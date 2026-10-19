from pathlib import Path
import json,time,os,random,signal,math,hashlib,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel,AutoTokenizer,Adafactor,get_linear_schedule_with_warmup
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);CFG=json.loads((ROOT/'protocol.json').read_text());STATE=json.loads((ROOT/'runpod_state.json').read_text());D=Path('/workspace/hgclr/private_data');STOP=False

def handle(*_):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle)
def write(name,d):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def log(event,**kw):
 d={'time':time.time(),'event':event,**kw};write('progress.json',d)
 with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
 print(json.dumps(d),flush=True)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(split):return [json.loads(l) for l in (D/('esaw_'+split+'.json')).open() if l.strip()]
META=json.loads((D/'metadata.json').read_text());ids={n:i for i,n in enumerate(META['labels'])};fields=list(META['field_leaves']);members=[np.array([ids[n] for n in META['field_leaves'][f]]) for f in fields];flat_meta=np.concatenate(members);starts=np.cumsum([0]+[len(v) for v in members]);mapping={int(v):i for i,v in enumerate(flat_meta)};parent={ids[c]:ids[p] for p,cs in META['children'].items() for c in cs};roles=json.loads((ROOT/'roles.json').read_text());assert [r['output_field'] for r in roles]==fields

def depth(v):
 n=0
 while v in parent:n+=1;v=parent[v]
 return n
class Data:
 def __init__(self,rs,tok):
  self.rows=rs;N=len(rs)*8;self.ids=torch.empty((N,256),dtype=torch.int32);self.mask=torch.empty((N,256),dtype=torch.uint8);self.gold=torch.empty(N,dtype=torch.long)
  all_allowed=[set(v.tolist()) for v in members]
  for base in range(0,N,2048):
   ix=range(base,min(base+2048,N));texts=[]
   for k in ix:
    r=rs[k//8];j=k%8;role=roles[j];texts.append(f"Variabile ESAW da classificare: {role['title']}. Criterio: {role['role']}. Infortunio: {r['token']}.")
    cand=all_allowed[j].intersection(r['label']);assert cand;self.gold[k]=mapping[max(cand,key=depth)]
   t=tok(texts,padding='max_length',max_length=256,truncation=True,return_tensors='pt');end=base+len(texts);self.ids[base:end]=t['input_ids'].int();self.mask[base:end]=t['attention_mask'].byte()
  log('tokenized',records=len(rs),queries=N)
 def __len__(self):return len(self.gold)
class Classifier(nn.Module):
 def __init__(self):
  super().__init__();self.backbone=AutoModel.from_pretrained(CFG['base_model'],revision=CFG['revision'],torch_dtype=torch.bfloat16,attn_implementation='sdpa');self.backbone.config.use_cache=False;self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});self.head=nn.Linear(1024,818,dtype=torch.float32);nn.init.normal_(self.head.weight,std=.02);nn.init.zeros_(self.head.bias)
  allowed=torch.zeros((8,818),dtype=torch.bool)
  for j in range(8):allowed[j,starts[j]:starts[j+1]]=True
  self.register_buffer('allowed',allowed,persistent=False)
 def forward(self,ids,mask,field):
  h=self.backbone(input_ids=ids,attention_mask=mask).last_hidden_state
  # Explicit mask-index pooling works for left OR right padding.
  pos=(mask.long()*torch.arange(mask.shape[1],device=mask.device)[None,:]).max(1).values;z=h[torch.arange(len(h),device=h.device),pos].float();z=F.normalize(z,dim=-1);return self.head(z).masked_fill(~self.allowed[field],float('-inf'))
def batch(data,index):
 return data.ids[index].long().cuda(),data.mask[index].long().cuda(),(index%8).cuda(),data.gold[index].cuda()
@torch.inference_mode()
def infer(model,data):
 model.eval();z=np.empty((len(data.rows),818),np.float32)
 for i in range(0,len(data),64):
  ix=torch.arange(i,min(i+64,len(data)));a,mask,f,y=batch(data,ix);scores=model(a,mask,f).float().cpu().numpy()
  for j in range(8):
   local=(ix.numpy()%8)==j;recs=ix.numpy()[local]//8;z[recs,starts[j]:starts[j+1]]=scores[local,starts[j]:starts[j+1]]
 return z

def exact(z,data):
 gold=data.gold.numpy().reshape(-1,8);hit=np.stack([z[:,starts[j]:starts[j+1]].argmax(1)+starts[j]==gold[:,j] for j in range(8)],1);return {'mean_top1':float(hit.mean()),'exact8':float(hit.all(1).mean())}
def main():
 random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 hashes={n:sha(D/n) for n in ['metadata.json','bert_value_dict.pt','slot.pt','esaw_train.json','esaw_dev.json']};write('data_manifest.json',hashes);write('protocol.json',CFG);log('loading_model',seed=CFG['seed'])
 tok=AutoTokenizer.from_pretrained(CFG['base_model'],revision=CFG['revision'],padding_side='left');model=Classifier().cuda();train=Data(rows('train'),tok);dev=Data(rows('dev'),tok);opt=Adafactor(model.parameters(),lr=2e-5,relative_step=False,scale_parameter=False,warmup_init=False);total=math.ceil(len(train)/80)*3;sch=get_linear_schedule_with_warmup(opt,int(total*.1),total)
 best=-1.;hist=[];step=0;cursor=0;start_epoch=1;order=None;generator=torch.Generator().manual_seed(CFG['seed']);start=time.time();losses=[]
 def save_checkpoint(name,epoch,cursor,order):
  path=OUT/name;tmp=path.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'step':step,'epoch':epoch,'batch_cursor':cursor,'permutation':order,'history':hist,'best_validation_top1':best,'generator':generator.get_state(),'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},'protocol':CFG},tmp);tmp.replace(path)
 # The frozen run is fresh. Exact continuation is supported only by this full state.
 if (OUT/'last.pt').exists():
  c=torch.load(OUT/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(c['model'],strict=True);opt.load_state_dict(c['optimizer']);sch.load_state_dict(c['scheduler']);step=c['step'];start_epoch=c['epoch'];cursor=c['batch_cursor'];order=c['permutation'];hist=c['history'];best=c['best_validation_top1'];generator.set_state(c['generator']);random.setstate(c['rng']['python']);np.random.set_state(c['rng']['numpy']);torch.set_rng_state(c['rng']['torch']);torch.cuda.set_rng_state_all(c['rng']['cuda']);del c;log('resumed',step=step,epoch=start_epoch,cursor=cursor)
 stop_reason='epochs_completed'
 for epoch in range(start_epoch,4):
  if order is None:order=torch.randperm(len(train),generator=generator)
  model.train();opt.zero_grad(set_to_none=True);finished=True
  for pos in range(cursor,len(train),80):
   if STOP or time.time()>=STATE['training_deadline_epoch']:
    stop_reason='budget_or_deadline';finished=False;cursor=pos;break
   ix=order[pos:pos+80];loss_value=0
   for b in ix.split(16):
    a,mask,f,y=batch(train,b);scores=model(a,mask,f);loss=F.cross_entropy(scores,y)*len(b)/len(ix);assert torch.isfinite(loss),('nonfinite',step);loss.backward();loss_value+=float(loss.detach())
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);assert torch.isfinite(norm);opt.step();sch.step();opt.zero_grad(set_to_none=True);step+=1;losses.append(loss_value);cursor=pos+len(ix)
   if step%50==0:log('train_step',epoch=epoch,step=step,loss=loss_value,gradient_norm=float(norm),seconds=time.time()-start,queries_per_second=(step*80)/(time.time()-start))
   if step%500==0:save_checkpoint('last.pt',epoch,cursor,order)
  log('validation_started',epoch=epoch,step=step);z=infer(model,dev);m=exact(z,dev);hist.append({'epoch':epoch,'epoch_complete':finished,'step':step,'validation':m,'mean_loss':float(np.mean(losses)) if losses else None});write('history.json',hist)
  improved=m['mean_top1']>best
  if improved:
   best=m['mean_top1'];save_checkpoint('best.pt',epoch,cursor,order);np.savez_compressed(OUT/'validation_logits.npz',logits=z,record_keys=np.array([r['record_key'] for r in dev.rows]),gold=dev.gold.numpy().reshape(-1,8),label_meta_ids=flat_meta);write('best.json',{'epoch':epoch,'step':step,'mean_validation_top1':best})
  log('epoch_complete' if finished else 'training_budget_stop',epoch=epoch,step=step,validation=m)
  if not finished:save_checkpoint('last.pt',epoch,cursor,order);break
  cursor=0;order=None;save_checkpoint('last.pt',epoch+1,cursor,order)
 # Model and epoch selection are now frozen before first opening of test records.
 write('selection_frozen.json',{'best_validation_top1':best,'best':json.loads((OUT/'best.json').read_text()),'time':time.time(),'test_opened':False,'stop_reason':stop_reason,'completed_epochs':sum(x['epoch_complete'] for x in hist),'seed':13})
 c=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(c['model'],strict=True);del c;test=Data(rows('test'),tok);log('test_started',step=step);z=infer(model,test);gold=test.gold.numpy().reshape(-1,8);np.savez_compressed(OUT/'test_logits.npz',logits=z,gold=gold,record_keys=np.array([r['record_key'] for r in test.rows]),label_meta_ids=flat_meta);metrics={'mean':{},'record':{},'stop_reason':stop_reason,'completed_epochs':sum(x['epoch_complete'] for x in hist),'single_seed':13,'retrospective':True}
 top=[]
 for j in range(8):top.append(np.argsort(-z[:,starts[j]:starts[j+1]],axis=1,kind='stable')[:,:10]+starts[j])
 for k in [1,5,10]:
  hit=np.stack([(top[j][:,:k]==gold[:,j,None]).any(1) for j in range(8)],1);metrics['mean']['top'+str(k)]=float(hit.mean());metrics['record']['all8_at'+str(k)]=float(hit.all(1).mean())
 codes=[]
 for f,vs in zip(fields,members):codes.extend([META['output_by_leaf'][f][META['labels'][int(v)]] for v in vs])
 with (OUT/'predictions.jsonl').open('w') as file:
  for i,row in enumerate(test.rows):file.write(json.dumps({'record_key':row['record_key'],'gold':{f:codes[gold[i,j]] for j,f in enumerate(fields)},'top10':{f:[codes[v] for v in top[j][i]] for j,f in enumerate(fields)}})+'\n')
 write('benchmark.json',metrics);log('completed',step=step,metrics=metrics)
if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback
  write('failure.json',{'error':repr(e),'traceback':traceback.format_exc()});log('failed',error=repr(e));raise
