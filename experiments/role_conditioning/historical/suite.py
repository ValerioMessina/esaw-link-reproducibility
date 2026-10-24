"""Matched role-conditioning ablation; no test-selected hyperparameters or seeds."""
from pathlib import Path
import os,sys,json,time,math,random,hashlib,gc,signal,traceback,shutil
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Adafactor,AutoTokenizer,get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer,losses
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from esaw_biencoder import data as data
from esaw_biencoder.schema import TARGET_FIELDS
from esaw_biencoder.evaluate import filter_records_by_split,record_split_key
R=ROOT/'results';R.mkdir(exist_ok=True);STATE=json.loads((ROOT/'runpod_state.json').read_text());P=json.loads((ROOT/'protocol.json').read_text());STOP=False

def stopping(*_):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def log(event,**kw):
 v={'time':time.time(),'event':event,**kw};print(json.dumps(v),flush=True)
 with (R/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
 write(R/'progress.json',v)
def gate():
 if STOP or time.time()>json.loads((ROOT/'runpod_state.json').read_text())['training_deadline_epoch']:raise TimeoutError('Budget deadline reached; incomplete stages are not scientific results')
def seed_all(seed=13):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def clear():
 gc.collect();torch.cuda.empty_cache()
def model_load(path):
 seed_all();m=SentenceTransformer(str(path),device='cuda',model_kwargs={'torch_dtype':torch.bfloat16});m.max_seq_length=256;return m

def query(row,target,arm):
 evidence=row['_common_evidence']
 if arm=='role_on':return f"Variabile ESAW da classificare: {target.title}. Criterio: {target.role}. Infortunio: {evidence}."
 assert arm=='role_off';return f'Infortunio: {evidence}.'
def candidate(entry,target,arm):
 if arm=='role_on':return data.candidate_text(entry,target,'v5')
 assert arm=='role_off'
 parts=[f'Classe ESAW: {entry.label}.']
 if entry.parent_label:parts.append(f'Macroclasse: {entry.parent_label}.')
 parts.append(f'Codice ESAW: {entry.output_code}.');return ' '.join(parts)

def prepare():
 torch.set_num_threads(8);assert torch.cuda.is_available()
 base=Path(snapshot_download('Qwen/Qwen3-Embedding-0.6B',ignore_patterns=['*.bin','*.msgpack','*.h5','onnx/*','openvino/*']))
 tok=AutoTokenizer.from_pretrained(base);rows=data.load_dataset(ROOT/'private/dataset.json');explicit=data.build_explicit_taxonomy(data.load_taxonomy(ROOT/'private/tassonomia.json'))
 # Both arms receive exactly the same narrative, cropped using only the tokenizer and fixed role templates.
 # This prevents the shorter ablated prefix from revealing additional narrative beyond the on-arm token budget.
 prefix=max(len(tok(f'Variabile ESAW da classificare: {t.title}. Criterio: {t.role}. Infortunio: ',add_special_tokens=False)['input_ids']) for t in TARGET_FIELDS)
 evidence_budget=256-prefix-8
 assert evidence_budget>100
 for row in rows:
  ids=tok(data.accident_to_text(row),add_special_tokens=False)['input_ids'];text=tok.decode(ids[:evidence_budget],skip_special_tokens=True)
  while max(len(tok(query({'_common_evidence':text},t,'role_on'),add_special_tokens=False)['input_ids']) for t in TARGET_FIELDS)>254:
   text=tok.decode(tok(text,add_special_tokens=False)['input_ids'][:-1],skip_special_tokens=True)
  row['_common_evidence']=text
 splits={n:filter_records_by_split(rows,str(ROOT/'private/split.json'),n) for n in ['train','validation','test']}
 assert [len(splits[k]) for k in ['train','validation','test']]==[40000,5000,5000]
 candidates={arm:{t.field:[{'code':e.output_code,'taxonomy_code':e.code,'text':candidate(e,t,arm)} for e in explicit[t.taxonomy]] for t in TARGET_FIELDS} for arm in ['role_on','role_off']}
 mats=[t for t in TARGET_FIELDS if t.taxonomy=='agente_materiale']
 for t in mats[1:]:assert candidates['role_off'][t.field]==candidates['role_off'][mats[0].field]
 probe=rows[0]
 assert len({query(probe,t,'role_off') for t in TARGET_FIELDS})==1
 assert len({query(probe,t,'role_on') for t in TARGET_FIELDS})==8
 for t in TARGET_FIELDS:
  assert t.title not in query(probe,t,'role_off') and t.role not in query(probe,t,'role_off')
 write(R/'data_verification.json',{'base_snapshot':str(base),'base_config_sha256':sha(base/'config.json'),'base_weights_sha256':sha(base/'model.safetensors'),'narrative_token_budget':evidence_budget,'common_evidence':True,'max_seq_length':256,'records':{k:len(v) for k,v in splits.items()},'private_sources_sha256':{p.name:sha(p) for p in (ROOT/'private').iterdir() if p.is_file()},'role_off_shared_material_candidates_identical':True,'role_off_query_independent_of_field':True,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'protocol_sha256':sha(ROOT/'protocol.json')})
 return base,splits,candidates

def encode_unique(m,texts,batch=32):
 # Reusing exact duplicate embeddings affects speed only; models are in eval mode.
 unique=list(dict.fromkeys(texts));lookup={t:i for i,t in enumerate(unique)};z=m.encode(unique,batch_size=batch,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
 return z[np.array([lookup[t] for t in texts])]
def metrics(pred,g):
 out={}
 for k in [1,5,10]:
  h=(pred[:,:,:k]==g[:,:,None]).any(-1);out[f'top{k}_pct']=float(h.mean()*100);out[f'record_all8_top{k}_pct']=float(h.all(1).mean()*100)
 out['per_field']={t.output_field:{f'top{k}_pct':float((pred[:,j,:k]==g[:,j,None]).any(-1).mean()*100) for k in [1,5,10]} for j,t in enumerate(TARGET_FIELDS)}
 mat=[j for j,t in enumerate(TARGET_FIELDS) if t.taxonomy=='agente_materiale'];out['material_top1_pct']=float((pred[:,mat,0]==g[:,mat]).mean()*100)
 return out

@torch.inference_mode()
def evaluate(m,rows,cands,arm,stage,split):
 gate();dest=R/arm/stage;dest.mkdir(parents=True,exist_ok=True);p=dest/f'{split}_predictions.npz'
 if p.exists():return
 m.eval();n=len(rows);gold=np.empty((n,8),np.int32);pred=np.empty((n,8,10),np.int32)
 shared_queries=None
 if arm=='role_off':shared_queries=encode_unique(m,[query(row,TARGET_FIELDS[0],arm) for row in rows])
 for j,t in enumerate(TARGET_FIELDS):
  gate();cs=cands[t.field];codes=[c['code'] for c in cs];idx={c:i for i,c in enumerate(codes)}
  ce=encode_unique(m,[c['text'] for c in cs]);qe=shared_queries if shared_queries is not None else encode_unique(m,[query(row,t,arm) for row in rows]);score=qe.astype(np.float32)@ce.astype(np.float32).T
  pred[:,j]=np.argsort(-score,axis=1,kind='stable')[:,:10];gold[:,j]=[idx[data.label_to_output_code(t,data.record_value(row,t.field))] for row in rows];log('eval_field',arm=arm,stage=stage,split=split,field=t.output_field)
 if arm=='role_off':
  mat=[j for j,t in enumerate(TARGET_FIELDS) if t.taxonomy=='agente_materiale'];assert np.array_equal(pred[:,mat[0]],pred[:,mat[1]]) and np.array_equal(pred[:,mat[0]],pred[:,mat[2]])
 np.savez_compressed(p,rank_indices=pred,gold_indices=gold,record_keys=np.array([record_split_key(row,i) for i,row in enumerate(rows)]))
 write(dest/'candidate_code_order.json',{t.output_field:[c['code'] for c in cands[t.field]] for t in TARGET_FIELDS});write(dest/f'{split}_metrics.json',metrics(pred,gold));log('evaluation_complete',arm=arm,stage=stage,split=split,metrics=metrics(pred,gold))

@torch.inference_mode()
def mine(m,rows,cands,arm):
 dest=R/arm/'hard_negatives.json'
 if dest.exists():return json.loads(dest.read_text())
 m.eval();out={};shared=None
 if arm=='role_off':shared=encode_unique(m,[query(row,TARGET_FIELDS[0],arm) for row in rows])
 for j,t in enumerate(TARGET_FIELDS):
  gate();cs=cands[t.field];ce=encode_unique(m,[c['text'] for c in cs]);qe=shared if shared is not None else encode_unique(m,[query(row,t,arm) for row in rows]);scores=qe.astype(np.float32)@ce.astype(np.float32).T;gold=[data.label_to_output_code(t,data.record_value(row,t.field)) for row in rows];ix={c['code']:k for k,c in enumerate(cs)};rank=np.argsort(-scores,axis=1,kind='stable')[:,:21]
  out[t.field]=[[int(k) for k in order if int(k)!=ix[g]][:20] for order,g in zip(rank,gold)];assert all(len(x)==20 for x in out[t.field]);log('mined_field',arm=arm,field=t.output_field)
 write(dest,out);return out

def examples(rows,cands):
 pairs=[];indexes={t.field:{c['code']:i for i,c in enumerate(cands[t.field])} for t in TARGET_FIELDS}
 for i,row in enumerate(rows):
  for j,t in enumerate(TARGET_FIELDS):pairs.append((i,j,indexes[t.field][data.label_to_output_code(t,data.record_value(row,t.field))]))
 assert len(pairs)==320000
 return pairs

def train(m,rows,cands,arm,stage,neg=None):
 dest=R/arm/stage;dest.mkdir(parents=True,exist_ok=True);modeldir=dest/'model';done=dest/'training_complete.json'
 if done.exists():return model_load(modeldir)
 seed_all();m.train();is_refine=stage=='refinement';epochs=P['training'][stage]['epochs'];batch=24 if is_refine else 80;lr=8e-6 if is_refine else 2e-5
 pairs=examples(rows,cands);steps_epoch=math.ceil(len(pairs)/batch);total=steps_epoch*epochs
 loss_kw={'mini_batch_size':8,'show_progress_bar':False}
 if is_refine:loss_kw.update(hardness_mode='hard_negatives',hardness_strength=.2)
 lossfn=losses.CachedMultipleNegativesRankingLoss(m,**loss_kw)
 opt=Adafactor(m.parameters(),lr=lr,relative_step=False,scale_parameter=False,warmup_init=False);sched=get_linear_schedule_with_warmup(opt,math.ceil(total*.1),total)
 step=0;elapsed0=time.time();order_hash=hashlib.sha256();history=[];seeded=random.Random(13);last=dest/'last.pt'
 resume=None
 if last.exists():
  resume=torch.load(last,map_location='cpu',weights_only=False)
  assert resume['protocol_sha256']==sha(ROOT/'protocol.json'),'Checkpoint protocol mismatch'
  assert 0<=resume['epoch']<epochs and 0<resume['step']<=total
  m.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);sched.load_state_dict(resume['scheduler']);step=resume['step']
  events=R/'events.jsonl'
  if events.exists():
   for line in events.read_text().splitlines():
    v=json.loads(line)
    if v.get('event')=='training_step' and v.get('arm')==arm and v.get('stage')==stage and v.get('step',0)<=step:history.append(v)
  if history:elapsed0-=history[-1]['elapsed_seconds']
  log('checkpoint_resumed',arm=arm,stage=stage,step=step,total=total,checkpoint_sha256=sha(last))
 try:
  for ep in range(epochs):
   order=list(range(len(pairs)));seeded.shuffle(order);order_hash.update(np.array(order,dtype=np.int32).tobytes())
   if resume is not None and ep<resume['epoch']:continue
   start_batch=0
   if resume is not None and ep==resume['epoch']:
    assert order==resume['epoch_order'],'Training order mismatch'
    assert seeded.getstate()==resume['shuffle_rng'],'Shuffle RNG mismatch'
    start_batch=resume['next_batch'];assert step==ep*steps_epoch+math.ceil(start_batch/batch)
    random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng']);torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
    del resume;resume=None
   for a in range(start_batch,len(order),batch):
    gate();chosen=[pairs[k] for k in order[a:a+batch]];cols=[[],[]]+([[],[],[],[]] if is_refine else [])
    for ri,fj,gi in chosen:
     t=TARGET_FIELDS[fj];cs=cands[t.field];cols[0].append(query(rows[ri],t,arm));cols[1].append(cs[gi]['text'])
     if is_refine:
      # Identical mining policy and count; each arm mines its own train-only candidates.
      ni=neg[t.field][ri][:4];assert gi not in ni
      for j,k in enumerate(ni):cols[j+2].append(cs[k]['text'])
    features=[{k:v.to('cuda') for k,v in m.tokenize(col).items() if torch.is_tensor(v)} for col in cols]
    opt.zero_grad(set_to_none=True);value=lossfn(features,torch.empty(0,device='cuda'));assert torch.isfinite(value);value.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step();sched.step();step+=1
    if step==1 or step%25==0:
     row={'step':step,'total':total,'epoch':ep+1,'loss':float(value.detach()),'gradient_norm':float(norm),'elapsed_seconds':time.time()-elapsed0,'seconds_per_step':(time.time()-elapsed0)/step};log('training_step',arm=arm,stage=stage,**row);history.append(row);write(dest/'training_progress.json',row)
    if step%500==0:
     tmp=last.with_suffix('.tmp');torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':ep,'next_batch':a+batch,'step':step,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'shuffle_rng':seeded.getstate(),'epoch_order':order,'protocol_sha256':sha(ROOT/'protocol.json')},tmp);tmp.replace(last)
   log('epoch_complete',arm=arm,stage=stage,epoch=ep+1,steps=step)
  m.save(str(modeldir));write(done,{'complete':True,'epochs':epochs,'steps':step,'examples_per_epoch':len(pairs),'batch_size':batch,'effective_batch_equal_across_arms':True,'order_sha256':order_hash.hexdigest(),'seconds':time.time()-elapsed0,'checkpoint_selection':'fixed final epoch, no test or validation performance selection','training_history':history})
  tmp=last.with_suffix('.tmp');torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':epochs-1,'next_batch':len(pairs),'step':step,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'shuffle_rng':seeded.getstate(),'epoch_order':order,'protocol_sha256':sha(ROOT/'protocol.json')},tmp);tmp.replace(last)
 except BaseException:
  # Preserve a usable interrupted checkpoint without claiming a completed stage.
  m.save(str(dest/'interrupted_model'));write(dest/'incomplete.json',{'step':step,'total':total,'epoch':ep+1,'reason':'exception or budget stop'});raise
 del lossfn,opt,sched;clear();return m

def compare():
 report={};rng=np.random.default_rng(20261021)
 for stage in ['frozen','alignment','refinement']:
  a=np.load(R/'role_on'/stage/'test_predictions.npz');b=np.load(R/'role_off'/stage/'test_predictions.npz');assert np.array_equal(a['record_keys'],b['record_keys']);assert np.array_equal(a['gold_indices'],b['gold_indices'])
  ga=a['gold_indices'];ha=(a['rank_indices'][:,:,0]==ga);hb=(b['rank_indices'][:,:,0]==ga);mat=[j for j,t in enumerate(TARGET_FIELDS) if t.taxonomy=='agente_materiale'];arrays={'top1':ha.mean(1)-hb.mean(1),'exact8':ha.all(1).astype(float)-hb.all(1).astype(float),'material_top1':ha[:,mat].mean(1)-hb[:,mat].mean(1)}
  report[stage]={}
  for key,delta in arrays.items():
   boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean()*100 for _ in range(10000)]);report[stage][key]={'role_on_minus_off_pp':float(delta.mean()*100),'paired_record_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
  if stage!='frozen':
   aa=json.loads((R/'role_on'/stage/'training_complete.json').read_text());bb=json.loads((R/'role_off'/stage/'training_complete.json').read_text());assert aa['order_sha256']==bb['order_sha256'] and aa['steps']==bb['steps']
 write(R/'paired_comparisons.json',{'results':report,'replicates':10000,'seed':20261021,'limitation':'One matched training seed; CI conditional on fixed trained models; common-evidence crop differs from historical paper exports. No model/backbone superiority claim.'})
 table=['# Role-conditioning ablation','', 'Matched seed 13; full training split; same common narrative token budget. New paired retraining; historical paper exports are not substituted.','', '| Stage | Arm | Top1 % | Top5 % | Top10 % | Exact8 % | RR@5 % | RR@10 % | Material Top1 % |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for stage in ['frozen','alignment','refinement']:
  for arm in ['role_on','role_off']:
   v=json.loads((R/arm/stage/'test_metrics.json').read_text());keys=['top1_pct','top5_pct','top10_pct','record_all8_top1_pct','record_all8_top5_pct','record_all8_top10_pct','material_top1_pct'];table.append('| '+stage+' | '+arm+' | '+' | '.join(f'{v[k]:.4f}' for k in keys)+' |')
 table+=['','The primary contrast is role-on minus role-off after refinement. Stage-wise contrasts describe the role effect at frozen, aligned and refined checkpoints; they do not separate query-side and document-side conditioning or compare different backbones. Arm-specific mining is part of the pipeline effect.','', 'Mean record bootstrap intervals are conditional on a single matched seed, not training-seed variability.']
 (R/'REPORT.md').write_text('\n'.join(table)+'\n')


def main():
 write(R/'start.json',{'time':time.time(),'protocol_sha256':sha(ROOT/'protocol.json')});base,splits,cands=prepare()
 # Finish the matched alignment pair before either refinement, so a budget cutoff preserves an interpretable comparison.
 for arm in ['role_on','role_off']:
  if all((R/arm/'alignment'/f'{split}_metrics.json').exists() for split in ['test','validation']) and (R/arm/'alignment/training_complete.json').exists():continue
  gate();m=model_load(base);evaluate(m,splits['test'],cands[arm],arm,'frozen','test');m=train(m,splits['train'],cands[arm],arm,'alignment');evaluate(m,splits['validation'],cands[arm],arm,'alignment','validation');evaluate(m,splits['test'],cands[arm],arm,'alignment','test');del m;clear()
 for arm in ['role_on','role_off']:
  gate();m=model_load(R/arm/'alignment/model');neg=mine(m,splits['train'],cands[arm],arm);m=train(m,splits['train'],cands[arm],arm,'refinement',neg);evaluate(m,splits['validation'],cands[arm],arm,'refinement','validation');evaluate(m,splits['test'],cands[arm],arm,'refinement','test');del m;clear()
 compare();write(R/'COMPLETE.json',{'complete':True,'time':time.time(),'protocol_sha256':sha(ROOT/'protocol.json')});log('suite_complete')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(R/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});log('suite_failed',error=repr(e));raise
