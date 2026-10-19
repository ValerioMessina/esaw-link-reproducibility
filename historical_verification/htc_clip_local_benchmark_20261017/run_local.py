"""Frozen retrospective HTC-CLIP benchmark; local Mac, original checkpoint/assets."""
from pathlib import Path
import sys,json,time,hashlib,platform,os
from unittest.mock import patch
import numpy as np
import torch
from transformers import AutoTokenizer
P=Path('/Volumes/university data/tesi/benchs/paper_reproductions');R=Path(__file__).resolve().parent;D=P/'data/esaw'
sys.path[:0]=[str(P),str(P/'upstream/contrastive-htc')]
from esaw_htc.common import load_hierarchy,read_jsonl
from esaw_htc.htc_clip_model import HtcClip
H=load_hierarchy(D);FIELDS=list(H.field_leaves);DEP=np.array([next(d for d,vs in H.depth2label.items() if i in vs) for i in range(len(H.labels))]);CK=P/'downloads/live/runpod_htc_clip_state/results/htc_clip_preliminary/tuning/g0.02_w0.05/best.pt';B=R.parent/'hgclr_full_20261017/pretrained'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,d):
 p=R/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def log(event,**kw):
 d={'time':time.time(),'event':event,**kw};save('progress.json',d);print(json.dumps(d),flush=True)
def gold(rows):
 g=np.empty((len(rows),8),np.int64)
 for j,f in enumerate(FIELDS):
  allowed=set(H.field_leaves[f])
  for i,r in enumerate(rows):
   ids=allowed.intersection(r['label']);assert ids;g[i,j]=max(ids,key=lambda n:DEP[n])
 return g
# Admissible order is the original taxonomy order. No leaf-only filtering.
def decode(z,offsets=None):
 pred=np.empty((len(z),8,10),np.int64)
 for j,f in enumerate(FIELDS):
  vs=np.array(H.field_leaves[f]);assert len(vs)>=10
  if offsets is None:ix=torch.topk(torch.from_numpy(z[:,vs]).sigmoid(),10,dim=1).indices.numpy()
  else:ix=np.argsort(-(z[:,vs].astype(np.float64)+offsets[f]*(DEP[vs]==2)),axis=1,kind='stable')[:,:10]
  pred[:,j]=vs[ix]
 return pred
GRID=np.arange(-12.,12.001,.25)
def fit(z,g):
 offsets={}
 for j,f in enumerate(FIELDS):
  vs=np.array(H.field_leaves[f]);a=z[:,vs].astype(np.float64);ch=DEP[vs]==2
  scores=[int((vs[(a+float(d)*ch).argmax(1)]==g[:,j]).sum()) for d in GRID]
  best=max(range(len(GRID)),key=lambda k:(scores[k],-abs(float(GRID[k]))));offsets[f]=float(GRID[best])
 return offsets

def metrics(pred,g):
 hit=pred[:,:,0]==g;child=DEP[g]==2;out={'records':len(g),'decisions':int(g.size),'mean_top1_pct':float(hit.mean()*100),'record_exact8_pct':float(hit.all(1).mean()*100),'parent_prediction_pct':float((DEP[pred[:,:,0]]==1).mean()*100),'gold_child_accuracy_pct':float(hit[child].mean()*100),'gold_parent_accuracy_pct':float(hit[~child].mean()*100),'by_field':{}}
 for k in [1,5,10]:out['mean_top'+str(k)+'_pct']=float((pred[:,:,:k]==g[:,:,None]).any(2).mean()*100);out['record_all8_top'+str(k)+'_pct']=float((pred[:,:,:k]==g[:,:,None]).any(2).all(1).mean()*100)
 for j,f in enumerate(FIELDS):out['by_field'][f]={'top1_pct':float(hit[:,j].mean()*100),'gold_parent_records':int((~child[:,j]).sum()),'gold_child_records':int(child[:,j].sum()),'parent_accuracy_pct':float(hit[~child[:,j],j].mean()*100),'child_accuracy_pct':float(hit[child[:,j],j].mean()*100)}
 return out

def main():
 torch.set_num_threads(4);torch.manual_seed(20261017);device='mps' if torch.backends.mps.is_available() else 'cpu';start=time.time()
 expected={'metadata.json':'00d38b9187515a2cce7e4663338c6d2a8978a1e8d795bfeab05246f8b1311446','bert_value_dict.pt':'f38974fdbd0029aae828910c8d5f5d130e7a68d9a7e205444f330d1f1e6a93b6','slot.pt':'536cc60b42df1506d2651302648b3a8f2513138eacc128c2aed4cb2fb133f842'}
 for n,h in expected.items():assert sha(D/n)==h
 assert sha(CK)=='8e47b37a3eb1dfae732dfa6f625e519aeeddf1f364c0a55f9098d35200e92487'
 source={str(p):sha(p) for p in [P/'esaw_htc/htc_clip_model.py',P/'esaw_htc/common.py',P/'upstream/contrastive-htc/model/graph.py',P/'upstream/contrastive-htc/model/contrast.py',Path(__file__)]}
 # Freeze choices before running any new validation or test inference.
 protocol={'id':'htc-clip-local-retrospective-20261017-v1','method':'HTC-CLIP paper reconstruction with explicit class-logit deviations','checkpoint':str(CK),'checkpoint_sha256':sha(CK),'seed':3,'epoch':50,'encoder':'bert-base-uncased','literal_paper_path':False,'BCE_reduction':'mean','historical_unused_embedding_dropout':'retained; eval inference unaffected','taxonomy_original_hashes':expected,'source_hashes':source,'head':'maximum of linear and path logits','decoder_A':'original torch.sigmoid(FP32) then torch.topk(10) on all admissible codes','decoder_B':'FP64 logit + one child offset per field; parent codes remain admissible; stable taxonomy-index tie break','offset_grid':[-12,12,.25],'offset_fit':'full 5000 validation only, maximize field Top1; tie smallest abs offset then first grid order','diagnostic_crossfit':'two folds seed20261016; assessment separated from offset fit, but checkpoint selected on full validation','test_policy':'evaluate once after offsets frozen; retrospective test already observed historically; no independent confirmation claim','device':device,'batch_size':8,'max_length':512,'precision':'FP32','padded_input':'dynamic right padding; original mask and tokenization','training_performed':False,'additional_Runpod_resources':False}
 save('protocol.json',protocol);save('runtime.json',{'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'device':device});log('load_checkpoint',device=device)
 tok=AutoTokenizer.from_pretrained(str(B),local_files_only=True)
 with patch('model.graph.AutoTokenizer.from_pretrained',return_value=tok):model=HtcClip(str(B),len(H.labels),H.depth2label,[H.display[n] for n in H.labels],str(D),.02,.05,1.,128)
 ck=torch.load(CK,map_location='cpu',mmap=True,weights_only=False);model.load_state_dict(ck['model'],strict=True);del ck;model.eval().to(device)
 @torch.inference_mode()
 def infer(rows,split):
  vals={k:[] for k in ['logits','linear_logits','path_logits']};t=time.time()
  for i in range(0,len(rows),8):
   batch=rows[i:i+8];inputs=tok([r['token'] for r in batch],padding=True,truncation=True,max_length=512,return_tensors='pt');out=model(**{k:v.to(device) for k,v in inputs.items()})
   for k in vals:vals[k].append(out[k].cpu().numpy())
   if i==0 or (i+len(batch))%400==0 or i+len(batch)==len(rows):log('inference',split=split,records=i+len(batch),total=len(rows),elapsed_seconds=time.time()-t)
  z={k:np.concatenate(v) for k,v in vals.items()}
  assert all(np.isfinite(v).all() for v in z.values());z['record_keys']=np.array([r['record_key'] for r in rows]);z['gold_ids']=gold(rows);np.savez_compressed(R/(split+'_scores.npz'),**z);return z
 dev=read_jsonl(D/'esaw_dev.json');assert len(dev)==5000;v=infer(dev,'validation');g=v['gold_ids'];z=v['logits'];log('fit_validation_offsets');offsets=fit(z,g)
 raw=decode(z);adapted=decode(z,offsets);rng=np.random.default_rng(20261016);order=rng.permutation(5000);fold=np.empty(5000,int);fold[order[:2500]]=0;fold[order[2500:]]=1;cross=np.empty_like(raw);cross_offsets={}
 for k in [0,1]:
  fitidx=fold==k;assess=~fitidx;o=fit(z[fitidx],g[fitidx]);cross[assess]=decode(z[assess],o);cross_offsets[str(k)]=o
 save('validation_metrics.json',{'A_raw':metrics(raw,g),'B_full_validation_resubstitution':metrics(adapted,g),'B_crossfit_diagnostic':metrics(cross,g),'fold_offsets':cross_offsets,'limitation':'Full-validation fit is resubstitution; crossfit still uses checkpoint selected on entire validation.'})
 # Freeze final offsets and protocol hashes BEFORE opening the test records.
 save('decoder_frozen.json',{'protocol_sha256':sha(R/'protocol.json'),'offsets':offsets,'validation_scores_sha256':sha(R/'validation_scores.npz'),'time_frozen_epoch':time.time(),'test_opened':False});frozen_hash=sha(R/'decoder_frozen.json');log('decoder_frozen',offsets=offsets,sha256=frozen_hash)
 test=read_jsonl(D/'esaw_test.json');assert len(test)==5000;assert sha(R/'decoder_frozen.json')==frozen_hash;t=infer(test,'test');g=t['gold_ids'];a=decode(t['logits']);b=decode(t['logits'],offsets)
 m={'A_raw':metrics(a,g),'B_validation_calibrated':metrics(b,g),'decoder_sha256':frozen_hash,'retrospective':True,'no_test_tuning':True}
 # Paired bootstrap over records; conditional on fixed weights and fitted offsets.
 delta=(b[:,:,0]==g).mean(1)-(a[:,:,0]==g).mean(1);rg=np.random.default_rng(20261017);boot=np.array([delta[rg.integers(5000,size=5000)].mean()*100 for _ in range(2000)]);m['paired_gain']={'points':float(delta.mean()*100),'record_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'limitation':'Conditional on one checkpoint and fixed offsets; not seed/model selection uncertainty.'};save('test_metrics.json',m)
 for name,preds in [('A_raw',a),('B_validation_calibrated',b)]:
  with (R/(name+'_test_predictions.jsonl')).open('w') as f:
   for i,row in enumerate(test):
    item={'record_key':row['record_key'],'gold':{field:H.output_by_leaf[field][int(g[i,j])] for j,field in enumerate(FIELDS)},'top10':{field:[H.output_by_leaf[field][int(n)] for n in preds[i,j]] for j,field in enumerate(FIELDS)}};f.write(json.dumps(item)+'\n')
 assert sha(R/'decoder_frozen.json')==frozen_hash
 for n,h in expected.items():assert sha(D/n)==h
 log('completed',elapsed_seconds=time.time()-start,raw_test_top1=m['A_raw']['mean_top1_pct'],calibrated_test_top1=m['B_validation_calibrated']['mean_top1_pct'])
if __name__=='__main__':
 try:main()
 except Exception as e:log('failed',error=repr(e));raise
