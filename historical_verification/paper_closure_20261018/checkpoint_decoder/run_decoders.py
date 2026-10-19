from pathlib import Path
import sys,json,time,copy,gc,importlib.util,argparse
from unittest.mock import patch
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer,BertModel
ROOT=Path(__file__).resolve().parents[2];R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ref',ROOT/'htc_clip_local_benchmark_20261017/run_local.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
P,D,H=ref.P,ref.D,ref.H;sys.path.insert(0,str(P))
from esaw_htc.recent_models import HisrModel,HyilrModel
from esaw_htc.common import read_jsonl
parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['hisr','hyilr'],required=True);args=parser.parse_args();name=args.method;R=R/name;R.mkdir(exist_ok=True);ref.R=R
sha,save,log=ref.sha,ref.save,ref.log
CK=P/f'downloads/live/runpod_{name}_state/results/{name}_preliminary/final/seed13/best_macro.pt';B=ROOT/'hgclr_full_20261017/pretrained';V=ROOT/f'audit_esaw_20261015/baseline_diagnostics_20261016/{name}_validation_scores.npz'
torch.set_num_threads(4);torch.manual_seed(20261018);device='mps' if torch.backends.mps.is_available() else 'cpu';start=time.time()
protocol={'method':name,'retrospective':True,'checkpoint':str(CK),'checkpoint_sha256':sha(CK),'validation_source':str(V),'validation_source_sha256':sha(V),'taxonomy_sha256':{n:sha(D/n) for n in ['metadata.json','bert_value_dict.pt','slot.pt']},'decoder_A':'original FP32 sigmoid then torch.topk10, all admissible codes','decoder_B':'FP64 logit plus per-field child offset, parent codes remain admissible, stable taxonomy order','offset_grid':[-12,12,.25],'selection':'full validation 5000 only; field Top1; tie smallest abs offset then first grid order','test_policy':'freeze offsets before opening full test; historical test already observed; retrospective only','training_performed':False,'additional_Runpod_resources':False,'source_sha256':sha(Path(__file__)),'reference_source_sha256':sha(ROOT/'htc_clip_local_benchmark_20261017/run_local.py'),'precision':'FP32','device':device,'batch_size':8,'max_length':512}
save('protocol.json',protocol);v=np.load(V);z=v['logits'];g=v['gold_ids'];assert z.shape[0]==5000;np.savez_compressed(R/'validation_scores.npz',logits=z,gold_ids=g,record_keys=v['record_keys']);o=ref.fit(z,g)
a=ref.decode(z);b=ref.decode(z,o);rng=np.random.default_rng(20261016);order=rng.permutation(5000);fold=np.empty(5000,int);fold[order[:2500]]=0;fold[order[2500:]]=1;cross=np.empty_like(a);cross_o={}
for k in [0,1]:
 ix=fold==k;oo=ref.fit(z[ix],g[ix]);cross[~ix]=ref.decode(z[~ix],oo);cross_o[str(k)]=oo
save('validation_metrics.json',{'A_raw':ref.metrics(a,g),'B_full_validation_resubstitution':ref.metrics(b,g),'B_crossfit_diagnostic':ref.metrics(cross,g),'fold_offsets':cross_o,'limitation':'Checkpoint previously selected on whole validation; crossfit is diagnostic only.'})
save('decoder_frozen.json',{'protocol_sha256':sha(R/'protocol.json'),'offsets':o,'validation_scores_sha256':sha(R/'validation_scores.npz'),'time_frozen_epoch':time.time(),'test_opened':False});frozen=sha(R/'decoder_frozen.json');log('decoder_frozen',offsets=o,sha256=frozen)
ck=torch.load(CK,map_location='cpu',mmap=True,weights_only=False);ca=ck['args'];cfg=AutoConfig.from_pretrained(str(B),local_files_only=True)
with torch.device('meta'),patch('esaw_htc.recent_models.AutoConfig.from_pretrained',return_value=copy.deepcopy(cfg)),patch('esaw_htc.recent_models.AutoModel.from_pretrained',side_effect=lambda *a,**k:BertModel(k.get('config',copy.deepcopy(cfg)))):
 if name=='hyilr':model=HyilrModel('offline',len(H.labels),H.depth2label,H.children,float(ca['contrastive_weight']),float(ca['temperature']))
 else:model=HisrModel('offline',len(H.labels),H.children,H.parent,{},node_loss_weight=float(ca['contrastive_weight']),sequence_loss_weight=float(ca['contrastive_weight']))
loaded=model.load_state_dict(ck['model'],strict=True,assign=True);save('checkpoint_load.json',{'epoch':ck['epoch'],'missing_keys':loaded.missing_keys,'unexpected_keys':loaded.unexpected_keys,'args':ca})
for mod in model.modules():
 for n,val in list(mod._buffers.items()):
  if val is None or val.device.type!='meta':continue
  if n=='position_ids':mod._buffers[n]=torch.arange(cfg.max_position_embeddings,dtype=val.dtype).expand(1,-1)
  elif n=='token_type_ids':mod._buffers[n]=torch.zeros(tuple(val.shape),dtype=val.dtype)
  else:raise RuntimeError(n)
assert not any(t.device.type=='meta' for t in list(model.parameters())+list(model.buffers()));model.eval().to(device);del ck;gc.collect();tok=AutoTokenizer.from_pretrained(str(B),local_files_only=True)
test=read_jsonl(D/'esaw_test.json');assert len(test)==5000;assert sha(R/'decoder_frozen.json')==frozen
@torch.inference_mode()
def infer():
 zs=[]
 for i in range(0,5000,8):
  inp=tok([r['token'] for r in test[i:i+8]],padding=True,truncation=True,max_length=512,return_tensors='pt');out=model(**{k:x.to(device) for k,x in inp.items()});zs.append(out['logits'].cpu().numpy())
  if i==0 or (i+8)%400==0 or i+8==5000:log('test_inference',records=min(i+8,5000),elapsed_seconds=time.time()-start)
 return np.concatenate(zs)
z=infer();assert np.isfinite(z).all();g=ref.gold(test);np.savez_compressed(R/'test_scores.npz',logits=z,gold_ids=g,record_keys=np.array([r['record_key'] for r in test]));a=ref.decode(z);b=ref.decode(z,o)
m={'A_raw':ref.metrics(a,g),'B_validation_calibrated':ref.metrics(b,g),'decoder_sha256':frozen,'retrospective':True,'no_test_tuning':True}
delta=(b[:,:,0]==g).mean(1)-(a[:,:,0]==g).mean(1);rg=np.random.default_rng(20261018);boot=np.array([delta[rg.integers(5000,size=5000)].mean()*100 for _ in range(10000)]);m['paired_gain']={'points':float(delta.mean()*100),'record_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'replicates':10000,'seed':20261018,'limitation':'Conditional on fixed checkpoint and offsets; excludes seed and selection uncertainty.'}
train_text={r['token'] for r in read_jsonl(D/'esaw_train.json')};keep=np.array([r['token'] not in train_text for r in test]);m['overlap_sensitivity']={'excluded':int((~keep).sum()),'definition':'exact token text equality to a training record','A_raw':ref.metrics(a[keep],g[keep]),'B_validation_calibrated':ref.metrics(b[keep],g[keep])};save('test_metrics.json',m)
for mode,preds in [('A_raw',a),('B_validation_calibrated',b)]:
 with (R/(mode+'_test_predictions.jsonl')).open('w') as f:
  for i,row in enumerate(test):f.write(json.dumps({'record_key':row['record_key'],'gold':{field:H.output_by_leaf[field][int(g[i,j])] for j,field in enumerate(ref.FIELDS)},'top10':{field:[H.output_by_leaf[field][int(n)] for n in preds[i,j]] for j,field in enumerate(ref.FIELDS)}})+'\n')
assert sha(R/'decoder_frozen.json')==frozen
for n,h in protocol['taxonomy_sha256'].items():assert sha(D/n)==h
log('completed',raw=m['A_raw']['mean_top1_pct'],calibrated=m['B_validation_calibrated']['mean_top1_pct'],elapsed_seconds=time.time()-start)
