from pathlib import Path
import sys,json,time,importlib.util,gc
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
# Load the established decoder and taxonomy before importing the pinned HGCLR encoder.
spec=importlib.util.spec_from_file_location('ref',ROOT/'htc_clip_local_benchmark_20261017/run_local.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
for k in list(sys.modules):
 if k=='model' or k.startswith('model.'):del sys.modules[k]
sys.path.insert(0,str(ROOT/'hgclr_full_20261017/vendor'))
from model.contrast import BertModel
B=ROOT/'hgclr_full_20261017/pretrained';src=ROOT/'hgclr_full_20261017/recovered/final'
torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu'
tok=AutoTokenizer.from_pretrained(B,local_files_only=True);cfg=AutoConfig.from_pretrained(B,local_files_only=True)
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def log(**v):print(json.dumps(v),flush=True)
protocol={'date':'2026-10-21','authorization':'User requested test evaluation of latest available hierarchical baseline checkpoints','selection':'best_macro.pt selected by validation, from latest HGCLR retraining; all five available seeds, no test-based selection','seeds':[3,13,21,42,87],'seed87':'incomplete training; report separately, no completed five-seed claim','decoder':'Established raw sigmoid+topk and validation-only depth offsets [-12,12] step .25','inference':'Pinned author BertModel encoder and classifier, strict loaded; eval forward is encoder CLS then linear classifier. Training-only graph/projector unused by original no-label forward.','device':device,'batch_size':8,'max_length':512,'training':False}
save(OUT/'protocol.json',protocol)
dev=r.read_jsonl(r.D/'esaw_dev.json');gd=r.gold(dev)
# Freeze all checkpoint choices and offsets before accessing test.
manifest={}
for seed in protocol['seeds']:
 p=src/f'seed{seed}';v=np.load(p/'validation_logits.npz');assert list(v['record_keys'])==[x['record_key'] for x in dev]
 manifest[str(seed)]={'checkpoint':str(p/'best_macro.pt'),'sha256':r.sha(p/'best_macro.pt'),'best':json.loads((p/'best.json').read_text()),'training_status':json.loads((p/'status.json').read_text()),'offsets':r.fit(v['logits'],gd),'validation_sha256':r.sha(p/'validation_logits.npz')}
save(OUT/'frozen_checkpoints_decoders.json',manifest);frozen=r.sha(OUT/'frozen_checkpoints_decoders.json')
test=r.read_jsonl(r.D/'esaw_test.json');assert len(test)==5000;gold=r.gold(test)
@torch.inference_mode()
def infer(enc,head,rows,label):
 zs=[];start=time.time()
 for i in range(0,len(rows),8):
  inp=tok([x['token'] for x in rows[i:i+8]],padding=True,truncation=True,max_length=512,return_tensors='pt');h=enc(**{k:v.to(device) for k,v in inp.items()},return_dict=True).last_hidden_state[:,0];zs.append(head(h).cpu().numpy())
  if i==0 or (i+8)%400==0:log(split=label,records=min(i+8,len(rows)),seconds=time.time()-start)
 return np.concatenate(zs)
results={}
for seed in protocol['seeds']:
 d=OUT/f'seed{seed}';d.mkdir(exist_ok=True);meta=manifest[str(seed)]
 if (d/'metrics.json').exists():results[str(seed)]=json.loads((d/'metrics.json').read_text());continue
 log(seed=seed,event='loading');ck=torch.load(meta['checkpoint'],map_location='cpu',mmap=True,weights_only=False);state=ck['model']
 enc=BertModel(cfg);enc.load_state_dict({k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')},strict=True)
 head=torch.nn.Linear(cfg.hidden_size,len(r.H.labels));head.load_state_dict({k.removeprefix('classifier.'):v for k,v in state.items() if k.startswith('classifier.')},strict=True);enc.eval().to(device);head.eval().to(device);del ck,state;gc.collect()
 probe=infer(enc,head,dev[:32],'validation_parity');cached=np.load(src/f'seed{seed}'/'validation_logits.npz')['logits'][:32];diff=float(np.max(np.abs(probe-cached)));assert np.allclose(probe,cached,atol=.003,rtol=.003),(seed,diff)
 save(d/'load_verification.json',{'strict_encoder_classifier':True,'validation_probe_max_abs_diff':diff,'checkpoint_sha256':meta['sha256']})
 z=infer(enc,head,test,f'test_seed{seed}');assert z.shape==(5000,len(r.H.labels)) and np.isfinite(z).all();np.savez_compressed(d/'test_scores.npz',logits=z,gold_ids=gold,record_keys=np.array([x['record_key'] for x in test]))
 a=r.decode(z);b=r.decode(z,meta['offsets']);m={'seed':seed,**meta,'A_raw':r.metrics(a,gold),'B_validation_calibrated':r.metrics(b,gold),'frozen_manifest_sha256':frozen};save(d/'metrics.json',m)
 with (d/'predictions.jsonl').open('w') as f:
  for i,row in enumerate(test):f.write(json.dumps({'record_key':row['record_key'],'gold':gold[i].tolist(),'raw_top10':a[i].tolist(),'calibrated_top10':b[i].tolist()})+'\n')
 results[str(seed)]=m;log(seed=seed,event='complete',raw=m['A_raw']['mean_top1_pct'],calibrated=m['B_validation_calibrated']['mean_top1_pct']);del enc,head;gc.collect();torch.mps.empty_cache()
 assert r.sha(OUT/'frozen_checkpoints_decoders.json')==frozen
save(OUT/'results.json',results)
