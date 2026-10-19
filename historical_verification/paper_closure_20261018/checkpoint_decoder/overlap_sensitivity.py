from pathlib import Path
import sys,json,importlib.util
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];D=Path('/Volumes/university data/tesi/benchs/dual encoder');sys.path.insert(0,str(D))
from esaw_biencoder.data import load_dataset,accident_to_text
sp=importlib.util.spec_from_file_location('ref',ROOT/'htc_clip_local_benchmark_20261017/run_local.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def key(row,i):
 for f in ('id','numero_caso'):
  v=row.get(f) or row.get(f.upper())
  if v is not None and str(v).strip():return str(v).strip()
 return f'index:{i}'
splits=json.loads((D/'artifacts/splits/esaw-fixed-seed13-test10-val10.json').read_text())['splits'];tr=set(splits['train']);records=load_dataset(D/'dataset.json');texts={key(r,i):accident_to_text(r) for i,r in enumerate(records)};traintexts={v for k,v in texts.items() if k in tr and v};testrows=m.read_jsonl(m.D/'esaw_test.json');bad=np.array([bool(texts[r['record_key']]) and texts[r['record_key']] in traintexts for r in testrows]);empty=np.array([not texts[r['record_key']] for r in testrows]);assert int(bad.sum())==12
out={'definition':'Nonempty original accident_to_text exact equality to training; same as audit_data.py','nonempty_overlap_test_records':int(bad.sum()),'test_empty_evidence_records':int(empty.sum()),'methods':{}}
for name,rd in [('htc_clip',ROOT/'htc_clip_local_benchmark_20261017'),('hyilr',R/'hyilr'),('hisr',R/'hisr')]:
 if not (rd/'test_scores.npz').exists():continue
 z=np.load(rd/'test_scores.npz');assert list(z['record_keys'])==[r['record_key'] for r in testrows];o=json.loads((rd/'decoder_frozen.json').read_text())['offsets'];a=m.decode(z['logits']);b=m.decode(z['logits'],o);keep=~bad;out['methods'][name]={'source_scores_sha256':m.sha(rd/'test_scores.npz'),'A_raw':m.metrics(a[keep],z['gold_ids'][keep]),'B_validation_calibrated':m.metrics(b[keep],z['gold_ids'][keep])}
(R/'overlap12_sensitivity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'excluded':out['nonempty_overlap_test_records'],'empty':out['test_empty_evidence_records'],'methods':list(out['methods'])}))
