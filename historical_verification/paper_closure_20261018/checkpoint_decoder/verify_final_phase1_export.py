from pathlib import Path
import json,hashlib,collections
D=Path('/Volumes/university data/tesi/benchs/dual encoder');R=Path(__file__).resolve().parent;P=D/'error_analysis/v5-qwen3-0.6b-mnrl-seq256-test-flat-top10-predictions.jsonl';F=['COD_LOC','COD_LAV','COD_ATV','COD_MAT_ATV','COD_DEV','COD_MAT_DEV','COD_CON','COD_MAT_CON'];hits=collections.Counter();n=0;seen=set()
for line in P.open():
 row=json.loads(line);assert row['split_key'] not in seen;seen.add(row['split_key']);n+=1;assert set(row['targets'])==set(F)
 for field in F:
  t=row['targets'][field];top=t['top10'];target=t['gold']['code'];assert len(top)==10
  for k in [1,5,10]:hits[k]+=target in [x['code'] for x in top[:k]]
assert n==5000
with P.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
out={'source':str(P),'source_sha256':h,'records':n,'decisions':n*8,'metrics':{str(k):{'hits':hits[k],'pct':hits[k]/(n*8)*100} for k in [1,5,10]},'provenance':'Export error_analysis generated with final-v5 directory per archived script and report. Hash identifies export exactly; current weight hash alone cannot retroactively prove historical runtime identity.','limitation':'Supports naming exact existing final-v5 export; does not create a new prospectively frozen matched ablation.'};(R/'phase1_final_export_verified.json').write_text(json.dumps(out,indent=2)+'\n');print(out['metrics'])
