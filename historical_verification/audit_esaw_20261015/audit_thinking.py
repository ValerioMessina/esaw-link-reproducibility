from pathlib import Path
from collections import Counter
import json
D=Path('/Volumes/university data/tesi/benchs/dual encoder');R=Path(__file__).resolve().parent;F=['COD_LOC','COD_LAV','COD_ATV','COD_MAT_ATV','COD_DEV','COD_MAT_DEV','COD_CON','COD_MAT_CON']
def rows(p):
 for l in p.open():
  if l.strip():yield json.loads(l)
ref={}
for r in rows(D/'benchmarks/gemma4_singleprompt_xlarge_final_full/deepseek-v4-flash-gold-audit-full-8fields.jsonl'):
 a=r['audit'];items=[]
 for n in ['field_audits','audited_fields','audits','fields_to_audit']:
  if isinstance(a.get(n),list):items.extend(a[n])
 for f in F:
  if isinstance(a.get(f),dict):items.append({'field':f,**a[f]})
 fs={i.get('field'):i for i in items if i.get('field') in F}
 if not r.get('validation_errors') and len(fs)==8 and all(i.get('audited_code') and 'gold_quality' in i for i in fs.values()):ref[str(r['record_key'])]={f:i['audited_code'] for f,i in fs.items()}
think={};dup=0
for p in (D/'benchmarks/gemma4_thinking_top10_runpod_partial_20260820/output').glob('*.predictions.jsonl'):
 for r in rows(p):
  k=r['record_key'];dup+=k in think;think[k]={'gold':r['gold'],'dual':{f:r['dual_top1'][f]['code'] for f in F},'llm':{f:r['llm_top1'][f]['code'] for f in F},'candidates':{f:[x['code'] for x in r['dual_candidates'][f]] for f in F}}
stats={'original':Counter(),'silver':Counter()};diff=Counter()
for r in rows(next((D/'benchmarks/gemma4_singleprompt_xlarge_final_full').glob('benchmark*.predictions.jsonl'))):
 k=r['record_key']
 if k not in think:continue
 t=think[k]
 for f in F:
  diff['lists']+=1;diff['top1_diff']+=r['dual_top1'][f]['code']!=t['dual'][f];diff['list_order_diff']+=[x['code'] for x in r['dual_candidates'][f]]!=t['candidates'][f]
 models={'input_nonthinking':{f:r['dual_top1'][f]['code'] for f in F},'nonthinking':{f:r['llm_top1'][f]['code'] for f in F},'input_thinking':t['dual'],'thinking':t['llm']}
 for reference,gold in [('original',r['gold']),('silver',ref.get(k))]:
  if not gold:continue
  c=stats[reference];c['n']+=1
  for m,preds in models.items():
   h=sum(preds[f]==gold[f] for f in F);c[m+'_hits']+=h;c[m+'_exact8']+=h==8;c[m+'_ge7']+=h>=7;c[m+'_ge6']+=h>=6
out={'thinking_unique_records':len(think),'duplicates':dup,'candidate_differences':dict(diff),'metrics':{name:{'n':c['n'],'systems':{m:{'mean_top1':100*c[m+'_hits']/(8*c['n']),'exact8':100*c[m+'_exact8']/c['n'],'ge7':100*c[m+'_ge7']/c['n'],'ge6':100*c[m+'_ge6']/c['n']} for m in models}} for name,c in stats.items()}}
(R/'thinking_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
