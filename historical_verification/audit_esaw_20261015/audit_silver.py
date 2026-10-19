from pathlib import Path
from collections import Counter,defaultdict
import json,sys
R=Path(__file__).resolve().parent;D=Path('/Volumes/university data/tesi/benchs/dual encoder');sys.path.insert(0,str(D))
from esaw_biencoder.schema import TARGET_FIELDS
from esaw_biencoder.data import load_taxonomy,label_to_taxonomy_code
F=[t.output_field for t in TARGET_FIELDS];T={t.output_field:t for t in TARGET_FIELDS};tax=load_taxonomy(D/'tassonomia.json')
def rows(p):
 for l in p.open():
  if l.strip():yield json.loads(l)
ap=D/'benchmarks/gemma4_singleprompt_xlarge_final_full/deepseek-v4-flash-gold-audit-full-8fields.jsonl';strict={};bad_records=set();counts=Counter();src=Counter();quality=Counter();errors=Counter();provenance=Counter()
for r in rows(ap):
 a=r.get('audit',{});items=[]
 for n in ['field_audits','audited_fields','audits','fields_to_audit']:
  if isinstance(a.get(n),list):items.extend(a[n])
 for f in F:
  if isinstance(a.get(f),dict):items.append({'field':f,**a[f]})
 fs={i.get('field'):i for i in items if i.get('field') in F}
 complete=not r.get('validation_errors') and set(fs)==set(F) and all(i.get('audited_code') and 'gold_quality' in i for i in fs.values())
 if not complete:continue
 strict[str(r['record_key'])]={f:i['audited_code'] for f,i in fs.items()}
 payloads={p['field']:p for p in r.get('request',{}).get('fields_to_audit',[])}
 for f,i in fs.items():
  src[i['source']]+=1;quality[i['gold_quality']]+=1
  if label_to_taxonomy_code(T[f],i['audited_code']) not in tax[T[f].taxonomy]:
   errors['invalid_taxonomy_code']+=1;bad_records.add(str(r['record_key']))
  p=payloads.get(f,{})
  allowed={c['code'] for c in p.get('allowed_choices',[])}
  if allowed and i['audited_code'] not in allowed:errors['outside_allowed_choices']+=1
  expected={'keep_gold':p.get('gold',{}).get('code'),'dual':p.get('dual_top1',{}).get('code'),'gemma':p.get('gemma_top1',{}).get('code')}.get(i['source'])
  if expected and i['audited_code']!=expected:errors['source_code_mismatch']+=1
 for n in ['model','provider','model_revision','prompt_hash','reviewer','timestamp','created_at','provenance']:
  if n in r:provenance[n]+=1
metrics=defaultdict(lambda:Counter());orig_fixed=Counter();fields=defaultdict(lambda:Counter())
pp=next((D/'benchmarks/gemma4_singleprompt_xlarge_final_full').glob('benchmark*.predictions.jsonl'))
for r in rows(pp):
 k=str(r['record_key'])
 if k not in strict:continue
 ref=strict[k];nonempty=bool(r['text'].strip());groups=['all','nonempty' if nonempty else 'empty']
 if k not in bad_records:groups.append('valid_taxonomy_records_only')
 for group in groups:
  c=metrics[group];c['records']+=1
  for model in ['dual_top1','llm_top1']:
   hit=sum(r[model][f]['code']==ref[f] for f in F);c[model+'_hits']+=hit;c[model+'_exact8']+=hit==8
   if group=='all':
    for f in F:fields[f][model]+=r[model][f]['code']==ref[f]
 orig_fixed['records']+=1
 for model in ['dual_top1','llm_top1']:
  hit=sum(r[model][f]['code']==r['gold'][f] for f in F);orig_fixed[model+'_hits']+=hit;orig_fixed[model+'_exact8']+=hit==8
out={'strict_records':len(strict),'strict_decisions':len(strict)*8,'source_strict':dict(src),'quality_strict':dict(quality),'strict_checks':dict(errors),'row_provenance_fields_present':dict(provenance),'metrics_by_evidence':{g:{'records':c['records'],**{m:{'mean_top1_pct':100*c[m+'_hits']/(8*c['records']),'exact8_pct':100*c[m+'_exact8']/c['records']} for m in ['dual_top1','llm_top1']}} for g,c in metrics.items()},'original_reference_on_silver_population':{'records':orig_fixed['records'],**{m:{'mean_top1_pct':100*orig_fixed[m+'_hits']/(8*orig_fixed['records']),'exact8_pct':100*orig_fixed[m+'_exact8']/orig_fixed['records']} for m in ['dual_top1','llm_top1']}}}
# Field counters increment only in the all-record group.
out['silver_field_accuracy_pct']={f:{m:100*n/len(strict) for m,n in c.items()} for f,c in fields.items()}
(R/'silver_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
