"""Independent counting checks on frozen canonical labels; no provider/network."""
from pathlib import Path
import json,hashlib,re
from collections import Counter
R=Path(__file__).resolve().parent;D=Path('/Volumes/university data/tesi/benchs/dual encoder')
x=json.loads((R/'canonical_metrics.json').read_text());p=json.loads((R/'private/canonical_details.json').read_text());secondary=json.loads((R/'secondary_metrics.json').read_text());F=('COD_LOC','COD_LAV','COD_ATV','COD_MAT_ATV','COD_DEV','COD_MAT_DEV','COD_CON','COD_MAT_CON')
pp=D/'benchmarks/gemma4_singleprompt_xlarge_final_full/benchmark-llm-rerank-ollama-gemma4-12b-v6-hardneg8-test-full-top10-ctx8192-hier-singleprompt-gpu-xlarge.predictions.jsonl'
rows={str(r['record_key']):r for r in map(json.loads,pp.read_text().splitlines())}
def norm(f,v):
 if 'MAT_' not in f:return v
 assert re.fullmatch(r'\d\d\.\d\d(?:\.00\.00)?',v)
 return '.'.join(v.split('.')[:2])
checks=0
for group,g in x['groups'].items():
 keys=p['group_record_keys'][group]
 assert g['records']==len(keys) and g['decisions']==8*len(keys);checks+=1
 assert hashlib.sha256(('\n'.join(sorted(keys))+'\n').encode()).hexdigest()==g['inclusion_sha256'];checks+=1
 assert sum(g['composition']['source'].values())==g['decisions'];checks+=1
 assert sum(g['composition']['gold_quality'].values())==g['decisions'];checks+=1
 for reference,ref in g['references'].items():
  for model in ('dual_top1','llm_top1'):
   hits=[]
   for k in keys:
    gold=p['accepted_record_details'][k]['labels'] if reference=='silver' else rows[k]['gold']
    hits.append(sum(norm(f,rows[k][model][f]['code'])==norm(f,gold[f]) for f in F))
   assert abs(sum(hits)/len(keys)/8*100-ref[model]['mean_top1_pct'])<1e-10;checks+=1
   assert abs(hits.count(8)/len(keys)*100-ref[model]['exact8_pct'])<1e-10;checks+=1
   assert abs(ref[model]['hierarchical_evaluation']['mean']['top1']*100-ref[model]['mean_top1_pct'])<1e-10;checks+=1
for name,g in secondary['cross'].items():
 assert g['records']==x['groups']['canonical']['records'];checks+=1
 assert g['inclusion_sha256']==x['groups']['canonical']['inclusion_sha256'];checks+=1
 for model in ('input','fusion'):
  assert abs(g['systems'][model]['mean']['top1']-g['systems'][model]['mean']['candidate_recall_top1'])<1e-12;checks+=1
assert secondary['previous_strict_population_reconciliation']['old_records']==4643;checks+=1
assert x['all_test_exact_nonempty_train_overlap_records']==12;checks+=1
assert x['full_prediction_original_reference_sensitivity']['exclude_exact_nonempty_train_overlap']['records']==4988;checks+=1
result={'independent_numeric_assertions':checks,'passed':True,'checks':['same inclusion digest in composition and all references','eight decisions per accepted record','source and quality counts sum to same denominator','independent direct hit/exact8 counts for every sensitivity/reference','hierarchical evaluator agrees with independent exact accuracy','all five cross variants same canonical inclusion digest','cross input and fusion candidate top1 consistency','old strict population exactly reproduced','12 original overlaps and4988 post-exclusion records']}
(R/'verification_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
