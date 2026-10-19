"""Offline matched populations for cross fusion and thinking; archival drift report."""
from pathlib import Path
import json,hashlib,sys
from collections import Counter,defaultdict
import numpy as np
from canonical_audit import *
R=Path(__file__).resolve().parent;D=Path('/Volumes/university data/tesi/benchs/dual encoder');C=D.parent/'cross encoder';sys.path.insert(0,str(D))
import evaluate_llm_predictions_audited_hierarchical as ev
T=json.loads((D/'tassonomia.json').read_text());ap=D/'benchmarks/gemma4_singleprompt_xlarge_final_full/deepseek-v4-flash-gold-audit-full-8fields.jsonl';pp=D/'benchmarks/gemma4_singleprompt_xlarge_final_full/benchmark-llm-rerank-ollama-gemma4-12b-v6-hardneg8-test-full-top10-ctx8192-hier-singleprompt-gpu-xlarge.predictions.jsonl'
def read(p):
 with p.open() as h:return [json.loads(l) for l in h if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kh(keys):return hashlib.sha256(('\n'.join(sorted(keys))+'\n').encode()).hexdigest()
A,_,_=load_canonical(ap,T);direct={str(r['record_key']):r for r in read(pp)};inputs={str(ap):sha(ap),str(pp):sha(pp)}
private={};out={'inputs':inputs,'bootstrap':{'replicates':10000,'seed':20261018,'unit':'paired record; eight fields together','ci':'percentile 95%'},'cross':{},'thinking':{}}
def results(rows,ks,models,reference):
 gold={k:A[k]['labels'] if reference=='silver' else direct[k]['gold'] for k in ks}
 o={'records':len(ks),'decisions':8*len(ks),'inclusion_sha256':kh(ks),'reference':reference,'systems':{}}
 for model in models:
  evalrows=[{**rows[k], 'dual_candidates':rows[k].get('_candidate_lists_by_model',{}).get(model,rows[k]['dual_candidates'])} for k in ks]
  v=ev.evaluate(evalrows,gold,T,model)
  o['systems'][model]=v
 return o
def delta(rows,ks,left,right,reference):
 vals=[]
 for k in ks:
  gold=A[k]['labels'] if reference=='silver' else direct[k]['gold'];h=[]
  for m in (left,right):h.append([normalize_code(f,rows[k][m][f].get('code'))==normalize_code(f,gold[f]) for f in FIELDS])
  vals.append([(sum(h[1])-sum(h[0]))/8,int(all(h[1]))-int(all(h[0]))])
 a=np.array(vals);rng=np.random.default_rng(20261018);b=np.concatenate([a[rng.integers(0,len(ks),(100,len(ks)))].mean(1)*100 for _ in range(100)])
 return {'direction':right+' minus '+left,**{m:{'delta_pp':float(a[:,i].mean()*100),'ci95_pp':np.quantile(b[:,i],[.025,.975]).tolist()} for i,m in enumerate(('mean_top1','exact8'))}}
# Exact filenames, sorted inventory recorded with hashes (never next(glob)).
crossfiles=sorted((C/'benchmarks/cluster_final').glob('predictions*.jsonl'))
for p in crossfiles:
 inputs[str(p)]=sha(p);raw=read(p);cnt=Counter((str(r['record_key']),r['field']) for r in raw);fields=defaultdict(dict)
 for r in raw:
  if cnt[(str(r['record_key']),r['field'])]==1:fields[str(r['record_key'])][r['field']]=r
 complete={k:v for k,v in fields.items() if set(v)==set(FIELDS)};ks=sorted(set(complete)&set(A)&set(direct));rows={}
 for k in ks:
  rows[k]={'record_key':k,'input':{f:{'code':complete[k][f]['retriever_code']} for f in FIELDS},'fusion':{f:{'code':complete[k][f]['rerank_code']} for f in FIELDS},'dual_candidates':{f:[{'code':v} for v in complete[k][f]['rerank_top10_codes']] for f in FIELDS}}
 for k in ks:rows[k]['_candidate_lists_by_model']={'input':{f:[{'code':v} for v in complete[k][f]['retriever_top10_codes']] for f in FIELDS},'fusion':rows[k]['dual_candidates']}
 res=results(rows,ks,['input','fusion'],'silver');res['paired_delta']=delta(rows,ks,'input','fusion','silver');res['raw_decisions']=len(raw);res['duplicate_record_field_pairs']=sum(n>1 for n in cnt.values());out['cross'][p.name]=res;private[p.name]=ks
# Thinking exports are incomplete execution, no completion/imputation.
thinkfiles=sorted((D/'benchmarks/gemma4_thinking_top10_runpod_partial_20260820/output').glob('*.predictions.jsonl'))
raw=[]
for p in thinkfiles:inputs[str(p)]=sha(p);raw.extend(read(p))
cnt=Counter(str(r['record_key']) for r in raw);thinking={str(r['record_key']):r for r in raw if cnt[str(r['record_key'])]==1};allks=sorted(set(thinking)&set(direct));silverks=sorted(set(allks)&set(A));rows={};differences=Counter()
for k in allks:
 t=thinking[k];d=direct[k];rows[k]={'record_key':k,'input_nonthinking':d['dual_top1'],'nonthinking':d['llm_top1'],'input_thinking':t['dual_top1'],'thinking':t['llm_top1'],'dual_candidates':t['dual_candidates']}
 for f in FIELDS:
  differences['lists']+=1;differences['input_top1_different']+=d['dual_top1'][f]['code']!=t['dual_top1'][f]['code'];differences['ordered_candidates_different']+=[x['code'] for x in d['dual_candidates'][f]]!=[x['code'] for x in t['dual_candidates'][f]]
for k in allks:rows[k]['_candidate_lists_by_model']={'input_nonthinking':direct[k]['dual_candidates'],'nonthinking':direct[k]['dual_candidates'],'input_thinking':thinking[k]['dual_candidates'],'thinking':thinking[k]['dual_candidates']}
for ref,ks in [('original',allks),('silver',silverks)]:
 res=results(rows,ks,['input_nonthinking','nonthinking','input_thinking','thinking'],ref)
 # Candidate coverage in result uses thinking lists for every system; make explicit.
 res['candidate_coverage_list_source']='each configuration uses its own input export; nonthinking and its input share direct lists, thinking and its input share thinking lists'
 res['paired_thinking_minus_nonthinking']=delta(rows,ks,'nonthinking','thinking',ref);out['thinking'][ref]=res;private['thinking_'+ref]=ks
out['thinking']['export_differences']=dict(differences);out['thinking']['raw_rows']=len(raw);out['thinking']['duplicate_record_keys']=sum(n>1 for n in cnt.values())
# Reproduce previous strict population exactly, then quantify removals and additions.
old={};oldrows={}
for r in read(ap):
 a=r.get('audit',{});items=[]
 for n in ALIASES:
  if isinstance(a.get(n),list):items.extend(a[n])
 for f in FIELDS:
  if isinstance(a.get(f),dict):items.append({'field':f,**a[f]})
 fs={i.get('field'):i for i in items if i.get('field') in FIELDS}
 if not r.get('validation_errors') and set(fs)==set(FIELDS) and all(i.get('audited_code') and 'gold_quality' in i for i in fs.values()):old[str(r['record_key'])]=fs;oldrows[str(r['record_key'])]=r
add=set(A)-set(old);remove=set(old)-set(A);reasons=Counter();_,_,reject=load_canonical(ap,T);rr={r['record_key']:r['reasons'] for r in reject}
for k in remove:reasons.update(rr[k])
out['previous_strict_population_reconciliation']={'old_records':len(old),'canonical_records':len(A),'intersection_records':len(set(old)&set(A)),'added_records':len(add),'removed_records':len(remove),'removed_reasons_nonexclusive':dict(reasons),'added_representation':dict(Counter(A[k]['representation'] for k in add)),'added_missing_metadata':dict(sum((Counter(A[k]['missing_metadata']) for k in add),Counter()))}
private['previous_strict_added']=sorted(add);private['previous_strict_removed']=[{'record_key':k,'reasons':rr[k]} for k in sorted(remove)]
# Preserve only aggregate legacy artifacts for evidence; no re-labeling their population.
legacyroot=D.parent/'reports/final_benchmark_artifacts';legacy=sorted(p for p in legacyroot.rglob('*.json') if ('dual12-cross5' in p.name or 'cross-only-deepseek' in p.name or 'same4334' in p.name))
out['unrecomputable_legacy_pipeline_artifacts']={'reason':'Only aggregate JSON found on SSD; corresponding per-record/field JSONL exports unavailable. Canonical re-evaluation and paired inference cannot be derived from aggregates.','action':'Remove old Cross only column and three-stage comparison from canonical silver tables or label them historical with their original population; do not present as canonically matched.','files':{str(p):sha(p) for p in legacy}}
(R/'private/secondary_inclusion.json').write_text(json.dumps(private,indent=2));(R/'secondary_metrics.json').write_text(json.dumps(out,indent=2));print(json.dumps({'reconciliation':out['previous_strict_population_reconciliation'],'cross_populations':{k:v['records'] for k,v in out['cross'].items()},'thinking_populations':{k:out['thinking'][k]['records'] for k in ('original','silver')}},indent=2))
