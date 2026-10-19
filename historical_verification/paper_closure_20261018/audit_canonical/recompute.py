"""Fixed-file, fixed-population offline recomputation; public outputs aggregates only."""
from pathlib import Path
import hashlib,json,sys
from collections import Counter,defaultdict
import numpy as np
from canonical_audit import *
R=Path(__file__).resolve().parent
D=Path('/Volumes/university data/tesi/benchs/dual encoder')
AP=D/'benchmarks/gemma4_singleprompt_xlarge_final_full/deepseek-v4-flash-gold-audit-full-8fields.jsonl'
PP=D/'benchmarks/gemma4_singleprompt_xlarge_final_full/benchmark-llm-rerank-ollama-gemma4-12b-v6-hardneg8-test-full-top10-ctx8192-hier-singleprompt-gpu-xlarge.predictions.jsonl'
TP=D/'tassonomia.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hash_keys(ks):return hashlib.sha256(('\n'.join(sorted(ks))+'\n').encode()).hexdigest()
tax=json.loads(TP.read_text());audits,stats,rejected=load_canonical(AP,tax)
raw=[json.loads(l) for l in PP.open() if l.strip()];pc=Counter(str(r['record_key']) for r in raw)
preds={str(r['record_key']):r for r in raw if pc[str(r['record_key'])]==1}
# Exact nonempty evidence overlap according to original accident_to_text and split.
sys.path.insert(0,str(D))
from esaw_biencoder.data import load_dataset,accident_to_text
import evaluate_llm_predictions_audited_hierarchical as hierarchical
records=load_dataset(D/'dataset.json')
def key(r,i):
 for f in ('id','numero_caso'):
  v=r.get(f) or r.get(f.upper())
  if v is not None and str(v).strip():return str(v).strip()
 return f'index:{i}'
splits=json.loads((D/'artifacts/splits/esaw-fixed-seed13-test10-val10.json').read_text())['splits']
bykey={key(r,i):accident_to_text(r) for i,r in enumerate(records)}
traintexts={bykey[k] for k in splits['train'] if bykey[k]}
overlap={k for k in splits['test'] if bykey[k] and bykey[k] in traintexts}
base=sorted(set(audits)&set(preds))
groups={'canonical':base,'complete_metadata_only':[k for k in base if not audits[k]['missing_metadata']], 'no_known_pending_review_records':[k for k in base if not audits[k]['review_pending']], 'nonempty_evidence':[k for k in base if str(preds[k].get('text','')).strip()], 'empty_evidence':[k for k in base if not str(preds[k].get('text','')).strip()], 'complete_metadata_no_pending_review':[k for k in base if not audits[k]['missing_metadata'] and not audits[k]['review_pending']], 'exclude_train_text_overlap':[k for k in base if k not in overlap]}
# Strict complete metadata excludes missing audit.record_key as well as field metadata.
out={'protocol':{'bootstrap_unit':'record; all eight fields stay paired','bootstrap_replicates':10000,'bootstrap_seed':20261018,'ci':'percentile 2.5/97.5; two-sided; no multiplicity adjustment','primary_population':'all unambiguous 8-field code-valid canonical archival labels including valid labels with pending human review; abstentions excluded','original_silver_comparison':'identical record inclusion and ordering within each group','material_normalization':'only dd.dd or dd.dd.00.00, never truncate nonzero levels'}, 'inputs':{str(p):sha(p) for p in (AP,PP,TP,D/'dataset.json',D/'artifacts/splits/esaw-fixed-seed13-test10-val10.json')},'loader':stats,'prediction_rows':len(raw),'prediction_duplicate_keys':sum(n>1 for n in pc.values()),'audit_accepted_missing_predictions':len(set(audits)-set(preds)),'all_test_exact_nonempty_train_overlap_records':len(overlap),'groups':{}}
private={'rejected_audit_rows':rejected,'accepted_record_details':audits,'group_record_keys':groups,'overlap_test_keys':sorted(overlap)}
for name,ks in groups.items():
 n=len(ks);summary={'records':n,'decisions':8*n,'inclusion_sha256':hash_keys(ks),'review_pending_records':sum(audits[k]['review_pending'] for k in ks),'references':{}}
 source=Counter();quality=Counter();missing=Counter();review=0
 for k in ks:
  missing.update(audits[k]['missing_metadata'])
  for x in audits[k]['items'].values():
   source[x.get('source','MISSING')]+=1;quality[x.get('gold_quality','MISSING')]+=1;review+=x.get('needs_human_review') is True
 summary['composition']={'source':dict(source),'gold_quality':dict(quality),'pending_review_decisions':review,'missing_metadata':dict(missing)}
 for refname in ('original','silver'):
  hits=np.zeros((n,2,8),dtype=np.int8)
  for i,k in enumerate(ks):
   row=preds[k];gold=audits[k]['labels'] if refname=='silver' else {f:normalize_code(f,row['gold'][f]) for f in FIELDS}
   for m,model in enumerate(('dual_top1','llm_top1')):
    for j,f in enumerate(FIELDS):hits[i,m,j]=normalize_code(f,row[model][f].get('code'))==gold[f]
  val={model:{'mean_top1_pct':float(hits[:,m,:].mean()*100),'exact8_pct':float(hits[:,m,:].all(1).mean()*100),'field_top1_pct':{f:float(hits[:,m,j].mean()*100) for j,f in enumerate(FIELDS)}} for m,model in enumerate(('dual_top1','llm_top1'))}
  diff=np.stack((hits[:,1,:].mean(1)-hits[:,0,:].mean(1),hits[:,1,:].all(1).astype(int)-hits[:,0,:].all(1).astype(int)),axis=1)
  rng=np.random.default_rng(20261018);samples=[]
  for _ in range(100):samples.append(diff[rng.integers(0,n,(100,n))].mean(1)*100)
  b=np.concatenate(samples)
  for model in ('dual_top1','llm_top1'):
   gold_map={k:(audits[k]['labels'] if refname=='silver' else {f:normalize_code(f,preds[k]['gold'][f]) for f in FIELDS}) for k in ks}
   ev=hierarchical.evaluate([preds[k] for k in ks],gold_map,tax,model)
   val[model]['hierarchical_evaluation']=ev
  val['dual_candidate_coverage_pct']={str(top):100*sum(normalize_code(f,(audits[k]['labels'] if refname=='silver' else preds[k]['gold'])[f]) in [normalize_code(f,x.get('code')) for x in preds[k]['dual_candidates'][f][:top]] for k in ks for f in FIELDS)/(8*n) for top in (1,5,10)}
  val['paired_gemma_minus_dual']={metric:{'delta_pp':float(diff[:,j].mean()*100),'ci95_pp':np.quantile(b[:,j],[.025,.975]).tolist()} for j,metric in enumerate(('mean_top1','exact8'))}
  summary['references'][refname]=val
 out['groups'][name]=summary
out['full_prediction_original_reference_sensitivity']={}
for name,ks in {'full':sorted(preds),'exclude_exact_nonempty_train_overlap':[k for k in sorted(preds) if k not in overlap]}.items():
 result={'records':len(ks),'inclusion_sha256':hash_keys(ks),'models':{}}
 for model in ('dual_top1','llm_top1'):
  h=np.array([[normalize_code(f,preds[k][model][f].get('code'))==normalize_code(f,preds[k]['gold'][f]) for f in FIELDS] for k in ks])
  result['models'][model]={'mean_top1_pct':float(h.mean()*100),'exact8_pct':float(h.all(1).mean()*100)}
 out['full_prediction_original_reference_sensitivity'][name]=result
 private['group_record_keys']['full_original_'+name]=ks
(R/'private/canonical_details.json').write_text(json.dumps(private,ensure_ascii=False,indent=2))
(R/'canonical_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({'loader':stats,'groups':{k:{'records':v['records'],'silver':v['references']['silver']['paired_gemma_minus_dual']} for k,v in out['groups'].items()}},indent=2))
