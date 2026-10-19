from pathlib import Path
from collections import Counter,defaultdict
import json,sys,hashlib,ast
ROOT=Path(__file__).resolve().parent; B=Path('/Volumes/university data/tesi/benchs');D=B/'dual encoder'
sys.path.insert(0,str(D))
from esaw_biencoder.data import load_dataset,load_taxonomy,record_value,label_to_output_code,parent_code_for,label_to_taxonomy_code
from esaw_biencoder.schema import TARGET_FIELDS
F=[t.output_field for t in TARGET_FIELDS];T={t.output_field:t for t in TARGET_FIELDS}
def rows(p):
 for l in p.open():
  if l.strip():yield json.loads(l)
def key(r,i):
 for f in ('id','numero_caso'):
  v=r.get(f) or r.get(f.upper())
  if v is not None and str(v).strip():return str(v).strip()
 return f'index:{i}'
records=load_dataset(D/'dataset.json'); tax=load_taxonomy(D/'tassonomia.json');golds={key(r,i):{t.output_field:label_to_output_code(t,record_value(r,t.field)) for t in TARGET_FIELDS} for i,r in enumerate(records)}
spl=json.loads((D/'artifacts/splits/esaw-fixed-seed13-test10-val10.json').read_text())['splits'];test=set(spl['test'])
def parent(f,c):return parent_code_for(T[f].taxonomy,label_to_taxonomy_code(T[f],c),tax[T[f].taxonomy])
def summarize(rs):
 c=Counter();n=0
 for r in rs:
  n+=1
  for k in [1,5,10]:
   h=sum(r['gold'][f] in r['top10'][f][:k] for f in F);c['top'+str(k)]+=h;c['rr'+str(k)]+=h==8
 return {'records':n,**{k:100*v/(n*(8 if k.startswith('top') else 1)) for k,v in c.items()}}
ps={
 'HGCLR':B/'paper_reproductions/downloads/extracted/results/preliminary_seed13/hgclr/predictions.jsonl',
 'HTC-CLIP':B/'paper_reproductions/downloads/live/runpod_htc_clip_state/results/htc_clip_preliminary/preliminary_seed3_test/predictions.jsonl',
 'HiSR':B/'paper_reproductions/downloads/live/runpod_hisr_state/results/hisr_preliminary/final/seed13/predictions.jsonl',
 'HyILR':B/'paper_reproductions/downloads/live/runpod_hyilr_state/results/hyilr_preliminary/final/seed13/predictions.jsonl'}
out={'baseline_recomputed':{}}
for name,p in ps.items():
 rs=list(rows(p));ks=[r['record_key'] for r in rs];qc=Counter();hier=Counter()
 for r in rs:
  if r['record_key'] not in test:qc['outside_test']+=1
  for f in F:
   if r['gold'][f]!=golds.get(r['record_key'],{}).get(f):qc['gold_mismatches']+=1
   if len(set(r['top10'][f]))!=len(r['top10'][f]):qc['duplicate_codes']+=1
   for c in r['top10'][f]:
    if label_to_taxonomy_code(T[f],c) not in tax[T[f].taxonomy]:qc['invalid_codes']+=1
   c=r['top10'][f][0];g=r['gold'][f]
   hier['parent_top1']+=parent(f,c) is None
   hier['gold_child']+=parent(f,g) is not None
   hier['pred_exact_parent_of_child_gold']+=parent(f,g)==label_to_taxonomy_code(T[f],c)
 out['baseline_recomputed'][name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'unique_records':len(set(ks)),'missing_test_records':len(test-set(ks)),'checks':dict(qc),'metrics_pct':summarize(rs),'parent_bias_pct_of_decisions':{k:100*v/(8*len(rs)) for k,v in hier.items()}}
# Real final phase-1 export, versus paper's checkpoint 10000.
p=D/'error_analysis/v5-qwen3-0.6b-mnrl-seq256-test-flat-top10-predictions.jsonl';cc=Counter();phase_qc=Counter();n=0
for r in rows(p):
 n+=1
 for f,i in r['targets'].items():
  for k,v in [(1,'top1_correct'),(5,'top5_hit'),(10,'top10_hit')]:
   hit=i['gold']['code'] in [x['code'] for x in i['top10'][:k]];cc['top'+str(k)]+=hit
   if hit!=bool(i[v]):phase_qc['saved_flag_mismatch']+=1
  if i['gold']['code']!=golds.get(r['split_key'],{}).get(f):phase_qc['gold_mismatch']+=1
out['phase1_final']={'path':str(p),'records':n,'checks':dict(phase_qc),'metrics_pct':{k:100*v/(8*n) for k,v in cc.items()}}
# Cross exports: row/field identity, fixed inventory, both reranker/input and code-level list invariance.
out['cross_recomputed']={}
for p in sorted((B/'cross encoder/benchmarks/cluster_final').glob('predictions*.jsonl')):
 cs=Counter();record=defaultdict(dict);errors=Counter();weights=defaultdict(set)
 for r in rows(p):
  k=str(r['record_key']);f=r['field'];record[k][f]=r
  if r['gold_code']!=golds.get(k,{}).get(f):errors['gold_mismatch']+=1
  if set(r['rerank_top10_codes'])!=set(r['retriever_top10_codes']):errors['list_membership_changed']+=1
  if k not in test:errors['outside_test']+=1
  weights[f].add(r['retriever_weight'])
  for name,col in [('rerank','rerank_top10_codes'),('retriever','retriever_top10_codes')]:
   for n in [1,5,10]:cs[name+str(n)]+=r['gold_code'] in r[col][:n]
 for rs in record.values():
  if set(rs)!=set(F):errors['incomplete_record']+=1
  for name,col in [('rerank','rerank_top10_codes'),('retriever','retriever_top10_codes')]:cs[name+'_exact8']+=all(rs[f]['gold_code'] in rs[f][col][:1] for f in F)
 out['cross_recomputed'][p.name]={'record_count':len(record),'checks':dict(errors),'metrics_pct':{k:100*v/(len(record)*(1 if 'exact8' in k else 8)) for k,v in cs.items()},'weights':{f:sorted(w) for f,w in weights.items()}}
# Zero-shot independent counting from all 3 prediction files and manifest class membership.
z=D/'runpod_downloads/labelwise_zero_shot_fold0/live';manifest=json.loads((z/'artifacts/labelwise_zero_shot/esaw-label-folds-seed13-v1.json').read_text());held=manifest['folds']['0']['heldout_taxonomy_codes_by_output_field'];out['zero_shot_recomputed']={}
for p in sorted((z/'benchmarks').glob('*test.predictions.jsonl')):
 c=Counter();q=Counter()
 for r in rows(p):
  c['records']+=1;unseen=[]
  for f,i in r['fields'].items():
   un=i['gold_taxonomy_code'] in held[f]
   if un!=i['is_unseen']:q['unseen_flag_mismatch']+=1
   if i['gold_output_code']!=golds.get(r['record_key'],{}).get(f):q['gold_mismatch']+=1
   group='unseen' if un else 'seen';c[group]+=1;c[group+'_generalized_hits']+=i['gold_output_code'] in i['generalized_top10'][:1]
   if un:c['unseen_restricted_hits']+=i['gold_output_code'] in i['restricted_unseen_top10'][:1];unseen.append(i['gold_output_code'] in i['generalized_top10'][:1])
  if unseen:c['records_with_unseen']+=1;c['all_unseen_correct']+=all(unseen)
 out['zero_shot_recomputed'][p.name]={'counts':dict(c),'checks':dict(q),'pct':{'seen_generalized':100*c['seen_generalized_hits']/c['seen'],'unseen_generalized':100*c['unseen_generalized_hits']/c['unseen'],'unseen_restricted':100*c['unseen_restricted_hits']/c['unseen'],'all_unseen_correct':100*c['all_unseen_correct']/c['records_with_unseen']}}
# Validate audit function without importing/running its provider callers.
src=D/'audit_esaw_with_openai.py';tree=ast.parse(src.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='validate_audit');mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]);ns={};exec(compile(ast.fix_missing_locations(mod),str(src),'exec'),ns)
payload={'fields_to_audit':[{'field':f,'allowed_choices':[{'code':'x'}],'gold':{'code':'x'},'dual_top1':{'code':'x'},'gemma_top1':{'code':'x'}} for f in F]}
out['validator_regression_probes']={
 'empty_audit_returns':ns['validate_audit'](payload,{}),
 'empty_field_list_returns':ns['validate_audit'](payload,{'field_audits':[]}),
 'single_field_missing_quality_and_confidence_returns':ns['validate_audit'](payload,{'field_audits':[{'field':F[0],'audited_code':'x','source':'keep_gold'}]}),
 'duplicate_field_returns':ns['validate_audit'](payload,{'field_audits':[{'field':F[0],'audited_code':'x','source':'keep_gold'}]*8})}
(ROOT/'prediction_audit.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
