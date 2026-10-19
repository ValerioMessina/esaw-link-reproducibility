"""Read-only audit: outputs only aggregates and file hashes, never narratives/record IDs."""
from pathlib import Path
from collections import Counter,defaultdict
import json,sys,hashlib,itertools,importlib.util
ROOT=Path(__file__).resolve().parent
B=Path('/Volumes/university data/tesi/benchs')
D=B/'dual encoder'
sys.path.insert(0,str(D))
from esaw_biencoder.data import load_dataset,load_taxonomy,build_explicit_taxonomy,build_candidates,accident_to_text,label_to_taxonomy_code,record_value,normalize_value,parent_code_for
from esaw_biencoder.schema import TARGET_FIELDS
F=[t.output_field for t in TARGET_FIELDS]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r,i):
 for f in ('id','numero_caso'):
  v=r.get(f) or r.get(f.upper())
  if v is not None and str(v).strip():return str(v).strip()
 return f'index:{i}'
def readl(p):
 with p.open() as h:
  for l in h:
   if l.strip():yield json.loads(l)
records=load_dataset(D/'dataset.json');tax=load_taxonomy(D/'tassonomia.json')
splitp=D/'artifacts/splits/esaw-fixed-seed13-test10-val10.json';spl=json.loads(splitp.read_text())['splits'];sets={k:set(v) for k,v in spl.items()}
keys=[key(r,i) for i,r in enumerate(records)]
texts=[accident_to_text(r) for r in records]
part={x:k for k,s in sets.items() for x in s}
out={'files':{str(p):sha(p) for p in [D/'dataset.json',D/'tassonomia.json',splitp,ROOT/'draft/main.tex']},'dataset':{'records':len(records),'nonempty':sum(bool(t) for t in texts),'unique_keys':len(set(keys)),'split_sizes':{k:len(v) for k,v in spl.items()},'split_unique_sizes':{k:len(v) for k,v in sets.items()},'pairwise_split_intersections':{a+'--'+b:len(sets[a]&sets[b]) for a,b in itertools.combinations(sets,2)},'unassigned_records':sum(k not in part for k in keys)}}
freq=Counter();byfield={};invalid=Counter();l2={}
for t in TARGET_FIELDS:
 es=tax[t.taxonomy];c=Counter(label_to_taxonomy_code(t,record_value(r,t.field)) for r in records)
 for code,n in c.items():
  if code not in es:invalid[t.output_field]+=n
  freq[t.output_field+'::'+code]=n
 children={x for x in es if parent_code_for(t.taxonomy,x,es)}
 byfield[t.output_field]={'candidates':len(es),'L1':len(es)-len(children),'L2':len(children),'observed':len(c),'child_assignment_pct':100*sum(c[x] for x in children)/len(records),'missing_codes':c['']}
out['dataset'].update(invalid_codes=dict(invalid),inventory=byfield,observed_classes=len(freq),rare_1_to_50=sum(1<=n<=50 for n in freq.values()),top30_share=sum(n for _,n in freq.most_common(30))/400000,top95_share=sum(n for _,n in freq.most_common(95))/400000)
# Exact evidence duplication across partitions, separately with/without empty evidence.
groups=defaultdict(list)
for i,(k,t) in enumerate(zip(keys,texts)):groups[t].append((part.get(k),i))
dup={}
for a,b in itertools.combinations(sets,2):
 both=[g for t,g in groups.items() if t and {a,b}<={p for p,i in g}]
 dup[a+'--'+b]={'nonempty_shared_texts':len(both),a+'_records':sum(p==a for g in both for p,i in g),b+'_records':sum(p==b for g in both for p,i in g)}
out['dataset']['exact_text_overlap']=dup
out['dataset']['empty_by_split']={s:sum(not t and part[k]==s for k,t in zip(keys,texts)) for s in sets}
# Concrete text-format equivalence check.
cand=build_candidates(build_explicit_taxonomy(tax),'v5')
mat=[t for t in TARGET_FIELDS if t.taxonomy=='agente_materiale']
out['candidate_role_conditioning']={'same_code_candidate_texts_equal':all([x.text for x in cand[mat[0].field]]==[x.text for x in cand[t.field]] for t in mat[1:]),'unique_candidate_texts':len({x.text for cs in cand.values() for x in cs}),'total_candidates':sum(map(len,cand.values()))}
# Audit schemas and strict/legacy loaders.
ap=D/'benchmarks/gemma4_singleprompt_xlarge_final_full/deepseek-v4-flash-gold-audit-full-8fields.jsonl'
audits={};strict={};source=Counter();quality=Counter();review=0;schema=Counter();topkeys=Counter();invalid_source=0;candidate_violations=0;duplicate_audit=0
for r in readl(ap):
 topkeys.update(r.keys());a=r.get('audit',{});k=str(r.get('record_key') or r.get('request',{}).get('record_key') or '')
 items=[]
 if isinstance(a,dict):
  for name in ('audited_fields','audits','fields_to_audit','field_audits'):
   if isinstance(a.get(name),list):items.extend(a[name])
  for f in F:
   if isinstance(a.get(f),dict):items.append({'field':f,**a[f]})
 if k in audits:duplicate_audit+=1
 fs={i.get('field'):i for i in items if i.get('field') in F}
 if r.get('validation_errors'):schema['validation_errors']+=1
 elif len(fs)!=8 or any(not i.get('audited_code') for i in fs.values()):schema['missing_fields']+=1
 elif any('gold_quality' not in i for i in fs.values()):schema['missing_quality']+=1
 else:
  schema['strict_complete']+=1;strict[k]={f:fs[f]['audited_code'] for f in F}
 if not r.get('validation_errors') and len(fs)==8 and all(i.get('audited_code') for i in fs.values()):audits[k]={f:fs[f]['audited_code'] for f in F}
 for i in items:
  if 'gold_quality' in i:
   quality[i['gold_quality']]+=1;source[i.get('source','MISSING')]+=1;review+=bool(i.get('needs_human_review'))
out['audit']={'file_sha256':sha(ap),'rows':sum(topkeys[x] for x in ['case_index']),'schema':dict(schema),'legacy_complete_labels':len(audits),'strict_complete_labels':len(strict),'duplicate_keys':duplicate_audit,'standard_quality_decisions':sum(quality.values()),'quality':dict(quality),'source':dict(source),'review_flags':review,'row_metadata_fields':dict(topkeys)}
# Independent recomputation on direct Gemma export.
pp=next((D/'benchmarks/gemma4_singleprompt_xlarge_final_full').glob('benchmark*.predictions.jsonl'))
results={};seenkeys=Counter();mismatch=Counter();outside=Counter();arrs=defaultdict(list)
def summarize(rows,refname,ref):
 pass
stats=defaultdict(lambda:{'n':0,'correct':Counter(),'exact8':Counter(),'ge7':Counter(),'ge6':Counter(),'hits':Counter(),'rr':Counter()})
for r in readl(pp):
 k=str(r['record_key']);seenkeys[k]+=1
 if k not in sets['test']:mismatch['non_test_row']+=1
 original={f:r['gold'][f] for f in F}
 # Persisted gold versus original dataset.
 if not isinstance(next(iter(original.values())),str):raise ValueError('unexpected gold schema')
 for refname,ref in [('original',original),('silver_legacy',audits.get(k)),('silver_strict',strict.get(k))]:
  if not ref:continue
  ss=stats[refname];ss['n']+=1;delta=[]
  for model in ['dual_top1','llm_top1']:
   correct=sum(str(r[model][f].get('code',''))==str(ref[f]) for f in F)
   ss['correct'][model]+=correct;ss['exact8'][model]+=correct==8;ss['ge7'][model]+=correct>=7;ss['ge6'][model]+=correct>=6
   delta.append(correct/8)
   for f in F:
    if str(r[model][f].get('code','')) not in [str(x['code']) for x in r['dual_candidates'][f]]:outside[model]+=1
  arrs[refname].append([delta[1]-delta[0],int(delta[1]==1)-int(delta[0]==1)])
  for n in [1,5,10]:
   h=sum(str(ref[f]) in [str(x['code']) for x in r['dual_candidates'][f][:n]] for f in F)
   ss['hits'][str(n)]+=h;ss['rr'][str(n)]+=h==8
for ref,ss in stats.items():
 n=ss['n'];results[ref]={'n':n,**{m:{'mean_top1':100*ss['correct'][m]/(8*n),'exact8':100*ss['exact8'][m]/n,'ge7':100*ss['ge7'][m]/n,'ge6':100*ss['ge6'][m]/n} for m in ['dual_top1','llm_top1']},'coverage':{k:100*v/(8*n) for k,v in ss['hits'].items()},'record_recall':{k:100*v/n for k,v in ss['rr'].items()}}
out['direct_gemma_recomputed']={'prediction_sha256':sha(pp),'unique_records':len(seenkeys),'duplicates':sum(v-1 for v in seenkeys.values()),'rows_outside_test':dict(mismatch),'outside_candidates_counts_across_evaluated_populations':dict(outside),'metrics':results}
try:
 import numpy as np
 rng=np.random.default_rng(20261015)
 for name,a in arrs.items():
  a=np.array(a);n=len(a);means=np.concatenate([a[rng.integers(0,n,(100,n))].mean(1) for _ in range(100)])*100
  results[name]['paired_record_bootstrap_95ci_delta_llm_minus_dual']={m:list(np.quantile(means[:,i],[.025,.975])) for i,m in enumerate(['mean_top1_pp','exact8_pp'])}
except ImportError:out['bootstrap']='numpy unavailable'
(ROOT/'data_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,indent=2,ensure_ascii=False))
