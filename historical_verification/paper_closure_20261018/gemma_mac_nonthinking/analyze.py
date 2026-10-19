"""Independent full-run request verification and record-paired uncertainty."""
from pathlib import Path
import json,hashlib,sys,shutil,time
import numpy as np
R=Path(__file__).resolve().parent;P=json.loads((R/'protocol.json').read_text());D=Path('/Volumes/university data/tesi/benchs/dual encoder');sys.path.insert(0,str(D))
import benchmark_llm_rerank_once as source
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(Path(P['tasks_path']))==P['tasks_sha256'];tasks=[json.loads(l) for l in Path(P['tasks_path']).open() if l.strip()]
files=list((R/'results').glob('*.json'));assert len(files)==5000,'No final significance claim on partial run'
model=[];dual=[];invalid=0;truncated=0;seen=set();proof=[]
for t in tasks:
    path=R/'results'/f"{t['pair_index']:05d}.json";r=json.loads(path.read_text());assert r['record_key']==t['record_key'] and r['record_key'] not in seen;seen.add(r['record_key']);assert r['protocol_sha256']==sha(R/'protocol.json') and r['think'] is False
    body={'model':P['model'],'messages':[{'role':'user','content':t['prompt']}],'stream':False,'think':False,'format':'json','options':P['options']};assert r['request_sha256']==hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();assert r['prompt_sha256']==hashlib.sha256(t['prompt'].encode()).hexdigest()
    payload=r.get('response',{});assert not payload.get('message',{}).get('thinking','').strip(),'Unexpected thinking'
    truncated+=payload.get('done_reason')=='length'
    try:parsed=source.extract_json_object(payload.get('message',{}).get('content',''))
    except Exception:parsed={}
    labels=parsed.get('labels',{}) if isinstance(parsed,dict) else {};labels=labels if isinstance(labels,dict) else {}
    valid={f:str(labels.get(f,'')).strip() in t['candidates'][f] for f in P['fields']};invalid+=sum(not v for v in valid.values())
    model.append([valid[f] and str(labels.get(f,'')).strip()==t['gold'][f] for f in P['fields']]);dual.append([t['dual'][f]==t['gold'][f] for f in P['fields']]);proof.append({'file':path.name,'sha256':sha(path)})
a=np.asarray(model,dtype=bool);b=np.asarray(dual,dtype=bool);rg=np.random.default_rng(20261018);boot=np.empty((10000,4))
for k in range(len(boot)):
    ix=rg.integers(len(a),size=len(a));aa=a[ix];bb=b[ix];boot[k]=[aa.mean(),aa.all(1).mean(),(aa.mean(1)-bb.mean(1)).mean(),(aa.all(1).astype(float)-bb.all(1)).mean()]
out={'time':time.time(),'verified':True,'records':5000,'decisions':40000,'reference':'original labels, retrospective test','mean_top1_pct':float(a.mean()*100),'exact8_pct':float(a.all(1).mean()*100),'dual_top1_pct':float(b.mean()*100),'gemma_minus_dual_pp':float((a.mean()-b.mean())*100),'bootstrap95_pct':dict(zip(['gemma_top1','gemma_exact8','gemma_minus_dual','exact8_gemma_minus_dual'],np.quantile(boot,[.025,.975],axis=0).T.__mul__(100).tolist())),'per_field_top1_pct':dict(zip(P['fields'],a.mean(0).__mul__(100).tolist())),'invalid_fields':invalid,'truncated_responses':truncated,'bootstrap_unit':'record (all8fields together),10000resamples,seed20261018','generation_seed':13,'limitations':['Full fixed historical test, not new independent holdout','Confidence intervals do not establish human semantic correctness of original labels','One frozen model and deterministic generation, no training-seed variability','No guarantee of significant superiority; all effects and intervals reported','Invalid outputs counted as errors, no hidden fallback'],'files':proof}
p=R/'independent_verification.json';p.write_text(json.dumps(out,indent=2));q=Path(P['ssd_directory'])/p.name;shutil.copy2(p,q);assert sha(p)==sha(q)
report=R/'REPORT_NONTHINKING_MAC.md';report.write_text('# Gemma4 12B non-thinking — full test Mac\n\n'+json.dumps({k:v for k,v in out.items() if k!='files'},ensure_ascii=False,indent=2)+'\n');shutil.copy2(report,Path(P['ssd_directory'])/report.name);assert sha(report)==sha(Path(P['ssd_directory'])/report.name)
print(json.dumps({k:v for k,v in out.items() if k!='files'},indent=2))
