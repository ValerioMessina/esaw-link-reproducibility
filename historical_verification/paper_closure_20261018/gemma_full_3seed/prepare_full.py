"""Prepare all 5000 original prompts offline. Does not allocate or call any model."""
from pathlib import Path
import sys,json,hashlib,random,time,shutil
R=Path(__file__).resolve().parent;M=R.parent/'gemma_matched'
D=Path('/Volumes/university data/tesi/benchs/dual encoder');sys.path.insert(0,str(D))
import benchmark_llm_rerank_once as source
from esaw_biencoder.data import load_dataset
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
old=json.loads((M/'protocol.json').read_text())
for n,h in old['sha256'].items():assert sha(Path(n))==h,n
pred_path=next(Path(n) for n in old['sha256'] if n.endswith('.predictions.jsonl'))
preds=[json.loads(l) for l in pred_path.open() if l.strip()];assert len(preds)==5000
data=load_dataset(D/'dataset.json');records={source.record_key(r,i):r for i,r in enumerate(data)}
assert len(set(str(r['record_key']) for r in preds))==5000
private=R/'private';private.mkdir(exist_ok=True);tasks=private/'tasks.jsonl';assert not tasks.exists(),'Do not overwrite frozen input'
order=list(range(5000));random.Random(20261018).shuffle(order)
with tasks.open('w') as f:
    for n,i in enumerate(order):
        row=preds[i];c=row['dual_candidates'];assert set(c)==set(old['fields'])
        assert all(len(c[k])==10 for k in old['fields'])
        prompt=source.single_record_prompt(records[str(row['record_key'])],c,conservative=False,hierarchical=True)
        item={'pair_index':n,'record_key':str(row['record_key']),'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'candidates':{k:[str(x['code']) for x in c[k]] for k in old['fields']},'gold':row['gold'],'dual':{k:row['dual_top1'][k]['code'] for k in old['fields']}}
        f.write(json.dumps(item,ensure_ascii=False)+'\n')
shutil.copy2(M/'model_manifest.json',R/'model_manifest.json')
out={'prepared_epoch':time.time(),'status':'prepared_not_started_budget_and_deadline_pending','model':old['model'],'manifest_sha256':old['manifest_sha256'],'runtime':old['runtime'],'records':5000,'fields':old['fields'],'generation_seeds':[13,21,42],'modes':[False,True],'planned_requests':30000,'sampling':'all original test records, fixed shuffled order seed20261018','options':{'temperature':0,'num_ctx':16384,'num_predict':8192},'prompt':'unmodified original SSD single_record_prompt(conservative=False,hierarchical=True), no mode-only hint','candidates':'same original ordered Top10 for all six requests per record','mode_order':'alternate by (pair_index + seed_index) parity','failure_policy':'no silent fallback, retain invalid/truncated/errors, primary invalid as wrong; dual fallback only sensitivity','primary_reference':'original labels; silver sensitivity separate and explicitly non-independent','retrospective':True,'test_already_observed':True,'seed_interpretation':'generation seeds only, no retraining; temperature0 may yield identical outputs and does not provide three independent training replications','token_limitation':'8192 is a maximum, not a guarantee reasoning completes; test ceiling must not be changed after outcome inspection','completion':'full result only if every record has both modes for each of all3 seeds; report any unfinished protocol','source_sha256':old['sha256'],'tasks_sha256':sha(tasks),'dataset_sha256':sha(D/'dataset.json'),'new_compute_authorized_within_required_budget':False,'required_before_start':['explicit total budget and revised deadline','new-limit throughput/completion pilot on validation or non-test data','resource price and capacity read','hard guards, cumulative cost tracking and verified SSD recovery']}
(R/'protocol_prepared.json').write_text(json.dumps(out,indent=2));print(json.dumps({'prepared':True,'records':5000,'requests':30000,'tasks_sha256':out['tasks_sha256'],'no_llm_calls':True}))
