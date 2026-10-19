"""Independent arithmetic/export and decoder checks; no training or parameter fit on test."""
from pathlib import Path
import sys
import json,hashlib,collections
import numpy as np
import torch
R=Path(__file__).resolve().parent/sys.argv[1];P=Path('/Volumes/university data/tesi/benchs/paper_reproductions/data/esaw')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
meta=json.loads((P/'metadata.json').read_text());names=meta['labels'];lookup={s:i for i,s in enumerate(names)};parent={lookup[c]:lookup[p] for p,cs in meta['children'].items() for c in cs}
def depth(v):
 d=0
 while v in parent:d+=1;v=parent[v]
 return d
fields=list(meta['field_leaves']);codes={f:{lookup[n]:c for n,c in meta['output_by_leaf'][f].items()} for f in fields};ids={f:[lookup[n] for n in meta['field_leaves'][f]] for f in fields};truth={}
for line in (P/'esaw_test.json').open():
 row=json.loads(line);y=set(row['label']);truth[row['record_key']]={f:codes[f][max(y.intersection(ids[f]),key=depth)] for f in fields}
assert len(truth)==5000
frozen=json.loads((R/'decoder_frozen.json').read_text());assert frozen['test_opened'] is False;assert sha(R/'protocol.json')==frozen['protocol_sha256'];assert sha(R/'validation_scores.npz')==frozen['validation_scores_sha256'];scores=np.load(R/'test_scores.npz');test_logits=scores['logits'];test_keys=scores['record_keys'];report={};saved=json.loads((R/'test_metrics.json').read_text())
devrows=[json.loads(line) for line in (P/'esaw_dev.json').open()]
valcheck=np.load(R/'validation_scores.npz')
assert list(valcheck['record_keys'])==[r['record_key'] for r in devrows]
for i,row in enumerate(devrows):
 for j,f in enumerate(fields):assert valcheck['gold_ids'][i,j]==max(set(row['label']).intersection(ids[f]),key=depth)
# Refitting final offsets from validation alone with an independent loop and exact integer hit counts.
val=np.load(R/'validation_scores.npz');refit={}
for j,f in enumerate(fields):
 vs=np.array(ids[f]);child=np.array([depth(v)==2 for v in vs]);base=val['logits'][:,vs].astype(np.float64);g=val['gold_ids'][:,j];candidates=[]
 for quarter in range(-48,49):
  offset=quarter/4;pred=vs[np.argmax(base+child*offset,axis=1)];candidates.append((int(np.count_nonzero(pred==g)),-abs(offset),-quarter,offset))
 refit[f]=max(candidates)[3]
assert refit==frozen['offsets'];report['validation_only_offsets_independently_verified']=True
for mode,metric_key in [('A_raw','A_raw'),('B_validation_calibrated','B_validation_calibrated')]:
 hits=collections.Counter();exact=collections.Counter();parent_hits=parent_n=child_hits=child_n=0;seen=set();rows=[json.loads(line) for line in (R/(mode+'_test_predictions.jsonl')).open()]
 assert len(rows)==5000
 for i,row in enumerate(rows):
  key=row['record_key'];assert key not in seen;seen.add(key);assert key==test_keys[i];assert row['gold']==truth[key];good=collections.Counter()
  for f in fields:
   allowed=set(codes[f].values());top=row['top10'][f];assert len(top)==10 and len(set(top))==10 and set(top)<=allowed
   vs=np.array(ids[f]);z=test_logits[i,vs]
   if mode=='A_raw':rank=torch.topk(torch.tensor(z).sigmoid(),10).indices.numpy()
   else:
    adj=z.astype(np.float64)+frozen['offsets'][f]*np.array([depth(v)==2 for v in vs]);rank=np.lexsort((np.arange(len(vs)),-adj))[:10]
   assert top==[codes[f][int(vs[k])] for k in rank],(mode,key,f)
   target=truth[key][f];node=next(n for n,c in codes[f].items() if c==target);success=top[0]==target
   if depth(node)==2:child_n+=1;child_hits+=success
   else:parent_n+=1;parent_hits+=success
   for k in [1,5,10]:h=target in top[:k];hits[k]+=h;good[k]+=h
  for k in [1,5,10]:exact[k]+=good[k]==8
 assert seen==set(truth)
 result={'records':len(rows),'decisions':len(rows)*8,'hits':dict(hits),'all8_counts':dict(exact),'parent_gold_count':parent_n,'parent_gold_hits':parent_hits,'child_gold_count':child_n,'child_gold_hits':child_hits}
 for k in [1,5,10]:
  assert abs(100*hits[k]/40000-saved[metric_key]['mean_top'+str(k)+'_pct'])<1e-9
  assert abs(100*exact[k]/5000-saved[metric_key]['record_all8_top'+str(k)+'_pct'])<1e-9
 assert abs(100*child_hits/child_n-saved[metric_key]['gold_child_accuracy_pct'])<1e-9;assert abs(100*parent_hits/parent_n-saved[metric_key]['gold_parent_accuracy_pct'])<1e-9
 report[mode]=result
report['decoder_sha256']=sha(R/'decoder_frozen.json');assert report['decoder_sha256']==saved['decoder_sha256'];report['passed']=True;report['scope']='Original test gold, every exported ranking and every top1/5/10 aggregate verified; final offset choice independently rederived from validation only.'
(R/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
