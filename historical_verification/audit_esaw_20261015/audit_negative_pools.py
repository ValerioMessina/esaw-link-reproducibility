from pathlib import Path
import json
from collections import Counter
D=Path('/Volumes/university data/tesi/benchs/dual encoder');O=Path(__file__).resolve().parent
spl=json.loads((D/'artifacts/splits/esaw-fixed-seed13-test10-val10.json').read_text())['splits'];train=set(spl['train']);z=D/'runpod_downloads/labelwise_zero_shot_fold0/live'
h=json.loads((z/'artifacts/labelwise_zero_shot/esaw-label-folds-seed13-v1.json').read_text())['folds']['0']['heldout_taxonomy_codes_by_output_field'];out={}
for p,held in [(D/'artifacts/hard_negatives/v5-train-top20.json',None),(z/'artifacts/labelwise_zero_shot/fold0-stage-a-train-top20.json',h)]:
 j=json.loads(p.read_text());c=Counter()
 for f,field in j['fields'].items():
  for k,i in field['items'].items():
   c['training_items']+=1
   if k not in train:c['outside_train_keys']+=1
   if held and i['gold_code'] in held[f]:c['heldout_positive_leaks']+=1
   ns=[]
   for n in i['negatives']:
    code=n if isinstance(n,str) else n.get('taxonomy_code') or n.get('code');ns.append(code);c['negative_codes']+=1
    if held and code in held[f]:c['heldout_negative_leaks']+=1
    if code==i['gold_code']:c['paired_gold_in_negatives']+=1
   if len(ns)!=len(set(ns)):c['duplicated_negative_codes_per_item']+=1
 out[p.name]=dict(c)
 del j
(O/'negative_pool_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
