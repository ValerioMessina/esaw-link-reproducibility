from pathlib import Path
from collections import defaultdict
import json,numpy as np
B=Path('/Volumes/university data/tesi/benchs/cross encoder/benchmarks/cluster_final');R=Path(__file__).resolve().parent;out={};rng=np.random.default_rng(20261015)
for p in sorted(B.glob('predictions*.jsonl')):
 d=defaultdict(list)
 for l in p.open():
  r=json.loads(l);d[r['record_key']].append((r['gold_code']==r['rerank_code'],r['gold_code']==r['retriever_code']))
 a=np.array([[sum(x for x,y in v)/8-sum(y for x,y in v)/8,int(all(x for x,y in v))-int(all(y for x,y in v))] for v in d.values()]);n=len(a)
 boot=np.concatenate([a[rng.integers(0,n,(100,n))].mean(1) for _ in range(50)])*100
 out[p.name]={'record_count':n,'paired_record_bootstrap_samples':5000,'delta_top1_pp':100*a[:,0].mean(),'delta_exact8_pp':100*a[:,1].mean(),'ci95_top1_pp':list(np.quantile(boot[:,0],[.025,.975])),'ci95_exact8_pp':list(np.quantile(boot[:,1],[.025,.975]))}
(R/'cross_uncertainty.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
