"""Paired record bootstrap; input NPZ is private, output only aggregate CI."""
import argparse,json
import numpy as np
from check_inputs import external

def compare(a,b,replicates=10000,seed=20261018):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=8 or not len(a):raise ValueError('Expected matched N x 8 matrices')
    if not (np.isin(a,[0,1]).all() and np.isin(b,[0,1]).all()):raise ValueError('Hit matrices must be binary')
    if replicates<1:raise ValueError('Positive bootstrap replicate count required')
    rng=np.random.default_rng(seed);out={}
    delta=np.stack([b.mean(1)-a.mean(1),b.all(1).astype(float)-a.all(1)],axis=1)
    samples=[]
    for start in range(0,replicates,100):
        ix=rng.integers(0,len(delta),(min(100,replicates-start),len(delta)))
        samples.append(delta[ix].mean(1)*100)
    boot=np.concatenate(samples)
    for j,name in enumerate(('mean_top1','exact8')):
        out[name]={'delta_percentage_points':float(delta[:,j].mean()*100),'ci95':np.quantile(boot[:,j],[.025,.975]).tolist()}
    return {'records':len(a),'replicates':replicates,'seed':seed,'paired_over':'records, all eight fields jointly',
            'limitation':'Conditional on fixed systems, data and reference; no training-seed or label uncertainty',**out}

def main():
    p=argparse.ArgumentParser();p.add_argument('--a',required=True);p.add_argument('--b',required=True)
    p.add_argument('--replicates',type=int,default=10000);p.add_argument('--seed',type=int,default=20261018)
    a=p.parse_args()
    with np.load(external(a.a),allow_pickle=False) as x,np.load(external(a.b),allow_pickle=False) as y:
        if not np.array_equal(x['record_keys'],y['record_keys']):raise ValueError('Record identities/order differ')
        print(json.dumps(compare(x['hits'],y['hits'],a.replicates,a.seed),indent=2))
if __name__=='__main__':main()
