"""Fit validation-only child offsets; freeze before reading external test scores.

Scores NPZ: logits[N,L], gold_ids[N,8], record_keys[N]. Metadata is the
prepared HTC metadata.json. This script implements adjusted-logit ranking;
historical sigmoid/torch.topk tie behavior is retained in the source snapshots.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from check_inputs import external

GRID=np.arange(-12.,12.001,.25)

def layout(meta):
    names=meta['labels'];ids={v:i for i,v in enumerate(names)}
    parent={ids[c]:ids[p] for p,cs in meta['children'].items() for c in cs}
    def depth(i):
        seen=set();d=0
        while i in parent:
            if i in seen:raise ValueError('Hierarchy cycle')
            seen.add(i);d+=1;i=parent[i]
        return d
    depths=np.array([depth(i) for i in range(len(names))])
    fields=list(meta['field_leaves'])
    if len(fields)!=8:raise ValueError('Expected eight fields')
    return fields,[np.array([ids[v] for v in meta['field_leaves'][f]]) for f in fields],depths

def fit(z,g,members,depths):
    out=[]
    for j,vs in enumerate(members):
        a=z[:,vs].astype(np.float64);child=depths[vs]==2
        scores=[int((vs[(a+d*child).argmax(1)]==g[:,j]).sum()) for d in GRID]
        best=max(range(len(GRID)),key=lambda k:(scores[k],-abs(float(GRID[k]))))
        out.append(float(GRID[best]))
    return out

def decode(z,members,depths,offsets,k=10):
    out=[]
    for vs,d in zip(members,offsets):
        a=z[:,vs].astype(np.float64)+d*(depths[vs]==2)
        out.append(vs[np.argsort(-a,axis=1,kind='stable')[:,:min(k,len(vs))]])
    if len({p.shape[1] for p in out})!=1:raise ValueError('Inconsistent candidate counts for requested k')
    return np.stack(out,axis=1)

def metrics(pred,g):
    hit=pred[:,:,0]==g
    out={'records':len(g),'mean_top1_pct':float(hit.mean()*100),'record_exact8_pct':float(hit.all(1).mean()*100)}
    for k in (5,10):
        if pred.shape[2]>=k:out[f'mean_top{k}_pct']=float((pred[:,:,:k]==g[:,:,None]).any(2).mean()*100)
    return out

def load_scores(path,members,labels):
    with np.load(path,allow_pickle=False) as v:z=v['logits'].copy();g=v['gold_ids'].copy();keys=v['record_keys'].copy()
    if z.shape!=(len(keys),labels) or g.shape!=(len(keys),8) or len(set(keys.tolist()))!=len(keys):
        raise ValueError('Invalid score, gold or identity dimensions')
    if not np.isfinite(z).all():raise ValueError('Nonfinite scores')
    for j,vs in enumerate(members):
        if not np.isin(g[:,j],vs).all():raise ValueError('Gold outside field inventory')
    return z,g,keys

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--test');p.add_argument('--output-dir',required=True)
    p.add_argument('--frozen',help='Previously persisted decoder; permits separate test export after fit-only stage')
    p.add_argument('--method',choices=['htc-clip','hisr','hyilr'],required=True,
        help='HGCLR test is gated in its five-seed trainer, not available through this adapter')
    a=p.parse_args();meta_path=external(a.metadata);val=external(a.validation);test=external(a.test) if a.test else None
    if val==test:raise ValueError('Validation and test must be distinct')
    meta=json.loads(meta_path.read_text());fields,members,depths=layout(meta)
    z,g,keys=load_scores(val,members,len(meta['labels']));offsets=fit(z,g,members,depths)
    with np.load(val,allow_pickle=False) as data:checkpoint_sha=str(data['checkpoint_sha256'].item())
    out=external(a.output_dir);out.mkdir(parents=True,exist_ok=True);freeze=out/'decoder_frozen.json'
    payload={'method':a.method,'offsets':dict(zip(fields,offsets)),'metadata_sha256':sha(meta_path),
             'validation_sha256':sha(val),'checkpoint_sha256':checkpoint_sha,'grid':[-12,12,.25],'test_read':False,
             'tie_rule':'smallest absolute offset then grid order; stable taxonomy order for ranking',
             'retrospective':True,'scope':'fixed weights, no source-score replication claim'}
    if a.frozen:
        freeze=external(a.frozen)
        if json.loads(freeze.read_text())!=payload:raise ValueError('Frozen decoder/provenance mismatch')
    else:
        with freeze.open('x') as f:json.dump(payload,f,indent=2)
    digest=sha(freeze)
    if test is None:
        print(json.dumps({'frozen':True,'decoder_sha256':digest,'test_read':False},indent=2));return
    # Test payload is opened only after the complete decoder has been persisted.
    z,g,test_keys=load_scores(test,members,len(meta['labels']))
    with np.load(test,allow_pickle=False) as data:
        if str(data['checkpoint_sha256'].item())!=checkpoint_sha:raise ValueError('Test checkpoint differs from validation')
    if set(keys.tolist())&set(test_keys.tolist()):raise ValueError('Validation/test identity overlap')
    pred=decode(z,members,depths,offsets);result=metrics(pred,g)
    assert sha(freeze)==digest
    result.update(decoder_sha256=digest,test_sha256=sha(test),retrospective=True)
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'private_hits.npz',hits=pred[:,:,0]==g,record_keys=test_keys)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

