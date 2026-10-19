"""Exact-code evaluation and validation-only depth calibration; no model imports."""
import numpy as np

def layout(meta):
    names=meta['labels'];ids={x:i for i,x in enumerate(names)}
    parent={ids[c]:ids[p] for p,cs in meta['children'].items() for c in cs}
    depth=np.zeros(len(names),dtype=np.int64)
    for i in range(len(names)):
        j=i
        while j in parent:depth[i]+=1;j=parent[j]
    fields=list(meta['field_leaves'])
    candidates=[np.array([ids[x] for x in meta['field_leaves'][f]]) for f in fields]
    codes=[np.array([meta['output_by_leaf'][f][names[i]] for i in cs]) for f,cs in zip(fields,candidates)]
    return fields,candidates,codes,depth

def truth_arrays(rows,meta):
    fields,cs,_,depth=layout(meta);y=np.zeros((len(rows),len(meta['labels'])),dtype=bool);gold=np.zeros((len(rows),len(fields)),dtype=np.int64)
    for i,r in enumerate(rows):
        y[i,r['label']]=True
        for j,c in enumerate(cs):
            valid=c[y[i,c]];assert len(valid)>0
            deepest=valid[depth[valid]==depth[valid].max()];assert len(deepest)==1
            gold[i,j]=deepest[0]
    return y,gold

def f1(y,p,mask=None):
    if mask is not None:y=y[:,mask];p=p[:,mask]
    tp=(y&p).sum(0);fp=(~y&p).sum(0);fn=(y&~p).sum(0)
    return {'micro':float(2*tp.sum()/max(1,2*tp.sum()+fp.sum()+fn.sum())),'macro':float(np.mean(2*tp/np.maximum(1,2*tp+fp+fn))),'exact_node_set':float(np.all(y==p,axis=1).mean())}

def calibrated_offsets(z,gold,meta):
    # One offset per field, candidates always include legitimate parent codes.
    _,cs,_,depth=layout(meta);out=[]
    for j,c in enumerate(cs):
        child=(depth[c]==2).astype(np.float32);best=(-1,-float('inf'));delta_best=0.
        for delta in np.arange(-12,12.001,.25):
            pred=c[(z[:,c]+delta*child).argmax(1)];hits=int((pred==gold[:,j]).sum())
            key=(hits,-abs(float(delta)))
            if key>best:best=key;delta_best=float(delta)
        out.append(delta_best)
    return out

def ranking(z,meta,offsets=None):
    _,cs,_,depth=layout(meta);result=[]
    for j,c in enumerate(cs):
        s=z[:,c].copy()
        if offsets is not None:s+=offsets[j]*(depth[c]==2)
        # Stable sorting documents the tie policy for the adapted decoder.
        result.append(c[np.argsort(-s,axis=1,kind='stable')[:,:min(10,len(c))]])
    return result

def ranking_metrics(ranks,gold,meta):
    fields,_,_,_=layout(meta);hits={k:np.stack([np.any(r[:,:min(k,r.shape[1])]==gold[:,j,None],axis=1) for j,r in enumerate(ranks)],axis=1) for k in [1,5,10]}
    return {'fields':{f:{f'top{k}':float(hits[k][:,j].mean()) for k in hits} for j,f in enumerate(fields)},'mean':{f'top{k}':float(hits[k].mean()) for k in hits},'records':{f'all8_at{k}':float(hits[k].all(1).mean()) for k in hits},'counts':{f'hits{k}':int(hits[k].sum()) for k in hits}}

def author_threshold(z):
    # Match released FP32 sigmoid > 0.5, including rounding at zero.
    import torch
    return torch.sigmoid(torch.from_numpy(np.asarray(z,dtype=np.float32))).numpy() > .5

def diagnose(z,rows,meta):
    y,gold=truth_arrays(rows,meta);_,cs,_,depth=layout(meta);r=ranking(z,meta);thresholded=author_threshold(z)
    # Training-capacity probe: deepest node predicted positive, with score fallback.
    deepest=[]
    for c in cs:
        positive=thresholded[:,c];d=np.where(positive,depth[c][None,:],-1);maxd=d.max(1,keepdims=True)
        s=np.where((d==maxd)&positive,z[:,c],-np.inf);no=~positive.any(1);s[no]=z[no][:,c]
        deepest.append(c[s.argmax(1)])
    pred=np.stack(deepest,1)
    return {'hierarchical':f1(y,thresholded),'nonroot':f1(y,thresholded,depth>0),'raw_logits':ranking_metrics(r,gold,meta),'deepest_positive_top1':float((pred==gold).mean()),'distinct_top1_tuples':int(len(np.unique(np.stack([a[:,0] for a in r],1),axis=0))),'mean_logit_std_nonroot':float(z[:,depth>0].std(0).mean()),'mean_probability_std_nonroot':float((1/(1+np.exp(-np.clip(z[:,depth>0],-50,50)))).std(0).mean())}
