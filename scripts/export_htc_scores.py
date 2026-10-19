"""Portable FP32 score export for the three retrospective HTC adapters.

Requires trusted researcher-supplied checkpoint; files are never distributed.
HGCLR uses its separate five-seed gated trainer and is deliberately excluded.
"""
import argparse,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
from check_inputs import ROOT,external

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['htc-clip','hisr','hyilr'],required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--data-dir',required=True)
    p.add_argument('--base-model',required=True);p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--decoder-frozen')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();data=external(a.data_dir);ckpath=external(a.checkpoint);out=external(a.output)
    checkpoint_sha=sha(ckpath)
    if a.split=='test':
        if not a.decoder_frozen:raise ValueError('Persist validation decoder before test export')
        frozen=json.loads(external(a.decoder_frozen).read_text())
        if frozen['method']!=a.method or frozen['metadata_sha256']!=sha(data/'metadata.json') or frozen['checkpoint_sha256']!=checkpoint_sha:
            raise ValueError('Frozen decoder has different checkpoint, method or taxonomy')
    if out.exists():raise FileExistsError('Refusing to replace a prior score export')
    sys.path[:0]=[str(ROOT/'baselines'),str(ROOT/'baselines/vendor')]
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from esaw_htc.common import load_hierarchy,read_jsonl
    from esaw_htc.htc_clip_model import HtcClip
    from esaw_htc.recent_models import HisrModel,HyilrModel
    torch.set_num_threads(4);device=torch.device(a.device)
    hierarchy=load_hierarchy(data);tok=AutoTokenizer.from_pretrained(a.base_model)
    ck=torch.load(ckpath,map_location='cpu',mmap=True,weights_only=False);args=ck.get('args',{})
    if a.method=='htc-clip':
        with patch('model.graph.AutoTokenizer.from_pretrained',return_value=tok):
            model=HtcClip(a.base_model,len(hierarchy.labels),hierarchy.depth2label,
                [hierarchy.display[n] for n in hierarchy.labels],str(data),
                float(args.get('gamma',.02)),float(args.get('contrastive_weight',.05)),
                float(args.get('temperature',1)),int(args.get('path_hidden',128)))
    elif a.method=='hisr':
        weight=float(args.get('contrastive_weight',.01))
        model=HisrModel(a.base_model,len(hierarchy.labels),hierarchy.children,hierarchy.parent,{},
                       node_loss_weight=weight,sequence_loss_weight=weight)
    else:
        model=HyilrModel(a.base_model,len(hierarchy.labels),hierarchy.depth2label,hierarchy.children,
                        float(args.get('contrastive_weight',.6)),float(args.get('temperature',.07)))
    model.load_state_dict(ck['model'],strict=True);del ck;model.eval().to(device)
    rows=read_jsonl(data/('esaw_dev.json' if a.split=='validation' else 'esaw_test.json'))
    depths={n:d for d,ns in hierarchy.depth2label.items() for n in ns};gold=[];logits=[]
    with torch.inference_mode():
        for i in range(0,len(rows),a.batch_size):
            batch=rows[i:i+a.batch_size]
            inputs=tok([r['token'] for r in batch],padding=True,truncation=True,max_length=512,return_tensors='pt')
            logits.append(model(**{k:v.to(device) for k,v in inputs.items()})['logits'].cpu().numpy())
            for row in batch:
                gold.append([max(set(vs)&set(row['label']),key=lambda n:depths[n]) for vs in hierarchy.field_leaves.values()])
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,logits=np.concatenate(logits),gold_ids=np.array(gold,dtype=np.int64),
                        record_keys=np.array([r['record_key'] for r in rows]),checkpoint_sha256=checkpoint_sha)
    print(json.dumps({'records':len(rows),'split':a.split,'checkpoint_sha256':checkpoint_sha,
                      'export_sha256':sha(out),'no_record_content_logged':True}))
if __name__=='__main__':main()
