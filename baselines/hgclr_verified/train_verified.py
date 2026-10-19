#!/usr/bin/env python3
"""Gated HGCLR retraining. No test reads until all confirmatory runs finish."""
from pathlib import Path
import argparse,os,sys,json,time,random,hashlib,gc,signal,subprocess,platform
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModel
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'vendor'))
from esaw_htc.hgclr_model import Hgclr
from esaw_htc.common import EsawTextDataset,load_hierarchy,read_jsonl,seed_everything
from model.optim import Adam
from metrics import layout,truth_arrays,f1,diagnose,ranking,ranking_metrics,calibrated_offsets,author_threshold
STOP=False

def signal_stop(*_):
    global STOP;STOP=True
signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def log(event,**kw):
    row={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':event,**kw};print(json.dumps(row,allow_nan=False),flush=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    write(OUT/'progress.json',row)

def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def save(path,model,opt,scheduler,epoch,step,mode,extra=None):
    value={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':None if scheduler is None else scheduler.state_dict(),'rng':rng(),'epoch':epoch,'global_step':step,'mode':mode,'config':CFG,'args':vars(ARGS),'data_manifest':DATA_MANIFEST,'protocol_sha256':hashlib.sha256((ROOT/'configs/protocol.json').read_bytes()).hexdigest(),**(extra or {})}
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);tmp.replace(path)

def budget_stop(diagnostic=False):
    now=time.time();deadline=ARGS.deadline
    if diagnostic:deadline=min(deadline,ARGS.diagnostic_deadline)
    return STOP or now>=deadline-120

def make_model(seed):
    seed_everything(seed)
    # Tokenizer is pinned and local, including inside the released graph constructor.
    with patch('model.graph.AutoTokenizer.from_pretrained',return_value=TOK):
        m=Hgclr(ARGS.pretrained,len(META['labels']),ARGS.data_dir,CFG['gamma'],CFG['contrastive_weight'],CFG['temperature'])
    ref=AutoModel.from_pretrained(ARGS.pretrained,local_files_only=True);loaded=m.encoder.load_state_dict(ref.state_dict(),strict=False)
    assert loaded.missing_keys==['embeddings.position_ids']
    assert sorted(loaded.unexpected_keys)==['pooler.dense.bias','pooler.dense.weight']
    delta=max(float((m.encoder.state_dict()[k]-v).abs().max()) for k,v in ref.state_dict().items() if k in m.encoder.state_dict())
    assert delta==0;del ref;gc.collect();return m.to(DEVICE)

def data_loader(rows,batch,shuffle=False,generator=None):
    ds=EsawTextDataset(rows,TOK,len(META['labels']),CFG['max_length']);ds.pretokenize()
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,collate_fn=ds.collate,generator=generator,num_workers=0,pin_memory=DEVICE.type=='cuda')

@torch.no_grad()
def infer(model,loader):
    model.eval();out=[]
    for b in loader:
        b.pop('record_key');b.pop('labels');z=model(**{k:v.to(DEVICE,non_blocking=True) for k,v in b.items()})['logits'];assert torch.isfinite(z).all();out.append(z.cpu().numpy())
    return np.concatenate(out)

def components(model,b,mode):
    h=model.encode(b['input_ids'],b['attention_mask'],b.get('token_type_ids'));z=model.classifier(h);a=F.binary_cross_entropy_with_logits(z,b['labels'])
    w=model.positive_weights(b['input_ids'],b['labels'],b['attention_mask']);hp=model.encode(b['input_ids'],b['attention_mask'],b.get('token_type_ids'),w);zp=model.classifier(hp);bce=F.binary_cross_entropy_with_logits(zp,b['labels']);cl=model.nt_xent(h,hp)
    total=a+bce+(CFG['contrastive_weight']*cl if mode=='full' else 0*cl)
    retained=(w>0)[b['attention_mask'].bool()].float().mean()
    return total,{'bce_original':float(a.detach()),'bce_positive':float(bce.detach()),'ntxent':float(cl.detach()),'token_retained':float(retained.detach())}

def update(model,b,opt,sch,mode):
    model.train();b={k:v.to(DEVICE,non_blocking=True) for k,v in b.items() if k!='record_key'};opt.zero_grad(set_to_none=True);loss,c=components(model,b,mode)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    loss.backward()
    norms=torch.stack([p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]);norm=float(norms.norm())
    if not np.isfinite(norm) or norm<=0:raise FloatingPointError('Nonfinite or zero gradient norm')
    c['gradient_norm']=norm
    opt.step()
    if sch is not None:sch.step()
    return float(loss.detach()),c

def optimizer(model,warmup):
    opt=Adam(model.parameters(),lr=CFG['learning_rate']);sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min((step+1)/warmup,1.)) if warmup else None
    return opt,sch

def diagnostic_run(kind,mode,rows,eval_rows,max_steps,seed,warmup):
    d=OUT/'diagnostics'/f'{kind}_{mode}';d.mkdir(parents=True,exist_ok=True);m=make_model(seed);gen=torch.Generator().manual_seed(seed);loader=data_loader(rows,CFG['batch_size'],True,gen);ev=data_loader(eval_rows,24);opt,sch=optimizer(m,warmup);step=0;hist=[];initial=None
    while step<max_steps:
        for b in loader:
            if step>=max_steps:break
            if budget_stop(True):
                save(d/'last.pt',m,opt,sch,step//len(loader),step,mode,{'dataloader_generator':gen.get_state(),'history':hist});log('diagnostic_budget_stop',kind=kind,mode=mode,step=step);return {'passed':False,'reason':'diagnostic_budget','history':hist}
            loss,c=update(m,b,opt,sch,mode);step+=1
            if step==1 or step%25==0:log('diagnostic_step',kind=kind,mode=mode,step=step,loss=loss,**c)
            if step%100==0 or step==max_steps:
                z=infer(m,ev);stats=diagnose(z,eval_rows,META);hist.append({'step':step,'loss':loss,**stats});write(d/'history.json',hist);np.savez_compressed(d/'validation_or_memorization_logits.npz',logits=z,record_keys=np.array([r['record_key'] for r in eval_rows]));log('diagnostic_evaluation',kind=kind,mode=mode,step=step,metrics=stats)
                if kind=='memorization':
                    gate=stats['nonroot']['micro']>=CFG['memorization']['gate_nonroot_micro_f1'] and stats['nonroot']['exact_node_set']>=CFG['memorization']['gate_exact_nonroot_set'] and stats['deepest_positive_top1']>=CFG['memorization']['gate_deepest_positive_field_top1']
                    if gate:
                        save(d/'last.pt',m,opt,sch,step//len(loader),step,mode,{'dataloader_generator':gen.get_state(),'history':hist});write(d/'gate.json',{'passed':True,'metrics':stats,'step':step});log('memorization_passed',mode=mode,step=step);del m,opt,sch;gc.collect();torch.cuda.empty_cache();return {'passed':True,'step':step,'metrics':stats}
    save(d/'last.pt',m,opt,sch,step//len(loader),step,mode,{'dataloader_generator':gen.get_state(),'history':hist})
    stats=hist[-1];passed=kind!='memorization';res={'passed':passed,'metrics':stats,'step':step};write(d/'gate.json',res);del m,opt,sch;gc.collect();torch.cuda.empty_cache();return res

def do_diagnostics(train,dev):
    gen=np.random.default_rng(CFG['memorization']['seed']);nonempty=[i for i,r in enumerate(train) if r['token'].strip()];empty=[i for i,r in enumerate(train) if not r['token'].strip()]
    idx=np.concatenate([gen.choice(nonempty,124,False),gen.choice(empty,4,False)]);mem=[train[i] for i in idx]
    subidx=gen.choice(len(train),CFG['subset']['records'],False);devix=gen.choice(len(dev),CFG['subset']['validation_records'],False);sub=[train[i] for i in subidx];ev=[dev[i] for i in devix]
    write(OUT/'diagnostics/sampling.json',{'seed':CFG['memorization']['seed'],'memorization_ids':[r['record_key'] for r in mem],'subset_train_ids':[r['record_key'] for r in sub],'subset_dev_ids':[r['record_key'] for r in ev],'test_accessed':False})
    # Contrast off still computes the same positive view and projector so random
    # draws and positive-BCE supervision stay matched up to diverging weights.
    results={}
    for mode in ['no_contrast','full']:
        results['memorization_'+mode]=diagnostic_run('memorization',mode,mem,mem,CFG['memorization']['steps'],CFG['memorization']['seed'],0)
        if not results['memorization_'+mode]['passed']:
            write(OUT/'diagnostics/gates.json',{'passed':False,'results':results,'reason':'Memorization gate failed; full training forbidden.'});return False
    for mode in ['no_contrast','full']:
        results['subset_'+mode]=diagnostic_run('subset',mode,sub,ev,int(np.ceil(len(sub)/CFG['batch_size']))*CFG['subset']['epochs'],CFG['subset']['seed'],CFG['subset']['warmup_steps'])
    # Dependence on text is assessed on nonempty narratives, not their trivial split.
    for mode in ['no_contrast','full']:
        d=OUT/'diagnostics'/('subset_'+mode);z=np.load(d/'validation_or_memorization_logits.npz')['logits'];mask=np.array([bool(r['token'].strip()) for r in ev]);stats=diagnose(z[mask],[r for r in ev if r['token'].strip()],META)
        _,g=truth_arrays([r for r in ev if r['token'].strip()],META);off=calibrated_offsets(z[mask],g,META);adapted=ranking(z[mask],META,off);distinct=len(np.unique(np.stack([a[:,0] for a in adapted],1),axis=0));stats['distinct_depth_calibrated_tuples']=int(distinct)
        passed=distinct>=10 and stats['mean_logit_std_nonroot']>=.02
        results['subset_'+mode]['text_dependence_nonempty']=stats;results['subset_'+mode]['passed']=passed
    passed=all(v['passed'] for v in results.values());write(OUT/'diagnostics/gates.json',{'passed':passed,'results':results,'test_accessed':False});log('diagnostics_complete',passed=passed);return passed

def raw_ranks(z):
    # Exactly the original decoder: FP32 sigmoid followed by torch.topk.
    s=torch.sigmoid(torch.from_numpy(z));_,cs,_,_=layout(META)
    return [c[torch.topk(s[:,c],k=min(10,len(c)),dim=1).indices.numpy()] for c in cs]

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])

def full_run(seed,train,dev):
    d=OUT/'final'/f'seed{seed}';d.mkdir(parents=True,exist_ok=True)
    if (d/'status.json').exists() and json.loads((d/'status.json').read_text())['complete']:return True
    m=make_model(seed);gen=torch.Generator().manual_seed(seed);cached=data_loader(train,CFG['batch_size']).dataset;ev=data_loader(dev,24);opt,sch=optimizer(m,CFG['full_warmup_steps']);best_macro=-1;best_micro=-1;patience=0;hist=[];step=0;converged=False;start_epoch=1;cursor=0;order=None;losses=[]
    if (d/'last.pt').exists():
        ck=torch.load(d/'last.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler']);restore_rng(ck['rng']);gen.set_state(ck['dataloader_generator']);best_macro=ck['best_macro'];best_micro=ck['best_micro'];patience=ck['patience'];hist=ck['history'];step=ck['global_step'];start_epoch=ck['next_epoch'];cursor=ck['batch_cursor'];order=ck.get('epoch_permutation');losses=ck.get('partial_epoch_losses',[]);del ck
        log('resumed',seed=seed,epoch=start_epoch,batch_cursor=cursor,step=step)
    for epoch in range(start_epoch,CFG['full_max_epochs']+1):
        if order is None:order=torch.randperm(len(train),generator=gen).tolist();losses=[]
        loader=DataLoader(cached,batch_size=CFG['batch_size'],sampler=order,collate_fn=cached.collate,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(seed+epoch));t=time.time()
        for ix,b in enumerate(loader):
            if ix<cursor:continue
            if budget_stop():
                save(d/'last.pt',m,opt,sch,epoch,step,'full',{'next_epoch':epoch,'batch_cursor':ix,'epoch_permutation':order,'partial_epoch_losses':losses,'dataloader_generator':gen.get_state(),'history':hist,'best_macro':best_macro,'best_micro':best_micro,'patience':patience,'epoch_boundary':False});write(d/'status.json',{'complete':False,'reason':'budget_or_signal','epoch':epoch,'step':step});return False
            loss,c=update(m,b,opt,sch,'full');step+=1;losses.append(loss)
            if step%100==0:log('full_step',seed=seed,epoch=epoch,step=step,loss=loss,**c)
        z=infer(m,ev);stats=diagnose(z,dev,META);micro=stats['hierarchical']['micro'];macro=stats['hierarchical']['macro'];patience+=1
        if micro>best_micro:best_micro=micro;patience=0
        improved=macro>best_macro
        if improved:best_macro=macro;patience=0
        row={'epoch':epoch,'step':step,'mean_loss':float(np.mean(losses)),'seconds_since_resume_or_epoch_start':time.time()-t,'patience':patience,'diagnostics':stats};hist.append(row)
        extra={'history':hist,'best_macro':best_macro,'best_micro':best_micro,'patience':patience,'dataloader_generator':gen.get_state(),'epoch_boundary':True,'next_epoch':epoch+1,'batch_cursor':0,'epoch_permutation':None,'partial_epoch_losses':[]}
        save(d/'last.pt',m,opt,sch,epoch,step,'full',extra)
        if improved:
            import shutil
            shutil.copyfile(d/'last.pt',d/'best_macro.pt');np.savez_compressed(d/'validation_logits.npz',logits=z,record_keys=np.array([r['record_key'] for r in dev]));write(d/'best.json',{'epoch':epoch,'step':step,'macro':macro,'micro':micro})
        write(d/'history.json',hist);log('full_epoch',seed=seed,**row);cursor=0;order=None
        if patience>=CFG['patience']:converged=True;break
    write(d/'status.json',{'complete':converged,'reason':'early_stopping' if converged else 'epoch_cap','epoch':epoch,'step':step});del m,opt,sch;gc.collect();torch.cuda.empty_cache();return converged

def export_test(seed,test,dev):
    d=OUT/'final'/f'seed{seed}';ck=torch.load(d/'best_macro.pt',map_location='cpu',weights_only=False);m=make_model(seed);load=m.load_state_dict(ck['model'],strict=True);assert not load.missing_keys and not load.unexpected_keys
    for k,v in m.state_dict().items():
        if v.is_floating_point():assert torch.isfinite(v).all(),k
    zdev=np.load(d/'validation_logits.npz')['logits'];_,golddev=truth_arrays(dev,META);offsets=calibrated_offsets(zdev,golddev,META);write(d/'decoder_B.json',{'fit_split':'validation','offsets':dict(zip(layout(META)[0],offsets)),'grid':CFG['decoder_B_grid'],'checkpoint_selection':'same source-style Macro-F1 as A','retrospective_development':True})
    # First use of test records by this training program.
    z=infer(m,data_loader(test,24));np.savez_compressed(d/'test_logits.npz',logits=z,record_keys=np.array([r['record_key'] for r in test]));y,gold=truth_arrays(test,META);a=raw_ranks(z);b=ranking(z,META,offsets);fields,cs,codes,depth=layout(META)
    summary={'seed':seed,'checkpoint_epoch':ck['epoch'],'A_raw_released':ranking_metrics(a,gold,META),'B_validation_depth_calibrated':ranking_metrics(b,gold,META),'hierarchical_threshold_f1':f1(y,author_threshold(z)),'nonroot_threshold_f1':f1(y,author_threshold(z),depth>0),'decoder_B_offsets':offsets,'test_records':len(test)}
    codeby={f:{int(n):str(code) for n,code in zip(c,co)} for f,c,co in zip(fields,cs,codes)}
    with (d/'predictions.jsonl').open('w') as f:
        for i,r in enumerate(test):
            row={'record_key':r['record_key'],'gold':{field:codeby[field][int(gold[i,j])] for j,field in enumerate(fields)}}
            for name,ranks in [('A',a),('B',b)]:row[name+'_top10']={field:[codeby[field][int(n)] for n in ranks[j][i]] for j,field in enumerate(fields)}
            f.write(json.dumps(row)+'\n')
    write(d/'benchmark.json',summary);log('test_export_complete',seed=seed,metrics=summary);del m,ck;gc.collect();torch.cuda.empty_cache();return summary

def main():
    global ARGS,CFG,OUT,DEVICE,TOK,META,DATA_MANIFEST
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',required=True);parser.add_argument('--pretrained',required=True);parser.add_argument('--output',required=True);parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--diagnostic-deadline',type=float,required=True);parser.add_argument('--diagnostics-only',action='store_true');parser.add_argument('--resume-after-diagnostics',action='store_true');ARGS=parser.parse_args();OUT=Path(ARGS.output);OUT.mkdir(parents=True,exist_ok=True);CFG=json.loads((ROOT/'configs/protocol.json').read_text());DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu');assert DEVICE.type=='cuda','GPU training required; refusing silent CPU fallback'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    TOK=AutoTokenizer.from_pretrained(ARGS.pretrained,local_files_only=True);META=json.loads((Path(ARGS.data_dir)/'metadata.json').read_text());assert len(META['labels'])==826
    DATA_MANIFEST={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(ARGS.data_dir).iterdir() if p.is_file() and p.suffix in ['.json','.pt']};write(OUT/'data_manifest.json',DATA_MANIFEST)
    write(OUT/'runtime.json',{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'platform':platform.platform(),'args':vars(ARGS),'config':CFG});write(OUT/'pip_freeze.json',subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines())
    train=read_jsonl(Path(ARGS.data_dir)/'esaw_train.json');dev=read_jsonl(Path(ARGS.data_dir)/'esaw_dev.json');assert len(train)==40000 and len(dev)==5000
    log('started',device=str(DEVICE),train=len(train),validation=len(dev),test_accessed=False)
    if ARGS.resume_after_diagnostics:passed=json.loads((OUT/'diagnostics/gates.json').read_text())['passed']
    else:passed=do_diagnostics(train,dev)
    if not passed:log('stopped_diagnostic_failure',full_training_started=False,test_accessed=False);return 2
    if ARGS.diagnostics_only:log('diagnostics_only_complete',passed=True);return 0
    for seed in CFG['seeds']:
        if not full_run(seed,train,dev):log('stopped_incomplete',seed=seed,test_accessed=False);return 3
    if budget_stop():log('budget_before_test');return 3
    test=read_jsonl(Path(ARGS.data_dir)/'esaw_test.json');assert len(test)==5000
    summaries=[export_test(seed,test,dev) for seed in CFG['seeds']]
    aggregate={name:{k:{'mean':float(np.mean([s[name]['mean'][k] for s in summaries])),'seed_sample_std':float(np.std([s[name]['mean'][k] for s in summaries],ddof=1))} for k in ['top1','top5','top10']} for name in ['A_raw_released','B_validation_depth_calibrated']}
    write(OUT/'summary.json',{'complete':True,'seeds':summaries,'aggregate':aggregate,'historical_run_overwritten':False});log('completed',aggregate=aggregate);return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        import traceback
        if 'OUT' in globals():write(OUT/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()})
        raise
