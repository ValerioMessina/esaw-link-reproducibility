"""Portable command adapter. Dry run by default; never provisions compute."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
from check_inputs import ROOT,external,inspect

def build(config,values):
    config_path=Path(config).resolve()
    if ROOT not in config_path.parents:raise ValueError('Use a reviewed repository configuration')
    spec=json.loads(config_path.read_text())
    def expand(v):return v.format(**values)
    return spec,[expand(v) for v in spec['argv']]

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--dataset',required=True)
    p.add_argument('--taxonomy',required=True);p.add_argument('--split-file',required=True)
    p.add_argument('--work-dir',required=True);p.add_argument('--phase1-model');p.add_argument('--refined-model')
    p.add_argument('--hard-negatives');p.add_argument('--data-dir');p.add_argument('--base-model',default='bert-base-uncased')
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    vals={k:str(external(getattr(a,k))) for k in ('dataset','taxonomy','split_file','work_dir')}
    vals.update(root=str(ROOT),python=sys.executable,base_model=a.base_model)
    for k in ('phase1_model','refined_model','hard_negatives','data_dir'):
        if getattr(a,k):vals[k]=str(external(getattr(a,k)))
    spec,argv=build(a.config,vals)
    env=os.environ.copy();paths=[str(ROOT/v) for v in spec.get('pythonpath',[])]
    env['PYTHONPATH']=os.pathsep.join(paths+([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    print(json.dumps({'id':spec['id'],'execution':a.execute,'environment':spec['environment'],
                      'PYTHONPATH':paths,'command':shlex.join(argv)},indent=2),flush=True)
    if a.execute:
        inspect(vals['dataset'],vals['taxonomy'],vals['split_file'],True)
        Path(vals['work_dir']).mkdir(parents=True,exist_ok=True)
        env['HF_HOME']=str(Path(vals['work_dir'])/'hf_cache')
        subprocess.run(argv,check=True,cwd=ROOT/spec['cwd'],env=env)
if __name__=='__main__':main()

