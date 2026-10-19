from pathlib import Path
import sys,os,json,hashlib,time,runpy
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
R=Path(__file__).resolve().parent/'qwen_final_reinference';D=Path('/Volumes/university data/tesi/benchs/dual encoder');R.mkdir(exist_ok=True);private=R/'private';private.mkdir(exist_ok=True,mode=0o700)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
import torch,transformers,sentence_transformers,platform
actual_runtime={'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'sentence_transformers':sentence_transformers.__version__,'platform':platform.platform()}
pilot=json.loads((R/'pilot_protocol_frozen.json').read_text());assert (R/'pilot_result.json').exists()
for p,h in pilot['files_sha256'].items():assert sha(Path(p))==h,p
protocol={**pilot,'runtime':actual_runtime,'pilot_runtime':pilot['runtime'],'phase':'full final-v5 test reinference','split':'test','records':5000,'sample':'all records original fixed split','pilot_protocol_sha256':sha(R/'pilot_protocol_frozen.json'),'pilot_result_sha256':sha(R/'pilot_result.json'),'runner':'original error_analysis_top10.py unmodified; same default32 batch; offline mps device automatic','retrospective':True,'selection_performance':False,'started_epoch':time.time(),'wrapper_sha256':sha(Path(__file__))};(R/'full_protocol_frozen.json').write_text(json.dumps(protocol,indent=2)+'\n');freeze=sha(R/'full_protocol_frozen.json')
import torch
from sentence_transformers import SentenceTransformer
assert torch.backends.mps.is_available();torch.set_num_threads(4)
os.chdir(D);sys.path.insert(0,str(D));sys.argv=[str(D/'error_analysis_top10.py'),'--model',str(D/'models/esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256'),'--dataset',str(D/'dataset.json'),'--taxonomy',str(D/'tassonomia.json'),'--split-file',str(D/'artifacts/splits/esaw-fixed-seed13-test10-val10.json'),'--split','test','--sample-size','0','--text-format','v5','--top-k','10','--batch-size','32','--predictions-output',str(private/'final_v5_predictions.jsonl'),'--summary-output',str(R/'final_v5_summary.json')]
runpy.run_path(str(D/'error_analysis_top10.py'),run_name='__main__')
for p,h in pilot['files_sha256'].items():assert sha(Path(p))==h,p
assert sha(R/'full_protocol_frozen.json')==freeze
out={'completed':True,'finished_epoch':time.time(),'protocol_sha256':freeze,'predictions_sha256':sha(private/'final_v5_predictions.jsonl'),'summary_sha256':sha(R/'final_v5_summary.json')};(R/'full_completed.json').write_text(json.dumps(out,indent=2)+'\n')
