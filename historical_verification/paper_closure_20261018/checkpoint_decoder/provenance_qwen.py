from pathlib import Path
import hashlib,json
D=Path('/Volumes/university data/tesi/benchs/dual encoder');R=Path(__file__).resolve().parent
files=['models/esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256/README.md','REPORT_RUNPOD_TRAINING.md','runpod_v5_qwen_mnrl_seq256_train_benchmark.sh','runpod_v6_qwen_hardneg_train_benchmark.sh','runpod_v6_qwen_hardneg8_ablation_train_benchmark.sh','run_local_v5_checkpoint10000_topks_once_benchmark.sh','esaw_biencoder/train.py','models/esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256/training_config.json','models/esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256/model.safetensors','models/esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256/candidates.json','models/esaw-biencoder-v6-qwen3-0.6b-mnrl-hardneg/training_config.json','models/esaw-biencoder-v6-ablation-hardneg8/training_config.json','benchmarks/benchmark-v5-qwen3-0.6b-mnrl-seq256-checkpoint10000-test-flat-top1-top5-top10.json']
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest={'conclusion':'Gli artefatti documentano fase2 inizializzata dal finale v5 a 3 epoche, non checkpoint10000 (~2.5 epoche).','evidence_strength':'Concordanza configurazioni/script/report storici; non esiste qui un hash registrato al momento dell inizializzazione remota, quindi il manifest attuale non dimostra da solo identita byte-per-byte storica.','phase1_reported_row':{'checkpoint':'checkpoint-10000','approx_epoch':2.5,'mean_top1_pct':48.16,'source':'REPORT_RUNPOD_TRAINING.md:1295-1355'},'phase1_final':{'directory':'models/esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256','epochs_configured':3,'historical_report_mean_top1_pct':48.28},'phase2':[],'files':[]}
for n in files:
 p=D/n;manifest['files'].append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
for model in ['esaw-biencoder-v6-qwen3-0.6b-mnrl-hardneg','esaw-biencoder-v6-ablation-hardneg8']:
 cfg=json.loads((D/'models'/model/'training_config.json').read_text());manifest['phase2'].append({'model':model,'base_model':cfg['base_model'],'epochs':cfg['epochs'],'hard_negatives_per_positive':cfg['mnrl_hard_negatives_per_positive'],'resume_from_checkpoint':cfg['resume_from_checkpoint'],'resume_note':'Flag allows continuation within phase2 checkpoint directory. It does not select v5 checkpoint10000.'})
(R/'qwen_checkpoint_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['conclusion'])
