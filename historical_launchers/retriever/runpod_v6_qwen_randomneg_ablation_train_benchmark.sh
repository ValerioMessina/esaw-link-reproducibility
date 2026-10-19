#!/usr/bin/env bash
set -euo pipefail

cd /workspace/esaw-biencoder

BASE_MODEL="models/esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256"
MODEL_DIR="models/esaw-biencoder-v6-ablation-randomneg"
CHECKPOINT_DIR="checkpoints/v6-ablation-randomneg"
SPLIT_FILE="artifacts/splits/esaw-fixed-seed13-test10-val10.json"
TRAIN_LOG="logs/train-v6-ablation-randomneg.log"
BENCH_BASE_VALIDATION_JSON="benchmarks/benchmark-v5-baseline-validation-flat-top1-top5-top10.json"
BENCH_VALIDATION_JSON="benchmarks/benchmark-v6-ablation-randomneg-validation-flat-top1-top5-top10.json"
BENCH_TEST_JSON="benchmarks/benchmark-v6-ablation-randomneg-test-flat-top1-top5-top10.json"
RECORD_VALIDATION_JSON="benchmarks/benchmark-v6-ablation-randomneg-validation-record-top1-top5-top10.json"
RECORD_TEST_JSON="benchmarks/benchmark-v6-ablation-randomneg-test-record-top1-top5-top10.json"
DELTA_JSON="benchmarks/v6-ablation-randomneg-vs-v5-validation-delta.json"
ARTIFACT="/workspace/esaw-biencoder-v6-ablation-randomneg-artifacts.tgz"
VENV_DIR="/root/esaw-biencoder-v6-randomneg-venv"

mkdir -p logs benchmarks checkpoints

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
grep -Ev '^[[:space:]]*torch([<=>[:space:]]|$)' requirements.txt > /tmp/esaw-requirements-no-torch.txt
python -m pip install -r /tmp/esaw-requirements-no-torch.txt datasets
python -m pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps torch==2.8.0

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA non disponibile")
print("cuda_device", torch.cuda.get_device_name(0))
PY

nvidia-smi
python - <<'PY'
import re
import subprocess

output = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
used, total = [int(part.strip()) for part in re.split(r",\s*", output.strip().splitlines()[0])]
print(f"gpu_memory_before_training_mib {used}/{total}")
if used > 5000:
    raise SystemExit(f"GPU memory not clean before training: {used} MiB already used.")
PY

python benchmark_topks_once.py \
  --model "$BASE_MODEL" \
  --dataset dataset.json \
  --taxonomy tassonomia.json \
  --split-file "$SPLIT_FILE" \
  --split validation \
  --sample-size 0 \
  --text-format v5 \
  --output "$BENCH_BASE_VALIDATION_JSON"

python -m esaw_biencoder.train \
  --dataset dataset.json \
  --taxonomy tassonomia.json \
  --output-dir "$MODEL_DIR" \
  --base-model "$BASE_MODEL" \
  --epochs 1 \
  --batch-size 24 \
  --device cuda \
  --loss mnrl \
  --mnrl-hard-negatives-per-positive 4 \
  --mnrl-hardness-mode hard-negatives \
  --mnrl-hardness-strength 0.2 \
  --negative-strategy random \
  --optimizer adafactor \
  --learning-rate 8e-6 \
  --max-seq-length 256 \
  --torch-dtype bfloat16 \
  --text-format v5 \
  --split-file "$SPLIT_FILE" \
  --validation-ratio 0.1 \
  --test-ratio 0.1 \
  --max-validation-examples 1000 \
  --evaluation-steps 0 \
  --checkpoint-path "$CHECKPOINT_DIR" \
  --checkpoint-save-steps 500 \
  --checkpoint-save-total-limit 4 \
  --resume-from-checkpoint \
  2>&1 | tee "$TRAIN_LOG"

python benchmark_topks_once.py \
  --model "$MODEL_DIR" \
  --dataset dataset.json \
  --taxonomy tassonomia.json \
  --split-file "$SPLIT_FILE" \
  --split validation \
  --sample-size 0 \
  --text-format v5 \
  --output "$BENCH_VALIDATION_JSON"

python benchmark_topks_once.py \
  --model "$MODEL_DIR" \
  --dataset dataset.json \
  --taxonomy tassonomia.json \
  --split-file "$SPLIT_FILE" \
  --split test \
  --sample-size 0 \
  --text-format v5 \
  --output "$BENCH_TEST_JSON"

python benchmark_record_topks_once.py \
  --model "$MODEL_DIR" \
  --dataset dataset.json \
  --taxonomy tassonomia.json \
  --split-file "$SPLIT_FILE" \
  --split validation \
  --sample-size 0 \
  --text-format v5 \
  --batch-size 64 \
  --output "$RECORD_VALIDATION_JSON"

python benchmark_record_topks_once.py \
  --model "$MODEL_DIR" \
  --dataset dataset.json \
  --taxonomy tassonomia.json \
  --split-file "$SPLIT_FILE" \
  --split test \
  --sample-size 0 \
  --text-format v5 \
  --batch-size 64 \
  --output "$RECORD_TEST_JSON"

python - <<PY
import json
from pathlib import Path

base_path = Path("$BENCH_BASE_VALIDATION_JSON")
candidate_path = Path("$BENCH_VALIDATION_JSON")
out_path = Path("$DELTA_JSON")
base = json.loads(base_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
delta = {
    key: round(candidate["mean"][key] - base["mean"][key], 4)
    for key in ("top1_accuracy", "top5_accuracy", "top10_accuracy")
}
payload = {
    "ablation": "random explicit negatives",
    "controlled_difference": "same as v6 hard-negative run, but explicit negatives are sampled randomly instead of mined from v5 retrieval errors",
    "baseline": str(base_path),
    "candidate": str(candidate_path),
    "baseline_mean": base["mean"],
    "candidate_mean": candidate["mean"],
    "delta_mean": delta,
}
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

tar -czf "$ARTIFACT" \
  "$MODEL_DIR" "$CHECKPOINT_DIR" "$SPLIT_FILE" benchmarks logs \
  runpod_v6_qwen_randomneg_ablation_train_benchmark.sh requirements.txt README.md REPORT_RUNPOD_TRAINING.md \
  benchmark_topks_once.py benchmark_record_topks_once.py esaw_biencoder

echo "Artifact salvato in $ARTIFACT"
