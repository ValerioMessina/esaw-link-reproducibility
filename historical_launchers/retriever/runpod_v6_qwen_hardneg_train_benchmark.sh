#!/usr/bin/env bash
set -euo pipefail

cd /workspace/esaw-biencoder

BASE_MODEL="models/esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256"
MODEL_DIR="models/esaw-biencoder-v6-qwen3-0.6b-mnrl-hardneg"
CHECKPOINT_DIR="checkpoints/v6-qwen3-0.6b-mnrl-hardneg"
SPLIT_FILE="artifacts/splits/esaw-fixed-seed13-test10-val10.json"
HARD_NEGATIVES="artifacts/hard_negatives/v5-train-top20.json"
TRAIN_LOG="logs/train-v6-qwen3-0.6b-mnrl-hardneg.log"
BENCH_BASE_VALIDATION_JSON="benchmarks/benchmark-v5-baseline-validation-flat-top1-top5-top10.json"
BENCH_VALIDATION_JSON="benchmarks/benchmark-v6-qwen3-0.6b-mnrl-hardneg-validation-flat-top1-top5-top10.json"
BENCH_TEST_JSON="benchmarks/benchmark-v6-qwen3-0.6b-mnrl-hardneg-test-flat-top1-top5-top10.json"
DELTA_JSON="benchmarks/v6-hardneg-vs-v5-validation-delta.json"
VALIDATION_GATE="benchmarks/v6-hardneg-validation-gate.txt"
ARTIFACT="/workspace/esaw-biencoder-v6-qwen3-0.6b-mnrl-hardneg-artifacts.tgz"
VENV_DIR="/root/esaw-biencoder-v6-venv"

mkdir -p logs benchmarks checkpoints artifacts/hard_negatives

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
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

if [ ! -f "$HARD_NEGATIVES" ]; then
  python -m esaw_biencoder.mine_hard_negatives \
    --model "$BASE_MODEL" \
    --dataset dataset.json \
    --taxonomy tassonomia.json \
    --split-file "$SPLIT_FILE" \
    --split train \
    --top-n 20 \
    --batch-size 64 \
    --device cuda \
    --output "$HARD_NEGATIVES"
fi

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
  --hard-negative-file "$HARD_NEGATIVES" \
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

python - <<PY
import json
from pathlib import Path

base_path = Path("$BENCH_BASE_VALIDATION_JSON")
v6_path = Path("$BENCH_VALIDATION_JSON")
out_path = Path("$DELTA_JSON")
base = json.loads(base_path.read_text(encoding="utf-8"))
v6 = json.loads(v6_path.read_text(encoding="utf-8"))
delta = {
    key: round(v6["mean"][key] - base["mean"][key], 4)
    for key in ("top1_accuracy", "top5_accuracy", "top10_accuracy")
}
payload = {
    "baseline": str(base_path),
    "candidate": str(v6_path),
    "baseline_mean": base["mean"],
    "candidate_mean": v6["mean"],
    "delta_mean": delta,
    "promote_to_test_threshold": {
        "min_top1_delta": 0.04,
        "max_top5_drop": -0.01
    },
    "passes_validation_gate": (
        delta["top1_accuracy"] >= 0.04 and delta["top5_accuracy"] >= -0.01
    ),
}
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
Path("$VALIDATION_GATE").write_text("PASS\\n" if payload["passes_validation_gate"] else "FAIL\\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

if [ "$(cat "$VALIDATION_GATE")" = "PASS" ]; then
  python benchmark_topks_once.py \
    --model "$MODEL_DIR" \
    --dataset dataset.json \
    --taxonomy tassonomia.json \
    --split-file "$SPLIT_FILE" \
    --split test \
    --sample-size 0 \
    --text-format v5 \
    --output "$BENCH_TEST_JSON"
else
  echo "Validation gate non superato: salto il benchmark test per evitare di usare il test come set di tuning."
fi

tar -czf "$ARTIFACT" \
  "$MODEL_DIR" "$CHECKPOINT_DIR" "$SPLIT_FILE" "$HARD_NEGATIVES" benchmarks logs \
  runpod_v6_qwen_hardneg_train_benchmark.sh requirements.txt README.md REPORT_RUNPOD_TRAINING.md esaw_biencoder

echo "Artifact salvato in $ARTIFACT"
