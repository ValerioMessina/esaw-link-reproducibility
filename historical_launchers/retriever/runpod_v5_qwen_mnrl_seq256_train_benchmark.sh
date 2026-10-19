#!/usr/bin/env bash
set -euo pipefail

cd /workspace/esaw-biencoder
mkdir -p logs benchmarks artifacts/splits models checkpoints

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_DIR="models/esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256"
SPLIT_FILE="artifacts/splits/esaw-fixed-seed13-test10-val10.json"
CHECKPOINT_DIR="checkpoints/v5-qwen3-0.6b-mnrl-seq256"
TRAIN_LOG="logs/train-v5-qwen3-0.6b-mnrl-3ep-b80-seq256-bf16-adafactor.log"
BENCH_TEST_TOP5="logs/benchmark-v5-qwen3-0.6b-mnrl-seq256-test-flat-top5.log"
BENCH_TEST_TOP10="logs/benchmark-v5-qwen3-0.6b-mnrl-seq256-test-flat-top10.log"
BENCH_ALL_TOP5="logs/benchmark-v5-qwen3-0.6b-mnrl-seq256-all-flat-top5.log"
BENCH_ALL_TOP10="logs/benchmark-v5-qwen3-0.6b-mnrl-seq256-all-flat-top10.log"
SUMMARY="benchmarks/v5-qwen3-0.6b-mnrl-seq256-summary.json"
ARTIFACT="/workspace/esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256-artifacts.tgz"
VENV_DIR="/root/esaw-biencoder-v5-venv"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
TORCH_VERSION="2.8.0"

package_artifact() {
  local paths=()
  for path in "$MODEL_DIR" "$CHECKPOINT_DIR" "$SPLIT_FILE" benchmarks logs runpod_v5_qwen_mnrl_seq256_train_benchmark.sh \
    requirements.txt README.md REPORT_RUNPOD_TRAINING.md esaw_biencoder; do
    [[ -e "$path" ]] && paths+=("$path")
  done
  if [[ ${#paths[@]} -gt 0 ]]; then
    tar --no-same-owner -czf "$ARTIFACT" "${paths[@]}" || true
  fi
}

trap 'echo "=== ERROR: packaging partial artifacts ==="; package_artifact' ERR

{
  echo "=== ENV ==="
  date -Iseconds
  nvidia-smi
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("cuda_capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
PY

  echo "=== PYTHON ENV ==="
  python -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION"
  grep -Ev '^[[:space:]]*torch([<=>[:space:]]|$)' requirements.txt > /tmp/esaw-requirements-no-torch.txt
  python -m pip install -r /tmp/esaw-requirements-no-torch.txt datasets
  python -m pip install --index-url "$TORCH_INDEX_URL" --force-reinstall --no-deps "torch==$TORCH_VERSION"
  python - <<'PY'
import sentence_transformers, torch, transformers
print("sentence_transformers", sentence_transformers.__version__)
print("transformers", transformers.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise SystemExit("CUDA non disponibile dopo installazione Torch cu128")
PY

  echo "=== TRAIN V5 QWEN 0.6B MNRL SEQ256 BATCH80 ==="
  RESUME_ARGS=()
  if find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | grep -q .; then
    RESUME_ARGS=(--resume-from-checkpoint)
    echo "Resume checkpoint: enabled from $CHECKPOINT_DIR"
  else
    echo "Resume checkpoint: no existing checkpoint in $CHECKPOINT_DIR"
  fi
  python -m esaw_biencoder.train \
    --dataset dataset.json \
    --taxonomy tassonomia.json \
    --output-dir "$MODEL_DIR" \
    --model-preset qwen3-embedding-0.6b \
    --epochs 3 \
    --batch-size 80 \
    --device cuda \
    --loss mnrl \
    --optimizer adafactor \
    --learning-rate 2e-5 \
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
    "${RESUME_ARGS[@]}"
} 2>&1 | tee "$TRAIN_LOG"

source "$VENV_DIR/bin/activate"

run_bench() {
  local label="$1"
  local outfile="$2"
  local top_k="$3"
  shift 3
  {
    echo "=== $label ==="
    date -Iseconds
    python -m esaw_biencoder.evaluate \
      --model "$MODEL_DIR" \
      --dataset dataset.json \
      --taxonomy tassonomia.json \
      --top-k "$top_k" \
      "$@"
  } 2>&1 | tee "$outfile"
}

run_bench "BENCH TEST FLAT TOP5" "$BENCH_TEST_TOP5" 5 \
  --split-file "$SPLIT_FILE" --split test --sample-size 0 --retrieval flat
cp "$BENCH_TEST_TOP5" benchmarks/benchmark-v5-qwen3-0.6b-mnrl-seq256-test-flat-top5.txt

run_bench "BENCH TEST FLAT TOP10" "$BENCH_TEST_TOP10" 10 \
  --split-file "$SPLIT_FILE" --split test --sample-size 0 --retrieval flat
cp "$BENCH_TEST_TOP10" benchmarks/benchmark-v5-qwen3-0.6b-mnrl-seq256-test-flat-top10.txt

run_bench "BENCH ALL FLAT TOP5" "$BENCH_ALL_TOP5" 5 \
  --sample-size 0 --retrieval flat
cp "$BENCH_ALL_TOP5" benchmarks/benchmark-v5-qwen3-0.6b-mnrl-seq256-all-flat-top5.txt

run_bench "BENCH ALL FLAT TOP10" "$BENCH_ALL_TOP10" 10 \
  --sample-size 0 --retrieval flat
cp "$BENCH_ALL_TOP10" benchmarks/benchmark-v5-qwen3-0.6b-mnrl-seq256-all-flat-top10.txt

cat > "$SUMMARY" <<JSON
{
  "model_dir": "$MODEL_DIR",
  "split_file": "$SPLIT_FILE",
  "base_model": "Qwen/Qwen3-Embedding-0.6B",
  "model_preset": "qwen3-embedding-0.6b",
  "loss": "mnrl",
  "optimizer": "adafactor",
  "learning_rate": 2e-5,
  "max_seq_length": 256,
  "text_format": "v5",
  "torch_dtype": "bfloat16",
  "use_amp": false,
  "gradient_checkpointing": false,
  "epochs": 3,
  "batch_size": 80,
  "device": "cuda",
  "retrieval_primary": "flat",
  "benchmarks": [
    "benchmark-v5-qwen3-0.6b-mnrl-seq256-test-flat-top5.txt",
    "benchmark-v5-qwen3-0.6b-mnrl-seq256-test-flat-top10.txt",
    "benchmark-v5-qwen3-0.6b-mnrl-seq256-all-flat-top5.txt",
    "benchmark-v5-qwen3-0.6b-mnrl-seq256-all-flat-top10.txt"
  ]
}
JSON

package_artifact

echo "=== DONE ==="
date -Iseconds
echo "artifact=$ARTIFACT"
