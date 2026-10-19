#!/usr/bin/env bash
set -euo pipefail

# Budget-constrained HiSR experiment on the fixed ESAW split. HiSR publishes
# one complete configuration, so no architecture or hyperparameter search is
# introduced. The best validation-Macro checkpoint from seed 13 is evaluated
# once on the retrospective test split.

cd /workspace/paper-reproductions
source ./runpod_runner_common.sh
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV=/root/esaw-hisr-venv
DATA_DIR=data/esaw
RESULTS=results/hisr_preliminary
mkdir -p "$RESULTS/final" logs

package_results() {
  tar -czf /workspace/esaw-paper-reproductions-results.tgz \
    results logs data/esaw protocols PROVENANCE.md \
    REPRODUCTION_STATUS.md README.md runpod_esaw_hisr_benchmark.sh \
    runpod_runner_common.sh train_recent_htc.py esaw_htc 2>/dev/null || true
}
trap package_results EXIT

ensure_gpu_venv "$VENV"

python - <<'PY'
import json
import platform
import torch

assert torch.cuda.is_available()
metadata = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "python": platform.python_version(),
    "scope": "ESAW-only HiSR matched-seed preliminary benchmark",
    "configuration_policy": "single published complete configuration; no ESAW tuning grid",
    "final_seed": 13,
}
print(json.dumps(metadata, indent=2), flush=True)
with open("results/hisr_preliminary/environment.json", "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)
    handle.write("\n")
PY

prepare_neural_assets_if_missing "$DATA_DIR" bert-base-uncased

if ! skip_completed_stage hisr_final_seed13 "$RESULTS/final/seed13/benchmark.json"; then
  archive_incomplete_stage "$RESULTS/final/seed13" hisr_final_seed13
  python train_recent_htc.py \
    --method hisr \
    --data-dir "$DATA_DIR" --data-name esaw \
    --output-dir "$RESULTS/final/seed13" \
    --seed 13 \
    --pretokenize --num-workers 4 \
    2>&1 | tee logs/hisr_final_seed13.log
fi

python - <<'PY'
import json
from pathlib import Path

benchmark = json.loads(
    Path("results/hisr_preliminary/final/seed13/benchmark.json").read_text()
)
summary = {
    "status": "completed",
    "interpretation": "single matched-seed preliminary ESAW result using the published HiSR configuration",
    "method": "HiSR",
    "configuration_policy": "single published complete configuration; no ESAW tuning grid",
    "final_seed": benchmark["seed"],
    "checkpoint_epoch": benchmark["checkpoint"],
    "mean": benchmark["mean"],
    "record_exact@1": benchmark["record_exact@1"],
    "record_recall@5": benchmark["record_recall@5"],
    "record_recall@10": benchmark["record_recall@10"],
    "claim_boundary": "One matched seed provides preliminary evidence only and does not estimate seed variance or establish a statistically confirmed SOTA claim.",
}
Path("results/hisr_preliminary/summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2), flush=True)
PY

package_results
