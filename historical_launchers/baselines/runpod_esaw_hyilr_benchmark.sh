#!/usr/bin/env bash
set -euo pipefail

cd /workspace/paper-reproductions
source ./runpod_runner_common.sh
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV=/root/esaw-hyilr-venv
DATA_DIR=data/esaw
RESULTS=results/hyilr_preliminary
mkdir -p "$RESULTS/tuning" "$RESULTS/final" logs

package_results() {
  tar -czf /workspace/esaw-paper-reproductions-results.tgz \
    results logs data/esaw protocols PROVENANCE.md REPRODUCTION_STATUS.md README.md \
    runpod_esaw_hyilr_benchmark.sh train_recent_htc.py select_tuning_result.py \
    check_hyilr_optimization.py runpod_runner_common.sh esaw_htc 2>/dev/null || true
}
trap package_results EXIT

ensure_gpu_venv "$VENV"

python check_hyilr_optimization.py

prepare_neural_assets_if_missing "$DATA_DIR" bert-base-uncased

python - <<'PY'
import json
from pathlib import Path
policy = {
    "cost_cap_usd": 5.0,
    "gpu_hourly_usd": 0.44,
    "selection_policy": "budget-limited validation proxy",
    "tuning_seed": 3,
    "tuning_train_examples": 10000,
    "tuning_epochs_per_configuration": 1,
    "tuning_weights": [0.3, 0.4, 0.6],
    "final_seed": 13,
    "final_train_examples": 40000,
    "final_max_epochs": 6,
    "test_policy": "test remains sealed until validation selects one configuration",
    "numerics": "FP32; BF16 disabled by user request",
}
Path("results/hyilr_preliminary/budget_policy.json").write_text(
    json.dumps(policy, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(policy, indent=2), flush=True)
PY

for weight in 0.3 0.4 0.6; do
  tag="w${weight}"
  marker="$RESULTS/tuning/$tag/validation.json"
  if ! skip_completed_stage "hyilr_tuning_${tag}" "$marker"; then
    archive_incomplete_stage "$RESULTS/tuning/$tag" "hyilr_tuning_${tag}"
    python train_recent_htc.py --method hyilr \
      --data-dir "$DATA_DIR" --data-name esaw \
      --output-dir "$RESULTS/tuning/$tag" \
      --contrastive-weight "$weight" --seed 3 --skip-test \
      --pretokenize --num-workers 4 \
      --max-train-examples 10000 --max-epochs 1 \
      2>&1 | tee "logs/hyilr_tuning_${tag}.log"
  fi
done

python select_tuning_result.py --output "$RESULTS/tuning/selected.json" \
  "$RESULTS"/tuning/w*/validation.json
HYILR_WEIGHT=$(python -c 'import json; print(json.load(open("results/hyilr_preliminary/tuning/selected.json"))["selected"]["hyperparameters"]["contrastive_weight"])')

if ! skip_completed_stage hyilr_final_seed13 "$RESULTS/final/seed13/benchmark.json"; then
  archive_incomplete_stage "$RESULTS/final/seed13" hyilr_final_seed13
  python train_recent_htc.py --method hyilr \
    --data-dir "$DATA_DIR" --data-name esaw \
    --output-dir "$RESULTS/final/seed13" \
    --contrastive-weight "$HYILR_WEIGHT" --seed 13 \
    --pretokenize --num-workers 4 \
    --max-epochs 6 \
    2>&1 | tee logs/hyilr_final_seed13.log
fi

python - <<'PY'
import json
from pathlib import Path
b = json.loads(Path("results/hyilr_preliminary/final/seed13/benchmark.json").read_text())
s = {"status": "completed", "method": "HyILR", "final_seed": b["seed"],
     "checkpoint_epoch": b["checkpoint"], "mean": b["mean"],
     "record_exact@1": b["record_exact@1"], "record_recall@5": b["record_recall@5"],
     "record_recall@10": b["record_recall@10"],
     "claim_boundary": "Single matched-seed preliminary result; not a statistical SOTA claim."}
Path("results/hyilr_preliminary/summary.json").write_text(json.dumps(s, indent=2) + "\n")
print(json.dumps(s, indent=2), flush=True)
PY

package_results
