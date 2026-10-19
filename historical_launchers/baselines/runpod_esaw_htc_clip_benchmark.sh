#!/usr/bin/env bash
set -euo pipefail

cd /workspace/paper-reproductions
source ./runpod_runner_common.sh
export PYTHONPATH="$PWD:$PWD/upstream/contrastive-htc"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV=/root/esaw-htc-clip-venv
DATA_DIR=data/esaw
RESULTS=results/htc_clip_preliminary
mkdir -p "$RESULTS/tuning" "$RESULTS/final" logs

package_results() {
  tar -czf /workspace/esaw-paper-reproductions-results.tgz \
    results logs data/esaw protocols PROVENANCE.md REPRODUCTION_STATUS.md README.md \
    runpod_esaw_htc_clip_benchmark.sh runpod_runner_common.sh train_htc_clip.py \
    select_tuning_result.py esaw_htc upstream/contrastive-htc 2>/dev/null || true
}
trap package_results EXIT

ensure_gpu_venv "$VENV"
prepare_neural_assets_if_missing "$DATA_DIR" bert-base-uncased

run_config() {
  local gamma=$1 weight=$2 output=$3 seed=$4 log=$5
  shift 5
  python train_htc_clip.py --architecture htc-clip \
    --data-dir "$DATA_DIR" --data-name esaw --output-dir "$output" \
    --base-model bert-base-uncased --batch-size 56 --eval-batch-size 128 \
    --learning-rate 3e-5 --gamma "$gamma" --contrastive-weight "$weight" \
    --temperature 1 --path-hidden 128 --max-length 512 --early-stop 6 --seed "$seed" \
    --pretokenize --num-workers 4 \
    "$@" 2>&1 | tee "$log"
}

for config in "0.02 0.05" "0.005 0.3"; do
  read -r gamma weight <<< "$config"
  tag="g${gamma}_w${weight}"
  marker="$RESULTS/tuning/${tag}.json"
  if ! skip_completed_stage "htc_clip_tuning_${tag}" "$marker"; then
    archive_incomplete_stage "$RESULTS/tuning/$tag" "htc_clip_tuning_${tag}"
    run_config "$gamma" "$weight" "$RESULTS/tuning/$tag" 3 "logs/htc_clip_tuning_${tag}.log" \
      --skip-test --validation-output "$marker"
  fi
done
python select_tuning_result.py --output "$RESULTS/tuning/selected.json" "$RESULTS"/tuning/g*.json
read -r GAMMA WEIGHT < <(python -c 'import json; x=json.load(open("results/htc_clip_preliminary/tuning/selected.json"))["selected"]["hyperparameters"]; print(x["gamma"], x["contrastive_weight"])')

if ! skip_completed_stage htc_clip_final_seed13 "$RESULTS/final/seed13/benchmark.json"; then
  archive_incomplete_stage "$RESULTS/final/seed13" htc_clip_final_seed13
  run_config "$GAMMA" "$WEIGHT" "$RESULTS/final/seed13" 13 logs/htc_clip_final_seed13.log
fi

python - <<'PY'
import json
from pathlib import Path
b = json.loads(Path("results/htc_clip_preliminary/final/seed13/benchmark.json").read_text())
s = {"status": "completed", "method": "HTC-CLIP", "final_seed": b["seed"],
     "checkpoint_epoch": b["checkpoint"], "mean": b["mean"],
     "record_exact@1": b["record_exact@1"], "record_recall@5": b["record_recall@5"],
     "record_recall@10": b["record_recall@10"],
     "claim_boundary": "Single matched-seed preliminary result; not a statistical SOTA claim."}
Path("results/htc_clip_preliminary/summary.json").write_text(json.dumps(s, indent=2) + "\n")
print(json.dumps(s, indent=2), flush=True)
PY

package_results
