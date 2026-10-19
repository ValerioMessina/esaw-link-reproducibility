#!/usr/bin/env bash

# Shared recovery helpers for the RunPod ESAW runners.  A stage is considered
# complete only when its JSON completion artifact exists and parses correctly.

json_is_valid() {
  local path=$1
  [[ -s "$path" ]] && python3 - "$path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    json.load(handle)
PY
}

skip_completed_stage() {
  local label=$1 marker=$2
  if json_is_valid "$marker"; then
    printf 'resume_skip stage=%s marker=%s\n' "$label" "$marker"
    return 0
  fi
  return 1
}

archive_incomplete_stage() {
  local stage_dir=$1 label=$2
  if [[ ! -d "$stage_dir" ]] || [[ -z "$(find "$stage_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    return 0
  fi
  local stamp destination
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  destination="results/interrupted/${label}_${stamp}"
  mkdir -p "$(dirname "$destination")"
  mv "$stage_dir" "$destination"
  printf 'resume_archive stage=%s destination=%s\n' "$label" "$destination"
}

ensure_gpu_venv() {
  local venv=$1
  if [[ ! -x "$venv/bin/python" ]]; then
    python3 -m venv "$venv"
    # shellcheck disable=SC1090
    source "$venv/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
    sed '/^torch==/d' requirements-gpu.txt > /tmp/requirements-no-torch.txt
    python -m pip install -r /tmp/requirements-no-torch.txt
  else
    # shellcheck disable=SC1090
    source "$venv/bin/activate"
    printf 'resume_reuse venv=%s\n' "$venv"
  fi
}

prepare_neural_assets_if_missing() {
  local data_dir=$1 model=$2
  local required=(bert_value_dict.pt slot.pt value_dict.pt label_symbol_dict.pt special_tokens.txt)
  local name
  for name in "${required[@]}"; do
    if [[ ! -s "$data_dir/$name" ]]; then
      python prepare_neural_assets.py --data-dir "$data_dir" --model "$model"
      return 0
    fi
  done
  printf 'resume_skip stage=neural_assets data_dir=%s\n' "$data_dir"
}
