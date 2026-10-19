# Reproduction guide

## 1. Environments and inputs

Use separate virtual environments. HTC/GHCLR import internal Transformers BERT
APIs and require the recorded 4.44.2 environment; upgrading it to the retriever
environment is not a transparent change. `environments/hgclr-cuda.txt` is the
minimal frozen retraining environment. `environments/htc-cuda.txt` preserves
the larger historical dependency declaration, including packages for methods
outside the four rows. `environments/retriever-original-ranges.txt` preserves
the original broad requirements; it is NOT a historical lockfile. The Mac
reinference versions are recorded separately. For retriever training, install
`datasets` in addition to the original declaration and confirm that the
installed Sentence Transformers version exposes the MNRL hardness arguments
used in `retriever/esaw_biencoder/train.py`. The F06 API smoke source is included.
For optional selection/gate experiments, install scikit-learn, scipy and joblib.
Record `pip freeze`, Python, device, precision and source hashes for every run.

```bash
python3 -m venv /secure/esaw/env-retriever
source /secure/esaw/env-retriever/bin/activate
pip install -r environments/retriever-original-ranges.txt datasets
python scripts/check_inputs.py --dataset /secure/esaw/dataset.json \
  --taxonomy /secure/esaw/tassonomia.json --split-file /secure/esaw/split.json
```

Install an appropriate Torch/CUDA wheel for the machine. The code release
does not certify that an arbitrary latest combination reproduces old floating
point outputs. Models and tokenizer revisions must be downloaded separately.
The verified flat control used Qwen revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; the verified HGCLR BERT revision
is in its protocol. Do not assign these hashes retroactively to a historical
run that did not record them. For exact reruns, retain your authorized frozen
local pretrained snapshot and pass it where the source accepts a model path.

If the original private split is unavailable, `scripts/make_split.py` can
generate a deterministic split externally. Label it regenerated and verify
against the authorized original before claiming historical membership parity.

## 2. Retriever: alignment then refinement

Run from the repository root. Reuse these common flags for each configuration:

```bash
python scripts/run_experiment.py configs/phase1.json \
  --dataset /secure/esaw/dataset.json --taxonomy /secure/esaw/tassonomia.json \
  --split-file /secure/esaw/split.json --work-dir /secure/esaw/runs --execute
```

This performs full-parameter supervised retrieval tuning, 3 epochs, batch80,
Adafactor LR2e-5, BF16, max length256, seed13, v5 texts, MNRL in-batch.
The historical launcher also enabled periodic checkpoint export/resume; that
machine-specific launcher is preserved separately. The portable configuration
starts fresh and does not silently resume an old experiment.

Next run `configs/mine_train.json`, adding
`--phase1-model /secure/esaw/runs/phase1`. It mines Top20 candidates from TRAIN
only and writes `/secure/esaw/runs/hard_negatives.json`.
Then run `configs/refinement_h4.json` with the same data flags and additionally:

```bash
--phase1-model /secure/esaw/runs/phase1 \
--hard-negatives /secure/esaw/runs/hard_negatives.json
```

H4: one epoch, batch24, four explicit negatives per positive, hardness strength
.2, LR8e-6. H8 uses `configs/refinement_h8.json` (batch22, eight negatives),
as a separate continuation from the SAME final phase1, not from H4. Choose
hyperparameters on validation; do not choose a seed/model on test metrics.

Use `configs/evaluate_retriever.json` for record Top1/5/10 and Exact8, and
`configs/export_retriever.json` for field metrics and PRIVATE prediction export;
pass `--refined-model /secure/esaw/runs/refinement_h8` (or the chosen H4).
Persist both metrics and an exact file hash. Do not reconstruct candidate lists
from rounded printed scores. Taxonomic scorers are in
`retriever/evaluate_hierarchical_gold_noise.py` and
`retriever/evaluate_llm_predictions_audited_hierarchical.py`; inspect their
`--help` for original/silver paired inputs. Audit evaluation uses
`retriever/canonical_audit.py`; its structural validity is not human adjudication.

## 3. HTC preparation and three historical adapters

In the HTC environment, invoke `configs/prepare_htc.json` through the launcher
with the same external data/taxonomy/split flags. Outputs go to
`/secure/esaw/runs/htc_data`. Then use `configs/prepare_htc_assets.json`, adding
`--data-dir /secure/esaw/runs/htc_data` and your frozen BERT directory via
`--base-model /secure/esaw/pretrained-bert`. Generated label/token tensors are
private execution artifacts; none are distributed here.

Use `configs/hisr.json`, `configs/hyilr.json`, or `configs/htc_clip.json`
with those flags. These source configurations disable test access during
training. HiSR uses the published configuration, seed13; HyILR uses selected
contrastive weight .6 and max6epochs; HTC-CLIP uses the selected seed3 tuning
family and an explicit 50epoch cap. The historic source launcher also planned
a separate seed13 final run: do not conflate that plan with the seed3 checkpoint
used by the subsequent local audit. The portable recipes reproduce declared
settings; this release has not rerun those long trainings.

For the retrospective calibrated decoder:

```bash
python scripts/export_htc_scores.py --method hisr \
  --checkpoint /secure/esaw/runs/hisr/best_macro.pt \
  --data-dir /secure/esaw/runs/htc_data --base-model /secure/esaw/pretrained-bert \
  --split validation --device cuda --output /secure/esaw/runs/hisr_validation.npz
python scripts/decode_scores.py --method hisr \
  --metadata /secure/esaw/runs/htc_data/metadata.json \
  --validation /secure/esaw/runs/hisr_validation.npz \
  --output-dir /secure/esaw/runs/hisr_decoder
```

The decoder is now frozen. Export test with `export_htc_scores.py --split test`
and `--decoder-frozen /secure/esaw/runs/hisr_decoder/decoder_frozen.json`, using
the same checkpoint, metadata and an external test-score output. Finally:

```bash
python scripts/decode_scores.py --method hisr \
  --metadata /secure/esaw/runs/htc_data/metadata.json \
  --validation /secure/esaw/runs/hisr_validation.npz \
  --frozen /secure/esaw/runs/hisr_decoder/decoder_frozen.json \
  --test /secure/esaw/runs/hisr_test.npz \
  --output-dir /secure/esaw/runs/hisr_decoder
```

The portable scorer writes only aggregate metrics plus external private hit
arrays; it implements stable adjusted-logit ranking. For historical raw
`sigmoid(FP32)->torch.topk` tie behavior, use the preserved original verification
sources. Fixed pretrained/local weight availability and a complete fresh
inference must be checked on the target hardware. The new inference adapter
has source/CLI and synthetic checks, not a new full-checkpoint parity result.

## 4. Corrected HGCLR, gated separately

Use `baselines/hgclr_verified/train_verified.py`, its own `vendor/` and
`configs/protocol.json`; do not substitute the older adapter:

```bash
python baselines/hgclr_verified/train_verified.py \
  --data-dir /secure/esaw/runs/htc_data --pretrained /secure/esaw/pretrained-bert \
  --output /secure/esaw/runs/hgclr --deadline UNIX_DEADLINE \
  --diagnostic-deadline UNIX_DIAGNOSTIC_DEADLINE --diagnostics-only
```

Deadlines are future UTC Unix timestamps chosen by the researcher for the
available compute. Remove `--diagnostics-only` for diagnostic admission and
five-seed full training in the same invocation. Do not manually fabricate
`diagnostics/gates.json` to bypass failed diagnostics. The trainer refuses CPU,
retains checkpoints/RNG, and opens test only after all five seeds finish under
the frozen protocol. The executed study did not complete seed87, so its test
remains closed. Do not route HGCLR through the three-method score-export helper.

## 5. Reranking, prompts and silver audit

Prompt construction source is preserved unmodified:
`retriever/benchmark_llm_rerank_once.py`, especially
`single_record_prompt(record, candidates, conservative=False, hierarchical=True)`.
The larger-cap Mac protocol is in `configs/gemma_local_protocol.json`.
For a new LOCAL Ollama experiment, the original CLI supports:

```bash
cd retriever
python benchmark_llm_rerank_once.py --model /secure/esaw/runs/refinement_h8 \
  --dataset /secure/esaw/dataset.json --taxonomy /secure/esaw/tassonomia.json \
  --split-file /secure/esaw/split.json --split test --sample-size 0 \
  --backend ollama --api-base http://127.0.0.1:11434 --llm-model gemma4:12b \
  --candidate-top-k 10 --text-format v5 --single-record-prompt --hierarchical-rerank \
  --ollama-num-ctx 16384 --max-new-tokens 8192 --temperature 0 --seed 13 \
  --llm-retries 0 --no-consistency-pass \
  --output /secure/esaw/runs/gemma_metrics.json \
  --predictions-output /secure/esaw/runs/gemma_predictions.jsonl
```

This is a new CLI run: exact reconstruction of the already-running Mac test
requires its original ordered candidate export/private tasks. The artifact-bound
runner and analyzer are in `historical_verification/`; their private protocols,
tasks and responses are intentionally excluded. Do not present a new CLI run
as byte-identical to that frozen task stream. Check Ollama version and model
digest, not just a mutable model tag. Keep invalid responses, truncation and
fallback separate from direct valid LLM predictions. Do not report partial
accuracy as a completed full-test result.

The audit request/schema/prompt source is in `retriever/audit_esaw_with_openai.py`.
It is included for reproducibility, but no API keys, audit responses or new
paid requests are provided or launched. A new independent gold reference needs
blind review/adjudication; changing the LLM alone cannot remove system-dependent
candidate bias from a silver reference.

## 6. Statistical comparisons and provenance

`scripts/paired_bootstrap.py --a /secure/a.npz --b /secure/b.npz` expects each
external NPZ to contain identical ordered `record_keys` and boolean `hits[N,8]`.
It resamples records with all eight fields together, seed20261018, 10,000
replicates, and reports Top1 and Exact8 differences. This interval is conditional
on fixed trained systems/reference, not an estimate of training-seed uncertainty
or independent semantic correctness. Avoid significance-based stopping and
outcome-selected subsets.

`historical_verification/` preserves the exact local/cloud artifact verification
sources, including split overlap, negative pools, canonical silver, thinking,
and original-reference statistics. They retain original path/hash guards and
require authorized private artifacts; they are not advertised as portable
entry points. The flat-control snapshot likewise requires its external frozen
protocol, role definitions, prepared data and runtime deadline state. Its
budget-stopped result is not a completed convergence benchmark.

The original code is MIT-licensed; preserved upstream notices apply separately.
Before uploading changes, run release checks and review the source diff.
A new clone cannot execute the institutional experiment without
authorized data and regenerated or authorized frozen private artifacts.
