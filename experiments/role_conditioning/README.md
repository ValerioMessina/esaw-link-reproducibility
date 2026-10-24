# Matched role-conditioning ablation

Completed 22 October 2026. Both arms were retrained from Qwen3-Embedding-0.6B
with the same seed (13), original 40,000/5,000/5,000 split, training-example
order, admissible inventories and common narrative crop. Compute budget
constraints reduced **both arms to one alignment epoch plus one refinement
epoch (1+1)**. This is a separate experiment from the main historical 3+1 run.

Role-off removes the target name and role definition from **both queries and
candidate documents**. Labels, parent context, official codes and each field's
admissible inventory remain. Both arms see the same narrative token budget
(195 tokens before adding prefixes, 256-token model limit); removing a prefix
does not expose additional narrative. Refinement mines each arm's own training
Top-20 lists and uses four negatives per positive, with hardness strength 0.2.
Both phases use cached MNRL with microbatch 8 and full-parameter fine-tuning.

## Verified results against the administrative reference

All values are percentages on the original 5,000 test records.

| Stage | Arm | Mean Top-1 | Mean Top-5 | Mean Top-10 | Exact 8/8 | All 8 in Top-10 |
|---|---|---:|---:|---:|---:|---:|
| Frozen | Role on | 10.0700 | 32.1925 | 48.9150 | 0.00 | 2.22 |
| Frozen | Role off | 7.4450 | 22.5175 | 35.4850 | 0.00 | 0.54 |
| Alignment | Role on | 48.4875 | 76.7425 | 85.8275 | 3.64 | 46.10 |
| Alignment | Role off | 44.9500 | 73.7375 | 83.9425 | 3.50 | 43.56 |
| Refinement | Role on | 61.6275 | 91.7425 | 96.4850 | 8.50 | 81.04 |
| Refinement | Role off | 59.3150 | 90.7475 | 96.1175 | 7.72 | 80.32 |

The primary contrast after refinement is **+2.3125 percentage points** in mean
Top-1, paired-record bootstrap 95% CI **[1.9050, 2.7125]**. Exact 8/8 changes
by +0.78 points, CI [0.20, 1.36]. All eight fields stay paired within records;
10,000 replicates, bootstrap seed 20261021. The aggregate evidence contains
the per-field metrics and frozen/alignment contrasts too.

This single training seed does not estimate training-seed variability. The
contrast is the total pipeline effect, including arm-specific negative mining;
it does not separate query-side from document-side conditioning. The three
role-off material-agent fields necessarily have identical rankings. The
shorter schedule, common evidence crop and cached loss differ from the main
historical experiment; its scores cannot serve as this ablation's control.

## Files and provenance

- `historical/suite.py`: byte-for-byte recovered final source snapshot, including
  optimizer/scheduler/RNG checkpoint recovery. Its SHA-256 is recorded in
  `source_provenance.json`; the same manifest pins the four retriever dependencies.
- `protocol.json`: scientific protocol, with instance/billing metadata omitted.
  Complete-record recall headings are clarified as All8 rather than RR.
- `aggregate_evidence.json`: aggregate metrics and paired intervals only.
- `run.py`: prepares a fresh **external private workspace**. Its only suite-code
  edit renames the two historical Markdown report headings RR@5/10 to All8@5/10.
- `recount.py`: independently recounts all six private prediction exports and
  their paired intervals; does not load model weights or contact providers.
- `requirements.txt`: experiment-specific versions; the recovered CUDA runtime
  used PyTorch 2.8.0+cu128. Install the appropriate CUDA wheel separately.

No dataset, record identifiers, prediction arrays, models, credentials or cloud
account state are included. The manuscript is maintained separately.

## Preparing and running with authorized data

Use Python 3.11+ and the recorded CUDA environment for training. Install
`requirements.txt` into a separate environment after installing PyTorch.
The following command **only prepares files and prints the training command**:

```bash
python experiments/role_conditioning/run.py \
  --dataset /secure/esaw/dataset.json \
  --taxonomy /secure/esaw/tassonomia.json \
  --split-file /secure/esaw/split.json \
  --work-dir /secure/esaw/role-ablation-new \
  --deadline-hours 12
```

Add `--execute` to that preparation command to train on an existing CUDA GPU.
The work directory must not already exist, preventing stale model, mining or
test caches from being reused. Select the process deadline for the available
compute budget; it is checked at operation boundaries. This launcher does not
provision or stop cloud instances. The original fixed-final-epoch protocol does
not select checkpoints, seeds or hyperparameters using test results. Interrupted
stages must not be reported as completed experiments.

To recount the completed private exports (NumPy only):

```bash
python experiments/role_conditioning/recount.py \
  --results /secure/esaw/role-ablation-completed/results \
  --check-published
```

`--check-published` applies to the original recovered experiment. Omit it for a
new run whose numerical results can differ. Output contains aggregates only.
The original six exports and all three paired-bootstrap stages were independently
recounted successfully before this release. This does not certify an independent
GPU retraining from the public launcher.
