# Hierarchical baseline tests — 21 October 2026

All five available corrected-HGCLR checkpoints were evaluated on the same
5,000 test records / 40,000 field decisions. Four trainings completed source
early stopping (seeds 3, 13, 21, 42); seed 87 stopped for budget/signal during
epoch 4 and is reported separately. The first strictly highest validation
Macro-F1 checkpoint was selected for each seed, not the last training epoch.
No training was performed for this evaluation and no seed was selected on test.

This is a **separate retrospective evaluation of available checkpoints**.
The original confirmatory trainer requires five completed trainings before
opening test; that gate remains unsatisfied and the trainer/protocol have not
been relaxed. These results must not be described as completing that protocol.
In particular, seed 13 is not a representative five-seed HGCLR result; the
large between-seed differences remain part of the evidence.

HTC-CLIP, HiSR and HyILR already had complete tests. Their checkpoint hashes,
record/gold alignment and metrics were rechecked; those results are reused,
not a new inference or new training. Interrupted tuning trials do not replace
the selected historical checkpoints. Historical adapters are distinct from
the corrected HGCLR retraining and are not original-paper benchmark replications.

All values below are percentages. Top-k is the unweighted mean over eight
fields; Exact8 requires all eight Top-1 predictions to be correct for a record.

| Method | Seed | Training | Checkpoint epoch | Decoder | Top-1 | Top-5 | Top-10 | Exact8 |
|---|---:|---|---:|---|---:|---:|---:|---:|
| HGCLR | 3 | Complete | 2 | Raw | 2.3000 | 37.2600 | 63.0175 | 0.0400 |
| HGCLR | 3 | Complete | 2 | Calibrated | 23.8125 | 46.5225 | 67.6750 | 1.2200 |
| HGCLR | 13 | Complete | 29 | Raw | 20.4675 | 82.5300 | 92.4100 | 0.0400 |
| HGCLR | 13 | Complete | 29 | Calibrated | 56.9350 | 84.7400 | 91.4700 | 4.6600 |
| HGCLR | 21 | Complete | 2 | Raw | 2.5100 | 35.7550 | 62.4925 | 0.0400 |
| HGCLR | 21 | Complete | 2 | Calibrated | 23.5300 | 46.2775 | 67.3275 | 1.2400 |
| HGCLR | 42 | Complete | 3 | Raw | 2.1900 | 35.7900 | 61.4425 | 0.0400 |
| HGCLR | 42 | Complete | 3 | Calibrated | 24.0175 | 46.8975 | 68.0625 | 1.2200 |
| HGCLR | 87 | **Incomplete** | 2 | Raw | 2.1900 | 34.0075 | 60.9925 | 0.0400 |
| HGCLR | 87 | **Incomplete** | 2 | Calibrated | 23.6325 | 42.6450 | 66.8825 | 1.2200 |
| HTC-CLIP | 3 | Historical | 50 | Raw | 21.1650 | 80.2325 | 90.4425 | 0.0600 |
| HTC-CLIP | 3 | Historical | 50 | Calibrated | 55.7975 | 82.3650 | 89.4875 | 4.8600 |
| HiSR | 13 | Historical | 15 | Raw | 24.1050 | 83.8875 | 93.8575 | 0.0400 |
| HiSR | 13 | Historical | 15 | Calibrated | 57.3575 | 86.5125 | 92.8550 | 4.9000 |
| HyILR | 13 | Historical | 6 | Raw | 25.7025 | 83.0650 | 93.2050 | 0.0400 |
| HyILR | 13 | Historical | 6 | Calibrated | 58.4250 | 86.6600 | 92.9550 | 6.9400 |

## Decoder and inference provenance

Raw decoding uses FP32 sigmoid then Torch top-k over the field-admissible
codes, including parents. Calibrated decoding retains the same weights and
candidate inventories, adds one child-logit offset per field, and ranks in
FP64 with stable taxonomy-index ties. Offsets were frozen from validation
before this test inference: grid [-12,12], step 0.25; maximum field Top-1,
then smallest absolute offset, then grid order. This is retrospective because
historical test results had already been observed. Decoder gains do not
establish encoder improvement or independent confirmation.

HGCLR evaluation used the pinned author BertModel encoder CLS and linear
classifier with strict state loading. Original no-label evaluation does not
use the training-only graph/projector. The first 32 validation examples were
compared with cached logits for each checkpoint (atol=rtol=0.003). Maximum
absolute differences range from 0.00000190735 to 0.00001811981. This is a
32-record loading/parity probe, not a full-validation inference parity claim.
Runtime: Python 3.12.14, Torch 2.8.0, Transformers 4.44.2, FP32, macOS/MPS,
batch 8, maximum length 512. It is separate from the CUDA training runtime.

## Verification and available files

The [aggregate evidence](BASELINE_TEST_20261021.json) records all five seeds,
training completion, selected epochs, frozen offsets, metrics, runtime, loading
probes and hashes of private evidence. It contains no record identifiers,
narratives, logits, predictions or weights. Per-record artifacts stay private.

The original executed evaluation and recount sources are preserved verbatim in
`historical_verification/baseline_test_20261021/`. Like other historical
verification scripts, they require the original authorized external artifact
layout and are not portable entry points. Their source hashes appear in
`provenance/source_manifest.json`. The recount checks predictions against saved
scores, record/gold identity, candidate membership, duplicate-free ranks,
Top-1/5/10 and complete-record arithmetic, and the three historical checkpoint
hashes. It does not certify published source-score recovery.

The repository update reran this recount in its recorded Torch/Transformers
environment without launching training, model inference or paid API requests.
Offline evidence tests check denominators, metric arithmetic constraints,
checkpoint provenance and the distinction between four complete trainings and
one incomplete training. They do not replace the private-artifact recount.
