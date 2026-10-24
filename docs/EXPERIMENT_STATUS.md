# Evidence boundary — updated 24 October 2026

These are verified aggregate results. The 21 October update adds the separate
retrospective HGCLR test and rechecks the existing HTC tests; it does not launch
new training or complete the original five-training protocol.
Metric values below are percentages. No per-record results are distributed.

| Experiment | Verified scope / limit |
|---|---|
| ESAW-Link phase 1 | Full 3 epochs; fresh final-checkpoint test: Top1 48.2675, Top5 79.6025, Top10 88.5200, Exact8 3.64; historical step10000 is a different export |
| Main refined retriever | Historical original-reference full test Top1 61.7475, Exact8 8.48; refinement starts from final phase 1 |
| HTC-CLIP | Historical adaptation raw Top1 21.1650; fixed-weight validation-calibrated decoder 55.7975, Exact8 4.86 |
| HiSR | Raw 24.1050; calibrated 57.3575, Exact8 4.90 |
| HyILR | Raw 25.7025; calibrated 58.4250, Exact8 6.94 |
| Corrected HGCLR | Five-training protocol still incomplete: 3,13,21,42 complete, 87 stopped. Separate retrospective full tests of all five available validation-selected checkpoints completed on 21 October; see the linked report. Seed13 raw/calibrated Top1 20.4675/56.9350 is one seed, not the five-seed result. |
| Flat Qwen control | Test Top1 48.1675, Exact8 4.78; single seed, budget stop during epoch3 after two complete epochs |
| Role-conditioning ablation | Completed matched 1+1-epoch retraining, seed13, original 5000-record test. Role-on/off Top1 61.6275/59.3150; delta +2.3125 pp, paired 95% CI [1.9050, 2.7125]. Single seed; common narrative crop; arm-specific mining included. See [source and aggregate evidence](../experiments/role_conditioning/README.md). |
| F06 timing ablation | Same-loss explicit-negative API smoke passed; matched ESAW timing training NOT performed; historical cosine-vs-MNRL comparison does not isolate timing |
| New Gemma Mac full test | Completion not certified in this release; do not infer final metrics from partial outputs; separate from historical full-test Gemma below |

The [21 October baseline report](BASELINE_TEST_20261021.md) and
[machine-readable aggregate evidence](BASELINE_TEST_20261021.json) list every
seed and decoder, completion status, Top1/5/10, Exact8, checkpoint hashes and
verification limits. Historical HTC-CLIP/HiSR/HyILR values above are unchanged.
The original HGCLR trainer retains its five-completed-training test gate; the
new retrospective evaluation is explicitly separate.

Calibration uses one child-logit offset per field selected on validation only,
grid [-12,12] step .25, maximum field Top1, tie smallest absolute offset then
grid order; candidates retain admissible parents. Ranking uses stable taxonomy
order on adjusted-score ties. Test inference follows frozen offsets. This is
retrospective: the historical test had already been observed. It neither proves
encoder improvement nor reproduces the authors' published source benchmarks.

HGCLR's corrected source addresses unused-dropout RNG behavior and was checked
for implementation/gradient/update parity under tested conditions. This does
not establish universal correctness, original published-score recovery or
unsuitability of HGCLR for this domain. The historical adapters and corrected
source are separated to avoid silently replacing the code behind old numbers.

Both retriever stages use full-parameter supervised retrieval fine-tuning of
one shared Qwen3-Embedding-0.6B encoder. No LoRA/adapters/layer freezing. v5
conditions queries AND candidate documents on the field/role. Phase1 MNRL uses
batch80 and no explicit hard negatives; refinement H4 uses batch24 with four
mined negatives, hardness strength .2; H8 is an alternative refinement, batch22.
Neither stage masks repeated field–code identities across document instances.
False-negative collisions are possible; their historical frequency and
performance effect were not isolated. Do not describe a corrected multi-positive
objective or duplicate-free sampling as if executed in the reported study.

Historical Gemma full test on original reference: Top1 55.5350, Exact8 5.90.
On the common canonical silver subset (4,646 records, 37,168 decisions),
retriever/Gemma Top1 76.0547/76.5928, Exact8 19.7589/20.6629. Paired-record
bootstrap 10,000 replicates, seed20261018: Top1 difference +.5381 points,
95% CI [-.1157,1.1919]; Exact8 +.9040, CI [-.4090,2.1954]. These increases
are statistically inconclusive, not evidence of equivalence. Silver labels
are system/candidate dependent, not independent human gold. There are 639
pending-review flags on 407 records. Canonical validation fixes structural
acceptance and denominators; it does not settle all semantic disagreements.

The matched 32-pair thinking pilot had all 32 thinking responses truncated at
2048 tokens. This diagnoses an insufficient operational budget, not intrinsic
thinking quality. The ongoing Mac configuration uses a larger cap and must
be evaluated only after all 5,000 records and final verification.

The shared split and scoring do not imply equal backbones, compute, historical
tuning or decoder assumptions across methods. Do not claim universal SOTA,
all review concerns experimentally resolved, or complete five-seed replication.

