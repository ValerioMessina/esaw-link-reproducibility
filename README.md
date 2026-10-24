# ESAW-Link — reproducibility sources

Source release assembled from the study's frozen sources on 19 October 2026.
It contains the shared-encoder retriever, candidate mining, reranking and audit
code, the four HTC adaptations (HGCLR, HTC-CLIP, HiSR, HyILR), verification tools
and execution configurations. **No institutional records, record identifiers,
prediction exports, trained checkpoints or credentials are distributed.**

This is a source-code release, not a claim that every experimental gate
has passed or that the experiments have been independently reproduced from it.
The 21 October update adds [verified HTC test results](docs/BASELINE_TEST_20261021.md),
including all five available HGCLR checkpoints (four complete trainings, one
incomplete), and the original evaluation/recount sources.
The 24 October update adds the [matched role-conditioning ablation](experiments/role_conditioning/README.md):
verified 1+1-epoch results, recovered training source and a private-workspace launcher.
See [experimental status](docs/EXPERIMENT_STATUS.md),
[reproduction instructions](docs/REPRODUCING.md) and
[third-party notices](THIRD_PARTY_NOTICES.md).

## Layout

| Directory | Contents |
|---|---|
| `retriever/` | SSD ESAW-Link source: training, inference, role-conditioned query/candidate formats, mining, zero-shot, reranking, canonical audit |
| `baselines/` | Historical HTC adapters, preparation, training, original HGCLR dependencies |
| `baselines/hgclr_verified/` | Corrected HGCLR source and frozen five-seed protocol, separate from historical adapters |
| `configs/` | Portable executable command specifications; does not provision cloud compute |
| `scripts/` | Input checks, split generation, training-command launcher, decoder scoring, paired bootstrap and release checks |
| `tests/` | Synthetic offline checks and aggregate-evidence consistency checks, without institutional examples |
| `environments/` | Separate original declarations / tested runtime versions; not one invented universal lockfile |
| `historical_launchers/` | Original launchers for provenance; contain machine-specific paths and artifact packaging, do not use as portable entry points |
| `historical_verification/` | Original artifact-bound verification sources; require authorized private inputs at their original layout |
| `provenance/` | SHA-256 source manifest and release manifest |

## Start with offline verification

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_release.py
```

With authorized data stored **outside this repository**, check input consistency:

```bash
python3 scripts/check_inputs.py \
  --dataset /secure/esaw/dataset.json \
  --taxonomy /secure/esaw/tassonomia.json \
  --split-file /secure/esaw/split.json
```

Generate a command without starting training:

```bash
python3 scripts/run_experiment.py configs/phase1.json \
  --dataset /secure/esaw/dataset.json --taxonomy /secure/esaw/tassonomia.json \
  --split-file /secure/esaw/split.json --work-dir /secure/esaw/runs
```

Use `--execute` only after installing the matching environment. Default behavior
prints the exact command; no cloud resources or paid API requests are created.
All configured outputs must be outside this repository. The source scripts can
write sensitive predictions: preserve their outputs privately too.

The original taxonomy is supplied by the authorized researcher, not silently
replaced with a corrected vocabulary. Its SHA-256 and the known activity-19
description issue are documented in `docs/DATA_AND_TAXONOMY.md`.

## Release boundary

Run `python3 scripts/check_release.py` before staging and before any upload.
The ignored paths are an additional guard, not a substitute for reviewing files.
The study's private data cannot be reconstructed from this release. Repeating
the institutional evaluation requires lawful access to the original archive
and frozen split, or a separately declared new study on a different corpus.

## License and citation

Original code and software documentation are released under the
[MIT license](LICENSE). It permits reuse, modification and redistribution,
including commercial use, with preservation of copyright/license notices.
Upstream HGCLR notices are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
This software license does not cover institutional datasets, model weights
or the manuscript. See [CITATION.cff](CITATION.cff) for software citation metadata;
no unpublished paper DOI or acceptance status is asserted.

