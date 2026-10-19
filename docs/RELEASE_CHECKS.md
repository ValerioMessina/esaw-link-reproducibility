# Checks performed on the initial source release

Verified locally on 19 October 2026:

- All copied source snapshots match SHA-256 values in the source manifest.
- Eleven synthetic offline tests pass: canonical response acceptance/rejection,
  strict booleans, material depth checks, split determinism/disjointness,
  external-output guards, role-conditioned candidates, offset selection/ties,
  and paired-record bootstrap behavior.
- A synthetic CLI fit → persist decoder → test-scoring sequence passes.
- CLI imports/help pass for retriever training, mining and LLM reranking;
  baseline preparation/assets, HTC training, HiSR/HyILR training and the separate
  corrected HGCLR trainer.
- External institutional input checks pass for archive size, unique identities,
  disjoint split membership, target validity, material depth and frozen taxonomy
  SHA-256. The data were read externally, never exported into this repository.
- Release scan checks source syntax, hashes, forbidden artifact types/names,
  oversized/binary files, symlinks and common credential patterns.
- An additional local comparison of exported source text against normalized
  64-character prefixes of the six narrative fields in all 50,000 external
  records found no matching source files. This specific scan is an extra check,
  not a universal anonymization guarantee.

No new full model training, full inference parity run or paid API call was
performed to validate this assembly. No model weights, institutional examples,
private task streams, per-record outputs, or private split IDs were copied.
The software release is not an independent reproduction certificate.
Long-run results and incomplete
experiments are distinguished in `EXPERIMENT_STATUS.md`.
