from __future__ import annotations

import argparse
import json

from esaw_biencoder.label_holdout import generate_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera fold label-wise ESAW pre-registrati e anti-leakage.")
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--taxonomy", default="tassonomia.json")
    parser.add_argument("--split-file", default="artifacts/splits/esaw-fixed-seed13-test10-val10.json")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-support", type=int, default=30)
    parser.add_argument("--min-material-support-per-role", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = generate_manifest(
        args.dataset,
        args.taxonomy,
        args.split_file,
        seed=args.seed,
        folds=args.folds,
        min_train_support=args.min_train_support,
        min_material_support_per_role=args.min_material_support_per_role,
    )
    write_manifest(args.output, payload)
    summary = {
        "output": args.output,
        "protocol": payload["protocol"],
        "fold_count": payload["fold_count"],
        "primary_fold": payload["primary_fold"],
        "heldout_by_fold": {
            fold: value["heldout_taxonomy_codes_by_output_field"]
            for fold, value in payload["folds"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
