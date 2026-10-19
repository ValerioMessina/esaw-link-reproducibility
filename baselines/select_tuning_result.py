#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one predeclared configuration using validation Macro-F1 only."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    candidates = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("selection_split") != "validation":
            raise ValueError(f"{path} is not a validation-only result")
        candidates.append({
            "path": str(path),
            "best_macro_f1": float(payload["best_macro_f1"]),
            "hyperparameters": payload["hyperparameters"],
        })
    selected = max(candidates, key=lambda row: row["best_macro_f1"])
    result = {
        "selection_metric": "validation Macro-F1",
        "test_observed": False,
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
