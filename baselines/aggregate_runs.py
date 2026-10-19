#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    scalar_names = [
        "test_micro_f1",
        "test_macro_f1",
        "record_exact@1",
        "record_recall@5",
        "record_recall@10",
    ]
    values: dict[str, list[float]] = {}
    for name in scalar_names:
        observed = [float(run[name]) for run in runs if name in run]
        if observed:
            values[name] = observed
    for name in ("top1_accuracy", "top5_accuracy", "top10_accuracy"):
        observed = [float(run["mean"][name]) for run in runs if "mean" in run and name in run["mean"]]
        if observed:
            values[name] = observed
    result = {
        "method": args.method,
        "runs": len(runs),
        "inputs": [str(path) for path in args.inputs],
        "mean": {name: statistics.fmean(observed) for name, observed in values.items()},
        "sample_std": {
            name: statistics.stdev(observed) if len(observed) > 1 else 0.0
            for name, observed in values.items()
        },
        "per_run": [
            {
                "seed": run.get("seed"),
                **{
                    name: (
                        run["mean"][name]
                        if name in {"top1_accuracy", "top5_accuracy", "top10_accuracy"}
                        else run[name]
                    )
                    for name in values
                    if name in run or ("mean" in run and name in run["mean"])
                },
            }
            for run in runs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
