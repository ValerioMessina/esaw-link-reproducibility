"""Recount private paired NPZ exports; emit aggregate metrics only, without record IDs."""
import argparse
import json
from pathlib import Path
import numpy as np

STAGES = ("frozen", "alignment", "refinement")
ARMS = ("role_on", "role_off")
FIELDS = ("COD_LOC", "COD_LAV", "COD_ATV", "COD_MAT_ATV", "COD_DEV",
          "COD_MAT_DEV", "COD_CON", "COD_MAT_CON")


def metrics(ranks, gold):
    result = {"per_field": {field: {} for field in FIELDS}}
    for k in (1, 5, 10):
        hits = (ranks[:, :, :k] == gold[:, :, None]).any(2)
        result[f"top{k}_pct"] = float(hits.mean() * 100)
        result[f"record_all8_top{k}_pct"] = float(hits.all(1).mean() * 100)
        for j, field in enumerate(FIELDS):
            result["per_field"][field][f"top{k}_pct"] = float(hits[:, j].mean() * 100)
    result["material_top1_pct"] = float((ranks[:, [3, 5, 7], 0] == gold[:, [3, 5, 7]]).mean() * 100)
    return result


def recount(root):
    root = Path(root)
    rng = np.random.default_rng(20261021)
    output = {"training_seed": 13, "test_records": 5000, "metrics": {},
              "paired_comparisons": {"replicates": 10000, "seed": 20261021, "results": {}}}
    for stage in STAGES:
        with np.load(root / "role_on" / stage / "test_predictions.npz", allow_pickle=False) as a, \
                np.load(root / "role_off" / stage / "test_predictions.npz", allow_pickle=False) as b:
            if not np.array_equal(a["record_keys"], b["record_keys"]) or not np.array_equal(a["gold_indices"], b["gold_indices"]):
                raise ValueError("Paired record order or gold labels differ")
            gold = a["gold_indices"]
            if gold.shape != (5000, 8) or len(np.unique(a["record_keys"])) != 5000:
                raise ValueError("Expected 5000 distinct complete test records")
            if any(v["rank_indices"].shape != (5000, 8, 10) for v in (a, b)):
                raise ValueError("Expected eight Top-10 lists per record")
            output["metrics"][stage] = {arm: metrics(v["rank_indices"], gold) for arm, v in zip(ARMS, (a, b))}
            on = a["rank_indices"][:, :, 0] == gold
            off = b["rank_indices"][:, :, 0] == gold
            differences = {"top1": on.mean(1) - off.mean(1),
                           "exact8": on.all(1).astype(float) - off.all(1).astype(float),
                           "material_top1": on[:, [3, 5, 7]].mean(1) - off[:, [3, 5, 7]].mean(1)}
            comparisons = {}
            for name, delta in differences.items():
                # Preserve historical RNG consumption across stages and metrics.
                bootstrap = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() * 100
                                      for _ in range(10000)])
                comparisons[name] = {"role_on_minus_off_pp": float(delta.mean() * 100),
                                     "paired_record_bootstrap95": np.quantile(bootstrap, [.025, .975]).tolist()}
            output["paired_comparisons"]["results"][stage] = comparisons
    return output


def verify(actual, expected):
    if isinstance(expected, dict):
        for key, value in expected.items():
            verify(actual[key], value)
    elif isinstance(expected, list):
        if not np.allclose(actual, expected, atol=1e-10, rtol=0):
            raise ValueError("Aggregate interval differs")
    elif not np.isclose(actual, expected, atol=1e-10, rtol=0):
        raise ValueError("Aggregate metric differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--check-published", action="store_true")
    args = parser.parse_args()
    output = recount(args.results)
    if args.check_published:
        expected = json.loads(Path(__file__).with_name("aggregate_evidence.json").read_text())
        for key in ("metrics", "paired_comparisons"):
            target = expected[key] if key == "metrics" else {k: expected[key][k] for k in ("replicates", "seed", "results")}
            verify(output[key], target)
        output["published_aggregates_verified"] = True
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
