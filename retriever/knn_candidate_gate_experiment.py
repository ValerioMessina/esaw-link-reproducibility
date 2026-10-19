from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from multiclass_candidate_gate_experiment import (
    group_normalize,
    multiclass_scores,
    tune_decode,
    write_predictions,
)
from three_source_holdout_gate import (
    FIELDS,
    baseline_report,
    load_audit,
    load_part,
    rows_by_key,
    score_components,
    transform_feature_spaces,
)


def weight_grid() -> list[tuple[float, float, float]]:
    # Griglia mirata: copre sorgenti singole e blend plausibili senza ripetere
    # centinaia di policy quasi equivalenti sulla stessa validation.
    return [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.5),
        (0.4, 0.2, 0.4),
        (0.4, 0.4, 0.2),
        (0.6, 0.2, 0.2),
        (0.2, 0.4, 0.4),
        (0.2, 0.2, 0.6),
    ]


def knn_option_scores(
    val_keys, val_groups, val_options, train_keys, audit, neighbor_indices, similarities,
    k: int, weighting: str,
) -> np.ndarray:
    output = np.zeros(len(val_options), dtype=np.float64)
    for row_index, key in enumerate(val_keys):
        indexes = neighbor_indices[row_index, :k]
        sims = np.maximum(similarities[row_index, :k], 0.0)
        if weighting == "uniform":
            weights = np.ones_like(sims)
        elif weighting == "sim1":
            weights = sims
        elif weighting == "sim2":
            weights = sims ** 2
        elif weighting == "sim4":
            weights = sims ** 4
        else:
            temperature = float(weighting.removeprefix("softmax"))
            shifted = (sims - float(sims.max())) / temperature
            weights = np.exp(np.clip(shifted, -40.0, 40.0))
        denominator = max(float(weights.sum()), 1e-12)
        for field in FIELDS:
            votes: dict[str, float] = {}
            for neighbor, weight in zip(indexes, weights):
                code = audit[train_keys[int(neighbor)]][field]
                votes[code] = votes.get(code, 0.0) + float(weight)
            for option_index in val_groups[(key, field)]:
                output[option_index] = votes.get(val_options[option_index].code, 0.0) / denominator
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top3", nargs="+", required=True)
    parser.add_argument("--top10", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--multiclass-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = load_audit(Path(args.audit_jsonl))
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))["split"]
    top3 = rows_by_key([Path(value) for value in args.top3])
    top10 = rows_by_key([Path(args.top10)])
    train_keys, train_options, train_groups = load_part("train", split, audit, top3, top10)
    val_keys, val_options, val_groups = load_part("validation", split, audit, top3, top10)

    payload = joblib.load(args.multiclass_gate)
    word, char = payload["word_vectorizer"], payload["char_vectorizer"]
    train_texts = [str(top10[key].get("text") or "") for key in train_keys]
    val_texts = [str(top10[key].get("text") or "") for key in val_keys]
    x_train = normalize(hstack([word.transform(train_texts), char.transform(train_texts)], format="csr"))
    x_val = normalize(hstack([word.transform(val_texts), char.transform(val_texts)], format="csr"))
    neighbors = NearestNeighbors(n_neighbors=80, metric="cosine", algorithm="brute", n_jobs=-1)
    neighbors.fit(x_train)
    distances, neighbor_indices = neighbors.kneighbors(x_val, return_distance=True)
    similarities = 1.0 - distances

    base_payload = payload["base_gate"]
    val_meta, val_full, val_tree = transform_feature_spaces(
        val_options,
        base_payload["dict_vectorizer"],
        base_payload["text_vectorizer"],
        base_payload["tree_vectorizer"],
    )
    base_scores = score_components(
        base_payload["components"],
        {"text_meta": val_full, "tree_dense": val_tree, "meta": val_meta},
    )
    base_prob = group_normalize(val_keys, val_groups, base_scores, 1.0)
    direct_scores = multiclass_scores(payload["models"], x_val, val_keys, val_groups, val_options)
    direct_prob = group_normalize(val_keys, val_groups, direct_scores, float(payload["temperature"]))

    best = None
    best_state = None
    trials = []
    for k in (3, 5, 10, 20, 40, 80):
        for weighting in ("uniform", "sim1", "sim2", "sim4", "softmax0.05", "softmax0.1", "softmax0.2"):
            knn_prob = knn_option_scores(
                val_keys, val_groups, val_options, train_keys, audit,
                neighbor_indices, similarities, k, weighting,
            )
            for base_weight, direct_weight, knn_weight in weight_grid():
                decisions = base_weight * base_prob + direct_weight * direct_prob + knn_weight * knn_prob
                name = (
                    f"knn{k}_{weighting}_b{base_weight:g}_d{direct_weight:g}_k{knn_weight:g}"
                )
                result = tune_decode(val_keys, val_groups, val_options, decisions, name)
                serializable = {key: value for key, value in result.items() if key not in {"rank", "choices"}}
                serializable.update({
                    "neighbors": k,
                    "weighting": weighting,
                    "base_weight": base_weight,
                    "direct_weight": direct_weight,
                    "knn_weight": knn_weight,
                })
                trials.append(serializable)
                if best is None or result["rank"] > best["rank"]:
                    best = result
                    best_state = serializable

    assert best is not None and best_state is not None
    report = {
        "schema": "esaw-knn-candidate-gate-validation-v1",
        "train_records": len(train_keys),
        "validation_records": len(val_keys),
        "test_evaluated": False,
        "nearest_similarity": {
            "mean_top1": float(similarities[:, 0].mean()),
            "median_top1": float(np.median(similarities[:, 0])),
            "p90_top1": float(np.quantile(similarities[:, 0], 0.9)),
        },
        "validation_baselines": baseline_report(val_keys, val_groups, val_options),
        "best": best_state,
        "trials": trials,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_predictions(
        output_dir / "validation_predictions.jsonl",
        val_keys, val_groups, val_options, best["choices"],
    )
    print(json.dumps({"best": best_state, "nearest_similarity": report["nearest_similarity"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
