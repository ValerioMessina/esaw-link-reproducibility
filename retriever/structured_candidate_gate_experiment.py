from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from knn_candidate_gate_experiment import knn_option_scores
from multiclass_candidate_gate_experiment import group_normalize, multiclass_scores, write_predictions
from three_source_holdout_gate import (
    FIELDS,
    baseline_report,
    load_audit,
    load_part,
    metrics,
    rows_by_key,
    score_components,
    source_choices,
    transform_feature_spaces,
)


RELATED_PAIRS = {
    ("COD_LOC", "COD_LAV"),
    ("COD_ATV", "COD_MAT_ATV"),
    ("COD_DEV", "COD_MAT_DEV"),
    ("COD_CON", "COD_MAT_CON"),
}


def compatibility_tables(train_keys, audit, smoothing: float):
    marginal = {field: Counter(audit[key][field] for key in train_keys) for field in FIELDS}
    pair_counts = {
        (left, right): Counter((audit[key][left], audit[key][right]) for key in train_keys)
        for left, right in itertools.combinations(FIELDS, 2)
    }
    code_counts = {field: max(len(marginal[field]), 1) for field in FIELDS}
    n = len(train_keys)
    tables = {}
    for (left, right), counts in pair_counts.items():
        right_size = code_counts[right]
        marginal_right_denominator = n + smoothing * right_size
        table = {}
        for (left_code, right_code), count in counts.items():
            conditional = (count + smoothing) / (marginal[left][left_code] + smoothing * right_size)
            marginal_probability = (marginal[right][right_code] + smoothing) / marginal_right_denominator
            table[(left_code, right_code)] = math.log(max(conditional, 1e-12) / max(marginal_probability, 1e-12))
        tables[(left, right)] = table
    return tables


def structured_decode(
    keys, groups, options, unary, tables, pair_mode: str, pair_weight: float,
    max_changes: int, beam_size: int,
):
    output = {}
    for key in keys:
        majority = {
            field: max(
                groups[(key, field)],
                key=lambda index: (options[index].features["votes"], options[index].features["support_dual"]),
            )
            for field in FIELDS
        }
        # state: (score, changes, selected_indices)
        beam = [(0.0, 0, [])]
        for field_position, field in enumerate(FIELDS):
            expanded = []
            for score, changes, selected in beam:
                for option_index in groups[(key, field)]:
                    next_changes = changes + int(option_index != majority[field])
                    if next_changes > max_changes:
                        continue
                    option = options[option_index]
                    next_score = score + math.log(max(float(unary[option_index]), 1e-8))
                    for previous_position, previous_index in enumerate(selected):
                        previous_field = FIELDS[previous_position]
                        pair = (previous_field, field)
                        if pair_mode == "related" and pair not in RELATED_PAIRS:
                            continue
                        previous_code = options[previous_index].code
                        pair_value = tables[pair].get((previous_code, option.code), 0.0)
                        next_score += pair_weight * float(np.clip(pair_value, -3.0, 3.0))
                    expanded.append((next_score, next_changes, selected + [option_index]))
            expanded.sort(key=lambda item: item[0], reverse=True)
            beam = expanded[:beam_size]
        selected = beam[0][2]
        for field, option_index in zip(FIELDS, selected):
            output[(key, field)] = option_index
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
    neighbors = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute", n_jobs=-1).fit(x_train)
    distances, neighbor_indices = neighbors.kneighbors(x_val, return_distance=True)

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
    knn_prob = knn_option_scores(
        val_keys, val_groups, val_options, train_keys, audit,
        neighbor_indices, 1.0 - distances, 5, "uniform",
    )
    unary_variants = {
        "base": base_prob,
        "multiclass_best": 0.4 * base_prob + 0.6 * direct_prob,
        "knn_best": 0.6 * base_prob + 0.2 * direct_prob + 0.2 * knn_prob,
    }

    trials = []
    best = None
    best_choices = None
    for smoothing in (1.0,):
        tables = compatibility_tables(train_keys, audit, smoothing)
        for unary_name in ("base", "knn_best"):
            unary = unary_variants[unary_name]
            for pair_mode in ("related", "all"):
                for pair_weight in (0.0, 0.05, 0.1, 0.2, 0.4):
                    for max_changes in (2, 3, 4, 8):
                        choices = structured_decode(
                            val_keys, val_groups, val_options, unary, tables,
                            pair_mode, pair_weight, max_changes, beam_size=64,
                        )
                        value = metrics(val_keys, val_options, choices)
                        rank = (value["exact8"], value["ge7"], value["field_accuracy"], -max_changes, -pair_weight)
                        trial = {
                            "unary": unary_name,
                            "smoothing": smoothing,
                            "pair_mode": pair_mode,
                            "pair_weight": pair_weight,
                            "max_changes": max_changes,
                            "metrics": value,
                        }
                        trials.append(trial)
                        if best is None or rank > best[0]:
                            best = (rank, trial)
                            best_choices = choices

    assert best is not None and best_choices is not None
    report = {
        "schema": "esaw-structured-candidate-gate-validation-v1",
        "train_records": len(train_keys),
        "validation_records": len(val_keys),
        "test_evaluated": False,
        "validation_baselines": baseline_report(val_keys, val_groups, val_options),
        "best": best[1],
        "trials": trials,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_predictions(
        output_dir / "validation_predictions.jsonl",
        val_keys, val_groups, val_options, best_choices,
    )
    print(json.dumps({"best": best[1]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
