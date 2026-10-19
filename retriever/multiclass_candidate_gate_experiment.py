from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from three_source_holdout_gate import (
    FIELDS,
    baseline_report,
    decode,
    load_audit,
    load_part,
    metrics,
    rows_by_key,
    score_components,
    source_choices,
    transform_feature_spaces,
)


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    shifted = (values - float(np.max(values))) / max(temperature, 1e-6)
    exponentials = np.exp(np.clip(shifted, -40.0, 40.0))
    return exponentials / max(float(exponentials.sum()), 1e-12)


def group_normalize(keys, groups, values, temperature: float) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.float64)
    for key in keys:
        for field in FIELDS:
            indices = list(groups[(key, field)])
            output[indices] = softmax(np.asarray(values[indices], dtype=np.float64), temperature)
    return output


def tune_decode(keys, groups, options, decisions, name: str) -> dict:
    best = None
    for threshold in (-1.0, -0.5, -0.2, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2):
        for max_changes in (0, 1, 2, 3, 4, 5, 6, 8):
            choices = decode(keys, groups, options, decisions, threshold, max_changes)
            value = metrics(keys, options, choices)
            rank = (value["exact8"], value["ge7"], value["field_accuracy"], -max_changes, -abs(threshold))
            candidate = {
                "name": name,
                "threshold": threshold,
                "max_changes": max_changes,
                "metrics": value,
                "rank": rank,
                "choices": choices,
            }
            if best is None or rank > best["rank"]:
                best = candidate
    assert best is not None
    return best


def multiclass_scores(models, matrix, keys, groups, options) -> np.ndarray:
    key_position = {key: index for index, key in enumerate(keys)}
    output = np.full(len(options), -1e6, dtype=np.float64)
    for field in FIELDS:
        model = models[field]
        decision = np.asarray(model.decision_function(matrix), dtype=np.float64)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        class_index = {str(code): index for index, code in enumerate(model.classes_)}
        for key in keys:
            row_index = key_position[key]
            for option_index in groups[(key, field)]:
                code = options[option_index].code
                if code in class_index:
                    output[option_index] = float(decision[row_index, class_index[code]])
    return output


def write_predictions(path: Path, keys, groups, options, choices) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key in keys:
            handle.write(json.dumps({
                "record_key": key,
                "labels": {field: options[choices[(key, field)]].code for field in FIELDS},
            }, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top3", nargs="+", required=True)
    parser.add_argument("--top10", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--base-gate", required=True)
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

    train_texts = [str(top10[key].get("text") or "") for key in train_keys]
    val_texts = [str(top10[key].get("text") or "") for key in val_keys]
    word = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2,
        max_df=0.995, max_features=80_000, sublinear_tf=True, dtype=np.float64,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", lowercase=True, strip_accents="unicode", ngram_range=(3, 5),
        min_df=2, max_features=160_000, sublinear_tf=True, dtype=np.float64,
    )
    train_word = word.fit_transform(train_texts)
    train_char = char.fit_transform(train_texts)
    x_train = hstack([train_word, train_char], format="csr")
    x_val = hstack([word.transform(val_texts), char.transform(val_texts)], format="csr")

    base_payload = joblib.load(args.base_gate)
    val_meta, val_full, val_tree = transform_feature_spaces(
        val_options,
        base_payload["dict_vectorizer"],
        base_payload["text_vectorizer"],
        base_payload["tree_vectorizer"],
    )
    base_decisions = score_components(
        base_payload["components"],
        {"text_meta": val_full, "tree_dense": val_tree, "meta": val_meta},
    )

    trials = []
    best = None
    best_payload = None
    train_key_position = {key: index for index, key in enumerate(train_keys)}
    for class_weight in (None, "balanced"):
        for c_value in (0.1, 0.3, 1.0, 3.0):
            models = {}
            for field in FIELDS:
                labels = np.asarray([audit[key][field] for key in train_keys])
                model = LinearSVC(C=c_value, class_weight=class_weight, random_state=13)
                model.fit(x_train, labels)
                models[field] = model
            direct = multiclass_scores(models, x_val, val_keys, val_groups, val_options)
            for temperature in (0.25, 0.5, 1.0, 2.0):
                direct_prob = group_normalize(val_keys, val_groups, direct, temperature)
                for alpha in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                    decisions = (1.0 - alpha) * base_decisions + alpha * direct_prob
                    name = f"svc_c{c_value:g}_{class_weight or 'plain'}_t{temperature:g}_a{alpha:g}"
                    result = tune_decode(val_keys, val_groups, val_options, decisions, name)
                    serializable = {key: value for key, value in result.items() if key not in {"rank", "choices"}}
                    serializable.update({
                        "c": c_value,
                        "class_weight": class_weight,
                        "temperature": temperature,
                        "alpha": alpha,
                    })
                    trials.append(serializable)
                    if best is None or result["rank"] > best["rank"]:
                        best = result
                        best_payload = {
                            "word_vectorizer": word,
                            "char_vectorizer": char,
                            "models": models,
                            "c": c_value,
                            "class_weight": class_weight,
                            "temperature": temperature,
                            "alpha": alpha,
                            "threshold": result["threshold"],
                            "max_changes": result["max_changes"],
                            "base_gate": base_payload,
                        }

    assert best is not None and best_payload is not None
    report = {
        "schema": "esaw-multiclass-candidate-gate-validation-v1",
        "train_records": len(train_keys),
        "validation_records": len(val_keys),
        "vocabulary": {"word": len(word.vocabulary_), "char": len(char.vocabulary_)},
        "validation_baselines": baseline_report(val_keys, val_groups, val_options),
        "best": {key: value for key, value in best.items() if key not in {"rank", "choices"}},
        "trials": trials,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(best_payload, output_dir / "multiclass_candidate_gate.joblib")
    write_predictions(
        output_dir / "validation_predictions.jsonl",
        val_keys, val_groups, val_options, best["choices"],
    )
    print(json.dumps({"best": report["best"], "vocabulary": report["vocabulary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
