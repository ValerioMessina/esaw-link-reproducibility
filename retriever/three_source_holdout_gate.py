from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from four_source_exact8_gate import (
    FIELDS,
    Option,
    code_of,
    decode,
    load_audit,
    metrics,
    model_candidates,
    model_score,
    rows_by_key,
)


EXPECTED_COUNTS = {"train": 3250, "validation": 696, "test": 697}


def build_options(
    keys: Sequence[str],
    audit: Mapping[str, Mapping[str, str]],
    top3: Mapping[str, Mapping[str, Any]],
    top10: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Option], dict[tuple[str, str], list[int]]]:
    options: list[Option] = []
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for key in keys:
        row3, row10 = top3[key], top10[key]
        record_sources = {
            field: {
                "dual": code_of(row10, "dual_top1", field),
                "top3": code_of(row3, "llm_top1", field),
                "top10": code_of(row10, "llm_top1", field),
            }
            for field in FIELDS
        }
        record_dual_margins = []
        for field in FIELDS:
            candidates = row10.get("dual_candidates", {}).get(field, [])
            scores = [float(item.get("score") or 0.0) for item in candidates[:2] if isinstance(item, Mapping)]
            record_dual_margins.append(scores[0] - scores[1] if len(scores) >= 2 else 0.0)
        record_features = {
            "record_unanimous3": sum(len(set(values.values())) == 1 for values in record_sources.values()),
            "record_majority3": sum(max(Counter(values.values()).values()) >= 2 for values in record_sources.values()),
            "record_dual_top3_agree": sum(values["dual"] == values["top3"] for values in record_sources.values()),
            "record_dual_top10_agree": sum(values["dual"] == values["top10"] for values in record_sources.values()),
            "record_top3_top10_agree": sum(values["top3"] == values["top10"] for values in record_sources.values()),
            "record_dual_margin_min": min(record_dual_margins),
            "record_dual_margin_mean": float(np.mean(record_dual_margins)),
            "record_dual_margin_max": max(record_dual_margins),
        }
        for field in FIELDS:
            sources = record_sources[field]
            if any(not code for code in sources.values()):
                raise RuntimeError(f"Codice sorgente mancante per {key}/{field}: {sources}")
            votes = Counter(sources.values())
            pattern = f"unique{len(votes)}"
            dual_candidates = row10.get("dual_candidates", {}).get(field, [])
            dual_map = {
                str(item.get("code") or ""): item
                for item in dual_candidates if isinstance(item, Mapping)
            }
            label_map = {
                str(item.get("code") or ""): str(item.get("label") or "")
                for item in dual_candidates if isinstance(item, Mapping)
            }
            for source_row in (row3, row10):
                item = source_row.get("llm_top1", {}).get(field, {})
                if isinstance(item, Mapping) and item.get("code"):
                    label_map[str(item["code"])] = str(item.get("label") or "")
            top_scores = [float(item.get("score") or 0.0) for item in dual_candidates[:3]]
            while len(top_scores) < 3:
                top_scores.append(0.0)
            for code in dict.fromkeys(sources.values()):
                dual_item = dual_map.get(code, {})
                support = {name: int(value == code) for name, value in sources.items()}
                features = {
                    "field": field,
                    "field_code": f"{field}:{code}",
                    "code_prefix1": f"{field}:{code[:1]}",
                    "code_prefix2": f"{field}:{code[:2]}",
                    "code_segments": len(code.split(".")),
                    "pattern": pattern,
                    "support_pattern": "_".join(name for name, value in support.items() if value),
                    "votes": votes[code],
                    **{f"support_{name}": value for name, value in support.items()},
                    "dual_rank": int(dual_item.get("rank_dual_encoder") or 99),
                    "dual_score": float(dual_item.get("score") or 0.0),
                    "dual_margin12": top_scores[0] - top_scores[1],
                    "dual_margin13": top_scores[0] - top_scores[2],
                    "text_input": (
                        f"{row10.get('text', '')} Variabile ESAW {field}. "
                        f"Codice candidato {code}. Classe candidata {label_map.get(code, '')}."
                    ),
                    **record_features,
                }
                index = len(options)
                options.append(Option(key, field, code, audit[key][field], features))
                groups[(key, field)].append(index)
    return options, groups


def load_part(
    part: str,
    split: Mapping[str, Sequence[str]],
    audit: Mapping[str, Mapping[str, str]],
    top3: Mapping[str, Mapping[str, Any]],
    top10: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[Option], dict[tuple[str, str], list[int]]]:
    keys = list(split[part])
    expected = EXPECTED_COUNTS[part]
    if len(keys) != expected or len(set(keys)) != expected:
        raise RuntimeError(f"Split {part}: attese {expected} chiavi uniche, trovate {len(keys)}/{len(set(keys))}")
    missing = {
        "audit": [key for key in keys if key not in audit],
        "top3": [key for key in keys if key not in top3],
        "top10": [key for key in keys if key not in top10],
    }
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        summary = {name: {"count": len(values), "sample": values[:5]} for name, values in missing.items()}
        raise RuntimeError(f"Input incompleti per {part}: {json.dumps(summary, ensure_ascii=False)}")
    options, groups = build_options(keys, audit, top3, top10)
    return keys, options, groups


def source_choices(keys, groups, options, source: str):
    output = {}
    for key in keys:
        for field in FIELDS:
            indices = groups[(key, field)]
            if source == "majority":
                output[(key, field)] = max(
                    indices,
                    key=lambda index: (
                        options[index].features["votes"],
                        options[index].features["support_dual"],
                    ),
                )
            elif source == "oracle":
                output[(key, field)] = next(
                    (index for index in indices if options[index].code == options[index].gold),
                    indices[0],
                )
            else:
                output[(key, field)] = next(
                    index for index in indices if options[index].features[f"support_{source}"]
                )
    return output


def decode_single_source(keys, groups, options, decisions):
    output = {}
    for key in keys:
        candidates = {}
        for source in ("dual", "top3", "top10"):
            indices = [
                next(
                    index for index in groups[(key, field)]
                    if options[index].features[f"support_{source}"]
                )
                for field in FIELDS
            ]
            candidates[source] = (sum(float(decisions[index]) for index in indices), indices)
        _source, (_score, selected) = max(
            candidates.items(), key=lambda item: (item[1][0], item[0] == "dual")
        )
        for field, index in zip(FIELDS, selected):
            output[(key, field)] = index
    return output


def decode_policy(keys, groups, options, decisions, policy):
    if policy["decoder"] == "single_source":
        return decode_single_source(keys, groups, options, decisions)
    return decode(
        keys, groups, options, decisions,
        float(policy["threshold"]), int(policy["max_changes"]),
    )


def baseline_report(keys, groups, options):
    return {
        source: metrics(keys, options, source_choices(keys, groups, options, source))
        for source in ("dual", "top3", "top10", "majority", "oracle")
    }


def fit_feature_spaces(options: Sequence[Option]):
    dict_vectorizer = DictVectorizer(sparse=True, dtype=np.float64)
    text_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=60_000,
        sublinear_tf=True,
        dtype=np.float64,
    )
    meta = dict_vectorizer.fit_transform([
        {key: value for key, value in option.features.items() if key != "text_input"}
        for option in options
    ])
    text = text_vectorizer.fit_transform([str(option.features["text_input"]) for option in options])
    tree_vectorizer = DictVectorizer(sparse=False, dtype=np.float64)
    tree = tree_vectorizer.fit_transform([
        {
            key: value for key, value in option.features.items()
            if key not in {"text_input", "field_code"}
        }
        for option in options
    ])
    return dict_vectorizer, text_vectorizer, tree_vectorizer, meta, hstack([meta, text], format="csr"), tree


def transform_feature_spaces(options, dict_vectorizer, text_vectorizer, tree_vectorizer):
    meta = dict_vectorizer.transform([
        {key: value for key, value in option.features.items() if key != "text_input"}
        for option in options
    ])
    text = text_vectorizer.transform([str(option.features["text_input"]) for option in options])
    tree = tree_vectorizer.transform([
        {
            key: value for key, value in option.features.items()
            if key not in {"text_input", "field_code"}
        }
        for option in options
    ])
    return meta, hstack([meta, text], format="csr"), tree


def holdout_model_candidates():
    candidates = model_candidates()
    candidates.extend(
        (
            f"random_forest_leaf{leaf}",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=leaf,
                max_features=0.75,
                class_weight="balanced",
                random_state=13,
                n_jobs=-1,
            ),
        )
        for leaf in (2, 5, 10)
    )
    candidates.extend(
        (
            f"hist_gradient_leaf{leaf}",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=leaf,
                l2_regularization=1.0,
                random_state=13,
            ),
        )
        for leaf in (20, 50, 100)
    )
    return candidates


def probability_score(model, x):
    values = model_score(model, x)
    if hasattr(model, "decision_function"):
        values = 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
    return np.asarray(values, dtype=np.float64)


def evaluate_decisions(keys, groups, options, decisions, model_name, feature_space):
    best = None
    thresholds = (-1.0, -0.5, -0.2, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.7, 1.0, 1.5, 2.0)
    for threshold in thresholds:
        for max_changes in (0, 1, 2, 3, 4, 5, 6, 8):
            choices = decode(keys, groups, options, decisions, threshold, max_changes)
            value = metrics(keys, options, choices)
            rank = (value["exact8"], value["ge7"], value["field_accuracy"], -max_changes, -abs(threshold))
            result = {
                "model_name": model_name, "feature_space": feature_space,
                "decoder": "field_gate", "threshold": threshold,
                "max_changes": max_changes, "metrics": value, "rank": rank,
            }
            if best is None or rank > best["rank"]:
                best = result
    choices = decode_single_source(keys, groups, options, decisions)
    value = metrics(keys, options, choices)
    rank = (value["exact8"], value["ge7"], value["field_accuracy"], -8, 0.0)
    result = {
        "model_name": model_name, "feature_space": feature_space,
        "decoder": "single_source", "threshold": None,
        "max_changes": None, "metrics": value, "rank": rank,
    }
    if best is None or rank > best["rank"]:
        best = result
    return best


def score_components(components, spaces):
    output = None
    for component in components:
        values = probability_score(component["model"], spaces[component["feature_space"]])
        weighted = float(component["weight"]) * values
        output = weighted if output is None else output + weighted
    if output is None:
        raise RuntimeError("Gate senza componenti")
    return output


def write_predictions(path: Path, keys, groups, options, choices):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key in keys:
            handle.write(json.dumps({
                "record_key": key,
                "labels": {field: options[choices[(key, field)]].code for field in FIELDS},
            }, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate holdout supervisionato dual/Gemma Top-3/Gemma Top-10.")
    parser.add_argument("--mode", choices=["train", "apply"], required=True)
    parser.add_argument("--part", choices=["validation", "test"], default="validation")
    parser.add_argument("--top3", nargs="+", required=True)
    parser.add_argument("--top10", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--predictions-output", required=True)
    args = parser.parse_args()

    audit = load_audit(Path(args.audit_jsonl))
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))["split"]
    top3 = rows_by_key([Path(value) for value in args.top3])
    top10 = rows_by_key([Path(args.top10)])

    if args.mode == "train":
        train_keys, train_options, train_groups = load_part("train", split, audit, top3, top10)
        val_keys, val_options, val_groups = load_part("validation", split, audit, top3, top10)
        (
            dict_vectorizer, text_vectorizer, tree_vectorizer,
            x_train_meta, x_train_full, x_train_tree,
        ) = fit_feature_spaces(train_options)
        x_val_meta, x_val_full, x_val_tree = transform_feature_spaces(
            val_options, dict_vectorizer, text_vectorizer, tree_vectorizer
        )
        y_train = np.asarray([option.code == option.gold for option in train_options], dtype=np.int8)
        actionable_train = np.asarray([
            len(train_groups[(option.key, option.field)]) > 1
            for option in train_options
        ], dtype=bool)
        if not actionable_train.any() or len(np.unique(y_train[actionable_train])) < 2:
            raise RuntimeError("Training gate senza disaccordi supervisionabili di entrambe le classi")

        best = None
        best_components = None
        candidate_reports = []
        trained = []
        for base_model_name, prototype in holdout_model_candidates():
            if base_model_name.startswith("logistic_"):
                feature_space = "text_meta"
                x_train, x_val = x_train_full, x_val_full
            elif base_model_name.startswith("hist_gradient_"):
                feature_space = "tree_dense"
                x_train, x_val = x_train_tree, x_val_tree
            else:
                feature_space = "meta"
                x_train, x_val = x_train_meta, x_val_meta
            for training_regime, fit_mask in (("all", np.ones(len(y_train), dtype=bool)), ("disagreements", actionable_train)):
                model_name = f"{base_model_name}_{training_regime}"
                model = clone(prototype)
                model.fit(x_train[fit_mask], y_train[fit_mask])
                decisions = probability_score(model, x_val)
                candidate_best = evaluate_decisions(
                    val_keys, val_groups, val_options, decisions, model_name, feature_space
                )
                candidate_best["training_regime"] = training_regime
                component = {
                    "model_name": model_name,
                    "model": model,
                    "feature_space": feature_space,
                    "probabilities": decisions,
                    "best": candidate_best,
                }
                trained.append(component)
                if best is None or candidate_best["rank"] > best["rank"]:
                    best = candidate_best
                    best_components = [{"model": model, "feature_space": feature_space, "weight": 1.0}]
                candidate_reports.append({key: value for key, value in candidate_best.items() if key != "rank"})

        broad_families = {}
        for component in trained:
            name = component["model_name"]
            family = next(
                prefix for prefix in ("logistic", "extra_trees", "random_forest", "hist_gradient")
                if name.startswith(prefix)
            )
            previous = broad_families.get(family)
            if previous is None or component["best"]["rank"] > previous["best"]["rank"]:
                broad_families[family] = component
        top_components = sorted(
            broad_families.values(), key=lambda item: item["best"]["rank"], reverse=True
        )
        ensemble_reports = []
        for left, right in itertools.combinations(top_components, 2):
            for left_weight in (0.25, 0.5, 0.75):
                right_weight = 1.0 - left_weight
                ensemble_name = (
                    f"blend_{left_weight:g}x{left['model_name']}_"
                    f"{right_weight:g}x{right['model_name']}"
                )
                decisions = left_weight * left["probabilities"] + right_weight * right["probabilities"]
                result = evaluate_decisions(
                    val_keys, val_groups, val_options, decisions, ensemble_name, "ensemble"
                )
                ensemble_reports.append({key: value for key, value in result.items() if key != "rank"})
                if best is None or result["rank"] > best["rank"]:
                    best = result
                    best_components = [
                        {"model": left["model"], "feature_space": left["feature_space"], "weight": left_weight},
                        {"model": right["model"], "feature_space": right["feature_space"], "weight": right_weight},
                    ]
        if len(top_components) >= 3:
            selected_three = top_components[:3]
            decisions = sum(item["probabilities"] for item in selected_three) / 3.0
            result = evaluate_decisions(
                val_keys, val_groups, val_options, decisions,
                "blend_uniform_top3", "ensemble",
            )
            ensemble_reports.append({key: value for key, value in result.items() if key != "rank"})
            if best is None or result["rank"] > best["rank"]:
                best = result
                best_components = [
                    {"model": item["model"], "feature_space": item["feature_space"], "weight": 1.0 / 3.0}
                    for item in selected_three
                ]

        assert best is not None and best_components is not None
        policy = {
            "decoder": best["decoder"], "threshold": best["threshold"],
            "max_changes": best["max_changes"],
        }
        payload = {
            "schema": "esaw-three-source-holdout-gate-v1",
            "dict_vectorizer": dict_vectorizer,
            "text_vectorizer": text_vectorizer,
            "tree_vectorizer": tree_vectorizer,
            "components": best_components,
            "policy": policy,
            "model_name": best["model_name"],
            "feature_space": best["feature_space"],
        }
        Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, args.model_output)
        validation_spaces = {
            "text_meta": x_val_full,
            "tree_dense": x_val_tree,
            "meta": x_val_meta,
        }
        val_decisions = score_components(best_components, validation_spaces)
        selected = decode_policy(val_keys, val_groups, val_options, val_decisions, policy)
        report = {
            "schema": "esaw-three-source-holdout-gate-training-v1",
            "train_records": len(train_keys),
            "train_candidate_options": len(train_options),
            "train_actionable_options": int(actionable_train.sum()),
            "validation_records": len(val_keys),
            "validation_best": {key: value for key, value in best.items() if key != "rank"},
            "validation_model_candidates": candidate_reports,
            "validation_ensemble_candidates": ensemble_reports,
            "validation_locked_model": metrics(val_keys, val_options, selected),
            "train_baselines": baseline_report(train_keys, train_groups, train_options),
            "validation_baselines": baseline_report(val_keys, val_groups, val_options),
        }
        output_keys, output_options, output_groups, output_choices = val_keys, val_options, val_groups, selected
    else:
        keys, options, groups = load_part(args.part, split, audit, top3, top10)
        payload = joblib.load(args.model_output)
        if payload.get("schema") != "esaw-three-source-holdout-gate-v1":
            raise RuntimeError("Modello gate Top-3 non riconosciuto")
        x_meta, x_full, x_tree = transform_feature_spaces(
            options, payload["dict_vectorizer"], payload["text_vectorizer"], payload["tree_vectorizer"]
        )
        spaces = {"text_meta": x_full, "tree_dense": x_tree, "meta": x_meta}
        decisions = score_components(payload["components"], spaces)
        selected = decode_policy(keys, groups, options, decisions, payload["policy"])
        report = {
            "schema": "esaw-three-source-holdout-gate-locked-evaluation-v1",
            "part": args.part,
            "records": len(keys),
            "locked_policy": payload["policy"],
            "gate": metrics(keys, options, selected),
            "baselines": baseline_report(keys, groups, options),
        }
        output_keys, output_options, output_groups, output_choices = keys, options, groups, selected

    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_predictions(Path(args.predictions_output), output_keys, output_groups, output_options, output_choices)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
