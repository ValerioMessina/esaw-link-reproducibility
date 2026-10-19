from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


FIELDS = [
    "COD_LOC", "COD_LAV", "COD_ATV", "COD_MAT_ATV",
    "COD_DEV", "COD_MAT_DEV", "COD_CON", "COD_MAT_CON",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rows_by_key(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    output = {}
    for path in paths:
        for row in read_jsonl(path):
            key = str(row.get("record_key") or "")
            if key:
                output[key] = row
    return output


def load_audit(path: Path) -> dict[str, dict[str, str]]:
    output = {}
    for row in read_jsonl(path):
        if row.get("validation_errors"):
            continue
        audit = row.get("audit")
        if not isinstance(audit, Mapping):
            continue
        key = str(row.get("record_key") or row.get("request", {}).get("record_key") or audit.get("record_key") or "")
        labels = {}
        for list_key in ("audited_fields", "audits", "fields_to_audit", "field_audits"):
            items = audit.get(list_key)
            if isinstance(items, list):
                for item in items:
                    if item.get("field") and item.get("audited_code"):
                        labels[str(item["field"])] = str(item["audited_code"])
        for field in FIELDS:
            item = audit.get(field)
            if isinstance(item, Mapping) and item.get("audited_code"):
                labels[field] = str(item["audited_code"])
        if key and all(field in labels for field in FIELDS):
            output[key] = labels
    return output


def code_of(row: Mapping[str, Any], prediction_key: str, field: str) -> str:
    item = row.get(prediction_key, {}).get(field, {})
    return str(item.get("code") or "") if isinstance(item, Mapping) else ""


def normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array - array.mean()) / max(float(array.max() - array.min()), 1e-6)


def cross_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["record_key"]), str(row["field"])): row for row in read_jsonl(path)}


@dataclass
class Option:
    key: str
    field: str
    code: str
    gold: str
    features: dict[str, Any]


def build_options(
    keys: Sequence[str],
    audit: Mapping[str, Mapping[str, str]],
    top3: Mapping[str, Mapping[str, Any]],
    top10: Mapping[str, Mapping[str, Any]],
    cross: Mapping[tuple[str, str], Mapping[str, Any]],
    cross_weights: Mapping[str, float],
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
            cross_row = cross[(key, field)]
            cross_candidates = cross_row["rerank_top_candidates"]
            cross_scores = normalize([float(item["cross_score"]) for item in cross_candidates])
            retriever_scores = normalize([float(item["retriever_score"]) for item in cross_candidates])
            fused_scores = cross_scores + float(cross_weights[field]) * retriever_scores
            cross_order = np.argsort(-fused_scores)
            cross_best_index = int(cross_order[0])
            cross_best = str(cross_candidates[cross_best_index]["code"])
            cross_margin = float(fused_scores[cross_best_index] - fused_scores[int(cross_order[1])])
            cross_map = {
                str(item["code"]): {
                    "cross_norm": float(cross_scores[index]),
                    "retriever_norm": float(retriever_scores[index]),
                    "fusion_score": float(fused_scores[index]),
                    "cross_rank": int(np.where(cross_order == index)[0][0]) + 1,
                }
                for index, item in enumerate(cross_candidates)
            }

            dual = code_of(row10, "dual_top1", field)
            gemma3 = code_of(row3, "llm_top1", field)
            gemma10 = code_of(row10, "llm_top1", field)
            sources = {"dual": dual, "top3": gemma3, "top10": gemma10, "cross": cross_best}
            votes = Counter(sources.values())
            pattern = f"unique{len(votes)}"
            dual_candidates = row10.get("dual_candidates", {}).get(field, [])
            dual_map = {str(item.get("code") or ""): item for item in dual_candidates if isinstance(item, Mapping)}
            top_scores = [float(item.get("score") or 0.0) for item in dual_candidates[:3]]
            while len(top_scores) < 3:
                top_scores.append(0.0)
            for code in dict.fromkeys(sources.values()):
                if not code:
                    continue
                dual_item = dual_map.get(code, {})
                cross_item = cross_map.get(code, {})
                support = {name: int(value == code) for name, value in sources.items()}
                feature = {
                    "field": field,
                    "field_code": f"{field}:{code}",
                    "pattern": pattern,
                    "support_pattern": "_".join(name for name, value in support.items() if value),
                    "votes": votes[code],
                    **{f"support_{name}": value for name, value in support.items()},
                    "dual_rank": int(dual_item.get("rank_dual_encoder") or 99),
                    "dual_score": float(dual_item.get("score") or 0.0),
                    "dual_margin12": top_scores[0] - top_scores[1],
                    "dual_margin13": top_scores[0] - top_scores[2],
                    "cross_norm": float(cross_item.get("cross_norm", -2.0)),
                    "cross_retriever_norm": float(cross_item.get("retriever_norm", -2.0)),
                    "cross_fusion_score": float(cross_item.get("fusion_score", -4.0)),
                    "cross_rank": int(cross_item.get("cross_rank", 99)),
                    "cross_margin": cross_margin if support["cross"] else 0.0,
                    "cross_margin_supported": cross_margin * support["cross"],
                    **record_features,
                }
                index = len(options)
                options.append(Option(key, field, code, audit[key][field], feature))
                groups[(key, field)].append(index)
    return options, groups


def majority_index(indices: Sequence[int], options: Sequence[Option]) -> int:
    return max(
        indices,
        key=lambda index: (
            options[index].features["support_dual"] + options[index].features["support_top3"] + options[index].features["support_top10"],
            options[index].features["support_dual"],
        ),
    )


def decode(
    keys: Sequence[str], groups: Mapping[tuple[str, str], Sequence[int]], options: Sequence[Option],
    decisions: np.ndarray, threshold: float, max_changes: int,
) -> dict[tuple[str, str], int]:
    output = {}
    for key in keys:
        changes = []
        for field in FIELDS:
            indices = groups[(key, field)]
            baseline = majority_index(indices, options)
            best = max(indices, key=lambda index: float(decisions[index]))
            output[(key, field)] = baseline
            margin = float(decisions[best] - decisions[baseline])
            if best != baseline and margin >= threshold:
                changes.append((margin, field, best))
        changes.sort(reverse=True)
        for _margin, field, best in changes[:max_changes]:
            output[(key, field)] = best
    return output


def decode_single_source(
    keys: Sequence[str], groups: Mapping[tuple[str, str], Sequence[int]], options: Sequence[Option],
    decisions: np.ndarray,
) -> dict[tuple[str, str], int]:
    output = {}
    for key in keys:
        source_choices = {}
        for source in ("dual", "top3", "top10", "cross"):
            indices = []
            for field in FIELDS:
                index = next(
                    item for item in groups[(key, field)]
                    if options[item].features[f"support_{source}"]
                )
                indices.append(index)
            source_choices[source] = (sum(float(decisions[index]) for index in indices), indices)
        _source, (_score, selected) = max(
            source_choices.items(),
            key=lambda item: (item[1][0], item[0] == "dual"),
        )
        for field, index in zip(FIELDS, selected):
            output[(key, field)] = index
    return output


def decode_policy(keys, groups, options, decisions, policy):
    if policy.get("decoder") == "single_source":
        return decode_single_source(keys, groups, options, decisions)
    return decode(
        keys, groups, options, decisions,
        float(policy["threshold"]), int(policy["max_changes"]),
    )


def fixed_source_choices(keys, groups, options, source: str):
    output = {}
    for key in keys:
        for field in FIELDS:
            indices = groups[(key, field)]
            if source == "majority":
                output[(key, field)] = majority_index(indices, options)
            elif source == "oracle":
                output[(key, field)] = next((index for index in indices if options[index].code == options[index].gold), indices[0])
            else:
                output[(key, field)] = next(index for index in indices if options[index].features[f"support_{source}"])
    return output


def metrics(keys, options, choices):
    exact = ge7 = ge6 = fields = 0
    by_field = Counter()
    for key in keys:
        count = 0
        for field in FIELDS:
            option = options[choices[(key, field)]]
            correct = int(option.code == option.gold)
            count += correct
            fields += correct
            by_field[field] += correct
        exact += count == 8
        ge7 += count >= 7
        ge6 += count >= 6
    n = max(len(keys), 1)
    return {
        "records": len(keys), "field_accuracy": fields / (n * 8), "exact8": exact / n,
        "ge7": ge7 / n, "ge6": ge6 / n, "exact8_count": exact,
        "by_field": {field: by_field[field] / n for field in FIELDS},
    }


def fold_of(key: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha256(f"four-source-oof:{key}".encode()).digest()[:8], "big") % folds


def inputs(args):
    audit = load_audit(Path(args.audit_jsonl))
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))["split"]
    top3 = rows_by_key([Path(value) for value in args.top3])
    top10 = rows_by_key([Path(args.top10)])
    cross = cross_rows(Path(args.cross_predictions))
    calibration = json.loads(Path(args.cross_calibration).read_text(encoding="utf-8"))
    weights = calibration["locked_field_weights"]
    expected_keys = list(split[args.part])
    expected_count = {"validation": 696, "test": 697}[args.part]
    if len(expected_keys) != expected_count or len(set(expected_keys)) != expected_count:
        raise RuntimeError(
            f"Split {args.part} non valido: attesi {expected_count} record unici, "
            f"trovati {len(expected_keys)} righe e {len(set(expected_keys))} chiavi"
        )
    missing = {
        "audit": [key for key in expected_keys if key not in audit],
        "top3": [key for key in expected_keys if key not in top3],
        "top10": [key for key in expected_keys if key not in top10],
        "cross": [
            key for key in expected_keys
            if any((key, field) not in cross for field in FIELDS)
        ],
    }
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        summary = {name: {"count": len(values), "sample": values[:5]} for name, values in missing.items()}
        raise RuntimeError(f"Input incompleti per {args.part}: {json.dumps(summary, ensure_ascii=False)}")
    part_keys = expected_keys
    options, groups = build_options(part_keys, audit, top3, top10, cross, weights)
    return part_keys, options, groups


def model_score(model: Any, x: Any) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=np.float64)
    probabilities = model.predict_proba(x)
    positive = int(np.where(model.classes_ == 1)[0][0])
    return np.asarray(probabilities[:, positive], dtype=np.float64)


def model_candidates() -> list[tuple[str, Any]]:
    candidates = [
        (
            f"logistic_c{c_value:g}",
            LogisticRegression(
                C=c_value, class_weight="balanced", solver="liblinear", max_iter=1000,
            ),
        )
        for c_value in (0.1, 0.3, 1.0, 3.0, 10.0)
    ]
    candidates.extend(
        (
            f"extra_trees_leaf{leaf}",
            ExtraTreesClassifier(
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
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate OOF tra dual, Gemma Top-3, Gemma Top-10 e reranker semantico.")
    parser.add_argument("--mode", choices=["train", "apply"], required=True)
    parser.add_argument("--part", choices=["validation", "test"], required=True)
    parser.add_argument("--top3", nargs="+", required=True)
    parser.add_argument("--top10", required=True)
    parser.add_argument("--cross-predictions", required=True)
    parser.add_argument("--cross-calibration", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--predictions-output", required=True)
    args = parser.parse_args()

    keys, options, groups = inputs(args)
    x_features = [option.features for option in options]
    y = np.asarray([option.code == option.gold for option in options], dtype=np.int8)

    if args.mode == "train":
        if args.part != "validation":
            raise SystemExit("Il training del gate e' consentito solo su validation")
        vectorizer = DictVectorizer(sparse=True, dtype=np.float64)
        x = vectorizer.fit_transform(x_features)
        best = None
        best_model = None
        candidate_reports = []
        for model_name, prototype in model_candidates():
            oof = np.zeros(len(options), dtype=np.float64)
            for fold in range(5):
                train_idx = np.asarray([i for i, option in enumerate(options) if fold_of(option.key, 5) != fold])
                valid_idx = np.asarray([i for i, option in enumerate(options) if fold_of(option.key, 5) == fold])
                model = prototype.__class__(**prototype.get_params())
                model.fit(x[train_idx], y[train_idx])
                oof[valid_idx] = model_score(model, x[valid_idx])
            candidate_best = None
            for threshold in (-1.0, -0.5, -0.2, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.7, 1.0, 1.5, 2.0):
                for max_changes in (0, 1, 2, 3, 4, 5, 6, 8):
                    choices = decode(keys, groups, options, oof, threshold, max_changes)
                    value = metrics(keys, options, choices)
                    rank = (value["exact8"], value["ge7"], value["field_accuracy"], -max_changes, -abs(threshold))
                    if candidate_best is None or rank > candidate_best["rank"]:
                        candidate_best = {
                            "model_name": model_name, "decoder": "field_gate", "threshold": threshold,
                            "max_changes": max_changes, "metrics": value, "rank": rank,
                        }
                    if best is None or rank > best["rank"]:
                        best = {
                            "model_name": model_name, "decoder": "field_gate", "threshold": threshold,
                            "max_changes": max_changes, "metrics": value, "rank": rank,
                        }
            source_choices = decode_single_source(keys, groups, options, oof)
            source_value = metrics(keys, options, source_choices)
            source_rank = (
                source_value["exact8"], source_value["ge7"], source_value["field_accuracy"], -8, 0.0,
            )
            source_result = {
                "model_name": model_name, "decoder": "single_source", "threshold": None,
                "max_changes": None, "metrics": source_value, "rank": source_rank,
            }
            if candidate_best is None or source_rank > candidate_best["rank"]:
                candidate_best = source_result
            if best is None or source_rank > best["rank"]:
                best = source_result
            assert candidate_best is not None
            candidate_reports.append({key: value for key, value in candidate_best.items() if key != "rank"})
            final = prototype.__class__(**prototype.get_params())
            final.fit(x, y)
            if best["model_name"] == model_name:
                best_model = final
        assert best is not None and best_model is not None
        model_payload = {
            "schema": "esaw-four-source-exact8-gate-v1", "vectorizer": vectorizer, "model": best_model,
            "policy": {
                "decoder": best["decoder"], "threshold": best["threshold"],
                "max_changes": best["max_changes"],
            },
        }
        Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_payload, args.model_output)
        decisions = model_score(best_model, x)
        choices = decode_policy(keys, groups, options, decisions, model_payload["policy"])
        report = {
            "schema": "esaw-four-source-exact8-gate-training-v1", "part": args.part,
            "records": len(keys), "oof_best": {key: value for key, value in best.items() if key != "rank"},
            "oof_model_candidates": candidate_reports,
            "in_sample_locked_model": metrics(keys, options, choices),
            "baselines": {source: metrics(keys, options, fixed_source_choices(keys, groups, options, source)) for source in ("dual", "top3", "top10", "cross", "majority", "oracle")},
        }
    else:
        payload = joblib.load(args.model_output)
        x = payload["vectorizer"].transform(x_features)
        decisions = model_score(payload["model"], x)
        choices = decode_policy(keys, groups, options, decisions, payload["policy"])
        report = {
            "schema": "esaw-four-source-exact8-gate-locked-evaluation-v1", "part": args.part,
            "records": len(keys), "locked_policy": payload["policy"], "gate": metrics(keys, options, choices),
            "baselines": {source: metrics(keys, options, fixed_source_choices(keys, groups, options, source)) for source in ("dual", "top3", "top10", "cross", "majority", "oracle")},
        }

    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.predictions_output).open("w", encoding="utf-8") as handle:
        for key in keys:
            handle.write(json.dumps({"record_key": key, "labels": {field: options[choices[(key, field)]].code for field in FIELDS}}, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
