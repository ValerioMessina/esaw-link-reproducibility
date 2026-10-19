from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


FIELDS = [
    "COD_LOC",
    "COD_LAV",
    "COD_ATV",
    "COD_MAT_ATV",
    "COD_DEV",
    "COD_MAT_DEV",
    "COD_CON",
    "COD_MAT_CON",
]
WORD_RE = re.compile(r"[0-9a-zà-ÿ]+", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_audited_gold(path: Path) -> dict[str, dict[str, str]]:
    audited: dict[str, dict[str, str]] = {}
    for row in read_jsonl(path):
        if row.get("validation_errors"):
            continue
        audit = row.get("audit")
        if not isinstance(audit, Mapping):
            continue
        record_key = str(
            row.get("record_key")
            or row.get("request", {}).get("record_key")
            or audit.get("record_key")
            or ""
        ).strip()
        labels: dict[str, str] = {}
        for list_key in ("audited_fields", "audits", "fields_to_audit", "field_audits"):
            values = audit.get(list_key)
            if isinstance(values, list):
                for item in values:
                    field = str(item.get("field") or "").strip()
                    code = str(item.get("audited_code") or "").strip()
                    if field and code:
                        labels[field] = code
        for field in FIELDS:
            item = audit.get(field)
            if isinstance(item, Mapping):
                code = str(item.get("audited_code") or "").strip()
                if code:
                    labels[field] = code
        if record_key and all(field in labels for field in FIELDS):
            audited[record_key] = labels
    return audited


def make_split(keys: Sequence[str], seed: int) -> dict[str, list[str]]:
    ranked = sorted(
        keys,
        key=lambda key: hashlib.sha256(f"meta-gate-v1:{seed}:{key}".encode()).digest(),
    )
    n = len(ranked)
    train_end = int(n * 0.70)
    validation_end = train_end + int(n * 0.15)
    return {
        "train": ranked[:train_end],
        "validation": ranked[train_end:validation_end],
        "test": ranked[validation_end:],
    }


def load_or_create_split(path: Path, audit_keys: Sequence[str], seed: int) -> dict[str, list[str]]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        split = payload["split"]
        assigned = [key for part in ("train", "validation", "test") for key in split[part]]
        if len(assigned) != len(set(assigned)):
            raise ValueError("Lo split contiene record duplicati")
        if set(assigned) != set(audit_keys):
            raise ValueError("Lo split esistente non corrisponde alle audit valide")
        return split
    split = make_split(audit_keys, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "esaw-meta-gate-record-split-v1",
                "seed": seed,
                "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
                "counts": {name: len(values) for name, values in split.items()},
                "split": split,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return split


def rows_by_key(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            key = str(row.get("record_key") or "")
            if key:
                rows[key] = row
    return rows


def code_of(row: Mapping[str, Any], prediction_key: str, field: str) -> str:
    item = row.get(prediction_key, {}).get(field, {})
    return str(item.get("code") or "") if isinstance(item, Mapping) else ""


def safe_token(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", value.lower()).strip("_") or "empty"


def code_depth(code: str) -> int:
    if "." in code:
        return sum(part != "00" for part in code.split("."))
    return int(not code.endswith("0"))


@dataclass
class Option:
    record_key: str
    field: str
    code: str
    gold: str
    text: str
    label_text: str
    features: dict[str, Any]
    lexical: str


def build_options(
    keys: Sequence[str],
    gold: Mapping[str, Mapping[str, str]],
    top3: Mapping[str, Mapping[str, Any]],
    top10: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Option], dict[tuple[str, str], list[int]]]:
    options: list[Option] = []
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for key in keys:
        row3 = top3[key]
        row10 = top10[key]
        text = str(row10.get("text") or row3.get("text") or "")
        record_selections = []
        for field in FIELDS:
            record_selections.append(
                (
                    code_of(row10, "dual_top1", field),
                    code_of(row3, "llm_top1", field),
                    code_of(row10, "llm_top1", field),
                )
            )
        record_all_agree = sum(len(set(values)) == 1 for values in record_selections)
        record_two_agree = sum(len(set(values)) == 2 for values in record_selections)
        record_all_diff = 8 - record_all_agree - record_two_agree

        for field, (dual, gemma3, gemma10) in zip(FIELDS, record_selections):
            selected = [dual, gemma3, gemma10]
            candidates = row10.get("dual_candidates", {}).get(field, [])
            by_code = {str(item.get("code") or ""): item for item in candidates if isinstance(item, Mapping)}
            top_scores = [float(item.get("score") or 0.0) for item in candidates[:3]]
            while len(top_scores) < 3:
                top_scores.append(0.0)
            margin12 = top_scores[0] - top_scores[1]
            margin13 = top_scores[0] - top_scores[2]
            pattern = "all_agree" if len(set(selected)) == 1 else ("two_agree" if len(set(selected)) == 2 else "all_diff")
            counts = Counter(selected)
            for code in dict.fromkeys(selected):
                if not code:
                    continue
                item = by_code.get(code, {})
                rank = int(item.get("rank_dual_encoder") or 99)
                score = float(item.get("score") or 0.0)
                label_text = str(item.get("label") or "")
                parent = str(item.get("parent_code") or "")
                support = {
                    "dual": int(code == dual),
                    "top3": int(code == gemma3),
                    "top10": int(code == gemma10),
                }
                support_name = "_".join(name for name, value in support.items() if value)
                text_words = WORD_RE.findall(text.lower())[:320]
                label_words = set(WORD_RE.findall(label_text.lower()))
                overlap = len(set(text_words) & label_words)
                field_token = safe_token(field)
                code_token = safe_token(code)
                crossed = [f"x_{field_token}_{code_token}_{word}" for word in text_words if len(word) >= 3]
                crossed += [
                    f"xb_{field_token}_{code_token}_{left}_{right}"
                    for left, right in zip(text_words, text_words[1:])
                    if len(left) >= 3 and len(right) >= 3
                ]
                lexical = " ".join(
                    [f"code_{field_token}_{code_token}"]
                    + [f"label_{field_token}_{code_token}_{word}" for word in label_words]
                    + crossed
                )
                features: dict[str, Any] = {
                    "field": field,
                    "field_code": f"{field}:{code}",
                    "field_parent": f"{field}:{parent or 'ROOT'}",
                    "support": support_name,
                    "pattern": pattern,
                    "rank": rank,
                    "rank_reciprocal": 1.0 / rank,
                    "dual_score": score,
                    "dual_top1_score": top_scores[0],
                    "dual_margin12": margin12,
                    "dual_margin13": margin13,
                    "score_gap_from_top1": top_scores[0] - score,
                    "votes": counts[code],
                    "support_dual": support["dual"],
                    "support_top3": support["top3"],
                    "support_top10": support["top10"],
                    "is_dual_top1": int(rank == 1),
                    "is_zero_code": int(code.startswith("00")),
                    "is_parent_code": int(code_depth(code) == 0),
                    "code_depth": code_depth(code),
                    "label_overlap": overlap,
                    "text_words": len(text_words),
                    "record_all_agree": record_all_agree,
                    "record_two_agree": record_two_agree,
                    "record_all_diff": record_all_diff,
                }
                index = len(options)
                options.append(
                    Option(
                        record_key=key,
                        field=field,
                        code=code,
                        gold=gold[key][field],
                        text=text,
                        label_text=label_text,
                        features=features,
                        lexical=lexical,
                    )
                )
                groups[(key, field)].append(index)
    return options, groups


def metrics_for_choices(
    keys: Sequence[str],
    groups: Mapping[tuple[str, str], Sequence[int]],
    options: Sequence[Option],
    choices: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    field_correct = Counter()
    exact = ge7 = ge6 = 0
    for key in keys:
        correct = 0
        for field in FIELDS:
            option = options[choices[(key, field)]]
            value = int(option.code == option.gold)
            field_correct[field] += value
            correct += value
        exact += int(correct == 8)
        ge7 += int(correct >= 7)
        ge6 += int(correct >= 6)
    n = max(len(keys), 1)
    return {
        "records": len(keys),
        "field_accuracy": sum(field_correct.values()) / (n * len(FIELDS)),
        "exact8": exact / n,
        "ge7": ge7 / n,
        "ge6": ge6 / n,
        "exact8_count": exact,
        "by_field": {field: field_correct[field] / n for field in FIELDS},
    }


def baseline_choices(
    keys: Sequence[str],
    groups: Mapping[tuple[str, str], Sequence[int]],
    options: Sequence[Option],
    mode: str,
) -> dict[tuple[str, str], int]:
    output: dict[tuple[str, str], int] = {}
    support_key = {"dual": "support_dual", "top3": "support_top3", "top10": "support_top10"}
    for key in keys:
        for field in FIELDS:
            indices = groups[(key, field)]
            if mode == "oracle":
                output[(key, field)] = next((idx for idx in indices if options[idx].code == options[idx].gold), indices[0])
            elif mode == "majority":
                output[(key, field)] = max(indices, key=lambda idx: (options[idx].features["votes"], options[idx].features["support_dual"]))
            else:
                output[(key, field)] = next(idx for idx in indices if options[idx].features[support_key[mode]])
    return output


def decode_gate(
    keys: Sequence[str],
    groups: Mapping[tuple[str, str], Sequence[int]],
    options: Sequence[Option],
    decisions: np.ndarray,
    vote_weight: float,
    threshold: float,
    max_changes: int,
) -> dict[tuple[str, str], int]:
    output: dict[tuple[str, str], int] = {}
    for key in keys:
        proposed = []
        for field in FIELDS:
            indices = list(groups[(key, field)])
            majority = max(indices, key=lambda idx: (options[idx].features["votes"], options[idx].features["support_dual"]))
            adjusted = {idx: float(decisions[idx]) + vote_weight * float(options[idx].features["votes"]) for idx in indices}
            best = max(indices, key=lambda idx: adjusted[idx])
            margin = adjusted[best] - adjusted[majority]
            output[(key, field)] = majority
            if best != majority and margin >= threshold:
                proposed.append((margin, field, best))
        proposed.sort(reverse=True)
        for _margin, field, best in proposed[:max_changes]:
            output[(key, field)] = best
    return output


def subset_indices(options: Sequence[Option], key_set: set[str]) -> np.ndarray:
    return np.fromiter((index for index, option in enumerate(options) if option.record_key in key_set), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-gate auditato dual/Gemma Top-3/Gemma Top-10 ottimizzato per exact8.")
    parser.add_argument("--top3", required=True, nargs="+")
    parser.add_argument("--top10", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument(
        "--eligible-records-jsonl",
        help="JSONL opzionale che definisce il perimetro ufficiale dei record valutabili.",
    )
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--prepare-split-only", action="store_true")
    parser.add_argument("--max-features", type=int, default=350_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audited = load_audited_gold(Path(args.audit_jsonl))
    eligible_keys = set(audited)
    if args.eligible_records_jsonl:
        eligible_keys &= {
            str(row.get("record_key") or "")
            for row in read_jsonl(Path(args.eligible_records_jsonl))
            if row.get("record_key")
        }
    split = load_or_create_split(Path(args.split_file), sorted(eligible_keys), args.seed)
    if args.prepare_split_only:
        print(json.dumps({"audit_records": len(audited), "split_counts": {k: len(v) for k, v in split.items()}}, indent=2))
        return

    top3 = rows_by_key([Path(value) for value in args.top3])
    top10 = rows_by_key([Path(args.top10)])
    available = set(audited) & set(top3) & set(top10)
    train_keys = [key for key in split["train"] if key in available]
    validation_keys = [key for key in split["validation"] if key in available]
    test_keys = [key for key in split["test"] if key in available]
    if not train_keys or not validation_keys:
        raise SystemExit("Record comuni insufficienti in train/validation")
    used_keys = train_keys + validation_keys + (test_keys if args.evaluate_test else [])
    options, groups = build_options(used_keys, audited, top3, top10)

    lex = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\S+\b",
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    vec = DictVectorizer(sparse=True, dtype=np.float32)
    train_idx = subset_indices(options, set(train_keys))
    validation_idx = subset_indices(options, set(validation_keys))
    lex_train = lex.fit_transform([options[index].lexical for index in train_idx])
    dict_train = vec.fit_transform([options[index].features for index in train_idx])
    x_train = hstack([lex_train, dict_train], format="csr")
    y_train = np.asarray([options[index].code == options[index].gold for index in train_idx], dtype=np.int8)

    all_lex = lex.transform([option.lexical for option in options])
    all_dict = vec.transform([option.features for option in options])
    x_all = hstack([all_lex, all_dict], format="csr")

    validation_baselines = {
        mode: metrics_for_choices(validation_keys, groups, options, baseline_choices(validation_keys, groups, options, mode))
        for mode in ("dual", "top3", "top10", "majority", "oracle")
    }
    best: dict[str, Any] | None = None
    best_model: SGDClassifier | None = None
    best_decisions: np.ndarray | None = None
    for alpha in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4):
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=1500,
            tol=1e-4,
            class_weight="balanced",
            average=True,
            random_state=args.seed,
        )
        model.fit(x_train, y_train)
        decisions = model.decision_function(x_all)
        for vote_weight in (-0.5, 0.0, 0.5, 1.0, 1.5):
            for threshold in (0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
                for max_changes in (1, 2, 3, 4, 8):
                    choices = decode_gate(validation_keys, groups, options, decisions, vote_weight, threshold, max_changes)
                    metrics = metrics_for_choices(validation_keys, groups, options, choices)
                    candidate = {
                        "alpha": alpha,
                        "vote_weight": vote_weight,
                        "threshold": threshold,
                        "max_changes": max_changes,
                        "metrics": metrics,
                    }
                    rank = (metrics["exact8"], metrics["ge7"], metrics["field_accuracy"], -max_changes, -threshold)
                    if best is None or rank > best["rank"]:
                        best = {**candidate, "rank": rank}
                        best_model = model
                        best_decisions = decisions.copy()

    assert best is not None and best_model is not None and best_decisions is not None
    report: dict[str, Any] = {
        "schema": "esaw-meta-gate-exact8-v1",
        "available_records": len(available),
        "split_available": {
            "train": len(train_keys),
            "validation": len(validation_keys),
            "test": len(test_keys),
        },
        "test_was_evaluated": bool(args.evaluate_test),
        "feature_counts": {"lexical": len(lex.vocabulary_), "structured": len(vec.feature_names_)},
        "training_examples": int(len(train_idx)),
        "training_positive_rate": float(y_train.mean()),
        "validation_baselines": validation_baselines,
        "best_validation_policy": {key: value for key, value in best.items() if key != "rank"},
    }

    final_parts = ["validation"] + (["test"] if args.evaluate_test else [])
    predictions = []
    for part in final_parts:
        keys = validation_keys if part == "validation" else test_keys
        choices = decode_gate(
            keys,
            groups,
            options,
            best_decisions,
            best["vote_weight"],
            best["threshold"],
            best["max_changes"],
        )
        report[f"{part}_gate"] = metrics_for_choices(keys, groups, options, choices)
        report[f"{part}_baselines"] = {
            mode: metrics_for_choices(keys, groups, options, baseline_choices(keys, groups, options, mode))
            for mode in ("dual", "top3", "top10", "majority", "oracle")
        }
        for key in keys:
            predictions.append(
                {
                    "split": part,
                    "record_key": key,
                    "labels": {field: options[choices[(key, field)]].code for field in FIELDS},
                }
            )

    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    joblib.dump(
        {
            "schema": "esaw-meta-gate-exact8-v1",
            "model": best_model,
            "lexical_vectorizer": lex,
            "dict_vectorizer": vec,
            "policy": {key: value for key, value in best.items() if key not in ("rank", "metrics")},
        },
        output_dir / "model.joblib",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
