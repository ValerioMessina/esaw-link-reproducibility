from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FIELDS = [
    "COD_LOC", "COD_LAV", "COD_ATV", "COD_MAT_ATV",
    "COD_DEV", "COD_MAT_DEV", "COD_CON", "COD_MAT_CON",
]
EXPECTED_HASHES = {
    "split": "0298c84f77ccb91958315fb242ae17c21ffcc61314021c9f3cb405c07298d3fb",
    "audit": "97cbfe6342750bd4dc9d0695722312ccf44d1f51065e0799f77fa7e37ef157ed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
        for name in ("audited_fields", "audits", "fields_to_audit", "field_audits"):
            values = audit.get(name)
            if isinstance(values, list):
                for item in values:
                    if item.get("field") and item.get("audited_code"):
                        labels[str(item["field"])] = str(item["audited_code"])
        for field in FIELDS:
            item = audit.get(field)
            if isinstance(item, Mapping) and item.get("audited_code"):
                labels[field] = str(item["audited_code"])
        if key and all(field in labels for field in FIELDS):
            output[key] = labels
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica indipendente del goal exact match 8/8 >40%.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--validation-summary", required=True)
    parser.add_argument("--gate-report", required=True)
    parser.add_argument("--gate-model", required=True)
    parser.add_argument("--top3", nargs="+", required=True)
    parser.add_argument("--top10", required=True)
    parser.add_argument("--cross-predictions", default="")
    parser.add_argument("--cross-calibration", default="")
    args = parser.parse_args()

    split_path = Path(args.split_file)
    audit_path = Path(args.audit_jsonl)
    artifact_hashes = {"split": sha256(split_path), "audit": sha256(audit_path)}
    split = json.loads(split_path.read_text(encoding="utf-8"))["split"]
    train, validation, test = map(set, (split["train"], split["validation"], split["test"]))
    split_counts = {"train": len(train), "validation": len(validation), "test": len(test)}
    expected_split_counts = {"train": 3250, "validation": 696, "test": 697}
    audit = load_audit(audit_path)
    rows = read_jsonl(Path(args.predictions))
    by_key = {str(row.get("record_key") or ""): row for row in rows}
    duplicate_rows = len(rows) - len(by_key)
    test_keys_match = set(by_key) == test
    complete = all(
        isinstance(row.get("labels"), Mapping) and all(str(row["labels"].get(field) or "") for field in FIELDS)
        for row in by_key.values()
    )
    exact = ge7 = ge6 = field_correct = 0
    by_field = {field: 0 for field in FIELDS}
    for key in sorted(test & set(by_key)):
        labels = by_key[key]["labels"]
        correct = 0
        for field in FIELDS:
            value = int(str(labels[field]) == audit[key][field])
            correct += value
            field_correct += value
            by_field[field] += value
        exact += correct == 8
        ge7 += correct >= 7
        ge6 += correct >= 6
    n = len(test)
    exact_rate = exact / max(n, 1)
    validation_summary = json.loads(Path(args.validation_summary).read_text(encoding="utf-8"))
    gate_report = json.loads(Path(args.gate_report).read_text(encoding="utf-8"))
    validation_exact8 = float(validation_summary.get("validation_gate_exact8", 0.0))
    selected_gate_variant = str(validation_summary.get("selected_gate", {}).get("selected_variant") or "")
    reported_test_exact8 = float(gate_report.get("gate", {}).get("exact8", -1.0))
    lineage_paths = {
        "validation_summary": Path(args.validation_summary),
        "gate_report": Path(args.gate_report),
        "gate_model": Path(args.gate_model),
        "top10": Path(args.top10),
    }
    if args.cross_predictions:
        lineage_paths["cross_predictions"] = Path(args.cross_predictions)
    if args.cross_calibration:
        lineage_paths["cross_calibration"] = Path(args.cross_calibration)
    lineage_paths.update({f"top3_shard_{index:02d}": Path(path) for index, path in enumerate(args.top3)})
    lineage_files_exist = all(path.is_file() and path.stat().st_size > 0 for path in lineage_paths.values())
    lineage_hashes = {
        name: sha256(path) for name, path in lineage_paths.items()
        if path.is_file() and path.stat().st_size > 0
    }
    lineage_valid = bool(
        validation_summary.get("ready") is True
        and validation_summary.get("test_evaluated") is False
        and validation_exact8 >= args.threshold
        and selected_gate_variant in {"four_source_oof", "three_source_holdout"}
        and (
            (
                selected_gate_variant == "four_source_oof"
                and gate_report.get("schema") == "esaw-four-source-exact8-gate-locked-evaluation-v1"
            )
            or (
                selected_gate_variant == "three_source_holdout"
                and gate_report.get("schema") == "esaw-three-source-holdout-gate-locked-evaluation-v1"
            )
        )
        and (
            selected_gate_variant != "four_source_oof"
            or (bool(args.cross_predictions) and bool(args.cross_calibration))
        )
        and gate_report.get("part") == "test"
        and int(gate_report.get("records") or 0) == 697
        and abs(reported_test_exact8 - exact_rate) <= 1e-12
        and len(args.top3) == 16
        and lineage_files_exist
        and len(lineage_hashes) == len(lineage_paths)
    )
    proof = {
        "schema": "esaw-exact8-goal-proof-v1",
        "threshold_strictly_greater_than": args.threshold,
        "artifact_hashes": artifact_hashes,
        "expected_artifact_hashes": EXPECTED_HASHES,
        "artifact_hashes_match": artifact_hashes == EXPECTED_HASHES,
        "split_counts": split_counts,
        "expected_split_counts": expected_split_counts,
        "split_counts_match_expected": split_counts == expected_split_counts,
        "split_disjoint": not (train & validation or train & test or validation & test),
        "prediction_rows": len(rows),
        "duplicate_prediction_rows": duplicate_rows,
        "test_keys_match_exactly": test_keys_match,
        "complete_8_fields": complete,
        "audited_test_records": n,
        "validation_gate_exact8": validation_exact8,
        "selected_gate_variant": selected_gate_variant,
        "validation_gate_threshold_passed": validation_exact8 >= args.threshold,
        "reported_test_gate_exact8": reported_test_exact8,
        "reported_test_gate_exact8_matches_recomputed": abs(reported_test_exact8 - exact_rate) <= 1e-12,
        "lineage_valid": lineage_valid,
        "lineage_hashes": lineage_hashes,
        "field_accuracy": field_correct / max(n * 8, 1),
        "exact8": exact_rate,
        "exact8_count": exact,
        "ge7": ge7 / max(n, 1),
        "ge6": ge6 / max(n, 1),
        "by_field": {field: by_field[field] / max(n, 1) for field in FIELDS},
    }
    proof["passed"] = bool(
        proof["artifact_hashes_match"] and proof["lineage_valid"]
        and proof["split_counts_match_expected"] and proof["split_disjoint"]
        and test_keys_match and complete and duplicate_rows == 0
        and len(rows) == n and exact_rate > args.threshold
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof, ensure_ascii=False, indent=2))
    raise SystemExit(0 if proof["passed"] else 2)


if __name__ == "__main__":
    main()
