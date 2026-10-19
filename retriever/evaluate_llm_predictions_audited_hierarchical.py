from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping


TARGETS = [
    "COD_LOC",
    "COD_LAV",
    "COD_ATV",
    "COD_MAT_ATV",
    "COD_DEV",
    "COD_MAT_DEV",
    "COD_CON",
    "COD_MAT_CON",
]

FIELD_TAXONOMY = {
    "COD_LOC": "tipo_luogo",
    "COD_LAV": "tipo_lavoro",
    "COD_ATV": "attivita_fisica_specifica",
    "COD_MAT_ATV": "agente_materiale",
    "COD_DEV": "deviazione",
    "COD_MAT_DEV": "agente_materiale",
    "COD_CON": "contatto_modalita_lesione",
    "COD_MAT_CON": "agente_materiale",
}


from canonical_audit import normalize_code, load_audited_gold


def parent_code(taxonomy: Dict[str, Dict[str, str]], field: str, code: str | None) -> str | None:
    code = normalize_code(field, code)
    entries = taxonomy[FIELD_TAXONOMY[field]]
    if not code:
        return None
    if FIELD_TAXONOMY[field] == "agente_materiale":
        parent = f"{code.split('.')[0]}.00"
        return parent if parent in entries and parent != code else None
    if len(code) == 3:
        parent = f"{code[:2]}0"
    elif len(code) == 2:
        parent = f"{code[0]}0"
    else:
        return None
    return parent if parent in entries and parent != code else None


def ancestors(taxonomy: Dict[str, Dict[str, str]], field: str, code: str | None) -> List[str]:
    out: List[str] = []
    current = parent_code(taxonomy, field, code)
    while current:
        out.append(current)
        current = parent_code(taxonomy, field, current)
    return out


def distance(taxonomy: Dict[str, Dict[str, str]], field: str, pred: str, gold: str) -> int:
    pred = normalize_code(field, pred)
    gold = normalize_code(field, gold)
    if pred == gold:
        return 0
    pred_path = [pred] + ancestors(taxonomy, field, pred)
    gold_path = [gold] + ancestors(taxonomy, field, gold)
    for i, code in enumerate(pred_path):
        if code in gold_path:
            return i + gold_path.index(code)
    return len(pred_path) + len(gold_path)


def is_relaxed(taxonomy: Dict[str, Dict[str, str]], field: str, pred: str, gold: str) -> bool:
    pred = normalize_code(field, pred)
    gold = normalize_code(field, gold)
    return pred == gold or pred in ancestors(taxonomy, field, gold) or gold in ancestors(taxonomy, field, pred)




def row_prediction_code(row: Mapping[str, Any], prediction_key: str, field: str) -> str:
    predictions = row.get(prediction_key, {})
    if isinstance(predictions, Mapping):
        item = predictions.get(field, {})
        if isinstance(item, Mapping):
            return normalize_code(field, item.get("code"))
    return ""


def candidate_codes(row: Mapping[str, Any], field: str, limit: int) -> List[str]:
    candidates = row.get("dual_candidates", {})
    if not isinstance(candidates, Mapping):
        return []
    values = candidates.get(field, [])
    if not isinstance(values, list):
        return []
    return [normalize_code(field, item.get("code")) for item in values[:limit] if isinstance(item, Mapping)]


def evaluate(
    rows: Iterable[Mapping[str, Any]],
    audited_gold: Mapping[str, Mapping[str, str]],
    taxonomy: Dict[str, Dict[str, str]],
    prediction_key: str,
) -> Dict[str, Any]:
    field_counts: Dict[str, Counter] = defaultdict(Counter)
    record_values: Dict[str, Dict[str, Dict[str, bool]]] = defaultdict(dict)
    total_records = 0
    skipped_records = 0

    for row in rows:
        record_key = str(row.get("record_key") or "")
        golds = audited_gold.get(record_key)
        if not golds:
            skipped_records += 1
            continue
        total_records += 1
        for field in TARGETS:
            gold = normalize_code(field, golds[field])
            pred = row_prediction_code(row, prediction_key, field)
            candidates5 = candidate_codes(row, field, 5)
            d = distance(taxonomy, field, pred, gold)
            field_counts[field]["n"] += 1
            field_counts[field]["top1"] += int(pred == gold)
            for k in range(1, 6):
                field_counts[field][f"candidate_recall_top{k}"] += int(gold in candidates5[:k])
            field_counts[field]["hier_relaxed_top1"] += int(is_relaxed(taxonomy, field, pred, gold))
            field_counts[field]["dist_le_1_top1"] += int(d <= 1)
            field_counts[field]["dist_le_2_top1"] += int(d <= 2)
            field_counts[field]["dist_sum"] += d
            field_counts[field]["hsim_sum"] += 1 / (1 + d)
            record_values[record_key][field] = {
                "top1": pred == gold,
                "hier_relaxed": is_relaxed(taxonomy, field, pred, gold),
                "dist_le_2": d <= 2,
            }

    record_counts: Dict[str, Counter] = defaultdict(Counter)
    for fields in record_values.values():
        if all(field in fields for field in TARGETS):
            record_counts["n"]["value"] += 1
            for metric in ("top1", "hier_relaxed", "dist_le_2"):
                count = sum(int(fields[field][metric]) for field in TARGETS)
                record_counts[metric]["exact8"] += int(count == 8)
                record_counts[metric]["ge7"] += int(count >= 7)
                record_counts[metric]["ge6"] += int(count >= 6)

    fields_out = {}
    for field in TARGETS:
        counts = field_counts[field]
        n = max(counts["n"], 1)
        fields_out[field] = {
            "records": counts["n"],
            "top1": counts["top1"] / n,
            **{f"candidate_recall_top{k}": counts[f"candidate_recall_top{k}"] / n for k in range(1, 6)},
            "hier_relaxed_top1": counts["hier_relaxed_top1"] / n,
            "dist_le_1_top1": counts["dist_le_1_top1"] / n,
            "dist_le_2_top1": counts["dist_le_2_top1"] / n,
            "avg_hier_distance_top1": counts["dist_sum"] / n,
            "hier_similarity_top1": counts["hsim_sum"] / n,
        }
    macro_keys = [
        "top1",
        "candidate_recall_top1",
        "candidate_recall_top2",
        "candidate_recall_top3",
        "candidate_recall_top4",
        "candidate_recall_top5",
        "hier_relaxed_top1",
        "dist_le_1_top1",
        "dist_le_2_top1",
        "avg_hier_distance_top1",
        "hier_similarity_top1",
    ]
    record_n = max(record_counts["n"]["value"], 1)
    return {
        "prediction_key": prediction_key,
        "records_seen": total_records,
        "records_skipped_no_valid_audit": skipped_records,
        "complete_records": record_counts["n"]["value"],
        "mean": {key: mean(fields_out[field][key] for field in TARGETS) for key in macro_keys},
        "record": {
            metric: {
                "exact8": record_counts[metric]["exact8"] / record_n,
                "ge7": record_counts[metric]["ge7"] / record_n,
                "ge6": record_counts[metric]["ge6"] / record_n,
            }
            for metric in ("top1", "hier_relaxed", "dist_le_2")
        },
        "fields": fields_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-key", default="llm_top1", choices=["llm_top1", "dual_top1"])
    args = parser.parse_args()

    with Path(args.predictions).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    audited_gold = load_audited_gold(Path(args.audit_jsonl), taxonomy)
    result = evaluate(rows, audited_gold, taxonomy, args.prediction_key)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
