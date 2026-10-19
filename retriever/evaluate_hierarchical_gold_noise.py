from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from esaw_biencoder.data import load_taxonomy, material_taxonomy_code, parent_code_for
from esaw_biencoder.schema import TARGET_FIELDS


FIELDS = [target.output_field for target in TARGET_FIELDS]
FIELD_TO_TAXONOMY = {target.output_field: target.taxonomy for target in TARGET_FIELDS}
CORRECTED_GOLD_COLUMNS = ("corrected_gold_code", "gold_corrected", "corrected_code", "new_gold_code")
CORRECTION_FLAG_COLUMNS = ("use_correction", "accepted", "is_corrected")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(value)).strip()


def to_taxonomy_code(field: str, code: Any) -> str:
    normalized = clean_text(code)
    if FIELD_TO_TAXONOMY[field] == "agente_materiale":
        return material_taxonomy_code(normalized)
    return normalized


def ancestors_for(raw_taxonomy: Mapping[str, Mapping[str, str]], field: str, taxonomy_code: str) -> list[str]:
    taxonomy_name = FIELD_TO_TAXONOMY[field]
    taxonomy = raw_taxonomy[taxonomy_name]
    ancestors: list[str] = []
    current = parent_code_for(taxonomy_name, taxonomy_code, taxonomy)
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        ancestors.append(current)
        current = parent_code_for(taxonomy_name, current, taxonomy)
    return ancestors


def path_for(raw_taxonomy: Mapping[str, Mapping[str, str]], field: str, code: Any) -> list[str]:
    taxonomy_code = to_taxonomy_code(field, code)
    if not taxonomy_code:
        return []
    return list(reversed(ancestors_for(raw_taxonomy, field, taxonomy_code))) + [taxonomy_code]


def label_for(raw_taxonomy: Mapping[str, Mapping[str, str]], field: str, code: Any) -> str:
    taxonomy = raw_taxonomy[FIELD_TO_TAXONOMY[field]]
    return taxonomy.get(to_taxonomy_code(field, code), "")


def common_prefix_len(left: list[str], right: list[str]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def set_based_scores(gold_path: list[str], pred_path: list[str]) -> dict[str, float]:
    gold_set = set(gold_path)
    pred_set = set(pred_path)
    intersection = len(gold_set & pred_set)
    precision = intersection / len(pred_set) if pred_set else 0.0
    recall = intersection / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "hierarchical_precision": precision,
        "hierarchical_recall": recall,
        "hierarchical_f1": f1,
    }


def compare(raw_taxonomy: Mapping[str, Mapping[str, str]], field: str, gold: Any, pred: Any) -> dict[str, Any]:
    gold_path = path_for(raw_taxonomy, field, gold)
    pred_path = path_for(raw_taxonomy, field, pred)
    common = common_prefix_len(gold_path, pred_path)
    min_depth = max(min(len(gold_path), len(pred_path)), 1)
    max_depth = max(len(gold_path), len(pred_path), 1)
    exact = to_taxonomy_code(field, gold) == to_taxonomy_code(field, pred)
    parent_match = bool(gold_path and pred_path and gold_path[0] == pred_path[0])
    ancestor_compatible = bool(common == min(len(gold_path), len(pred_path)) and common > 0)
    pred_more_specific = bool(ancestor_compatible and len(pred_path) > len(gold_path))
    pred_more_general = bool(ancestor_compatible and len(pred_path) < len(gold_path))
    sibling_same_parent = bool(parent_match and not ancestor_compatible and not exact)
    return {
        "exact": exact,
        "parent_match": parent_match,
        "ancestor_compatible": ancestor_compatible,
        "pred_more_specific_same_branch": pred_more_specific,
        "pred_more_general_same_branch": pred_more_general,
        "sibling_same_parent": sibling_same_parent,
        "hier_score_max_depth": common / max_depth,
        "hier_score_min_depth": common / min_depth,
        **set_based_scores(gold_path, pred_path),
        "common_prefix_len": common,
        "gold_path": gold_path,
        "pred_path": pred_path,
    }


def candidate_by_code(candidates: list[Mapping[str, Any]], code: Any) -> Mapping[str, Any] | None:
    target = clean_text(code)
    for item in candidates:
        if clean_text(item.get("code")) == target:
            return item
    return None


def candidate_rank(candidates: list[Mapping[str, Any]], code: Any) -> int | None:
    item = candidate_by_code(candidates, code)
    if not item:
        return None
    rank = item.get("rank_dual_encoder")
    return int(rank) if rank is not None else None


def candidate_score(candidates: list[Mapping[str, Any]], code: Any) -> float | None:
    item = candidate_by_code(candidates, code)
    if not item:
        return None
    score = item.get("score")
    return float(score) if score is not None else None


def init_stats() -> dict[str, float]:
    return {
        "n": 0,
        "exact": 0,
        "parent_match": 0,
        "ancestor_compatible": 0,
        "pred_more_specific_same_branch": 0,
        "pred_more_general_same_branch": 0,
        "sibling_same_parent": 0,
        "hier_score_max_depth": 0.0,
        "hier_score_min_depth": 0.0,
        "hierarchical_precision": 0.0,
        "hierarchical_recall": 0.0,
        "hierarchical_f1": 0.0,
    }


def add_stats(stats: dict[str, float], cmp: Mapping[str, Any]) -> None:
    stats["n"] += 1
    for key in (
        "exact",
        "parent_match",
        "ancestor_compatible",
        "pred_more_specific_same_branch",
        "pred_more_general_same_branch",
        "sibling_same_parent",
    ):
        stats[key] += int(bool(cmp[key]))
    stats["hier_score_max_depth"] += float(cmp["hier_score_max_depth"])
    stats["hier_score_min_depth"] += float(cmp["hier_score_min_depth"])
    stats["hierarchical_precision"] += float(cmp["hierarchical_precision"])
    stats["hierarchical_recall"] += float(cmp["hierarchical_recall"])
    stats["hierarchical_f1"] += float(cmp["hierarchical_f1"])


def finalize(stats: Mapping[str, float]) -> dict[str, float]:
    n = max(int(stats["n"]), 1)
    return {
        "n": int(stats["n"]),
        "exact_accuracy": stats["exact"] / n,
        "parent_accuracy": stats["parent_match"] / n,
        "ancestor_compatible_accuracy": stats["ancestor_compatible"] / n,
        "more_specific_same_branch_rate": stats["pred_more_specific_same_branch"] / n,
        "more_general_same_branch_rate": stats["pred_more_general_same_branch"] / n,
        "sibling_same_parent_rate": stats["sibling_same_parent"] / n,
        "hier_score_max_depth_mean": stats["hier_score_max_depth"] / n,
        "hier_score_min_depth_mean": stats["hier_score_min_depth"] / n,
        "hierarchical_precision": stats["hierarchical_precision"] / n,
        "hierarchical_recall": stats["hierarchical_recall"] / n,
        "hierarchical_f1": stats["hierarchical_f1"] / n,
    }


def top_candidates_text(candidates: list[Mapping[str, Any]], limit: int = 5) -> str:
    parts = []
    for item in candidates[:limit]:
        parts.append(
            f"{item.get('rank_dual_encoder')}:{item.get('code')} {clean_text(item.get('label'))}"
        )
    return " | ".join(parts)


def truthy(value: Any) -> bool:
    return clean_text(value).lower() in {"1", "true", "yes", "y", "si", "sì", "ok"}


def load_gold_overrides(path: Path | None) -> dict[tuple[str, str], str]:
    if not path:
        return {}
    overrides: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record_key = clean_text(row.get("record_key"))
            field = clean_text(row.get("field"))
            if not record_key or not field:
                continue
            corrected = ""
            for column in CORRECTED_GOLD_COLUMNS:
                corrected = clean_text(row.get(column))
                if corrected:
                    break
            if not corrected:
                continue
            flags = [clean_text(row.get(column)) for column in CORRECTION_FLAG_COLUMNS if row.get(column) is not None]
            if flags and not any(truthy(flag) for flag in flags):
                continue
            overrides[(record_key, field)] = corrected
    return overrides


def suspicion_categories(
    *,
    gold: str,
    pred: str,
    dual: str,
    gold_rank: int | None,
    pred_rank: int | None,
    dual_rank: int | None,
    gold_score: float | None,
    pred_score: float | None,
    dual_score: float | None,
    cmp_pred: Mapping[str, Any],
    cmp_dual: Mapping[str, Any],
    llm_changed_dual: bool,
) -> list[str]:
    categories: list[str] = []
    if gold_rank is None:
        categories.append("gold_absent_from_retriever_topk")
    if pred != gold and pred == dual and gold_rank is not None:
        categories.append("model_and_retriever_agree_against_gold")
    if pred != gold and cmp_pred["pred_more_specific_same_branch"]:
        categories.append("prediction_more_specific_than_gold_same_branch")
    if pred != gold and cmp_pred["pred_more_general_same_branch"]:
        categories.append("prediction_more_general_than_gold_same_branch")
    if pred != gold and cmp_pred["sibling_same_parent"]:
        categories.append("prediction_sibling_of_gold_same_parent")
    if pred != gold and cmp_pred["parent_match"]:
        categories.append("gold_parent_supported_but_leaf_differs")
    if pred != gold and not cmp_pred["parent_match"]:
        categories.append("different_parent_than_gold")
    if pred != gold and gold_rank is not None and pred_rank is not None and pred_rank < gold_rank:
        categories.append("prediction_ranked_above_gold_by_retriever")
    if (
        pred != gold
        and gold_score is not None
        and pred_score is not None
        and pred_score >= gold_score + 0.05
    ):
        categories.append("prediction_score_margin_over_gold")
    if llm_changed_dual and cmp_pred["exact"] and not cmp_dual["exact"]:
        categories.append("llm_corrected_retriever_vs_gold")
    if llm_changed_dual and not cmp_pred["exact"] and cmp_dual["exact"]:
        categories.append("llm_broke_gold_exact")
    return categories or ["no_suspicion"]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    raw_taxonomy = load_taxonomy(args.taxonomy)
    rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    gold_overrides = load_gold_overrides(args.gold_overrides_csv)
    models = ["dual", "llm"]
    overall = {model: init_stats() for model in models}
    by_field = {model: {field: init_stats() for field in FIELDS} for model in models}
    record_level = {
        model: {"exact_all": 0, "parent_all": 0, "ancestor_all": 0}
        for model in models
    }
    suspicion_counter: Counter[str] = Counter()
    changed_counter: Counter[str] = Counter()
    detail_rows: list[dict[str, Any]] = []
    suspicious_rows: list[dict[str, Any]] = []

    for record in rows:
        record_flags = {
            model: {"exact": True, "parent": True, "ancestor": True}
            for model in models
        }
        for field in FIELDS:
            candidates = list(record["dual_candidates"][field])
            original_gold = clean_text(record["gold"][field])
            gold = gold_overrides.get((clean_text(record.get("record_key")), field), original_gold)
            dual = clean_text(record["dual_top1"][field]["code"])
            llm = clean_text(record["llm_top1"][field]["code"])
            predictions = {"dual": dual, "llm": llm}
            cmps = {model: compare(raw_taxonomy, field, gold, pred) for model, pred in predictions.items()}

            for model in models:
                add_stats(overall[model], cmps[model])
                add_stats(by_field[model][field], cmps[model])
                record_flags[model]["exact"] &= bool(cmps[model]["exact"])
                record_flags[model]["parent"] &= bool(cmps[model]["parent_match"])
                record_flags[model]["ancestor"] &= bool(cmps[model]["ancestor_compatible"])

            gold_rank = candidate_rank(candidates, gold)
            dual_rank = candidate_rank(candidates, dual)
            llm_rank = candidate_rank(candidates, llm)
            gold_score = candidate_score(candidates, gold)
            dual_score = candidate_score(candidates, dual)
            llm_score = candidate_score(candidates, llm)
            changed = dual != llm
            categories = suspicion_categories(
                gold=gold,
                pred=llm,
                dual=dual,
                gold_rank=gold_rank,
                pred_rank=llm_rank,
                dual_rank=dual_rank,
                gold_score=gold_score,
                pred_score=llm_score,
                dual_score=dual_score,
                cmp_pred=cmps["llm"],
                cmp_dual=cmps["dual"],
                llm_changed_dual=changed,
            )
            for category in categories:
                suspicion_counter[category] += 1
            if changed:
                if cmps["llm"]["exact"] and not cmps["dual"]["exact"]:
                    changed_counter["exact_improve"] += 1
                elif cmps["dual"]["exact"] and not cmps["llm"]["exact"]:
                    changed_counter["exact_worse"] += 1
                elif cmps["llm"]["hierarchical_f1"] > cmps["dual"]["hierarchical_f1"]:
                    changed_counter["hier_improve"] += 1
                elif cmps["llm"]["hierarchical_f1"] < cmps["dual"]["hierarchical_f1"]:
                    changed_counter["hier_worse"] += 1
                else:
                    changed_counter["hier_equal"] += 1

            raw = record.get("llm_raw", {}).get(field, {})
            detail = {
                "record_key": record.get("record_key"),
                "record_index": record.get("record_index"),
                "field": field,
                "gold_code": gold,
                "gold_label": label_for(raw_taxonomy, field, gold),
                "original_gold_code": original_gold,
                "original_gold_label": label_for(raw_taxonomy, field, original_gold),
                "gold_was_overridden": gold != original_gold,
                "dual_code": dual,
                "dual_label": clean_text(record["dual_top1"][field].get("label")),
                "llm_code": llm,
                "llm_label": clean_text(record["llm_top1"][field].get("label")),
                "llm_changed_dual": changed,
                "gold_rank_in_topk": gold_rank,
                "llm_rank_in_topk": llm_rank,
                "dual_exact": cmps["dual"]["exact"],
                "llm_exact": cmps["llm"]["exact"],
                "dual_parent_match": cmps["dual"]["parent_match"],
                "llm_parent_match": cmps["llm"]["parent_match"],
                "dual_ancestor_compatible": cmps["dual"]["ancestor_compatible"],
                "llm_ancestor_compatible": cmps["llm"]["ancestor_compatible"],
                "dual_hier_score": round(cmps["dual"]["hier_score_max_depth"], 4),
                "llm_hier_score": round(cmps["llm"]["hier_score_max_depth"], 4),
                "dual_hierarchical_f1": round(cmps["dual"]["hierarchical_f1"], 4),
                "llm_hierarchical_f1": round(cmps["llm"]["hierarchical_f1"], 4),
                "dual_hierarchical_precision": round(cmps["dual"]["hierarchical_precision"], 4),
                "llm_hierarchical_precision": round(cmps["llm"]["hierarchical_precision"], 4),
                "dual_hierarchical_recall": round(cmps["dual"]["hierarchical_recall"], 4),
                "llm_hierarchical_recall": round(cmps["llm"]["hierarchical_recall"], 4),
                "suspicion_categories": ";".join(categories),
                "gold_path": " > ".join(cmps["llm"]["gold_path"]),
                "llm_path": " > ".join(cmps["llm"]["pred_path"]),
                "selected_parent_code": clean_text(raw.get("selected_parent_code")),
                "parent_reason": clean_text(raw.get("parent_reason")),
                "leaf_reason": clean_text(raw.get("leaf_reason")),
                "reason": clean_text(raw.get("reason")),
                "top5_candidates": top_candidates_text(candidates),
                "text": clean_text(record.get("text")),
            }
            detail_rows.append(detail)
            if args.include_no_suspicion or categories != ["no_suspicion"]:
                suspicious_rows.append(detail)

        for model in models:
            record_level[model]["exact_all"] += int(record_flags[model]["exact"])
            record_level[model]["parent_all"] += int(record_flags[model]["parent"])
            record_level[model]["ancestor_all"] += int(record_flags[model]["ancestor"])

    summary = {
        "input_predictions": str(args.predictions),
        "taxonomy": str(args.taxonomy),
        "gold_overrides_csv": str(args.gold_overrides_csv) if args.gold_overrides_csv else "",
        "gold_overrides_applied": sum(1 for row in detail_rows if row["gold_was_overridden"]),
        "records": len(rows),
        "field_decisions": len(rows) * len(FIELDS),
        "metric_definitions": {
            "exact_accuracy": "Predizione uguale alla gold; per agenti materiali normalizza xx.yy.00.00 a xx.yy.",
            "parent_accuracy": "Gold e predizione condividono il primo nodo del percorso tassonomico.",
            "ancestor_compatible_accuracy": "Gold e predizione sono sullo stesso ramo e una e' antenata/discendente dell'altra.",
            "hier_score_max_depth_mean": "Prefisso comune del percorso / profondita' massima fra gold e predizione.",
            "hierarchical_precision": "Set-based PH: |antenati(gold) intersecato antenati(pred)| / |antenati(pred)|, includendo il codice stesso.",
            "hierarchical_recall": "Set-based RH: |antenati(gold) intersecato antenati(pred)| / |antenati(gold)|, includendo il codice stesso.",
            "hierarchical_f1": "Set-based FH: media armonica di hierarchical_precision e hierarchical_recall.",
            "more_specific_same_branch_rate": "Quota di predizioni piu' specifiche della gold nello stesso ramo.",
            "sibling_same_parent_rate": "Quota di predizioni con stesso parent della gold ma foglia diversa.",
        },
        "overall": {model: finalize(overall[model]) for model in models},
        "by_field": {model: {field: finalize(by_field[model][field]) for field in FIELDS} for model in models},
        "record_level": {
            model: {
                "record_exact_all_fields": record_level[model]["exact_all"] / max(len(rows), 1),
                "record_parent_all_fields": record_level[model]["parent_all"] / max(len(rows), 1),
                "record_ancestor_all_fields": record_level[model]["ancestor_all"] / max(len(rows), 1),
            }
            for model in models
        },
        "delta_llm_minus_dual": {},
        "changed_decisions": dict(changed_counter),
        "gold_noise_suspicion_counts": dict(suspicion_counter),
    }
    for metric in summary["overall"]["dual"]:
        if metric == "n":
            continue
        summary["delta_llm_minus_dual"][metric] = summary["overall"]["llm"][metric] - summary["overall"]["dual"][metric]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_detail_csv, detail_rows)
    write_csv(args.output_suspicious_csv, suspicious_rows)
    write_markdown(args.output_report_md, summary, suspicious_rows[: args.report_examples])
    return summary


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: Mapping[str, Any], examples: list[Mapping[str, Any]]) -> None:
    lines = [
        "# Valutazione Gerarchica E Gold Noise",
        "",
        f"- Records: `{summary['records']}`",
        f"- Decisioni campo: `{summary['field_decisions']}`",
        "",
        "## Metriche Globali",
        "",
        "| metrica | dual | llm | delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, dual_value in summary["overall"]["dual"].items():
        if metric == "n":
            continue
        llm_value = summary["overall"]["llm"][metric]
        delta = summary["delta_llm_minus_dual"][metric]
        lines.append(f"| {metric} | `{dual_value:.4f}` | `{llm_value:.4f}` | `{delta:+.4f}` |")
    lines.extend(["", "## Sospetti Gold Label", ""])
    for key, value in sorted(summary["gold_noise_suspicion_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Esempi Da Auditare", ""])
    for row in examples:
        lines.extend(
            [
                f"### {row['record_key']} - {row['field']} - {row['suspicion_categories']}",
                "",
                f"- Gold: `{row['gold_code']}` {row['gold_label']}",
                f"- Dual: `{row['dual_code']}` {row['dual_label']}",
                f"- LLM: `{row['llm_code']}` {row['llm_label']}",
                f"- Gold rank topK: `{row['gold_rank_in_topk']}`; LLM rank topK: `{row['llm_rank_in_topk']}`",
                f"- Gold path: `{row['gold_path']}`",
                f"- LLM path: `{row['llm_path']}`",
                f"- Parent reason: {row['parent_reason']}",
                f"- Leaf reason: {row['leaf_reason']}",
                f"- Reason: {row['reason']}",
                f"- Testo: {row['text']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rivaluta prediction ESAW con metriche gerarchiche e segnali di gold noise.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=Path("tassonomia.json"))
    parser.add_argument(
        "--gold-overrides-csv",
        type=Path,
        default=None,
        help=(
            "CSV opzionale con record_key, field e corrected_gold_code. "
            "Se contiene accepted/use_correction/is_corrected, applica solo le righe vere."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-detail-csv", type=Path, required=True)
    parser.add_argument("--output-suspicious-csv", type=Path, required=True)
    parser.add_argument("--output-report-md", type=Path, required=True)
    parser.add_argument("--include-no-suspicion", action="store_true")
    parser.add_argument("--report-examples", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps({
        "overall": summary["overall"],
        "delta_llm_minus_dual": summary["delta_llm_minus_dual"],
        "gold_noise_suspicion_counts": summary["gold_noise_suspicion_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
