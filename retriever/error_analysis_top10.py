from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from esaw_biencoder.data import (
    Candidate,
    INPUT_FIELDS,
    TEXT_FORMATS,
    build_candidates,
    build_explicit_taxonomy,
    label_to_output_code,
    load_dataset,
    load_taxonomy,
    normalize_value,
    query_text,
    record_value,
)
from esaw_biencoder.evaluate import filter_records_by_split, record_split_key
from esaw_biencoder.schema import TARGET_FIELDS


def cosine_matrix(query_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    queries = query_embeddings / np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
    candidates = candidate_embeddings / np.maximum(np.linalg.norm(candidate_embeddings, axis=1, keepdims=True), 1e-12)
    return queries @ candidates.T


def load_candidates(model_dir: Path, taxonomy_path: str | Path, text_format: str) -> Dict[str, List[Candidate]]:
    candidates_path = model_dir / "candidates.json"
    if candidates_path.exists():
        payload = json.loads(candidates_path.read_text(encoding="utf-8"))
        return {
            field: [Candidate(**{**candidate, "ancestors": candidate.get("ancestors", [])}) for candidate in field_candidates]
            for field, field_candidates in payload.items()
        }
    return build_candidates(build_explicit_taxonomy(load_taxonomy(taxonomy_path)), text_format=text_format)


def round_score(score: float) -> float:
    return round(float(score), 6)


def candidate_payload(candidate: Candidate, score: float, rank: int, is_gold: bool) -> Dict[str, Any]:
    return {
        "rank": rank,
        "code": candidate.code,
        "label": candidate.label,
        "score": round_score(score),
        "is_gold": is_gold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Error analysis ESAW con Top10, score e gold label.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--taxonomy", default="tassonomia.json")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--split", choices=["all", "train", "validation", "test"], default="test")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--text-format", choices=TEXT_FORMATS, default="v5")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    model_dir = Path(args.model)
    records = filter_records_by_split(load_dataset(args.dataset), args.split_file, args.split)
    if args.sample_size and len(records) > args.sample_size:
        records = rng.sample(records, args.sample_size)

    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        rows.append(
            {
                "split_key": record_split_key(record, index),
                "numero_caso": normalize_value(record_value(record, "numero_caso")),
                "inputs": {field: normalize_value(record_value(record, field)) for field in INPUT_FIELDS},
                "targets": {},
            }
        )

    model = SentenceTransformer(args.model)
    candidates = load_candidates(model_dir, args.taxonomy, args.text_format)
    summary_fields: Dict[str, Dict[str, Any]] = {}

    for target in TARGET_FIELDS:
        usable_indexes = [index for index, record in enumerate(records) if record_value(record, target.field)]
        usable_records = [records[index] for index in usable_indexes]
        field_candidates = candidates[target.field]
        candidate_by_code = {candidate.code: candidate for candidate in field_candidates}
        candidate_embeddings = model.encode(
            [candidate.text for candidate in field_candidates],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        query_embeddings = model.encode(
            [query_text(record, target, args.text_format) for record in usable_records],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        scores = cosine_matrix(query_embeddings, candidate_embeddings)
        top_k = min(max(1, args.top_k), len(field_candidates))
        best_indexes = np.argsort(-scores, axis=1)[:, :top_k]

        top1_hits = 0
        top5_hits = 0
        top10_hits = 0
        rescued_top10 = 0
        same_parent_top1_errors = 0
        confusion_pairs: Counter[tuple[str, str]] = Counter()
        missed_gold_top10: Counter[str] = Counter()
        rank_distribution: Counter[str] = Counter()
        top1_error_margins: List[float] = []

        for row_index, record_index in enumerate(usable_indexes):
            gold_code = label_to_output_code(target, record_value(records[record_index], target.field))
            gold_candidate = candidate_by_code.get(gold_code)
            if gold_candidate is None:
                raise ValueError(f"Gold code non trovato tra i candidati: {target.output_field}={gold_code}")

            all_scores = scores[row_index]
            gold_index = field_candidates.index(gold_candidate)
            gold_score = float(all_scores[gold_index])
            gold_rank = int(np.count_nonzero(all_scores > gold_score)) + 1
            prediction_indexes = [int(index) for index in best_indexes[row_index]]
            top_candidates = [
                candidate_payload(field_candidates[index], all_scores[index], rank + 1, field_candidates[index].code == gold_code)
                for rank, index in enumerate(prediction_indexes)
            ]
            predicted = field_candidates[prediction_indexes[0]]
            top1_correct = predicted.code == gold_code
            top5_hit = gold_rank <= min(5, top_k)
            top10_hit = gold_rank <= top_k
            top1_hits += int(top1_correct)
            top5_hits += int(top5_hit)
            top10_hits += int(top10_hit)

            if gold_rank == 1:
                rank_distribution["rank_1"] += 1
            elif gold_rank <= 5:
                rank_distribution["rank_2_5"] += 1
            elif gold_rank <= 10:
                rank_distribution["rank_6_10"] += 1
            else:
                rank_distribution["outside_top10"] += 1

            if not top1_correct:
                confusion_pairs[(gold_code, predicted.code)] += 1
                top1_error_margins.append(float(all_scores[prediction_indexes[0]] - gold_score))
                if top10_hit:
                    rescued_top10 += 1
                else:
                    missed_gold_top10[gold_code] += 1
                if gold_candidate.parent_code and gold_candidate.parent_code == predicted.parent_code:
                    same_parent_top1_errors += 1

            rows[record_index]["targets"][target.output_field] = {
                "gold": {
                    "code": gold_code,
                    "label": gold_candidate.label,
                    "score": round_score(gold_score),
                    "rank": gold_rank,
                },
                "top1_correct": top1_correct,
                "top5_hit": top5_hit,
                "top10_hit": top10_hit,
                "top10": top_candidates,
            }

        total = max(1, len(usable_records))
        top1_errors = len(usable_records) - top1_hits
        summary_fields[target.output_field] = {
            "records": len(usable_records),
            "top1_accuracy": round(top1_hits / total, 4),
            "top5_accuracy": round(top5_hits / total, 4),
            "top10_accuracy": round(top10_hits / total, 4),
            "top1_errors": top1_errors,
            "recovered_within_top10_after_top1_error": rescued_top10,
            "outside_top10": total - top10_hits,
            "rank_distribution": dict(rank_distribution),
            "same_parent_top1_errors": same_parent_top1_errors,
            "mean_top1_to_gold_score_gap_on_errors": (
                round_score(sum(top1_error_margins) / len(top1_error_margins)) if top1_error_margins else 0.0
            ),
            "top_confusions": [
                {
                    "gold_code": gold_code,
                    "gold_label": candidate_by_code[gold_code].label,
                    "predicted_code": predicted_code,
                    "predicted_label": candidate_by_code[predicted_code].label,
                    "count": count,
                }
                for (gold_code, predicted_code), count in confusion_pairs.most_common(10)
            ],
            "most_frequent_gold_outside_top10": [
                {
                    "gold_code": code,
                    "gold_label": candidate_by_code[code].label,
                    "count": count,
                }
                for code, count in missed_gold_top10.most_common(10)
            ],
        }

    predictions_path = Path(args.predictions_output)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for row in rows:
            predictions_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = ("top1_accuracy", "top5_accuracy", "top10_accuracy")
    mean = {
        metric: round(sum(float(field[metric]) for field in summary_fields.values()) / len(summary_fields), 4)
        for metric in metrics
    }
    summary = {
        "model": str(model_dir),
        "dataset": args.dataset,
        "split": args.split,
        "split_file": args.split_file,
        "text_format": args.text_format,
        "retrieval": "flat",
        "top_k": args.top_k,
        "records": len(records),
        "predictions_output": str(predictions_path),
        "fields": summary_fields,
        "mean": mean,
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
