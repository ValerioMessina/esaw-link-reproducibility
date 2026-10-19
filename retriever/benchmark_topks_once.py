from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from esaw_biencoder.data import (
    Candidate,
    TEXT_FORMATS,
    build_candidates,
    build_explicit_taxonomy,
    label_to_output_code,
    load_dataset,
    load_taxonomy,
    query_text,
    record_value,
)
from esaw_biencoder.evaluate import filter_records_by_split
from esaw_biencoder.retrieval import top_candidate_codes
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ESAW Top1/Top5/Top10 in un solo passaggio.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--taxonomy", default="tassonomia.json")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--split", choices=["all", "train", "validation", "test"], default="test")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--text-format", choices=TEXT_FORMATS, default="v5")
    parser.add_argument(
        "--rebuild-candidates",
        action="store_true",
        help="Ricostruisce i candidati dalla tassonomia invece di usare candidates.json del modello.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    model_dir = Path(args.model)
    records = filter_records_by_split(load_dataset(args.dataset), args.split_file, args.split)
    if args.sample_size and len(records) > args.sample_size:
        records = rng.sample(records, args.sample_size)

    model = SentenceTransformer(args.model)
    if args.rebuild_candidates:
        candidates = build_candidates(build_explicit_taxonomy(load_taxonomy(args.taxonomy)), text_format=args.text_format)
    else:
        candidates = load_candidates(model_dir, args.taxonomy, args.text_format)
    report: Dict[str, Dict[str, float]] = {}

    for target in TARGET_FIELDS:
        usable_records = [record for record in records if record_value(record, target.field)]
        field_candidates = candidates[target.field]
        candidate_codes = [candidate.code for candidate in field_candidates]
        candidate_embeddings = model.encode(
            [candidate.text for candidate in field_candidates],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        query_embeddings = model.encode(
            [query_text(record, target, args.text_format) for record in usable_records],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        scores = cosine_matrix(query_embeddings, candidate_embeddings)
        gold = [label_to_output_code(target, record_value(record, target.field)) for record in usable_records]

        hits = {1: 0, 5: 0, 10: 0}
        for row_index, gold_code in enumerate(gold):
            predicted_codes = top_candidate_codes(scores[row_index], field_candidates, 10)
            for top_k in hits:
                hits[top_k] += int(gold_code in predicted_codes[:top_k])

        total = max(1, len(usable_records))
        report[target.output_field] = {
            "records": len(usable_records),
            "top1_hits": hits[1],
            "top5_hits": hits[5],
            "top10_hits": hits[10],
            "top1_accuracy": hits[1] / total,
            "top5_accuracy": hits[5] / total,
            "top10_accuracy": hits[10] / total,
        }

    means = {
        key: sum(field[key] for field in report.values()) / len(report)
        for key in ("top1_accuracy", "top5_accuracy", "top10_accuracy")
    }
    payload = {"fields": report, "mean": means}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
