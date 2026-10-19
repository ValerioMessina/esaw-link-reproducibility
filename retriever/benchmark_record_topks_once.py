from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
from sentence_transformers import SentenceTransformer

from esaw_biencoder.data import (
    TEXT_FORMATS,
    build_candidates,
    build_explicit_taxonomy,
    label_to_output_code,
    load_dataset,
    load_taxonomy,
    query_text,
    record_value,
)
from esaw_biencoder.evaluate import (
    cosine_matrix,
    filter_records_by_split,
    load_candidates,
    resolve_text_format,
)
from esaw_biencoder.retrieval import top_candidate_codes
from esaw_biencoder.schema import TARGET_FIELDS

TEXT_FORMAT_AUTO = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark record-level ESAW: tutti gli 8 campi corretti insieme a Top1/Top5/Top10."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--taxonomy", default="tassonomia.json")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--split", choices=["all", "train", "validation", "test"], default="test")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--text-format",
        choices=(TEXT_FORMAT_AUTO, *TEXT_FORMATS),
        default=TEXT_FORMAT_AUTO,
    )
    parser.add_argument(
        "--rebuild-candidates",
        action="store_true",
        help="Ricostruisce i candidati dalla tassonomia invece di usare candidates.json del modello.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def complete_target_records(records: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        record
        for record in records
        if all(record_value(record, target.field) for target in TARGET_FIELDS)
    ]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    model_dir = Path(args.model)
    text_format = resolve_text_format(model_dir, args.text_format)

    records = filter_records_by_split(load_dataset(args.dataset), args.split_file, args.split)
    records = complete_target_records(records)
    if args.sample_size and len(records) > args.sample_size:
        records = rng.sample(records, args.sample_size)

    model = SentenceTransformer(args.model)
    if args.rebuild_candidates:
        candidates = build_candidates(build_explicit_taxonomy(load_taxonomy(args.taxonomy)), text_format=text_format)
    else:
        candidates = load_candidates(model_dir, args.taxonomy, text_format)

    field_hits: Dict[str, Dict[str, int]] = {
        target.output_field: {"top1": 0, "top5": 0, "top10": 0}
        for target in TARGET_FIELDS
    }
    record_hits = {1: np.ones(len(records), dtype=bool), 5: np.ones(len(records), dtype=bool), 10: np.ones(len(records), dtype=bool)}

    for target in TARGET_FIELDS:
        field_candidates = candidates[target.field]
        candidate_embeddings = model.encode(
            [candidate.text for candidate in field_candidates],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        query_embeddings = model.encode(
            [query_text(record, target, text_format) for record in records],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        scores = cosine_matrix(query_embeddings, candidate_embeddings)

        for row_index, record in enumerate(records):
            gold_code = label_to_output_code(target, record_value(record, target.field))
            predicted_codes = top_candidate_codes(scores[row_index], field_candidates, 10)
            hit1 = bool(predicted_codes) and predicted_codes[0] == gold_code
            hit5 = gold_code in predicted_codes[:5]
            hit10 = gold_code in predicted_codes[:10]
            field_hits[target.output_field]["top1"] += int(hit1)
            field_hits[target.output_field]["top5"] += int(hit5)
            field_hits[target.output_field]["top10"] += int(hit10)
            record_hits[1][row_index] &= hit1
            record_hits[5][row_index] &= hit5
            record_hits[10][row_index] &= hit10

    total = max(1, len(records))
    fields = {
        field: {
            "top1_hits": hits["top1"],
            "top5_hits": hits["top5"],
            "top10_hits": hits["top10"],
            "top1_accuracy": hits["top1"] / total,
            "top5_accuracy": hits["top5"] / total,
            "top10_accuracy": hits["top10"] / total,
        }
        for field, hits in field_hits.items()
    }
    payload = {
        "model": args.model,
        "split": args.split,
        "records": len(records),
        "text_format": text_format,
        "record_exact@1": float(record_hits[1].mean()) if len(records) else 0.0,
        "record_recall@5": float(record_hits[5].mean()) if len(records) else 0.0,
        "record_recall@10": float(record_hits[10].mean()) if len(records) else 0.0,
        "record_hits": {
            "exact@1": int(record_hits[1].sum()),
            "recall@5": int(record_hits[5].sum()),
            "recall@10": int(record_hits[10].sum()),
        },
        "field_metrics": fields,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
