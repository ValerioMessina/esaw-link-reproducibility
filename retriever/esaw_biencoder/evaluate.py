from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .data import (
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
from .retrieval import top_candidate_codes
from .schema import TARGET_FIELDS

TEXT_FORMAT_AUTO = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valuta top-k accuracy del bi-encoder ESAW.")
    parser.add_argument("--model", default="models/esaw-biencoder")
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--taxonomy", default="tassonomia.json")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--split-file", default=None, help="JSON con split fisso train/validation/test.")
    parser.add_argument("--split", choices=["all", "train", "validation", "test"], default="all")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--retrieval",
        choices=["flat", "hierarchical"],
        default="flat",
        help="'flat' confronta tutte le classi; 'hierarchical' predice prima la macroclasse e poi cerca nei figli.",
    )
    parser.add_argument(
        "--hierarchy-beam-size",
        type=int,
        default=1,
        help="Numero di macroclassi candidate da espandere nel retrieval gerarchico.",
    )
    parser.add_argument(
        "--text-format",
        choices=(TEXT_FORMAT_AUTO, *TEXT_FORMATS),
        default=TEXT_FORMAT_AUTO,
        help="Formato testi query/candidati. 'auto' legge training_config.json del modello se presente.",
    )
    return parser.parse_args()


def resolve_text_format(model_dir: Path, requested: str) -> str:
    if requested != TEXT_FORMAT_AUTO:
        return requested
    config_path = model_dir / "training_config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        text_format = payload.get("text_format")
        if text_format in TEXT_FORMATS:
            return str(text_format)
    return "legacy"


def load_candidates(model_dir: Path, taxonomy_path: str | Path, text_format: str) -> Dict[str, List[Candidate]]:
    candidates_path = model_dir / "candidates.json"
    if candidates_path.exists():
        payload = json.loads(candidates_path.read_text(encoding="utf-8"))
        return {
            field: [Candidate(**{**candidate, "ancestors": candidate.get("ancestors", [])}) for candidate in field_candidates]
            for field, field_candidates in payload.items()
        }
    return build_candidates(build_explicit_taxonomy(load_taxonomy(taxonomy_path)), text_format=text_format)


def record_split_key(record: Mapping[str, Any], index: int) -> str:
    for field in ("id", "numero_caso"):
        value = record.get(field) or record.get(field.upper())
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"index:{index}"


def filter_records_by_split(
    records: Sequence[Mapping[str, Any]],
    split_file: str | Path | None,
    split_name: str,
) -> List[Mapping[str, Any]]:
    if split_file is None or split_name == "all":
        return list(records)
    payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
    splits = payload.get("splits", payload)
    keys = set(splits.get(split_name, []))
    return [
        record
        for index, record in enumerate(records)
        if record_split_key(record, index) in keys
    ]


def cosine_matrix(query_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    queries = query_embeddings / np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
    candidates = candidate_embeddings / np.maximum(np.linalg.norm(candidate_embeddings, axis=1, keepdims=True), 1e-12)
    return queries @ candidates.T


def macro_code_for_candidate(candidate: Candidate) -> str:
    return candidate.parent_code or candidate.taxonomy_code


def hierarchical_best_indexes(
    scores: np.ndarray,
    candidates: Sequence[Candidate],
    top_k: int,
    hierarchy_beam_size: int,
) -> tuple[np.ndarray, List[List[str]]]:
    macro_indexes = [index for index, candidate in enumerate(candidates) if candidate.parent_code is None]
    if not macro_indexes:
        best = np.argsort(-scores, axis=1)[:, :top_k]
        return best, [[] for _ in range(scores.shape[0])]

    macro_scores = scores[:, macro_indexes]
    macro_order = np.argsort(-macro_scores, axis=1)[:, : max(1, hierarchy_beam_size)]
    rows: List[List[int]] = []
    selected_macros: List[List[str]] = []

    for row_index in range(scores.shape[0]):
        macro_codes = [candidates[macro_indexes[local_index]].taxonomy_code for local_index in macro_order[row_index]]
        scope = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.taxonomy_code in macro_codes
            or candidate.parent_code in macro_codes
            or any(ancestor in macro_codes for ancestor in candidate.ancestors)
        ]
        if not scope:
            scope = list(range(len(candidates)))
        ordered_scope = sorted(scope, key=lambda index: float(scores[row_index, index]), reverse=True)
        rows.append(ordered_scope[:top_k])
        selected_macros.append(macro_codes)

    width = max((len(row) for row in rows), default=0)
    padded = np.full((len(rows), width), -1, dtype=int)
    for row_index, row in enumerate(rows):
        padded[row_index, : len(row)] = row
    return padded, selected_macros


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    model_dir = Path(args.model)
    text_format = resolve_text_format(model_dir, args.text_format)
    records = filter_records_by_split(load_dataset(args.dataset), args.split_file, args.split)
    if args.sample_size and len(records) > args.sample_size:
        records = rng.sample(records, args.sample_size)

    model = SentenceTransformer(args.model)
    candidates = load_candidates(model_dir, args.taxonomy, text_format)
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
            [query_text(record, target, text_format) for record in usable_records],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        scores = cosine_matrix(query_embeddings, candidate_embeddings)
        if args.retrieval == "hierarchical":
            best, predicted_macros = hierarchical_best_indexes(
                scores,
                field_candidates,
                args.top_k,
                args.hierarchy_beam_size,
            )
        else:
            best = np.argsort(-scores, axis=1)[:, : args.top_k]
            predicted_macros = [[] for _ in usable_records]

        gold = [label_to_output_code(target, record_value(record, target.field)) for record in usable_records]
        candidate_by_code = {candidate.code: candidate for candidate in field_candidates}

        top1 = 0
        topk = 0
        macro_correct = 0
        for row_index, gold_code in enumerate(gold):
            if args.retrieval == "flat":
                predicted_codes = top_candidate_codes(scores[row_index], field_candidates, args.top_k)
            else:
                row_indexes = [int(index) for index in best[row_index] if int(index) >= 0]
                predicted_codes = [candidate_codes[i] for i in row_indexes]
            top1 += int(bool(predicted_codes) and predicted_codes[0] == gold_code)
            topk += int(gold_code in predicted_codes)
            gold_candidate = candidate_by_code.get(gold_code)
            if args.retrieval == "hierarchical" and gold_candidate is not None:
                macro_correct += int(macro_code_for_candidate(gold_candidate) in predicted_macros[row_index])

        total = max(1, len(usable_records))
        field_report: Dict[str, float] = {
            "records": len(usable_records),
            "top1_accuracy": round(top1 / total, 4),
            f"top{args.top_k}_accuracy": round(topk / total, 4),
        }
        if args.retrieval == "hierarchical":
            field_report["macro_accuracy"] = round(macro_correct / total, 4)
        report[target.output_field] = field_report

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
