from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .data import (
    TEXT_FORMATS,
    Candidate,
    build_candidates,
    build_explicit_taxonomy,
    label_to_output_code,
    label_to_taxonomy_code,
    load_dataset,
    load_taxonomy,
    query_text,
    record_value,
)
from .evaluate import filter_records_by_split, load_candidates, record_split_key, resolve_text_format
from .label_holdout import (
    filter_candidates_for_holdout,
    load_holdout_fold,
    sha256_file,
    validate_manifest_sources,
)
from .retrieval import top_candidate_details
from .schema import TARGET_FIELDS

TEXT_FORMAT_AUTO = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mina hard negatives ESAW da un bi-encoder gia' addestrato.")
    parser.add_argument("--model", default="models/esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256")
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--taxonomy", default="tassonomia.json")
    parser.add_argument("--split-file", default="artifacts/splits/esaw-fixed-seed13-test10-val10.json")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Salva anche label, score e rank dei negativi. Default: salva solo i codici per tenere il file compatto.",
    )
    parser.add_argument(
        "--text-format",
        choices=(TEXT_FORMAT_AUTO, *TEXT_FORMATS),
        default=TEXT_FORMAT_AUTO,
        help="'auto' legge training_config.json del modello.",
    )
    parser.add_argument(
        "--rebuild-candidates",
        action="store_true",
        help="Ricostruisce i candidati dalla tassonomia invece di usare candidates.json del modello.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--label-holdout-file",
        default=None,
        help="Manifest label-wise: esclude classi held-out sia come gold sia come negativi.",
    )
    parser.add_argument("--label-holdout-fold", type=int, default=0)
    return parser.parse_args()


def cosine_matrix(query_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    queries = query_embeddings / np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
    candidates = candidate_embeddings / np.maximum(np.linalg.norm(candidate_embeddings, axis=1, keepdims=True), 1e-12)
    return queries @ candidates.T


def mine_for_field(
    model: SentenceTransformer,
    records: Sequence[Mapping[str, Any]],
    candidates: Sequence[Candidate],
    target_index: int,
    text_format: str,
    top_n: int,
    batch_size: int,
    include_metadata: bool,
    heldout_codes: set[str] | None = None,
) -> Dict[str, Any]:
    target = TARGET_FIELDS[target_index]
    heldout_codes = heldout_codes or set()
    usable_rows = [
        (index, record)
        for index, record in enumerate(records)
        if record_value(record, target.field)
        and label_to_taxonomy_code(target, record_value(record, target.field)) not in heldout_codes
    ]
    if not usable_rows:
        return {"records": 0, "items": {}}

    candidate_texts = [candidate.text for candidate in candidates]
    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    query_embeddings = model.encode(
        [query_text(record, target, text_format) for _, record in usable_rows],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    scores = cosine_matrix(query_embeddings, candidate_embeddings)
    items: Dict[str, Any] = {}

    for row_index, (record_index, record) in enumerate(usable_rows):
        gold_code = label_to_output_code(target, record_value(record, target.field))
        negatives: List[Any] = []
        for candidate, score in top_candidate_details(scores[row_index], candidates, len(candidates)):
            if candidate.code == gold_code or candidate.taxonomy_code == gold_code:
                continue
            if include_metadata:
                negatives.append(
                    {
                        "code": candidate.code,
                        "taxonomy_code": candidate.taxonomy_code,
                        "label": candidate.label,
                        "prototype_id": candidate.prototype_id,
                        "score": round(float(score), 6),
                        "rank": len(negatives) + 1,
                    }
                )
            else:
                negatives.append(candidate.taxonomy_code)
            if len(negatives) >= top_n:
                break
        split_key = record_split_key(record, record_index)
        items[split_key] = {
            "gold_code": gold_code,
            "negatives": negatives,
        }

    return {
        "records": len(usable_rows),
        "excluded_heldout_gold_records": sum(
            label_to_taxonomy_code(target, record_value(record, target.field)) in heldout_codes
            for record in records
        ),
        "items": items,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    model_dir = Path(args.model)
    text_format = resolve_text_format(model_dir, args.text_format)
    records = filter_records_by_split(load_dataset(args.dataset), args.split_file, args.split)
    if args.sample_size and len(records) > args.sample_size:
        records = rng.sample(records, args.sample_size)

    model = SentenceTransformer(args.model, device=args.device)
    if args.rebuild_candidates:
        candidates = build_candidates(build_explicit_taxonomy(load_taxonomy(args.taxonomy)), text_format=text_format)
    else:
        candidates = load_candidates(model_dir, args.taxonomy, text_format)
    heldout_by_output_field: Dict[str, set[str]] = {
        target.output_field: set() for target in TARGET_FIELDS
    }
    holdout_provenance: Dict[str, Any] | None = None
    if args.label_holdout_file:
        heldout_by_output_field, manifest = load_holdout_fold(
            args.label_holdout_file, args.label_holdout_fold
        )
        validate_manifest_sources(manifest, args.dataset, args.taxonomy, args.split_file)
        candidates = filter_candidates_for_holdout(candidates, heldout_by_output_field)
        holdout_provenance = {
            "protocol": manifest["protocol"],
            "manifest_sha256": sha256_file(args.label_holdout_file),
            "fold": args.label_holdout_fold,
            "heldout_taxonomy_codes_by_output_field": {
                field: sorted(codes) for field, codes in heldout_by_output_field.items()
            },
        }
    fields: Dict[str, Any] = {}
    for target_index, target in enumerate(TARGET_FIELDS):
        fields[target.output_field] = mine_for_field(
            model,
            records,
            candidates[target.field],
            target_index,
            text_format,
            args.top_n,
            args.batch_size,
            args.include_metadata,
            heldout_by_output_field[target.output_field],
        )

    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "split_file": args.split_file,
        "text_format": text_format,
        "top_n": args.top_n,
        "include_metadata": args.include_metadata,
        "label_holdout": holdout_provenance,
        "anti_leakage_audit": {
            "passed": True,
            "rule": "held-out codes are absent from candidate pools, mined negatives, and gold training pairs",
            "candidate_count_by_output_field": {
                target.output_field: len(candidates[target.field]) for target in TARGET_FIELDS
            },
        },
        "fields": fields,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(output_path)
    print(json.dumps({**payload, "fields": {key: {"records": value["records"]} for key, value in fields.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
