from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from esaw_biencoder.data import (
    Candidate,
    accident_to_text,
    build_candidates,
    build_explicit_taxonomy,
    label_to_output_code,
    load_dataset,
    load_taxonomy,
    query_text,
    record_value,
)
from esaw_biencoder.evaluate import cosine_matrix, filter_records_by_split, record_split_key
from esaw_biencoder.retrieval import top_candidate_details
from esaw_biencoder.schema import TARGET_FIELDS, TargetField


PROTOCOL_NAME = "esaw-task-level-zero-shot-semantic-retrieval-v1"
CANDIDATE_FORMATS = ("official-label", "official-label-role", "official-label-role-path")
QUERY_FORMATS = ("plain-role", "qwen-instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline task-level zero-shot ESAW: un encoder general-purpose congelato "
            "confronta le narrative con le descrizioni ufficiali delle classi."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--taxonomy", default="tassonomia.json")
    parser.add_argument(
        "--split-file",
        default="artifacts/splits/esaw-fixed-seed13-test10-val10.json",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Lo split train non e' ammesso dal runner zero-shot.",
    )
    parser.add_argument("--sample-size", type=int, default=0, help="0 usa tutto lo split.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="Per esempio mps, cuda o cpu.")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--candidate-format",
        choices=CANDIDATE_FORMATS,
        default="official-label-role",
    )
    parser.add_argument(
        "--query-format",
        choices=QUERY_FORMATS,
        default="qwen-instruct",
    )
    parser.add_argument(
        "--allow-trained-checkpoint",
        action="store_true",
        help=(
            "Consente deliberatamente un path con training_config.json. "
            "Il risultato non puo' essere dichiarato task-level zero-shot."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Default: <output senza estensione>.predictions.jsonl",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zero_shot_model_source(model_name_or_path: str, allow_trained_checkpoint: bool) -> None:
    model_path = Path(model_name_or_path).expanduser()
    if not model_path.exists():
        return
    training_markers = [
        model_path / "training_config.json",
        model_path / "candidates.json",
        model_path / "trainer_state.json",
    ]
    present = [path.name for path in training_markers if path.exists()]
    if present and not allow_trained_checkpoint:
        raise ValueError(
            "Il modello locale contiene marker di training ESAW "
            f"({', '.join(present)}). Usa un backbone originale oppure passa "
            "--allow-trained-checkpoint sapendo che il risultato non e' zero-shot."
        )


def complete_target_records(records: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        record
        for record in records
        if all(record_value(record, target.field) for target in TARGET_FIELDS)
    ]


def render_candidate(candidate: Candidate, target: TargetField, candidate_format: str) -> str:
    parts = [
        f"Classe ESAW per {target.title}: {candidate.label}.",
    ]
    if candidate_format in ("official-label-role", "official-label-role-path"):
        parts.append(f"Ruolo della variabile: {target.role}.")
    if candidate_format == "official-label-role-path" and candidate.parent_label:
        parts.append(f"Macroclasse ufficiale: {candidate.parent_label}.")
    return " ".join(parts)


def zero_shot_candidates(
    taxonomy: Mapping[str, Mapping[str, str]],
    candidate_format: str,
) -> Dict[str, List[Candidate]]:
    # legacy creates exactly one candidate per official taxonomy entry and does
    # not inject the v7 hand-written prototypes.
    base = build_candidates(build_explicit_taxonomy(taxonomy), text_format="legacy")
    rendered: Dict[str, List[Candidate]] = {}
    for target in TARGET_FIELDS:
        rendered[target.field] = [
            replace(candidate, text=render_candidate(candidate, target, candidate_format))
            for candidate in base[target.field]
        ]
    return rendered


def render_query(record: Mapping[str, Any], target: TargetField, query_format: str) -> str:
    if query_format == "qwen-instruct":
        return query_text(record, target, "v6")
    return (
        f"Narrativa di infortunio: {accident_to_text(record)}. "
        f"Variabile ESAW richiesta: {target.title}. "
        f"Ruolo della variabile: {target.role}."
    )


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> tuple[float, int]:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0, 0
    scores: List[float] = []
    for label in labels:
        true_positive = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        false_positive = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        false_negative = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(scores)), len(labels)


def mean_metric(fields: Mapping[str, Mapping[str, float]], metric: str) -> float:
    values = [float(field[metric]) for field in fields.values()]
    return float(np.mean(values)) if values else 0.0


def resolved_model_revision(model: SentenceTransformer) -> str | None:
    try:
        return getattr(model[0].auto_model.config, "_commit_hash", None)
    except (AttributeError, IndexError, TypeError):
        return None


def predictions_path_for(output: str | Path, requested: str | None) -> Path:
    if requested:
        return Path(requested)
    output_path = Path(output)
    if output_path.suffix:
        return output_path.with_suffix("").with_name(output_path.stem + ".predictions.jsonl")
    return output_path.with_name(output_path.name + ".predictions.jsonl")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    validate_zero_shot_model_source(args.model, args.allow_trained_checkpoint)

    rng = random.Random(args.seed)
    all_records = load_dataset(args.dataset)
    split_records = filter_records_by_split(all_records, args.split_file, args.split)
    records = complete_target_records(split_records)
    excluded_incomplete = len(split_records) - len(records)
    if args.sample_size and len(records) > args.sample_size:
        records = rng.sample(records, args.sample_size)
    if not records:
        raise ValueError("Nessun record completo disponibile per il benchmark.")

    model_kwargs: Dict[str, Any] = {
        "device": args.device,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision
    model = SentenceTransformer(args.model, **model_kwargs)
    model.max_seq_length = args.max_seq_length
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    resolved_revision = resolved_model_revision(model)
    if trainable_parameters:
        raise RuntimeError("Il congelamento del backbone zero-shot non e' riuscito.")

    candidates = zero_shot_candidates(load_taxonomy(args.taxonomy), args.candidate_format)
    record_keys = [record_split_key(record, index) for index, record in enumerate(records)]
    prediction_rows: List[Dict[str, Any]] = [
        {"record_key": key, "fields": {}} for key in record_keys
    ]
    field_metrics: Dict[str, Dict[str, float | int]] = {}
    record_hits = {
        1: np.ones(len(records), dtype=bool),
        5: np.ones(len(records), dtype=bool),
        10: np.ones(len(records), dtype=bool),
    }
    correct_fields_at_1 = np.zeros(len(records), dtype=int)

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
            [render_query(record, target, args.query_format) for record in records],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        scores = cosine_matrix(query_embeddings, candidate_embeddings)

        hits = {1: 0, 5: 0, 10: 0}
        gold_codes: List[str] = []
        predicted_top1: List[str] = []
        candidate_codes = {candidate.code for candidate in field_candidates}
        gold_missing_from_inventory = 0

        for row_index, record in enumerate(records):
            gold_code = label_to_output_code(target, record_value(record, target.field))
            ranked = top_candidate_details(scores[row_index], field_candidates, 10)
            ranked_codes = [candidate.code for candidate, _ in ranked]
            gold_codes.append(gold_code)
            predicted_top1.append(ranked_codes[0] if ranked_codes else "")
            gold_missing_from_inventory += int(gold_code not in candidate_codes)

            for top_k in hits:
                hit = gold_code in ranked_codes[:top_k]
                hits[top_k] += int(hit)
                record_hits[top_k][row_index] &= hit
            correct_fields_at_1[row_index] += int(bool(ranked_codes) and ranked_codes[0] == gold_code)

            prediction_rows[row_index]["fields"][target.output_field] = {
                "gold": gold_code,
                "top10": [
                    {
                        "rank": rank,
                        "code": candidate.code,
                        "label": candidate.label,
                        "score": float(score),
                    }
                    for rank, (candidate, score) in enumerate(ranked, start=1)
                ],
            }

        total = len(records)
        field_macro_f1, macro_labels = macro_f1(gold_codes, predicted_top1)
        field_metrics[target.output_field] = {
            "records": total,
            "candidate_classes": len(candidate_codes),
            "gold_missing_from_candidate_inventory": gold_missing_from_inventory,
            "top1_hits": hits[1],
            "top5_hits": hits[5],
            "top10_hits": hits[10],
            "top1_accuracy": hits[1] / total,
            "top5_accuracy": hits[5] / total,
            "top10_accuracy": hits[10] / total,
            "macro_f1": field_macro_f1,
            "macro_f1_labels": macro_labels,
        }

    predictions_path = predictions_path_for(args.output, args.predictions_output)
    write_jsonl(predictions_path, prediction_rows)

    output_path = Path(args.output)
    payload: Dict[str, Any] = {
        "protocol": {
            "name": PROTOCOL_NAME,
            "claim_scope": "task-level zero-shot; not controlled label-wise zero-shot",
            "task_specific_esaw_weight_updates": False,
            "backbone_frozen": True,
            "trainable_parameters_at_inference": trainable_parameters,
            "total_parameters": total_parameters,
            "candidate_source": "official ESAW taxonomy; rebuilt independently of trained checkpoints",
            "candidate_format": args.candidate_format,
            "query_format": args.query_format,
        },
        "run": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "model": args.model,
            "model_revision_requested": args.model_revision,
            "model_revision_resolved": resolved_revision,
            "device": str(model.device),
            "max_seq_length": args.max_seq_length,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "split": args.split,
            "sample_size_requested": args.sample_size,
            "records": len(records),
            "incomplete_records_excluded_before_sampling": excluded_incomplete,
            "predictions": str(predictions_path),
        },
        "input_checksums_sha256": {
            "dataset": sha256_file(args.dataset),
            "taxonomy": sha256_file(args.taxonomy),
            "split_file": sha256_file(args.split_file),
        },
        "mean_field_metrics": {
            metric: mean_metric(field_metrics, metric)
            for metric in ("top1_accuracy", "top5_accuracy", "top10_accuracy", "macro_f1")
        },
        "record_metrics": {
            "exact_8_of_8_at_1": float(record_hits[1].mean()),
            "at_least_7_of_8_at_1": float((correct_fields_at_1 >= 7).mean()),
            "at_least_6_of_8_at_1": float((correct_fields_at_1 >= 6).mean()),
            "recall_8_of_8_at_5": float(record_hits[5].mean()),
            "recall_8_of_8_at_10": float(record_hits[10].mean()),
            "exact_8_of_8_hits": int(record_hits[1].sum()),
            "recall_8_of_8_at_5_hits": int(record_hits[5].sum()),
            "recall_8_of_8_at_10_hits": int(record_hits[10].sum()),
        },
        "field_metrics": field_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
