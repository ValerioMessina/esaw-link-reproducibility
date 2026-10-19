from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from esaw_biencoder.data import (
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
from esaw_biencoder.evaluate import (
    cosine_matrix,
    filter_records_by_split,
    load_candidates,
    record_split_key,
    resolve_text_format,
)
from esaw_biencoder.label_holdout import (
    PROTOCOL_NAME,
    load_holdout_fold,
    sha256_file,
    validate_manifest_sources,
)
from esaw_biencoder.retrieval import top_candidate_codes
from esaw_biencoder.schema import TARGET_FIELDS


TEXT_FORMAT_AUTO = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valuta generalized e conventional label-wise zero-shot ESAW su classi "
            "pre-registrate, separando decisioni seen/unseen."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--taxonomy", default="tassonomia.json")
    parser.add_argument("--split-file", default="artifacts/splits/esaw-fixed-seed13-test10-val10.json")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--label-holdout-file", required=True)
    parser.add_argument("--label-holdout-fold", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--text-format", choices=(TEXT_FORMAT_AUTO, *TEXT_FORMATS), default=TEXT_FORMAT_AUTO)
    parser.add_argument("--rebuild-candidates", action="store_true")
    parser.add_argument("--allow-protocol-mismatch", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output", default=None)
    return parser.parse_args()


def wilson_interval(hits: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = hits / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def macro_f1_for_labels(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(true == label and predicted == label for true, predicted in zip(y_true, y_pred))
        fp = sum(true != label and predicted == label for true, predicted in zip(y_true, y_pred))
        fn = sum(true == label and predicted != label for true, predicted in zip(y_true, y_pred))
        denominator = 2 * tp + fp + fn
        if tp + fn > 0:
            scores.append((2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def topk_summary(rows: Sequence[Mapping[str, Any]], prefix: str = "generalized") -> Dict[str, Any]:
    total = len(rows)
    result: Dict[str, Any] = {"decisions": total}
    for k in (1, 5, 10):
        key = f"{prefix}_top{k}_hit"
        hits = sum(bool(row[key]) for row in rows)
        result[f"top{k}_hits"] = hits
        result[f"top{k}_accuracy"] = hits / total if total else 0.0
        if k == 1:
            result["top1_wilson_95ci"] = wilson_interval(hits, total)
    return result


def output_path_for_predictions(output: str, requested: str | None) -> Path:
    if requested:
        return Path(requested)
    path = Path(output)
    return path.with_suffix("").with_name(path.stem + ".predictions.jsonl")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_model_protocol(model_path: str, manifest_sha: str, fold: int, allow_mismatch: bool) -> Dict[str, Any]:
    config_path = Path(model_path) / "training_config.json"
    if not config_path.exists():
        return {"training_config_present": False, "protocol_match": None}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trained = config.get("label_holdout") or {}
    match = trained.get("manifest_sha256") == manifest_sha and trained.get("fold") == fold
    if trained and not match and not allow_mismatch:
        raise ValueError(
            "Checkpoint addestrato con manifest/fold differenti: "
            f"checkpoint={trained.get('manifest_sha256')} fold={trained.get('fold')}, "
            f"benchmark={manifest_sha} fold={fold}."
        )
    return {
        "training_config_present": True,
        "checkpoint_declares_label_holdout": bool(trained),
        "protocol_match": match if trained else None,
        "training_config": str(config_path),
    }


def main() -> None:
    args = parse_args()
    heldout, manifest = load_holdout_fold(args.label_holdout_file, args.label_holdout_fold)
    validate_manifest_sources(manifest, args.dataset, args.taxonomy, args.split_file)
    manifest_sha = sha256_file(args.label_holdout_file)
    protocol_check = verify_model_protocol(
        args.model, manifest_sha, args.label_holdout_fold, args.allow_protocol_mismatch
    )

    records = filter_records_by_split(load_dataset(args.dataset), args.split_file, args.split)
    if args.sample_size:
        records = records[: args.sample_size]
    if not records:
        raise ValueError("Split di benchmark vuoto.")

    model_path = Path(args.model)
    text_format = (
        resolve_text_format(model_path, args.text_format)
        if args.text_format == TEXT_FORMAT_AUTO
        else args.text_format
    )
    model_kwargs: Dict[str, Any] = {"device": args.device}
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision
    model = SentenceTransformer(args.model, **model_kwargs)
    model.max_seq_length = args.max_seq_length
    model.eval()

    if args.rebuild_candidates or not (model_path / "candidates.json").exists():
        candidates = build_candidates(
            build_explicit_taxonomy(load_taxonomy(args.taxonomy)), text_format=text_format
        )
    else:
        candidates = load_candidates(model_path, args.taxonomy, text_format)

    prediction_rows: list[Dict[str, Any]] = [
        {"record_key": record_split_key(record, index), "fields": {}}
        for index, record in enumerate(records)
    ]
    all_decisions: list[Dict[str, Any]] = []
    field_metrics: Dict[str, Any] = {}

    for target in TARGET_FIELDS:
        field_candidates = candidates[target.field]
        unseen_candidates = [
            candidate
            for candidate in field_candidates
            if candidate.taxonomy_code in heldout[target.output_field]
        ]
        present_unseen = {candidate.taxonomy_code for candidate in unseen_candidates}
        missing = heldout[target.output_field] - present_unseen
        if missing:
            raise ValueError(f"Classi held-out mancanti nel candidate pool {target.output_field}: {sorted(missing)}")

        usable = [
            (index, record)
            for index, record in enumerate(records)
            if label_to_taxonomy_code(target, record_value(record, target.field))
        ]
        candidate_embeddings = model.encode(
            [candidate.text for candidate in field_candidates],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        query_embeddings = model.encode(
            [query_text(record, target, text_format) for _, record in usable],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        scores = cosine_matrix(query_embeddings, candidate_embeddings)
        unseen_indices = [
            index
            for index, candidate in enumerate(field_candidates)
            if candidate.taxonomy_code in heldout[target.output_field]
        ]
        field_rows: list[Dict[str, Any]] = []
        candidate_by_output = {candidate.code: candidate for candidate in field_candidates}

        for row_index, (record_index, record) in enumerate(usable):
            gold_taxonomy_code = label_to_taxonomy_code(target, record_value(record, target.field))
            gold_output_code = label_to_output_code(target, record_value(record, target.field))
            generalized_codes = top_candidate_codes(scores[row_index], field_candidates, 10)
            restricted_codes = top_candidate_codes(
                scores[row_index, unseen_indices], unseen_candidates, 10
            )
            is_unseen = gold_taxonomy_code in heldout[target.output_field]
            decision: Dict[str, Any] = {
                "record_index": record_index,
                "output_field": target.output_field,
                "gold_taxonomy_code": gold_taxonomy_code,
                "gold_output_code": gold_output_code,
                "is_unseen": is_unseen,
                "generalized_top10": generalized_codes,
                "restricted_unseen_top10": restricted_codes,
            }
            for k in (1, 5, 10):
                decision[f"generalized_top{k}_hit"] = gold_output_code in generalized_codes[:k]
                decision[f"restricted_top{k}_hit"] = is_unseen and gold_output_code in restricted_codes[:k]
            predicted_candidate = candidate_by_output.get(generalized_codes[0]) if generalized_codes else None
            gold_candidate = candidate_by_output.get(gold_output_code)
            decision["generalized_parent_hit"] = bool(
                is_unseen
                and predicted_candidate
                and gold_candidate
                and predicted_candidate.parent_code == gold_candidate.parent_code
            )
            field_rows.append(decision)
            all_decisions.append(decision)
            prediction_rows[record_index]["fields"][target.output_field] = {
                key: value for key, value in decision.items() if key not in {"record_index", "output_field"}
            }

        seen_rows = [row for row in field_rows if not row["is_unseen"]]
        unseen_rows = [row for row in field_rows if row["is_unseen"]]
        seen_summary = topk_summary(seen_rows)
        unseen_generalized = topk_summary(unseen_rows)
        unseen_restricted = topk_summary(unseen_rows, prefix="restricted")
        unseen_true = [row["gold_output_code"] for row in unseen_rows]
        unseen_pred = [row["generalized_top10"][0] for row in unseen_rows if row["generalized_top10"]]
        if len(unseen_pred) != len(unseen_true):
            raise RuntimeError("Predizione top-1 mancante.")
        label_outputs = sorted({candidate.code for candidate in unseen_candidates})
        seen_top1 = seen_summary["top1_accuracy"]
        unseen_top1 = unseen_generalized["top1_accuracy"]
        harmonic = 2 * seen_top1 * unseen_top1 / (seen_top1 + unseen_top1) if seen_top1 + unseen_top1 else 0.0
        field_metrics[target.output_field] = {
            "heldout_taxonomy_codes": sorted(heldout[target.output_field]),
            "heldout_candidate_count": len(unseen_candidates),
            "seen_generalized": seen_summary,
            "unseen_generalized": {
                **unseen_generalized,
                "macro_f1_over_supported_heldout_labels": macro_f1_for_labels(
                    unseen_true, unseen_pred, label_outputs
                ),
                "parent_accuracy": (
                    sum(row["generalized_parent_hit"] for row in unseen_rows) / len(unseen_rows)
                    if unseen_rows else 0.0
                ),
            },
            "unseen_restricted": unseen_restricted,
            "seen_unseen_harmonic_top1": harmonic,
        }

    seen_all = [row for row in all_decisions if not row["is_unseen"]]
    unseen_all = [row for row in all_decisions if row["is_unseen"]]
    pooled_seen = topk_summary(seen_all)
    pooled_unseen = topk_summary(unseen_all)
    pooled_restricted = topk_summary(unseen_all, prefix="restricted")
    denominator = pooled_seen["top1_accuracy"] + pooled_unseen["top1_accuracy"]
    pooled_harmonic = (
        2 * pooled_seen["top1_accuracy"] * pooled_unseen["top1_accuracy"] / denominator
        if denominator else 0.0
    )

    records_with_unseen = [
        row for row in prediction_rows if any(field["is_unseen"] for field in row["fields"].values())
    ]
    records_all_unseen_correct = sum(
        all(
            (not field["is_unseen"]) or field["generalized_top1_hit"]
            for field in row["fields"].values()
        )
        for row in records_with_unseen
    )

    payload = {
        "protocol": PROTOCOL_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision_requested": args.model_revision,
        "model_protocol_check": protocol_check,
        "dataset": args.dataset,
        "dataset_sha256": sha256_file(args.dataset),
        "taxonomy": args.taxonomy,
        "taxonomy_sha256": sha256_file(args.taxonomy),
        "record_split_file": args.split_file,
        "record_split_sha256": sha256_file(args.split_file),
        "split": args.split,
        "records": len(records),
        "text_format": text_format,
        "max_seq_length": args.max_seq_length,
        "label_holdout_file": args.label_holdout_file,
        "label_holdout_manifest_sha256": manifest_sha,
        "label_holdout_fold": args.label_holdout_fold,
        "evaluation_regimes": {
            "generalized": "rank among all seen and unseen taxonomy classes",
            "restricted": "rank only among held-out taxonomy classes; reported only for unseen gold decisions",
        },
        "pooled_decision_metrics": {
            "seen_generalized": pooled_seen,
            "unseen_generalized": pooled_unseen,
            "unseen_restricted": pooled_restricted,
            "seen_unseen_harmonic_top1": pooled_harmonic,
        },
        "record_metrics": {
            "records_with_at_least_one_unseen_gold": len(records_with_unseen),
            "all_unseen_decisions_top1_correct_hits": records_all_unseen_correct,
            "all_unseen_decisions_top1_correct_accuracy": (
                records_all_unseen_correct / len(records_with_unseen) if records_with_unseen else 0.0
            ),
        },
        "field_metrics": field_metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output_path_for_predictions(args.output, args.predictions_output)
    write_jsonl(predictions_path, prediction_rows)
    payload["predictions_output"] = str(predictions_path)
    payload["predictions_sha256"] = sha256_file(predictions_path)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
