from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .data import Candidate, TEXT_FORMATS, build_candidates, build_explicit_taxonomy, load_taxonomy, query_text
from .schema import TARGET_FIELDS

TEXT_FORMAT_AUTO = "auto"


@dataclass(frozen=True)
class Prediction:
    field: str
    classe: str
    score: float
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predice le 8 variabili ESAW con un bi-encoder addestrato.")
    parser.add_argument("--model", default="models/esaw-biencoder", help="Path del modello SentenceTransformer.")
    parser.add_argument("--taxonomy", default="tassonomia.json", help="Path tassonomia, usata se mancano i candidati nel modello.")
    parser.add_argument("--input", default="-", help="File JSON con un infortunio. Usa '-' per stdin.")
    parser.add_argument("--top-k", type=int, default=1, help="Numero di classi da restituire per variabile.")
    parser.add_argument("--explain", action="store_true", help="Include score e label descrittiva nella risposta.")
    parser.add_argument(
        "--retrieval",
        choices=["flat", "hierarchical"],
        default="flat",
        help="'flat' cerca tra tutte le classi; 'hierarchical' predice prima la macroclasse e poi cerca nei figli.",
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


def load_input(path: str) -> Dict[str, Any]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("L'input deve essere un singolo oggetto JSON.")
    return payload


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

    raw_taxonomy = load_taxonomy(taxonomy_path)
    return build_candidates(build_explicit_taxonomy(raw_taxonomy), text_format=text_format)


def cosine_scores(query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    query = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)
    candidates = candidate_embeddings / np.maximum(np.linalg.norm(candidate_embeddings, axis=1, keepdims=True), 1e-12)
    return candidates @ query


def hierarchical_best_indexes(
    scores: np.ndarray,
    candidates: Sequence[Candidate],
    top_k: int,
    hierarchy_beam_size: int,
) -> List[int]:
    macro_indexes = [index for index, candidate in enumerate(candidates) if candidate.parent_code is None]
    if not macro_indexes:
        return [int(index) for index in np.argsort(-scores)[:top_k]]

    macro_order = np.argsort(-scores[macro_indexes])[: max(1, hierarchy_beam_size)]
    macro_codes = [candidates[macro_indexes[int(local_index)]].taxonomy_code for local_index in macro_order]
    scope = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.taxonomy_code in macro_codes
        or candidate.parent_code in macro_codes
        or any(ancestor in macro_codes for ancestor in candidate.ancestors)
    ]
    if not scope:
        scope = list(range(len(candidates)))
    return sorted(scope, key=lambda index: float(scores[index]), reverse=True)[:top_k]


def predict_record(
    model: SentenceTransformer,
    record: Mapping[str, Any],
    candidates: Mapping[str, Sequence[Candidate]],
    top_k: int,
    retrieval: str,
    hierarchy_beam_size: int,
    text_format: str,
) -> Dict[str, List[Prediction]]:
    predictions: Dict[str, List[Prediction]] = {}
    for target in TARGET_FIELDS:
        field_candidates = list(candidates[target.field])
        candidate_embeddings = model.encode(
            [candidate.text for candidate in field_candidates],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        query_embedding = model.encode(
            query_text(record, target, text_format),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        scores = cosine_scores(query_embedding, candidate_embeddings)
        if retrieval == "hierarchical":
            best_indexes = hierarchical_best_indexes(scores, field_candidates, top_k, hierarchy_beam_size)
        else:
            best_indexes = [int(index) for index in np.argsort(-scores)[:top_k]]
        predictions[target.field] = [
            Prediction(
                field=target.output_field,
                classe=field_candidates[index].code,
                score=float(scores[index]),
                label=field_candidates[index].label,
            )
            for index in best_indexes
        ]
    return predictions


def response_json(predictions: Mapping[str, Sequence[Prediction]], explain: bool, top_k: int) -> Any:
    if top_k == 1:
        rows = [items[0] for items in predictions.values()]
        if explain:
            return [
                {"field": row.field, "classe": row.classe, "score": round(row.score, 6), "label": row.label}
                for row in rows
            ]
        return [{"field": row.field, "classe": row.classe} for row in rows]

    payload = []
    for items in predictions.values():
        payload.append(
            {
                "field": items[0].field,
                "candidates": [
                    (
                        {"classe": row.classe, "score": round(row.score, 6), "label": row.label}
                        if explain
                        else {"classe": row.classe}
                    )
                    for row in items
                ],
            }
        )
    return payload


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model)
    record = load_input(args.input)
    text_format = resolve_text_format(model_dir, args.text_format)
    model = SentenceTransformer(str(model_dir))
    candidates = load_candidates(model_dir, args.taxonomy, text_format)
    predictions = predict_record(
        model,
        record,
        candidates,
        max(1, args.top_k),
        args.retrieval,
        max(1, args.hierarchy_beam_size),
        text_format,
    )
    print(json.dumps(response_json(predictions, args.explain, max(1, args.top_k)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
