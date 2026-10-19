from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .data import Candidate


def top_candidate_codes(
    row_scores: np.ndarray,
    candidates: Sequence[Candidate],
    top_k: int,
) -> List[str]:
    best_by_code: Dict[str, float] = {}
    for index, candidate in enumerate(candidates):
        score = float(row_scores[index])
        current = best_by_code.get(candidate.code)
        if current is None or score > current:
            best_by_code[candidate.code] = score
    ordered = sorted(best_by_code.items(), key=lambda item: (-item[1], item[0]))
    return [code for code, _ in ordered[:top_k]]


def top_candidate_details(
    row_scores: np.ndarray,
    candidates: Sequence[Candidate],
    top_k: int,
) -> List[tuple[Candidate, float]]:
    best_by_code: Dict[str, tuple[Candidate, float]] = {}
    for index, candidate in enumerate(candidates):
        score = float(row_scores[index])
        current = best_by_code.get(candidate.code)
        if current is None or score > current[1]:
            best_by_code[candidate.code] = (candidate, score)
    ordered = sorted(best_by_code.values(), key=lambda item: (-item[1], item[0].code))
    return ordered[:top_k]
