#!/usr/bin/env python3
"""Join cross-encoder rankings with the full ESAW candidate metadata."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from esaw_biencoder.data import build_candidates, build_explicit_taxonomy, load_taxonomy
from esaw_biencoder.schema import TARGET_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-lists", required=True, type=Path)
    parser.add_argument("--cross-predictions", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_by_output = {target.output_field: target for target in TARGET_FIELDS}

    taxonomy_candidates = build_candidates(
        build_explicit_taxonomy(load_taxonomy(args.taxonomy)),
        text_format="v5",
    )
    taxonomy_by_field_code = {
        target.output_field: {
            candidate.code: asdict(candidate)
            for candidate in taxonomy_candidates[target.field]
            if candidate.prototype_id == "base"
        }
        for target in TARGET_FIELDS
    }

    retrieval_by_pair: dict[tuple[str, str], dict[str, dict]] = {}
    with args.retrieval_lists.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["record_key"]), str(row["field"]))
            retrieval_by_pair[key] = {
                str(item["candidate"]["code"]): dict(item["candidate"])
                for item in row["candidates"]
            }

    output_rows = 0
    taxonomy_fallbacks = 0
    seen_pairs: set[tuple[str, str]] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.cross_predictions.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            record_key = str(row["record_key"])
            field = str(row["field"])
            pair = (record_key, field)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate cross prediction: {pair}")
            seen_pairs.add(pair)
            if field not in target_by_output:
                raise ValueError(f"Unknown ESAW field: {field}")

            ranked_candidates = []
            for rank, code_value in enumerate(row["rerank_top10_codes"][: args.top_k], start=1):
                code = str(code_value)
                candidate = retrieval_by_pair.get(pair, {}).get(code)
                if candidate is None:
                    candidate = taxonomy_by_field_code[field].get(code)
                    taxonomy_fallbacks += 1
                if candidate is None:
                    raise ValueError(f"Candidate metadata unavailable for {pair}, code={code}")
                ranked_candidates.append(
                    {
                        "rank": rank,
                        "rerank_score": float(args.top_k - rank + 1),
                        "candidate": candidate,
                    }
                )

            destination.write(
                json.dumps(
                    {
                        "record_key": record_key,
                        "field": field,
                        "rerank_top5_candidates": ranked_candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_rows += 1

    expected_pairs = 5000 * len(TARGET_FIELDS)
    if output_rows != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} record/field pairs, found {output_rows}")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": output_rows,
                "records": output_rows // len(TARGET_FIELDS),
                "top_k": args.top_k,
                "taxonomy_fallbacks": taxonomy_fallbacks,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
