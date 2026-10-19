from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_candidates, build_explicit_taxonomy, dataclass_to_dict, load_taxonomy, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rende esplicita la tassonomia ESAW e genera i candidati del bi-encoder.")
    parser.add_argument("--taxonomy", default="tassonomia.json", help="Path del file tassonomia.json.")
    parser.add_argument("--out-dir", default="artifacts", help="Cartella di output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_taxonomy = load_taxonomy(args.taxonomy)
    explicit = build_explicit_taxonomy(raw_taxonomy)
    candidates = build_candidates(explicit)

    write_json(
        out_dir / "explicit_taxonomy.json",
        {name: [dataclass_to_dict(entry) for entry in entries] for name, entries in explicit.items()},
    )
    write_json(
        out_dir / "candidates.json",
        {field: [dataclass_to_dict(candidate) for candidate in field_candidates] for field, field_candidates in candidates.items()},
    )
    print(f"Scritti {out_dir / 'explicit_taxonomy.json'} e {out_dir / 'candidates.json'}")


if __name__ == "__main__":
    main()
