#!/usr/bin/env python3
"""Prepare the fixed ESAW split for paper-faithful HTC reproductions.

ESAW is represented as one multi-path hierarchy.  Every output field is a
root-level role; the corresponding taxonomy groups and codes are the second
and third levels.  Agent-material nodes are duplicated by role because the
same code has a different semantic function in activity, deviation, and
contact.  This preserves the eight simultaneous decisions while satisfying
the tree assumption made by DPT and HTC-CLIP.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


INPUT_FIELDS = (
    "luogo_infortunio",
    "lavorazione",
    "dettaglio_lavorazione",
    "imprevisto",
    "conseguenza",
    "des_infortunio",
)


@dataclass(frozen=True)
class Target:
    field: str
    taxonomy: str
    role: str


TARGETS = (
    Target("cod_loc", "tipo_luogo", "COD_LOC"),
    Target("cod_lav", "tipo_lavoro", "COD_LAV"),
    Target("cod_atv", "attivita_fisica_specifica", "COD_ATV"),
    Target("cod_mat_atv", "agente_materiale", "COD_MAT_ATV"),
    Target("cod_dev", "deviazione", "COD_DEV"),
    Target("cod_mat_dev", "agente_materiale", "COD_MAT_DEV"),
    Target("cod_con", "contatto_modalita_lesione", "COD_CON"),
    Target("cod_mat_con", "agente_materiale", "COD_MAT_CON"),
)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, list) and results and isinstance(results[0].get("items"), list):
        return results[0]["items"]
    raise ValueError(f"Unsupported dataset format: {path}")


def value(record: Mapping[str, Any], field: str) -> str:
    raw = record.get(field, record.get(field.upper(), ""))
    return " ".join(str(raw or "").replace("\n", " ").split()).strip()


def split_key(record: Mapping[str, Any], index: int) -> str:
    for field in ("id", "numero_caso"):
        raw = record.get(field, record.get(field.upper()))
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return f"index:{index}"


def output_code(target: Target, raw: str) -> str:
    if target.taxonomy != "agente_materiale":
        return raw
    parts = raw.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}.00.00"
    return raw


def taxonomy_code(target: Target, raw: str) -> str:
    if target.taxonomy == "agente_materiale":
        return ".".join(raw.split(".")[:2])
    return raw


def parent_code(target: Target, code: str, entries: Mapping[str, str]) -> str | None:
    if target.taxonomy == "agente_materiale":
        parent = f"{code.split('.')[0]}.00"
    elif len(code) == 3:
        parent = f"{code[:2]}0"
    elif len(code) == 2:
        parent = f"{code[0]}0"
    else:
        return None
    return parent if parent in entries and parent != code else None


def accident_text(record: Mapping[str, Any]) -> str:
    labels = {
        "luogo_infortunio": "Luogo infortunio",
        "lavorazione": "Lavorazione",
        "dettaglio_lavorazione": "Dettaglio lavorazione",
        "imprevisto": "Imprevisto",
        "conseguenza": "Conseguenza",
        "des_infortunio": "Descrizione infortunio",
    }
    parts = [f"{labels[field]}: {value(record, field)}" for field in INPUT_FIELDS if value(record, field)]
    return ". ".join(parts)


def role_node(role: str) -> str:
    return f"ROLE::{role}"


def group_node(role: str, code: str) -> str:
    return f"GROUP::{role}::{code}"


def leaf_node(role: str, code: str) -> str:
    return f"LEAF::{role}::{code}"


def make_tree(taxonomy: Mapping[str, Mapping[str, str]]) -> tuple[list[str], dict[str, str], dict[str, list[str]], dict[str, list[str]], dict[str, dict[str, str]]]:
    labels: list[str] = []
    display: dict[str, str] = {}
    children: dict[str, list[str]] = defaultdict(list)
    field_leaves: dict[str, list[str]] = {}
    output_by_leaf: dict[str, dict[str, str]] = {}

    def add(name: str, text: str) -> None:
        if name not in display:
            labels.append(name)
            display[name] = text

    for target in TARGETS:
        role = role_node(target.role)
        add(role, target.role)
        entries = taxonomy[target.taxonomy]
        leaves: list[str] = []
        leaf_to_code: dict[str, str] = {}
        for code, description in sorted(entries.items()):
            parent = parent_code(target, code, entries)
            if parent is not None:
                parent_name = group_node(target.role, parent)
                add(parent_name, f"{target.role} {entries[parent]}")
                if parent_name not in children[role]:
                    children[role].append(parent_name)
                leaf = leaf_node(target.role, code)
                add(leaf, f"{target.role} {description}")
                children[parent_name].append(leaf)
            else:
                leaf = group_node(target.role, code)
                add(leaf, f"{target.role} {description}")
                if leaf not in children[role]:
                    children[role].append(leaf)
            leaves.append(leaf)
            leaf_to_code[leaf] = output_code(target, code)
        field_leaves[target.role] = leaves
        output_by_leaf[target.role] = leaf_to_code
    return labels, display, dict(children), field_leaves, output_by_leaf


def paths_for_record(record: Mapping[str, Any], taxonomy: Mapping[str, Mapping[str, str]]) -> list[str]:
    out: list[str] = []
    for target in TARGETS:
        raw = value(record, target.field)
        if not raw:
            continue
        code = taxonomy_code(target, raw)
        entries = taxonomy[target.taxonomy]
        if code not in entries:
            continue
        role = role_node(target.role)
        out.append(role)
        parent = parent_code(target, code, entries)
        if parent is not None:
            out.append(group_node(target.role, parent))
            out.append(leaf_node(target.role, code))
        else:
            out.append(group_node(target.role, code))
    return out


def ordered_unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_dataset(args.dataset)
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    split_payload = json.loads(args.split_file.read_text(encoding="utf-8"))
    split_ids = split_payload.get("splits", split_payload)
    record_by_key = {split_key(record, i): record for i, record in enumerate(records)}

    labels, display, children, field_leaves, output_by_leaf = make_tree(taxonomy)
    label_to_id = {label: i for i, label in enumerate(labels)}
    args.output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "representation": "single multi-path tree with eight ESAW role roots",
        "labels": labels,
        "display": display,
        "children": children,
        "field_leaves": field_leaves,
        "output_by_leaf": output_by_leaf,
        "label_to_id": label_to_id,
        "source_split": str(args.split_file.resolve()),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        keys = split_ids[split]
        target_name = "dev" if split == "validation" else split
        path = args.output / f"esaw_{target_name}.json"
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for key in keys:
                record = record_by_key.get(str(key))
                if record is None:
                    continue
                label_names = ordered_unique(paths_for_record(record, taxonomy))
                row = {
                    "record_key": str(key),
                    "token": accident_text(record),
                    "label": [label_to_id[name] for name in label_names],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        counts[split] = count

    manifest = {
        "dataset": str(args.dataset.resolve()),
        "taxonomy": str(args.taxonomy.resolve()),
        "split_file": str(args.split_file.resolve()),
        "counts": counts,
        "num_labels": len(labels),
        "num_role_roots": len(TARGETS),
        "max_depth": 3,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
