from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .data import (
    Candidate,
    label_to_taxonomy_code,
    load_dataset,
    load_taxonomy,
    parent_code_for,
    record_value,
)
from .evaluate import filter_records_by_split
from .schema import TARGET_FIELDS


PROTOCOL_NAME = "esaw-generalized-label-wise-zero-shot-v1"
MATERIAL_OUTPUT_FIELDS = ("COD_MAT_ATV", "COD_MAT_DEV", "COD_MAT_CON")
ADMINISTRATIVE_LABEL_MARKERS = (
    "nessuna informazione",
    "nessun agente",
    "non precisat",
    "non specificat",
    "altro agente",
    "altro tipo",
    "altra attivita",
    "altra attività",
    "altra deviazione",
    "altro contatto",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def leaf_codes(taxonomy_name: str, taxonomy: Mapping[str, str]) -> set[str]:
    parents = {
        parent
        for code in taxonomy
        if (parent := parent_code_for(taxonomy_name, code, taxonomy)) is not None
    }
    return set(taxonomy) - parents


def is_administrative_label(label: str) -> bool:
    normalized = " ".join(label.lower().split())
    return any(marker in normalized for marker in ADMINISTRATIVE_LABEL_MARKERS)


def training_support(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Counter[str]]:
    support: Dict[str, Counter[str]] = {
        target.output_field: Counter() for target in TARGET_FIELDS
    }
    for record in records:
        for target in TARGET_FIELDS:
            code = label_to_taxonomy_code(target, record_value(record, target.field))
            if code:
                support[target.output_field][code] += 1
    return support


def stratified_fold_assignment(
    codes: Iterable[str],
    taxonomy_name: str,
    taxonomy: Mapping[str, str],
    folds: int,
    seed: int,
) -> Dict[str, int]:
    """Assign codes without using validation/test outcomes.

    Round-robin is performed independently inside each immediate-parent stratum.
    A deterministic parent-specific offset prevents fold 0 from always receiving
    the first/smallest member of small strata.
    """
    by_parent: Dict[str, list[str]] = defaultdict(list)
    for code in codes:
        parent = parent_code_for(taxonomy_name, code, taxonomy) or "__root__"
        by_parent[parent].append(code)

    assignment: Dict[str, int] = {}
    for parent, members in sorted(by_parent.items()):
        ordered = sorted(members, key=lambda code: stable_hash(seed, taxonomy_name, parent, code))
        offset = int(stable_hash(seed, taxonomy_name, parent)[:8], 16) % folds
        for index, code in enumerate(ordered):
            assignment[code] = (offset + index) % folds
    return assignment


def generate_manifest(
    dataset_path: str | Path,
    taxonomy_path: str | Path,
    split_path: str | Path,
    *,
    seed: int = 13,
    folds: int = 5,
    min_train_support: int = 30,
    min_material_support_per_role: int = 20,
) -> Dict[str, Any]:
    if folds < 2:
        raise ValueError("Servono almeno due fold di etichette.")
    dataset = load_dataset(dataset_path)
    train_records = filter_records_by_split(dataset, split_path, "train")
    taxonomy = load_taxonomy(taxonomy_path)
    support = training_support(train_records)

    groups: Dict[str, Dict[str, Any]] = {}
    assignments_by_output: Dict[str, Dict[str, int]] = {}
    processed_taxonomies: set[str] = set()

    for target in TARGET_FIELDS:
        if target.taxonomy == "agente_materiale":
            continue
        tax = taxonomy[target.taxonomy]
        eligible = sorted(
            code
            for code in leaf_codes(target.taxonomy, tax)
            if support[target.output_field][code] >= min_train_support
            and not is_administrative_label(tax[code])
        )
        assignment = stratified_fold_assignment(eligible, target.taxonomy, tax, folds, seed)
        assignments_by_output[target.output_field] = assignment
        groups[target.output_field] = {
            "taxonomy": target.taxonomy,
            "shared_across_output_fields": [target.output_field],
            "eligible_codes": eligible,
            "train_support": {code: support[target.output_field][code] for code in eligible},
            "fold_by_code": assignment,
        }
        processed_taxonomies.add(target.taxonomy)

    material_tax = taxonomy["agente_materiale"]
    material_eligible = sorted(
        code
        for code in leaf_codes("agente_materiale", material_tax)
        if min(support[field][code] for field in MATERIAL_OUTPUT_FIELDS) >= min_material_support_per_role
        and not is_administrative_label(material_tax[code])
    )
    material_assignment = stratified_fold_assignment(
        material_eligible, "agente_materiale", material_tax, folds, seed
    )
    for output_field in MATERIAL_OUTPUT_FIELDS:
        assignments_by_output[output_field] = dict(material_assignment)
    groups["MATERIAL_SHARED"] = {
        "taxonomy": "agente_materiale",
        "shared_across_output_fields": list(MATERIAL_OUTPUT_FIELDS),
        "eligible_codes": material_eligible,
        "train_support_by_output_field": {
            field: {code: support[field][code] for code in material_eligible}
            for field in MATERIAL_OUTPUT_FIELDS
        },
        "fold_by_code": material_assignment,
    }

    fold_payload: Dict[str, Any] = {}
    for fold in range(folds):
        heldout_by_output = {
            target.output_field: sorted(
                code
                for code, assigned_fold in assignments_by_output[target.output_field].items()
                if assigned_fold == fold
            )
            for target in TARGET_FIELDS
        }
        fold_payload[str(fold)] = {
            "heldout_taxonomy_codes_by_output_field": heldout_by_output,
            "heldout_label_count": sum(len(codes) for codes in heldout_by_output.values()),
            "heldout_unique_material_count": len(heldout_by_output["COD_MAT_ATV"]),
        }

    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": 1,
        "seed": seed,
        "fold_count": folds,
        "primary_fold": 0,
        "selection_uses": "taxonomy structure and train split labels only; validation/test labels and metrics are not used",
        "selection": {
            "leaf_only": True,
            "exclude_administrative_fallback_labels": True,
            "administrative_label_markers": list(ADMINISTRATIVE_LABEL_MARKERS),
            "min_train_support": min_train_support,
            "min_material_support_per_role": min_material_support_per_role,
            "material_codes_are_shared_across_all_three_material_roles": True,
        },
        "sources": {
            "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
            "taxonomy": {"path": str(taxonomy_path), "sha256": sha256_file(taxonomy_path)},
            "record_split": {"path": str(split_path), "sha256": sha256_file(split_path)},
        },
        "train_record_count": len(train_records),
        "groups": groups,
        "folds": fold_payload,
    }


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def load_holdout_fold(path: str | Path, fold: int) -> tuple[Dict[str, set[str]], Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL_NAME:
        raise ValueError(f"Protocollo holdout inatteso: {payload.get('protocol')!r}")
    fold_payload = payload.get("folds", {}).get(str(fold))
    if not isinstance(fold_payload, Mapping):
        raise ValueError(f"Fold {fold} non presente nel manifest.")
    raw = fold_payload.get("heldout_taxonomy_codes_by_output_field", {})
    expected = {target.output_field for target in TARGET_FIELDS}
    if set(raw) != expected:
        raise ValueError(f"Campi holdout mancanti/eccedenti: attesi {sorted(expected)}, trovati {sorted(raw)}")
    heldout = {field: {str(code) for code in codes} for field, codes in raw.items()}
    material_sets = [heldout[field] for field in MATERIAL_OUTPUT_FIELDS]
    if not all(codes == material_sets[0] for codes in material_sets[1:]):
        raise ValueError("Le classi agente materiale devono essere identiche nei tre ruoli.")
    return heldout, payload


def validate_manifest_sources(
    payload: Mapping[str, Any],
    dataset_path: str | Path,
    taxonomy_path: str | Path,
    split_path: str | Path,
) -> None:
    actual = {
        "dataset": sha256_file(dataset_path),
        "taxonomy": sha256_file(taxonomy_path),
        "record_split": sha256_file(split_path),
    }
    for key, digest in actual.items():
        expected = payload.get("sources", {}).get(key, {}).get("sha256")
        if expected != digest:
            raise ValueError(f"SHA-256 {key} non coincide: manifest={expected}, attuale={digest}")


def filter_candidates_for_holdout(
    candidates: Mapping[str, Sequence[Candidate]],
    heldout_by_output_field: Mapping[str, set[str]],
) -> Dict[str, list[Candidate]]:
    output_by_raw_field = {target.field: target.output_field for target in TARGET_FIELDS}
    return {
        raw_field: [
            candidate
            for candidate in field_candidates
            if candidate.taxonomy_code not in heldout_by_output_field[output_by_raw_field[raw_field]]
        ]
        for raw_field, field_candidates in candidates.items()
    }


def audit_training_candidates(
    candidates: Mapping[str, Sequence[Candidate]],
    heldout_by_output_field: Mapping[str, set[str]],
) -> Dict[str, Any]:
    by_target = {target.field: target for target in TARGET_FIELDS}
    leaked: Dict[str, list[str]] = {}
    counts: Dict[str, int] = {}
    for raw_field, field_candidates in candidates.items():
        target = by_target[raw_field]
        heldout = heldout_by_output_field[target.output_field]
        leaks = sorted({candidate.taxonomy_code for candidate in field_candidates if candidate.taxonomy_code in heldout})
        if leaks:
            leaked[target.output_field] = leaks
        counts[target.output_field] = len(field_candidates)
    return {
        "passed": not leaked,
        "training_candidate_count_by_output_field": counts,
        "leaked_heldout_codes": leaked,
    }


def audit_hard_negative_lookup(
    lookup: Mapping[str, Mapping[str, Sequence[str]]],
    heldout_by_output_field: Mapping[str, set[str]],
) -> Dict[str, Any]:
    leaked: Dict[str, Dict[str, list[str]]] = {}
    inspected = 0
    for output_field, items in lookup.items():
        heldout = heldout_by_output_field.get(output_field, set())
        for record_key, codes in items.items():
            inspected += len(codes)
            bad = sorted({str(code) for code in codes if str(code) in heldout})
            if bad:
                leaked.setdefault(output_field, {})[record_key] = bad
    return {
        "passed": not leaked,
        "inspected_negative_codes": inspected,
        "leaked_heldout_codes": leaked,
    }
