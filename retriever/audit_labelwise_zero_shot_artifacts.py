from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from esaw_biencoder.label_holdout import load_holdout_fold, sha256_file
from esaw_biencoder.schema import TARGET_FIELDS


def negative_code(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        return str(item.get("taxonomy_code") or item.get("code") or "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit bloccante degli artefatti label-wise zero-shot.")
    parser.add_argument("--label-holdout-file", required=True)
    parser.add_argument("--label-holdout-fold", type=int, default=0)
    parser.add_argument("--hard-negative-file", required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    heldout, manifest = load_holdout_fold(args.label_holdout_file, args.label_holdout_fold)
    hard_payload = json.loads(Path(args.hard_negative_file).read_text(encoding="utf-8"))
    leaks: list[dict[str, str]] = []
    inspected_negatives = 0
    inspected_training_items = 0
    for output_field, field_payload in hard_payload.get("fields", {}).items():
        for record_key, item in field_payload.get("items", {}).items():
            inspected_training_items += 1
            gold = str(item.get("gold_code") or "")
            if gold in heldout.get(output_field, set()):
                leaks.append({"field": output_field, "record": record_key, "role": "gold", "code": gold})
            for raw_negative in item.get("negatives", []):
                code = negative_code(raw_negative)
                inspected_negatives += 1
                if code in heldout.get(output_field, set()):
                    leaks.append({"field": output_field, "record": record_key, "role": "negative", "code": code})

    model_checks: list[dict[str, Any]] = []
    manifest_sha = sha256_file(args.label_holdout_file)
    for model in args.model:
        config_path = Path(model) / "training_config.json"
        if not config_path.exists():
            model_checks.append({"model": model, "passed": False, "reason": "training_config.json missing"})
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        label_config = config.get("label_holdout") or {}
        passed = (
            label_config.get("manifest_sha256") == manifest_sha
            and label_config.get("fold") == args.label_holdout_fold
            and label_config.get("candidate_audit", {}).get("passed") is True
            and label_config.get("hard_negative_audit", {}).get("passed") is True
        )
        model_checks.append(
            {
                "model": model,
                "passed": passed,
                "manifest_sha256": label_config.get("manifest_sha256"),
                "fold": label_config.get("fold"),
            }
        )

    passed = not leaks and all(check["passed"] for check in model_checks)
    payload = {
        "protocol": manifest["protocol"],
        "manifest_sha256": manifest_sha,
        "fold": args.label_holdout_fold,
        "passed": passed,
        "inspected_training_items": inspected_training_items,
        "inspected_negative_codes": inspected_negatives,
        "leaks": leaks,
        "model_checks": model_checks,
        "rules": [
            "held-out codes cannot be hard-negative gold labels",
            "held-out codes cannot be mined negatives",
            "every trained checkpoint must declare the exact pre-registered manifest and fold",
            "material holdout is shared across activity, deviation and contact roles",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
