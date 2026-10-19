from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping

import importlib.util


def load_audit_module(path: Path):
    spec = importlib.util.spec_from_file_location("audit_esaw_with_openai", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossibile caricare {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner parallelo resumable per audit ESAW.")
    parser.add_argument("--audit-script", default="")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=6400)
    parser.add_argument("--deepseek-reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument("--deepseek-no-thinking", action="store_true")
    return parser.parse_args()


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            completed.add(str(json.loads(line)["record_key"]))
    return completed


def append_jsonl(path: Path, row: Mapping[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    audit_script = Path(args.audit_script) if args.audit_script else Path(__file__).with_name("audit_esaw_with_openai.py")
    mod = load_audit_module(audit_script)
    records = mod.load_dataset(args.dataset)
    by_key = {mod.record_key(record, index): record for index, record in enumerate(records)}
    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    predictions = mod.load_predictions(args.predictions)[: args.limit]

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_jsonl)
    lock = threading.Lock()

    tasks = []
    for index, prediction in enumerate(predictions, 1):
        key = str(prediction["record_key"])
        if key in completed:
            continue
        tasks.append((index, prediction, list(mod.TARGETS)))

    total = len(predictions)
    print(f"total={total} completed={len(completed)} pending={len(tasks)} workers={args.workers}", flush=True)

    def run_one(task):
        index, prediction, fields = task
        key = str(prediction["record_key"])
        payload = mod.build_user_payload(taxonomy, by_key[key], prediction, fields)
        audit = mod.call_model(
            args.provider,
            args.model,
            payload,
            dry_run=False,
            retries=args.retries,
            timeout=args.timeout,
            max_output_tokens=args.max_output_tokens,
            deepseek_thinking=not args.deepseek_no_thinking,
            deepseek_reasoning_effort=args.deepseek_reasoning_effort,
        )
        errors = mod.validate_audit(payload, audit)
        row = {
            "case_index": index,
            "record_key": key,
            "fields_requested": fields,
            "request": payload,
            "audit": audit,
            "validation_errors": errors,
        }
        append_jsonl(output_jsonl, row, lock)
        return index, key, len(errors)

    done = len(completed)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            index, key, error_count = future.result()
            done += 1
            print(f"[{done}/{total}] record={key} original_index={index} errors={error_count}", flush=True)

    csv_rows: list[Dict[str, Any]] = []
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                csv_rows.extend(mod.csv_rows_from_result(json.loads(line)))
    write_csv(Path(args.output_csv), csv_rows)


if __name__ == "__main__":
    main()
