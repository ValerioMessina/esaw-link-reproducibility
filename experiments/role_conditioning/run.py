"""Prepare an external private workspace; execute CUDA training only with --execute."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def prepare(dataset, taxonomy, split_file, work_dir, deadline_hours):
    work = Path(work_dir).resolve()
    if work == REPO or REPO in work.parents:
        raise ValueError("All private inputs and generated artifacts must stay outside the repository")
    if deadline_hours <= 0:
        raise ValueError("deadline-hours must be positive")
    inputs = {"dataset.json": Path(dataset).resolve(),
              "tassonomia.json": Path(taxonomy).resolve(),
              "split.json": Path(split_file).resolve()}
    for path in inputs.values():
        if not path.is_file() or path == REPO or REPO in path.parents:
            raise ValueError("Supply existing authorized inputs outside the repository")
    provenance = json.loads((HERE / "source_provenance.json").read_text())
    for name, expected in provenance["source_files"].items():
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen dependency changed: {name}")
    source = HERE / "historical/suite.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != provenance["original_suite_sha256"]:
        raise ValueError("Historical suite hash mismatch")
    # Never reuse previous test/model/mining caches under a different protocol.
    work.mkdir(parents=True, exist_ok=False)
    (work / "private").mkdir()
    for name, path in inputs.items():
        shutil.copy2(path, work / "private" / name)
    shutil.copytree(REPO / "retriever/esaw_biencoder", work / "src/esaw_biencoder",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(HERE / "protocol.json", work / "protocol.json")
    # Historical numerical code unchanged; only fix the two Markdown column names.
    suite = source.read_text().replace("RR@5 %", "All8@5 %").replace("RR@10 %", "All8@10 %")
    (work / "suite.py").write_text(suite)
    state = {"training_deadline_epoch": time.time() + deadline_hours * 3600}
    (work / "runpod_state.json").write_text(json.dumps(state) + "\n")
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "taxonomy", "split-file", "work-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline-hours", type=float, required=True,
                        help="Local process deadline; does not stop or provision cloud instances")
    parser.add_argument("--execute", action="store_true", help="Start training on an existing CUDA GPU")
    args = parser.parse_args()
    work = prepare(args.dataset, args.taxonomy, args.split_file, args.work_dir, args.deadline_hours)
    command = [sys.executable, "-u", str(work / "suite.py")]
    print(json.dumps({"prepared": str(work), "command": command, "execute": args.execute}))
    if args.execute:
        return subprocess.run(command, cwd=work, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
