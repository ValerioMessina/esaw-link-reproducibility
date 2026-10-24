"""Offline checks for the ablation publication boundary and metric definitions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/role_conditioning"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RoleAblationTests(unittest.TestCase):
    def test_field_mean_differs_from_complete_record_recall(self):
        gold = np.zeros((2, 8), dtype=int)
        ranks = np.ones((2, 8, 10), dtype=int)
        ranks[0, :, 0] = 0
        ranks[1, :4, 0] = 0
        values = load("recount").metrics(ranks, gold)
        self.assertEqual(values["top1_pct"], 75.0)
        self.assertEqual(values["record_all8_top1_pct"], 50.0)

    def test_preparation_uses_external_fresh_workspace(self):
        runner = load("run")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths = [base / name for name in ("dataset.json", "taxonomy.json", "split.json")]
            for path in paths:
                path.write_text("{}")
            with self.assertRaises(ValueError):
                runner.prepare(*paths, ROOT / "private", 1)
            work = base / "work"
            runner.prepare(*paths, work, 1)
            with self.assertRaises(FileExistsError):
                runner.prepare(*paths, work, 1)
            self.assertTrue((work / "src/esaw_biencoder/data.py").is_file())
            self.assertNotIn("RR@5 %", (work / "suite.py").read_text())
            state = json.loads((work / "runpod_state.json").read_text())
            self.assertEqual(set(state), {"training_deadline_epoch"})

    def test_bootstrap_does_not_claim_training_seed_variability(self):
        evidence = json.loads((EXP / "aggregate_evidence.json").read_text())
        pair = evidence["paired_comparisons"]
        self.assertEqual(pair["replicates"], 10000)
        self.assertEqual(evidence["training_seed"], 13)
        self.assertEqual(pair["results"]["refinement"]["top1"]["role_on_minus_off_pp"], 2.3125)


if __name__ == "__main__":
    unittest.main()
