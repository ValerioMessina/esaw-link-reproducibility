"""Aggregate-evidence integrity; no institutional inputs or model inference."""
import json, math, re, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class BaselineEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads((ROOT / 'docs/BASELINE_TEST_20261021.json').read_text())

    def test_incomplete_training_is_not_a_completed_five_seed_benchmark(self):
        e = self.evidence
        self.assertFalse(e['new_training'])
        self.assertTrue(e['retrospective'])
        self.assertFalse(e['hgclr_five_training_protocol_complete'])
        self.assertEqual([r['seed'] for r in e['hgclr']], [3, 13, 21, 42, 87])
        complete = [r['seed'] for r in e['hgclr'] if r['training_status']['complete']]
        self.assertEqual(complete, [3, 13, 21, 42])
        for row in e['hgclr']:
            status = row['training_status']
            self.assertIs(type(status['complete']), bool)
            self.assertLessEqual(row['selected_validation_checkpoint_epoch'], status['epoch'])
            self.assertEqual(status['reason'], 'early_stopping' if status['complete'] else 'budget_or_signal')

    def test_denominators_and_nested_topk_metrics(self):
        e = self.evidence
        self.assertEqual((e['records'], e['fields'], e['decisions']), (5000, 8, 40000))
        self.assertEqual(e['decisions'], e['records'] * e['fields'])
        for row in e['hgclr'] + e['historical_adapters']:
            for mode in ('raw', 'validation_calibrated'):
                with self.subTest(seed=row['seed'], method=row.get('method', 'HGCLR'), mode=mode):
                    m = row[mode]
                    for key, value in m.items():
                        self.assertTrue(math.isfinite(value))
                        self.assertTrue(0 <= value <= 100)
                        denominator = e['records'] if key.startswith('record_') else e['decisions']
                        count = value * denominator / 100
                        self.assertAlmostEqual(count, round(count), places=7)
                    field = [m['mean_top1_pct'], m['mean_top5_pct'], m['mean_top10_pct']]
                    record = [m['record_exact8_pct'], m['record_all8_top5_pct'], m['record_all8_top10_pct']]
                    self.assertEqual(field, sorted(field))
                    self.assertEqual(record, sorted(record))
                    for a, b in zip(record, field):
                        self.assertLessEqual(a, b)

    def test_strict_loading_probes_and_checkpoint_hashes_agree(self):
        for row in self.evidence['hgclr']:
            probe = row['validation_probe']
            self.assertTrue(probe['strict_encoder_classifier'])
            self.assertEqual(probe['checkpoint_sha256'], row['checkpoint_sha256'])
            self.assertRegex(row['checkpoint_sha256'], r'^[0-9a-f]{64}$')
            self.assertTrue(0 <= probe['validation_probe_max_abs_diff'] < 0.003)
            self.assertTrue(row['test_evidence']['passed'])
            self.assertEqual(row['test_evidence']['records'], self.evidence['records'])
            for key in ('scores_sha256', 'predictions_sha256'):
                self.assertRegex(row['test_evidence'][key], r'^[0-9a-f]{64}$')
            self.assertEqual(len(row['offsets']), 8)
            for offset in row['offsets'].values():
                self.assertTrue(-12 <= offset <= 12)
                self.assertEqual(offset * 4, round(offset * 4))

    def test_historical_adapters_remain_identified_separately(self):
        rows = self.evidence['historical_adapters']
        self.assertEqual([(r['method'], r['seed'], r['checkpoint_epoch']) for r in rows],
                         [('HTC-CLIP', 3, 50), ('HiSR', 13, 15), ('HyILR', 13, 6)])
        for row in rows:
            self.assertRegex(row['checkpoint_sha256'], r'^[0-9a-f]{64}$')
        hgclr_hashes = {r['checkpoint_sha256'] for r in self.evidence['hgclr']}
        self.assertFalse(hgclr_hashes & {r['checkpoint_sha256'] for r in rows})

if __name__ == '__main__':
    unittest.main()
