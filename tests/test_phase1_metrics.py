r"""Phase 1: metric and patient-clustered bootstrap tests. Run: .venv\Scripts\python.exe -I tests\test_phase1_metrics.py"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import eval_metrics as em  # noqa: E402


class T_(unittest.TestCase):
    def test_perfect_and_inverted_ranking(self):
        y = np.array([[1], [1], [0], [0]])
        self.assertEqual(em.macro_auroc(y, np.array([[.9], [.8], [.2], [.1]])), 1.0)
        self.assertEqual(em.macro_auroc(y, np.array([[.1], [.2], [.8], [.9]])), 0.0)
        self.assertEqual(em.macro_auroc(y, np.full((4, 1), .5)), 0.5)

    def test_undefined_class_is_reported_not_zero(self):
        y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
        p = np.array([[.9, .5], [.1, .5], [.8, .5], [.2, .5]])
        r = em.report(y, p, [0.5, 0.5], ["A", "B"])
        self.assertEqual(r["undefined_auroc_classes"], ["B"])
        self.assertEqual(r["macro_auroc"], 1.0)                      # mean over the defined class only
        self.assertIsNone(r["per_class"]["B"]["auroc"])

    def test_brier_and_ece_known_values(self):
        y = np.array([[1], [0], [1], [0]])
        p = np.array([[1.0], [0.0], [0.5], [0.5]])
        self.assertAlmostEqual(float(em.brier_columns(y, p)[0]), (0 + 0 + .25 + .25) / 4)
        self.assertAlmostEqual(float(em.ece_columns(np.array([[1], [0], [1], [0]]), np.full((4, 1), .5))[0]), 0.0)    # calibrated bin
        self.assertAlmostEqual(float(em.ece_columns(np.zeros((5, 1)), np.ones((5, 1)))[0]), 1.0)                       # confident and wrong
        yy = np.array([[1], [0]] * 10)
        self.assertAlmostEqual(float(em.ece_columns(yy, np.full((20, 1), .9))[0]), 0.4)                                 # overconfident

    def test_threshold_metrics_and_rare_class_sensitivity(self):
        y = np.array([[1, 1], [1, 0], [0, 0], [0, 1]])
        p = np.array([[.9, .2], [.4, .1], [.3, .6], [.1, .8]])
        r = em.report(y, p, [0.5, 0.5], ["common", "rare"], train_support=[5000, 20], rare_support=100)
        self.assertEqual(r["per_class"]["common"]["sensitivity"], 0.5)
        self.assertEqual(r["per_class"]["rare"]["sensitivity"], 0.5)
        self.assertEqual(r["rare_class_sensitivity"], 0.5)
        self.assertEqual(r["n_rare_classes"], 1)
        self.assertEqual(r["per_class"]["common"]["specificity"], 1.0)
        self.assertAlmostEqual(r["micro_f1"], 2 * 2 / (2 * 2 + 1 + 2))

    def test_cluster_bootstrap_keeps_width_for_clustered_data_and_record_bootstrap_is_too_narrow(self):
        rng = np.random.RandomState(0)
        n = 600
        y = (rng.rand(n, 1) < 0.4).astype(int)
        p = np.clip(0.5 * y + rng.rand(n, 1) * 0.8, 0, 1)
        f = lambda a, b: em.macro_auroc(a, b)
        _, lo1, hi1, _ = em.cluster_bootstrap(y, p, np.arange(n), f, n_boot=400, seed=1)
        y5, p5 = np.repeat(y, 5, 0), np.repeat(p, 5, 0)                       # every patient has 5 identical records
        _, lo5c, hi5c, _ = em.cluster_bootstrap(y5, p5, np.repeat(np.arange(n), 5), f, n_boot=400, seed=1)
        _, lo5r, hi5r, _ = em.cluster_bootstrap(y5, p5, np.arange(5 * n), f, n_boot=400, seed=1)    # wrongly treating records as independent
        self.assertAlmostEqual(hi1 - lo1, hi5c - lo5c, delta=0.35 * (hi1 - lo1))
        self.assertLess(hi5r - lo5r, 0.75 * (hi5c - lo5c))

    def test_cluster_bootstrap_is_deterministic_and_paired_difference_detects_gain(self):
        rng = np.random.RandomState(1)
        n = 400
        y = (rng.rand(n, 1) < 0.5).astype(int)
        pa = np.clip(0.6 * y + rng.rand(n, 1) * 0.6, 0, 1)
        pb = np.clip(0.2 * y + rng.rand(n, 1) * 0.6, 0, 1)
        g = rng.randint(0, 150, n)
        f = lambda a, b: em.macro_auroc(a, b)
        a1 = em.cluster_bootstrap(y, pa, g, f, n_boot=200, seed=3)[:3]
        a2 = em.cluster_bootstrap(y, pa, g, f, n_boot=200, seed=3)[:3]
        self.assertEqual(a1, a2)
        d, lo, hi, _ = em.paired_cluster_bootstrap(y, pa, pb, g, f, n_boot=300, seed=4)
        self.assertGreater(d, 0)
        self.assertGreater(lo, 0)
        self.assertLess(lo, hi)

    def test_select_thresholds_uses_only_given_predictions_and_handles_empty_class(self):
        y = np.array([[1, 0], [1, 0], [0, 0], [0, 0], [1, 0], [0, 0]])
        p = np.array([[.7, .9], [.65, .9], [.2, .9], [.4, .9], [.6, .9], [.1, .9]])
        thr = em.select_thresholds(y, p)
        self.assertTrue(0.4 <= thr[0] < 0.6)                      # separates 0.6..0.7 positives from <=0.4 negatives
        self.assertEqual(thr[1], 0.5)                             # no positives -> default, not fitted



if __name__ == "__main__":
    unittest.main(verbosity=1)
