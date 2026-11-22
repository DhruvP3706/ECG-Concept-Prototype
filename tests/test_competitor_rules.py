"""Tests for the competitor-selection rules A/B/C of Experiment 4. Run: .venv\Scripts\python.exe -I tests\test_competitor_rules.py"""
import importlib.util
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
spec = importlib.util.spec_from_file_location("cr", os.path.join(ROOT, "scripts", "xai_competitor_robustness.py"))
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)
xc = cr.xc


class T(unittest.TestCase):
    thr = np.full(6, 0.5)
    logits = np.array([2.0, 1.5, 0.8, -0.2, -1.0, -3.0])        # probs .88 .82 .69 .45 .27 .05 ; classes 1,2 are predicted positive

    def test_rule_a_highest_probability_non_positive(self):
        self.assertEqual(xc.pick_competitor(self.logits, self.thr, 0), (3, False))

    def test_rule_b_ignores_thresholds(self):
        self.assertEqual(cr.pick_highest_logit(self.logits, 0), 1)                 # strongest non-target even though it is predicted positive
        self.assertEqual(cr.pick_highest_logit(self.logits, 1), 0)

    def test_rule_c_second_eligible_and_unavailable(self):
        self.assertEqual(cr.pick_second_eligible(self.logits, self.thr, 0), 4)     # eligible = {3, 4, 5}; second by probability = 4
        self.assertIsNone(cr.pick_second_eligible(np.array([2.0, -1.0, 1.0]), np.full(3, 0.5), 0))   # only one eligible class

    def test_rules_never_return_the_target_and_are_deterministic(self):
        rng = np.random.RandomState(0)
        for _ in range(200):
            lg = rng.randn(45) * 2
            thr = rng.uniform(0.2, 0.85, 45)
            c = int(rng.randint(45))
            a, _ = xc.pick_competitor(lg, thr, c)
            b = cr.pick_highest_logit(lg, c)
            cc = cr.pick_second_eligible(lg, thr, c)
            self.assertNotIn(c, (a, b))
            if cc is not None:
                self.assertNotEqual(cc, c)
                self.assertNotEqual(cc, a)                                          # C is the runner-up of A's candidate list
            self.assertEqual((a, b, cc), (xc.pick_competitor(lg, thr, c)[0], cr.pick_highest_logit(lg, c), cr.pick_second_eligible(lg, thr, c)))

    def test_ties_go_to_lowest_index(self):
        lg = np.array([3.0, 0.0, 0.0, -1.0])
        self.assertEqual(cr.pick_highest_logit(lg, 0), 1)
        self.assertEqual(cr.pick_second_eligible(lg, np.full(4, 0.9), 0), 2)

    def test_perturbed_multi_matches_single_class_function(self):
        import torch
        from ecg_concept import attribution as at
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12 * 40, 5))
        x, b = np.random.RandomState(1).randn(12, 40).astype(np.float32), np.zeros((12, 40), np.float32)
        rid, n = at.region_ids("lead", 12, 40, 5)
        ms = at.prefix_masks(np.arange(12), rid, [0, 3, 12])
        multi = cr.perturbed_multi(m, x, b, [1, 4], ms, "delete")
        self.assertTrue(np.allclose(multi[:, 0], at.perturbed_logits(m, x, b, 1, ms, "delete"), atol=1e-5))
        self.assertTrue(np.allclose(multi[:, 1], at.perturbed_logits(m, x, b, 4, ms, "delete"), atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=1)
