"""Tests for margin (class-contrastive) Integrated Gradients. Run: .venv\Scripts\python.exe -I tests\test_contrastive.py"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ecg_concept import attribution as at  # noqa: E402
from ecg_concept.lightecgnetv2 import LightECGNetV2  # noqa: E402


class Nonlin(torch.nn.Module):
    def __init__(self, T=40, C=4):
        super().__init__()
        torch.manual_seed(1)
        self.l1, self.l2 = torch.nn.Linear(12 * T, 16), torch.nn.Linear(16, C)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x.reshape(len(x), -1))))


def rnd(T=40, seed=0):
    return np.random.RandomState(seed).randn(12, T).astype(np.float32)


class T(unittest.TestCase):
    def test_margin_ig_is_difference_of_class_igs(self):
        m, x = Nonlin(), rnd(seed=2)
        a_c = at.integrated_gradients(m, x, 1, steps=64)
        a_k = at.integrated_gradients(m, x, 3, steps=64)
        a_m = at.integrated_gradients(m, x, (1, 3), steps=64)
        self.assertTrue(np.allclose(a_m["attr"], a_c["attr"] - a_k["attr"], atol=1e-5))     # linearity of the path integral

    def test_margin_completeness_and_values(self):
        m, x = Nonlin(), rnd(seed=3)
        r = at.integrated_gradients(m, x, (0, 2), steps=256)
        with torch.no_grad():
            o = m(torch.from_numpy(x)[None])[0]
            o0 = m(torch.zeros(1, 12, 40))[0]
        self.assertAlmostEqual(r["f_x"], float(o[0] - o[2]), places=5)
        self.assertAlmostEqual(r["f_baseline"], float(o0[0] - o0[2]), places=5)
        self.assertLess(r["completeness_abs_err"], 5e-3)

    def test_margin_perturbation_endpoints(self):
        m, x, b = Nonlin(), rnd(seed=4), np.zeros((12, 40), np.float32)
        rid, n = at.region_ids("lead", 12, 40, 5)
        ms = at.prefix_masks(np.arange(12), rid, [0, 12])
        d = at.perturbed_logits(m, x, b, (1, 2), ms, "delete")
        with torch.no_grad():
            o, o0 = m(torch.from_numpy(x)[None])[0], m(torch.from_numpy(b)[None])[0]
        self.assertAlmostEqual(float(d[0]), float(o[1] - o[2]), places=4)
        self.assertAlmostEqual(float(d[1]), float(o0[1] - o0[2]), places=4)

    def test_integer_target_unchanged_by_the_extension(self):
        m, x = Nonlin(), rnd(seed=5)
        r = at.integrated_gradients(m, x, 2, steps=16)
        with torch.no_grad():
            self.assertAlmostEqual(r["f_x"], float(m(torch.from_numpy(x)[None])[0, 2]), places=5)

    def test_real_architecture_margin_shape_and_frozen(self):
        m = LightECGNetV2(6).eval()
        h = [v.clone() for v in m.state_dict().values()]
        r = at.integrated_gradients(m, rnd(T=5000, seed=6), (4, 1), steps=2, chunk=2)
        self.assertEqual(r["attr"].shape, (12, 5000))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(h, m.state_dict().values())))

    def test_competitor_rule_helper(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cx", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "xai_contrastive_ig.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        spec.loader.exec_module(mod)
        thr = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        logits = np.array([3.0, 2.0, -1.0, -3.0, 0.5])                                      # probs .95 .88 .27 .05 .62
        k, flag = mod.pick_competitor(logits, thr, 0)
        self.assertEqual((k, flag), (2, False))                                             # strongest class NOT predicted positive (class 1 and 4 are positive)
        k, flag = mod.pick_competitor(np.array([3.0, 2.0, 1.0]), np.array([0.5] * 3), 0)
        self.assertEqual((k, flag), (1, True))                                              # all positive -> strongest non-target, flagged
        k, _ = mod.pick_competitor(np.array([1.0, -1.0, -1.0]), np.array([0.5] * 3), 0)
        self.assertEqual(k, 1)                                                              # tie -> lowest index (deterministic)


if __name__ == "__main__":
    unittest.main(verbosity=1)
