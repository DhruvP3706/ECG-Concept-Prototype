"""Focused tests for src/ecg_concept/attribution.py (synthetic; no dataset, no checkpoint). Run: .venv\\Scripts\\python.exe -I tests\\test_attribution.py"""
import hashlib
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ecg_concept import attribution as at  # noqa: E402
from ecg_concept.lightecgnetv2 import LightECGNetV2  # noqa: E402


class Lin(nn.Module):
    """logit_c = sum(w_c * x) + b_c over a (12, T) input: attribution is exactly w * (x - baseline)."""

    def __init__(self, T=40, C=3):
        super().__init__()
        torch.manual_seed(0)
        self.w = nn.Parameter(torch.randn(C, 12 * T))
        self.b = nn.Parameter(torch.randn(C))

    def forward(self, x):
        return x.reshape(len(x), -1) @ self.w.T + self.b


class Nonlin(nn.Module):
    def __init__(self, T=40, C=3):
        super().__init__()
        torch.manual_seed(1)
        self.l1, self.l2 = nn.Linear(12 * T, 16), nn.Linear(16, C)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x.reshape(len(x), -1))))


def rnd(T=40, seed=0):
    return np.random.RandomState(seed).randn(12, T).astype(np.float32)


class TestIG(unittest.TestCase):
    def test_linear_model_is_exact_and_shape_preserved(self):
        m, x = Lin(), rnd()
        r = at.integrated_gradients(m, x, 1, steps=4)
        self.assertEqual(r["attr"].shape, (12, 40))
        self.assertTrue(np.allclose(r["attr"], m.w[1].detach().numpy().reshape(12, 40) * x, atol=1e-5))
        self.assertLess(r["completeness_rel_err"], 1e-4)

    def test_completeness_improves_with_steps_for_nonlinear_model(self):
        m, x = Nonlin(), rnd(seed=3)
        e = [at.integrated_gradients(m, x, 2, steps=s)["completeness_abs_err"] for s in (4, 32, 256)]
        self.assertLess(e[2], e[0] + 1e-9)
        self.assertLess(e[2], 5e-3)

    def test_baseline_handling(self):
        m, x = Nonlin(), rnd(seed=4)
        r0 = at.integrated_gradients(m, x, 0, baseline=x.copy(), steps=8)             # x == baseline -> zero attribution
        self.assertTrue(np.allclose(r0["attr"], 0))
        b = rnd(seed=5)
        r1 = at.integrated_gradients(m, x, 0, baseline=b, steps=64)
        self.assertLess(r1["completeness_abs_err"], 5e-3)                               # sum(attr) = f(x) - f(b) for a non-zero baseline too
        self.assertAlmostEqual(r1["f_baseline"], float(m(torch.from_numpy(b)[None])[0, 0]), places=5)
        with self.assertRaises(ValueError):
            at.integrated_gradients(m, x, 0, baseline=rnd(T=30))

    def test_numerical_stability_float32_vs_reference(self):
        m, x = Nonlin(), rnd(seed=6)
        a64 = at.integrated_gradients(m, x, 1, steps=64)["attr"]
        a128 = at.integrated_gradients(m, x, 1, steps=128)["attr"]
        self.assertTrue(np.isfinite(a64).all())
        self.assertGreater(np.corrcoef(a64.ravel(), a128.ravel())[0, 1], 0.999)

    def test_model_and_input_are_not_changed(self):
        m, x = LightECGNetV2(5), rnd(T=5000, seed=7)
        m.eval()
        h0 = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in m.state_dict().values())).hexdigest()
        xin = torch.from_numpy(x.copy())
        before = m(xin[None]).detach().clone()
        flags = [p.requires_grad for p in m.parameters()]
        at.integrated_gradients(m, xin, 2, steps=2, chunk=2)
        self.assertEqual(h0, hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in m.state_dict().values())).hexdigest())
        self.assertTrue(torch.equal(xin, torch.from_numpy(x)))
        self.assertEqual(flags, [p.requires_grad for p in m.parameters()])
        self.assertFalse(m.training)
        self.assertTrue(torch.equal(before, m(xin[None]).detach()))                    # prediction invariance

    def test_real_architecture_shape_and_finite(self):
        m = LightECGNetV2(7).eval()
        r = at.integrated_gradients(m, rnd(T=5000, seed=8), 3, steps=4, chunk=4)
        self.assertEqual(r["attr"].shape, (12, 5000))
        self.assertTrue(np.isfinite(r["attr"]).all())


class TestPerturbation(unittest.TestCase):
    def test_region_ids_and_scores(self):
        rid, n = at.region_ids("time", 12, 5000, 125)
        self.assertEqual((rid.shape, n), ((12, 5000), 40))
        rid, n = at.region_ids("cell", 12, 5000, 125)
        self.assertEqual((n, int(rid.max()) + 1), (480, 480))
        rid, n = at.region_ids("lead", 12, 5000, 125)
        a = np.zeros((12, 5000))
        a[3] = 1.0
        s = at.region_scores(a, rid, n)
        self.assertEqual(int(np.argmax(s)), 3)
        self.assertEqual(float(s.sum()), 5000.0)

    def test_prefix_masks_nested_and_exact_fraction(self):
        rid, n = at.region_ids("time", 12, 5000, 125)
        order = np.random.RandomState(0).permutation(n)
        ms = at.prefix_masks(order, rid, [0, 4, 8, 40])
        self.assertFalse(ms[0].any())
        self.assertTrue(ms[3].all())
        self.assertAlmostEqual(ms[1].mean(), 4 / 40)
        self.assertTrue(((ms[1] & ~ms[2]).sum()) == 0)                                  # nested prefixes
        self.assertTrue(ms[1].reshape(12, -1)[:, :].any(axis=0).sum() == 4 * 125)       # contiguous windows removed from every lead

    def test_orderings_directions(self):
        rng = np.random.RandomState(0)
        ss, sa = np.array([3., -1., 0.5, -4.]), np.array([3., 1., 0.5, 4.])
        o = at.orderings(ss, sa, rng, n_random=3)
        self.assertEqual(o["pos"].tolist(), [0, 2, 1, 3])
        self.assertEqual(o["neg"].tolist(), [3, 1, 2, 0])
        self.assertEqual(o["abs"].tolist(), [3, 0, 1, 2])
        self.assertEqual(o["low_abs"].tolist(), [2, 1, 0, 3])
        self.assertEqual(o["random"].shape, (3, 4))

    def test_delete_insert_semantics_and_endpoints(self):
        m = Lin()
        x, b = rnd(seed=1), np.zeros((12, 40), np.float32)
        rid, n = at.region_ids("lead", 12, 40, 5)
        order = np.arange(12)
        ms = at.prefix_masks(order, rid, [0, 12])
        d = at.perturbed_logits(m, x, b, 0, ms, "delete")
        i = at.perturbed_logits(m, x, b, 0, ms, "insert")
        fx, fb = float(m(torch.from_numpy(x)[None])[0, 0]), float(m(torch.from_numpy(b)[None])[0, 0])
        self.assertAlmostEqual(float(d[0]), fx, places=4)           # nothing deleted = original
        self.assertAlmostEqual(float(d[1]), fb, places=4)           # everything deleted = baseline
        self.assertAlmostEqual(float(i[0]), fb, places=4)           # nothing restored = baseline
        self.assertAlmostEqual(float(i[1]), fx, places=4)           # everything restored = original

    def test_attribution_order_beats_random_on_linear_model(self):
        m = Lin()
        x, b = rnd(seed=2), np.zeros((12, 40), np.float32)
        r = at.integrated_gradients(m, x, 2, steps=4)
        rid, n = at.region_ids("lead", 12, 40, 5)
        ss = at.region_scores(r["attr"], rid, n)
        ks = list(range(13))
        top = at.perturbed_logits(m, x, b, 2, at.prefix_masks(np.argsort(-ss), rid, ks), "delete")
        rng = np.random.RandomState(0)
        rand = np.mean([at.perturbed_logits(m, x, b, 2, at.prefix_masks(rng.permutation(12), rid, ks), "delete") for _ in range(30)], axis=0)
        self.assertLess(at.curve_auc(top, np.array(ks) / 12), at.curve_auc(rand, np.array(ks) / 12))
        self.assertAlmostEqual(at.curve_auc([1, 1, 1], [0, .5, 1]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
