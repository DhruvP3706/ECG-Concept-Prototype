"""Risk-focused tests for the Step 4.4-6.1 concept/prototype components. Run: .venv\\Scripts\\python.exe -I tests\\test_step4_4.py"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ecg_concept import concept_proto as cp  # noqa: E402


class TestMaskedLoss(unittest.TestCase):
    def test_missing_targets_do_not_contribute_or_propagate_nan(self):
        pred = torch.zeros(4, 2, requires_grad=True)
        target = torch.tensor([[1.0, float("nan")], [float("nan"), float("nan")], [3.0, 2.0], [float("nan"), 4.0]])
        mask = torch.isfinite(target)
        loss = cp.masked_mse(pred, target, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(pred.grad).all())
        # concept 0 valid targets {1,3}: mean sq = (1+9)/2 = 5 ; concept 1 valid {2,4}: (4+16)/2 = 10 ; average 7.5
        self.assertAlmostEqual(float(loss), 7.5, places=6)
        self.assertEqual(float(pred.grad[1].abs().sum()), 0.0)                       # a row with no valid target gets no gradient

    def test_all_missing_is_zero_not_nan(self):
        pred = torch.ones(3, 2)
        loss = cp.masked_mse(pred, torch.full((3, 2), float("nan")), torch.zeros(3, 2, dtype=torch.bool))
        self.assertEqual(float(loss), 0.0)


class TestScaling(unittest.TestCase):
    def test_fit_uses_only_supplied_rows_and_ignores_nan(self):
        train = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
        val = np.array([[100.0, 1000.0]])
        s = cp.Standardizer.fit(train)
        np.testing.assert_allclose(s.mean, [3.0, 20.0])                               # validation row had no influence
        np.testing.assert_allclose(s.std, [np.std([1, 3, 5]), 10.0])
        np.testing.assert_allclose(s.inverse(s.transform(val)), val)
        self.assertTrue(np.isnan(s.transform(train)[1, 1]))                           # missing stays missing, never becomes 0

    def test_std_floor(self):
        s = cp.Standardizer.fit(np.ones((5, 1)))
        self.assertGreater(s.std[0], 0)


class TestPrototypes(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.Z = np.vstack([rng.normal(m, 0.2, (40, 4)) for m in (-2, 0, 2)])

    def test_kmeans_deterministic_and_anchors_are_real_distinct_records(self):
        c1, a1 = cp.kmeans_det(self.Z, 5)
        c2, a2 = cp.kmeans_det(self.Z, 5)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(a1, a2)
        idx = cp.anchor_records(self.Z, c1)
        self.assertEqual(len(set(idx)), 5)
        self.assertTrue(all(0 <= i < len(self.Z) for i in idx))

    def test_anchor_collision_resolved(self):
        centres = np.array([self.Z[0], self.Z[0]])                                    # two centres on the same record
        idx = cp.anchor_records(self.Z, centres)
        self.assertEqual(len(set(idx)), 2)

    def test_tau_safeguard_and_similarity_range(self):
        P = self.Z[:3]
        tau0 = cp.tau_rule(P, P)                                                       # all training points ARE prototypes: distance 0
        self.assertGreaterEqual(tau0, 1e-6)
        s = cp.similarities(self.Z, P, cp.tau_rule(self.Z, P))
        self.assertTrue(np.isfinite(s).all() and (s > 0).all() and (s <= 1).all())
        np.testing.assert_allclose(cp.similarities(P, P, 1.0)[np.arange(3), np.arange(3)], 1.0)

    def test_logits_equal_bias_plus_contributions_and_head_sees_only_similarities(self):
        torch.manual_seed(0)
        head = cp.ProtoHead(16, 45)
        s = torch.rand(7, 16)
        contrib = head.contributions(s)
        recon = head.fc.bias[None, :] + contrib.sum(-1)
        torch.testing.assert_close(head(s), recon, atol=1e-6, rtol=1e-5)
        self.assertEqual(head.fc.in_features, 16)                                      # no extra (embedding/mask) input path
        self.assertEqual([n for n, _ in head.named_parameters()], ["fc.weight", "fc.bias"])


class TestTrainingDeterminism(unittest.TestCase):
    def test_same_seed_same_result_and_best_state_restored(self):
        rng = np.random.RandomState(1)
        X = rng.normal(size=(300, 6))
        Y = (X[:, :3] > 0).astype(np.float32)
        out = []
        for _ in range(2):
            torch.manual_seed(5)                                                       # seed BEFORE constructing the model (weight init)
            m, h = cp.fit_head(torch.nn.Linear(6, 3), X[:200], Y[:200], X[200:], Y[200:], seed=5, lr=1e-2, max_epochs=15, patience=5)
            out.append((cp.predict(m, X[200:]), h["best_val"]))
        np.testing.assert_array_equal(out[0][0], out[1][0])
        self.assertEqual(out[0][1], out[1][1])

    def test_concept_regression_with_missing_targets_trains_finite(self):
        rng = np.random.RandomState(2)
        X = rng.normal(size=(400, 8))
        T = np.stack([X[:, 0] * 2, X[:, 1] - X[:, 2]], 1)
        M = rng.rand(400, 2) > 0.4
        Tn = np.where(M, T, np.nan)
        net = cp.ConceptMLP(8, 16, 2)
        net, h = cp.fit_head(net, X[:300], np.nan_to_num(Tn[:300]), X[300:], np.nan_to_num(Tn[300:]), seed=3, lr=1e-2, max_epochs=30, patience=8,
                             loss="mse", mask_tr=M[:300], mask_va=M[300:])
        self.assertTrue(np.isfinite(h["val"]).all())
        self.assertLess(h["best_val"], h["val"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
