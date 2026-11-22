r"""Phase 1: masked multi-task loss, missing labels, single-dataset batches, augmentation, sampling. Run: .venv\Scripts\python.exe -I tests\test_phase1_losses_data.py"""
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import data_v3 as dv  # noqa: E402
from ecg_concept.lightecgnet_paper import DualHeadLightECGNet  # noqa: E402
from ecg_concept.losses_v3 import MultiTaskLoss, masked_bce_with_logits, pos_weight_from_labels  # noqa: E402


class T_(unittest.TestCase):
    def test_masked_equals_standard_when_everything_valid(self):
        g = torch.Generator().manual_seed(0)
        lg, y = torch.randn(8, 5, generator=g), (torch.rand(8, 5, generator=g) > 0.7).float()
        pw = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        l, n = masked_bce_with_logits(lg, y, pos_weight=pw)
        self.assertAlmostEqual(float(l), float(F.binary_cross_entropy_with_logits(lg, y, pos_weight=pw)), places=6)
        self.assertEqual(n, 40)

    def test_nan_labels_and_masks_are_ignored_with_zero_gradient(self):
        lg = torch.randn(4, 3, requires_grad=True)
        nan = float("nan")
        y = torch.tensor([[1.0, nan, 0.0], [0.0, 1.0, nan], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        mask = torch.ones(4, 3, dtype=torch.bool)
        mask[3] = False
        l, n = masked_bce_with_logits(lg, y, mask)
        l.backward()
        self.assertEqual(n, 7)
        self.assertTrue(torch.isfinite(l))
        self.assertEqual(float(lg.grad[0, 1]), 0.0)
        self.assertEqual(float(lg.grad[1, 2]), 0.0)
        self.assertEqual(float(lg.grad[3].abs().sum()), 0.0)
        self.assertGreater(float(lg.grad[2].abs().sum()), 0.0)

    def test_all_missing_gives_zero_loss_and_zero_gradient(self):
        lg = torch.randn(2, 3, requires_grad=True)
        l, n = masked_bce_with_logits(lg, torch.full((2, 3), float("nan")))
        l.backward()
        self.assertEqual((float(l), n, float(lg.grad.abs().sum())), (0.0, 0, 0.0))

    def test_single_dataset_batch_leaves_other_head_untouched(self):
        d = DualHeadLightECGNet()
        x = torch.randn(4, 12, 5000)
        outs = {"ptbxl": d(x, "ptbxl"), "physionet": None}
        loss = MultiTaskLoss(weights={"ptbxl": 1.0, "physionet": 1.0})
        y = {"ptbxl": (torch.rand(4, 5) > 0.5).float()}
        total, parts = loss(outs, y)
        total.backward()
        self.assertEqual(list(parts), ["ptbxl"])
        self.assertTrue(all(p.grad is None for p in d.heads["physionet"].parameters()))
        with self.assertRaises(ValueError):
            loss({"ptbxl": torch.zeros(0, 5), "physionet": None}, {})

    def test_mixed_batch_uses_each_heads_own_labels_only(self):
        a, b = torch.randn(3, 45, requires_grad=True), torch.randn(2, 5, requires_grad=True)
        ya, yb = (torch.rand(3, 45) > 0.9).float(), (torch.rand(2, 5) > 0.5).float()
        total, parts = MultiTaskLoss(weights={"physionet": 0.5, "ptbxl": 2.0})({"physionet": a, "ptbxl": b}, {"physionet": ya, "ptbxl": yb})
        la, _ = masked_bce_with_logits(a, ya)
        lb, _ = masked_bce_with_logits(b, yb)
        self.assertAlmostEqual(float(total), 0.5 * float(la) + 2.0 * float(lb), places=5)
        self.assertEqual({k: v[1] for k, v in parts.items()}, {"physionet": 135, "ptbxl": 10})

    def test_pos_weight_cap_floor_and_empty_class(self):
        y = np.zeros((100, 4))
        y[:50, 0] = 1          # balanced -> 1
        y[:1, 1] = 1           # 99/1 -> capped at 50
        y[:90, 2] = 1          # 10/90 -> floored at 1
        w = pos_weight_from_labels(y)                                   # class 3 has no positives -> cap
        self.assertEqual(w.tolist(), [1.0, 50.0, 1.0, 50.0])
        y[0, 0] = np.nan
        self.assertTrue(np.isfinite(pos_weight_from_labels(y)).all())

    def test_cache_roundtrip_error_is_small_on_zscored_signals(self):
        x = np.random.RandomState(0).randn(12, 5000).astype(np.float32)
        self.assertLess(dv.cache_error(x), 3e-3)

    def test_augmentation_is_deterministic_and_shapes_preserved(self):
        x = np.random.RandomState(1).randn(6, 12, 5000).astype(np.float32)
        x0 = x.copy()
        a = dv.paper_augment(x, np.random.RandomState(5))
        b = dv.paper_augment(x, np.random.RandomState(5))
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.shape, x.shape)
        self.assertFalse(np.array_equal(a, x))
        self.assertTrue(np.array_equal(x, x0))                                                      # input untouched
        z = dv.paper_augment(x, np.random.RandomState(2), p=0.0, p_lead_drop=1.0)
        self.assertTrue(all((np.abs(z[i]).sum(1) == 0).sum() in (1, 2, 3) for i in range(6)))

    def test_mixup_keeps_nan_where_a_source_label_is_missing(self):
        x = np.random.RandomState(0).randn(4, 12, 10).astype(np.float32)
        y = np.array([[1, np.nan], [0, 1], [1, 1], [0, 0]], np.float32)
        xm, ym, lam, perm = dv.mixup(x, y, np.random.RandomState(3))
        self.assertEqual(xm.shape, x.shape)
        for i in range(4):
            if np.isnan(y[i, 1]) or np.isnan(y[perm[i], 1]):
                self.assertTrue(np.isnan(ym[i, 1]))
        self.assertEqual(dv.mixup(x, y, np.random.RandomState(3), alpha=0)[2], 1.0)

    def test_sampling_weights_favour_rare_classes(self):
        y = np.zeros((100, 2))
        y[:90, 0] = 1
        y[90:92, 1] = 1
        w = dv.sampling_weights(y)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertGreater(w[90], w[0])


if __name__ == "__main__":
    unittest.main(verbosity=1)
