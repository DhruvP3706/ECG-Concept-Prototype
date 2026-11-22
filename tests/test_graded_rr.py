"""Tests for target-CV RR construction (Experiment 7): CV accuracy, interval validity, order/count/duration preservation, beat alignment and waveform preservation.
Run: .venv\\Scripts\\python.exe -I tests\\test_graded_rr.py"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ecg_concept import rr_warp as rw  # noqa: E402

T = 5000
REG = np.arange(250, 4800, 450)                                             # regular, CV = 0
IRR = np.array([200, 600, 1100, 1500, 2100, 2500, 3200, 3600, 4100, 4500, 4800])


def synth(beats):
    t = np.arange(T)
    out = []
    for a in (1.0, 0.7, -0.8, 0.5, 1.2, 0.9, -0.6, 1.1, 1.3, 0.8, 0.6, 1.0):
        x = np.zeros(T)
        for b in beats:
            x += a * 1000 * np.exp(-0.5 * ((t - b) / 6.0) ** 2) + 0.3 * a * 1000 * np.exp(-0.5 * ((t - (b + 90)) / 22.0) ** 2)
        out.append(x)
    return np.round(np.array(out)).astype(np.int16)


class TG(unittest.TestCase):
    def test_target_cv_is_met_within_tolerance_for_all_patterns(self):
        for kind in ("iid", "smooth", "perm_orig"):
            for target in (0.03, 0.08, 0.2):
                new, tries, why = rw.target_cv_beats(REG, target, kind, np.random.RandomState(1)) if kind != "perm_orig" else rw.target_cv_beats(IRR, target, kind, np.random.RandomState(1))
                self.assertIsNotNone(new, (kind, target, why))
                self.assertLessEqual(abs(rw.rr_cv(new) - target), max(0.005, 0.05 * target))

    def test_invariants_count_order_duration_mean_and_interval_validity(self):
        for kind in ("iid", "smooth", "perm_orig"):
            base = IRR
            new, _, _ = rw.target_cv_beats(base, 0.15, kind, np.random.RandomState(2))
            self.assertEqual(len(new), len(base))
            self.assertTrue((np.diff(new) >= rw.MIN_RR).all())
            self.assertEqual((new[0], new[-1]), (base[0], base[-1]))                              # total duration preserved
            self.assertAlmostEqual(float(np.diff(new).mean()), float(np.diff(base).mean()), places=9)   # mean RR preserved exactly (fixed ends)
            self.assertTrue((np.diff(new) > 0).all() and rw.valid_beats(new))

    def test_multiple_distinct_sequences_with_the_same_cv_are_reproducible(self):
        a = rw.target_cv_beats(REG, 0.1, "iid", np.random.RandomState(10))[0]
        b = rw.target_cv_beats(REG, 0.1, "iid", np.random.RandomState(11))[0]
        a2 = rw.target_cv_beats(REG, 0.1, "iid", np.random.RandomState(10))[0]
        self.assertTrue(np.array_equal(a, a2))                                                    # same seed -> same sequence
        self.assertFalse(np.array_equal(a, b))                                                    # different seed -> different sequence ...
        self.assertAlmostEqual(rw.rr_cv(a), rw.rr_cv(b), delta=0.01)                              # ... at the same CV

    def test_perm_orig_at_own_cv_reproduces_the_rr_multiset(self):
        cv0 = rw.rr_cv(IRR)
        new, _, _ = rw.target_cv_beats(IRR, cv0, "perm_orig", np.random.RandomState(3))
        a, b = np.sort(np.diff(new)), np.sort(np.diff(IRR))
        self.assertLessEqual(np.abs(a - b).max(), 2)                                              # same distribution up to rounding

    def test_impossible_targets_are_rejected_with_a_reason(self):
        fast = np.arange(100, 4000, 130)                                                         # mean RR 130 samples, floor is 100
        new, tries, why = rw.target_cv_beats(fast, 0.4, "iid", np.random.RandomState(4), max_tries=30)
        self.assertIsNone(new)
        self.assertEqual(why, "min_rr_violation")
        self.assertEqual(rw.target_cv_beats(np.array([100, 500, 900]), 0.1, "iid", np.random.RandomState(0))[2], "too_few_intervals")

    def test_waveform_preservation_and_beat_alignment_after_warping(self):
        x = synth(IRR)
        new, _, _ = rw.target_cv_beats(IRR, 0.04, "smooth", np.random.RandomState(5))
        y = rw.to_int16(rw.warp(x, IRR, new, rigid_post=150, rigid_pre=50))
        for o, n in zip(IRR, new):
            self.assertTrue(np.array_equal(x[:, o - 40:o + 41], y[:, n - 40:n + 41]))               # QRS windows bit-identical
            self.assertLessEqual(abs(int(np.argmax(np.abs(y[0, n - 30:n + 31]))) - 30), 1)           # spike at the new beat position
        q = rw.quality(x, y, IRR, new, rigid_post=150, rigid_pre=50)
        self.assertTrue(q["qrs_windows_identical"])
        self.assertLess(q["junction_ratio_max"], 5.0)
        self.assertGreater(q["std_ratio_min"], 0.85)
        self.assertLess(q["std_ratio_max"], 1.15)

    def test_increasing_then_decreasing_cv_moves_in_the_requested_direction(self):
        up = rw.target_cv_beats(REG, 0.12, "iid", np.random.RandomState(6))[0]
        down = rw.target_cv_beats(up, 0.02, "perm_orig", np.random.RandomState(7))[0]
        self.assertGreater(rw.rr_cv(up), rw.rr_cv(REG))
        self.assertLess(rw.rr_cv(down), rw.rr_cv(up))


if __name__ == "__main__":
    unittest.main(verbosity=1)
