r"""Tests for Experiment 10 operations (P-like localisation, patch extraction/placement, surrogate patch). Run: .venv\Scripts\python.exe -I tests\test_pwave_ops.py"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import pwave_ops as pw  # noqa: E402


def synth_beats(n=12, lag=-80, noise=2.0, seed=0, amp=40.0):
    rng = np.random.RandomState(seed)
    t = np.arange(pw.REG) + pw.START0
    out = []
    for _ in range(n):
        d = np.zeros((12, pw.REG))
        for l in range(12):
            d[l] = (amp * (1 + 0.1 * l)) * np.exp(-0.5 * ((t - lag) / 8.0) ** 2) + rng.randn(pw.REG) * noise
        d -= d[:, :1] + (d[:, -1:] - d[:, :1]) * np.linspace(0, 1, pw.REG)[None]
        out.append(d)
    return np.stack(out)


class T_(unittest.TestCase):
    def test_localize_finds_planted_lag_and_beat_correlation(self):
        D = synth_beats(lag=-80)
        r = pw.localize(D)
        self.assertLessEqual(abs(r["lag"] + 80), 2)
        self.assertGreater(r["snr"], 5)
        self.assertTrue((r["corr"] > 0.6).all())

    def test_noise_only_record_has_low_snr(self):
        D = synth_beats(amp=0.0, noise=5.0, seed=3)
        self.assertLess(pw.localize(D)["snr"], 5)

    def test_patch_contains_peak_and_tapers_to_zero(self):
        D = synth_beats(lag=-80)
        p = pw.patch_from(D[0], -80, 40)
        self.assertEqual(p.shape, (12, 40))
        self.assertLess(np.abs(p[:, 0]).max(), 1e-9)
        self.assertGreater(np.abs(p[:, 25]).max(), 20)                     # lag - pre = sample 25 of the patch

    def test_placement_limits_and_region_ends_stay_zero(self):
        self.assertIsNotNone(pw.placement(-80, 0))
        self.assertIsNone(pw.placement(-80, 40))                           # would run into the QRS protection margin
        self.assertIsNone(pw.placement(-80, -60))                          # would leave the region start
        s = pw.placement(-80, -30)
        d = pw.place(np.ones((12, 40)), s)
        self.assertEqual(d[:, :2].sum() + d[:, -2:].sum(), 0)
        self.assertEqual(int(d[0].sum()), 40)

    def test_shift_moves_the_peak_by_the_offset(self):
        D = synth_beats(lag=-80)
        p = pw.patch_from(D[0], -80, 40)
        for off in (-30, -15, 0, 8):
            s = pw.placement(-80, off)
            d = pw.place(p, s)
            self.assertLessEqual(abs(int(np.argmax(np.abs(d[3]))) + pw.START0 - (-80 + off)), 3)

    def test_surrogate_matches_rms_and_position_but_not_waveform(self):
        D = synth_beats(lag=-80)
        p = pw.patch_from(D[0], -80, 40)
        s = pw.surrogate_patch(p, np.random.RandomState(1))
        self.assertTrue(np.allclose(np.sqrt(np.mean(s ** 2, 1)), np.sqrt(np.mean(p ** 2, 1)), rtol=1e-6))
        self.assertLess(np.abs(s[:, 0]).max(), 1e-9)
        c = np.corrcoef(s[3], p[3])[0, 1]
        self.assertLess(abs(c), 0.9)

    def test_control_window_is_before_p_window(self):
        for lag in (-90, -70):
            pre, _ = pw.WIDTHS[40]
            self.assertLess(pw.CTL_START + 40, lag - pw.START0 - pre + 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
