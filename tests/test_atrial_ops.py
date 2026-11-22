r"""Tests for the atrial-region interventions: QRS preservation, exact non-target regions, region eligibility, flattening, sham magnitude matching, f-wave calibration, QC. Run: .venv\Scripts\python.exe -I tests\test_atrial_ops.py"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ecg_concept import atrial_ops as ao  # noqa: E402

T = 5000
BEATS = np.arange(300, 4800, 420)                                        # RR = 420 samples (840 ms)


def synth(beats=BEATS, p_amp=60.0, noise=2.0, seed=0):
    t = np.arange(T)
    rng = np.random.RandomState(seed)
    out = []
    for a in (1.0, 0.7, -0.8, 0.5, 1.2, 0.9, -0.6, 1.1, 1.3, 0.8, 0.6, 1.0):
        x = np.zeros(T)
        for b in beats:
            x += a * 1000 * np.exp(-0.5 * ((t - b) / 6.0) ** 2)                       # QRS
            x += a * p_amp * np.exp(-0.5 * ((t - (b - 100)) / 14.0) ** 2)              # P wave 100 samples before R
            x += 0.25 * a * 1000 * np.exp(-0.5 * ((t - (b + 90)) / 22.0) ** 2)         # T wave
        x += rng.randn(T) * noise
        out.append(x)
    return np.round(np.array(out)).astype(np.int16)


class T_(unittest.TestCase):
    def test_regions_eligibility_follows_the_rr_rule(self):
        regs, idx = ao.regions(BEATS)
        self.assertEqual(idx, list(range(1, len(BEATS))))                                # RR 420 >= 160 + 220
        for (a, e), i in zip(regs, idx):
            self.assertEqual((a, e), (BEATS[i] - 160, BEATS[i] - 50))
            self.assertGreaterEqual(a - BEATS[i - 1], 220)
            self.assertLess(e, BEATS[i] - 40)                                            # never touches the QRS window
        short = np.array([300, 600, 1000, 1400])                                         # first RR 300 < 380 -> beat 1 not eligible
        self.assertEqual(ao.regions(short)[1], [2, 3])

    def test_flatten_removes_deviation_keeps_ends_and_everything_else_exact(self):
        x = synth()
        regs, _ = ao.regions(BEATS)
        y = ao.to_int16(ao.gain_region(x, regs, 0.0))
        q = ao.qc(x, y, BEATS, regs)
        self.assertTrue(q["qrs_identical"] and q["outside_identical"] and q["shape_dtype_ok"] and q["amplitude_ok"])
        self.assertGreater(q["rms_modification"], 2.0)
        a, e = regs[2]
        seg = y[0, a:e].astype(float)
        line = np.linspace(seg[0], seg[-1], len(seg))
        self.assertLess(np.abs(seg - line).max(), 1.01)                                  # straight line up to int16 rounding
        self.assertEqual((int(y[0, a]), int(y[0, e - 1])), (int(x[0, a]), int(x[0, e - 1])))   # region ends unchanged -> no step

    def test_gain_is_linear_in_the_region_deviation(self):
        x = synth()
        regs, _ = ao.regions(BEATS)
        d0, d2 = ao.gain_region(x, regs, 0.0) - x, ao.gain_region(x, regs, 2.0) - x
        self.assertTrue(np.allclose(d2, -d0, atol=1e-9))                                  # s=2 adds exactly what s=0 removes
        self.assertTrue(np.allclose(ao.gain_region(x, regs, 1.0), x))                     # s=1 is the identity
        self.assertTrue(np.allclose(ao.gain_region(x, regs, 0.5) - x, 0.5 * d0))

    def test_sham_matches_modification_rms_per_lead_and_region_and_keeps_ends(self):
        x = synth()
        regs, _ = ao.regions(BEATS)
        mods = ao.modification(x, regs, 0.0)
        ys = ao.random_sham(x, regs, mods, np.random.RandomState(1))
        for (a, e), m in zip(regs, mods):
            d = ys[:, a:e] - x[:, a:e]
            for l in range(12):
                self.assertAlmostEqual(float(np.sqrt(np.mean(d[l] ** 2))), float(np.sqrt(np.mean(m[l] ** 2))), delta=1e-6 * max(1, float(np.sqrt(np.mean(m[l] ** 2)))) + 1e-6)
            self.assertLess(np.abs(d[:, 0]).max() + np.abs(d[:, -1]).max(), 3.0)             # (nearly) zero at the region ends
        q = ao.qc(x, ao.to_int16(ys), BEATS, regs, strict_amp=False)
        self.assertTrue(q["qrs_identical"] and q["outside_identical"])
        ys2 = ao.random_sham(x, regs, mods, np.random.RandomState(2))
        self.assertFalse(np.allclose(ys, ys2))                                           # independent patterns differ

    def test_sham_is_not_a_removal_of_the_p_wave(self):
        x = synth(p_amp=80.0, noise=0.5)
        regs, _ = ao.regions(BEATS)
        a, e = regs[3]
        ys = ao.random_sham(x, regs, ao.modification(x, regs, 0.0), np.random.RandomState(3))
        flat = ao.gain_region(x, regs, 0.0)
        p_orig = np.abs(x[1, a:e].astype(float) - np.linspace(x[1, a], x[1, e - 1], e - a)).max()
        p_flat = np.abs(flat[1, a:e] - np.linspace(flat[1, a], flat[1, e - 1], e - a)).max()
        self.assertLess(p_flat, 1e-9)
        self.assertGreater(p_orig, 40)
        self.assertGreater(np.abs(ys[1, a:e] - np.linspace(ys[1, a], ys[1, e - 1], e - a)).max(), 5.0)   # the sham leaves a non-flat region

    def test_fwave_has_requested_rms_band_and_zero_ends(self):
        rng = np.random.RandomState(4)
        f = ao.fwave_noise(110, 25.0, rng)
        self.assertAlmostEqual(float(np.sqrt(np.mean(f ** 2))), 25.0, places=6)
        self.assertLess(abs(f[0]) + abs(f[-1]), 5.0)
        x = synth()
        regs, _ = ao.regions(BEATS)
        tgt = np.full(12, 20.0)
        y = ao.synthetic_fwave(x, regs, tgt, np.random.RandomState(5))
        a, e = regs[1]
        dev = y[:, a:e] - np.linspace(y[:, a], y[:, e - 1], e - a, axis=1)
        self.assertTrue(np.allclose(np.sqrt(np.mean(dev ** 2, axis=1)), 20.0, rtol=0.2))
        self.assertTrue(ao.qc(x, ao.to_int16(y), BEATS, regs, strict_amp=False)["outside_identical"])

    def test_qc_detects_violations(self):
        x = synth()
        regs, _ = ao.regions(BEATS)
        y = x.copy()
        y[3, BEATS[4] + 5] += 1                                                          # touch a QRS window
        self.assertFalse(ao.qc(x, y, BEATS, regs)["qrs_identical"])
        y = x.copy()
        y[0, 20] += 1                                                                    # touch a sample outside every region
        self.assertFalse(ao.qc(x, y, BEATS, regs)["outside_identical"])
        y = x.copy()
        y[0, regs[1][0] + 50] = 30000                                                    # huge spike inside a region
        q = ao.qc(x, y, BEATS, regs)
        self.assertFalse(q["amplitude_ok"])
        self.assertGreater(q["max_region_diff_ratio"], 1.0)
        self.assertEqual(ao.qc(x, x, BEATS, regs)["rms_modification"], 0.0)               # an unchanged record is reported as unchanged

    def test_region_deviation_rms_calibration_helper(self):
        x = synth(p_amp=0.0, noise=10.0)
        regs, _ = ao.regions(BEATS)
        r = ao.region_deviation_rms(x, regs)
        self.assertTrue(np.all((r > 8) & (r < 18)))                               # noise sd 10, inflated by the two noisy anchor samples of the straight line
        self.assertTrue(np.isnan(ao.region_deviation_rms(x, [])).all())


if __name__ == "__main__":
    unittest.main(verbosity=1)
