"""Tests for beat-anchored time warping and the RR interventions. Run: .venv\\Scripts\\python.exe -I tests\\test_rr_warp.py"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ecg_concept import rr_warp as rw  # noqa: E402

T = 5000
BEATS = np.array([300, 640, 1100, 1400, 1950, 2300, 2900, 3200, 3700, 4300, 4700])


def synth(beats, amp=1000.0):
    t = np.arange(T)
    out = []
    for a in (1.0, 0.7, -0.8, 0.5, 1.2, 0.9, -0.6, 1.1, 1.3, 0.8, 0.6, 1.0):
        x = np.zeros(T)
        for b in beats:
            x += a * amp * np.exp(-0.5 * ((t - b) / 6.0) ** 2) + 0.3 * a * amp * np.exp(-0.5 * ((t - (b + 90)) / 22.0) ** 2)
        out.append(x)
    return np.round(np.array(out)).astype(np.int16)


class T_(unittest.TestCase):
    def test_identity_warp_is_exact(self):
        x = synth(BEATS)
        y = rw.to_int16(rw.warp(x, BEATS, BEATS))
        self.assertTrue(np.array_equal(x, y))

    def test_qrs_windows_are_copied_rigidly_and_land_at_new_positions(self):
        x = synth(BEATS)
        new = rw.shuffle_rr(BEATS, np.random.RandomState(0))
        self.assertTrue(rw.valid_beats(new))
        y = rw.to_int16(rw.warp(x, BEATS, new))
        for o, n in zip(BEATS, new):
            self.assertTrue(np.array_equal(x[:, o - 40:o + 41], y[:, n - 40:n + 41]))
            self.assertLessEqual(abs(int(np.argmax(np.abs(y[0, n - 30:n + 31]))) - 30), 1)                     # the spike sits at the new beat position
        self.assertTrue(rw.quality(x, y, BEATS, new)["qrs_windows_identical"])

    def test_order_count_and_duration_preserved(self):
        rng = np.random.RandomState(1)
        for new in (rw.shuffle_rr(BEATS, rng), rw.regularise(BEATS, 0.0), rw.regularise(BEATS, 0.5)):
            self.assertEqual(len(new), len(BEATS))
            self.assertTrue((np.diff(new) > 0).all())
            self.assertEqual((new[0], new[-1]), (BEATS[0], BEATS[-1]))                                       # total span unchanged
            tm = rw.time_map(BEATS, new)
            self.assertTrue((np.diff(tm) > 0).all())                                                          # monotone time map: no folding

    def test_shuffle_preserves_the_interval_distribution(self):
        new = rw.shuffle_rr(BEATS, np.random.RandomState(2))
        self.assertEqual(sorted(np.diff(new).tolist()), sorted(np.diff(BEATS).tolist()))
        self.assertFalse((np.diff(new) == np.diff(BEATS)).all())
        s0, s1 = rw.rr_stats(BEATS), rw.rr_stats(new)
        self.assertAlmostEqual(s0["cv"], s1["cv"], places=9)

    def test_regularisation_reduces_variability_and_keeps_mean(self):
        full, half_ = rw.regularise(BEATS, 0.0), rw.regularise(BEATS, 0.5)
        c0, c1, c2 = (rw.rr_stats(b)["cv"] for b in (BEATS, half_, full))
        self.assertLess(c2, 0.01)
        self.assertLess(c2, c1)
        self.assertLess(c1, c0)
        self.assertAlmostEqual(c1 / c0, 0.5, delta=0.08)
        self.assertLessEqual(abs(rw.rr_stats(full)["mean"] - rw.rr_stats(BEATS)["mean"]), 1e-9)

    def test_sham_matches_requested_magnitude_and_is_valid(self):
        target = 25.0
        new = rw.sham_smooth(BEATS, target, np.random.RandomState(3))
        self.assertTrue(rw.valid_beats(new))
        self.assertAlmostEqual(rw.displacement_rms(BEATS, new), target, delta=4.0)
        rel = rw.rel_rr_change_rms(BEATS, rw.regularise(BEATS, 0.0))
        nn = rw.sham_noise(BEATS, rel, np.random.RandomState(4))
        self.assertTrue(rw.valid_beats(nn))
        self.assertAlmostEqual(rw.rel_rr_change_rms(BEATS, nn), rel, delta=0.25 * rel)
        self.assertEqual((nn[0], nn[-1]), (BEATS[0], BEATS[-1]))

    def test_overlapping_windows_and_short_rr_are_rejected(self):
        with self.assertRaises(ValueError):
            rw.time_map(BEATS, np.array([300, 340, 1100, 1400, 1950, 2300, 2900, 3200, 3700, 4300, 4700]))
        self.assertFalse(rw.valid_beats(np.array([300, 380, 1000])))

    def test_quality_metrics_detect_distortion(self):
        x = synth(BEATS)
        mild = rw.regularise(BEATS, 0.9)
        strong = rw.regularise(BEATS, 0.0)
        qm = rw.quality(x, rw.to_int16(rw.warp(x, BEATS, mild)), BEATS, mild)
        qs = rw.quality(x, rw.to_int16(rw.warp(x, BEATS, strong)), BEATS, strong)
        qi = rw.quality(x, x, BEATS, BEATS)
        self.assertAlmostEqual(qi["post_qrs_corr_median"], 1.0, places=6)
        self.assertLessEqual(qi["junction_ratio_max"], 1.0)
        self.assertGreaterEqual(qm["post_qrs_corr_median"], qs["post_qrs_corr_median"] - 1e-9)
        self.assertGreater(qs["rms_log_stretch"], qm["rms_log_stretch"])

    def test_rigid_st_t_and_p_regions_are_copied_and_only_the_gap_is_stretched(self):
        x = synth(BEATS)
        new = rw.shuffle_rr(BEATS, np.random.RandomState(0))
        y = rw.to_int16(rw.warp(x, BEATS, new, rigid_post=150, rigid_pre=50))
        n_rigid = 0
        fallback = rw.uniform_segments(BEATS, new, rigid_post=150, rigid_pre=50)
        for i in range(len(BEATS) - 1):
            if i not in fallback:
                o, n = BEATS[i], new[i]
                self.assertTrue(np.array_equal(x[:, o + 41:o + 41 + 150], y[:, n + 41:n + 41 + 150]))                      # ST-T region identical
                o2, n2 = BEATS[i + 1], new[i + 1]
                self.assertTrue(np.array_equal(x[:, o2 - 40 - 50:o2 - 40], y[:, n2 - 40 - 50:n2 - 40]))                    # P region identical
                n_rigid += 1
        self.assertGreater(n_rigid, 0)
        tm = rw.time_map(BEATS, new, rigid_post=150, rigid_pre=50)
        self.assertTrue((np.diff(tm) > 0).all())
        self.assertEqual((new[0], new[-1]), (BEATS[0], BEATS[-1]))
        q = rw.quality(x, y, BEATS, new, rigid_post=150, rigid_pre=50)
        self.assertTrue(q["qrs_windows_identical"])

    def test_short_segments_fall_back_to_uniform_stretching(self):
        old = np.array([300, 700, 880, 1300])
        new = np.array([300, 700 + 40, 880 + 30, 1300])
        self.assertIn(1, rw.uniform_segments(old, new, rigid_post=150, rigid_pre=50))
        tm = rw.time_map(old, new, rigid_post=150, rigid_pre=50)
        self.assertTrue((np.diff(tm) > 0).all())

    def test_warping_a_smooth_signal_does_not_create_jumps(self):
        t = np.arange(T)
        x = np.tile(np.round(800 * np.sin(2 * np.pi * t / 700) + 200 * np.sin(2 * np.pi * t / 90)), (12, 1)).astype(np.int16)
        new = rw.shuffle_rr(BEATS, np.random.RandomState(5))
        y = rw.warp(x, BEATS, new)
        self.assertLess(np.abs(np.diff(y, axis=1)).max(), 3.0 * np.abs(np.diff(x.astype(float), axis=1)).max())
        self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main(verbosity=1)
