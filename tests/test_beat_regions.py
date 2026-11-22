"""Tests for sample alignment of detected beats, region masks and attribution-mass bookkeeping. Run: .venv\\Scripts\\python.exe -I tests\\test_beat_regions.py"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ecg_concept import beat_regions as br  # noqa: E402
from ecg_concept import rr_concepts as rc  # noqa: E402

FS = 500


def synth(beat_samples, n=5000, amp=1000.0):
    t = np.arange(n)
    amps = [0.8, 1.0, 0.5, -0.9, 0.4, 0.7, -0.6, 1.2, 1.5, 1.4, 1.1, 0.9]
    out = []
    for a in amps:
        x = np.zeros(n)
        for b in beat_samples:
            x += a * amp * np.exp(-0.5 * ((t - b) / 6.0) ** 2)
            x += 0.25 * a * amp * np.exp(-0.5 * ((t - (b + 125)) / 20.0) ** 2)
        out.append(x)
    return np.round(np.array(out)).astype(np.int16)


class T(unittest.TestCase):
    def test_extractor_beat_indices_are_sample_aligned(self):
        truth = np.arange(300, 4700, 400)
        res = rc.extract_record(synth(truth), rc.CONFIG, return_beats=True)
        got = np.round(np.asarray(res["beats"]["consensus"]) * FS).astype(int)
        self.assertEqual(len(got), len(truth))
        self.assertLessEqual(np.abs(got - truth).max(), 3)                                   # within 6 ms of the true spike sample
        self.assertTrue(((got >= 0) & (got < 5000)).all())
        self.assertTrue((np.diff(got) > 0).all())

    def test_masks_partition_the_record(self):
        r = br.regions_from_beats([100, 600, 1100, 4990], 5000, 40)
        total = r["qrs"].astype(int) + r["rr"].astype(int) + r["out"].astype(int)
        self.assertTrue((total == 1).all())
        self.assertEqual(int(r["qrs"][4999]), 1)
        self.assertEqual(int(r["qrs"].sum()), 3 * 81 + 50)                  # last window clipped at the record end
        self.assertEqual(int(r["out"][:60].sum()), 60)                                         # before the first beat window: outside
        self.assertEqual(int(r["rr"][141:560].sum()), 560 - 141)                               # between windows: inter-beat

    def test_overlapping_windows_are_a_union_and_short_rr_has_no_segment(self):
        r = br.regions_from_beats([200, 250, 700], 5000, 40)
        self.assertEqual(int(r["qrs"][160:291].sum()), 131)                                   # 200-40 .. 250+40
        self.assertEqual(len(br.rr_segments([200, 250, 700], 5000, 40)), 1)                    # the 50-sample RR has no region left after removing both windows

    def test_fewer_than_two_beats_means_everything_outside(self):
        r = br.regions_from_beats([2500], 5000, 40)
        self.assertEqual(int(r["rr"].sum()), 0)
        self.assertEqual(int(r["out"].sum()), 5000 - 81)
        self.assertEqual(int(br.regions_from_beats([], 5000, 40)["out"].sum()), 5000)

    def test_tiles_do_not_overlap_and_stay_in_segments(self):
        beats = [100, 700, 1200]
        tiles = br.rr_tiles(beats, 5000, 40)
        reg = br.regions_from_beats(beats, 5000, 40)
        for s, e in tiles:
            self.assertEqual(e - s, 81)
            self.assertTrue(reg["rr"][s:e].all())
        self.assertTrue(all(tiles[i][1] <= tiles[i + 1][0] for i in range(len(tiles) - 1)))
        self.assertEqual(len(br.qrs_tiles(beats, 5000, 40)), 3)

    def test_attribution_mass_bookkeeping(self):
        a = np.random.RandomState(0).randn(12, 5000)
        reg = br.regions_from_beats(np.arange(300, 4700, 400), 5000, 40)
        absm, pos, neg = br.time_mass(a)
        self.assertTrue(np.allclose(absm, np.abs(a).sum(0)))
        self.assertTrue((neg <= 0).all() and (pos >= 0).all())
        s = br.shares(absm, reg)
        self.assertAlmostEqual(s["qrs"] + s["rr"] + s["out"], 1.0, places=9)
        self.assertAlmostEqual(s["qrs"], np.abs(a[:, reg["qrs"]]).sum() / np.abs(a).sum(), places=9)
        self.assertAlmostEqual(sum(br.duration_shares(reg).values()), 1.0, places=9)
        c = np.zeros((12, 5000))
        c[:, 100], c[:, 101] = 1.0, -1.0                                                       # cancelling signs: zero net, non-zero absolute mass
        self.assertEqual(float(c.sum()), 0.0)
        self.assertGreater(float(br.time_mass(c)[0].sum()), 0)

    def test_shift_control_preserves_sizes_and_random_windows_match(self):
        reg = br.regions_from_beats(np.arange(300, 4700, 400), 5000, 40)
        for sft in (250, 1234, 4000):
            sh = br.shifted(reg, sft)
            self.assertEqual(int(sh["qrs"].sum()), int(reg["qrs"].sum()))
            self.assertEqual(int(sh["rr"].sum()), int(reg["rr"].sum()))
        self.assertEqual(int(br.random_windows(11, 81, 5000, np.random.RandomState(1)).sum()), 11 * 81)

    def test_flat_attribution_gives_duration_proportional_shares(self):
        reg = br.regions_from_beats(np.arange(300, 4700, 400), 5000, 40)
        s, d = br.shares(np.ones(5000), reg), br.duration_shares(reg)
        for k in s:
            self.assertAlmostEqual(s[k], d[k], places=9)


if __name__ == "__main__":
    unittest.main(verbosity=1)
