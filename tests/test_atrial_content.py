r"""Tests for Experiment 9 operations (deviation transplant, reversal, shuffling, spectrum-matched sham, masks). Run: .venv\Scripts\python.exe -I tests\test_atrial_content.py"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import atrial_content as ac  # noqa: E402
from ecg_concept import atrial_ops as ao  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_atrial_ops import BEATS, synth  # noqa: E402


class T_(unittest.TestCase):
    def setUp(self):
        self.x = synth(p_amp=60.0, noise=3.0, seed=1)
        self.regs, _ = ao.regions(BEATS)
        self.old = ac.region_devs(self.x, self.regs)

    def check_untouched(self, y, mask=None):
        q = ao.qc(self.x, ao.to_int16(y), BEATS, self.regs, strict_amp=False)
        self.assertTrue(q["qrs_identical"] and q["outside_identical"] and q["shape_dtype_ok"])
        for (a, e) in self.regs:
            self.assertTrue(np.allclose(y[:, a], self.x[:, a]) and np.allclose(y[:, e - 1], self.x[:, e - 1]))   # region ends unchanged -> no step

    def test_flat_equals_experiment8_flatten(self):
        y = ac.with_devs(self.x, self.regs, [np.zeros_like(d) for d in self.old])
        self.assertTrue(np.allclose(y, ao.gain_region(self.x, self.regs, 0.0)))

    def test_reverse_keeps_energy_and_is_involution(self):
        rev = ac.reverse_devs(self.old)
        for o, r in zip(self.old, rev):
            self.assertTrue(np.allclose(np.sum(o ** 2, 1), np.sum(r ** 2, 1)))
        self.assertTrue(all(np.allclose(a, b) for a, b in zip(ac.reverse_devs(rev), self.old)))
        self.check_untouched(ac.with_devs(self.x, self.regs, rev))

    def test_shuffle_is_permutation_of_blocks_deterministic_and_nontrivial(self):
        s1 = ac.shuffle_devs(self.old, np.random.RandomState(3))
        s2 = ac.shuffle_devs(self.old, np.random.RandomState(3))
        for o, a, b in zip(self.old, s1, s2):
            self.assertTrue(np.array_equal(a, b))
            self.assertFalse(np.array_equal(a, o))
            self.assertFalse(np.array_equal(a, o[:, ::-1]))
            self.assertLess(abs(np.sqrt(np.mean(a ** 2)) / np.sqrt(np.mean(o ** 2)) - 1), 0.75)   # energy roughly preserved (end ramp removal changes it; reported in the analysis)
        self.check_untouched(ac.with_devs(self.x, self.regs, s1), None)

    def test_donor_transplant_uses_recipient_baseline_and_donor_waveform(self):
        donor = synth(p_amp=-80.0, noise=6.0, seed=9)
        bd = np.stack(ac.region_devs(donor, self.regs))
        dd = ac.donor_devs(bd, len(self.regs), 2)
        y = ac.with_devs(self.x, self.regs, dd)
        self.check_untouched(y)
        a, e = self.regs[0]
        self.assertTrue(np.allclose(y[:, a:e] - ao._line(self.x[:, a:e].astype(float)), dd[0]))
        sc = ac.donor_devs(bd, 2, 0, scale=np.full(12, 0.5))
        self.assertTrue(np.allclose(sc[0], 0.5 * bd[0]))

    def test_sham_matches_rms_spectrum_ends_and_has_no_structure(self):
        donor = synth(p_amp=-80.0, noise=6.0, seed=9)
        new = ac.donor_devs(np.stack(ac.region_devs(donor, self.regs)), len(self.regs), 0)
        sh = ac.sham_devs(self.old, new, np.random.RandomState(5))
        for o, nw, s in zip(self.old, new, sh):
            m, ms = nw - o, s - o
            self.assertTrue(np.allclose(np.sqrt(np.mean(m ** 2, 1)), np.sqrt(np.mean(ms ** 2, 1)), rtol=1e-6))
            self.assertLess(np.abs(ms[:, 0]).max() + np.abs(ms[:, -1]).max(), 1e-6)          # zero ends
            self.assertFalse(np.allclose(m, ms))
        y_real = ao.to_int16(ac.with_devs(self.x, self.regs, new))
        y_sh = ao.to_int16(ac.with_devs(self.x, self.regs, sh))
        self.assertLess(ac.band_l1(self.x, y_real, y_sh, self.regs), 0.35)                    # similar band-energy profile
        sh2 = ac.sham_devs(self.old, new, np.random.RandomState(6))
        self.assertFalse(np.allclose(sh[0], sh2[0]))
        self.check_untouched(ac.with_devs(self.x, self.regs, sh))

    def test_lead_mask_leaves_other_leads_identical(self):
        mask = np.zeros(12, bool)
        mask[[1]] = True
        y = ac.with_devs(self.x, self.regs, [np.zeros_like(d) for d in self.old], mask)
        self.assertTrue(np.array_equal(y[~mask], self.x[~mask].astype(float)))
        self.assertFalse(np.allclose(y[1], self.x[1]))
        self.assertGreater(ac.masked_rms(self.x, y, self.regs, mask), ac.masked_rms(self.x, y, self.regs))

    def test_raw_substitution_changes_only_regions_and_can_step(self):
        donor = synth(p_amp=0.0, noise=1.0, seed=2) + 500
        segs = [donor[:, a:e] for a, e in self.regs]
        y = ac.raw_substitute(self.x, self.regs, segs)
        q = ao.qc(self.x, ao.to_int16(y), BEATS, self.regs, strict_amp=False)
        self.assertTrue(q["qrs_identical"] and q["outside_identical"])
        self.assertGreater(abs(y[0, self.regs[0][0]] - self.x[0, self.regs[0][0]]), 100)       # boundary step exists -> variant is flagged by QC


if __name__ == "__main__":
    unittest.main(verbosity=1)
