"""Engineering tests for the Step 4.3 RR-concept extractor (synthetic signals only).

Synthetic recovery and cross-detector agreement are ENGINEERING evidence, not clinical validation.
Run:  .venv\\Scripts\\python.exe -I tests\\test_rr_concepts.py
"""
import copy
import inspect
import os
import sys
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ecg_concept import beat_detection as bd  # noqa: E402
from ecg_concept import rr_concepts as rc  # noqa: E402

FS = 500
AMPS = [0.8, 1.0, 0.5, -0.9, 0.4, 0.7, -0.6, 1.2, 1.5, 1.4, 1.1, 0.9]       # mixed amplitudes and polarities across leads


def synth(times_s, n=5000, amp=1000.0, flip=1.0, sigma=0.012, noise=0.0, seed=0, wander=True, as_int=True):
    """12-lead int16-like array with Gaussian QRS spikes at times_s, T waves, baseline wander and optional noise."""
    t = np.arange(n) / FS
    rng = np.random.RandomState(seed)
    leads = []
    for a in AMPS:
        x = np.zeros(n)
        for tb in times_s:
            x += a * amp * flip * np.exp(-0.5 * ((t - tb) / sigma) ** 2)
            x += 0.25 * a * amp * flip * np.exp(-0.5 * ((t - (tb + 0.25)) / 0.04) ** 2)
        if wander:
            x += 0.2 * amp * np.sin(2 * np.pi * 0.3 * t + rng.rand())
        if noise:
            x += rng.normal(0, noise * amp, n)
        leads.append(x)
    return np.round(np.array(leads)).astype(np.int16) if as_int else np.array(leads)


def match_count(true_s, found_s, tol_s=0.03):
    pairs, ua, ub = bd.match_one_to_one(np.asarray(true_s), np.asarray(found_s), tol_s)
    return len(pairs), len(ua), len(ub)          # matched, missed truth, false detections


class TestFormulas(unittest.TestCase):
    def test_regular_and_irregular(self):
        t = np.concatenate([[1.0], 1.0 + np.cumsum([0.8, 1.2, 0.8, 1.2])])
        r = rc.rr_concepts(t)
        self.assertEqual((r["n_beats"], r["n_rr"]), (5, 4))
        self.assertAlmostEqual(r["rate_mean_bpm"], 60.0)
        self.assertAlmostEqual(r["rate_median_bpm"], 60.0)
        self.assertAlmostEqual(r["rr_cv"], np.sqrt(0.16 / 3) / 1.0)           # sample SD (ddof=1)
        self.assertAlmostEqual(r["rr_nrmssd"], 0.4)
        self.assertAlmostEqual(r["rr_mad_succ"], 0.4)
        reg = rc.rr_concepts(np.arange(0, 6.0, 1.0))
        self.assertAlmostEqual(reg["rr_cv"], 0.0)
        self.assertAlmostEqual(reg["rr_nrmssd"], 0.0)

    def test_mean_vs_median_differ_and_nothing_removed(self):
        t = np.concatenate([[0.0], np.cumsum([0.6, 0.6, 0.6, 2.4])])
        r = rc.rr_concepts(t)
        self.assertAlmostEqual(r["rate_mean_bpm"], 60 / 1.05)
        self.assertAlmostEqual(r["rate_median_bpm"], 100.0)
        self.assertEqual(r["n_rr"], 4)                                           # the 2.4 s interval is kept

    def test_missing_is_nan_never_zero(self):
        r1 = rc.rr_concepts([1.0])
        self.assertTrue(all(np.isnan(r1[k]) for k in ("rate_mean_bpm", "rate_median_bpm", "rr_cv", "rr_nrmssd", "rr_mad_succ")))
        r2 = rc.rr_concepts([0.0, 1.0, 2.0])                                     # 2 RR: rate yes, irregularity not
        self.assertFalse(np.isnan(r2["rate_mean_bpm"]))
        self.assertTrue(np.isnan(r2["rr_cv"]) and np.isnan(r2["rr_nrmssd"]) and np.isnan(r2["rr_mad_succ"]))
        r0 = rc.rr_concepts([])
        self.assertTrue(np.isnan(r0["rate_mean_bpm"]) and r0["n_beats"] == 0)


class TestMatchingAndFusion(unittest.TestCase):
    def test_one_to_one(self):
        pairs, ua, ub = bd.match_one_to_one([100, 200, 300], [110, 210, 500], 25)
        self.assertEqual(pairs, [(0, 0), (1, 1)])
        self.assertEqual((ua, ub), ([2], [2]))
        pairs, ua, ub = bd.match_one_to_one([100], [90, 110], 25)                 # two candidates, one slot
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(ub), 1)

    def test_one_to_one_property_random(self):
        rng = np.random.RandomState(1)
        for _ in range(50):
            a, b = np.sort(rng.randint(0, 5000, 12)), np.sort(rng.randint(0, 5000, 14))
            pairs, ua, ub = bd.match_one_to_one(a, b, 25)
            self.assertEqual(len({i for i, _ in pairs}), len(pairs))
            self.assertEqual(len({j for _, j in pairs}), len(pairs))
            self.assertTrue(all(abs(a[i] - b[j]) <= 25 for i, j in pairs))
            self.assertEqual(len(pairs) + len(ua), len(a))
            self.assertEqual(len(pairs) + len(ub), len(b))

    def test_fusion_no_chaining(self):
        per_lead = {l: [t] for l, t in enumerate([0, 40, 80, 120, 160])}          # each within 50 of its neighbour, but chain spans 160
        times, support = bd.fuse_leads(per_lead, tol=50, k_min=1, refractory=1)
        self.assertGreaterEqual(len(times), 2)                                    # a chain would have produced ONE cluster
        self.assertTrue(support.max() <= 2)
        times3, _ = bd.fuse_leads(per_lead, tol=50, k_min=3, refractory=1)
        self.assertEqual(len(times3), 0)                                          # nothing reaches 3 leads within 50 of its first member

    def test_fusion_consecutive_beats_not_merged(self):
        rng = np.random.RandomState(3)
        truth = np.arange(250, 4800, 300)                                         # 0.6 s spacing
        per_lead = {l: list(truth + rng.randint(-8, 9, truth.size)) for l in range(8)}
        times, support = bd.fuse_leads(per_lead, tol=25, k_min=3, refractory=100)
        self.assertEqual(len(times), len(truth))
        self.assertTrue((support == 8).all())


class TestDetectors(unittest.TestCase):
    def _check(self, times, **kw):
        sig = synth(times, **kw)
        out = rc.extract_record(sig, return_beats=True)
        inner = [t for t in times if 0.1 <= t <= 9.9]
        res = {}
        for det in ("A", "B", "consensus"):
            res[det] = match_count(inner, out["beats"][det])
        return out, res, len(inner)

    def test_regular_rates_and_polarity(self):
        for rate in (40, 60, 75, 120, 160):
            for flip in (1.0, -1.0):
                times = np.arange(0.8, 9.6, 60.0 / rate)
                out, res, n = self._check(times, flip=flip)
                for det in ("A", "B", "consensus"):
                    m, missed, extra = res[det]
                    self.assertEqual((m, missed, extra), (n, 0, 0), f"{det} rate={rate} flip={flip} {res[det]}")
                self.assertEqual(out["rate_state"], "ACCEPTED", f"rate={rate} flip={flip} {out['reason_codes']}")
                self.assertAlmostEqual(out["rate_mean_bpm_cons"], rate, delta=0.6)

    def test_irregular_spacing_not_masked(self):
        rng = np.random.RandomState(7)
        rr = rng.uniform(0.45, 1.3, 14)
        times = 0.7 + np.cumsum(rr)
        times = times[times < 9.7]
        out, res, n = self._check(times)
        for det in ("A", "B", "consensus"):
            self.assertEqual(res[det], (n, 0, 0), det)
        self.assertEqual(out["rate_state"], "ACCEPTED")
        self.assertEqual(out["irr_state"], "ACCEPTED")                            # irregular rhythm alone is never a mask
        truth = rc.rr_concepts(times[(times >= 0.1) & (times <= 9.9)])
        self.assertAlmostEqual(out["rr_cv_cons"], truth["rr_cv"], delta=0.01)
        self.assertAlmostEqual(out["rr_nrmssd_cons"], truth["rr_nrmssd"], delta=0.01)

    def test_wide_qrs_not_masked(self):
        times = np.arange(0.8, 9.6, 0.9)
        out, res, n = self._check(times, sigma=0.035)
        self.assertEqual(out["rate_state"], "ACCEPTED", out["reason_codes"])
        self.assertEqual(res["consensus"][0], n)

    def test_noise_engineering_recovery(self):
        times = np.arange(0.8, 9.6, 0.8)
        out, res, n = self._check(times, noise=0.08, seed=5)
        for det in ("A", "B", "consensus"):
            m, missed, extra = res[det]
            self.assertGreaterEqual(m, n - 1, f"{det} {res[det]}")
            self.assertLessEqual(extra, 1, f"{det} {res[det]}")

    def test_short_record_edges(self):
        times = [0.04, 0.9, 1.9, 2.96]                                            # first/last inside the 0.1 s edge zone
        sig = synth(times, n=1500)
        out = rc.extract_record(sig, return_beats=True)
        self.assertTrue(out["structural_ok"])
        inner = [0.9, 1.9]
        m, missed, extra = match_count(inner, out["beats"]["consensus"])
        self.assertEqual((m, missed), (2, 0))
        self.assertTrue(all(0.1 <= b <= 2.9 for b in out["beats"]["consensus"]))
        short = rc.extract_record(synth([0.5], n=800))                            # 1.6 s < 2 s minimum
        self.assertFalse(short["structural_ok"])
        self.assertIn("STRUCTURAL_SHAPE", short["reason_codes"])
        self.assertEqual(short["rate_state"], "UNAVAILABLE")

    def test_two_beats_is_insufficient_not_absence(self):
        out = rc.extract_record(synth([2.0]))                                     # one beat only
        self.assertEqual(out["rate_state"], "UNAVAILABLE")
        self.assertIn("INSUFFICIENT_BEATS", out["reason_codes"])
        self.assertTrue(np.isnan(out["rate_mean_bpm_cons"]))
        two = rc.extract_record(synth([2.0, 3.0]))
        self.assertNotEqual(two["rate_state"], "UNAVAILABLE")
        self.assertEqual(two["irr_state"], "UNAVAILABLE")
        self.assertIn("INSUFFICIENT_BEATS_FOR_IRREGULARITY", two["reason_codes"])


class TestQuality(unittest.TestCase):
    def test_flat_signal(self):
        out = rc.extract_record(np.zeros((12, 5000), dtype=np.int16))
        self.assertEqual(out["rate_state"], "UNAVAILABLE")
        self.assertEqual(out["irr_state"], "UNAVAILABLE")
        self.assertIn("FLAT_LEAD", out["reason_codes"])
        self.assertIn("INSUFFICIENT_USABLE_LEADS", out["reason_codes"])
        self.assertTrue(np.isnan(out["accepted_rate_mean_bpm"]) and np.isnan(out["rate_mean_bpm_cons"]))

    def test_some_flat_leads_still_processes(self):
        sig = synth(np.arange(0.8, 9.6, 0.8))
        sig[[0, 6, 7]] = 0
        out = rc.extract_record(sig)
        self.assertEqual(out["n_flat_leads"], 3)
        self.assertEqual(out["n_usable_leads"], 5)
        self.assertIn(out["rate_state"], ("ACCEPTED", "UNCERTAIN"))
        self.assertAlmostEqual(out["rate_mean_bpm_cons"], 75.0, delta=0.6)

    def test_clipping_indicator(self):
        raw = synth(np.arange(0.8, 9.6, 0.8), amp=60000.0, as_int=False)
        sig = np.clip(np.round(raw), -32768, 32767).astype(np.int16)             # ADC saturation at the int16 limits
        out = rc.extract_record(sig)
        self.assertGreater(out["n_clip_leads"], 0)
        self.assertIn("CLIPPING_SUSPECT", out["reason_codes"])
        self.assertGreaterEqual(out["n_beats_cons"], 8)                           # timing still recovered
        hw = np.clip(np.round(synth(np.arange(0.8, 9.6, 0.8), amp=30000.0, as_int=False)), -15000, 15000).astype(np.int16)
        self.assertGreater(rc.extract_record(hw)["n_clip_leads"], 0)              # plateau below the int16 limit with steep flanks
        clean = rc.extract_record(synth(np.arange(0.8, 9.6, 0.8)))
        self.assertEqual(clean["n_clip_leads"], 0)                                # smooth baseline minima are not clipping
        self.assertNotIn("CLIPPING_SUSPECT", clean["reason_codes"])

    def test_nonfinite(self):
        sig = synth(np.arange(0.8, 9.6, 0.8)).astype(np.float64)
        sig[1, 100] = np.nan
        out = rc.extract_record(sig)
        self.assertEqual(out["n_nonfinite_leads"], 1)
        self.assertIn("NONFINITE_LEAD", out["reason_codes"])
        self.assertNotEqual(out["rate_state"], "UNAVAILABLE")
        allnan = rc.extract_record(np.full((12, 5000), np.nan))
        self.assertEqual(allnan["rate_state"], "UNAVAILABLE")
        self.assertIn("INSUFFICIENT_USABLE_LEADS", allnan["reason_codes"])
        none = rc.extract_record(None)
        self.assertEqual(none["rate_state"], "UNAVAILABLE")
        self.assertIn("STRUCTURAL_SHAPE", none["reason_codes"])

    def test_detector_disagreement_marks_uncertain_not_deleted(self):
        sig = synth(np.arange(0.8, 9.6, 0.8))
        base = rc.extract_record(sig)
        self.assertEqual(base["rate_state"], "ACCEPTED")
        orig = rc.DETECTORS["B"]

        def drops_every_third(x, fs, c):                                          # a detector that misses some beats
            d = orig(x, fs, c)
            return [p for i, p in enumerate(d) if i % 3 != 2]

        with unittest.mock.patch.dict(rc.DETECTORS, {"B": drops_every_third}):
            out = rc.extract_record(sig, return_beats=True)
        self.assertIn("DETECTOR_DISAGREEMENT", out["reason_codes"])
        self.assertEqual(out["rate_state"], "UNCERTAIN")
        self.assertGreater(out["n_unmatched_A"], 0)
        self.assertEqual(out["n_unmatched_B"], 0)
        self.assertTrue(np.isnan(out["accepted_rate_mean_bpm"]))                  # uncertain values are not accepted values
        self.assertFalse(np.isnan(out["rate_mean_bpm_cons"]))                     # but the value is still reported separately
        self.assertEqual(out["n_beats_A"], base["n_beats_A"])                     # detector-specific output kept intact


class TestInterfaceAndDeterminism(unittest.TestCase):
    def test_interface_has_no_metadata_inputs(self):
        self.assertEqual(list(inspect.signature(rc.extract_record).parameters), ["signal", "c", "return_beats"])

    def test_determinism_and_duplicate_identity(self):
        sig = synth(np.arange(0.8, 9.6, 0.73), noise=0.05, seed=11)
        a = rc.extract_record(sig.copy(), return_beats=True)
        b = rc.extract_record(sig.copy(), return_beats=True)
        for k in a:
            if k in ("runtime_s",):
                continue
            if k == "beats":
                for d in a[k]:
                    np.testing.assert_array_equal(a[k][d], b[k][d])
            elif isinstance(a[k], float) and np.isnan(a[k]):
                self.assertTrue(np.isnan(b[k]))
            else:
                self.assertEqual(a[k], b[k], k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
