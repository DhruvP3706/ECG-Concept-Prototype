"""Synthetic tests for the PTB-XL ingestion layer (no download, no WFDB, no real records). Run: .venv\\Scripts\\python.exe -I tests\\test_ptbxl.py"""
import csv
import os
import sys
import tempfile
import types
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ecg_concept import ptbxl as px  # noqa: E402

STATEMENTS = [  # code, diagnostic, form, rhythm, diagnostic_class, diagnostic_subclass
    ("NORM", 1.0, "", "", "NORM", "NORM"), ("IMI", 1.0, "", "", "MI", "IMI"), ("AMI", 1.0, "", "", "MI", "AMI"),
    ("LVH", 1.0, "", "", "HYP", "LVH"), ("NDT", 1.0, "", "", "STTC", "NDT"),
    ("ABQRS", "", 1.0, "", "", ""), ("SR", "", "", 1.0, "", ""), ("AFIB", "", "", 1.0, "", ""),
]


def write_statements(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "description", "diagnostic", "form", "rhythm", "diagnostic_class", "diagnostic_subclass"])
        for code, d, fo, rh, dc, ds in STATEMENTS:
            w.writerow([code, "x", d, fo, rh, dc, ds])


def write_db(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ecg_id", "patient_id", "age", "scp_codes", "strat_fold", "filename_lr", "filename_hr"])
        for ecg_id, pid, scp, fold in rows:
            w.writerow([ecg_id, f"{pid}.0", 50, scp, fold, f"records100/{ecg_id:05d}_lr", f"records500/{ecg_id:05d}_hr"])


def rec(ecg_id, pid, fold, scp=None):
    return px.PtbxlRecord(ecg_id, pid, fold, f"records500/{ecg_id:05d}_hr", f"records100/{ecg_id:05d}_lr", scp or {"NORM": 100.0})


class TestScpAndMetadata(unittest.TestCase):
    def test_parse_valid_and_rejects_code_execution(self):
        self.assertEqual(px.parse_scp_codes("{'NORM': 100.0, 'SR': 0.0}"), {"NORM": 100.0, "SR": 0.0})
        self.assertEqual(px.parse_scp_codes("{}"), {})
        for bad in ("__import__('os').system('echo x')", "[1, 2]", "{'A': 'x'}", "{'A': True}", "{'A': -1}", "{'A': float('nan')}", "{1: 2.0}", "", None, "{'A': "):
            with self.assertRaises(ValueError, msg=repr(bad)):
                px.parse_scp_codes(bad)

    def test_read_metadata_preserves_ids_folds_paths_and_raw_likelihoods(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ptbxl_database.csv")
            write_db(p, [(1, 15709, "{'NORM': 100.0, 'SR': 0.0}", 3), (2, 15709, "{'IMI': 35.0}", 3)])
            r = px.read_metadata(p)
            self.assertEqual([x.ecg_id for x in r], [1, 2])
            self.assertEqual((r[0].patient_id, r[0].strat_fold), (15709, 3))
            self.assertEqual(r[0].filename_hr, "records500/00001_hr")
            self.assertEqual(r[0].scp_codes, {"NORM": 100.0, "SR": 0.0})
            self.assertEqual(r[1].scp_raw, "{'IMI': 35.0}")

    def test_read_metadata_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.csv")
            write_db(p, [(1, 1, "{'NORM': 100.0}", 1), (1, 2, "{'NORM': 100.0}", 2)])
            with self.assertRaisesRegex(ValueError, "duplicate ecg_id"):
                px.read_metadata(p)
            q = os.path.join(d, "b.csv")
            with open(q, "w", newline="") as f:
                csv.writer(f).writerow(["ecg_id", "scp_codes"])
            with self.assertRaisesRegex(ValueError, "missing columns"):
                px.read_metadata(q)
            r = os.path.join(d, "c.csv")
            write_db(r, [(1, 1, "{'NORM': 100.0}", "2.5")])
            with self.assertRaisesRegex(ValueError, "strat_fold"):
                px.read_metadata(r)

    def test_read_statements(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "scp_statements.csv")
            write_statements(p)
            s = px.read_statements(p)
            self.assertTrue(s["IMI"]["diagnostic"] and not s["IMI"]["rhythm"])
            self.assertEqual((s["IMI"]["diagnostic_class"], s["IMI"]["diagnostic_subclass"]), ("MI", "IMI"))
            self.assertTrue(s["SR"]["rhythm"] and not s["SR"]["diagnostic"])
            self.assertIsNone(s["SR"]["diagnostic_class"])


class TestLabels(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.csv")
            write_statements(p)
            self.st = px.read_statements(p)
        self.recs = [rec(1, 1, 1, {"NORM": 100.0, "SR": 0.0}), rec(2, 2, 1, {"IMI": 35.0, "AMI": 0.0, "ABQRS": 0.0}), rec(3, 3, 2, {"IMI": 100.0, "LVH": 80.0, "AFIB": 100.0})]

    def kw(self, **o):
        base = dict(min_likelihood=50, unknown_as_positive=False)
        base.update(o)
        return base

    def test_likelihood_options_are_required(self):
        with self.assertRaises(TypeError):
            px.build_targets(self.recs, self.st, ["NORM"], level="statement")
        with self.assertRaises(TypeError):
            px.derive_vocabulary(self.recs, self.st, level="statement", min_likelihood=50)
        with self.assertRaises(ValueError):
            px.build_targets(self.recs, self.st, ["NORM"], level="statement", min_likelihood=0, unknown_as_positive=False)

    def test_statement_level_threshold_and_unknown_policy(self):
        v = ["AMI", "IMI", "LVH", "NORM"]
        Y = px.build_targets(self.recs, self.st, v, level="statement", **self.kw())
        self.assertEqual(Y.tolist(), [[0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 1, 0]])         # IMI 35 < 50 -> absent; AMI 0 (unknown) -> absent
        Y2 = px.build_targets(self.recs, self.st, v, level="statement", **self.kw(min_likelihood=30))
        self.assertEqual(Y2[1].tolist(), [0, 1, 0, 0])                                  # threshold is inclusive/configurable
        self.assertEqual(px.build_targets(self.recs, self.st, v, level="statement", **self.kw(min_likelihood=35))[1, 1], 1)   # >= is inclusive
        Y3 = px.build_targets(self.recs, self.st, v, level="statement", **self.kw(unknown_as_positive=True))
        self.assertEqual(Y3[1].tolist(), [1, 0, 0, 0])                                  # unknown AMI counted only when asked; IMI 35 < 50 stays absent

    def test_categories_are_not_mixed_silently(self):
        with self.assertRaisesRegex(ValueError, "allow_mixed_categories"):
            px.derive_vocabulary(self.recs, self.st, level="statement", categories=("diagnostic", "rhythm"), **self.kw())
        v = px.derive_vocabulary(self.recs, self.st, level="statement", categories=("diagnostic", "rhythm"), allow_mixed_categories=True, **self.kw())
        self.assertEqual(v, ["AFIB", "IMI", "LVH", "NORM"])
        self.assertEqual(px.derive_vocabulary(self.recs, self.st, level="statement", categories=("rhythm",), **self.kw()), ["AFIB"])
        self.assertNotIn("AFIB", px.derive_vocabulary(self.recs, self.st, level="statement", **self.kw()))     # default = diagnostic only

    def test_superclass_and_subclass_levels_are_diagnostic_only(self):
        v = px.derive_vocabulary(self.recs, self.st, level="superclass", **self.kw())
        self.assertEqual(v, ["HYP", "MI", "NORM"])
        Y = px.build_targets(self.recs, self.st, v, level="superclass", **self.kw())
        self.assertEqual(Y.tolist(), [[0, 0, 1], [0, 0, 0], [1, 1, 0]])                # rhythm/form statements never create superclasses
        self.assertEqual(px.derive_vocabulary(self.recs, self.st, level="subclass", **self.kw()), ["IMI", "LVH", "NORM"])
        with self.assertRaises(ValueError):
            px.derive_vocabulary(self.recs, self.st, level="superclass", categories=("rhythm",), **self.kw())

    def test_unknown_code_policy_and_vocabulary_rules(self):
        bad = [rec(9, 9, 1, {"NOPE": 100.0})]
        with self.assertRaisesRegex(ValueError, "not in scp_statements"):
            px.derive_vocabulary(bad, self.st, level="statement", **self.kw())
        self.assertEqual(px.derive_vocabulary(bad, self.st, level="statement", unknown_codes="ignore", **self.kw()), [])
        with self.assertRaises(ValueError):
            px.build_targets(self.recs, self.st, [], level="statement", **self.kw())
        with self.assertRaises(ValueError):
            px.build_targets(self.recs, self.st, ["NORM", "NORM"], level="statement", **self.kw())

    def test_vocabulary_is_deterministic_and_support_filtered(self):
        a = px.derive_vocabulary(self.recs, self.st, level="statement", **self.kw())
        b = px.derive_vocabulary(list(reversed(self.recs)), self.st, level="statement", **self.kw())
        self.assertEqual(a, b)
        self.assertEqual(px.derive_vocabulary(self.recs, self.st, level="statement", min_support=2, **self.kw()), [])


class TestSplits(unittest.TestCase):
    def test_fold_assignment_and_counts(self):
        recs = [rec(i, i, f) for i, f in enumerate([1, 2, 3, 8, 8, 9, 10, 10], start=1)]
        s = px.make_splits(recs)
        self.assertEqual((len(s.train), len(s.val), s.n_test), (5, 1, 2))
        self.assertEqual({r.strat_fold for r in s.train}, {1, 2, 3, 8})
        self.assertTrue(all(r.strat_fold == 9 for r in s.val))
        self.assertEqual(px.record_role(rec(1, 1, 10)), "test")

    def test_invalid_folds_and_duplicate_ids(self):
        for fold in (0, 11, 3.5, -1, True, None):
            with self.assertRaises(ValueError, msg=repr(fold)):
                px.validate_records([rec(1, 1, fold)])
        with self.assertRaisesRegex(ValueError, "duplicate record IDs"):
            px.validate_records([rec(1, 1, 1), rec(1, 2, 2)])

    def test_patient_overlap_detected_across_roles_but_allowed_within_train(self):
        px.validate_records([rec(1, 7, 1), rec(2, 7, 5), rec(3, 8, 9), rec(4, 9, 10)])           # same patient inside train is fine
        for a, b in ((1, 9), (1, 10), (9, 10)):
            with self.assertRaisesRegex(ValueError, "more than one of train/val/test"):
                px.make_splits([rec(1, 7, a), rec(2, 7, b)])

    def test_test_records_are_locked_and_released_once(self):
        s = px.make_splits([rec(1, 1, 1), rec(2, 2, 9), rec(3, 3, 10)])
        self.assertFalse(hasattr(s, "test"))
        self.assertNotIn(3, [r.ecg_id for r in s.train + s.val])
        self.assertNotIn("3", repr(s).replace("test=1", ""))
        with self.assertRaises(PermissionError):
            s.open_test("yes")
        t = s.open_test(px.TEST_CONFIRMATION)
        self.assertEqual([r.ecg_id for r in t], [3])
        with self.assertRaisesRegex(PermissionError, "already released"):
            s.open_test(px.TEST_CONFIRMATION)
        self.assertEqual(len(s.test_ids_sha256), 64)

    def test_check_files_is_metadata_only_and_skips_test_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "records500"))
            for name in ("00001_hr.hea", "00001_hr.dat", "00002_hr.hea"):
                open(os.path.join(d, "records500", name), "w").close()
            recs = [rec(1, 1, 1), rec(2, 2, 2), rec(3, 3, 10)]
            miss = px.check_files(recs, d)
            self.assertEqual([os.path.basename(m) for m in miss], ["00002_hr.dat"])            # fold-10 files not examined
            self.assertEqual(len(px.check_files(recs, d, include_test=True)), 3)


class TestLeadsAndHeader(unittest.TestCase):
    def test_model_lead_order_comes_from_existing_source(self):
        self.assertEqual(px.MODEL_LEAD_ORDER, ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"))

    def test_reorder_by_name_with_shuffled_uppercase_header(self):
        names = ["V1", "AVL", "I", "V6", "AVR", "III", "II", "V4", "AVF", "V2", "V5", "V3"]
        d = np.tile(np.arange(12), (7, 1))                                                  # column j holds the value j
        out = px.reorder_leads(d, names)
        self.assertEqual(out.shape, (12, 7))
        for target_i, lead in enumerate(px.MODEL_LEAD_ORDER):
            self.assertTrue((out[target_i] == [n.lower() for n in names].index(lead.lower())).all(), lead)

    def test_lead_failures(self):
        names = list(px.MODEL_LEAD_ORDER)
        d = np.zeros((5, 12))
        with self.assertRaisesRegex(ValueError, "unexpected lead"):
            px.reorder_leads(d, names[:-1] + ["ECG"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            px.reorder_leads(d, names[:3] + ["AVR", "aVR"] + names[5:])
        with self.assertRaisesRegex(ValueError, "expected 12 leads"):
            px.reorder_leads(np.zeros((5, 11)), names[:11])
        with self.assertRaisesRegex(ValueError, "does not match"):
            px.reorder_leads(np.zeros((12, 5)), names)
        with self.assertRaisesRegex(ValueError, "missing leads"):                           # reachable with a custom target set
            px.reorder_leads(np.zeros((5, 3)), ["I", "II", "III"], target=("I", "II", "V1"))

    def good_info(self, **o):
        info = {"names": list(px.MODEL_LEAD_ORDER), "fs": 500, "sig_len": 5000, "gains": [1000.0] * 12, "baselines": [0] * 12, "units": ["mV"] * 12}
        info.update(o)
        return info

    def test_header_checks(self):
        h = px.check_header(self.good_info())
        self.assertEqual((h["fs"], h["gain"], h["baseline"], h["units"]), (500, 1000.0, 0, "mv"))
        for bad in ({"fs": 100}, {"sig_len": 1000}, {"gains": [1000.0] * 11 + [200.0]}, {"gains": [200.0] * 12}, {"baselines": [0] * 11 + [5]},
                    {"units": ["uV"] * 12}, {"units": ["mV"] * 11 + ["uV"]}, {"names": ["I"] * 11}):
            with self.assertRaises(ValueError, msg=str(bad)):
                px.check_header(self.good_info(**bad))
        relaxed = px.check_header(self.good_info(gains=[200.0] * 12), px.HeaderExpectation(gain=None))
        self.assertEqual(relaxed["gain"], 200.0)


class TestLoader(unittest.TestCase):
    def fake_wfdb(self, names=None, dtype=np.int32, fs=500, gain=1000.0, scale=1):
        names = names or ["V1", "AVL", "I", "V6", "AVR", "III", "II", "V4", "AVF", "V2", "V5", "V3"]
        d = (np.tile(np.arange(12), (5000, 1)) * scale).astype(dtype)
        r = types.SimpleNamespace(d_signal=d, sig_name=names, fs=fs, sig_len=5000, adc_gain=[gain] * 12, baseline=[0] * 12, units=["mV"] * 12)
        return types.SimpleNamespace(rdrecord=lambda path, physical=False: r), names

    def test_missing_wfdb_gives_clear_error_without_installing(self):
        with unittest.mock.patch("importlib.import_module", side_effect=ImportError("no wfdb")):
            with self.assertRaisesRegex(ImportError, "optional 'wfdb' package"):
                px.load_signal(rec(1, 1, 1), "/nonexistent")
        self.assertEqual(px.MODEL_LEAD_ORDER[0], "I")                                       # module itself imports without wfdb

    def test_load_reorders_and_returns_int16_with_header(self):
        fake, names = self.fake_wfdb()
        with unittest.mock.patch.object(px, "_require_wfdb", return_value=fake):
            s = px.load_signal(rec(1, 1, 1), "/root")
        self.assertEqual((s.signal.shape, s.signal.dtype), ((12, 5000), np.int16))
        for i, lead in enumerate(px.MODEL_LEAD_ORDER):
            self.assertTrue((s.signal[i] == [n.lower() for n in names].index(lead.lower())).all())
        self.assertEqual(s.header["gain"], 1000.0)
        self.assertEqual(s.original_lead_names, names)
        self.assertEqual(px.backbone_input(s).shape, (12, 5000))                             # existing preprocessing runs; equivalence NOT claimed

    def test_load_rejects_bad_dtype_range_header_and_test_fold(self):
        with unittest.mock.patch.object(px, "_require_wfdb", return_value=self.fake_wfdb(dtype=np.float64)[0]):
            with self.assertRaisesRegex(ValueError, "expected integer"):
                px.load_signal(rec(1, 1, 1), "/root")
        with unittest.mock.patch.object(px, "_require_wfdb", return_value=self.fake_wfdb(scale=5000)[0]):
            with self.assertRaisesRegex(ValueError, "int16 range"):
                px.load_signal(rec(1, 1, 1), "/root")
        with unittest.mock.patch.object(px, "_require_wfdb", return_value=self.fake_wfdb(fs=100)[0]):
            with self.assertRaisesRegex(ValueError, "500 Hz"):
                px.load_signal(rec(1, 1, 1), "/root")
        with unittest.mock.patch.object(px, "_require_wfdb", return_value=self.fake_wfdb(names=["I"] * 12)[0]):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                px.load_signal(rec(1, 1, 1), "/root")
        called = []
        fake = self.fake_wfdb()[0]
        fake.rdrecord = lambda *a, **k: called.append(1)
        with unittest.mock.patch.object(px, "_require_wfdb", return_value=fake):
            with self.assertRaises(PermissionError):
                px.load_signal(rec(5, 5, 10), "/root")
        self.assertEqual(called, [])                                                          # waveform reader never invoked for fold 10


if __name__ == "__main__":
    unittest.main(verbosity=2)
