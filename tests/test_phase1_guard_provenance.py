r"""Phase 1: fold-10 persistent guard, checkpoint immutability, provenance, preprocessing parity and bit-identical reference inference.
No fold-10 waveform is ever opened: guard tests use temporary fake files; real-manifest tests only compare path strings.
Run: .venv\Scripts\python.exe -I tests\test_phase1_guard_provenance.py"""
import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import data_v3 as dv  # noqa: E402
from ecg_concept import provenance as pv  # noqa: E402
from ecg_concept.inference import load_model  # noqa: E402
from ecg_concept.preprocessing import load_ecg, preprocess, read_raw_mat  # noqa: E402
from ecg_concept.ptbxl_guard import TOKEN_ENV, Fold10Guard  # noqa: E402

REAL_TEST_MAN = os.path.join(ROOT, "benchmarks", "ptbxl_v1", "manifest_test_reserved.csv")
TRAIN_MAN = os.path.join(ROOT, "benchmarks", "ptbxl_v1", "manifest_train_val.csv")
CKPT = os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")
DS = os.path.join(ROOT, "data", "raw", "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0")
PTBXL = os.path.join(ROOT, "data", "raw", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")


def sha(b):
    return hashlib.sha256(b).hexdigest()


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.man = os.path.join(self.d, "man.csv")
        with open(self.man, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ecg_id", "filename_hr"])
            w.writerow([1, "records500/21000/21999_hr"])
        self.protocol = os.path.join(self.d, "protocol.md")
        with open(self.protocol, "w") as f:
            f.write("protocol v1")
        self.fake = os.path.join(self.d, "ptbxl", "records500", "21000", "21999_hr.dat")
        self.ok = os.path.join(self.d, "ptbxl", "records500", "00000", "00001_hr.dat")
        for p in (self.fake, self.ok):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(b"FAKE")
        self.log, self.rel = os.path.join(self.d, "log.jsonl"), os.path.join(self.d, "release.json")
        self.g = Fold10Guard(self.man, self.log, self.rel, self.protocol)

    def tearDown(self):
        Fold10Guard.uninstall_hooks()
        os.environ.pop(TOKEN_ENV, None)

    def _release(self, token="tok", protocol_sha=None):
        with open(self.rel, "w") as f:
            json.dump({"approved_by": "tester", "approval_text": "ok", "protocol_sha256": protocol_sha or sha(open(self.protocol, "rb").read()), "config_sha256": "c" * 64, "token_sha256": sha(token.encode())}, f)

    def test_membership_is_path_based_across_separators(self):
        self.assertTrue(self.g.is_fold10(self.fake))
        self.assertTrue(self.g.is_fold10("X:\\data\\records500\\21000\\21999_hr.hea"))
        self.assertFalse(self.g.is_fold10(self.ok))
        self.assertFalse(self.g.is_fold10("records500/21000/21999_lr.dat"))

    def test_denied_by_default_logged_and_hash_chain_verifies(self):
        with self.assertRaises(PermissionError):
            self.g.check(self.fake)
        self.g.check(self.ok)                                                                # allowed, not logged (not fold 10)
        lines = [json.loads(l) for l in open(self.log)]
        self.assertEqual([l["event"] for l in lines], ["READ_DENIED"])
        self.assertTrue(self.g.verify_log())
        txt = open(self.log).read().replace("READ_DENIED", "READ_ALLOWED")
        open(self.log, "w").write(txt)
        self.assertFalse(Fold10Guard(self.man, self.log, self.rel, self.protocol).verify_log())          # tampering detected

    def test_log_persists_across_guard_instances(self):
        for _ in range(2):
            with self.assertRaises(PermissionError):
                Fold10Guard(self.man, self.log, self.rel, self.protocol).check(self.fake)
        self.assertEqual(len(open(self.log).read().strip().splitlines()), 2)
        self.assertTrue(Fold10Guard(self.man, self.log, self.rel, self.protocol).verify_log())

    def test_hooks_block_open_os_open_and_fromfile(self):
        self.g.install_hooks()
        with self.assertRaises(PermissionError):
            open(self.fake, "rb")
        with self.assertRaises(PermissionError):
            os.open(self.fake, os.O_RDONLY)
        with self.assertRaises(PermissionError):
            np.fromfile(self.fake, dtype=np.uint8)
        with open(self.ok, "rb") as f:                                                       # other files unaffected
            self.assertEqual(f.read(), b"FAKE")
        denied = [json.loads(l)["event"] for l in open(self.log, "rb").read().decode().splitlines()]
        Fold10Guard.uninstall_hooks()
        self.assertEqual(denied, ["READ_DENIED"] * 3)
        with open(self.fake, "rb") as f:                                                     # hooks removed (and no release) -> raw access works again: guard is not an OS barrier
            self.assertEqual(f.read(), b"FAKE")

    def test_release_requires_every_condition_and_is_one_shot(self):
        with self.assertRaises(PermissionError):
            self.g.open_release_session()                                                    # no release file
        self._release("tok")
        with self.assertRaises(PermissionError):
            self.g.open_release_session()                                                    # token env missing
        os.environ[TOKEN_ENV] = "wrong"
        with self.assertRaises(PermissionError):
            self.g.open_release_session()
        os.environ[TOKEN_ENV] = "tok"
        self._release("tok", protocol_sha="0" * 64)
        with self.assertRaises(PermissionError):
            self.g.open_release_session()                                                    # protocol changed after approval
        self._release("tok")
        with self.assertRaises(PermissionError):
            self.g.check(self.fake)                                                          # approved but no session opened yet
        self.g.open_release_session()
        self.g.check(self.fake)                                                              # allowed inside the session and logged
        self.assertEqual(self.g.n_allowed_reads(), 1)
        with self.assertRaises(PermissionError):
            Fold10Guard(self.man, self.log, self.rel, self.protocol).open_release_session()   # a second session (even from a new process) is refused
        g2 = Fold10Guard(self.man, self.log, self.rel, self.protocol)
        with self.assertRaises(PermissionError):
            g2.check(self.fake)                                                              # new process without a session cannot read
        self.assertTrue(g2.verify_log())

    def test_real_manifest_lists_all_2198_fold10_records_without_opening_files(self):
        g = Fold10Guard()
        self.assertEqual(len(g.keys), 2198)
        with open(REAL_TEST_MAN, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(g.is_fold10(os.path.join(PTBXL, r["filename_hr"] + ".dat")) for r in rows))
        with open(TRAIN_MAN, newline="", encoding="utf-8") as f:
            trows = list(csv.DictReader(f))
        self.assertFalse(any(g.is_fold10(os.path.join(PTBXL, r["filename_hr"] + ".dat")) for r in trows))
        self.assertTrue(all(int(r["strat_fold"]) == 10 for r in rows))
        self.assertTrue(all(int(r["strat_fold"]) <= 9 for r in trows))


class ProvenanceTests(unittest.TestCase):
    def test_reference_registry_matches_files_and_reference_checkpoint_hash(self):
        ok = pv.verify_registry()
        self.assertEqual(ok["models/pretrained/lightv2_ft_fold0.pt"], "7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72")
        self.assertEqual(pv.sha256_file(CKPT), ok["models/pretrained/lightv2_ft_fold0.pt"])

    def test_registry_detects_modification(self):
        d = tempfile.mkdtemp()
        f = os.path.join(d, "a.bin")
        open(f, "wb").write(b"x")
        reg = os.path.join(d, "reg.json")
        json.dump({"files": {"a.bin": sha(b"x")}}, open(reg, "w"))
        self.assertEqual(list(pv.verify_registry(reg, d)), ["a.bin"])
        open(f, "wb").write(b"y")
        with self.assertRaises(RuntimeError):
            pv.verify_registry(reg, d)
        os.remove(f)
        with self.assertRaises(FileNotFoundError):
            pv.verify_registry(reg, d)

    def test_new_outputs_cannot_target_reference_files_or_pretrained_dir(self):
        with self.assertRaises(PermissionError):
            pv.assert_not_reference_output(CKPT)
        with self.assertRaises(PermissionError):
            pv.assert_not_reference_output(os.path.join(ROOT, "models", "pretrained", "new.pt"))
        pv.assert_not_reference_output(os.path.join(ROOT, "results", "phase1_runs", "x", "best.pt"))

    def test_run_manifest_contains_hashes_and_refuses_overwrite(self):
        d = os.path.join(tempfile.mkdtemp(), "run")
        m = pv.write_run_manifest(d, {"a": 1}, [42], ["src/ecg_concept/provenance.py"])
        self.assertEqual(m["seeds"], [42])
        self.assertEqual(len(m["code_sha256"]["src/ecg_concept/provenance.py"]), 64)
        self.assertIn("models/pretrained/lightv2_ft_fold0.pt", m["reference_files_verified"])
        with self.assertRaises(FileExistsError):
            pv.write_run_manifest(d, {"a": 1}, [42], [])

    def test_reference_inference_bit_identical_across_loads_and_matches_stored_logits(self):
        z = np.load(os.path.join(ROOT, "results", "step4_4", "cache", "merged.npz"))
        ids = [str(i) for i in z["ids"]]
        role = {}
        with open(os.path.join(ROOT, "results", "step3_7", "prospective_role_manifest.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["role"] == "train":
                    role[r["record_id"]] = r
        pick = [i for i in ids if i in role][:4]
        self.assertEqual(len(pick), 4)
        paths = {}
        for root, _, fs in os.walk(os.path.join(DS, "WFDBRecords")):
            for fn in fs:
                if fn.endswith(".mat") and fn[:-4] in pick:
                    paths[fn[:-4]] = os.path.join(root, fn)
        x = torch.from_numpy(np.stack([load_ecg(paths[i]) for i in pick]))
        m1, m2 = load_model(CKPT, "cpu"), load_model(CKPT, "cpu")
        with torch.inference_mode():
            a, b = m1(x), m2(x)
        self.assertTrue(torch.equal(a, b))
        stored = np.stack([z["orig_logits"][ids.index(i)] for i in pick])
        self.assertLess(float(np.abs(a.numpy() - stored).max()), 1e-3)                        # CPU fp32 vs stored (GPU) logits

    def test_preprocessing_parity_and_float16_cache_on_training_records(self):
        with open(TRAIN_MAN, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["role"] == "train" and r["excluded"] in ("False", "")][:2]
        self.assertTrue(all(int(r["strat_fold"]) <= 8 for r in rows))
        from ecg_concept.ptbxl_reader import read_record
        for r in rows:
            sig, _ = read_record(os.path.join(PTBXL, r["filename_hr"]))
            x = preprocess(sig)
            self.assertEqual((x.shape, x.dtype), ((12, 5000), np.float32))
            self.assertTrue(np.array_equal(x, preprocess(sig.copy())))                       # deterministic
            self.assertLess(dv.cache_error(x), 5e-3)
        role = []
        with open(os.path.join(ROOT, "results", "step3_7", "prospective_role_manifest.csv"), encoding="utf-8") as f:
            role = [r["record_id"] for r in csv.DictReader(f) if r["role"] == "train"][:2]
        for root, _, fs in os.walk(os.path.join(DS, "WFDBRecords")):
            for fn in fs:
                if fn.endswith(".mat") and fn[:-4] in role:
                    p = os.path.join(root, fn)
                    self.assertTrue(np.array_equal(load_ecg(p), preprocess(read_raw_mat(p))))
                    self.assertLess(dv.cache_error(load_ecg(p)), 5e-3)


if __name__ == "__main__":
    unittest.main(verbosity=1)
