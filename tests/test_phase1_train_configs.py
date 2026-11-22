r"""Phase 1: configuration invariants and a CPU smoke test of the training driver on a tiny SYNTHETIC cache (throw-away random model; no real ECG, no project weights).
Run: .venv\Scripts\python.exe -I tests\test_phase1_train_configs.py"""
import copy
import csv
import glob
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_v3 as tv  # noqa: E402
from ecg_concept import provenance as pv  # noqa: E402
from ecg_concept.ptbxl_guard import Fold10Guard  # noqa: E402

CFG = sorted(glob.glob(os.path.join(ROOT, "experiments", "phase1", "*.json")))


def tiny_cache(d, n=48, c=5, with_eval_rows=True):
    rng = np.random.RandomState(0)
    t = np.arange(5000) / 500.0
    x = np.zeros((n, 12, 5000), np.float16)
    y = (rng.rand(n, c) < 0.4).astype(np.float32)
    y[0] = 1.0
    y[1] = 0.0
    for i in range(n):
        for l in range(12):
            x[i, l] = (np.sin(2 * np.pi * (1.0 + y[i, 0]) * t + l) + 0.1 * rng.randn(5000)).astype(np.float16)
    fold = np.arange(n) % 3
    if with_eval_rows:
        fold[-4:] = -1                                                           # evaluation-only rows: must never be trained or validated on
    np.save(os.path.join(d, "x.npy"), x)
    np.save(os.path.join(d, "y.npy"), y)
    with open(os.path.join(d, "meta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "inner_fold", "role"])
        w.writerows([[i, i, int(fold[i]), "train"] for i in range(n)])
    return fold


class ConfigTests(unittest.TestCase):
    def test_all_configs_parse_and_obey_protocol_invariants(self):
        self.assertGreaterEqual(len(CFG), 7)
        for p in CFG:
            with open(p, encoding="utf-8") as f:
                c = json.load(f)
            txt = json.dumps(c).lower()
            self.assertNotIn("fold10", txt, p)
            self.assertNotIn("fold9", txt.replace("fold 9", ""), p)
            self.assertIn("status", c, p)
            if "train" in c:
                t = c["train"]
                self.assertEqual(t["physical_batch"] * t["accumulation"], 256, p)
                self.assertEqual(t["precision"], "fp32", p)
                self.assertEqual(c["loss"]["pos_weight_cap"], 50.0, p)
                self.assertTrue(c["cache_dir"].startswith("data/processed/"), p)
                self.assertFalse(any(k in c["cache_dir"] for k in ("fold9", "fold10", "test", "validation_role")), p)
                if c["init"]["type"] != "scratch":
                    self.assertTrue(c["init"]["path"].startswith("models/pretrained/") or "EDIT_ME" in c["init"]["path"], p)

    def test_initialisation_from_reference_is_read_only_and_head_is_new(self):
        with open(os.path.join(ROOT, "experiments", "phase1", "B_ptbxl_init_v2.json"), encoding="utf-8") as f:
            c = json.load(f)
        before = pv.sha256_file(os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt"))
        m = tv.build_model(c, 5)
        self.assertEqual(tuple(m.head[3].weight.shape), (5, 256))
        self.assertEqual(pv.sha256_file(os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")), before)
        import torch
        ref = torch.load(os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt"), map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(m.state_dict()["tcn3b.pw.weight"], ref["tcn3b.pw.weight"]))

    def test_learning_rate_schedule_matches_paper_description(self):
        s = {"lr": 1e-3, "lr_min": 1e-5, "warmup_epochs": 5, "epochs": 120}
        spe = 100
        self.assertAlmostEqual(tv.lr_at(0, spe, s), 1e-3 / 500)
        self.assertAlmostEqual(tv.lr_at(5 * spe - 1, spe, s), 1e-3)
        self.assertAlmostEqual(tv.lr_at(5 * spe, spe, s), 1e-3)
        self.assertAlmostEqual(tv.lr_at(int(120 * spe) - 1, spe, s), 1e-5, delta=1e-7)
        self.assertAlmostEqual(tv.lr_at(int(62.5 * spe), spe, s), 1e-5 + 0.5 * (1e-3 - 1e-5), delta=2e-5)


class SmokeTest(unittest.TestCase):
    def tearDown(self):
        Fold10Guard.uninstall_hooks()

    def test_two_epoch_cpu_run_on_synthetic_cache(self):
        d, out = tempfile.mkdtemp(), tempfile.mkdtemp()
        fold = tiny_cache(d)
        with open(os.path.join(ROOT, "experiments", "phase1", "B_ptbxl_scratch.json"), encoding="utf-8") as f:
            c = json.load(f)
        c = copy.deepcopy(c)
        c.update(cache_dir=d, name="smoke")
        c["train"].update(epochs=2, patience=10, eval_every=1, physical_batch=8, accumulation=1, warmup_epochs=1)
        before = pv.sha256_file(os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt"))
        x, y, meta, tr, va = tv.load_cache(d, 0)
        self.assertTrue(set(tr).isdisjoint(va))
        self.assertFalse(set(np.where(fold == -1)[0]) & (set(tr) | set(va)))                 # evaluation-only rows never reach training or inner validation
        self.assertTrue(all(fold[i] in (1, 2) for i in tr) and all(fold[i] == 0 for i in va))
        s = tv.run(c, 0, out_root=out, device="cpu")
        run_dir = s["run_dir"]
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "best.pt")))
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "run_manifest.json")))
        self.assertEqual(len(json.load(open(os.path.join(run_dir, "history.json")))), 2)
        self.assertEqual(np.load(os.path.join(run_dir, "oof_val_probs.npy")).shape, (len(va), 5))
        self.assertEqual(s["fold10_waveform_reads_allowed"], 0)
        self.assertNotEqual(os.path.dirname(os.path.abspath(run_dir)), os.path.join(ROOT, "models", "pretrained"))
        self.assertEqual(pv.sha256_file(os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")), before)
        # pooled OOF report on the same synthetic cache
        import eval_oof
        json.dump(["a", "b", "c", "d", "e"], open(os.path.join(d, "class_names.json"), "w"))
        rep_path = os.path.join(out, "report.json")
        old = sys.argv
        sys.argv = ["eval_oof", "--cache", d, "--runs", run_dir, "--out", rep_path, "--n-boot", "50"]
        try:
            eval_oof.main()
        finally:
            sys.argv = old
        rep = json.load(open(rep_path))
        self.assertEqual(rep["n_records"], len(va))
        self.assertEqual(len(rep["thresholds"]), 5)
        self.assertIn("macro_auroc_cluster_ci", rep)


if __name__ == "__main__":
    unittest.main(verbosity=1)
