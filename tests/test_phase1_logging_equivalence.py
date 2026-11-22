r"""Phase 1: the logging-only change to scripts/train_v3.py must not alter training. The pre-change driver is preserved as scripts/train_v3_original_fold0.py (SHA-256 66f16d26...).
Runs both drivers on the same tiny SYNTHETIC cache (throw-away models, no real ECG) on CPU (and CUDA when available) with the approved B configuration reduced only in size, and compares per-epoch training loss,
learning rate, validation macro-AUROC, saved checkpoint tensors and out-of-fold probabilities. Run: .venv\Scripts\python.exe -I tests\test_phase1_logging_equivalence.py"""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from ecg_concept import provenance as pv  # noqa: E402
from ecg_concept.ptbxl_guard import Fold10Guard  # noqa: E402
from test_phase1_train_configs import tiny_cache  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ORIG = load(os.path.join(ROOT, "scripts", "train_v3_original_fold0.py"), "tv_orig")
NEW = load(os.path.join(ROOT, "scripts", "train_v3.py"), "tv_new")


def run(mod, cfg, cache, device, seed_fold=0, global_seed=1234):
    out = tempfile.mkdtemp()
    c = copy.deepcopy(cfg)
    c["cache_dir"] = cache
    torch.manual_seed(global_seed)                                                    # the ORIGINAL driver builds the model before seeding (init = global RNG state); the NEW driver seeds first (init independent of this call)
    s = mod.run(c, seed_fold, out_root=out, device=device)
    d = s["run_dir"]
    hist = json.load(open(os.path.join(d, "history.json")))
    ck = torch.load(os.path.join(d, "best.pt"), map_location="cpu", weights_only=True)
    return hist, ck, np.load(os.path.join(d, "oof_val_probs.npy")), s


class T_(unittest.TestCase):
    def tearDown(self):
        Fold10Guard.uninstall_hooks()

    def test_original_driver_is_preserved_unchanged(self):
        self.assertEqual(pv.sha256_file(os.path.join(ROOT, "scripts", "train_v3_original_fold0.py")), "66f16d26044f35e2dd285a6a9437a319c316093c87d921c9fe2b20df583a063d")

    def test_config_file_unchanged(self):
        self.assertTrue(pv.sha256_file(os.path.join(ROOT, "experiments", "phase1", "B_ptbxl_scratch.json")).startswith("dc2cdae51b275b7a"))

    def _compare(self, device):
        cache = tempfile.mkdtemp()
        tiny_cache(cache)
        cfg = json.load(open(os.path.join(ROOT, "experiments", "phase1", "B_ptbxl_scratch.json")))
        cfg["name"] = "eq"
        cfg["train"].update(epochs=3, patience=10, eval_every=1, physical_batch=8, accumulation=2, warmup_epochs=1)
        seed0 = cfg["seed"]                                                              # fold 0 -> seed 42: seeding the global RNG with it before the original driver reproduces the new driver's initial weights
        h0, c0, p0, _ = run(ORIG, cfg, cache, device, global_seed=seed0)
        h0b, c0b, p0b, _ = run(ORIG, cfg, cache, device, global_seed=seed0)
        h1, c1, p1, _ = run(NEW, cfg, cache, device, global_seed=999)
        deterministic = all(float(a["train_loss"]) == float(b["train_loss"]) for a, b in zip(h0, h0b)) and all(torch.equal(c0[k], c0b[k]) for k in c0)
        tol = 0.0 if deterministic else 1e-4
        self.assertEqual(len(h0), len(h1))
        for a, b in zip(h0, h1):
            self.assertAlmostEqual(a["train_loss"], b["train_loss"], delta=tol)
            self.assertEqual(a["lr"], b["lr"])
            self.assertAlmostEqual(a["val_macro_auroc"], b["val_macro_auroc"], delta=tol)
        for k in c0:
            self.assertLessEqual(float((c0[k].float() - c1[k].float()).abs().max()), tol, k)
        self.assertLessEqual(float(np.abs(p0 - p1).max()), tol)
        for r in h1:                                                                      # new logging fields exist and are finite
            for k in ("val_loss_weighted", "val_loss_unweighted", "train_macro_auroc_eval", "train_loss_weighted_eval", "grad_norm_mean", "grad_norm_max", "grad_norm_frac_clipped"):
                self.assertTrue(np.isfinite(r[k]), k)
            self.assertEqual(r["grad_nonfinite_steps"], 0)
        self.assertNotIn("val_loss_weighted", h0[0])
        return deterministic

    def test_new_driver_init_is_reproducible_recorded_and_fold_specific(self):
        cache = tempfile.mkdtemp()
        tiny_cache(cache)
        with open(os.path.join(cache, "cache_manifest.json"), "w") as f:
            json.dump({"x_sha256": "abc"}, f)
        cfg = json.load(open(os.path.join(ROOT, "experiments", "phase1", "B_ptbxl_scratch.json")))
        cfg["name"] = "seedtest"
        cfg["train"].update(epochs=2, patience=10, eval_every=1, physical_batch=8, accumulation=1, warmup_epochs=1)
        out = {}
        for tag, gs, fold in (("a", 11, 0), ("b", 22, 0), ("c", 11, 1)):
            h, ck, p, s = run(NEW, cfg, cache, "cpu", seed_fold=fold, global_seed=gs)
            d = s["run_dir"]
            man = json.load(open(os.path.join(d, "run_manifest.json")))
            self.assertEqual(man["init_weights_sha256"], pv.sha256_file(os.path.join(d, "init.pt")))
            for k in ("seed_policy", "config_sha256", "code_sha256", "seeds", "cache_manifest_sha256", "x_cache_sha256_from_manifest"):
                self.assertIn(k, man)
            self.assertEqual(man["seeds"], [cfg["seed"] + fold])
            self.assertIn("scripts/train_v3.py", man["code_sha256"])
            out[tag] = (man["init_weights_sha256"], h, ck, p)
        self.assertEqual(out["a"][0], out["b"][0])                                          # same fold, different global RNG state -> identical initial weights
        self.assertNotEqual(out["a"][0], out["c"][0])                                       # different fold -> different seed -> different initial weights
        for a, b in zip(out["a"][1], out["b"][1]):                                          # whole CPU run (batch order, augmentation, dropout) reproducible
            self.assertEqual((a["train_loss"], a["val_macro_auroc"]), (b["train_loss"], b["val_macro_auroc"]))
        self.assertTrue(all(torch.equal(out["a"][2][k], out["b"][2][k]) for k in out["a"][2]))
        self.assertTrue(np.array_equal(out["a"][3], out["b"][3]))

    def test_cpu_training_identical_before_and_after_logging_change(self):
        self.assertTrue(self._compare("cpu"))                                              # CPU is deterministic: bit-identical required

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA")
    def test_cuda_training_identical_before_and_after_logging_change(self):
        det = self._compare("cuda")
        print("cuda original driver bitwise deterministic:", det)


if __name__ == "__main__":
    unittest.main(verbosity=1)
