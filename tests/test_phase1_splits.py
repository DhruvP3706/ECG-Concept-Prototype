r"""Phase 1: split tests (metadata only; no waveform is opened). Run: .venv\Scripts\python.exe -I tests\test_phase1_splits.py"""
import collections
import csv
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import splits_v3 as sp  # noqa: E402

MAN = os.path.join(ROOT, "benchmarks", "ptbxl_v1", "manifest_train_val.csv")
ROLE = os.path.join(ROOT, "results", "step3_7", "prospective_role_manifest.csv")
DUP = os.path.join(ROOT, "results", "step3_7", "duplicate_group_manifest.csv")


class T_(unittest.TestCase):
    def test_ptbxl_inner_folds_real_manifest_patient_disjoint_and_exclude_fold9_and_10(self):
        with open(MAN, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["ecg_id"] = int(r["ecg_id"])
        inner = sp.ptbxl_inner_folds(rows)                                                  # raises if a patient spans two inner folds
        by_fold = {int(r["strat_fold"]) for r in rows if r["ecg_id"] in inner}
        self.assertEqual(by_fold, set(range(1, 9)))
        self.assertTrue(all(int(r["strat_fold"]) in range(1, 9) for r in rows if r["ecg_id"] in inner))
        self.assertEqual(len(inner), 17418)                                                 # all training-role records
        pats = collections.defaultdict(set)
        for r in rows:
            if r["ecg_id"] in inner:
                pats[r["patient_id"]].add(inner[int(r["ecg_id"])])
        self.assertTrue(all(len(v) == 1 for v in pats.values()))
        sizes = collections.Counter(inner.values())
        self.assertEqual(set(sizes), {0, 1, 2, 3})
        self.assertLess(max(sizes.values()) / min(sizes.values()), 1.1)
        self.assertEqual(sorted(sp.PTBXL_INNER_GROUPS.values()), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_ptbxl_inner_folds_detect_patient_leak(self):
        rows = [{"ecg_id": 1, "patient_id": 7, "strat_fold": 1}, {"ecg_id": 2, "patient_id": 7, "strat_fold": 3}]
        with self.assertRaises(ValueError):
            sp.ptbxl_inner_folds(rows)
        ok = [{"ecg_id": 1, "patient_id": 7, "strat_fold": 1}, {"ecg_id": 2, "patient_id": 7, "strat_fold": 2}, {"ecg_id": 3, "patient_id": 8, "strat_fold": 9}, {"ecg_id": 4, "patient_id": 9, "strat_fold": 10}]
        self.assertEqual(sp.ptbxl_inner_folds(ok), {1: 0, 2: 0})                            # folds 9 and 10 never receive an inner fold

    def test_group_kfold_keeps_groups_together_is_deterministic_and_balances(self):
        rng = np.random.RandomState(0)
        n = 600
        ids = [f"r{i}" for i in range(n)]
        groups = [f"g{i // 2}" if i < 100 else f"s{i}" for i in range(n)]               # 50 duplicate pairs + singletons
        labels = [[int(rng.randint(0, 5))] + ([5] if i % 40 == 0 else []) for i in range(n)]
        a = sp.group_kfold(ids, groups, 4, seed=3, labels=labels, n_classes=6)
        b = sp.group_kfold(ids, groups, 4, seed=3, labels=labels, n_classes=6)
        self.assertEqual(a, b)
        for g in {g for g in groups if g.startswith("g")}:
            self.assertEqual(len({a[i] for i, gg in zip(ids, groups) if gg == g}), 1)
        sizes = collections.Counter(a.values())
        self.assertLess(max(sizes.values()) / min(sizes.values()), 1.15)
        for c in range(6):                                                                    # every class appears in every fold
            self.assertEqual(len({a[i] for i, l in zip(ids, labels) if c in l}), 4)
        self.assertNotEqual(a, sp.group_kfold(ids, groups, 4, seed=4, labels=labels, n_classes=6))

    def test_physionet_training_role_duplicate_aware_folds(self):
        with open(ROLE, encoding="utf-8") as f:
            role = list(csv.DictReader(f))
        with open(DUP, encoding="utf-8") as f:
            dups = {r["record_id"]: r["dup_group_id"] for r in csv.DictReader(f)}
        train = [r for r in role if r["role"] == "train"]
        ids = [r["record_id"] for r in train]
        groups = [dups.get(r["record_id"]) or f"SINGLE-{r['record_id']}" for r in train]
        names = sorted({l for r in train for l in r["labels"].split("|") if l})
        cidx = {c: i for i, c in enumerate(names)}
        labels = [[cidx[l] for l in r["labels"].split("|") if l] for r in train]
        folds = sp.group_kfold(ids, groups, 3, seed=20261120, labels=labels, n_classes=len(names))
        self.assertEqual(len(folds), 35568)
        by_group = collections.defaultdict(set)
        for i, g in zip(ids, groups):
            by_group[g].add(folds[i])
        self.assertTrue(all(len(v) == 1 for v in by_group.values()))                         # duplicate partners never straddle folds
        self.assertLess(max(collections.Counter(folds.values()).values()) / min(collections.Counter(folds.values()).values()), 1.05)
        for c in names:                                                                       # classes with >=3 train records appear in all 3 folds
            n = sum(1 for l in labels if cidx[c] in l)
            if n >= 3:
                self.assertEqual(len({folds[i] for i, l in zip(ids, labels) if cidx[c] in l}), 3, c)
        # records of the historical validation / test_reserved / excluded roles are not part of the training development pool
        self.assertFalse({r["record_id"] for r in role if r["role"] != "train"} & set(folds))

    def test_duplicate_groups_helper(self):
        g = sp.duplicate_groups([{"record_id": "A", "dup_group_id": "DUP-1"}, {"record_id": "B", "dup_group_id": "DUP-1"}, {"record_id": "C", "dup_group_id": ""}])
        self.assertEqual(g["A"], g["B"])
        self.assertNotEqual(g["A"], g["C"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
