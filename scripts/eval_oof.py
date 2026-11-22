r"""Pool the out-of-fold inner-validation predictions of one experiment (all inner folds) and report the Phase 1 metrics. Reads run directories written by train_v3.py and the SAME cache; never opens another partition.
Thresholds are selected on the pooled OOF predictions only (no fold 9 / fold 10 / historical test involvement). Patient/group-clustered bootstrap CI for macro-AUROC.
Usage: .venv\Scripts\python.exe -I scripts\eval_oof.py --cache data\processed\ptbxl_train_v1 --runs <run_dir_fold0> <run_dir_fold1> ... --out results\phase1_runs\B_report.json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import eval_metrics as em  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    y_all = np.load(os.path.join(a.cache, "y.npy"))
    with open(os.path.join(a.cache, "meta.csv"), newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    names = json.load(open(os.path.join(a.cache, "class_names.json")))
    idx, probs = [], []
    for r in a.runs:
        idx.append(np.load(os.path.join(r, "oof_val_idx.npy")))
        probs.append(np.load(os.path.join(r, "oof_val_probs.npy")))
    idx, p = np.concatenate(idx), np.concatenate(probs)
    assert len(set(idx.tolist())) == len(idx), "an inner-fold record appears in two runs"
    y = y_all[idx].astype(int)
    groups = np.array([meta[i]["group"] for i in idx])
    train_support = np.nansum(y_all[[i for i in range(len(meta)) if int(meta[i]["inner_fold"]) >= 0]], 0)
    thr = em.select_thresholds(y, p)
    rep = em.report(y, p, thr, names, train_support=train_support)
    pt, lo, hi, _ = em.cluster_bootstrap(y, p, groups, em.macro_auroc, n_boot=a.n_boot)
    rep.update(macro_auroc_cluster_ci=[pt, lo, hi], n_records=int(len(idx)), n_groups=int(len(set(groups))), thresholds=thr.tolist(), runs=a.runs,
               note="pooled out-of-fold inner-validation predictions; thresholds fitted on the same OOF predictions (optimistic F1; use only for relative comparison); no exposed partition used")
    with open(a.out, "x") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("macro_auroc", "macro_auroc_cluster_ci", "macro_auprc", "macro_f1", "ece_macro", "brier_macro", "rare_class_sensitivity")}, indent=1))


if __name__ == "__main__":
    main()
