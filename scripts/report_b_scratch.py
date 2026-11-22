r"""Combined Experiment B scratch baseline report (inner folds 0-3 pooled out-of-fold). Reads run directories and the training cache only; no fold-9/10 data.
Threshold policy (explicit): (a) fixed 0.5 = unbiased with respect to thresholds; (b) 'same-set optimistic' = thresholds fitted on the pooled OOF predictions and evaluated on the same predictions (NOT an unbiased estimate, shown only to quantify the optimism);
(c) 'nested leave-one-fold-out' = thresholds for fold k fitted on the OOF predictions of the other three folds and applied to fold k (approximately unbiased threshold-selection estimate).
Usage: .venv\Scripts\python.exe -I scripts\report_b_scratch.py --runs <run_f0> <run_f1> <run_f2> <run_f3> --out results\phase1_runs\B_scratch_combined_report.json
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_v3 as tv  # noqa: E402
from ecg_concept import eval_metrics as em  # noqa: E402
from ecg_concept import provenance as pv  # noqa: E402
from ecg_concept.losses_v3 import masked_bce_with_logits, pos_weight_from_labels  # noqa: E402
from ecg_concept.metrics import ap_columns, auroc_columns  # noqa: E402

CACHE = os.path.join(ROOT, "data", "processed", "ptbxl_train_v1")
NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
CFG = os.path.join(ROOT, "experiments", "phase1", "B_ptbxl_scratch.json")


def f1s(y, p, thr):
    pr = p > np.asarray(thr)[None, :]
    tp, fp, fn = (pr & (y == 1)).sum(0).astype(float), (pr & (y == 0)).sum(0).astype(float), (~pr & (y == 1)).sum(0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(tp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return float(np.nanmean(f)), float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()))


def multi_stat(y, p):
    a, ap = auroc_columns(y, p), ap_columns(y, p)
    mf, uf = f1s(y, p, [0.5] * 5)
    return np.concatenate([[np.nanmean(a), np.nanmean(ap), mf, uf], a, ap])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=4, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    cfg = json.load(open(CFG))
    x, y_all, meta, _, _ = tv.load_cache(CACHE, 0)
    fold = np.array([int(m["inner_fold"]) for m in meta])
    group = np.array([m["group"] for m in meta])
    res = {"cache": "data/processed/ptbxl_train_v1", "config_sha256": pv.sha256_file(CFG), "folds": {}, "checks": {}}
    ypool, ppool, gpool, fpool = [], [], [], []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for r in a.runs:
        man = json.load(open(os.path.join(r, "run_manifest.json")))
        k = man["inner_fold"]
        va_exp = np.where(fold == k)[0]
        tr_exp = np.where((fold >= 0) & (fold != k))[0]
        oi, op = np.load(os.path.join(r, "oof_val_idx.npy")), np.load(os.path.join(r, "oof_val_probs.npy"))
        hist = json.load(open(os.path.join(r, "history.json")))
        summ = json.load(open(os.path.join(r, "summary.json")))
        chk = {"val_records_equal_expected": bool(np.array_equal(np.sort(oi), va_exp)), "record_overlap": int(len(set(tr_exp) & set(va_exp))), "patient_overlap": int(len(set(group[tr_exp]) & set(group[va_exp]))),
               "n_train": int(len(tr_exp)), "n_val": int(len(va_exp)), "n_train_patients": int(len(set(group[tr_exp]))), "n_val_patients": int(len(set(group[va_exp]))),
               "manifest_n_train": man["n_train"], "manifest_n_val": man["n_val"]}
        sel = max((h for h in hist if "val_macro_auroc" in h), key=lambda h: h["val_macro_auroc"])
        chk["best_epoch_equals_argmax_val_macro_auroc"] = bool(sel["epoch"] == summ["best_epoch"])
        chk["best_ckpt_sha256_matches_summary"] = pv.sha256_file(os.path.join(r, "best.pt")) == summ["best_ckpt_sha256"]
        # reload reproduction and training-subset metrics at the best checkpoint (eval mode, full training subset)
        m = tv.build_model(cfg, 5)
        m.load_state_dict(torch.load(os.path.join(r, "best.pt"), map_location="cpu", weights_only=True), strict=True)
        m.to(dev)
        p_val, lg_val = tv.predict_both(m, x, oi, dev)
        chk["reload_max_abs_diff_val_probs"] = float(np.abs(p_val - op).max())
        p_tr, lg_tr = tv.predict_both(m, x, tr_exp, dev)
        pw = torch.as_tensor(pos_weight_from_labels(y_all[tr_exp], cap=50.0))
        y_tr, y_va = y_all[tr_exp], y_all[oi]
        train_auc = em.macro_auroc(y_tr.astype(int), p_tr)
        wl_tr = float(masked_bce_with_logits(lg_tr, torch.from_numpy(y_tr), pos_weight=pw)[0])
        wl_va = float(masked_bce_with_logits(lg_val, torch.from_numpy(y_va), pos_weight=pw)[0])
        tsup = np.nansum(y_tr, 0)
        rep = em.report(y_va.astype(int), op, [0.5] * 5, NAMES, train_support=tsup)
        fin = [bool(np.isfinite(h["train_loss"])) for h in hist]
        rr = {"seed": man["seeds"][0], "epochs_run": len(hist), "best_epoch": summ["best_epoch"], "val_macro_auroc_best": summ["best_inner_val_macro_auroc"], "train_macro_auroc_eval_at_best_ckpt": train_auc,
              "gap_train_minus_val_auroc": train_auc - summ["best_inner_val_macro_auroc"], "weighted_bce_train_eval": wl_tr, "weighted_bce_val": wl_va, "ckpt_sha256": summ["best_ckpt_sha256"], "run_dir": os.path.relpath(r, ROOT),
              "total_train_seconds": float(sum(h["sec"] for h in hist)), "mean_epoch_seconds_train_only": float(np.mean([h["sec"] for h in hist])), "gpu_peak_allocated_gb": max(h["gpu_peak_gb"] for h in hist if h.get("gpu_peak_gb")),
              "all_train_losses_finite": all(fin), "has_per_epoch_extras": "val_loss_weighted" in hist[0],
              "metrics_at_0.5": {kk: rep[kk] for kk in ("macro_auroc", "macro_auprc", "macro_f1", "micro_f1", "macro_sensitivity", "macro_specificity", "ece_macro", "brier_macro")}, "per_class": rep["per_class"]}
        if "val_loss_weighted" in hist[0]:
            rr["grad_norm"] = {"mean_of_epoch_means": float(np.mean([h["grad_norm_mean"] for h in hist])), "max": float(max(h["grad_norm_max"] for h in hist)), "mean_frac_clipped": float(np.mean([h["grad_norm_frac_clipped"] for h in hist])), "nonfinite_steps": int(sum(h["grad_nonfinite_steps"] for h in hist))}
            rr["curves"] = [{k2: h[k2] for k2 in ("epoch", "train_loss", "train_loss_weighted_eval", "val_loss_weighted", "val_loss_unweighted", "train_macro_auroc_eval", "val_macro_auroc", "grad_norm_mean", "grad_norm_max")} for h in hist]
            rr["extra_sec_per_epoch_incl_eval_not_in_sec"] = None
        rr["checks"] = chk
        res["folds"][str(k)] = rr
        order = np.argsort(oi)
        ypool.append(y_va.astype(int))
        ppool.append(op)
        gpool.append(group[oi])
        fpool.append(np.full(len(oi), k))
        del m
        torch.cuda.empty_cache()
    y, p, g, f = np.concatenate(ypool), np.concatenate(ppool), np.concatenate(gpool), np.concatenate(fpool)
    res["checks"]["pooled_n"] = int(len(y))
    res["checks"]["pooled_equals_all_16289_records"] = bool(len(y) == 16289)
    res["checks"]["patients_in_more_than_one_validation_fold"] = int(sum(1 for pt in set(g) if len(set(f[g == pt])) > 1))
    # pooled metrics with patient-clustered CIs (one resampling, many statistics)
    t0 = time.time()
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.RandomState(20261120)
    point = multi_stat(y, p)
    boots = np.empty((a.n_boot, len(point)))
    for b in range(a.n_boot):
        idx = np.concatenate([members[i] for i in rng.randint(0, len(uniq), len(uniq))])
        boots[b] = multi_stat(y[idx], p[idx])
    lo, hi = np.nanpercentile(boots, 2.5, 0), np.nanpercentile(boots, 97.5, 0)
    lab = ["macro_auroc", "macro_auprc", "macro_f1_at_0.5", "micro_f1_at_0.5"] + [f"auroc_{c}" for c in NAMES] + [f"auprc_{c}" for c in NAMES]
    res["pooled_oof"] = {"n_records": int(len(y)), "n_patients": int(len(uniq)), "bootstrap": f"patient-clustered, B={a.n_boot}, seed 20261120 ({time.time() - t0:.0f}s)", "stats_with_95ci": {l: [float(point[i]), float(lo[i]), float(hi[i])] for i, l in enumerate(lab)}}
    full = em.report(y, p, [0.5] * 5, NAMES, train_support=np.nansum(y_all[fold >= 0], 0))
    res["pooled_oof"]["at_threshold_0.5"] = {kk: full[kk] for kk in ("macro_sensitivity", "macro_specificity", "ece_macro", "brier_macro", "macro_f1", "micro_f1")}
    res["pooled_oof"]["per_class_at_0.5"] = full["per_class"]
    # thresholds: three clearly separated estimates
    thr_same = em.select_thresholds(y, p)
    res["thresholds"] = {"a_fixed_0.5": {"macro_f1": f1s(y, p, [0.5] * 5)[0], "micro_f1": f1s(y, p, [0.5] * 5)[1], "status": "unbiased with respect to thresholds"},
                         "b_same_set_optimistic": {"thresholds": thr_same.tolist(), "macro_f1": f1s(y, p, thr_same)[0], "micro_f1": f1s(y, p, thr_same)[1], "status": "OPTIMISTIC: thresholds fitted on the same OOF predictions; not an estimate of generalisation"}}
    pr = np.zeros_like(p, bool)
    nested_thr = {}
    for k in sorted(set(f.tolist())):
        thr_k = em.select_thresholds(y[f != k], p[f != k])
        nested_thr[str(k)] = thr_k.tolist()
        pr[f == k] = p[f == k] > thr_k[None, :]
    tp, fp, fn = (pr & (y == 1)).sum(0).astype(float), (pr & (y == 0)).sum(0).astype(float), (~pr & (y == 1)).sum(0).astype(float)
    res["thresholds"]["c_nested_leave_one_fold_out"] = {"thresholds_per_heldout_fold": nested_thr, "macro_f1": float(np.nanmean(2 * tp / (2 * tp + fp + fn))), "micro_f1": float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())),
                                                         "per_class_sensitivity": (tp / np.maximum(tp + fn, 1)).tolist(), "per_class_specificity": ((y == 0).sum(0) - fp).astype(float).__truediv__((y == 0).sum(0)).tolist(),
                                                         "status": "approximately unbiased threshold-selection estimate (thresholds never fitted on the fold they are applied to); models were still early-stopped on their own fold"}
    # fold-wise variability
    keys = ["val_macro_auroc_best", "train_macro_auroc_eval_at_best_ckpt", "gap_train_minus_val_auroc"]
    res["fold_variability"] = {kk: {"values": [res["folds"][s][kk] for s in sorted(res["folds"])], "mean": float(np.mean([res["folds"][s][kk] for s in res["folds"]])), "sd": float(np.std([res["folds"][s][kk] for s in res["folds"]], ddof=1))} for kk in keys}
    for m_ in ("macro_auprc", "macro_f1", "micro_f1", "ece_macro", "brier_macro"):
        v = [res["folds"][s]["metrics_at_0.5"][m_] for s in sorted(res["folds"])]
        res["fold_variability"][m_ + "_at_0.5"] = {"values": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
    for c in NAMES:
        for q in ("auroc", "auprc", "sensitivity", "specificity"):
            v = [res["folds"][s]["per_class"][c][q] for s in sorted(res["folds"])]
            res["fold_variability"][f"{c}_{q}"] = {"values": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
