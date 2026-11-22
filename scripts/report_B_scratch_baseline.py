r"""Combined report for the Experiment B scratch baseline (inner folds 0-3 of the PTB-XL patient-grouped CV). Uses only the training cache and the four runs' saved artefacts.
Reads no fold-9 / fold-10 waveform. Threshold-dependent numbers are reported three ways and labelled: fixed 0.5 (no selection), NESTED leave-one-fold-out thresholds (thresholds fitted on the other three folds' OOF
predictions and applied to the held-out fold: an honest estimate of the threshold-selection procedure) and SAME-SET thresholds (fitted on the pooled OOF predictions they are scored on: optimistic, not an estimate).
Usage: .venv\Scripts\python.exe -I scripts\report_B_scratch_baseline.py --runs RUN0 RUN1 RUN2 RUN3 --out results\phase1_runs\B_scratch_baseline_report
"""
import argparse
import csv
import json
import os
import sys

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


def f1_macro(y, p, thr):
    pr = p > np.asarray(thr)[None, :]
    tp, fp, fn = (pr & (y == 1)).sum(0), (pr & (y == 0)).sum(0), (~pr & (y == 1)).sum(0)
    return float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1)))


def f1_micro(y, p, thr):
    pr = p > np.asarray(thr)[None, :]
    tp, fp, fn = (pr & (y == 1)).sum(), (pr & (y == 0)).sum(), (~pr & (y == 1)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=4, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for ext in (".json", ".md"):
        assert not os.path.exists(a.out + ext), a.out + ext
    y_all = np.load(os.path.join(CACHE, "y.npy")).astype(int)
    meta = list(csv.DictReader(open(os.path.join(CACHE, "meta.csv"), newline="", encoding="utf-8")))
    fold = np.array([int(m["inner_fold"]) for m in meta])
    grp = np.array([m["group"] for m in meta])
    x = np.load(os.path.join(CACHE, "x.npy"), mmap_mode="r")
    cfg = json.load(open(os.path.join(ROOT, "experiments", "phase1", "B_ptbxl_scratch.json")))
    R = {"runs": a.runs, "folds": {}}
    P, I, F = [], [], []
    ref_before = pv.verify_registry()
    for k, rd in enumerate(a.runs):
        va = np.where(fold == k)[0]
        tr = np.where((fold >= 0) & (fold != k))[0]
        info = {"n_train": int(len(tr)), "n_val": int(len(va))}
        info["patients_train"], info["patients_val"] = int(len(set(grp[tr]))), int(len(set(grp[va])))
        info["patient_overlap"], info["record_overlap"] = int(len(set(grp[tr]) & set(grp[va]))), int(len(set(tr) & set(va)))
        oi, op = np.load(os.path.join(rd, "oof_val_idx.npy")), np.load(os.path.join(rd, "oof_val_probs.npy"))
        info["oof_idx_equals_fold_rows"] = bool(np.array_equal(np.sort(oi), va))
        summ = json.load(open(os.path.join(rd, "summary.json")))
        info["ckpt_sha256"] = pv.sha256_file(os.path.join(rd, "best.pt"))
        info["ckpt_hash_matches_summary"] = info["ckpt_sha256"] == summ["best_ckpt_sha256"]
        hist = json.load(open(os.path.join(rd, "history.json")))
        man = json.load(open(os.path.join(rd, "run_manifest.json")))
        info["seed"], info["config_sha256"] = man["seeds"][0], man["config_sha256"]
        ev = [h for h in hist if "val_macro_auroc" in h]
        best = max(ev, key=lambda h: h["val_macro_auroc"])
        info.update(epochs_run=len(hist), best_epoch=best["epoch"], best_val_macro_auroc=best["val_macro_auroc"], summary_best_epoch=summ["best_epoch"],
                    best_epoch_matches_max_val_auroc=best["epoch"] == summ["best_epoch"], total_train_seconds=float(sum(h["sec"] for h in hist)), mean_epoch_seconds=float(np.mean([h["sec"] for h in hist])), gpu_peak_allocated_gb=float(max(h["gpu_peak_gb"] for h in hist)),
                    all_losses_finite=bool(all(np.isfinite(h["train_loss"]) for h in hist)), early_stopped=len(hist) < cfg["train"]["epochs"])
        if "grad_norm_max" in hist[0]:
            info.update(grad_norm_mean_over_epochs=float(np.mean([h["grad_norm_mean"] for h in hist])), grad_norm_max=float(max(h["grad_norm_max"] for h in hist)), grad_clipped_fraction_mean=float(np.mean([h["grad_norm_frac_clipped"] for h in hist])),
                        grad_nonfinite_steps=int(sum(h["grad_nonfinite_steps"] for h in hist)))
        # exact reproduction after reload and post-hoc train/val diagnostics at the selected checkpoint (eval mode; same code path for all folds, fold 0 included)
        m = tv.build_model(cfg, 5)
        m.load_state_dict(torch.load(os.path.join(rd, "best.pt"), map_location="cpu", weights_only=True), strict=True)
        m.to("cuda")
        pv_, lg_va = tv.predict_both(m, x, va, "cuda")
        info["reload_max_abs_diff_vs_saved_oof"] = float(np.abs(pv_ - op).max()) if np.array_equal(oi, va) else None
        pw = torch.as_tensor(pos_weight_from_labels(y_all[tr].astype(np.float32), cap=cfg["loss"]["pos_weight_cap"]))
        p_tr, lg_tr = tv.predict_both(m, x, tr, "cuda")
        info["best_ckpt_train_macro_auroc_full_train_eval"] = em.macro_auroc(y_all[tr], p_tr)
        info["best_ckpt_val_macro_auroc"] = em.macro_auroc(y_all[va], pv_)
        info["best_ckpt_train_loss_weighted_eval"] = float(masked_bce_with_logits(lg_tr, torch.from_numpy(y_all[tr].astype(np.float32)), pos_weight=pw)[0])
        info["best_ckpt_val_loss_weighted"] = float(masked_bce_with_logits(lg_va, torch.from_numpy(y_all[va].astype(np.float32)), pos_weight=pw)[0])
        info["best_ckpt_val_loss_unweighted"] = float(masked_bce_with_logits(lg_va, torch.from_numpy(y_all[va].astype(np.float32)))[0])
        rep = em.report(y_all[va], op, [0.5] * 5, NAMES)
        info["val_at_0.5"] = {k2: rep[k2] for k2 in ("macro_auroc", "macro_auprc", "macro_f1", "micro_f1", "macro_sensitivity", "macro_specificity", "ece_macro", "brier_macro")}
        info["per_class_val_auroc"] = {c: rep["per_class"][c]["auroc"] for c in NAMES}
        info["per_class_val_support"] = {c: rep["per_class"][c]["support"] for c in NAMES}
        info["history_trajectory"] = [{k2: h.get(k2) for k2 in ("epoch", "train_loss", "train_macro_auroc_eval", "val_macro_auroc", "val_loss_weighted", "val_loss_unweighted", "grad_norm_mean", "grad_norm_max", "sec")} for h in hist]
        R["folds"][k] = info
        P.append(op)
        I.append(oi)
        F.append(np.full(len(oi), k))
        del m
        torch.cuda.empty_cache()
    idx, p, fo = np.concatenate(I), np.concatenate(P), np.concatenate(F)
    assert len(set(idx.tolist())) == len(idx) == len(y_all), "OOF predictions must cover every training record exactly once"
    order = np.argsort(idx)
    idx, p, fo = idx[order], p[order], fo[order]
    y, g = y_all[idx], grp[idx]
    R["pooled"] = {"n_records": int(len(idx)), "n_patients": int(len(set(g))), "covers_all_records_once": True}
    # fold-wise variability
    fm = {k2: [R["folds"][k]["val_at_0.5"][k2] for k in range(4)] for k2 in ("macro_auroc", "macro_auprc", "macro_f1", "micro_f1", "ece_macro", "brier_macro")}
    R["fold_wise_mean_sd"] = {k2: [float(np.mean(v)), float(np.std(v, ddof=1))] for k2, v in fm.items()}
    R["fold_wise_per_class_auroc"] = {c: [R["folds"][k]["per_class_val_auroc"][c] for k in range(4)] for c in NAMES}
    R["fold_wise_per_class_auroc_mean_sd"] = {c: [float(np.mean(v)), float(np.std(v, ddof=1))] for c, v in R["fold_wise_per_class_auroc"].items()}
    # pooled metrics with patient-clustered CIs
    macro_auc = lambda yy, pp: float(np.nanmean(auroc_columns(yy, pp)))
    macro_ap = lambda yy, pp: float(np.nanmean(ap_columns(yy, pp)))
    R["pooled"]["macro_auroc"] = em.cluster_bootstrap(y, p, g, macro_auc, n_boot=5000)[:3]
    R["pooled"]["macro_auprc"] = em.cluster_bootstrap(y, p, g, macro_ap, n_boot=5000)[:3]
    R["pooled"]["per_class"] = {}
    for c, nm in enumerate(NAMES):
        R["pooled"]["per_class"][nm] = {"support": int(y[:, c].sum()), "auroc": em.cluster_bootstrap(y, p, g, lambda yy, pp, c=c: float(auroc_columns(yy, pp)[c]), n_boot=2000)[:3],
                                        "auprc": em.cluster_bootstrap(y, p, g, lambda yy, pp, c=c: float(ap_columns(yy, pp)[c]), n_boot=2000)[:3]}
    thr05 = [0.5] * 5
    rep05 = em.report(y, p, thr05, NAMES)
    R["pooled"]["at_0.5_no_selection"] = {k2: rep05[k2] for k2 in ("macro_f1", "micro_f1", "macro_sensitivity", "macro_specificity", "ece_macro", "brier_macro")}
    R["pooled"]["at_0.5_no_selection"]["macro_f1_ci"] = em.cluster_bootstrap(y, p, g, lambda yy, pp: f1_macro(yy, pp, thr05), n_boot=2000)[:3]
    R["pooled"]["at_0.5_no_selection"]["micro_f1_ci"] = em.cluster_bootstrap(y, p, g, lambda yy, pp: f1_micro(yy, pp, thr05), n_boot=2000)[:3]
    R["pooled"]["at_0.5_no_selection"]["per_class"] = {c: {k2: rep05["per_class"][c][k2] for k2 in ("sensitivity", "specificity", "ece", "brier")} for c in NAMES}
    # threshold selection: nested (honest) vs same-set (optimistic)
    nested = np.zeros_like(p, dtype=bool)
    nested_thr = {}
    for k in range(4):
        tr_m, te_m = fo != k, fo == k
        thr = em.select_thresholds(y[tr_m], p[tr_m])
        nested_thr[k] = thr.tolist()
        nested[te_m] = p[te_m] > thr[None, :]
    tp, fp, fn = (nested & (y == 1)).sum(0), (nested & (y == 0)).sum(0), (~nested & (y == 1)).sum(0)
    R["pooled"]["nested_leave_one_fold_out_thresholds"] = {"thresholds_per_heldout_fold": nested_thr, "macro_f1": float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1))), "micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
                                                            "macro_sensitivity": float(np.mean(tp / np.maximum(tp + fn, 1))), "macro_specificity": float(np.mean(((~nested) & (y == 0)).sum(0) / np.maximum((y == 0).sum(0), 1))),
                                                            "note": "honest for the threshold-selection procedure (thresholds never see the scored fold); still conditional on epoch selection by inner-validation AUROC"}
    thr_same = em.select_thresholds(y, p)
    R["pooled"]["same_set_thresholds_OPTIMISTIC_not_an_estimate"] = {"thresholds": thr_same.tolist(), "macro_f1": f1_macro(y, p, thr_same), "micro_f1": f1_micro(y, p, thr_same)}
    # HYP focus
    h = NAMES.index("HYP")
    R["HYP"] = {"fold_auroc": R["fold_wise_per_class_auroc"]["HYP"], "fold_support": [R["folds"][k]["per_class_val_support"]["HYP"] for k in range(4)],
                "fold_auprc": [float(ap_columns(y[fo == k], p[fo == k])[h]) for k in range(4)],
                "fold_sensitivity_0.5": [float(((p[fo == k, h] > .5) & (y[fo == k, h] == 1)).sum() / max((y[fo == k, h] == 1).sum(), 1)) for k in range(4)],
                "fold_specificity_0.5": [float(((p[fo == k, h] <= .5) & (y[fo == k, h] == 0)).sum() / max((y[fo == k, h] == 0).sum(), 1)) for k in range(4)],
                "fold_ece": [float(em.ece_columns(y[fo == k], p[fo == k])[h]) for k in range(4)]}
    # reference unchanged
    R["reference_files_verified_before_and_after"] = ref_before == pv.verify_registry()
    with open(a.out + ".json", "x") as f:
        json.dump(R, f, indent=1)
    # epoch metrics csv
    with open(a.out + "_epoch_metrics.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "epoch", "train_loss", "train_macro_auroc_eval", "val_macro_auroc", "val_loss_weighted", "val_loss_unweighted", "grad_norm_mean", "grad_norm_max", "sec"])
        for k in range(4):
            for r in R["folds"][k]["history_trajectory"]:
                w.writerow([k] + [r[c] for c in ("epoch", "train_loss", "train_macro_auroc_eval", "val_macro_auroc", "val_loss_weighted", "val_loss_unweighted", "grad_norm_mean", "grad_norm_max", "sec")])
    print(json.dumps({k2: R[k2] for k2 in ("fold_wise_mean_sd", "HYP")}, indent=1))
    print(json.dumps({"pooled": {k2: R["pooled"][k2] for k2 in ("macro_auroc", "macro_auprc", "at_0.5_no_selection", "nested_leave_one_fold_out_thresholds", "same_set_thresholds_OPTIMISTIC_not_an_estimate")}}, indent=1))
    open(a.out + ".md", "x").write("see JSON; narrative report in the chat response\n")


if __name__ == "__main__":
    main()
