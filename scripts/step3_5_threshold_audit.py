"""Step 3.5 stage 2: threshold / split provenance audit. VALIDATION FOLDS ONLY (0,1,2,5); fold 9 is never read.

Writes results/step3_5/threshold_audit.json and regenerated_thresholds.npz (regenerated artifacts, kept
separate from models/pretrained). Reproduces, from cached fp32 outputs of the current checkpoints:
  * each checkpoint's notebook-recorded validation macro-F1 (cells 19/20/21 outputs)
  * the original threshold files vs thresholds re-fitted with the notebook's search on the checkpoint's own val fold
  * the ensemble CV thresholds printed by notebook cell 22 (fit on combined val folds 0,1,2)
  * a split-membership sanity check (held-out vs trained-on folds)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept.metrics import (aggregate_ranking, auroc_columns, find_optimal_thresholds,  # noqa: E402
                                 notebook_phase1_thresholds, threshold_metrics)

OUT = os.path.join(ROOT, "results", "step3_5")
MP = os.path.join(ROOT, "models", "pretrained")
meta = json.load(open(os.path.join(OUT, "cache", "meta.json")))
CLASSES = meta["classes"]
CK = ["lightv2_ft_fold0", "lightv2_p1_fold0", "lightv2_p1_fold1", "lightv2_p1_fold2"]
VAL_FOLD = {"lightv2_ft_fold0": 0, "lightv2_p1_fold0": 0, "lightv2_p1_fold1": 1, "lightv2_p1_fold2": 2}
TRAIN_FOLDS = {c: [f for f in range(9) if f != VAL_FOLD[c]] for c in CK}     # per notebook cells 19-21
# validation macro-F1 recorded by the notebook at the saved epoch (cell 19: 0.4541; cell 20: 0.4211; cell 21: 0.4321, 0.4432)
NB_VAL_F1 = {"lightv2_p1_fold0": 0.4541, "lightv2_ft_fold0": 0.4211, "lightv2_p1_fold1": 0.4321, "lightv2_p1_fold2": 0.4432}
# notebook cell 22 output: ensemble CV thresholds printed for classes with |t-0.5| > 0.1
NB_ENS_PRINTED = {
    "1AVB": .85, "2AVB": .80, "3AVB": .70, "AF": .60, "AFIB": .70, "ALS": .85, "APB": .65, "AQW": .85, "ARS": .80, "AT": .75,
    "AVB": .85, "AVRT": .30, "CCR": .85, "CR": .85, "ERV": .75, "IVB": .85, "JEB": .75, "LFBBB": .85, "LVH": .80, "LVQRSAL": .75,
    "MISW": .85, "PRIE": .10, "PWC": .70, "QTIE": .75, "RBBB": .85, "SB": .70, "SR": .75, "ST": .60, "STDD": .85, "STE": .85,
    "STTC": .75, "SVT": .85, "TWC": .65, "TWO": .70, "UW": .65, "VEB": .75, "VFW": .85, "VPB": .85, "VPE": .80, "WPW": .85,
}

fold = {k: np.load(os.path.join(OUT, "cache", f"fold{k}.npz")) for k in (0, 1, 2, 5)}
P = {(c, k): fold[k][c + "|probs"] for c in CK for k in fold}
Y = {k: fold[k]["targets"].astype(bool) for k in fold}
for k in fold:
    assert list(fold[k]["ids"]) == meta["fold_ids"][str(k)] and Y[k].shape[1] == 45

audit = {"note": "validation folds 0,1,2,5 only; fold 9 not read", "per_checkpoint": {}}
regen = {}
for c in CK:
    v = VAL_FOLD[c]
    p, y = P[(c, v)], Y[v]
    thr_re, f1_re = notebook_phase1_thresholds(p, y)
    regen[c] = thr_re
    entry = {"val_fold": v, "train_folds": TRAIN_FOLDS[c], "notebook_recorded_val_macro_f1": NB_VAL_F1[c],
             "regenerated_val_macro_f1": float(f1_re), "regenerated_equals_all_0.5": bool(np.all(thr_re == 0.5)),
             "val_macro_f1_at_0.5": threshold_metrics(y, p, 0.5)["macro_f1_nb"]}
    entry["regenerated_vs_notebook_f1_abs_diff"] = abs(f1_re - NB_VAL_F1[c])
    tpath = os.path.join(MP, c + "_thresh.npy")
    if os.path.exists(tpath):
        tf = np.load(tpath, allow_pickle=False)
        entry["file_exists"] = True
        entry["file_shape"] = list(tf.shape)
        entry["file_val_macro_f1"] = threshold_metrics(y, p, tf)["macro_f1_nb"]
        entry["file_vs_regenerated_equal_classes"] = int(np.isclose(tf, thr_re, atol=1e-9).sum())
        entry["file_vs_regenerated_max_abs_diff"] = float(np.abs(tf - thr_re).max())
        entry["file_vs_regenerated_diff_classes"] = {CLASSES[i]: [float(tf[i]), float(thr_re[i])]
                                                       for i in np.where(~np.isclose(tf, thr_re, atol=1e-9))[0]}
        d = abs(entry["file_val_macro_f1"] - NB_VAL_F1[c])
        entry["file_f1_abs_diff_from_notebook"] = d
        entry["status"] = "VERIFIED" if d <= 0.003 else "UNVERIFIED"
    else:
        entry["file_exists"] = False
        entry["status"] = "NO_ORIGINAL_FILE"
    audit["per_checkpoint"][c] = entry

# Ensemble CV thresholds (cell 22): each model on ITS OWN val fold, clean pass, combined 0,1,2; plain find_optimal_thresholds
ens = ["lightv2_ft_fold0", "lightv2_p1_fold1", "lightv2_p1_fold2"]
cp = np.vstack([P[(c, VAL_FOLD[c])] for c in ens])
cy = np.vstack([Y[VAL_FOLD[c]] for c in ens])
ens_thr = find_optimal_thresholds(cp, cy)
regen["ensemble_cv"] = ens_thr
printed_ok, printed_bad = 0, {}
for i, cl in enumerate(CLASSES):
    if cl in NB_ENS_PRINTED:
        if abs(ens_thr[i] - NB_ENS_PRINTED[cl]) < 1e-9:
            printed_ok += 1
        else:
            printed_bad[cl] = [float(NB_ENS_PRINTED[cl]), float(ens_thr[i])]
unprinted_ok = sum(1 for i, cl in enumerate(CLASSES) if cl not in NB_ENS_PRINTED and abs(ens_thr[i] - 0.5) <= 0.1 + 1e-9)
audit["ensemble_cv_thresholds"] = {
    "n_combined_val_records": int(cp.shape[0]), "notebook_reported_n": 13344,
    "printed_classes": len(NB_ENS_PRINTED), "printed_exact_match": printed_ok, "printed_mismatch": printed_bad,
    "unprinted_classes": 45 - len(NB_ENS_PRINTED), "unprinted_within_0.1_of_0.5": unprinted_ok,
    "values": dict(zip(CLASSES, map(float, ens_thr)))}

# Split-membership sanity: macro-AUROC of each checkpoint on each fold (val vs trained-on)
auc = {}
for c in CK:
    for k in fold:
        auc[(c, k)] = float(np.nanmean(auroc_columns(Y[k], P[(c, k)])))
audit["macro_auroc_by_fold"] = {c: {str(k): auc[(c, k)] for k in sorted(fold)} for c in CK}
audit["split_sanity"] = {}
for c in CK:
    v = VAL_FOLD[c]
    tr = [k for k in fold if k != v]            # folds 0,1,2,5 minus val: all were TRAINING folds for c
    audit["split_sanity"][c] = {"val_fold": v, "val_macro_auroc": auc[(c, v)],
                                "mean_macro_auroc_on_its_training_folds": float(np.mean([auc[(c, k)] for k in tr])),
                                "gap_train_minus_val": float(np.mean([auc[(c, k)] for k in tr]) - auc[(c, v)])}
# same-fold contrast: a fold is held-out for one model and training data for another
audit["same_fold_contrast"] = {}
for k, held, others in ((0, "lightv2_p1_fold0", ["lightv2_p1_fold1", "lightv2_p1_fold2"]),
                         (1, "lightv2_p1_fold1", ["lightv2_p1_fold0", "lightv2_p1_fold2"]),
                         (2, "lightv2_p1_fold2", ["lightv2_p1_fold0", "lightv2_p1_fold1"])):
    audit["same_fold_contrast"][f"fold{k}"] = {"held_out_model": held, "held_out_auroc": auc[(held, k)],
                                               "trained_on_models": {o: auc[(o, k)] for o in others}}

json.dump(audit, open(os.path.join(OUT, "threshold_audit.json"), "w"), indent=2)
np.savez_compressed(os.path.join(OUT, "regenerated_thresholds.npz"), classes=np.array(CLASSES),
                    **{k.replace("lightv2_", "") if k != "ensemble_cv" else k: v for k, v in regen.items()})
print(json.dumps({k: v for k, v in audit.items() if k != "per_checkpoint"}, indent=1)[:6000])
for c, e in audit["per_checkpoint"].items():
    print(c, json.dumps({k: v for k, v in e.items() if k != "file_vs_regenerated_diff_classes"}))
    if e.get("file_vs_regenerated_diff_classes"):
        print("   differing classes (file, regenerated):", e["file_vs_regenerated_diff_classes"])
