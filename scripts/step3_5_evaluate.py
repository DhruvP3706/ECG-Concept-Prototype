"""Step 3.5 stage 3: full held-out (folds[9]) evaluation of the four recovered checkpoints.

Thresholds come ONLY from (a) the original files whose status the validation-only audit marked VERIFIED, or
(b) thresholds re-fitted on each checkpoint's own validation fold (regenerated, labelled). Nothing is tuned on
the test fold. Bootstrap: record-level resampling (multi-label records stay whole), B=1000, shared resamples
across models (paired), percentile 95% CI; undefined per-class values in a resample are skipped (counts reported).
"""
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import metrics as M  # noqa: E402

OUT = os.path.join(ROOT, "results", "step3_5")
MP = os.path.join(ROOT, "models", "pretrained")
B = 1000
SEED = 20261013
meta = json.load(open(os.path.join(OUT, "cache", "meta.json")))
audit = json.load(open(os.path.join(OUT, "threshold_audit.json")))
CLASSES = meta["classes"]
CK = ["lightv2_ft_fold0", "lightv2_p1_fold0", "lightv2_p1_fold1", "lightv2_p1_fold2"]
ENS = ["lightv2_ft_fold0", "lightv2_p1_fold1", "lightv2_p1_fold2"]       # members of the notebook's 3-fold ensemble
regen = np.load(os.path.join(OUT, "regenerated_thresholds.npz"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


f9 = np.load(os.path.join(OUT, "cache", "fold9.npz"))
ids = f9["ids"]
assert list(ids) == meta["fold_ids"]["9"] and len(set(ids)) == len(ids) == 4447
Y = f9["targets"].astype(bool)
assert Y.shape == (4447, 45) and Y.any(1).all(), "every test record must carry >=1 of the 45 labels"
support = Y.sum(0)

scores = {}                                                     # (model, score_config) -> (N,45)
for c in CK:
    scores[(c, "clean")] = f9[c + "|probs"]
    scores[(c, "tta5")] = f9[c + "|tta5"]
    scores[(c, "floatfilt")] = f9[c + "|floatfilt"]
for cfg in ("clean", "tta5"):
    scores[("ensemble3", cfg)] = np.mean([scores[(c, cfg)] for c in ENS], axis=0)
for k, v in scores.items():
    assert v.shape == (4447, 45) and np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1, k

# ---- threshold configurations (model -> list of (name, array, status, role))
thr_cfg = {}
for c in CK:
    a = audit["per_checkpoint"][c]
    short = c.replace("lightv2_", "")
    lst = []
    if a["file_exists"]:
        tf = np.load(os.path.join(MP, c + "_thresh.npy"), allow_pickle=False)
        if a["status"] == "VERIFIED":
            lst.append(("original_file", tf, "VERIFIED", "primary"))
            lst.append(("regenerated_val", regen[short], "REGENERATED (own val fold)", "secondary"))
        else:
            lst.append(("regenerated_val", regen[short], "REGENERATED (own val fold)", "primary"))
            lst.append(("original_file_UNVERIFIED", tf, "UNVERIFIED (not used for primary)", "diagnostic"))
    else:
        lst.append(("regenerated_val", regen[short], "REGENERATED (own val fold); no original file exists", "primary"))
    lst.append(("fixed_0.5", np.full(45, 0.5), "diagnostic only", "diagnostic"))
    thr_cfg[c] = lst
thr_cfg["ensemble3"] = [
    ("ensemble_cv_regenerated", regen["ensemble_cv"], "REGENERATED (combined val folds 0,1,2)", "primary"),
    ("fixed_0.5", np.full(45, 0.5), "diagnostic only", "diagnostic")]

# ---- which (model, score_config) get bootstrap CIs
BOOT = [(c, "clean") for c in CK] + [("ensemble3", "clean"), ("lightv2_ft_fold0", "tta5"), ("ensemble3", "tta5")]
rng_idx = M.bootstrap_indices(4447, B, SEED)
RANK_KEYS = ["macro_auroc", "weighted_auroc", "micro_auroc", "macro_auprc", "micro_auprc"]

boot_rank = {}      # key -> dict arrays
for key in BOOT:
    S = scores[key]
    agg = {k: np.full(B, np.nan) for k in RANK_KEYS}
    pc_auc = np.full((B, 45), np.nan)
    pc_ap = np.full((B, 45), np.nan)
    t0 = time.time()
    for b, idx in enumerate(rng_idx):
        yb, sb = Y[idx], S[idx]
        a = M.auroc_columns(yb, sb)
        p = M.ap_columns(yb, sb)
        pc_auc[b], pc_ap[b] = a, p
        sup = yb.sum(0)
        w = np.where(np.isnan(a), 0.0, sup)
        agg["macro_auroc"][b] = np.nanmean(a)
        agg["weighted_auroc"][b] = np.nansum(np.nan_to_num(a) * w) / w.sum()
        agg["macro_auprc"][b] = np.nanmean(p)
        agg["micro_auroc"][b] = M.auroc_micro(yb, sb)
        agg["micro_auprc"][b] = M.ap_micro(yb, sb)
    boot_rank[key] = (agg, pc_auc, pc_ap)
    log("bootstrap ranking", key, f"{time.time() - t0:.0f}s")


def boot_thr(S, thr):
    keys = None
    agg = {}
    pc = {"precision": np.full((B, 45), np.nan), "recall": np.full((B, 45), np.nan), "f1": np.full((B, 45), np.nan)}
    for b, idx in enumerate(rng_idx):
        yb, sb = Y[idx], S[idx]
        m = M.threshold_metrics(yb, sb, thr)
        if keys is None:
            keys = list(m)
            agg = {k: np.full(B, np.nan) for k in keys}
        for k in keys:
            agg[k][b] = m[k]
        pr, rc, f1, _ = M.per_class_prf(yb, sb > thr[None, :])
        pc["precision"][b], pc["recall"][b], pc["f1"][b] = pr, rc, f1
    return agg, pc


agg_rows, pc_rows = [], []
npz = {"record_ids": ids.astype("U12"), "class_names": np.array(CLASSES), "targets": Y.astype(np.uint8)}
comparison = {}
point = {}
for (model, scfg), S in scores.items():
    rk = M.aggregate_ranking(Y, S)
    a_pc = M.auroc_columns(Y, S)
    ap_pc = M.ap_columns(Y, S)
    boot = boot_rank.get((model, scfg))
    for tname, thr, status, role in thr_cfg[model]:
        tm = M.threshold_metrics(Y, S, thr)
        role_eff = role if scfg == "clean" else ("secondary" if (model, scfg) in BOOT else "diagnostic")
        if scfg == "floatfilt":
            role_eff = "diagnostic"
        if scfg == "tta5" and role == "diagnostic" and tname != "fixed_0.5":
            continue
        row = {"model": model, "score_config": scfg, "threshold_config": tname, "threshold_status": status,
               "role": role_eff, "n_records": 4447, "n_classes_with_positives": int((support > 0).sum())}
        ci_r = ci_t = {}
        pc_ci = {}
        if boot is not None:
            ci_r = {k: M.percentile_ci(boot[0][k]) for k in RANK_KEYS}
            agg_t, pc_t = boot_thr(S, thr)
            ci_t = {k: M.percentile_ci(v) for k, v in agg_t.items()}
            pc_ci = {k: [M.percentile_ci(v[:, j]) for j in range(45)] for k, v in pc_t.items()}
        for k, v in {**rk, **tm}.items():
            row[k] = v
            ci = ci_r.get(k) or ci_t.get(k)
            row[k + "_lo"], row[k + "_hi"] = (ci[0], ci[1]) if ci else ("", "")
        agg_rows.append(row)
        pr, rc, f1, (tp, fp, fn, tn) = M.per_class_prf(Y, S > thr[None, :])
        for j, cl in enumerate(CLASSES):
            r = {"model": model, "score_config": scfg, "threshold_config": tname, "threshold_status": status, "role": role_eff,
                 "class_index": j, "class": cl, "n_pos": int(support[j]), "n_neg": int(4447 - support[j]),
                 "prevalence": support[j] / 4447, "threshold": float(thr[j]),
                 "auroc": a_pc[j], "auprc": ap_pc[j], "precision": pr[j], "recall": rc[j], "f1": f1[j],
                 "tp": int(tp[j]), "fp": int(fp[j]), "fn": int(fn[j]), "tn": int(tn[j])}
            if boot is not None:
                lo, hi, nv = M.percentile_ci(boot[1][:, j]); r.update(auroc_lo=lo, auroc_hi=hi, auroc_boot_valid=nv)
                lo, hi, nv = M.percentile_ci(boot[2][:, j]); r.update(auprc_lo=lo, auprc_hi=hi)
                for kk in ("precision", "recall", "f1"):
                    lo, hi, nv = pc_ci[kk][j]; r.update({kk + "_lo": lo, kk + "_hi": hi})
                r["f1_boot_valid"] = pc_ci["f1"][j][2]
            pc_rows.append(r)
        point[(model, scfg, tname)] = (rk, tm)
    npz[f"probs__{model}__{scfg}"] = S.astype(np.float32)
    log("done", model, scfg)

for c in CK + ["ensemble3"]:
    for tname, thr, *_ in thr_cfg[c]:
        npz[f"thr__{c}__{tname}"] = np.asarray(thr, np.float64)
np.savez_compressed(os.path.join(OUT, "evaluation_predictions.npz"), **npz)


def write_csv(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.6f}" if isinstance(v, float) else v))
                        for k, v in r.items()})


write_csv(os.path.join(OUT, "aggregate_metrics.csv"), agg_rows)
write_csv(os.path.join(OUT, "per_class_metrics.csv"), pc_rows)

# ---- notebook comparison (cells 22 and 24)
def pick(model, scfg, tname):
    rk, tm = point[(model, scfg, tname)]
    return {"precision_macro": tm["macro_precision_nb"], "recall_macro": tm["macro_recall_nb"], "f1_macro": tm["macro_f1_nb"],
            "f1_micro": tm["micro_f1"], "auroc_macro": rk["macro_auroc"], "accuracy_mean": tm["mean_class_accuracy_nb"]}


comparison = {
    "ft_fold0_tta5_original_thr": {"reproduced": pick("lightv2_ft_fold0", "tta5", "original_file") if ("lightv2_ft_fold0", "tta5", "original_file") in point else None,
                                   "notebook_cell24": dict(precision_macro=.3095, recall_macro=.5726, f1_macro=.3725, f1_micro=.6763, auroc_macro=.9628, accuracy_mean=.9715)},
    "ft_fold0_tta5_fixed0.5": {"reproduced": pick("lightv2_ft_fold0", "tta5", "fixed_0.5"),
                               "notebook_cell24": dict(precision_macro=.2721, recall_macro=.6585, f1_macro=.3478, f1_micro=.6150, auroc_macro=.9628, accuracy_mean=.9598)},
    "ensemble3_tta5_cv_thr": {"reproduced": pick("ensemble3", "tta5", "ensemble_cv_regenerated"),
                              "notebook_cell22": dict(precision_macro=.3673, recall_macro=.4786, f1_macro=.3856, f1_micro=.7172, auroc_macro=.9626, accuracy_mean=.9771)},
    "ensemble3_tta5_fixed0.5": {"reproduced": pick("ensemble3", "tta5", "fixed_0.5"),
                                "notebook_cell22": dict(precision_macro=.2891, recall_macro=.6489, f1_macro=.3670, f1_micro=.6415, auroc_macro=.9626, accuracy_mean=.9641)},
}
json.dump(comparison, open(os.path.join(OUT, "notebook_comparison.json"), "w"), indent=2)
log("wrote csv/npz/json")
print(json.dumps(comparison, indent=1))
