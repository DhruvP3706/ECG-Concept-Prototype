"""Step 3.7 stage B: leakage sensitivity analysis from SAVED Step 3.5 predictions (no inference, no fitting).

Cohorts and rules were declared in analysis_config.json (stage A) before any cohort metric was computed.
Reuses src/ecg_concept/metrics.py unchanged. Reproduces the Step 3.5 headline metrics and paired bootstrap on cohort A first.
"""
import csv
import hashlib
import itertools
import json
import os
import platform
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import metrics as M  # noqa: E402
from ecg_concept.labels import CLASS_NAMES  # noqa: E402

OUT, R5 = os.path.join(ROOT, "results", "step3_7"), os.path.join(ROOT, "results", "step3_5")
CFG = os.path.join(OUT, "analysis_config.json")
for f in ("sensitivity_metrics.csv", "sensitivity_per_class.csv", "paired_comparisons.csv"):
    assert not os.path.exists(os.path.join(OUT, f)), f"{f} exists; refusing to overwrite"
cfg = json.load(open(CFG))
decl = dict(cfg["declaration"])
claimed = decl.pop("declaration_sha256")
assert hashlib.sha256(json.dumps(decl, sort_keys=True).encode()).hexdigest() == claimed, "declaration was altered after being written"
decl["declaration_sha256"] = claimed
B, SEED = decl["paired_bootstrap"]["B"], decl["paired_bootstrap"]["seed"]

pred = np.load(os.path.join(R5, "evaluation_predictions.npz"))
ids = [str(x) for x in pred["record_ids"]]
assert list(pred["class_names"]) == CLASS_NAMES and len(set(ids)) == len(ids) == 4447
Y = pred["targets"].astype(bool)
MODELS = {"ft_fold0": ("lightv2_ft_fold0", "original_file"), "p1_fold0": ("lightv2_p1_fold0", "original_file"),
          "p1_fold1": ("lightv2_p1_fold1", "regenerated_val"), "p1_fold2": ("lightv2_p1_fold2", "regenerated_val"),
          "ensemble3": ("ensemble3", "ensemble_cv_regenerated")}
S = {m: pred[f"probs__{c}__clean"].astype(np.float64) for m, (c, _) in MODELS.items()}
T = {m: pred[f"thr__{c}__{t}"] for m, (c, t) in MODELS.items()}

# ---- provenance / definition checks (no guessing)
mem = np.mean([pred[f"probs__lightv2_{k}__clean"] for k in ("ft_fold0", "p1_fold1", "p1_fold2")], axis=0)
assert np.abs(mem - pred["probs__ensemble3__clean"]).max() < 1e-7, "ensemble is not mean(ft_fold0,p1_fold1,p1_fold2)"
regen = np.load(os.path.join(R5, "regenerated_thresholds.npz"))
MP = os.path.join(ROOT, "models", "pretrained")
for k in ("ft_fold0", "p1_fold0"):
    assert np.array_equal(T[k], np.load(os.path.join(MP, f"lightv2_{k}_thresh.npy"))), k          # original, VERIFIED files
for k, name in (("p1_fold1", "p1_fold1"), ("p1_fold2", "p1_fold2"), ("ensemble3", "ensemble_cv")):
    assert np.array_equal(T[k], regen[name]), k                                                    # regenerated on validation folds
audit = json.load(open(os.path.join(R5, "threshold_audit.json")))["per_checkpoint"]
assert audit["lightv2_ft_fold0"]["status"] == audit["lightv2_p1_fold0"]["status"] == "VERIFIED" and audit["lightv2_p1_fold1"]["status"] == "UNVERIFIED"
assert not np.array_equal(T["p1_fold1"], np.load(os.path.join(MP, "lightv2_p1_fold1_thresh.npy"))), "stale p1_fold1 file must not be used"

masks = {}
for name, c in decl["cohorts"].items():
    masks[name] = ~np.isin(ids, c["excluded_ids"])
    assert masks[name].sum() == c["n"], name
    assert set(c["excluded_ids"]) <= set(ids)


def point(mask):
    out = {}
    for m in MODELS:
        y, s = Y[mask], S[m][mask]
        rk, tm = M.aggregate_ranking(y, s), M.threshold_metrics(y, s, T[m])
        auc, ap = M.auroc_columns(y, s), M.ap_columns(y, s)
        pr, rc, f1, (tp, fp, fn, tn) = M.per_class_prf(y, s > T[m][None, :])
        out[m] = dict(rk=rk, tm=tm, auc=auc, ap=ap, pr=pr, rc=rc, f1=f1, cm=(tp, fp, fn, tn))
    return out


pts = {c: point(masks[c]) for c in masks}

# ---- reproduce Step 3.5 headline metrics (cohort A) within tolerance. CSV was printed with 6 decimals => tolerance 5.1e-7
rows5 = list(csv.DictReader(open(os.path.join(R5, "aggregate_metrics.csv"))))
TOL_CSV = 5.1e-7
maxdiff = 0.0
for m, (c, t) in MODELS.items():
    r = [x for x in rows5 if x["model"] == c and x["score_config"] == "clean" and x["threshold_config"] == t and x["role"] == "primary"]
    assert len(r) == 1, (m, len(r))
    for k, v in {**pts["A_all_test"][m]["rk"], **pts["A_all_test"][m]["tm"]}.items():
        if k in r[0] and r[0][k] != "":
            d = abs(float(r[0][k]) - v)
            maxdiff = max(maxdiff, d)
            assert d <= TOL_CSV, ("headline mismatch", m, k, float(r[0][k]), v)

# ---- evaluable classes
n_pos = {c: Y[masks[c]].sum(0) for c in masks}
n_neg = {c: masks[c].sum() - n_pos[c] for c in masks}
evalble = {c: (n_pos[c] > 0) & (n_neg[c] > 0) for c in masks}
common = np.logical_and.reduce(list(evalble.values()))
unevaluable = {c: [CLASS_NAMES[j] for j in np.where(~evalble[c])[0]] for c in masks}

KEYS = ["macro_auroc", "weighted_auroc", "micro_auroc", "macro_auprc", "micro_auprc", "macro_f1_nb", "macro_f1_valid", "micro_f1"]
met_rows, cls_rows = [], []
for c in masks:
    for m in MODELS:
        p = pts[c][m]
        v = {**p["rk"], **p["tm"]}
        v["macro_auroc_common"] = float(np.nanmean(p["auc"][common]))
        v["macro_auprc_common"] = float(np.nanmean(p["ap"][common]))
        v["macro_f1_common"] = float(np.nanmean(p["f1"][common]))
        row = {"cohort": c, "model": m, "thresholds": MODELS[m][1], "n": int(masks[c].sum()), "n_classes_evaluable": int(evalble[c].sum()),
               "n_classes_common_evaluable": int(common.sum())}
        for k in KEYS + ["macro_auroc_common", "macro_auprc_common", "macro_f1_common"]:
            row[k] = v[k]
            row["d_" + k + "_vs_A"] = v[k] - ({**pts["A_all_test"][m]["rk"], **pts["A_all_test"][m]["tm"], "macro_auroc_common": float(np.nanmean(pts["A_all_test"][m]["auc"][common])),
                                               "macro_auprc_common": float(np.nanmean(pts["A_all_test"][m]["ap"][common])),
                                               "macro_f1_common": float(np.nanmean(pts["A_all_test"][m]["f1"][common]))}[k])
        met_rows.append(row)
        for j, cl in enumerate(CLASS_NAMES):
            a = pts["A_all_test"][m]
            cls_rows.append({"cohort": c, "model": m, "class": cl, "n_pos": int(n_pos[c][j]), "n_neg": int(n_neg[c][j]), "evaluable": bool(evalble[c][j]),
                             "common_evaluable": bool(common[j]), "threshold": float(T[m][j]), "auroc": p["auc"][j], "auprc": p["ap"][j], "precision": p["pr"][j],
                             "recall": p["rc"][j], "f1": p["f1"][j], "tp": int(p["cm"][0][j]), "fp": int(p["cm"][1][j]), "fn": int(p["cm"][2][j]), "tn": int(p["cm"][3][j]),
                             "d_auroc_vs_A": p["auc"][j] - a["auc"][j], "d_auprc_vs_A": p["ap"][j] - a["ap"][j], "d_f1_vs_A": p["f1"][j] - a["f1"][j]})


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(x, float) and np.isnan(x) else (f"{x:.6f}" if isinstance(x, (float, np.floating)) else x)) for k, x in r.items()})


write(os.path.join(OUT, "sensitivity_metrics.csv"), met_rows)
write(os.path.join(OUT, "sensitivity_per_class.csv"), cls_rows)

# ---- paired bootstrap per cohort (shared resamples across models); reproduce Step 3.5 on cohort A
PK = decl["paired_bootstrap"]["metrics"]
names5 = {"ft_fold0": "lightv2_ft_fold0", "p1_fold0": "lightv2_p1_fold0", "p1_fold1": "lightv2_p1_fold1", "p1_fold2": "lightv2_p1_fold2", "ensemble3": "ensemble3"}
prow, repro = [], {}
for c in masks:
    y_c = Y[masks[c]]
    n = int(masks[c].sum())
    idxs = M.bootstrap_indices(n, B, SEED)
    arr, ncls = {}, np.zeros(B)
    for m in MODELS:
        s_c = S[m][masks[c]]
        a = {k: np.full(B, np.nan) for k in PK}
        for b, i in enumerate(idxs):
            col = M.auroc_columns(y_c[i], s_c[i])
            a["macro_auroc"][b] = np.nanmean(col)
            tm = M.threshold_metrics(y_c[i], s_c[i], T[m])
            a["macro_f1_valid"][b], a["micro_f1"][b] = tm["macro_f1_valid"], tm["micro_f1"]
            ncls[b] = np.sum(~np.isnan(col))
        arr[m] = a
    for m1, m2 in itertools.combinations(MODELS, 2):
        for k in PK:
            d = arr[m1][k] - arr[m2][k]
            lo, hi, nv = M.percentile_ci(d)
            pt = {**pts[c][m1]["rk"], **pts[c][m1]["tm"]}[k] - {**pts[c][m2]["rk"], **pts[c][m2]["tm"]}[k]
            prow.append({"cohort": c, "model_a": m1, "model_b": m2, "metric": k, "point_diff": pt, "boot_mean_diff": float(np.nanmean(d)), "ci95_lo": lo, "ci95_hi": hi,
                         "frac_boot_positive": float((d[~np.isnan(d)] > 0).mean()), "n_valid_replicates": nv, "B": B, "seed": SEED, "n_records": n,
                         "mean_evaluable_classes_per_replicate": float(ncls.mean())})
            repro[(c, m1, m2, k)] = (float(np.nanmean(d)), lo, hi, float((d[~np.isnan(d)] > 0).mean()))
write(os.path.join(OUT, "paired_comparisons.csv"), prow)

ref = json.load(open(os.path.join(R5, "paired_bootstrap.json")))
pdiff = 0.0
for (c, m1, m2, k), (mean, lo, hi, fp_) in repro.items():
    if c != "A_all_test":
        continue
    r = ref[f"{names5[m1]}_minus_{names5[m2]}|{k}"]
    pdiff = max(pdiff, abs(r["mean_diff"] - mean), abs(r["ci95"][0] - lo), abs(r["ci95"][1] - hi), abs(r["frac_positive"] - fp_))
assert pdiff <= 1e-8, ("paired bootstrap not reproduced", pdiff)

cfg["completed"] = {
    "reproduction": {"headline_vs_aggregate_metrics_csv": {"tolerance": TOL_CSV, "max_abs_diff": maxdiff,
                     "note": "declared tolerance was 1e-8 but aggregate_metrics.csv stores 6 decimals, so 5.1e-7 is the achievable check; paired bootstrap was checked at 1e-8"},
                     "paired_bootstrap_vs_step3_5": {"tolerance": 1e-8, "max_abs_diff": pdiff}},
    "checks": ["ensemble == mean(ft_fold0,p1_fold1,p1_fold2) (<1e-7)", "original thresholds == models/pretrained files for ft_fold0,p1_fold0",
               "regenerated thresholds == regenerated_thresholds.npz", "p1_fold1 stale file not used", "cohort sizes == declared", "rows matched by record ID"],
    "evaluable_classes": {c: int(evalble[c].sum()) for c in masks}, "unevaluable_classes": unevaluable,
    "common_evaluable_classes": int(common.sum()), "software": {"python": platform.python_version(), "numpy": np.__version__},
    "command": "python -I scripts/step3_7_manifests.py ; python -I scripts/step3_7_sensitivity.py",
}
json.dump(cfg, open(CFG, "w"), indent=1)
print(json.dumps(cfg["completed"], indent=1))
