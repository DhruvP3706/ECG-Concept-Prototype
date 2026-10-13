"""Step 3.5: paired record-level bootstrap of differences between models on the test fold (same resamples,
B=1000, same seed as step3_5_evaluate.py). Metrics: macro-AUROC (defined classes), macro-F1 (valid classes) and
micro-F1 at each model's PRIMARY thresholds. Uses only saved predictions; no tuning."""
import json, os, sys, itertools
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import metrics as M
OUT = os.path.join(ROOT, "results", "step3_5")
z = np.load(os.path.join(OUT, "evaluation_predictions.npz"))
Y = z["targets"].astype(bool)
prim = {"lightv2_ft_fold0": "original_file", "lightv2_p1_fold0": "original_file", "lightv2_p1_fold1": "regenerated_val",
        "lightv2_p1_fold2": "regenerated_val", "ensemble3": "ensemble_cv_regenerated"}
idxs = M.bootstrap_indices(4447, 1000, 20261013)
arr = {}
for m, t in prim.items():
    S, thr = z[f"probs__{m}__clean"], z[f"thr__{m}__{t}"]
    a, f, mf = np.zeros(1000), np.zeros(1000), np.zeros(1000)
    for b, i in enumerate(idxs):
        a[b] = np.nanmean(M.auroc_columns(Y[i], S[i]))
        tm = M.threshold_metrics(Y[i], S[i], thr)
        f[b], mf[b] = tm["macro_f1_valid"], tm["micro_f1"]
    arr[m] = {"macro_auroc": a, "macro_f1_valid": f, "micro_f1": mf}
out = {}
for m1, m2 in itertools.combinations(prim, 2):
    for k in ("macro_auroc", "macro_f1_valid", "micro_f1"):
        d = arr[m1][k] - arr[m2][k]
        lo, hi, _ = M.percentile_ci(d)
        out[f"{m1}_minus_{m2}|{k}"] = {"mean_diff": float(d.mean()), "ci95": [lo, hi], "frac_positive": float((d > 0).mean())}
json.dump(out, open(os.path.join(OUT, "paired_bootstrap.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k.replace('lightv2_',''):70s} {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] P(>0)={v['frac_positive']:.2f}")
