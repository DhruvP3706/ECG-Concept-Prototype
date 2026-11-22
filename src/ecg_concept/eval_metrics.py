"""Evaluation report and patient/cluster-level bootstrap for Phase 1 (NumPy/SciPy only; scikit-learn is not installed).

Definitions fixed in docs/PROTOCOL_PHASE1.md:
* macro-AUROC / macro-AUPRC = mean over classes with >=1 positive and >=1 negative (undefined classes are NaN and listed, never set to 0).
* thresholded metrics use a strict '>' rule with externally supplied thresholds (never fitted inside this module).
* ECE: equal-width bins (default 15) of predicted probability, per class (binary), then mean over defined classes. Brier: mean squared error of the probability, per class then macro mean.
* rare-class sensitivity = mean sensitivity over classes whose TRAINING support is < `rare_support` (default 100) and that have >=1 positive in the evaluated set.
* cluster bootstrap: resample clusters (patients) with replacement, recompute the statistic on the pooled records of the sampled clusters.
"""
import numpy as np

from .metrics import ap_columns, auroc_columns, per_class_prf


def ece_columns(y, p, n_bins=15):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = np.zeros(y.shape[1])
    n = y.shape[0]
    for b in range(n_bins):
        m = idx == b
        cnt = m.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            conf = np.where(cnt > 0, (p * m).sum(0) / np.maximum(cnt, 1), 0.0)
            acc = np.where(cnt > 0, (y * m).sum(0) / np.maximum(cnt, 1), 0.0)
        out += cnt / n * np.abs(acc - conf)
    return out


def brier_columns(y, p):
    return ((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean(0)


def report(y, p, thresholds, class_names=None, train_support=None, rare_support=100, n_bins=15):
    """Full metric report. y (N,C) {0,1}; p (N,C) probabilities; thresholds (C,)."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    c = y.shape[1]
    class_names = class_names or [str(i) for i in range(c)]
    auc, ap = auroc_columns(y, p), ap_columns(y, p)
    pred = p > np.asarray(thresholds)[None, :]
    prec, rec, f1, (tp, fp, fn, tn) = per_class_prf(y, pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        spec = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
    ece, brier = ece_columns(y, p, n_bins), brier_columns(y, p)
    sup = y.sum(0)
    rare = np.zeros(c, bool) if train_support is None else (np.asarray(train_support) < rare_support) & (sup > 0)
    micro_f1 = float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-12))
    return {
        "macro_auroc": float(np.nanmean(auc)), "macro_auprc": float(np.nanmean(ap)), "macro_f1": float(np.nanmean(f1)), "micro_f1": micro_f1,
        "macro_sensitivity": float(np.nanmean(rec)), "macro_specificity": float(np.nanmean(spec)),
        "rare_class_sensitivity": float(np.nanmean(rec[rare])) if rare.any() else None, "n_rare_classes": int(rare.sum()),
        "ece_macro": float(np.nanmean(ece[sup > 0])) if (sup > 0).any() else None, "brier_macro": float(np.mean(brier)),
        "undefined_auroc_classes": [class_names[i] for i in np.where(np.isnan(auc))[0]],
        "per_class": {class_names[i]: {"support": int(sup[i]), "auroc": None if np.isnan(auc[i]) else float(auc[i]), "auprc": None if np.isnan(ap[i]) else float(ap[i]),
                                       "sensitivity": None if np.isnan(rec[i]) else float(rec[i]), "specificity": None if np.isnan(spec[i]) else float(spec[i]), "ece": float(ece[i]), "brier": float(brier[i])} for i in range(c)},
    }


def macro_auroc(y, p):
    return float(np.nanmean(auroc_columns(y, p)))


def cluster_bootstrap(y, p, groups, stat_fn, n_boot=5000, seed=20261120, alpha=0.05):
    """Percentile CI of stat_fn(y, p) with clusters resampled. Returns (point, lo, hi, boot_array)."""
    y, p = np.asarray(y), np.asarray(p)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        boots[b] = stat_fn(y[idx], p[idx])
    return float(stat_fn(y, p)), float(np.nanpercentile(boots, 100 * alpha / 2)), float(np.nanpercentile(boots, 100 * (1 - alpha / 2))), boots


def paired_cluster_bootstrap(y, p_a, p_b, groups, stat_fn, n_boot=5000, seed=20261120, alpha=0.05):
    """CI of stat(a) - stat(b) on the SAME resampled clusters."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.RandomState(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.randint(0, len(uniq), len(uniq))])
        d[b] = stat_fn(y[idx], p_a[idx]) - stat_fn(y[idx], p_b[idx])
    return float(stat_fn(y, p_a) - stat_fn(y, p_b)), float(np.nanpercentile(d, 100 * alpha / 2)), float(np.nanpercentile(d, 100 * (1 - alpha / 2))), d


def select_thresholds(y, p, grid=None, default=0.5):
    """Per-class F1-maximising thresholds on the given (out-of-fold) predictions; strict '>' rule; classes without positives keep `default`."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    grid = np.arange(0.05, 0.951, 0.05) if grid is None else np.asarray(grid)
    thr = np.full(y.shape[1], default, float)
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0:
            continue
        best = -1.0
        for g in grid:
            pr = p[:, c] > g
            tp, fp, fn = (pr & (y[:, c] == 1)).sum(), (pr & (y[:, c] == 0)).sum(), (~pr & (y[:, c] == 1)).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best + 1e-12:
                best, thr[c] = f1, g
    return thr
