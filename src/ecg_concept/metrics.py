"""Multi-label evaluation metrics (NumPy/SciPy only; scikit-learn is not installed).

Conventions
-----------
y : (N, C) {0,1} targets; s : (N, C) continuous scores; thr : (C,) thresholds.
Binary prediction rule (as in the original notebook): s > thr (strict).
Undefined per-class values are returned as NaN (AUROC/AUPRC need >=1 positive and
>=1 negative; recall/F1 need >=1 positive; precision needs >=1 predicted positive).
`*_nb` variants follow the notebook / scikit-learn `zero_division=0` convention
(undefined -> 0, macro averaged over ALL classes) so numbers can be compared with it.
"""
import numpy as np
from scipy.stats import rankdata


# ---------------------------------------------------------------- ranking metrics
def auroc_columns(y, s):
    """Per-class AUROC via the rank-sum (Mann-Whitney) statistic, ties averaged."""
    y = np.asarray(y, bool)
    n_pos = y.sum(0).astype(float)
    n_neg = y.shape[0] - n_pos
    ranks = rankdata(s, axis=0)                      # average ranks for ties
    auc = (ranks * y).sum(0) - n_pos * (n_pos + 1) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = auc / (n_pos * n_neg)
    auc[(n_pos == 0) | (n_neg == 0)] = np.nan
    return auc


def ap_columns(y, s):
    """Per-class average precision, identical to scikit-learn's step-wise definition (ties grouped)."""
    y = np.asarray(y, bool)
    n, c = y.shape
    order = np.argsort(-s, axis=0, kind="stable")
    ss = np.take_along_axis(s, order, 0)
    ys = np.take_along_axis(y, order, 0)
    tps = np.cumsum(ys, 0).astype(float)
    k = np.arange(1, n + 1, dtype=float)[:, None]
    n_pos = tps[-1]
    distinct = np.vstack([ss[1:] != ss[:-1], np.ones((1, c), bool)])   # last item of each tie group
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = tps / k
        rec = tps / n_pos
    cummax = np.maximum.accumulate(np.where(distinct, rec, 0.0), axis=0)
    prev = np.vstack([np.zeros((1, c)), cummax[:-1]])
    ap = (distinct * (rec - prev) * prec).sum(0)
    ap[n_pos == 0] = np.nan
    return ap


def auroc_micro(y, s):
    return float(auroc_columns(np.asarray(y).reshape(-1, 1), np.asarray(s).reshape(-1, 1))[0])


def ap_micro(y, s):
    return float(ap_columns(np.asarray(y).reshape(-1, 1), np.asarray(s).reshape(-1, 1))[0])


# ----------------------------------------------------------- thresholded metrics
def confusion(y, pred):
    y = np.asarray(y, bool)
    pred = np.asarray(pred, bool)
    tp = (pred & y).sum(0).astype(float)
    fp = (pred & ~y).sum(0).astype(float)
    fn = (~pred & y).sum(0).astype(float)
    tn = (~pred & ~y).sum(0).astype(float)
    return tp, fp, fn, tn


def per_class_prf(y, pred):
    """precision, recall, f1 (NaN where undefined) and the confusion counts."""
    tp, fp, fn, tn = confusion(y, pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        rec = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(tp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)   # defined iff >=1 positive
    return prec, rec, f1, (tp, fp, fn, tn)


def _f1_zero(tp, fp, fn):
    d = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, 2 * tp / d, 0.0)


def threshold_metrics(y, s, thr):
    """Aggregate thresholded metrics at per-class thresholds `thr` (scalar or (C,))."""
    y = np.asarray(y, bool)
    pred = s > np.broadcast_to(np.asarray(thr, dtype=np.float64), (s.shape[1],))[None, :]
    prec, rec, f1, (tp, fp, fn, tn) = per_class_prf(y, pred)
    support = y.sum(0)
    valid = support > 0
    n, c = y.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        prec0 = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec0 = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1_0 = _f1_zero(tp, fp, fn)
    out = {
        "macro_f1_nb": float(f1_0.mean()),                       # all classes, undefined -> 0 (notebook definition)
        "macro_precision_nb": float(prec0.mean()),
        "macro_recall_nb": float(rec0.mean()),
        "macro_f1_valid": float(np.nanmean(f1[valid])),          # classes with >=1 positive only
        "macro_precision_valid": float(prec0[valid].mean()),      # no predictions -> 0 precision
        "macro_recall_valid": float(np.nanmean(rec[valid])),
        "micro_f1": float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())),
        "micro_precision": float(tp.sum() / max(tp.sum() + fp.sum(), 1)),
        "micro_recall": float(tp.sum() / max(tp.sum() + fn.sum(), 1)),
        "weighted_f1": float((f1_0 * support).sum() / support.sum()),
        "exact_match": float((pred == y).all(1).mean()),
        "hamming_loss": float((pred != y).mean()),
        "mean_class_accuracy_nb": float(((tp + tn) / n).mean()),  # notebook's "Accuracy (mean)"
    }
    return out


def aggregate_ranking(y, s):
    y = np.asarray(y, bool)
    auc = auroc_columns(y, s)
    ap = ap_columns(y, s)
    support = y.sum(0)
    w = np.where(np.isnan(auc), 0.0, support)
    return {
        "macro_auroc": float(np.nanmean(auc)),
        "weighted_auroc": float(np.nansum(np.nan_to_num(auc) * w) / w.sum()),
        "micro_auroc": auroc_micro(y, s),
        "macro_auprc": float(np.nanmean(ap)),
        "micro_auprc": ap_micro(y, s),
        "n_classes_auroc_defined": int((~np.isnan(auc)).sum()),
    }


# -------------------------------------------- original threshold search (cell 15)
GRID = np.arange(0.1, 0.9, 0.05)   # exactly the notebook's grid (float64 values)


def find_optimal_thresholds(probs, trues):
    """Replicates the notebook's find_optimal_thresholds: per class, strict-greater F1 maximum
    over GRID, default 0.5, classes with no positives keep 0.5, first best wins (`f1 > best`)."""
    probs = np.asarray(probs)
    trues = np.asarray(trues, bool)
    n_cls = probs.shape[1]
    th = np.full(n_cls, 0.5)
    for c in range(n_cls):
        if trues[:, c].sum() == 0:
            continue
        best_f1, best_t = 0.0, 0.5
        for t in GRID:
            p = probs[:, c] > t
            tp = float((p & trues[:, c]).sum())
            fp = float((p & ~trues[:, c]).sum())
            fn = float((~p & trues[:, c]).sum())
            d = 2 * tp + fp + fn
            f1 = 2 * tp / d if d > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        th[c] = best_t
    return th


def notebook_phase1_thresholds(probs, trues):
    """What cells 19/20 saved: optimal thresholds if their macro-F1 >= the 0.5-default macro-F1, else all 0.5."""
    opt = find_optimal_thresholds(probs, trues)
    f_opt = threshold_metrics(trues, probs, opt)["macro_f1_nb"]
    f_def = threshold_metrics(trues, probs, 0.5)["macro_f1_nb"]
    return (opt, f_opt) if f_opt >= f_def else (np.full(probs.shape[1], 0.5), f_def)


# ------------------------------------------------------------------- bootstrap
def bootstrap_indices(n, b, seed):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, n, n) for _ in range(b)]


def percentile_ci(values, alpha=0.05):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (np.nan, np.nan, 0)
    return (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))), int(v.size))
