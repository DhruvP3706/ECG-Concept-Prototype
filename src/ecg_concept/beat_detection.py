"""R-peak (beat timing) detection for the Step 4.3 concept pilot. Timing only; amplitude is never used downstream.

Two detectors are implemented from scratch with SciPy (no ECG package is installed). They are INSPIRED by, and are
NOT verified reproductions of, the published algorithms they are named after:

* `detect_adaptive`  - "ADAPT": Pan-Tompkins-style energy detector (band-pass -> derivative -> square -> moving-window
  integration -> adaptive signal/noise thresholds with a simple search-back). The original paper was not readable
  here, so details (learning-phase initialisation, search-back rule) are our own choices.
* `detect_tma`       - "TMA": Elgendi-style two-moving-average detector (band-pass 8-20 Hz, squaring, event window
  97 ms vs beat window 611 ms, offset 8% of the mean squared signal). Parameters follow the public description only.

All windows are specified in seconds and converted with the sampling rate. Polarity is handled by using energy for
detection and |x| for localisation. Peak times are samples (ints) at the sampling rate of the input.
Detection preprocessing here is independent of the LightECGNetV2 backbone preprocessing.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, find_peaks, sosfiltfilt

# file lead order: I, II, III, aVR, aVL, aVF, V1..V6 ; II = I + III and aVR+aVL+aVF = 0 (dataset paper), so
# I, II, V1-V6 are the 8 non-redundant channels (they are not statistically independent votes).
NONREDUNDANT_LEADS = (0, 1, 6, 7, 8, 9, 10, 11)


def _bandpass(x, fs, lo, hi, order=2):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64))   # zero phase: no detection delay


def _refine(xl_abs, centers, half):
    """Move each candidate to the maximum of |localisation signal| within +-half samples."""
    out = []
    for p in centers:
        lo, hi = max(0, int(p) - half), min(len(xl_abs), int(p) + half + 1)
        out.append(lo + int(np.argmax(xl_abs[lo:hi])))
    return out


def _dedupe(peaks, strength, refractory):
    """Keep the stronger of any two peaks closer than the refractory period (deterministic, earlier wins ties)."""
    keep = []
    for p in sorted(set(peaks)):
        if keep and p - keep[-1] < refractory:
            if strength[p] > strength[keep[-1]]:
                keep[-1] = p
        else:
            keep.append(p)
    return keep


def detect_adaptive(x, fs, c):
    """Pan-Tompkins-style adaptive-threshold detector. Returns sorted sample indices."""
    a, refr = c["A"], max(1, int(round(c["refractory_s"] * fs)))
    xb = _bandpass(x, fs, *a["band"])
    mwi = uniform_filter1d(np.gradient(xb) ** 2, size=max(1, int(round(a["mwi_s"] * fs))), mode="nearest")
    cand, _ = find_peaks(mwi, distance=refr)
    if cand.size == 0:
        return []
    n0 = min(len(mwi), int(round(a["init_s"] * fs)))
    early = mwi[cand[cand < n0]]
    spk = float(np.median(early)) if early.size else float(mwi[:n0].max())
    npk = float(mwi[:n0].mean())
    thr = npk + a["thr_frac"] * (spk - npk)
    signal, noise = [], []
    for p in cand:
        v = float(mwi[p])
        if v > thr:
            signal.append(int(p))
            spk = 0.125 * v + 0.875 * spk
        else:
            noise.append((int(p), v, thr))
            npk = 0.125 * v + 0.875 * npk
        thr = npk + a["thr_frac"] * (spk - npk)
    # search-back: a gap longer than 1.66x the recent mean RR gets its strongest sub-threshold candidate (>0.5x threshold)
    final = list(signal)
    for i in range(1, len(signal)):
        recent = np.diff(signal[max(0, i - 9):i])
        if recent.size == 0:
            continue
        lo, hi = signal[i - 1], signal[i]
        if hi - lo > a["searchback_factor"] * recent.mean():
            inside = [(v, p) for p, v, th in noise if lo + refr < p < hi - refr and v > a["searchback_thr"] * th]
            if inside:
                final.append(max(inside)[1])
    xl_abs = np.abs(_bandpass(x, fs, *c["loc_band"]))
    half = max(1, int(round(c["loc_halfwin_s"] * fs)))
    return _dedupe(_refine(xl_abs, sorted(final), half), xl_abs, refr)


def detect_tma(x, fs, c):
    """Elgendi-style two-moving-average detector. Returns sorted sample indices."""
    b, refr = c["B"], max(1, int(round(c["refractory_s"] * fs)))
    y = _bandpass(x, fs, *b["band"]) ** 2
    w1, w2 = max(1, int(round(b["w1_s"] * fs))), max(1, int(round(b["w2_s"] * fs)))
    thr = uniform_filter1d(y, size=w2, mode="nearest") + b["beta"] * y.mean()
    above = uniform_filter1d(y, size=w1, mode="nearest") > thr
    edges = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    xl_abs = np.abs(_bandpass(x, fs, *c["loc_band"]))
    peaks = []
    for s, e in zip(starts, ends):
        if e - s >= w1:                                   # block of interest must be at least one event window wide
            peaks.append(s + int(np.argmax(xl_abs[s:e])))
    return _dedupe(peaks, xl_abs, refr)


DETECTORS = {"A": detect_adaptive, "B": detect_tma}


def fuse_leads(per_lead, tol, k_min, refractory):
    """Fuse per-lead beat times (dict lead -> sample indices) into consensus beats.

    Greedy time-ordered clustering where every member must lie within `tol` samples of the cluster's FIRST member
    (never of the previous member), so consecutive beats cannot be linked into a chain; at most one beat per lead per
    cluster. Clusters supported by < k_min leads are dropped; the beat time is the median of member times; beats
    closer than `refractory` keep the better-supported one (earlier wins ties). Returns (times, support) arrays.
    """
    items = sorted((int(t), lead) for lead, ts in per_lead.items() for t in ts)
    consumed = [False] * len(items)
    clusters = []
    for i, (t0, l0) in enumerate(items):
        if consumed[i]:
            continue
        members, used = [i], {l0}
        for j in range(i + 1, len(items)):
            if items[j][0] - t0 > tol:
                break
            if not consumed[j] and items[j][1] not in used:
                members.append(j)
                used.add(items[j][1])
        for j in members:
            consumed[j] = True
        clusters.append((float(np.median([items[j][0] for j in members])), len(members)))
    clusters = [cl for cl in clusters if cl[1] >= k_min]
    kept = []
    for t, s in sorted(clusters):
        if kept and t - kept[-1][0] < refractory:
            if s > kept[-1][1]:
                kept[-1] = (t, s)
        else:
            kept.append((t, s))
    return np.array([t for t, _ in kept], dtype=float), np.array([s for _, s in kept], dtype=int)


def match_one_to_one(a, b, tol):
    """Optimal one-to-one assignment of two sorted time arrays within `tol` (minimises total |dt|, deterministic).

    Returns (pairs [(ia, ib)], unmatched_a indices, unmatched_b indices). Each element is used at most once."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return [], list(range(a.size)), list(range(b.size))
    cost = np.abs(a[:, None] - b[None, :])
    big = 1e9
    r, c = linear_sum_assignment(np.where(cost <= tol, cost, big))
    pairs = [(int(i), int(j)) for i, j in zip(r, c) if cost[i, j] <= tol]
    ua = sorted(set(range(a.size)) - {i for i, _ in pairs})
    ub = sorted(set(range(b.size)) - {j for _, j in pairs})
    return pairs, ua, ub
