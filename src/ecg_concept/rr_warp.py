"""Beat-anchored time warping for controlled RR-timing interventions (counterfactual sensitivity tests; not clinical simulation).

Method: every detected R peak r_i gets a new location r'_i. The QRS-centred window [r_i-half, r_i+half] is copied RIGIDLY (slope 1, the samples are identical), and only the inter-beat segment between
neighbouring windows is linearly stretched or compressed (cubic-spline interpolation of the original samples) so that consecutive windows land exactly at the new spacing. The same time map is used for all
leads. The first and the last beat (and everything outside them) are not moved by the interventions below, so the number of beats, their order and the total duration are preserved.
Limitation (documented, measured by `quality`): the P/T/ST segments between beats are stretched with the RR interval, so their DURATION changes with the new RR; the QRS windows are untouched.
"""
import numpy as np
from scipy.interpolate import CubicSpline

HALF = 40
MIN_RR = 100                                  # samples (200 ms = the extractor's refractory period)


def valid_beats(new, half=HALF, min_rr=MIN_RR, T=5000):
    new = np.asarray(new)
    return bool(len(new) >= 2 and (np.diff(new) >= max(min_rr, 2 * half + 2)).all() and new[0] - half >= -half and new[-1] <= T - 1)


def time_map(old, new, T=5000, half=HALF, rigid_post=0, rigid_pre=0):
    """old sample index for every NEW sample index t' (piecewise linear, slope 1 inside windows and outside the beat span).
    rigid_post / rigid_pre: this many samples after / before each QRS window are ALSO copied rigidly (ST-T / P-wave region); only the remaining middle part of the inter-beat segment is stretched.
    If the middle part would be shorter than 10 samples in the old or the new timing, that segment falls back to uniform stretching (see `uniform_segments`)."""
    old, new = np.asarray(old, float), np.asarray(new, float)
    if len(old) != len(new) or not (np.diff(new) > 2 * half).all():
        raise ValueError("beat windows must not overlap and the beat count must be preserved")
    L = 10000.0
    kn, ko = [new[0] - half - L], [old[0] - half - L]
    for i, (o, n_) in enumerate(zip(old, new)):
        kn += [n_ - half, n_ + half]
        ko += [o - half, o + half]
        if i + 1 < len(old) and (rigid_post or rigid_pre):
            Mo = (old[i + 1] - half - rigid_pre) - (o + half + rigid_post)
            Mn = (new[i + 1] - half - rigid_pre) - (n_ + half + rigid_post)
            if Mo >= 10 and Mn >= 10:
                kn += [n_ + half + rigid_post, new[i + 1] - half - rigid_pre]
                ko += [o + half + rigid_post, old[i + 1] - half - rigid_pre]
    kn.append(new[-1] + half + L)
    ko.append(old[-1] + half + L)
    return np.interp(np.arange(T, dtype=float), kn, ko)


def uniform_segments(old, new, half=HALF, rigid_post=0, rigid_pre=0):
    """indices of inter-beat segments that fall back to uniform stretching (the rigid parts do not fit)."""
    out = []
    for i in range(len(old) - 1):
        Mo = (old[i + 1] - half - rigid_pre) - (old[i] + half + rigid_post)
        Mn = (new[i + 1] - half - rigid_pre) - (new[i] + half + rigid_post)
        if (rigid_post or rigid_pre) and not (Mo >= 10 and Mn >= 10):
            out.append(i)
    return out


def warp(signal, old, new, half=HALF, rigid_post=0, rigid_pre=0):
    """signal (12, T) -> warped float array (12, T). Outside the original support the nearest sample is used (never reached when the end beats are fixed)."""
    s = np.asarray(signal, float)
    T = s.shape[1]
    tm = np.clip(time_map(old, new, T, half, rigid_post, rigid_pre), 0, T - 1)
    return CubicSpline(np.arange(T), s, axis=1)(tm)


def to_int16(x):
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


def _positions(first, rr):
    return np.concatenate([[first], first + np.cumsum(np.asarray(rr, float))]).round().astype(int)


def regularise(beats, lam):
    """RR'_i = mean + lam * (RR_i - mean); lam=0 -> equal intervals. First and last beat fixed (total duration and beat count preserved)."""
    b = np.asarray(beats, int)
    rr = np.diff(b).astype(float)
    new = b[0] + np.concatenate([[0], np.cumsum(rr.mean() + lam * (rr - rr.mean()))])
    new = np.rint(new).astype(int)
    new[-1] = b[-1]
    return new


def shuffle_rr(beats, rng, tries=50):
    """random permutation of the RR intervals (multiset and total preserved); a permutation that changes the order is preferred."""
    b = np.asarray(beats, int)
    rr = np.diff(b)
    for _ in range(tries):
        p = rng.permutation(len(rr))
        if len(rr) < 2 or not (rr[p] == rr).all():
            break
    return _positions(b[0], rr[p])


def sham_smooth(beats, target_rms_disp, rng, half=HALF, shrink=0.85, max_shrink=30):
    """smooth, low-frequency random displacement of the beats (zero at both ends) with RMS displacement = target; RR irregularity is not specifically changed."""
    b = np.asarray(beats, int)
    u = np.linspace(0, 1, len(b))
    g = sum(rng.randn() * np.sin(k * np.pi * u) for k in (1, 2, 3))
    d = g / (np.sqrt(np.mean(g ** 2)) + 1e-12) * target_rms_disp
    for _ in range(max_shrink):
        new = np.rint(b + d).astype(int)
        new[0], new[-1] = b[0], b[-1]
        if valid_beats(new, half):
            return new
        d = d * shrink
    return None


def sham_noise(beats, target_rel_rms, rng, half=HALF, shrink=0.85, max_shrink=30):
    """RR'_i = RR_i (1 + eps_i), eps ~ N(0, s) scaled so the RMS relative change of the intervals equals the target; total duration preserved (this IS a random RR perturbation, not a pure sham)."""
    b = np.asarray(beats, int)
    rr = np.diff(b).astype(float)
    z = rng.randn(len(rr))
    scale = target_rel_rms / (np.sqrt(np.mean(z ** 2)) + 1e-12)
    for _ in range(max_shrink):
        rr2 = rr * (1 + scale * z)
        if (rr2 > 0).all():
            rr2 = rr2 * rr.sum() / rr2.sum()
            new = np.rint(b[0] + np.concatenate([[0], np.cumsum(rr2)])).astype(int)
            new[-1] = b[-1]
            if valid_beats(new, half):
                return new
        scale *= shrink
    return None


def rr_stats(beats):
    rr = np.diff(np.asarray(beats, float))
    return {"mean": float(rr.mean()), "sd": float(rr.std(ddof=1)) if len(rr) > 1 else float("nan"), "cv": float(rr.std(ddof=1) / rr.mean()) if len(rr) > 1 else float("nan"),
            "rmssd": float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else float("nan")}


def displacement_rms(old, new):
    return float(np.sqrt(np.mean((np.asarray(new, float) - np.asarray(old, float)) ** 2)))


def rel_rr_change_rms(old, new):
    a, b = np.diff(np.asarray(old, float)), np.diff(np.asarray(new, float))
    return float(np.sqrt(np.mean(((b - a) / a) ** 2)))


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def quality(x, y, old, new, half=HALF, post=150, pre=100, rigid_post=0, rigid_pre=0):
    """x, y: original and transformed int16 (12, T) arrays. Returns morphology-preservation and discontinuity metrics."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    old, new = np.asarray(old, int), np.asarray(new, int)
    T = x.shape[1]
    exact = all(np.array_equal(x[:, o - half:o + half + 1], y[:, n_ - half:n_ + half + 1]) for o, n_ in zip(old, new) if o - half >= 0 and o + half < T and n_ - half >= 0 and n_ + half < T)
    postc, prec, factors = [], [], []
    for i in range(len(old) - 1):
        Lo, Ln = (old[i + 1] - half) - (old[i] + half + 1), (new[i + 1] - half) - (new[i] + half + 1)
        factors.append(Ln / Lo if Lo > 0 else np.nan)
        m = int(min(post, Lo, Ln))
        if m > 5:
            so, sn = x[:, old[i] + half + 1:old[i] + half + 1 + m], y[:, new[i] + half + 1:new[i] + half + 1 + m]
            c = [_corr(so[l], sn[l]) for l in range(12)]
            postc.append(np.nanmean(c) if np.isfinite(c).any() else np.nan)
        m2 = int(min(pre, Lo, Ln))
        if m2 > 5:
            so, sn = x[:, old[i + 1] - half - m2:old[i + 1] - half], y[:, new[i + 1] - half - m2:new[i + 1] - half]
            c = [_corr(so[l], sn[l]) for l in range(12)]
            prec.append(np.nanmean(c) if np.isfinite(c).any() else np.nan)
    d2x = np.abs(np.diff(x, 2, axis=1))
    d2y = np.abs(np.diff(y, 2, axis=1))
    p99 = np.percentile(d2x, 99, axis=1) + 1e-9
    tm = time_map(old, new, T, half, rigid_post, rigid_pre)
    slope = np.diff(tm)
    jidx = np.where(np.abs(np.diff(slope)) > 1e-9)[0] + 1                      # samples where the time map changes slope = where the warp can create a kink
    junc = []
    for j in jidx:
        jo = int(round(tm[j]))
        if 3 <= j < T - 3 and 3 <= jo < T - 3:
            base = np.maximum(p99, d2x[:, max(jo - 3, 0):jo + 2].max(axis=1))           # the original's own curvature at the corresponding place
            junc.append((d2y[:, j - 2:j + 1].max(axis=1) / base).max())
    sdx, sdy = x.std(1) + 1e-9, y.std(1)
    return {"qrs_windows_identical": bool(exact), "post_qrs_corr_median": float(np.nanmedian(postc)) if postc else float("nan"), "pre_qrs_corr_median": float(np.nanmedian(prec)) if prec else float("nan"),
            "junction_ratio_max": float(max(junc)) if junc else 0.0, "std_ratio_min": float((sdy / sdx).min()), "std_ratio_max": float((sdy / sdx).max()),
            "stretch_factor_min": float(np.nanmin(factors)), "stretch_factor_max": float(np.nanmax(factors)), "rms_log_stretch": float(np.sqrt(np.nanmean(np.log(np.array(factors)) ** 2))),
            "n_uniform_fallback_segments": int(len(uniform_segments(old, new, half, rigid_post, rigid_pre))), "n_segments": int(len(old) - 1), "p999_abs_diff_ratio": float(np.percentile(np.abs(np.diff(y, axis=1)), 99.9) / (np.percentile(np.abs(np.diff(x, axis=1)), 99.9) + 1e-9))}


# ---------------------------------------------------------------- graded RR variability (Experiment 7)
def rr_cv(beats):
    rr = np.diff(np.asarray(beats, float))
    return float(rr.std(ddof=1) / rr.mean()) if len(rr) > 1 else float("nan")


def _pattern(kind, n, rng, rr_orig=None):
    if kind == "iid":
        z = rng.randn(n)
    elif kind == "smooth":                                                       # AR(1), rho = 0.8: slow, respiration-like fluctuation
        e = rng.randn(n)
        z = np.zeros(n)
        z[0] = e[0]
        for i in range(1, n):
            z[i] = 0.8 * z[i - 1] + np.sqrt(1 - 0.64) * e[i]
    elif kind == "perm_orig":                                                    # the record's own RR deviations in random order, rescaled to the target CV
        z = rng.permutation(np.asarray(rr_orig, float) - np.mean(rr_orig))
    else:
        raise ValueError(kind)
    z = z - z.mean()
    sd = z.std(ddof=1) if n > 1 else 0.0
    return None if sd < 1e-9 else z / sd


def target_cv_beats(beats, target_cv, kind, rng, min_rr=MIN_RR, tol_abs=0.005, tol_rel=0.05, max_tries=200):
    """New beat positions with RR sample-CV = target (within max(tol_abs, tol_rel*target) after rounding), mean RR, beat count, order and total duration preserved
    (first and last beat fixed). Returns (positions or None, tries, reason). Every RR >= min_rr samples."""
    b = np.asarray(beats, int)
    rr = np.diff(b).astype(float)
    if len(rr) < 3:
        return None, 0, "too_few_intervals"
    m = rr.mean()
    reason = "min_rr_violation"
    for t in range(1, max_tries + 1):
        z = _pattern(kind, len(rr), rng, rr)
        if z is None:
            return None, t, "degenerate_pattern"
        rr2 = m * (1 + target_cv * z)
        if rr2.min() < min_rr:
            continue
        new = np.rint(b[0] + np.concatenate([[0], np.cumsum(rr2)])).astype(int)
        new[-1] = b[-1]
        if not valid_beats(new):
            continue
        if abs(rr_cv(new) - target_cv) > max(tol_abs, tol_rel * target_cv):
            reason = "cv_tolerance_after_rounding"
            continue
        return new, t, ""
    return None, max_tries, reason
