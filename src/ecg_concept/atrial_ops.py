"""Atrial-region interventions on raw 12-lead records (counterfactual sensitivity tests; NOT physiological simulation; torch-free).

Atrial region of beat i (R peak r_i): samples [r_i - a1, r_i - 51] (inclusive; default a1 = 160, i.e. 320 ms ... 102 ms before the R peak). The QRS-centred window (r_i +- 40) and everything outside the
atrial regions is never touched. A beat is ELIGIBLE only if the preceding RR interval is long enough that the region starts >= 220 samples after the previous R peak (so that the previous T wave is not inside the region).
Operation  `gain_region`: y = L + s * (x - L) inside each region, where L is the straight line joining the first and the last sample of the region (so y == x at both region ends: no step is introduced).
  s = 0 flattens the region (removes the deviation from a straight baseline = P wave / fibrillatory fluctuation / noise), s in (0, 1) attenuates it, s > 1 amplifies it.
Sham `random_sham`: adds a smooth zero-edge random perturbation (low-order sine series) to each region and lead, scaled so that its RMS equals the RMS of the real intervention's modification of that region and lead:
  same location, same size, but not specifically a removal of the atrial waveform.
Sensitivity `synthetic_fwave`: flattened region plus band-limited (4-9 Hz) random activity with a prescribed per-lead RMS (independent across leads = unrealistic; documented limitation).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

HALF = 40
A_END = 51                                                                   # region ends 51 samples before the R peak (QRS window starts at r-40)
GAP_AFTER_PREV = 220                                                         # region must start >= 220 samples after the previous R peak


def regions(beats, a1=160, T=5000):
    """list of (start, stop) half-open slices of the atrial regions of ELIGIBLE beats, plus the indices of those beats."""
    b = np.asarray(beats, int)
    out, idx = [], []
    for i in range(1, len(b)):
        s, e = b[i] - a1, b[i] - A_END + 1
        if s >= 0 and e <= T and (b[i] - b[i - 1]) >= a1 + GAP_AFTER_PREV and s - b[i - 1] >= GAP_AFTER_PREV:
            out.append((int(s), int(e)))
            idx.append(i)
    return out, idx


def _line(seg):
    n = seg.shape[1]
    u = np.linspace(0.0, 1.0, n)[None, :]
    return seg[:, :1] + (seg[:, -1:] - seg[:, :1]) * u


def gain_region(x, regs, s):
    y = np.array(x, float, copy=True)
    for a, e in regs:
        seg = x[:, a:e].astype(float)
        L = _line(seg)
        y[:, a:e] = L + s * (seg - L)
    return y


def to_int16(y):
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def modification(x, regs, s):
    """the real intervention's modification d = y - x per region: list of (n_leads, len) arrays."""
    out = []
    for a, e in regs:
        seg = x[:, a:e].astype(float)
        out.append((s - 1.0) * (seg - _line(seg)))
    return out


def random_sham(x, regs, target_mods, rng, n_terms=4):
    y = np.array(x, float, copy=True)
    for (a, e), d_real in zip(regs, target_mods):
        n = e - a
        u = np.linspace(0.0, 1.0, n)
        for l in range(x.shape[0]):
            c = rng.randn(n_terms) / np.arange(1, n_terms + 1)
            d = sum(ck * np.sin((k + 1) * np.pi * u) for k, ck in enumerate(c))
            rms_t = np.sqrt(np.mean(d_real[l] ** 2))
            rms_d = np.sqrt(np.mean(d ** 2)) + 1e-12
            y[l, a:e] += d * rms_t / rms_d
    return y


def fwave_noise(n, target_rms, rng, band=(4.0, 9.0), fs=500, edge=10):
    """band-limited random activity of length n with RMS = target_rms and zero ends (Tukey-like edge taper)."""
    pad = 300
    z = rng.randn(n + 2 * pad)
    sos = butter(3, band, btype="band", fs=fs, output="sos")
    f = sosfiltfilt(sos, z)[pad:pad + n]
    w = np.ones(n)
    k = min(edge, n // 2)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
    w[:k], w[-k:] = ramp, ramp[::-1]
    f = f * w
    return f / (np.sqrt(np.mean(f ** 2)) + 1e-12) * target_rms


def synthetic_fwave(x, regs, target_rms_per_lead, rng):
    y = np.array(x, float, copy=True)
    for a, e in regs:
        seg = x[:, a:e].astype(float)
        L = _line(seg)
        for l in range(x.shape[0]):
            y[l, a:e] = L[l] + fwave_noise(e - a, target_rms_per_lead[l], rng)
    return y


def region_deviation_rms(x, regs):
    """per-lead RMS of (x - L) over all regions (used to calibrate the synthetic f-wave amplitude from real AFIB records)."""
    if not regs:
        return np.full(x.shape[0], np.nan)
    v = np.concatenate([(x[:, a:e].astype(float) - _line(x[:, a:e].astype(float))) for a, e in regs], axis=1)
    return np.sqrt(np.mean(v ** 2, axis=1))


def qc(x0, y0, beats, regs, half=HALF, strict_amp=True):
    """x0, y0: original and modified int16 (12, T). Returns checks (all must hold) and magnitudes."""
    x0, y0 = np.asarray(x0), np.asarray(y0)
    T = x0.shape[1]
    b = np.asarray(beats, int)
    qrs_ok = all(np.array_equal(x0[:, max(r - half, 0):r + half + 1], y0[:, max(r - half, 0):r + half + 1]) for r in b)
    mask = np.ones(T, bool)
    for a, e in regs:
        mask[a:e] = False
    outside_ok = bool(np.array_equal(x0[:, mask], y0[:, mask]))
    shape_ok = x0.shape == y0.shape and x0.dtype == y0.dtype
    d = (y0.astype(float) - x0.astype(float))
    rms_mod = float(np.mean([np.sqrt(np.mean(d[:, a:e] ** 2)) for a, e in regs])) if regs else 0.0
    xs, ys = x0.astype(float), y0.astype(float)
    p999 = np.percentile(np.abs(np.diff(xs, axis=1)), 99.9, axis=1) + 1e-9
    worst = 0.0
    for a, e in regs:
        lo, hi = max(a - 2, 0), min(e + 2, T)
        worst = max(worst, float((np.abs(np.diff(ys[:, lo:hi], axis=1)).max(axis=1) / p999).max()))
    amp_ok = True
    clip = int(((y0 == 32767) | (y0 == -32768)).sum() - ((x0 == 32767) | (x0 == -32768)).sum())
    if strict_amp:
        for a, e in regs:
            amp_ok &= bool((np.abs(ys[:, a:e]).max(axis=1) <= np.abs(xs).max(axis=1) + 1e-9).all())
    return {"qrs_identical": bool(qrs_ok), "outside_identical": outside_ok, "shape_dtype_ok": bool(shape_ok), "rms_modification": rms_mod, "max_region_diff_ratio": worst, "amplitude_ok": bool(amp_ok), "new_clipped_samples": clip}
