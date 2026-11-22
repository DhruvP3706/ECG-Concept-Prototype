"""Signal regions around ALGORITHMICALLY detected R peaks (Step 4.3 extractor; automatic, not expert-validated) and attribution-mass bookkeeping.

All indices are samples (500 Hz) into the same (12, T) array the attribution was computed on (the preprocessing neither crops nor pads 5000-sample records and its filter is zero-phase,
so sample i of the raw record and of the preprocessed record are the same instant).

Regions (a partition of the record):
  qrs : union of windows [r - half, r + half] (inclusive, clipped to the record) around every detected R peak  -> "QRS-centred region" (NOT a verified QRS complex)
  rr  : samples between the first and the last detected R peak that are not in `qrs`  (the inter-beat segments)
  out : everything else (before the first / after the last detected beat, and the whole record if fewer than 2 beats)
"""
import numpy as np


def regions_from_beats(beats, T=5000, half=40):
    b = np.unique(np.asarray(beats, dtype=int))
    b = b[(b >= 0) & (b < T)]
    qrs = np.zeros(T, bool)
    for r in b:
        qrs[max(0, r - half):min(T, r + half + 1)] = True
    span = np.zeros(T, bool)
    if len(b) >= 2:
        span[b[0]:b[-1] + 1] = True
    rr = span & ~qrs
    out = ~qrs & ~rr
    return {"qrs": qrs, "rr": rr, "out": out, "beats": b}


def qrs_tiles(beats, T=5000, half=40):
    """one (start, stop) half-open window per beat, clipped; windows of neighbouring beats may overlap."""
    return [(max(0, int(r) - half), min(T, int(r) + half + 1)) for r in np.unique(np.asarray(beats, int)) if 0 <= r < T]


def rr_tiles(beats, T=5000, half=40):
    """non-overlapping tiles of width 2*half+1 inside each inter-beat segment (between neighbouring QRS windows); remainders shorter than a tile are dropped."""
    w = 2 * half + 1
    b = np.unique(np.asarray(beats, int))
    out = []
    for r0, r1 in zip(b[:-1], b[1:]):
        s, e = int(r0) + half + 1, int(r1) - half                     # half-open [s, e)
        while s + w <= e:
            out.append((s, s + w))
            s += w
    return out


def rr_segments(beats, T=5000, half=40):
    """one (start, stop) segment per RR interval (QRS windows removed); empty segments are omitted."""
    b = np.unique(np.asarray(beats, int))
    return [(int(r0) + half + 1, int(r1) - half) for r0, r1 in zip(b[:-1], b[1:]) if int(r1) - half > int(r0) + half + 1]


def time_mass(attr):
    """per-sample attribution mass summed over leads: (abs, positive, negative) with negative <= 0. attr: (12, T)."""
    a = np.asarray(attr, float)
    pos, neg = np.maximum(a, 0).sum(0), np.minimum(a, 0).sum(0)
    return pos - neg, pos, neg


def shares(mass_t, regions):
    tot = float(np.sum(mass_t))
    return {k: (float(mass_t[regions[k]].sum()) / tot if tot != 0 else float("nan")) for k in ("qrs", "rr", "out")}


def duration_shares(regions):
    T = len(regions["qrs"])
    return {k: float(regions[k].sum()) / T for k in ("qrs", "rr", "out")}


def shifted(regions, shift):
    """circular shift of all masks by the same amount (keeps spacing and total duration)."""
    return {k: (np.roll(v, shift) if k != "beats" else v) for k, v in regions.items()}


def random_windows(n_windows, width, T, rng, tries=200):
    """n non-overlapping windows of `width` samples at random positions (falls back to overlapping if packing fails)."""
    for _ in range(tries):
        starts = np.sort(rng.randint(0, T - width + 1, n_windows))
        if n_windows == 0 or (np.diff(starts) >= width).all():
            m = np.zeros(T, bool)
            for s in starts:
                m[s:s + width] = True
            return m
    m = np.zeros(T, bool)
    for s in rng.randint(0, T - width + 1, n_windows):
        m[s:s + width] = True
    return m
