"""Atrial-content substitution / re-arrangement on raw 12-lead records (counterfactual sensitivity tests, torch-free).

All operations act on the DEVIATION of an atrial region from its straight baseline (dev = x - L, L = line through the first and last region sample), exactly as in Experiment 8 (atrial_ops.py):
  y = L_recipient + new_dev   -> the region ends are unchanged (no step), QRS windows and every non-target sample are untouched.
  flat = new_dev 0 | rev = time-reversed dev | shuf = block-shuffled dev | donor = dev of a donor region (same lead, same length), optionally per-lead amplitude-scaled.
  raw donor substitution (`raw_substitute`) copies the donor's raw samples (keeps the donor baseline) and therefore creates boundary steps (kept as a documented variant).
Matched sham: surrogate of the REAL modification m = new_dev - old_dev with the same per-lead per-region amplitude spectrum and RMS (random common phase across leads, linear ramp removed so ends are zero),
  i.e. same location, energy, boundary behaviour and spectral disturbance, but no donor waveform / no temporal structure.
"""
import json
import os

import numpy as np

from . import atrial_ops as ao

BANDS = ((0.0, 4.0), (4.0, 9.0), (9.0, 20.0), (20.0, 250.0))


class Bank:
    """memory-mapped donor deviation bank: dev_<a1>.npy (n_regions, 12, a1-50), raw_160.npy; index.json {a1: {rid: [start, stop]}, raw: {...}}."""

    def __init__(self, d):
        self.idx = json.load(open(os.path.join(d, "index.json")))
        self.dev = {a: np.load(os.path.join(d, f"dev_{a}.npy"), mmap_mode="r") for a in (120, 160, 200)}
        self.raw = np.load(os.path.join(d, "raw_160.npy"), mmap_mode="r")

    def devs(self, rid, a1):
        e = self.idx[str(a1)].get(rid)
        return None if e is None or e[1] <= e[0] else np.asarray(self.dev[a1][e[0]:e[1]], float)

    def raw160(self, rid):
        e = self.idx["raw"].get(rid)
        return None if e is None or e[1] <= e[0] else np.asarray(self.raw[e[0]:e[1]])


def region_devs(x, regs):
    out = []
    for a, e in regs:
        seg = x[:, a:e].astype(float)
        out.append(seg - ao._line(seg))
    return out


def rms_lead(devs):
    """per-lead median over regions of the region RMS."""
    return np.median(np.stack([np.sqrt(np.mean(d ** 2, axis=1)) for d in devs]), axis=0)


def with_devs(x, regs, devs, mask=None):
    y = np.array(x, float, copy=True)
    for (a, e), d in zip(regs, devs):
        seg = x[:, a:e].astype(float)
        new = ao._line(seg) + d
        if mask is None:
            y[:, a:e] = new
        else:
            y[mask, a:e] = new[mask]
    return y


def raw_substitute(x, regs, segs, mask=None):
    y = np.array(x, float, copy=True)
    for (a, e), s in zip(regs, segs):
        if mask is None:
            y[:, a:e] = s
        else:
            y[mask, a:e] = s[mask]
    return y


def reverse_devs(devs):
    return [d[:, ::-1].copy() for d in devs]


def shuffle_devs(devs, rng, k=5):
    """cut each region deviation into k equal blocks and re-order them with one (non-identity, non-reversing) permutation shared by all leads; the end-to-end ramp is removed afterwards so region ends stay unchanged."""
    out = []
    for d in devs:
        n = d.shape[1]
        edges = np.linspace(0, n, k + 1).astype(int)
        blocks = [d[:, edges[i]:edges[i + 1]] for i in range(k)]
        while True:
            p = rng.permutation(k)
            if not np.array_equal(p, np.arange(k)) and not np.array_equal(p, np.arange(k)[::-1]):
                break
        z = np.concatenate([blocks[i] for i in p], axis=1)
        out.append(z - (z[:, :1] + (z[:, -1:] - z[:, :1]) * np.linspace(0, 1, n)[None, :]))   # re-impose zero ends (otherwise the moved blocks create a step at the region boundary)
    return out


def donor_devs(bank_devs, n_regions, offset, scale=None):
    """one donor region per recipient region (cycling through the donor's eligible regions starting at `offset`); optional per-lead amplitude scale."""
    nd = len(bank_devs)
    out = []
    for i in range(n_regions):
        d = np.array(bank_devs[(offset + i) % nd], float)
        if scale is not None:
            d = d * scale[:, None]
        out.append(d)
    return out


def phase_surrogate(m, rng):
    """same per-lead amplitude spectrum and RMS as m; random phase common to all leads; endpoints forced to zero by removing the end-to-end ramp."""
    n = m.shape[1]
    F = np.fft.rfft(m, axis=1)
    ph = rng.uniform(0, 2 * np.pi, (1, F.shape[1]))
    ph[:, 0] = 0.0
    if n % 2 == 0:
        ph[:, -1] = 0.0
    s = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=n, axis=1)
    s = s - (s[:, :1] + (s[:, -1:] - s[:, :1]) * np.linspace(0, 1, n)[None, :])
    r_t, r_s = np.sqrt(np.mean(m ** 2, axis=1)), np.sqrt(np.mean(s ** 2, axis=1)) + 1e-12
    return s * (r_t / r_s)[:, None]


def sham_devs(old, new, rng):
    return [o + phase_surrogate(nw - o, rng) for o, nw in zip(old, new)]


def band_fractions(mods):
    f = np.fft.rfftfreq(mods[0].shape[1], 1 / 500)
    e = np.zeros(len(BANDS))
    for m in mods:
        P = np.abs(np.fft.rfft(m, axis=1)) ** 2
        for i, (lo, hi) in enumerate(BANDS):
            e[i] += P[:, (f >= lo) & (f < hi)].sum()
    return e / max(e.sum(), 1e-12)


def band_l1(x0, y_a, y_b, regs, mask=None):
    """L1 distance between normalised band-energy profiles of two modifications (0 = identical spectral shape, 2 = disjoint)."""
    def mods(y):
        out = []
        for a, e in regs:
            m = y[:, a:e].astype(float) - x0[:, a:e].astype(float)
            out.append(m if mask is None else m[mask])
        return out
    return float(np.abs(band_fractions(mods(y_a)) - band_fractions(mods(y_b))).sum())


def masked_rms(x0, y, regs, mask=None):
    v = []
    for a, e in regs:
        d = y[:, a:e].astype(float) - x0[:, a:e].astype(float)
        v.append(np.sqrt(np.mean((d if mask is None else d[mask]) ** 2)))
    return float(np.mean(v)) if v else 0.0
