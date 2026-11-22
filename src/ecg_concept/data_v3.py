"""Data utilities for Phase 1: float16 signal caches, paper-described augmentation, mixup, rare-class sampling weights.

Cache convention (one directory per dataset/partition): x.npy (N,12,5000) float16 = `preprocessing.preprocess(raw)` (original V2 pipeline, int16-truncation quirk kept) stored as float16;
y.npy (N,C) float32 with NaN for unknown labels; meta.csv (id, group, inner_fold, ...). A training partition is a separate cache from fold 9 / fold 10, so a training process cannot touch them by accident.
Augmentation follows the paper text (p=0.5 each unless noted): Gaussian noise sigma 0.02, circular time shift up to +-250 samples, amplitude scale U(0.8,1.2), random lead dropout (p=0.3, zeroing leads), mixup alpha 0.2.
Details the paper does not give (how many leads are dropped, whether the shift is circular, noise relative to z-scored signal) are chosen here and recorded in configs: 1-3 leads, circular roll, noise on z-scored signal.
"""
import numpy as np


def to_cache(x32):
    x = np.asarray(x32, np.float32)
    return x.astype(np.float16)


def cache_error(x32):
    """max abs error of the float16 round trip (z-scored signals)."""
    x = np.asarray(x32, np.float32)
    return float(np.abs(x.astype(np.float16).astype(np.float32) - x).max())


def paper_augment(x, rng, p=0.5, noise_sigma=0.02, max_shift=250, scale=(0.8, 1.2), p_lead_drop=0.3, max_drop=3):
    """x: (B,12,T) float32 (modified copy returned). Per-sample independent decisions."""
    x = np.array(x, np.float32, copy=True)
    B, L, T = x.shape
    for i in range(B):
        if rng.rand() < p:
            x[i] += rng.randn(L, T).astype(np.float32) * noise_sigma
        if rng.rand() < p:
            x[i] = np.roll(x[i], int(rng.randint(-max_shift, max_shift + 1)), axis=1)
        if rng.rand() < p:
            x[i] *= rng.uniform(*scale)
        if rng.rand() < p_lead_drop:
            x[i, rng.choice(L, int(rng.randint(1, max_drop + 1)), replace=False)] = 0.0
    return x


def mixup(x, y, rng, alpha=0.2):
    """Returns mixed x, mixed y (NaN labels stay NaN where either source is NaN), lambda, permutation."""
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = rng.permutation(len(x))
    xm = lam * x + (1 - lam) * x[perm]
    ym = lam * y + (1 - lam) * y[perm]
    return xm.astype(np.float32), ym.astype(np.float32), lam, perm


def sampling_weights(y, power=0.5):
    """WeightedRandomSampler weights proportional to 1/freq**power of each record's rarest positive class (paper: 1/sqrt(freq))."""
    y = np.nan_to_num(np.asarray(y, float), nan=0.0)
    freq = np.maximum(y.sum(0), 1.0)
    w = np.ones(len(y))
    for i in range(len(y)):
        pos = np.where(y[i] > 0)[0]
        if len(pos):
            w[i] = (1.0 / freq[pos]).max() ** power
    return w / w.sum()
