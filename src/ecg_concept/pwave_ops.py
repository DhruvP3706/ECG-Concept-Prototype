"""P-wave-like (phase-locked pre-QRS deflection) localisation in SR records and patch transplantation (counterfactual sensitivity tests, torch-free).

Localisation (template based, NOT expert validated): per SR record, the deviations of all eligible atrial regions (region r-160..r-51, deviation = x - straight line through the region ends, Experiments 8/9) are averaged
(per lead); the P-like peak lag is the maximum of the lead-pooled, 9-sample-smoothed template energy; the record is accepted if the lead-pooled template SNR >= 5 (template amplitude / standard error over beats); a beat is
accepted if its deviation correlates >= 0.6 (pooled over leads and a +-20 sample window) with the template. This identifies a phase-locked deflection, which is called "P-wave-like", not a delineated P wave.
Patch = donor deviation in [lag-pre, lag+post) tapered to zero at both ends (cosine edges); it is placed on the recipient's FLATTENED region (straight line between region ends) with its peak at lag+offset.
"""
import numpy as np

REG = 110                                                      # region length (r-160 .. r-51)
START0 = -160
EDGE = 6
WIDTHS = {40: (25, 15), 50: (30, 20), 30: (18, 12)}            # patch length -> (samples before lag, samples after lag)
CTL_START = 5                                                  # control window = region samples 5..5+w (offsets -155..-115): before every accepted P-like window
MIN_START, MAX_END = 2, 108                                    # a patch must stay inside the region with >=2 samples margin at both ends
SNR_MIN, CORR_MIN, MIN_BEATS = 5.0, 0.6, 4


def taper(n, edge=EDGE):
    w = np.ones(n)
    r = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    w[:edge], w[-edge:] = r, r[::-1]
    return w


def localize(D):
    """D: (n_beats, 12, 110) region deviations of one record. Returns lag (samples relative to R), snr, per-beat corr, template."""
    T = D.mean(0)
    se = D.std(0, ddof=1) / np.sqrt(len(D))
    E = np.convolve((T ** 2).sum(0), np.ones(9) / 9, "same")
    k = int(np.argmax(E))
    snr = float(np.sqrt((T ** 2).sum(0)[k]) / np.sqrt((se ** 2).sum(0)[k] + 1e-9))
    lo, hi = max(k - 20, 0), min(k + 20, REG)
    tv = T[:, lo:hi].ravel()
    corr = np.array([np.corrcoef(d[:, lo:hi].ravel(), tv)[0, 1] for d in D])
    return {"lag": START0 + k, "snr": snr, "corr": corr, "template": T}


def patch_from(dev, lag, width=40):
    pre, post = WIDTHS[width]
    s = lag - START0 - pre
    return dev[:, s:s + pre + post] * taper(pre + post)


def control_from(dev, width=40):
    return dev[:, CTL_START:CTL_START + width] * taper(width)


def placement(lag, offset, width=40):
    """start index inside the region, or None if the patch would leave the protected area."""
    pre, post = WIDTHS[width]
    s = lag + offset - START0 - pre
    return s if (s >= MIN_START and s + pre + post <= MAX_END) else None


def place(patch, start):
    d = np.zeros((patch.shape[0], REG))
    d[:, start:start + patch.shape[1]] = patch
    return d


def surrogate_patch(patch, rng):
    """same per-lead amplitude spectrum and RMS as the patch, random common phase, same taper and position: local energy without the P-like waveform."""
    from .atrial_content import phase_surrogate
    s = phase_surrogate(patch, rng) * taper(patch.shape[1])
    r_t, r_s = np.sqrt(np.mean(patch ** 2, axis=1)), np.sqrt(np.mean(s ** 2, axis=1)) + 1e-12
    return s * (r_t / r_s)[:, None]
