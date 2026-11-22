"""Masked multi-label losses for dataset-specific heads (Phase 1). No label space is shared between datasets.

Conventions: targets (N,C) in {0,1}, NaN = unknown/missing label; `mask` (N,C) bool optional extra mask.
The loss is the mean of the per-element BCE over VALID elements only (== torch BCEWithLogitsLoss(mean) when everything is valid).
`pos_weight` follows the paper/notebook: N_neg / N_pos per class from the TRAINING partition only, clipped to [1, cap] (cap 50); classes without positives get the cap.
"""
import numpy as np
import torch
import torch.nn.functional as F


def pos_weight_from_labels(y, cap=50.0, floor=1.0):
    y = np.asarray(y, float)
    valid = np.isfinite(y)
    pos = np.nansum(np.where(valid, y, 0.0), 0)
    neg = np.nansum(np.where(valid, 1.0 - y, 0.0), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(pos > 0, neg / np.maximum(pos, 1e-12), cap)
    return np.clip(w, floor, cap).astype(np.float32)


def masked_bce_with_logits(logits, targets, mask=None, pos_weight=None):
    """Returns (loss, n_valid). If no element is valid the loss is a zero that still carries the graph (zero gradients)."""
    valid = torch.isfinite(targets)
    if mask is not None:
        valid = valid & mask.bool()
    t = torch.where(valid, targets, torch.zeros_like(targets))
    per = F.binary_cross_entropy_with_logits(logits.float(), t.float(), pos_weight=pos_weight, reduction="none")
    n = valid.sum()
    if n == 0:
        return logits.sum() * 0.0, 0
    return (per * valid).sum() / n, int(n)


class MultiTaskLoss:
    """Sum of weighted per-head masked losses. A head with no samples in the batch contributes exactly zero (and zero gradient)."""

    def __init__(self, weights=None, pos_weights=None):
        self.weights = weights or {}
        self.pos_weights = pos_weights or {}

    def __call__(self, outputs, targets, masks=None):
        total, parts = None, {}
        for head, logits in outputs.items():
            if logits is None or logits.shape[0] == 0:
                continue
            pw = self.pos_weights.get(head)
            l, n = masked_bce_with_logits(logits, targets[head], (masks or {}).get(head), None if pw is None else torch.as_tensor(pw, device=logits.device, dtype=torch.float32))
            parts[head] = (float(l.detach()), n)
            term = self.weights.get(head, 1.0) * l
            total = term if total is None else total + term
        if total is None:
            raise ValueError("batch contains no samples for any head")
        return total, parts
