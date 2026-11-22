"""Input-level attribution for LightECGNetV2 (explains the model's own pre-sigmoid logit; nothing here trains or modifies a model).

* `integrated_gradients`  : midpoint-rule Integrated Gradients along the straight path baseline -> x for ONE class logit. Input shape is unchanged ((12, T) in, (12, T) out).
                            Reports completeness: sum(attr) vs f(x) - f(baseline). The model is used in eval mode with parameter gradients disabled and is restored afterwards.
* `region_ids`            : map every (lead, sample) to a region id for temporal windows, lead x window cells, or whole leads.
* `region_scores`         : aggregate an attribution map to region scores (signed sum or sum of magnitudes).
* `orderings`             : region orders used in the perturbation tests (descending signed, descending magnitude, ascending magnitude, ascending signed, random).
* `perturbed_logits`      : logits of inputs in which a chosen set of regions is replaced by a baseline (deletion) or restored from a baseline (insertion).
"""
import numpy as np
import torch


def _logit(model, x, class_idx):
    """class_idx: int -> that class's pre-sigmoid logit; (c, k) tuple -> the pairwise logit MARGIN f_c - f_k."""
    out = model(x)
    if isinstance(class_idx, (tuple, list)):
        return out[:, class_idx[0]] - out[:, class_idx[1]]
    return out[:, class_idx]


def integrated_gradients(model, x, class_idx, baseline=None, steps=64, chunk=64):
    """x: (12, T) array/tensor in NORMALISED signal space. baseline: same shape (default zeros). Returns dict with attr (12, T) float64 numpy,
    f_x, f_baseline (logits), completeness_abs_err, completeness_rel_err. Midpoint rule alphas (i + 0.5) / steps."""
    dev = next(model.parameters()).device
    was_training = model.training
    flags = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
        base = torch.zeros_like(x) if baseline is None else torch.as_tensor(np.asarray(baseline), dtype=torch.float32, device=dev)
        if x.ndim != 2 or base.shape != x.shape:
            raise ValueError(f"x and baseline must both be (12, T); got {tuple(x.shape)} and {tuple(base.shape)}")
        diff = x - base
        alphas = (torch.arange(steps, dtype=torch.float32, device=dev) + 0.5) / steps
        grad_sum = torch.zeros_like(x, dtype=torch.float64)
        for i in range(0, steps, chunk):
            a = alphas[i:i + chunk].view(-1, 1, 1)
            pts = (base[None] + a * diff[None]).requires_grad_(True)
            out = _logit(model, pts, class_idx).sum()
            g, = torch.autograd.grad(out, pts)
            grad_sum += g.double().sum(0)
        attr = (grad_sum / steps) * diff.double()
        with torch.no_grad():
            fx = float(_logit(model, x[None], class_idx))
            fb = float(_logit(model, base[None], class_idx))
        tot = float(attr.sum())
        err = abs(tot - (fx - fb))
        return {"attr": attr.cpu().numpy(), "f_x": fx, "f_baseline": fb, "attr_sum": tot, "completeness_abs_err": err,
                "completeness_rel_err": err / max(abs(fx - fb), 1e-6)}
    finally:
        for p, f in zip(model.parameters(), flags):
            p.requires_grad_(f)
        model.train(was_training)


def region_ids(kind, n_leads=12, T=5000, win=125):
    """(n_leads, T) int array of region ids. kind: 'time' (one region per temporal window, shared by all leads), 'cell' (lead x window), 'lead'."""
    t = np.arange(T) // win
    nw = int(t.max()) + 1
    if kind == "time":
        return np.broadcast_to(t[None, :], (n_leads, T)).copy(), nw
    if kind == "cell":
        return (np.arange(n_leads)[:, None] * nw + t[None, :]).astype(np.int64), n_leads * nw
    if kind == "lead":
        return np.broadcast_to(np.arange(n_leads)[:, None], (n_leads, T)).copy(), n_leads
    raise ValueError(kind)


def region_scores(attr, rid, n_regions, mode="signed"):
    a = np.abs(attr) if mode == "abs" else attr
    return np.bincount(rid.reshape(-1), weights=np.asarray(a, np.float64).reshape(-1), minlength=n_regions)


def orderings(scores_signed, scores_abs, rng, n_random=20):
    """Region orders (arrays of region indices, first = removed/restored first). random: (n_random, R)."""
    R = len(scores_signed)
    return {"pos": np.argsort(-scores_signed, kind="stable"), "abs": np.argsort(-scores_abs, kind="stable"),
            "low_abs": np.argsort(scores_abs, kind="stable"), "neg": np.argsort(scores_signed, kind="stable"),
            "random": np.stack([rng.permutation(R) for _ in range(n_random)])}


def prefix_masks(order, rid, ks):
    """bool (len(ks), n_leads, T): the first k regions of `order` selected, for each k in ks."""
    rank = np.empty(len(order), np.int64)
    rank[np.asarray(order)] = np.arange(len(order))
    er = rank[rid]
    return er[None] < np.asarray(ks)[:, None, None]


@torch.no_grad()
def perturbed_logits(model, x, base, class_idx, masks, mode, bs=96):
    """x, base: (12, T). masks: bool (M, 12, T) selected regions. mode 'delete': selected regions replaced by base; 'insert': start from base,
    selected regions restored from x. Returns (M,) logits (numpy)."""
    dev = next(model.parameters()).device
    model.eval()
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
    base = torch.as_tensor(np.asarray(base), dtype=torch.float32, device=dev)
    out = []
    for i in range(0, len(masks), bs):
        m = torch.as_tensor(masks[i:i + bs], device=dev)
        xin = torch.where(m, base[None], x[None]) if mode == "delete" else torch.where(m, x[None], base[None])
        out.append(_logit(model, xin, class_idx).float().cpu())
    return torch.cat(out).numpy()


def curve_auc(curve, fractions):
    """Mean of the curve over the fraction axis (trapezoid, normalised to [0, 1])."""
    f = np.asarray(fractions, float)
    return float(np.trapezoid(np.asarray(curve, float), f) / (f[-1] - f[0]))


# ---------------------------------------------------------------- lead-attention probing and an inference-only intervention
import contextlib  # noqa: E402
import types  # noqa: E402

import torch.nn.functional as _F  # noqa: E402


def lead_attention_weights(model, x):
    """x: (B, 12, T) normalised input. Returns (s, mu): s (B, 12) = the LeadAttention gates sigmoid(fc2(relu(fc1(mean_t x)))), mu (B, 12) = mean_t x.
    Computed with the model's own weights (no state change)."""
    la = model.lead_attn
    with torch.no_grad():
        mu = x.mean(-1)
        return torch.sigmoid(la.fc2(_F.relu(la.fc1(mu)))), mu


@contextlib.contextmanager
def lead_attention_intervention(model, mode):
    """Temporarily change ONLY the lead-attention gating in memory (the checkpoint and parameters are never touched; the original forward is restored on exit).
    mode 'original'     : the original formula, recomputed (must be identical to the unmodified model)
         'uniform_mean' : every lead gets the SAME gate = the mean over leads of the record's own original gates (detached constant) - overall gain preserved
         'ones'         : gate 1 for every lead (no gating)"""
    if mode not in ("original", "uniform_mean", "ones"):
        raise ValueError(mode)
    la = model.lead_attn

    def fwd(self, x):
        s = torch.sigmoid(self.fc2(_F.relu(self.fc1(x.mean(-1)))))
        if mode == "uniform_mean":
            s = s.detach().mean(1, keepdim=True).expand_as(s)
        elif mode == "ones":
            s = torch.ones_like(s)
        return x * s.unsqueeze(-1)

    la.forward = types.MethodType(fwd, la)
    try:
        yield model
    finally:
        del la.forward
