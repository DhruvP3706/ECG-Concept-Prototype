"""Minimal concept -> anchored-prototype -> diagnosis components for the first concept-prototype experiment (Step 6.1).

* `Standardizer`   : per-column scaler fitted on supplied (training) data only; NaN entries are ignored when fitting.
* `masked_mse`     : masked, per-concept-averaged regression loss; missing targets never contribute and never produce NaN.
* `ConceptMLP`     : 512 -> 128 -> 4 concept predictor (outputs standardized concepts).
* `kmeans_det`     : deterministic k-means (farthest-point initialisation, Lloyd iterations; no randomness).
* `anchor_records` : prototypes are ACTUAL training records nearest the cluster centres (anchored, not learned).
* `similarities`   : s_k = exp(-||z - p_k||^2 / tau); `tau_rule`: median training nearest-prototype squared distance, floored.
* `ProtoHead`      : linear K -> C head; logits == bias + sum_k w_ck s_k exactly (`contributions`).
* `fit_head`       : generic BCE/MSE trainer with declared optimiser and validation early stopping; uses no test data.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Standardizer:
    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean, np.float64), np.asarray(std, np.float64)

    @classmethod
    def fit(cls, X, floor=1e-6):
        X = np.asarray(X, np.float64)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        return cls(mean, np.maximum(std, floor))

    def transform(self, X):
        return (np.asarray(X, np.float64) - self.mean) / self.std

    def inverse(self, Z):
        return np.asarray(Z, np.float64) * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def masked_mse(pred, target, mask):
    """pred/target (N,C) tensors, mask (N,C) bool. Mean over valid entries per concept, then mean over concepts that have any.
    Missing targets are replaced inside the loss only by 0 AND multiplied by the mask, so NaN cannot propagate."""
    t = torch.where(mask, target, torch.zeros_like(target))
    m = mask.to(pred.dtype)
    per = (((pred - t) ** 2) * m).sum(0) / m.sum(0).clamp(min=1.0)
    has = (m.sum(0) > 0).to(pred.dtype)
    return (per * has).sum() / has.sum().clamp(min=1.0)


class ConceptMLP(nn.Module):
    def __init__(self, d_in=512, hidden=128, d_out=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def kmeans_det(X, k, max_iter=100):
    """Deterministic k-means. Init: the point nearest the mean, then farthest-point; no random numbers."""
    X = np.asarray(X, np.float64)
    c = [X[int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))]]
    d = ((X - c[0]) ** 2).sum(1)
    for _ in range(1, k):
        i = int(np.argmax(d))
        c.append(X[i])
        d = np.minimum(d, ((X - X[i]) ** 2).sum(1))
    C = np.array(c)
    a = None
    for _ in range(max_iter):
        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        na = D.argmin(1)
        if a is not None and np.array_equal(na, a):
            break
        a = na
        C = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
    return C, a


def anchor_records(X, centres):
    """Index (into X) of the distinct record nearest each centre (ties/collisions resolved by next nearest)."""
    used, out = set(), []
    for c in centres:
        order = np.argsort(((X - c) ** 2).sum(1), kind="stable")
        i = next(int(j) for j in order if int(j) not in used)
        used.add(i)
        out.append(i)
    return out


def sqdist(Z, P):
    return ((np.asarray(Z)[:, None, :] - np.asarray(P)[None, :, :]) ** 2).sum(2)


def tau_rule(train_Z, P, floor=1e-6):
    """tau = median over TRAINING records of the squared distance to the nearest prototype (floored for numerical safety)."""
    return float(max(np.median(sqdist(train_Z, P).min(1)), floor))


def similarities(Z, P, tau):
    return np.exp(-sqdist(Z, P) / tau)


class ProtoHead(nn.Module):
    def __init__(self, k, n_classes=45):
        super().__init__()
        self.fc = nn.Linear(k, n_classes)

    def forward(self, s):
        return self.fc(s)

    def contributions(self, s):
        """(N, C, K) signed contributions w_ck * s_k; logits = bias + contributions.sum(-1)."""
        return s[:, None, :] * self.fc.weight[None, :, :]


def fit_head(model, Xtr, Ytr, Xva, Yva, seed, lr, max_epochs=100, patience=10, batch=256, wd=1e-4, loss="bce", mask_tr=None, mask_va=None):
    """Train with AdamW; early stopping on the validation loss; restore the best state. Returns (model, history dict).
    NOTE: call torch.manual_seed(seed) BEFORE constructing `model` so the weight initialisation is reproducible."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xtr, Ytr, Xva, Yva = (torch.as_tensor(a, dtype=torch.float32) for a in (Xtr, Ytr, Xva, Yva))
    if loss == "mse":
        mask_tr, mask_va = torch.as_tensor(mask_tr), torch.as_tensor(mask_va)

    def lossf(out, y, m=None):
        return F.binary_cross_entropy_with_logits(out, y) if loss == "bce" else masked_mse(out, y, m)

    best, best_state, bad, hist = float("inf"), None, 0, {"train": [], "val": []}
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot, n = 0.0, 0
        for b0 in range(0, len(perm), batch):
            ix = perm[b0:b0 + batch]
            opt.zero_grad()
            l = lossf(model(Xtr[ix]), Ytr[ix], None if mask_tr is None else mask_tr[ix])
            l.backward()
            opt.step()
            tot += float(l) * len(ix)
            n += len(ix)
        model.eval()
        with torch.no_grad():
            v = float(lossf(model(Xva), Yva, mask_va))
        hist["train"].append(tot / n)
        hist["val"].append(v)
        assert np.isfinite(v) and np.isfinite(tot), "non-finite loss"
        if v < best - 1e-7:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    hist.update(best_val=best, best_epoch=int(np.argmin(hist["val"])) + 1, epochs_run=len(hist["val"]))
    return model, hist


@torch.no_grad()
def predict(model, X, batch=4096):
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    return torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)]).numpy()
