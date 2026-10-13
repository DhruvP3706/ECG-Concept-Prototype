"""Step 3.5 stage 1: run the four recovered checkpoints over original folds and cache sigmoid outputs.

Inference only (eval + inference_mode, fp32). No training, no threshold fitting, no waveform output.
Folds: 9 = original held-out test; 0,1,2 = validation folds of the checkpoints (used ONLY for
threshold/provenance audit); 5 = a CV fold every checkpoint trained on (train-reference for the
split sanity check). For the test fold two extra configurations are produced:
  * tta5      : notebook-style 5-pass TTA (clean + 4 noisy/scaled passes, seeded), probs averaged
  * floatfilt : same as primary but WITHOUT int16 truncation after filtering (diagnostic)
Writes results/step3_5/cache/fold{k}.npz and manifest/meta JSON.
"""
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ecg_concept.inference import load_model  # noqa: E402
from ecg_concept.labels import CLASS_NAMES, build_index, reproduce_folds  # noqa: E402
from ecg_concept.preprocessing import load_ecg  # noqa: E402

OUT = os.path.join(ROOT, "results", "step3_5")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
MP = os.path.join(ROOT, "models", "pretrained")
CKPTS = ["lightv2_ft_fold0", "lightv2_p1_fold0", "lightv2_p1_fold1", "lightv2_p1_fold2"]
FOLDS = [9, 0, 1, 2, 5]
BATCH = 64
DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tta_augment(sig, rng):
    """Notebook tta_augment: +N(0,0.005) noise, then x U(0.95,1.05) scale (per record)."""
    sig = sig + rng.normal(0, 0.005, sig.shape).astype(np.float32)
    return sig * np.float32(rng.uniform(0.95, 1.05))


log("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
t0 = time.time()
classes, recs, labs = build_index(DS)
assert classes == CLASS_NAMES, "class order mismatch"
ids = np.array([os.path.basename(r) for r in recs])
assert len(set(ids)) == len(ids)
folds = reproduce_folds(len(recs))
log("index built", len(recs), "records", f"{time.time() - t0:.0f}s", "fold sizes", [len(f) for f in folds])
Y = np.zeros((len(recs), len(classes)), np.uint8)
for i, l in enumerate(labs):
    Y[i, l] = 1
order_fingerprint = hashlib.sha256("\n".join(ids).encode()).hexdigest()

meta = {
    "classes": classes, "n_records": len(recs), "record_order_sha256": order_fingerprint,
    "fold_sizes": [len(f) for f in folds], "split": "RandomState(42).permutation(N) -> np.array_split(10); folds[9]=test",
    "checkpoints": {c: sha256(os.path.join(MP, c + ".pt")) for c in CKPTS},
    "threshold_files": {c: sha256(os.path.join(MP, c + "_thresh.npy")) for c in CKPTS if os.path.exists(os.path.join(MP, c + "_thresh.npy"))},
    "batch_size": BATCH, "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
    "torch": torch.__version__, "numpy": np.__version__, "dataset_dir": os.path.basename(DS),
    "fold_ids": {str(k): ids[folds[k]].tolist() for k in FOLDS},
}
json.dump(meta, open(os.path.join(OUT, "cache", "meta.json"), "w"))

models = {c: load_model(os.path.join(MP, c + ".pt"), dev) for c in CKPTS}


@torch.inference_mode()
def run(x, rngs=None):
    xt = torch.from_numpy(x).to(dev)
    lo = {c: m(xt).float() for c, m in models.items()}
    return {c: (l.cpu().numpy(), torch.sigmoid(l).cpu().numpy()) for c, l in lo.items()}


pool = ThreadPoolExecutor(6)
for k in FOLDS:
    path = os.path.join(CACHE, f"fold{k}.npz")
    if os.path.exists(path):
        log("fold", k, "cached, skipping")
        continue
    idx = folds[k]
    n = len(idx)
    res = {c: {"logits": np.zeros((n, 45), np.float32), "probs": np.zeros((n, 45), np.float32)} for c in CKPTS}
    extra = {}
    if k == 9:
        for c in CKPTS:
            extra[c + "|tta5"] = np.zeros((n, 45), np.float32)
            extra[c + "|floatfilt"] = np.zeros((n, 45), np.float32)
    seen = 0
    for b0 in range(0, n, BATCH):
        sel = idx[b0:b0 + BATCH]
        x = np.stack(list(pool.map(lambda i: load_ecg(recs[i] + ".mat", quantize_int16=True), sel)))
        assert x.shape[1:] == (12, 5000) and np.isfinite(x).all()
        out = run(x)
        for c in CKPTS:
            res[c]["logits"][b0:b0 + len(sel)], res[c]["probs"][b0:b0 + len(sel)] = out[c]
        if k == 9:
            accum = {c: out[c][1].copy() for c in CKPTS}
            for p in range(1, 5):
                rng = np.random.RandomState(1000 * p + b0)
                xa = np.stack([tta_augment(s, rng) for s in x])
                o = run(xa)
                for c in CKPTS:
                    accum[c] += o[c][1]
            for c in CKPTS:
                extra[c + "|tta5"][b0:b0 + len(sel)] = accum[c] / 5.0
            xf = np.stack(list(pool.map(lambda i: load_ecg(recs[i] + ".mat", quantize_int16=False), sel)))
            of = run(xf)
            for c in CKPTS:
                extra[c + "|floatfilt"][b0:b0 + len(sel)] = of[c][1]
        seen += len(sel)
        if (b0 // BATCH) % 10 == 0:
            log(f"fold {k}: {seen}/{n}")
    assert seen == n
    save = {"ids": ids[idx], "targets": Y[idx], "index": idx}
    for c in CKPTS:
        save[c + "|logits"] = res[c]["logits"]
        save[c + "|probs"] = res[c]["probs"]
        assert np.isfinite(res[c]["logits"]).all()
    for key, v in extra.items():
        assert np.isfinite(v).all()
        save[key] = v
    np.savez_compressed(path, **save)
    log("fold", k, "saved", path)
log("done", f"{time.time() - t0:.0f}s")
