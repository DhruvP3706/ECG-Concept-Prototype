"""Step 3 engineering smoke test: 5 records x 4 checkpoints. No training, no dataset copies.

Record choice: the first test-fold (folds[9], seed 42) records with distinct leading
labels and a clean signal (no constant lead, no int16 saturation). Fixed here by ID.
Writes results/step3/inference_smoke_test.json.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ecg_concept.inference import apply_thresholds, load_model, load_thresholds, predict_probs  # noqa: E402
from ecg_concept.labels import CLASS_NAMES, snomed_to_acronym  # noqa: E402
from ecg_concept.preprocessing import preprocess, read_raw_mat  # noqa: E402

RECORDS = ["JS42769", "JS37210", "JS10192", "JS33756", "JS11191"]
CKPTS = ["lightv2_ft_fold0", "lightv2_p1_fold0", "lightv2_p1_fold1", "lightv2_p1_fold2"]
MP = os.path.join(ROOT, "models", "pretrained")
DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
dev = "cuda" if torch.cuda.is_available() else "cpu"

s2a = snomed_to_acronym(DS)
raws, truth = [], []
for rid in RECORDS:
    base = glob.glob(os.path.join(DS, "WFDBRecords", "*", "*", rid + ".hea"))[0][:-4]
    raws.append(read_raw_mat(base + ".mat"))
    dx = next(l for l in open(base + ".hea") if l.startswith("#Dx")).split(":")[1].strip().split(",")
    truth.append(sorted(s2a[c] for c in dx if c in s2a and s2a[c] in CLASS_NAMES))
x = torch.from_numpy(np.stack([preprocess(r) for r in raws]))
x_float = torch.from_numpy(np.stack([preprocess(r, quantize_int16=False) for r in raws]))

out = {
    "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
    "torch": torch.__version__, "numpy": np.__version__, "dtype": "float32 (no autocast)",
    "preprocessing": {"source": "HonsExAI training notebook load_ecg_mat", "lead_order": "I,II,III,aVR,aVL,aVF,V1-V6",
                      "fs": 500, "length": 5000, "pad_crop": "zero-pad / crop tail",
                      "filter": "Butterworth order 3 bandpass 0.5-40 Hz, filtfilt (zero phase)",
                      "int16_truncation_after_filter": True, "normalisation": "per-lead z-score (eps 1e-8)"},
    "input": {"shape": list(x.shape), "dtype": str(x.dtype), "finite": bool(torch.isfinite(x).all()),
              "per_lead_mean_abs_max": float(x.mean(-1).abs().max()), "per_lead_std_range": [float(x.std(-1).min()), float(x.std(-1).max())]},
    "records": [{"id": r, "labels": t} for r, t in zip(RECORDS, truth)],
    "checkpoints": {},
}

probs_by = {}
for name in CKPTS:
    model = load_model(os.path.join(MP, name + ".pt"), dev)
    with torch.inference_mode():
        logits = model(x.to(dev))
        feats = model.forward_features(x.to(dev))
    probs = torch.sigmoid(logits.float()).cpu().numpy()
    probs_by[name] = probs
    cpu = load_model(os.path.join(MP, name + ".pt"), "cpu")
    with torch.inference_mode():
        probs_cpu = torch.sigmoid(cpu(x)).numpy()
        probs_noq = torch.sigmoid(cpu(x_float)).numpy()
    tpath = os.path.join(MP, name + "_thresh.npy")
    thr = load_thresholds(tpath) if os.path.exists(tpath) else None
    entry = {
        "load": "strict=True, weights_only=True OK", "trainable_params": sum(p.numel() for p in model.parameters()),
        "training_mode": model.training, "logits_shape": list(logits.shape), "probs_shape": list(probs.shape),
        "logits_range": [float(logits.min()), float(logits.max())], "probs_range": [float(probs.min()), float(probs.max())],
        "finite": bool(torch.isfinite(logits).all()),
        "max_abs_prob_diff_cuda_vs_cpu": float(np.abs(probs - probs_cpu).max()),
        "max_abs_prob_diff_int16trunc_vs_float_filter": float(np.abs(probs_cpu - probs_noq).max()),
        "forward_equals_forward_features_logits": bool(torch.equal(logits, feats["logits"])),
        "threshold_file": os.path.basename(tpath) if thr is not None else None,
        "per_record": [],
    }
    pred = apply_thresholds(probs, thr) if thr is not None else None
    for i, rid in enumerate(RECORDS):
        top = np.argsort(-probs[i])[:3]
        rec = {"id": rid, "truth": truth[i], "top3": [[CLASS_NAMES[j], round(float(probs[i, j]), 4)] for j in top]}
        if pred is not None:
            rec["pred_idx"] = np.where(pred[i])[0].tolist()
            rec["pred_labels"] = [CLASS_NAMES[j] for j in rec["pred_idx"]]
        else:
            rec["pred_idx"] = None
            rec["note"] = "no threshold file; no thresholded prediction made"
        entry["per_record"].append(rec)
    out["checkpoints"][name] = entry

# ---- Phase E: feature taps (p1_fold0) ----
model = load_model(os.path.join(MP, "lightv2_p1_fold0.pt"), dev)
cap = {}
h1 = model.tcn3b.register_forward_hook(lambda m, i, o: cap.__setitem__("tcn3b", o.detach()))
h2 = model.head[0].register_forward_hook(lambda m, i, o: cap.__setitem__("head0_pre_relu", o.detach()))
h3 = model.head[1].register_forward_hook(lambda m, i, o: cap.__setitem__("head1_relu", o.detach()))
with torch.inference_mode():
    plain = model(x.to(dev))
    f = model.forward_features(x.to(dev))
h1.remove(); h2.remove(); h3.remove()
fm = f["feature_map"]
ref = load_model(os.path.join(MP, "lightv2_p1_fold0.pt"), dev)
with torch.inference_mode():
    ref_logits = ref(x.to(dev))
xg = x[:1].clone().to(dev).requires_grad_(True)
model.eval()
g, = torch.autograd.grad(model.forward_features(xg)["feature_map"][0, :, 312].sum(), xg)
nz = torch.nonzero(g.abs().sum(1)[0] > 0).flatten()
out["feature_extraction"] = {
    "checkpoint": "lightv2_p1_fold0",
    "feature_map_tcn3b": {"shape": list(fm.shape), "min": float(fm.min()), "max": float(fm.max()), "per_time_step": True,
                          "hook_equals_forward_features": bool(torch.equal(cap["tcn3b"], fm)),
                          "samples_per_time_step": 5000 // fm.shape[-1]},
    "pooled_512": {"shape": list(f["pooled"].shape), "layout": "[global max (256) | global mean (256)]",
                   "equals_cat_of_feature_map_max_mean": bool(torch.equal(f["pooled"], torch.cat([fm.max(-1).values, fm.mean(-1)], 1)))},
    "hidden_256": {"shape": list(f["hidden"].shape), "post_relu": True, "hook_equals": bool(torch.equal(cap["head1_relu"], f["hidden"])),
                   "min": float(f["hidden"].min()), "zero_fraction": float((f["hidden"] == 0).float().mean())},
    "predictions_unchanged_by_extraction_and_hooks": bool(torch.equal(plain, ref_logits) and torch.equal(plain, f["logits"])),
    "receptive_field_of_one_feature_step_samples": [int(nz.min()), int(nz.max()), int(nz.numel())],
    "note_rf": "extent of non-zero input gradient for time step 312 of the feature map (centre); border effects absent",
}
os.makedirs(os.path.join(ROOT, "results", "step3"), exist_ok=True)
with open(os.path.join(ROOT, "results", "step3", "inference_smoke_test.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "checkpoints"}, indent=1))
for n, e in out["checkpoints"].items():
    print(n, {k: v for k, v in e.items() if k != "per_record"})
    for r in e["per_record"]:
        print("   ", r)
