"""PTB-XL baseline, stage 1: frozen LightECGNetV2 (ft_fold0) pooled 512-d embeddings for ELIGIBLE TRAIN (folds 1-8) and VALIDATION (fold 9)
records under three PREDECLARED preprocessing variants. Fold-10 records are never read (the manifest rows of role test_reserved are not used).

Variants (declared here and in config.json BEFORE any embedding is computed; selection rule is in the config):
  A_exact   : digital int16 as read -> existing `preprocess(quantize_int16=True)` (reference: the training-time pipeline, incl. int16 truncation)
  B_float   : digital int16 -> existing `preprocess(quantize_int16=False)` (no post-filter truncation)
  C_scaled  : digital counts / 4.88 (PTB-XL 1 uV/count -> hypothesised PhysioNet 4.88 uV/count), rounded, int16 -> `preprocess(quantize_int16=True)`
Per-lead z-scoring removes any per-lead gain; so the variants differ only through the int16 truncation (A vs C: its size relative to the signal; B: absent).
Z-scoring is per record and per lead (no dataset statistics), so no preprocessing parameter is fitted on any split.

Run: .venv\\Scripts\\python.exe -I scripts\\ptbxl_baseline_embed.py   (creates results/ptbxl_baseline/run_<stamp>/; never overwrites)
"""
import csv
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import ptbxl as px  # noqa: E402
from ecg_concept.inference import load_model  # noqa: E402
from ecg_concept.preprocessing import preprocess  # noqa: E402

BENCH = os.path.join(ROOT, "benchmarks", "ptbxl_v1")
CKPT = os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")
CKPT_SHA = "7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72"
VARIANTS = ["A_exact", "B_float", "C_scaled"]
CHUNK, BATCH = 1000, 64
PHYSIONET_UV_PER_COUNT = 4.88


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def variants(sig_i16):
    return {"A_exact": preprocess(sig_i16, quantize_int16=True),
            "B_float": preprocess(sig_i16, quantize_int16=False),
            "C_scaled": preprocess(np.rint(sig_i16.astype(np.float64) / PHYSIONET_UV_PER_COUNT).astype(np.int16), quantize_int16=True)}


def main():
    cfg = json.load(open(os.path.join(BENCH, "config.json"), encoding="utf-8"))
    res = json.load(open(os.path.join(BENCH, "reader_validation_results.json"), encoding="utf-8"))
    tvp, tep = os.path.join(BENCH, "manifest_train_val.csv"), os.path.join(BENCH, "manifest_test_reserved.csv")
    assert sha_file(tvp) == res["manifest_sha256"]["train_val"], "train/val manifest changed since freeze"
    assert sha_file(os.path.join(BENCH, "config.json")) == res["config_sha256"], "config changed since freeze"
    assert sha_file(CKPT) == CKPT_SHA, "checkpoint hash mismatch"
    rows = list(csv.DictReader(open(tvp, encoding="utf-8")))
    elig = [r for r in rows if r["excluded"] == "0" and r["reader_status"] == "ok"]
    n_tr, n_va = sum(r["role"] == "train" for r in elig), sum(r["role"] == "val" for r in elig)
    assert (n_tr, n_va) == (16289, 2034), (n_tr, n_va)
    assert all(int(r["strat_fold"]) <= 9 for r in elig)
    data = os.path.join(ROOT, cfg["dataset"]["root"])
    meta = {r.ecg_id: r for r in px.read_metadata(os.path.join(data, "ptbxl_database.csv"))}

    run = os.path.join(ROOT, "results", "ptbxl_baseline", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run)                                                          # fails if it exists: no overwrite
    os.makedirs(os.path.join(run, "embeddings"))
    config = {"stage": "embed", "variants": VARIANTS, "physionet_uv_per_count_hypothesis": PHYSIONET_UV_PER_COUNT, "chunk": CHUNK, "batch": BATCH,
              "selection_rule": {"metric": "mean over seeds 42,43,44 of validation macro-AUROC of the LINEAR head",
                                 "tie_rule": "if best variant exceeds A_exact by < 0.002 macro-AUROC, keep A_exact", "candidates": VARIANTS},
              "checkpoint": {"path": "models/pretrained/lightv2_ft_fold0.pt", "sha256": CKPT_SHA},
              "frozen_inputs": {"manifest_train_val_sha256": res["manifest_sha256"]["train_val"], "benchmark_config_sha256": res["config_sha256"],
                                "ptbxl_py_sha256": res["code_sha256"]["ptbxl.py"], "ptbxl_reader_py_sha256": res["code_sha256"]["ptbxl_reader.py"]},
              "eligible": {"train": n_tr, "validation": n_va}, "label_vocabulary": cfg["labels"]["vocabulary"], "fold10_waveforms_read": 0}
    json.dump(config, open(os.path.join(run, "config.json"), "x"), indent=1)
    log = open(os.path.join(run, "embed_log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CKPT, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    h0 = state_hash(model)
    say(f"device {dev}; eligible train {n_tr}, validation {n_va}; backbone state hash {h0[:16]}")
    t0 = time.time()
    n_bad = 0
    for ci in range(0, len(elig), CHUNK):
        part = elig[ci:ci + CHUNK]
        pooled = {v: np.zeros((len(part), 512), np.float32) for v in VARIANTS}
        stats = {v: [] for v in VARIANTS}
        for b in range(0, len(part), BATCH):
            sub = part[b:b + BATCH]
            xs = {v: [] for v in VARIANTS}
            for r in sub:
                ls = px.load_signal_native(meta[int(r["ecg_id"])], data)       # fold <= 9 only; raises for fold 10
                for v, x in variants(ls.signal).items():
                    xs[v].append(x)
            for v in VARIANTS:
                x = torch.from_numpy(np.stack(xs[v])).to(dev)
                with torch.inference_mode():
                    pooled[v][b:b + len(sub)] = model.forward_features(x)["pooled"].float().cpu().numpy()
        for v in VARIANTS:
            n_bad += int((~np.isfinite(pooled[v])).any(1).sum())
        np.savez(os.path.join(run, "embeddings", f"chunk_{ci // CHUNK:03d}.npz"), ids=np.array([int(r["ecg_id"]) for r in part]),
                 **{v: pooled[v] for v in VARIANTS})
        say(f"chunk {ci // CHUNK}: {ci + len(part)}/{len(elig)} done, {time.time() - t0:.0f}s elapsed, non-finite rows so far {n_bad}")
    h1 = state_hash(model)
    assert h0 == h1 and not model.training, "backbone changed"
    json.dump({"seconds": round(time.time() - t0), "backbone_state_hash_before": h0, "backbone_state_hash_after": h1, "unchanged": h0 == h1,
               "non_finite_rows": n_bad, "n_records": len(elig), "fold10_waveforms_read": 0}, open(os.path.join(run, "embed_summary.json"), "x"), indent=1)
    say(f"done in {time.time() - t0:.0f}s; backbone unchanged; non-finite rows {n_bad}")


if __name__ == "__main__":
    main()
