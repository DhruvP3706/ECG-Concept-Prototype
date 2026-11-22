r"""Single-dataset training/fine-tuning driver for Phase 1 (Experiments A1, A2, B and the single-dataset stages of D). NOT YET EXECUTED on real data.

Input: a JSON config (see experiments/phase1/*.json) and a cache directory built by scripts/build_cache.py (x.npy float16 (N,12,5000), y.npy float32 with NaN for unknown labels, meta.csv with columns id, inner_fold).
Rows with inner_fold == k are the INNER VALIDATION set of the run; rows with inner_fold >= 0 and != k are the training set. Fold 9 / fold 10 / historical test partitions are separate caches this script never reads.
Selection metric: inner-validation macro-AUROC (eval_metrics.macro_auroc). Nothing else is used to pick epochs. Thresholds are NOT fitted here.
Usage:
  .venv\Scripts\python.exe -I scripts\train_v3.py --config experiments\phase1\B_ptbxl_scratch.json --inner-fold 0 --dry-run     # no training, no waveform reading beyond shape checks
  .venv\Scripts\python.exe -I scripts\train_v3.py --config experiments\phase1\B_ptbxl_scratch.json --inner-fold 0                 # trains (requires approval)
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import data_v3 as dv  # noqa: E402
from ecg_concept import eval_metrics as em  # noqa: E402
from ecg_concept import provenance as pv  # noqa: E402
from ecg_concept.lightecgnet_paper import ArchConfig, LightECGNet, count_params  # noqa: E402
from ecg_concept.losses_v3 import masked_bce_with_logits, pos_weight_from_labels  # noqa: E402
from ecg_concept.ptbxl_guard import Fold10Guard  # noqa: E402

CODE = ["scripts/train_v3.py", "src/ecg_concept/lightecgnet_paper.py", "src/ecg_concept/losses_v3.py", "src/ecg_concept/data_v3.py", "src/ecg_concept/eval_metrics.py", "src/ecg_concept/provenance.py", "src/ecg_concept/ptbxl_guard.py"]


def lr_at(step, steps_per_epoch, cfg):
    e = step / steps_per_epoch
    lr_max, lr_min, w, T = cfg["lr"], cfg["lr_min"], cfg["warmup_epochs"], cfg["epochs"]
    if e < w:
        return lr_max * (step + 1) / (w * steps_per_epoch)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (e - w) / max(T - w, 1e-9)))


def build_model(cfg, num_classes):
    m = LightECGNet(num_classes, ArchConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["arch"].items()}))
    init = cfg.get("init", {"type": "scratch"})
    if init["type"] in ("v2_checkpoint", "run_checkpoint"):
        state = torch.load(os.path.join(ROOT, init["path"]), map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not (k.startswith("head.") and init.get("drop_head", True))}
        missing = m.load_state_dict(state, strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        if init.get("drop_head", True):
            assert all(k.startswith("head.") for k in missing.missing_keys), missing.missing_keys
    return m


def load_cache(cache_dir, inner_fold):
    x = np.load(os.path.join(cache_dir, "x.npy"), mmap_mode="r")
    y = np.load(os.path.join(cache_dir, "y.npy"))
    with open(os.path.join(cache_dir, "meta.csv"), newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    fold = np.array([int(r["inner_fold"]) for r in meta])
    assert len(meta) == len(x) == len(y)
    tr, va = np.where((fold >= 0) & (fold != inner_fold))[0], np.where(fold == inner_fold)[0]
    assert len(va) > 0 and not set(tr) & set(va)
    return x, y, meta, tr, va


@torch.no_grad()
def predict(model, x, idx, device, bs=128):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        xb = torch.from_numpy(np.asarray(x[idx[i:i + bs]], np.float32)).to(device)
        out.append(torch.sigmoid(model(xb).float()).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_both(model, x, idx, device, bs=128):
    """Same computation as `predict` (eval mode, fp32, sigmoid of the fp32 logits) but also returns the logits. Logging-only helper."""
    model.eval()
    probs, logits = [], []
    for i in range(0, len(idx), bs):
        xb = torch.from_numpy(np.asarray(x[idx[i:i + bs]], np.float32)).to(device)
        lg = model(xb).float()
        probs.append(torch.sigmoid(lg).cpu().numpy())
        logits.append(lg.cpu())
    return np.concatenate(probs), torch.cat(logits)


def run(cfg, inner_fold, dry_run=False, out_root=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pv.verify_registry()
    guard = Fold10Guard().install_hooks()                                                   # any attempt to read a fold-10 file in this process is denied and logged
    seed = cfg["seed"] + inner_fold                                                         # approved fold-specific seed policy: config seed + inner fold (fold 0 -> 42, fold 1 -> 43, ...)
    # SEED FIX (Phase 1 correction): seed Python, NumPy, PyTorch CPU and CUDA and fix the deterministic flags BEFORE the model is built, so the initial weights are reproducible.
    # The fold-0 pilot built the model before seeding; its initial weights were not recorded and cannot be regenerated.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    cache = os.path.join(ROOT, cfg["cache_dir"]) if not os.path.isabs(cfg["cache_dir"]) else cfg["cache_dir"]
    nc = cfg["num_classes"]
    model = build_model(cfg, nc)
    t = cfg["train"]
    steps_note = {"parameters": count_params(model), "physical_batch": t["physical_batch"], "accumulation": t["accumulation"], "effective_batch": t["physical_batch"] * t["accumulation"], "precision": t["precision"], "device": device}
    if dry_run:
        with torch.no_grad():
            o = model.eval()(torch.randn(2, 12, 5000))
        info = {"config": cfg["name"], "inner_fold": inner_fold, **steps_note, "forward_check_shape": list(o.shape), "cache_exists": os.path.isfile(os.path.join(cache, "x.npy")), "fold10_hooks_installed": True}
        if info["cache_exists"]:
            x, y, meta, tr, va = load_cache(cache, inner_fold)
            info.update(n_train=int(len(tr)), n_val=int(len(va)), x_shape=list(x.shape), x_dtype=str(x.dtype), pos_weight_head=pos_weight_from_labels(y[tr])[:5].tolist())
        print(json.dumps(info, indent=1))
        return info
    x, y, meta, tr, va = load_cache(cache, inner_fold)
    out_dir = os.path.join(out_root or os.path.join(ROOT, "results", "phase1_runs"), f"{cfg['name']}_f{inner_fold}_s{seed}_{time.strftime('%Y%m%d_%H%M%S')}")
    pv.assert_not_reference_output(os.path.join(out_dir, "best.pt"))
    os.makedirs(out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "init.pt"))                                    # initial weights (CPU, before any update), hashed into the run manifest
    init_sha = pv.sha256_file(os.path.join(out_dir, "init.pt"))
    cache_man = os.path.join(cache, "cache_manifest.json")
    man = pv.write_run_manifest(out_dir, cfg, [seed], CODE, {"inner_fold": inner_fold, "n_train": int(len(tr)), "n_val": int(len(va)), "cache_dir": cfg["cache_dir"], "init_weights_sha256": init_sha,
                                                          "seed_policy": "seed = config.seed + inner_fold; python/numpy/torch(cpu+cuda) seeded and cudnn deterministic BEFORE model construction; torch+numpy re-seeded after construction (approved order) so dropout/augmentation streams are unchanged; augmentation and batch order use RandomState(seed); no DataLoader workers",
                                                          "cache_manifest_sha256": pv.sha256_file(cache_man) if os.path.isfile(cache_man) else None, "x_cache_sha256_from_manifest": json.load(open(cache_man)).get("x_sha256") if os.path.isfile(cache_man) else None})
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model.to(device)
    pw = torch.as_tensor(pos_weight_from_labels(y[tr], cap=cfg["loss"]["pos_weight_cap"]), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    amp = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[t["precision"]]
    scaler = torch.amp.GradScaler("cuda", enabled=(t["precision"] == "fp16" and device == "cuda"))
    pb, acc = t["physical_batch"], t["accumulation"]
    eff = pb * acc
    spe = max(1, len(tr) // eff)
    hist, best, best_ep, bad, step = [], -1.0, -1, 0, 0
    sw = dv.sampling_weights(y[tr]) if cfg.get("sampler") == "inverse_sqrt_freq" else None
    for ep in range(1, t["epochs"] + 1):
        model.train()
        t0 = time.time()
        order = rng.choice(len(tr), len(tr), p=sw) if sw is not None else rng.permutation(len(tr))
        tot, nb = 0.0, 0
        gns = []                                                                            # logging only: pre-clipping gradient norms
        for s in range(spe):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, spe, {"lr": t["lr"], "lr_min": t["lr_min"], "warmup_epochs": t["warmup_epochs"], "epochs": t["epochs"]})
            opt.zero_grad(set_to_none=True)
            for a in range(acc):
                idx = tr[order[(s * acc + a) * pb:(s * acc + a + 1) * pb]]
                xb, yb = np.asarray(x[np.sort(idx)], np.float32), y[np.sort(idx)]
                if cfg["augment"]["enabled"]:
                    xb = dv.paper_augment(xb, rng, **cfg["augment"].get("params", {}))
                if cfg["augment"].get("mixup_alpha", 0) > 0:
                    xb, yb, _, _ = dv.mixup(xb, yb, rng, cfg["augment"]["mixup_alpha"])
                xt, yt = torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device)
                with torch.autocast(device, dtype=amp, enabled=amp is not None):
                    lg = model(xt)
                loss, _ = masked_bce_with_logits(lg.float(), yt, pos_weight=pw)
                scaler.scale(loss / acc).backward()
                tot += float(loss.detach())
                nb += 1
            scaler.unscale_(opt)
            gns.append(torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]).detach())   # return value (pre-clip norm) is only logged
            scaler.step(opt)
            scaler.update()
            step += 1
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0, "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None}
        if ep % t["eval_every"] == 0 or ep == t["epochs"]:
            p, lg_va = predict_both(model, x, va, device)                                   # identical probabilities to predict()
            auc = em.macro_auroc(y[va].astype(int), p)
            rec["val_macro_auroc"] = auc
            # ---- logging only (no influence on training or on selection): validation loss, training-subset AUROC/loss in eval mode, gradient norms
            yv = torch.from_numpy(y[va]).to(lg_va.device)
            rec["val_loss_weighted"] = float(masked_bce_with_logits(lg_va, yv, pos_weight=pw.to(lg_va.device))[0])
            rec["val_loss_unweighted"] = float(masked_bce_with_logits(lg_va, yv)[0])
            p_tr, lg_tr = predict_both(model, x, tr, device)
            rec["train_macro_auroc_eval"] = em.macro_auroc(y[tr].astype(int), p_tr)
            rec["train_loss_weighted_eval"] = float(masked_bce_with_logits(lg_tr, torch.from_numpy(y[tr]), pos_weight=pw.cpu())[0])
            g = torch.stack(gns).float().cpu()
            rec.update(grad_norm_mean=float(g.mean()), grad_norm_max=float(g.max()), grad_norm_frac_clipped=float((g > t["grad_clip"]).float().mean()), grad_nonfinite_steps=int((~torch.isfinite(g)).sum()))
            if auc > best:
                best, best_ep, bad = auc, ep, 0
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
                np.save(os.path.join(out_dir, "oof_val_probs.npy"), p)
                np.save(os.path.join(out_dir, "oof_val_idx.npy"), va)
            else:
                bad += t["eval_every"]
        hist.append(rec)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)
        print(rec, flush=True)
        if bad >= t["patience"]:
            break
    summary = {"best_epoch": best_ep, "best_inner_val_macro_auroc": best, "run_dir": out_dir, "best_ckpt_sha256": pv.sha256_file(os.path.join(out_dir, "best.pt")) if best_ep > 0 else None, "fold10_waveform_reads_allowed": guard.n_allowed_reads()}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--inner-fold", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    with open(a.config, encoding="utf-8") as f:
        c = json.load(f)
    run(c, a.inner_fold, dry_run=a.dry_run)
