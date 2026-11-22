"""PTB-XL experiment 2: END-TO-END fine-tuning of LightECGNetV2 (ft_fold0 init) + new 5-output linear head, seeds 42/43/44.
Same frozen manifests, labels (NORM, MI, STTC, CD, HYP), A_exact preprocessing and split as the frozen-linear baseline (results/ptbxl_baseline/run_20261120_002007).
Train = eligible folds 1-8 (16,289), validation = eligible fold 9 (2,034). Fold 10 is never read. The pretrained checkpoint is only READ (hash checked before and after).

Model: LightECGNetV2 feature extractor (every layer trainable, BatchNorm in train mode, dropout active) -> pooled 512-d -> NEW nn.Linear(512, 5).
The original 512->256->45 head stays in the module but is unused and frozen (excluded from the optimizer).
Optimizer: AdamW, backbone lr 1e-4, head lr 1e-3, weight_decay 1e-2 (both groups); schedule: 1 epoch linear warm-up (from 10% to 100% of each group's lr),
then cosine decay to 0 at epoch MAX_EPOCHS. Batch 64, fp32, no augmentation, grad-clip 5.0. Loss BCEWithLogits with train-only pos_weight.
Early stopping on validation (weighted BCE) loss, patience 5, max 30 epochs; best checkpoint = lowest validation loss (validation macro-AUROC is logged each epoch but is NOT used for selection).
Run: .venv\\Scripts\\python.exe -I scripts\\ptbxl_finetune.py        (creates results/ptbxl_finetune/run_<stamp>/ and models/ptbxl_finetune/run_<stamp>/)
"""
import csv
import datetime
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ecg_concept import metrics as M  # noqa: E402
from ecg_concept import ptbxl as px  # noqa: E402
from ecg_concept.inference import load_model  # noqa: E402
from ecg_concept.preprocessing import preprocess  # noqa: E402
import ptbxl_baseline_train as bt  # noqa: E402  (evaluate / pick_thresholds / VOCAB reused; its main() is not run)

BASE_RUN = os.path.join(ROOT, "results", "ptbxl_baseline", "run_20261120_002007")
CKPT = os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")
CKPT_SHA = "7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72"
SEEDS, VOCAB = (42, 43, 44), bt.VOCAB
LR_BACKBONE, LR_HEAD, WD, BATCH, MAX_EPOCHS, PATIENCE, CLIP, WARM_FRAC = 1e-4, 1e-3, 1e-2, 64, 30, 5, 5.0, 0.1


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class FineTuneNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.cls = nn.Linear(512, 5)

    def forward(self, x):
        return self.cls(self.backbone.forward_features(x)["pooled"])


def main():
    base_cfg = json.load(open(os.path.join(BASE_RUN, "config.json")))
    base_res = json.load(open(os.path.join(BASE_RUN, "train", "results.json")))
    bench = os.path.join(ROOT, "benchmarks", "ptbxl_v1", "manifest_train_val.csv")
    assert sha_file(bench) == base_cfg["frozen_inputs"]["manifest_train_val_sha256"], "manifest changed"
    assert sha_file(CKPT) == CKPT_SHA, "pretrained checkpoint hash mismatch"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = os.path.join(ROOT, "results", "ptbxl_finetune", "run_" + stamp)
    ck_dir = os.path.join(ROOT, "models", "ptbxl_finetune", "run_" + stamp)
    os.makedirs(run)
    os.makedirs(ck_dir)
    log = open(os.path.join(run, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    # ---- data: same eligible records and order as the frozen baseline's embedding cache
    ids = np.concatenate([np.load(f)["ids"] for f in sorted(glob.glob(os.path.join(BASE_RUN, "embeddings", "chunk_*.npz")))])
    man = {int(r["ecg_id"]): r for r in csv.DictReader(open(bench, encoding="utf-8"))}
    role = np.array([man[i]["role"] for i in ids])
    Y = np.array([[int(man[i][c]) for c in VOCAB] for i in ids], np.float32)
    tr, va = np.where(role == "train")[0], np.where(role == "val")[0]
    assert (len(tr), len(va)) == (16289, 2034) and (Y.sum(1) > 0).all() and all(int(man[i]["strat_fold"]) <= 9 for i in ids)
    pos_weight = torch.from_numpy(((len(tr) - Y[tr].sum(0)) / Y[tr].sum(0)).astype(np.float32))      # TRAIN only
    data = os.path.join(ROOT, base_cfg_root(base_cfg))
    meta = {r.ecg_id: r for r in px.read_metadata(os.path.join(data, "ptbxl_database.csv"))}
    cache = os.path.join(run, "a_exact_f32.npy")
    X = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32, shape=(len(ids), 12, 5000))
    t0 = time.time()
    for k, i in enumerate(ids):
        X[k] = preprocess(px.load_signal_native(meta[int(i)], data).signal, quantize_int16=True)       # A_exact; load_signal_native refuses fold 10
        if k % 3000 == 0:
            say(f"cached {k}/{len(ids)} ({time.time() - t0:.0f}s)")
    X.flush()
    say(f"A_exact cache built in {time.time() - t0:.0f}s: {cache}")
    X = np.load(cache, mmap_mode="r")
    dev = "cuda"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    # ---- sanity gate: cached inputs + pretrained backbone must reproduce the frozen baseline's A_exact embeddings
    frozen_emb = np.concatenate([np.load(f)["A_exact"] for f in sorted(glob.glob(os.path.join(BASE_RUN, "embeddings", "chunk_*.npz")))])
    probe = load_model(CKPT, dev)
    with torch.inference_mode():
        sel = np.array([0, 5000, 12000, 17000, int(np.where(ids == 12722)[0][0])])
        got = probe.forward_features(torch.from_numpy(np.stack([X[j] for j in sel])).to(dev))["pooled"].cpu().numpy()
    err = float(np.abs(got - frozen_emb[sel]).max())
    say(f"sanity: max |pooled - frozen baseline pooled| on 5 records = {err:.2e}")
    assert err < 1e-3, "cached inputs/backbone do not reproduce the frozen embeddings"
    del probe

    def run_eval(model, idx, bs=128):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                out.append(model(torch.from_numpy(np.stack([X[j] for j in idx[i:i + bs]])).to(dev)).cpu())
        return torch.cat(out)

    cfg = {"experiment": "ptbxl_finetune_end_to_end", "init_checkpoint": {"path": "models/pretrained/lightv2_ft_fold0.pt", "sha256": CKPT_SHA},
           "frozen_baseline_run": os.path.relpath(BASE_RUN, ROOT), "preprocessing": "A_exact (existing preprocess(quantize_int16=True) on digital int16)",
           "model": "LightECGNetV2 feature extractor (all layers trainable) -> pooled 512 -> new nn.Linear(512,5) (default PyTorch init, seeded); original 512-256-45 head unused and frozen",
           "optimizer": {"name": "AdamW", "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD, "weight_decay": WD, "grad_clip": CLIP},
           "scheduler": f"1 epoch linear warm-up from {WARM_FRAC:.0%} to 100% of each group lr, then cosine to 0 at epoch {MAX_EPOCHS}",
           "batch": BATCH, "max_epochs": MAX_EPOCHS, "early_stopping": f"validation weighted-BCE loss, patience {PATIENCE}, best = lowest val loss; val macro-AUROC logged only",
           "precision": "fp32", "augmentation": "none", "seeds": SEEDS, "pos_weight_train_only": pos_weight.tolist(),
           "counts": {"train": len(tr), "validation": len(va)}, "manifest_sha256": sha_file(bench), "fold10_waveforms_read": 0,
           "determinism": "cudnn.deterministic=True, benchmark=False; bitwise reproducibility on GPU not guaranteed"}
    json.dump(cfg, open(os.path.join(run, "config.json"), "x"), indent=1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dev))
    yva_t = torch.from_numpy(Y[va])
    all_res, curves = {}, []
    for seed in SEEDS:
        torch.manual_seed(seed)                                               # before ANY construction for this seed
        backbone = load_model(CKPT, "cpu")
        net = FineTuneNet(backbone)
        net.to(dev)
        for p in list(backbone.head.parameters()):
            p.requires_grad_(False)
        bb = [p for n, p in net.backbone.named_parameters() if not n.startswith("head.")]
        opt = torch.optim.AdamW([{"params": bb, "lr": LR_BACKBONE}, {"params": net.cls.parameters(), "lr": LR_HEAD}], weight_decay=WD)
        base_lrs = [LR_BACKBONE, LR_HEAD]
        steps_per = math.ceil(len(tr) / BATCH)
        total = MAX_EPOCHS * steps_per
        rng = np.random.RandomState(seed)
        best, bad, best_state, best_ep = float("inf"), 0, None, 0
        say(f"seed {seed}: trainable params {sum(p.numel() for p in bb) + sum(p.numel() for p in net.cls.parameters())} "
            f"(backbone {sum(p.numel() for p in bb)}, head {sum(p.numel() for p in net.cls.parameters())})")
        step, t_seed = 0, time.time()
        for ep in range(1, MAX_EPOCHS + 1):
            net.train()
            perm = rng.permutation(tr)
            tl, t_ep = 0.0, time.time()
            for i in range(0, len(perm), BATCH):
                idx = np.sort(perm[i:i + BATCH])
                warm = min(1.0, WARM_FRAC + (1 - WARM_FRAC) * step / steps_per)
                cos = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, step - steps_per) / max(1, total - steps_per))))
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * (warm if step < steps_per else cos)
                xb = torch.from_numpy(np.stack([X[j] for j in idx])).to(dev)
                yb = torch.from_numpy(Y[idx]).to(dev)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(net(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], CLIP)
                opt.step()
                tl += float(loss.detach()) * len(idx)
                step += 1
            logits = run_eval(net, va)
            vl = float(nn.functional.binary_cross_entropy_with_logits(logits, yva_t, pos_weight=pos_weight))
            pv = torch.sigmoid(logits).numpy().astype(np.float64)
            auc = float(np.nanmean(M.auroc_columns(Y[va].astype(bool), pv)))
            curves.append((seed, ep, tl / len(tr), vl, auc, opt.param_groups[0]["lr"], opt.param_groups[1]["lr"], round(time.time() - t_ep)))
            say(f"seed {seed} epoch {ep}: train loss {tl / len(tr):.4f} val loss {vl:.4f} val macro-AUROC {auc:.4f} ({time.time() - t_ep:.0f}s)")
            if vl < best - 1e-6:
                best, bad, best_ep = vl, 0, ep
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    say(f"seed {seed}: early stop at epoch {ep}")
                    break
        net.load_state_dict(best_state)
        logits = run_eval(net, va)
        p = torch.sigmoid(logits).numpy().astype(np.float64)
        r = bt.evaluate(Y[va].astype(bool), p)
        r.update(best_epoch=best_ep, epochs_run=ep, val_loss_best=best, seconds=round(time.time() - t_seed))
        all_res[str(seed)] = r
        ck = os.path.join(ck_dir, f"seed{seed}_best.pt")
        torch.save({"state_dict": best_state, "thresholds": r["thresholds"], "vocab": VOCAB, "seed": seed, "best_epoch": best_ep, "init_ckpt_sha256": CKPT_SHA}, ck)
        r["checkpoint_sha256"] = sha_file(ck)
        base = base_res["results"][f"A_exact|linear|{seed}"]
        say(f"seed {seed} BEST epoch {best_ep}: val loss {best:.4f} macro-AUROC {r['macro_auroc']:.4f} (frozen linear {base['macro_auroc']:.4f}), "
            f"macro-AP {r['macro_ap']:.4f} ({base['macro_ap']:.4f}), macro-F1@0.5 {r['t0.5']['macro_f1']:.4f} ({base['t0.5']['macro_f1']:.4f})")
        json.dump({"results": all_res, "vocab": VOCAB}, open(os.path.join(run, f"results_partial_seed{seed}.json"), "x"), indent=1)
        del net, opt, backbone
        torch.cuda.empty_cache()
    assert sha_file(CKPT) == CKPT_SHA, "pretrained checkpoint changed"
    json.dump({"results": all_res, "vocab": VOCAB, "config": cfg, "pretrained_checkpoint_unchanged": True, "frozen_baseline_results": os.path.relpath(os.path.join(BASE_RUN, "train", "results.json"), ROOT)},
              open(os.path.join(run, "results.json"), "x"), indent=1)
    with open(os.path.join(run, "training_curves.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "epoch", "train_loss_running", "val_loss", "val_macro_auroc", "lr_backbone", "lr_head", "epoch_seconds"])
        w.writerows(curves)
    say("done; pretrained checkpoint hash unchanged")


def base_cfg_root(_):
    return "data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"


if __name__ == "__main__":
    main()
