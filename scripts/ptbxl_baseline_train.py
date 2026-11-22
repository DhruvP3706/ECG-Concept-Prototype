"""PTB-XL baseline, stage 2: linear (mandatory) and small-MLP (secondary) heads on frozen 512-d embeddings, all preprocessing variants.
Train = folds 1-8 eligible (16,289), validation = fold 9 eligible (2,034). Fold 10 is not touched (not in the embedding cache).
Everything data-dependent (embedding mean/std, pos_weight) is fitted on eligible TRAINING rows only. Fold 9 is used for early stopping,
variant selection and per-class operating thresholds. Metrics are therefore validation-stage and optimistic for thresholded numbers.
Run: .venv\\Scripts\\python.exe -I scripts\\ptbxl_baseline_train.py <results/ptbxl_baseline/run_dir>   (writes <run>/train/ and models/ptbxl_heads/<run name>/; never overwrites)
"""
import csv
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import metrics as M  # noqa: E402

SEEDS = (42, 43, 44)
VOCAB = ["NORM", "MI", "STTC", "CD", "HYP"]
HEADS = ("linear", "mlp")
LR, WD, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 1e-4, 256, 100, 10
THR_GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def make_head(kind):
    if kind == "linear":
        return nn.Linear(512, 5)
    return nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 5))


def fit(kind, seed, Xtr, Ytr, Xva, Yva, pos_weight):
    torch.manual_seed(seed)                                                   # immediately before construction
    model = make_head(kind)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.RandomState(seed)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    xv, yv = torch.from_numpy(Xva), torch.from_numpy(Yva)
    best, best_state, bad, curve = float("inf"), None, 0, []
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(len(xt))
        tl = 0.0
        for i in range(0, len(perm), BATCH):
            idx = torch.from_numpy(perm[i:i + BATCH])
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tl += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(xv), yv))
            trl = float(loss_fn(model(xt), yt))                              # eval-mode training loss (comparable to validation loss)
        curve.append((ep, tl / len(perm), trl, vl))
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(xv)).numpy().astype(np.float64)
    best_ep = int(np.argmin([c[3] for c in curve])) + 1
    return model, p, curve, best_ep


def pick_thresholds(y, p):
    """Per class: the threshold in THR_GRID (0.05..0.95, step 0.01) maximising validation F1 with rule p > thr; first maximum wins."""
    out = np.zeros(y.shape[1])
    for c in range(y.shape[1]):
        f = [M.per_class_prf(y[:, [c]], (p[:, [c]] > t))[2][0] for t in THR_GRID]
        out[c] = THR_GRID[int(np.nanargmax(np.nan_to_num(f, nan=-1)))]
    return out


def evaluate(y, p):
    thr = pick_thresholds(y, p)
    rank = M.aggregate_ranking(y, p)
    auc, ap = M.auroc_columns(y, p), M.ap_columns(y, p)
    res = {"thresholds": thr.tolist(), "macro_auroc": rank["macro_auroc"], "micro_auroc": rank["micro_auroc"], "macro_ap": rank["macro_auprc"],
           "micro_ap": rank["micro_auprc"], "per_class_auroc": auc.tolist(), "per_class_ap": ap.tolist()}
    for tag, t in (("tuned", thr), ("t0.5", np.full(y.shape[1], 0.5))):
        pred = p > t[None, :]
        pr, rc, f1, (tp, fp, fn, tn) = M.per_class_prf(y, pred)
        tm = M.threshold_metrics(y, p, t)
        res[tag] = {"macro_f1": tm["macro_f1_valid"], "micro_f1": tm["micro_f1"], "micro_precision": tm["micro_precision"], "micro_recall": tm["micro_recall"],
                    "exact_match": tm["exact_match"], "hamming_loss": tm["hamming_loss"], "per_class_precision": pr.tolist(), "per_class_recall": rc.tolist(),
                    "per_class_f1": f1.tolist(), "confusion_tp_fp_fn_tn": [[float(a[c]) for a in (tp, fp, fn, tn)] for c in range(y.shape[1])]}
    return res


def main():
    run = os.path.abspath(sys.argv[1])
    cfg = json.load(open(os.path.join(run, "config.json")))
    summ = json.load(open(os.path.join(run, "embed_summary.json")))
    assert summ["unchanged"] and summ["non_finite_rows"] == 0, "embedding stage not clean"
    out = os.path.join(run, "train")
    heads_dir = os.path.join(ROOT, "models", "ptbxl_heads", os.path.basename(run))
    os.makedirs(out)
    os.makedirs(heads_dir)                                                    # both fail if present: no overwrite
    bench = os.path.join(ROOT, "benchmarks", "ptbxl_v1", "manifest_train_val.csv")
    assert sha_file(bench) == cfg["frozen_inputs"]["manifest_train_val_sha256"]
    man = {int(r["ecg_id"]): r for r in csv.DictReader(open(bench, encoding="utf-8"))}
    ids, emb = [], {v: [] for v in cfg["variants"]}
    for f in sorted(glob.glob(os.path.join(run, "embeddings", "chunk_*.npz"))):
        z = np.load(f)
        ids.append(z["ids"])
        for v in emb:
            emb[v].append(z[v])
    ids = np.concatenate(ids)
    emb = {v: np.concatenate(a).astype(np.float32) for v, a in emb.items()}
    assert len(ids) == len(set(ids)) == 18323
    role = np.array([man[i]["role"] for i in ids])
    Y = np.array([[int(man[i][c]) for c in VOCAB] for i in ids], np.float32)
    tr, va = role == "train", role == "val"
    assert (tr.sum(), va.sum()) == (16289, 2034) and (Y.sum(1) > 0).all()
    ytr, yva = Y[tr], Y[va]
    pos_weight = torch.from_numpy(((len(ytr) - ytr.sum(0)) / ytr.sum(0)).astype(np.float32))        # train-only
    log = open(os.path.join(out, "train_log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    say(f"train {tr.sum()} val {va.sum()}; train positives {ytr.sum(0).astype(int).tolist()}; val positives {yva.sum(0).astype(int).tolist()}; pos_weight {np.round(pos_weight.numpy(), 3).tolist()}")
    const_idx = np.where(ids == 12722)[0]
    say(f"record 12722 (constant-zero V5) present: {len(const_idx) == 1}; role {role[const_idx[0]] if len(const_idx) else None}; finite in all variants: "
        f"{all(np.isfinite(emb[v][const_idx]).all() for v in emb)}")
    results, curves, t0 = {}, [], time.time()
    for v in emb:
        X = emb[v]
        mu, sd = X[tr].mean(0), X[tr].std(0)                                  # TRAIN rows only
        sd = np.maximum(sd, 1e-6)
        Xs = ((X - mu) / sd).astype(np.float32)
        for kind in HEADS:
            for seed in SEEDS:
                model, p, curve, best_ep = fit(kind, seed, Xs[tr], ytr, Xs[va], yva, pos_weight)
                r = evaluate(yva.astype(bool), p)
                r.update(best_epoch=best_ep, epochs_run=len(curve), val_loss_best=min(c[3] for c in curve), train_loss_at_best=curve[best_ep - 1][2])
                results[f"{v}|{kind}|{seed}"] = r
                curves += [(v, kind, seed) + c for c in curve]
                ck = os.path.join(heads_dir, f"{v}_{kind}_seed{seed}.pt")
                torch.save({"state_dict": model.state_dict(), "emb_mean": mu, "emb_std": sd, "thresholds": r["thresholds"], "vocab": VOCAB,
                            "variant": v, "kind": kind, "seed": seed}, ck)
                say(f"{v} {kind} seed {seed}: best epoch {best_ep}/{len(curve)}, val loss {r['val_loss_best']:.4f}, macro-AUROC {r['macro_auroc']:.4f}, "
                    f"macro-AP {r['macro_ap']:.4f}, tuned macro-F1 {r['tuned']['macro_f1']:.4f}")
    # variant selection by the predeclared rule (linear head, mean val macro-AUROC over seeds)
    mean_auc = {v: float(np.mean([results[f"{v}|linear|{s}"]["macro_auroc"] for s in SEEDS])) for v in emb}
    best = max(mean_auc, key=mean_auc.get)
    selected = best if mean_auc[best] - mean_auc["A_exact"] >= 0.002 else "A_exact"
    say(f"mean linear validation macro-AUROC by variant: {mean_auc}; best {best}; SELECTED {selected} (rule: keep A_exact unless best exceeds it by >= 0.002)")
    json.dump({"results": results, "mean_linear_macro_auroc": mean_auc, "selected_variant": selected, "vocab": VOCAB, "seeds": SEEDS, "seconds": round(time.time() - t0),
               "hyper": {"lr": LR, "weight_decay": WD, "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "threshold_grid": [0.05, 0.95, 0.01]},
               "pos_weight": pos_weight.tolist(), "heads_dir": heads_dir, "head_checkpoint_sha256": {os.path.basename(f): sha_file(f) for f in sorted(glob.glob(os.path.join(heads_dir, "*.pt")))}},
              open(os.path.join(out, "results.json"), "x"), indent=1)
    with open(os.path.join(out, "training_curves.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "head", "seed", "epoch", "train_loss_running", "train_loss_eval", "val_loss"])
        w.writerows(curves)
    say(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
