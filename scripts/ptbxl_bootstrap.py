"""PTB-XL validation-only paired patient-level cluster bootstrap (fold 9 ONLY; no training, no tuning, no fold-10 access).
Models (each with seeds 42/43/44, all pre-existing): frozen_linear and frozen_mlp (A_exact embeddings + saved heads, baseline run_20261120_002007),
finetuned (end-to-end checkpoints, finetune run_20261120_003238). Predictions are regenerated from the hash-verified saved checkpoints on the 2,034 eligible
fold-9 records, in one fixed record order, then aligned with the manifest (labels, patient IDs). The same sampled patient clusters are used for every model.
Reported statistic = per replicate, the metric averaged over the three seeds of a model (per-seed results are saved too). The CIs describe fold-9 sampling
uncertainty only; fold 9 was already used for early stopping, variant selection and thresholds, so they are NOT independent test evidence.
Run: .venv\\Scripts\\python.exe -I scripts\\ptbxl_bootstrap.py        (new directory results/ptbxl_bootstrap/run_<stamp>/)
"""
import csv
import datetime
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ecg_concept import metrics as M  # noqa: E402
from ecg_concept.inference import load_model  # noqa: E402
import ptbxl_baseline_train as bt  # noqa: E402
import ptbxl_finetune as ft  # noqa: E402

BASE = os.path.join(ROOT, "results", "ptbxl_baseline", "run_20261120_002007")
FT = os.path.join(ROOT, "results", "ptbxl_finetune", "run_20261120_003238")
HEADS = os.path.join(ROOT, "models", "ptbxl_heads", "run_20261120_002007")
FTCK = os.path.join(ROOT, "models", "ptbxl_finetune", "run_20261120_003238")
SEEDS = (42, 43, 44)
MODELS = ("frozen_linear", "frozen_mlp", "finetuned")
B, BOOT_SEED = 5000, 20261120
VOCAB = bt.VOCAB
HYP = VOCAB.index("HYP")
COMPARISONS = (("finetuned", "frozen_linear"), ("finetuned", "frozen_mlp"), ("frozen_mlp", "frozen_linear"))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    base_res = json.load(open(os.path.join(BASE, "train", "results.json")))
    ft_res = json.load(open(os.path.join(FT, "results.json")))
    base_cfg = json.load(open(os.path.join(BASE, "config.json")))
    ft_cfg = json.load(open(os.path.join(FT, "config.json")))
    bench = os.path.join(ROOT, "benchmarks", "ptbxl_v1", "manifest_train_val.csv")
    assert sha_file(bench) == base_cfg["frozen_inputs"]["manifest_train_val_sha256"] == ft_cfg["manifest_sha256"], "manifest mismatch"
    out = os.path.join(ROOT, "results", "ptbxl_bootstrap", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    log = open(os.path.join(out, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    # ---- checkpoints: hash-verified before loading
    paths = {}
    for s in SEEDS:
        paths[("frozen_linear", s)] = os.path.join(HEADS, f"A_exact_linear_seed{s}.pt")
        paths[("frozen_mlp", s)] = os.path.join(HEADS, f"A_exact_mlp_seed{s}.pt")
        paths[("finetuned", s)] = os.path.join(FTCK, f"seed{s}_best.pt")
        for m in ("frozen_linear", "frozen_mlp"):
            assert sha_file(paths[(m, s)]) == base_res["head_checkpoint_sha256"][os.path.basename(paths[(m, s)])], (m, s)
        assert sha_file(paths[("finetuned", s)]) == ft_res["results"][str(s)]["checkpoint_sha256"], ("ft", s)
    assert sha_file(ft.CKPT) == ft.CKPT_SHA
    say("all 9 prediction checkpoints and the pretrained backbone match their recorded hashes")

    # ---- fold-9 records in one fixed order (same construction as the baseline/finetune runs)
    ids_all = np.concatenate([np.load(f)["ids"] for f in sorted(glob.glob(os.path.join(BASE, "embeddings", "chunk_*.npz")))])
    emb_all = np.concatenate([np.load(f)["A_exact"] for f in sorted(glob.glob(os.path.join(BASE, "embeddings", "chunk_*.npz")))])
    man = {int(r["ecg_id"]): r for r in csv.DictReader(open(bench, encoding="utf-8"))}
    role = np.array([man[i]["role"] for i in ids_all])
    va = np.where(role == "val")[0]
    ids = ids_all[va]
    assert len(va) == 2034 and all(man[i]["strat_fold"] == "9" and man[i]["excluded"] == "0" for i in ids)
    Y = np.array([[int(man[i][c]) for c in VOCAB] for i in ids], np.int8)
    pat = np.array([int(man[i]["patient_id"]) for i in ids])
    assert (Y.sum(1) > 0).all() and not np.isnan(pat).any()
    say(f"fold-9 eligible records {len(ids)}, patients {len(set(pat))}, positives {Y.sum(0).tolist()}")

    # ---- predictions
    P = {}
    for s in SEEDS:
        for kind in ("frozen_linear", "frozen_mlp"):
            ck = torch.load(paths[(kind, s)], map_location="cpu", weights_only=False)             # own file, hash verified above
            head = bt.make_head("linear" if kind == "frozen_linear" else "mlp")
            head.load_state_dict(ck["state_dict"])
            head.eval()
            x = torch.from_numpy(((emb_all[va] - ck["emb_mean"]) / ck["emb_std"]).astype(np.float32))
            with torch.no_grad():
                P[(kind, s)] = torch.sigmoid(head(x)).numpy().astype(np.float64)
    cache = np.load(os.path.join(FT, "a_exact_f32.npy"), mmap_mode="r")
    for s in SEEDS:
        net = ft.FineTuneNet(load_model(ft.CKPT, "cpu"))
        net.load_state_dict(torch.load(paths[("finetuned", s)], map_location="cpu", weights_only=True)["state_dict"])
        net.cuda().eval()
        outl = []
        with torch.no_grad():
            for i in range(0, len(va), 128):
                outl.append(net(torch.from_numpy(np.stack([cache[j] for j in va[i:i + 128]])).cuda()).cpu())
        P[("finetuned", s)] = torch.sigmoid(torch.cat(outl)).numpy().astype(np.float64)
        del net
        torch.cuda.empty_cache()
    # ---- reproduction check against the recorded runs (stop if a model does not reproduce)
    obs = {}
    for (m, s), p in P.items():
        auc = float(np.nanmean(M.auroc_columns(Y.astype(bool), p)))
        rec = (ft_res["results"][str(s)] if m == "finetuned" else base_res["results"][f"A_exact|{'linear' if m == 'frozen_linear' else 'mlp'}|{s}"])["macro_auroc"]
        say(f"{m} seed {s}: reproduced macro-AUROC {auc:.6f} vs recorded {rec:.6f} (diff {auc - rec:+.2e})")
        assert abs(auc - rec) < 5e-4, f"{m} seed {s} does not reproduce the recorded result"
        obs[(m, s)] = auc
    np.savez(os.path.join(out, "predictions.npz"), ecg_id=ids, patient_id=pat, y=Y,
             **{f"{m}_seed{s}": p for (m, s), p in P.items()})

    # ---- cluster bootstrap
    upat, inv = np.unique(pat, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(upat)))
    lens = np.bincount(inv, minlength=len(upat))
    rng = np.random.RandomState(BOOT_SEED)
    draws = rng.randint(0, len(upat), size=(B, len(upat))).astype(np.int32)                 # patient-index draws, saved for exact replay
    np.savez_compressed(os.path.join(out, "bootstrap_patient_draws.npz"), draws=draws, unique_patient_ids=upat, seed=BOOT_SEED)
    keys = [(m, s) for m in MODELS for s in SEEDS]
    auc_b = np.full((B, len(keys), 5), np.nan)
    ap_b = np.full((B, len(keys), 5), np.nan)
    t0 = time.time()
    Yb = Y.astype(bool)
    for b in range(B):
        d = draws[b]
        idx = order[np.concatenate([np.arange(starts[k], starts[k] + lens[k]) for k in d])]
        yb = Yb[idx]
        for j, k in enumerate(keys):
            pb = P[k][idx]
            auc_b[b, j] = M.auroc_columns(yb, pb)
            ap_b[b, j] = M.ap_columns(yb, pb)
        if b % 500 == 0:
            say(f"replicate {b}/{B} ({time.time() - t0:.0f}s)")
    say(f"bootstrap done in {time.time() - t0:.0f}s")

    # ---- statistics. Invalid = a class with no positive or no negative in the replicate (metric undefined).
    def macro(a):                                                                           # nan-aware mean over classes
        with np.errstate(all="ignore"):
            return np.nanmean(a, axis=-1)

    invalid_cls = {"auroc": int(np.isnan(auc_b).any(axis=(1, 2)).sum()), "ap": int(np.isnan(ap_b).any(axis=(1, 2)).sum()),
                   "hyp_auroc": int(np.isnan(auc_b[:, :, HYP]).any(axis=1).sum())}
    say(f"replicates with any undefined class value: {invalid_cls} of {B}")
    stats = {}                                                                              # (model, metric) -> (B,) seed-mean per replicate
    perseed = {}
    for m in MODELS:
        js = [keys.index((m, s)) for s in SEEDS]
        stats[(m, "macro_auroc")] = macro(auc_b[:, js]).mean(1)
        stats[(m, "macro_ap")] = macro(ap_b[:, js]).mean(1)
        stats[(m, "hyp_auroc")] = auc_b[:, js, HYP].mean(1)
        for s, j in zip(SEEDS, js):
            perseed[(m, s)] = {"macro_auroc": macro(auc_b[:, j]), "macro_ap": macro(ap_b[:, j]), "hyp_auroc": auc_b[:, j, HYP]}
    METR = ("macro_auroc", "macro_ap", "hyp_auroc")
    full = {}
    for m in MODELS:
        a = np.mean([M.auroc_columns(Yb, P[(m, s)]) for s in SEEDS], axis=0)                # seed-mean point estimate on full fold 9
        full[(m, "macro_auroc")] = float(np.mean([np.nanmean(M.auroc_columns(Yb, P[(m, s)])) for s in SEEDS]))
        full[(m, "macro_ap")] = float(np.mean([np.nanmean(M.ap_columns(Yb, P[(m, s)])) for s in SEEDS]))
        full[(m, "hyp_auroc")] = float(a[HYP])

    def ci(v):
        ok = ~np.isnan(v)
        lo, hi = np.percentile(v[ok], [2.5, 97.5])
        return float(lo), float(hi), int((~ok).sum())

    rows, summary = [], {"models": {}, "comparisons": {}}
    for m in MODELS:
        for k in METR:
            lo, hi, nbad = ci(stats[(m, k)])
            rows.append(["model", m, "", k, round(full[(m, k)], 5), round(float(np.nanmean(stats[(m, k)])), 5), round(lo, 5), round(hi, 5), nbad, B - nbad])
            summary["models"].setdefault(m, {})[k] = {"fold9_point_estimate_seed_mean": full[(m, k)], "bootstrap_mean": float(np.nanmean(stats[(m, k)])), "ci95": [lo, hi], "invalid_replicates": nbad}
    for a, b2 in COMPARISONS:
        for k in METR:
            d = stats[(a, k)] - stats[(b2, k)]
            lo, hi, nbad = ci(d)
            pt = full[(a, k)] - full[(b2, k)]
            frac = float(np.nanmean(d > 0))
            rows.append(["difference", f"{a} - {b2}", "", k, round(pt, 5), round(float(np.nanmean(d)), 5), round(lo, 5), round(hi, 5), nbad, B - nbad])
            summary["comparisons"][f"{a} - {b2}"] = summary["comparisons"].get(f"{a} - {b2}", {})
            summary["comparisons"][f"{a} - {b2}"][k] = {"point_difference": pt, "bootstrap_mean_difference": float(np.nanmean(d)), "ci95": [lo, hi], "share_replicates_difference_gt_0": frac, "invalid_replicates": nbad}
        for s in SEEDS:                                                                      # supplementary: same-seed pairs
            for k in METR:
                d = perseed[(a, s)][k] - perseed[(b2, s)][k]
                lo, hi, nbad = ci(d)
                rows.append(["difference_same_seed", f"{a} - {b2}", s, k, round(float(obs[(a, s)] - obs[(b2, s)]) if k == "macro_auroc" else float("nan"), 5), round(float(np.nanmean(d)), 5), round(lo, 5), round(hi, 5), nbad, B - nbad])
    with open(os.path.join(out, "summary.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "model_or_comparison", "seed", "metric", "fold9_point_estimate", "bootstrap_mean", "ci95_low", "ci95_high", "invalid_replicates", "valid_replicates"])
        w.writerows(rows)
    np.savez_compressed(os.path.join(out, "bootstrap_distributions.npz"), auroc_per_class=auc_b.astype(np.float32), ap_per_class=ap_b.astype(np.float32),
                        keys=np.array([f"{m}|{s}" for m, s in keys]), classes=np.array(VOCAB),
                        **{f"seedmean_{m}_{k}": stats[(m, k)] for m in MODELS for k in METR})
    cfg = {"analysis": "paired patient-level cluster bootstrap, fold 9 only", "replicates": B, "bootstrap_seed": BOOT_SEED, "ci": "percentile 2.5/97.5",
           "unit": "patient_id (all eligible records of a sampled patient included; same draws for all models)",
           "statistic": "per replicate the metric is computed for each seed's model, then averaged over seeds 42/43/44; per-seed values in bootstrap_distributions.npz",
           "invalid_handling": "AUROC/AP of a class is undefined (NaN) if the resample has no positive or no negative; macro = mean over defined classes (flagged); a replicate undefined for a metric is excluded from that metric's CI and counted",
           "n_records": int(len(ids)), "n_patients": int(len(upat)), "n_positives": Y.sum(0).tolist(), "vocab": VOCAB,
           "inputs": {"baseline_run": os.path.relpath(BASE, ROOT), "finetune_run": os.path.relpath(FT, ROOT), "manifest_sha256": sha_file(bench),
                      "pretrained_ckpt_sha256": ft.CKPT_SHA, "head_checkpoints": base_res["head_checkpoint_sha256"],
                      "finetune_checkpoints": {s: ft_res["results"][s]["checkpoint_sha256"] for s in ("42", "43", "44")}},
           "invalid_replicates_any_class": invalid_cls, "fold10_waveforms_read": 0, "script_sha256": sha_file(os.path.abspath(__file__))}
    json.dump(cfg, open(os.path.join(out, "config.json"), "x"), indent=1)
    json.dump(summary, open(os.path.join(out, "summary.json"), "x"), indent=1)
    for p in ("predictions.npz", "bootstrap_patient_draws.npz", "bootstrap_distributions.npz", "summary.csv", "summary.json", "config.json"):
        say(f"{p}: {sha_file(os.path.join(out, p))}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
