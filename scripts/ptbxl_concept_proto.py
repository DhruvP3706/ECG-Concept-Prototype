"""PTB-XL strict RR-concept -> anchored-prototype bottleneck (validation-stage; fold 10 never read).
Reuses the Step 4.4 components (src/ecg_concept/concept_proto.py) with PTB-XL specifics: 5 diagnostic superclasses, BCE with TRAIN-only pos_weight, eligible records only.
Everything is fitted on eligible TRAIN rows (folds 1-8): embedding/concept standardizers, the concept predictor, k-means, prototype anchors (real training records
whose four automatic concepts were all ACCEPTED), tau, the heads. Fold 9 is used for early stopping (weighted BCE / masked MSE) and evaluation only.
Models (seeds 42, 43, 44):
  C2  strict concept-only   Linear(4,5) on FROZEN predicted standardized concepts            (no embedding path)
  D   anchored prototypes   16 anchors in predicted-concept space; s=exp(-d^2/tau); Linear(16,5) on similarities only
  CTRL_mask_only            Linear(19+,5) on acceptance/quality/reason flags (no concept values) - control
  CTRL_shuffled_concepts    C2 trained on train concept vectors permuted across records (labels intact) - control; validation uses the real predicted concepts
The concept predictor (ConceptMLP 512-128-4, masked MSE on ACCEPTED automatic targets) is frozen before any diagnosis head is trained and is not updated by the diagnosis loss.
Concept R2/MAE/RMSE measure agreement with AUTOMATIC extractor targets, not clinical validity.
Run: .venv\\Scripts\\python.exe -I scripts\\ptbxl_concept_proto.py <results/ptbxl_concepts/run_dir>
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
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ecg_concept import concept_proto as cp  # noqa: E402
from ecg_concept import metrics as M  # noqa: E402
import ptbxl_baseline_train as bt  # noqa: E402

BASE = os.path.join(ROOT, "results", "ptbxl_baseline", "run_20261120_002007")
BOOT = sorted(glob.glob(os.path.join(ROOT, "results", "ptbxl_bootstrap", "run_*")))[-1]
CONCEPTS = ["accepted_rate_mean_bpm", "accepted_rr_cv", "accepted_rr_nrmssd", "accepted_rr_mad_succ"]
SEEDS, VOCAB, K = (42, 43, 44), bt.VOCAB, 16
LR_C, LR_H, WD, BATCH, MAXEP, PAT = 1e-3, 1e-2, 1e-4, 256, 100, 10
B, BOOT_SEED = 5000, 20261120
STATES = {"ACCEPTED": 2, "UNCERTAIN": 1, "UNAVAILABLE": 0}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_pw(model, Xtr, Ytr, Xva, Yva, seed, lr, pos_weight):
    """Step 4.4 fit_head semantics (AdamW, early stopping on validation loss, best state restored) with the baseline's BCE pos_weight."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    Xtr, Ytr, Xva, Yva = (torch.as_tensor(a, dtype=torch.float32) for a in (Xtr, Ytr, Xva, Yva))
    best, state, bad, hist = float("inf"), None, 0, {"train": [], "val": []}
    for ep in range(MAXEP):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for b0 in range(0, len(perm), BATCH):
            ix = perm[b0:b0 + BATCH]
            opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(model(Xtr[ix]), Ytr[ix], pos_weight=pos_weight)
            l.backward()
            opt.step()
            tot += float(l) * len(ix)
        model.eval()
        with torch.no_grad():
            v = float(F.binary_cross_entropy_with_logits(model(Xva), Yva, pos_weight=pos_weight))
        hist["train"].append(tot / len(perm))
        hist["val"].append(v)
        assert np.isfinite(v)
        if v < best - 1e-7:
            best, state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PAT:
                break
    model.load_state_dict(state)
    model.eval()
    hist.update(best_val=best, best_epoch=int(np.argmin(hist["val"])) + 1, epochs_run=len(hist["val"]))
    return model, hist


def load():
    ids = np.concatenate([np.load(f)["ids"] for f in sorted(glob.glob(os.path.join(BASE, "embeddings", "chunk_*.npz")))]).astype(int)
    emb = np.concatenate([np.load(f)["A_exact"] for f in sorted(glob.glob(os.path.join(BASE, "embeddings", "chunk_*.npz")))]).astype(np.float32)
    man = {int(r["ecg_id"]): r for r in csv.DictReader(open(os.path.join(ROOT, "benchmarks", "ptbxl_v1", "manifest_train_val.csv"), encoding="utf-8"))}
    return ids, emb, man


def main():
    run = os.path.abspath(sys.argv[1])
    chk = json.load(open(os.path.join(run, "validation_checks.json")))
    assert chk["all_pass"], "concept extraction did not pass its predeclared criteria; training is not allowed"
    ccfg = json.load(open(os.path.join(run, "config.json")))
    cs_path = os.path.join(run, "concepts.csv")
    assert sha_file(cs_path) == chk["concepts_csv_sha256"]
    out = os.path.join(run, "proto")
    os.makedirs(out)
    ckd = os.path.join(ROOT, "models", "ptbxl_concept_proto", os.path.basename(run))
    os.makedirs(ckd)
    log = open(os.path.join(out, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    ids, emb, man = load()
    rows = {int(r["ecg_id"]): r for r in csv.DictReader(open(cs_path, encoding="utf-8"))}
    assert list(rows) == ids.tolist()
    role = np.array([man[i]["role"] for i in ids])
    tr, va = role == "train", role == "val"
    assert (tr.sum(), va.sum()) == (16289, 2034)
    Y = np.array([[int(man[i][c]) for c in VOCAB] for i in ids], np.float32)
    T = np.array([[float(rows[i][c]) for c in CONCEPTS] for i in ids])
    acc = np.isfinite(T)
    full = acc.all(1)
    rs = np.array([STATES[rows[i]["rate_state"]] for i in ids])
    irs = np.array([STATES[rows[i]["irr_state"]] for i in ids])
    assert (full == ((rs == 2) & (irs == 2))).all(), "accepted masks inconsistent with states"
    reason_names = sorted({c for i in ids for c in str(rows[i]["reason_codes"]).split(";") if c})
    rf = np.array([[c in str(rows[i]["reason_codes"]).split(";") for c in reason_names] for i in ids], np.float64)
    nf = lambda c: np.array([float(rows[i][c]) for i in ids])
    Fm = np.column_stack([rs == 2, irs == 2, rs == 1, irs == 1, nf("n_usable_leads"), nf("n_flat_leads"), nf("n_clip_leads"), nf("n_nonfinite_leads"), rf]).astype(np.float64)
    pos_weight = torch.from_numpy(((tr.sum() - Y[tr].sum(0)) / Y[tr].sum(0)).astype(np.float32))
    pops = {"val_all": np.ones(int(va.sum()), bool), "val_full_accepted": full[va], "val_not_full_accepted": ~full[va]}
    say(f"train {tr.sum()} val {va.sum()}; fully-accepted train {int((full & tr).sum())} val {int((full & va).sum())}; accepted per concept train "
        f"{acc[tr].sum(0).tolist()} val {acc[va].sum(0).tolist()}; reason flags {reason_names}")
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)

    def run_seed(seed):
        o = {"probs": {}, "hist": {}, "state": {}}
        es = cp.Standardizer.fit(emb[tr])
        X = es.transform(emb).astype(np.float32)
        cs = cp.Standardizer.fit(T[tr])
        Tz = cs.transform(T)
        torch.manual_seed(seed)
        mC = cp.ConceptMLP(512, 128, 4)
        mC, h = cp.fit_head(mC, X[tr], np.nan_to_num(Tz[tr]), X[va], np.nan_to_num(Tz[va]), seed, LR_C, MAXEP, PAT, BATCH, loss="mse", mask_tr=acc[tr], mask_va=acc[va])
        for p in mC.parameters():
            p.requires_grad_(False)
        mC.eval()
        hb = hashlib.sha256(b"".join(v.numpy().tobytes() for v in mC.state_dict().values())).hexdigest()
        Zhat = cp.predict(mC, X).astype(np.float64)
        assert np.isfinite(Zhat).all()
        o["hist"]["concept_predictor"], o["state"]["concept_predictor"], o["Zhat"] = h, mC.state_dict(), Zhat
        const = np.nanmean(T[tr], axis=0)
        rows_c = []
        for split, sel in (("train", tr), ("validation", va)):
            pred = cs.inverse(Zhat[sel])
            for i, c in enumerate(CONCEPTS):
                m = acc[sel, i]
                e = pred[m, i] - T[sel][m, i]
                ce = const[i] - T[sel][m, i]
                rows_c.append({"seed": seed, "split": split, "concept": c, "n_accepted": int(m.sum()), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                               "const_mae": float(np.abs(ce).mean()), "const_rmse": float(np.sqrt((ce ** 2).mean())), "r2_vs_train_mean": float(1 - (e ** 2).sum() / (ce ** 2).sum()),
                               "pearson_r": float(np.corrcoef(pred[m, i], T[sel][m, i])[0, 1])})
        o["concept_eval"] = rows_c
        # C2 strict concept-only
        torch.manual_seed(seed)
        m2, h = fit_pw(nn.Linear(4, 5), Zhat[tr], Y[tr], Zhat[va], Y[va], seed, LR_H, pos_weight)
        o["probs"]["C2_concept_only"] = sig(cp.predict(m2, Zhat[va])).astype(np.float64)
        o["hist"]["C2_concept_only"], o["state"]["C2_concept_only"] = h, m2.state_dict()
        # D anchored prototypes (candidates: TRAIN records with all four concepts accepted)
        fa = np.where(full & tr)[0]
        centres, assign = cp.kmeans_det(Zhat[fa], K)
        anchor = fa[cp.anchor_records(Zhat[fa], centres)]
        assert tr[anchor].all() and not va[anchor].any()
        P = Zhat[anchor]
        tau = cp.tau_rule(Zhat[tr], P)
        S = cp.similarities(Zhat, P, tau).astype(np.float32)
        torch.manual_seed(seed)
        mD, h = fit_pw(cp.ProtoHead(K, 5), S[tr], Y[tr], S[va], Y[va], seed, LR_H, pos_weight)
        assert hashlib.sha256(b"".join(v.numpy().tobytes() for v in mC.state_dict().values())).hexdigest() == hb, "concept predictor changed"
        o["probs"]["D_prototype"] = sig(cp.predict(mD, S[va])).astype(np.float64)
        o["hist"]["D_prototype"], o["state"]["D_prototype"] = h, mD.state_dict()
        o["proto"] = {"anchor_ecg_id": ids[anchor].tolist(), "anchor_index": anchor.tolist(), "tau": tau, "cluster_sizes": np.bincount(assign, minlength=K).tolist(),
                      "P": P.tolist(), "head_weight": mD.fc.weight.detach().numpy().tolist(), "head_bias": mD.fc.bias.detach().numpy().tolist(),
                      "mean_similarity_train": S[tr].mean(0).tolist(), "concept_predictor_sha": hb,
                      "anchor_concepts_original_units": cs.inverse(P).tolist()}
        # controls
        fs = cp.Standardizer.fit(Fm[tr])
        Fz = fs.transform(Fm).astype(np.float32)
        torch.manual_seed(seed)
        mM, h = fit_pw(nn.Linear(Fz.shape[1], 5), Fz[tr], Y[tr], Fz[va], Y[va], seed, LR_H, pos_weight)
        o["probs"]["CTRL_mask_only"] = sig(cp.predict(mM, Fz[va])).astype(np.float64)
        o["hist"]["CTRL_mask_only"], o["state"]["CTRL_mask_only"] = h, mM.state_dict()
        perm = np.random.RandomState(seed).permutation(int(tr.sum()))
        torch.manual_seed(seed)
        mS, h = fit_pw(nn.Linear(4, 5), Zhat[tr][perm], Y[tr], Zhat[va], Y[va], seed, LR_H, pos_weight)     # concepts shuffled across training records
        o["probs"]["CTRL_shuffled_concepts"] = sig(cp.predict(mS, Zhat[va])).astype(np.float64)
        o["hist"]["CTRL_shuffled_concepts"], o["state"]["CTRL_shuffled_concepts"] = h, mS.state_dict()
        return o

    t0 = time.time()
    runs = {}
    for s in SEEDS:
        runs[s] = run_seed(s)
        eps = {k: v["epochs_run"] for k, v in runs[s]["hist"].items()}
        say(f"seed {s} done ({time.time() - t0:.0f}s); epochs {eps}; tau {runs[s]['proto']['tau']:.4f}")
    rep = run_seed(42)
    repro = {k: float(np.abs(rep["probs"][k] - runs[42]["probs"][k]).max()) for k in rep["probs"]}
    say(f"reproducibility (seed-42 rerun) max |prob diff|: {repro}")

    # ---- evaluation
    Yva = Y[va].astype(bool)
    models = ["C2_concept_only", "D_prototype", "CTRL_mask_only", "CTRL_shuffled_concepts"]
    res = {}
    for m in models:
        for s in SEEDS:
            for pop, sel in pops.items():
                if sel.sum() == 0:
                    continue
                r = bt.evaluate(Yva[sel], runs[s]["probs"][m][sel])
                res[f"{m}|{pop}|{s}"] = r
    # frozen baselines (recorded, for the table) and the exact same fold-9 predictions
    pred = np.load(os.path.join(BOOT, "predictions.npz"))
    assert (pred["ecg_id"] == ids[va]).all()
    base_p = {f"{m}": {s: pred[f"{m}_seed{s}"] for s in SEEDS} for m in ("frozen_linear", "frozen_mlp", "finetuned")}
    # ---- bootstrap with the SAME patient draws (seed 20261120, 5000 replicates) as the earlier bootstrap
    dz = np.load(os.path.join(BOOT, "bootstrap_patient_draws.npz"))
    draws, upat = dz["draws"], dz["unique_patient_ids"]
    pat = np.array([int(man[i]["patient_id"]) for i in ids[va]])
    assert int(dz["seed"]) == BOOT_SEED and draws.shape[0] == B and (np.unique(pat) == upat).all() and (pred["patient_id"] == pat).all() and (pred["y"] == Y[va]).all()
    _, inv = np.unique(pat, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(upat)))
    lens = np.bincount(inv, minlength=len(upat))
    allm = {**{k: base_p[k] for k in base_p}, **{m: {s: runs[s]["probs"][m] for s in SEEDS} for m in models}}
    names = list(allm)
    keys = [(m, s) for m in names for s in SEEDS]
    HYP = VOCAB.index("HYP")
    auc_b = np.zeros((B, len(keys), 5))
    ap_b = np.zeros((B, len(keys), 5))
    t1 = time.time()
    for b in range(B):
        idx = order[np.concatenate([np.arange(starts[k], starts[k] + lens[k]) for k in draws[b]])]
        yb = Yva[idx]
        for j, (m, s) in enumerate(keys):
            pb = allm[m][s][idx]
            auc_b[b, j] = M.auroc_columns(yb, pb)
            ap_b[b, j] = M.ap_columns(yb, pb)
    say(f"bootstrap {B} replicates in {time.time() - t1:.0f}s; replicates with an undefined class value: {int(np.isnan(auc_b).any(axis=(1, 2)).sum())} (AUROC), {int(np.isnan(ap_b).any(axis=(1, 2)).sum())} (AP)")
    stat = {}
    for m in names:
        js = [keys.index((m, s)) for s in SEEDS]
        stat[(m, "macro_auroc")] = np.nanmean(auc_b[:, js], axis=2).mean(1)
        stat[(m, "macro_ap")] = np.nanmean(ap_b[:, js], axis=2).mean(1)
        stat[(m, "hyp_auroc")] = auc_b[:, js, HYP].mean(1)
    full_pt = {}
    for m in names:
        full_pt[(m, "macro_auroc")] = float(np.mean([np.nanmean(M.auroc_columns(Yva, allm[m][s])) for s in SEEDS]))
        full_pt[(m, "macro_ap")] = float(np.mean([np.nanmean(M.ap_columns(Yva, allm[m][s])) for s in SEEDS]))
        full_pt[(m, "hyp_auroc")] = float(np.mean([M.auroc_columns(Yva, allm[m][s])[HYP] for s in SEEDS]))
    METR = ("macro_auroc", "macro_ap", "hyp_auroc")
    ci = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])] + [int(np.isnan(v).sum())]
    boot_rows = []
    for m in names:
        for k in METR:
            lo, hi, nb = ci(stat[(m, k)])
            boot_rows.append(["model", m, k, full_pt[(m, k)], lo, hi, nb])
    comps = [("D_prototype", "frozen_linear"), ("D_prototype", "frozen_mlp"), ("D_prototype", "finetuned"), ("C2_concept_only", "frozen_linear"),
             ("C2_concept_only", "frozen_mlp"), ("D_prototype", "C2_concept_only"), ("D_prototype", "CTRL_mask_only"), ("C2_concept_only", "CTRL_mask_only"),
             ("D_prototype", "CTRL_shuffled_concepts"), ("C2_concept_only", "CTRL_shuffled_concepts")]
    for a_, b_ in comps:
        for k in METR:
            d = stat[(a_, k)] - stat[(b_, k)]
            lo, hi, nb = ci(d)
            boot_rows.append(["difference", f"{a_} - {b_}", k, full_pt[(a_, k)] - full_pt[(b_, k)], lo, hi, nb])
    with open(os.path.join(out, "bootstrap_summary.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "model_or_comparison", "metric", "fold9_point_estimate_seed_mean", "ci95_low", "ci95_high", "invalid_replicates"])
        w.writerows(boot_rows)
    # ---- artefacts
    np.savez(os.path.join(out, "predictions_fold9.npz"), ecg_id=ids[va], patient_id=pat, y=Y[va], **{f"{m}_seed{s}": runs[s]["probs"][m] for m in models for s in SEEDS})
    np.savez(os.path.join(out, "concept_predictions_all.npz"), ecg_id=ids, **{f"zhat_seed{s}": runs[s]["Zhat"] for s in SEEDS}, targets=T, accepted=acc)
    for s in SEEDS:
        torch.save({"state": runs[s]["state"], "proto": runs[s]["proto"], "seed": s}, os.path.join(ckd, f"seed{s}.pt"))
    with open(os.path.join(out, "concept_prediction_metrics.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[42]["concept_eval"][0]))
        w.writeheader()
        for s in SEEDS:
            w.writerows(runs[s]["concept_eval"])
    json.dump({"results": res, "seeds": SEEDS, "vocab": VOCAB, "repro": repro, "hist": {str(s): {k: {x: v[x] for x in ("best_epoch", "epochs_run", "best_val")} for k, v in runs[s]["hist"].items()} for s in SEEDS},
               "proto": {str(s): {k: v for k, v in runs[s]["proto"].items() if k not in ("P",)} for s in SEEDS}, "pos_weight": pos_weight.tolist(),
               "populations": {"train_eligible": int(tr.sum()), "val_eligible": int(va.sum()), "train_fully_accepted": int((full & tr).sum()), "val_fully_accepted": int((full & va).sum()),
                               "train_accepted_per_concept": acc[tr].sum(0).tolist(), "val_accepted_per_concept": acc[va].sum(0).tolist()},
               "hyper": {"lr_concept": LR_C, "lr_heads": LR_H, "weight_decay": WD, "batch": BATCH, "max_epochs": MAXEP, "patience": PAT, "K": K, "bootstrap_B": B, "bootstrap_seed": BOOT_SEED,
                         "bootstrap_draws_file": os.path.relpath(os.path.join(BOOT, "bootstrap_patient_draws.npz"), ROOT), "bootstrap_draws_sha256": sha_file(os.path.join(BOOT, "bootstrap_patient_draws.npz"))},
               "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("scripts/ptbxl_concept_proto.py", "src/ecg_concept/concept_proto.py", "src/ecg_concept/rr_concepts.py", "src/ecg_concept/beat_detection.py")},
               "inputs": {"concepts_csv_sha256": chk["concepts_csv_sha256"], "extraction_config": ccfg["extractor_params_sha256"], "manifest_sha256": ccfg["manifest_sha256"],
                          "baseline_run": os.path.relpath(BASE, ROOT), "bootstrap_run": os.path.relpath(BOOT, ROOT)}, "fold10_waveforms_read": 0},
              open(os.path.join(out, "results.json"), "x"), indent=1)
    say(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
