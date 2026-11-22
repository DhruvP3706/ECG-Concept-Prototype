"""Stages 2-4: SCP form/rhythm concept predictor -> strict concept bottleneck (linear head, anchored-prototype head) on PTB-XL. Validation-stage; fold 10 never read.
Preconditions: the Stage-1 audit passed (audit_results.json all_pass) and the vocabulary file hash matches.
Configurations (all declared here before running):
  main   : audit vocabulary (17 concepts), label policy P1 (absence is NEGATIVE only inside a documented category, otherwise UNKNOWN and masked)       <- primary
  P2     : same vocabulary, absence = negative (sensitivity; an assumption)
  reduced: P1 with the semantic-overlap-watch concepts and ABQRS removed (VCLVH, QWAVE, LPR, STD_, NT_, INVT, LOWT, ABQRS) -> 9 concepts (sensitivity)
Per configuration and seed (42, 43, 44):
  predictor : ConceptMLP 512-128-C on frozen A_exact embeddings, masked BCE (unknown entries excluded), per-concept pos_weight = known negatives / known positives (TRAIN),
              AdamW lr 1e-3, wd 1e-4, batch 256, <=100 epochs, patience 10 on the validation masked weighted BCE; FROZEN (hash asserted) before any diagnosis head trains.
  inputs    : predicted concept PROBABILITIES (sigmoid), standardized with TRAIN mean/std. No true concept, no embedding and no mask reaches a diagnosis head.
  linear    : Linear(C,5).  proto: K=16 anchors = real eligible TRAIN records nearest the deterministic k-means centres in the standardized predicted-concept space
              (farthest-point init), s=exp(-||z-p||^2/tau), tau = median train nearest-prototype squared distance (floor 1e-6), head Linear(16,5) on similarities only.
  controls  : shuffled predicted concepts (permuted across TRAIN records, linear head); annotation-pattern control (Linear on [has any form statement, has any rhythm statement],
              from metadata, no ECG) to measure how much diagnostic information the unknown/known pattern alone carries.
Diagnosis heads: BCE with TRAIN-only pos_weight, AdamW lr 1e-2, wd 1e-4, batch 256, patience 10 on validation weighted BCE.
Note: concept probabilities of TRAIN records are in-sample for the predictor (optimistic) while validation ones are out-of-sample; heads train on the former, as in Step 4.4.
Validation concept targets are used for predictor early stopping and for concept evaluation only. SCP statements are report-derived annotations, not independent ground truth.
Run: .venv\\Scripts\\python.exe -I scripts\\ptbxl_scp_bottleneck.py <results/ptbxl_scp_concepts/run_dir>
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
from ecg_concept import ptbxl as px  # noqa: E402
from ecg_concept import scp_concepts as sc  # noqa: E402
import ptbxl_baseline_train as bt  # noqa: E402
import ptbxl_concept_proto as rr  # noqa: E402  (fit_pw, sha_file reused; its main() is not run)

BASE = rr.BASE
BOOT = rr.BOOT
RRRUN = os.path.join(ROOT, "results", "ptbxl_concepts", "run_20261120_015723", "proto")
DATA = os.path.join(ROOT, "data", "raw", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
SEEDS, VOCAB5, K = (42, 43, 44), bt.VOCAB, 16
LR_C, LR_H, WD, BATCH, MAXEP, PAT = 1e-3, 1e-2, 1e-4, 256, 100, 10
B, BOOT_SEED = 5000, 20261120
DROP_REDUCED = {"VCLVH", "QWAVE", "LPR", "STD_", "NT_", "INVT", "LOWT", "ABQRS"}
sha_file, sig = rr.sha_file, rr.sig


def masked_wbce(logit, y, mask, pw):
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none", pos_weight=pw) * mask
    per = bce.sum(0) / mask.sum(0).clamp(min=1.0)
    has = (mask.sum(0) > 0).float()
    return (per * has).sum() / has.sum().clamp(min=1.0)


def fit_concepts(model, Xtr, Ttr, Xva, Tva, seed, pw):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_C, weight_decay=WD)
    Xtr, Xva = torch.as_tensor(Xtr), torch.as_tensor(Xva)
    ytr, yva = torch.as_tensor(np.clip(Ttr, 0, 1), dtype=torch.float32), torch.as_tensor(np.clip(Tva, 0, 1), dtype=torch.float32)
    mtr, mva = torch.as_tensor(Ttr != -1, dtype=torch.float32), torch.as_tensor(Tva != -1, dtype=torch.float32)
    best, state, bad, hist = float("inf"), None, 0, {"train": [], "val": []}
    for ep in range(MAXEP):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for b0 in range(0, len(perm), BATCH):
            ix = perm[b0:b0 + BATCH]
            opt.zero_grad()
            l = masked_wbce(model(Xtr[ix]), ytr[ix], mtr[ix], pw)
            l.backward()
            opt.step()
            tot += float(l) * len(ix)
        model.eval()
        with torch.no_grad():
            v = float(masked_wbce(model(Xva), yva, mva, pw))
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


def concept_metrics(P, T, vocab, split):
    """Per-concept metrics on KNOWN targets only. P (N,C) probabilities, T (N,C) in {1,0,-1}."""
    rows = []
    for j, c in enumerate(vocab):
        k = T[:, j] != -1
        y, p = T[k, j].astype(bool), P[k, j]
        pos, neg = int(y.sum()), int((~y).sum())
        auc = float(M.auroc_columns(y[:, None], p[:, None])[0]) if pos and neg else float("nan")
        ap = float(M.ap_columns(y[:, None], p[:, None])[0]) if pos else float("nan")
        pr, rc, f1, _ = M.per_class_prf(y[:, None], (p > 0.5)[:, None])
        rows.append({"split": split, "concept": c, "n_pos": pos, "n_known_neg": neg, "n_unknown": int((~k).sum()), "prevalence_known": pos / max(pos + neg, 1), "auroc": auc, "ap": ap,
                     "precision@0.5": float(pr[0]), "recall@0.5": float(rc[0]), "f1@0.5": float(f1[0])})
    return rows


def main():
    run = os.path.abspath(sys.argv[1])
    aud = os.path.join(run, "audit")
    ares = json.load(open(os.path.join(aud, "audit_results.json")))
    assert ares["all_pass"], "audit did not pass; training is not allowed"
    vfile = os.path.join(aud, "concept_vocabulary.json")
    assert sha_file(vfile) == ares["vocabulary_file_sha256"]
    vdoc = json.load(open(vfile))
    vocab_full = vdoc["vocabulary"]
    cat_of = vdoc["category"]
    out = os.path.join(run, "bottleneck")
    ckd = os.path.join(ROOT, "models", "ptbxl_scp_concepts", os.path.basename(run))
    os.makedirs(out)
    os.makedirs(ckd)
    log = open(os.path.join(out, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    ids, emb, man = rr.load()
    records = {r.ecg_id: r for r in px.read_metadata(os.path.join(DATA, "ptbxl_database.csv"))}
    stm = px.read_statements(os.path.join(DATA, "scp_statements.csv"))
    form, rhythm, _ = sc.split_statements(stm)
    members = {"form": form, "rhythm": rhythm}
    recs = [records[i] for i in ids]
    role = np.array([man[i]["role"] for i in ids])
    tr, va = role == "train", role == "val"
    assert (tr.sum(), va.sum()) == (16289, 2034) and all(r.strat_fold <= 9 for r in recs)
    Y = np.array([[int(man[i][c]) for c in VOCAB5] for i in ids], np.float32)
    pos_weight = torch.from_numpy(((tr.sum() - Y[tr].sum(0)) / Y[tr].sum(0)).astype(np.float32))
    pat = np.array([int(man[i]["patient_id"]) for i in ids[va]])
    annot = np.array([[any(c in r.scp_codes for c in form), any(c in r.scp_codes for c in rhythm)] for r in recs], np.float64)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    X = cp.Standardizer.fit(emb[tr]).transform(emb).astype(np.float32)
    configs = {"main": ("P1", vocab_full), "P2": ("P2", vocab_full), "reduced": ("P1", [c for c in vocab_full if c not in DROP_REDUCED])}
    say(f"vocabulary {len(vocab_full)}: {vocab_full}; reduced {len(configs['reduced'][1])}: {configs['reduced'][1]}")

    def run_cfg(name, seed):
        policy, vocab = configs[name]
        T = sc.concept_targets(recs, vocab, cat_of, members, policy)
        Cn = len(vocab)
        kn = (T[tr] != -1)
        npos, nneg = (T[tr] == 1).sum(0), (T[tr] == 0).sum(0)
        cpw = torch.from_numpy((nneg / np.maximum(npos, 1)).astype(np.float32))
        torch.manual_seed(seed)
        mC = cp.ConceptMLP(512, 128, Cn)
        mC, h = fit_concepts(mC, X[tr], T[tr], X[va], T[va], seed, cpw)
        for p in mC.parameters():
            p.requires_grad_(False)
        mC.eval()
        hb = hashlib.sha256(b"".join(v.numpy().tobytes() for v in mC.state_dict().values())).hexdigest()
        Ph = sig(cp.predict(mC, X)).astype(np.float64)                         # predicted concept PROBABILITIES, all eligible rows
        assert np.isfinite(Ph).all()
        cs = cp.Standardizer.fit(Ph[tr])
        Z = cs.transform(Ph).astype(np.float64)
        o = {"T": T, "Ph": Ph, "hist": {"concept_predictor": h}, "probs": {}, "state": {"concept_predictor": mC.state_dict()}, "vocab": vocab, "concept_pw": cpw.tolist()}
        o["concept_eval"] = [{"config": name, "seed": seed, **r} for r in concept_metrics(Ph[tr], T[tr], vocab, "train") + concept_metrics(Ph[va], T[va], vocab, "validation")]
        T2 = (T == 1).astype(np.int8)                                           # presence-vs-all-absent view (P2 semantics), supplementary
        o["concept_eval_absent_is_negative"] = [{"config": name, "seed": seed, **r} for r in concept_metrics(Ph[va], T2[va], vocab, "validation")]
        torch.manual_seed(seed)
        m_lin, h = rr.fit_pw(nn.Linear(Cn, 5), Z[tr], Y[tr], Z[va], Y[va], seed, LR_H, pos_weight)
        o["probs"]["linear"] = sig(cp.predict(m_lin, Z[va])).astype(np.float64)
        o["hist"]["linear"], o["state"]["linear"] = h, m_lin.state_dict()
        centres, assign = cp.kmeans_det(Z[tr], K)
        tr_idx = np.where(tr)[0]
        anchor = tr_idx[cp.anchor_records(Z[tr], centres)]
        assert tr[anchor].all() and not va[anchor].any()
        P = Z[anchor]
        tau = cp.tau_rule(Z[tr], P)
        S = cp.similarities(Z, P, tau).astype(np.float32)
        torch.manual_seed(seed)
        m_pro, h = rr.fit_pw(cp.ProtoHead(K, 5), S[tr], Y[tr], S[va], Y[va], seed, LR_H, pos_weight)
        assert hashlib.sha256(b"".join(v.numpy().tobytes() for v in mC.state_dict().values())).hexdigest() == hb, "concept predictor changed"
        o["probs"]["proto"] = sig(cp.predict(m_pro, S[va])).astype(np.float64)
        o["hist"]["proto"], o["state"]["proto"] = h, m_pro.state_dict()
        o["proto"] = {"anchor_ecg_id": ids[anchor].tolist(), "tau": tau, "cluster_sizes": np.bincount(assign, minlength=K).tolist(), "head_weight": m_pro.fc.weight.detach().numpy().tolist(),
                      "anchor_true_concepts": {vocab[j]: int((T[anchor, j] == 1).sum()) for j in range(Cn)}, "anchor_unknown_concept_entries": int((T[anchor] == -1).sum()),
                      "predictor_sha": hb, "anchor_diag_labels": {VOCAB5[k]: int(Y[anchor, k].sum()) for k in range(5)}}
        perm = np.random.RandomState(seed).permutation(int(tr.sum()))
        torch.manual_seed(seed)
        m_shuf, h = rr.fit_pw(nn.Linear(Cn, 5), Z[tr][perm], Y[tr], Z[va], Y[va], seed, LR_H, pos_weight)
        o["probs"]["shuffled"] = sig(cp.predict(m_shuf, Z[va])).astype(np.float64)
        o["hist"]["shuffled"], o["state"]["shuffled"] = h, m_shuf.state_dict()
        if name == "main":
            Az = cp.Standardizer.fit(annot[tr]).transform(annot).astype(np.float32)
            torch.manual_seed(seed)
            m_an, h = rr.fit_pw(nn.Linear(2, 5), Az[tr], Y[tr], Az[va], Y[va], seed, LR_H, pos_weight)
            o["probs"]["annotation_pattern"] = sig(cp.predict(m_an, Az[va])).astype(np.float64)
            o["hist"]["annotation_pattern"], o["state"]["annotation_pattern"] = h, m_an.state_dict()
        return o

    t0 = time.time()
    R = {n: {} for n in configs}
    for n in configs:
        for s in SEEDS:
            R[n][s] = run_cfg(n, s)
            eps = {k: v["epochs_run"] for k, v in R[n][s]["hist"].items()}
            say(f"{n} seed {s} done ({time.time() - t0:.0f}s) epochs {eps} tau {R[n][s]['proto']['tau']:.3f}")
    rep = run_cfg("main", 42)
    repro = {k: float(np.abs(rep["probs"][k] - R["main"][42]["probs"][k]).max()) for k in rep["probs"]}
    repro["concept_probs"] = float(np.abs(rep["Ph"] - R["main"][42]["Ph"]).max())
    say(f"reproducibility (main, seed 42 rerun) max abs diff: {repro}")

    # ---- diagnosis metrics + concept tables
    Yva = Y[va].astype(bool)
    res = {}
    for n in configs:
        for m in R[n][42]["probs"]:
            for s in SEEDS:
                res[f"{n}|{m}|{s}"] = bt.evaluate(Yva, R[n][s]["probs"][m])
    with open(os.path.join(out, "concept_prediction_metrics.csv"), "x", newline="") as f:
        rows = [r for n in configs for s in SEEDS for r in R[n][s]["concept_eval"]]
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "concept_prediction_metrics_absent_is_negative.csv"), "x", newline="") as f:
        rows = [r for n in configs for s in SEEDS for r in R[n][s]["concept_eval_absent_is_negative"]]
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- paired bootstrap with the SAME patient draws as the earlier experiments
    pred = np.load(os.path.join(BOOT, "predictions.npz"))
    rrp = np.load(os.path.join(RRRUN, "predictions_fold9.npz"))
    assert (pred["ecg_id"] == ids[va]).all() and (rrp["ecg_id"] == ids[va]).all() and (pred["y"] == Y[va]).all()
    dz = np.load(os.path.join(BOOT, "bootstrap_patient_draws.npz"))
    draws, upat = dz["draws"], dz["unique_patient_ids"]
    assert int(dz["seed"]) == BOOT_SEED and draws.shape[0] == B and (np.unique(pat) == upat).all() and (pred["patient_id"] == pat).all()
    allm = {m: {s: pred[f"{m}_seed{s}"] for s in SEEDS} for m in ("frozen_linear", "frozen_mlp", "finetuned")}
    allm["rr_concept_linear"] = {s: rrp[f"C2_concept_only_seed{s}"] for s in SEEDS}
    allm["rr_prototype"] = {s: rrp[f"D_prototype_seed{s}"] for s in SEEDS}
    for n in configs:
        for m in R[n][42]["probs"]:
            allm[f"scp_{n}_{m}"] = {s: R[n][s]["probs"][m] for s in SEEDS}
    _, inv = np.unique(pat, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(upat)))
    lens = np.bincount(inv, minlength=len(upat))
    names = list(allm)
    keys = [(m, s) for m in names for s in SEEDS]
    HYP = VOCAB5.index("HYP")
    auc_b, ap_b = np.zeros((B, len(keys), 5)), np.zeros((B, len(keys), 5))
    t1 = time.time()
    for b in range(B):
        idx = order[np.concatenate([np.arange(starts[k], starts[k] + lens[k]) for k in draws[b]])]
        yb = Yva[idx]
        for j, (m, s) in enumerate(keys):
            auc_b[b, j] = M.auroc_columns(yb, allm[m][s][idx])
            ap_b[b, j] = M.ap_columns(yb, allm[m][s][idx])
    say(f"bootstrap {B} replicates in {time.time() - t1:.0f}s; undefined-class replicates: AUROC {int(np.isnan(auc_b).any(axis=(1, 2)).sum())}, AP {int(np.isnan(ap_b).any(axis=(1, 2)).sum())}")
    stat, pt = {}, {}
    for m in names:
        js = [keys.index((m, s)) for s in SEEDS]
        stat[(m, "macro_auroc")] = np.nanmean(auc_b[:, js], axis=2).mean(1)
        stat[(m, "macro_ap")] = np.nanmean(ap_b[:, js], axis=2).mean(1)
        stat[(m, "hyp_auroc")] = auc_b[:, js, HYP].mean(1)
        pt[(m, "macro_auroc")] = float(np.mean([np.nanmean(M.auroc_columns(Yva, allm[m][s])) for s in SEEDS]))
        pt[(m, "macro_ap")] = float(np.mean([np.nanmean(M.ap_columns(Yva, allm[m][s])) for s in SEEDS]))
        pt[(m, "hyp_auroc")] = float(np.mean([M.auroc_columns(Yva, allm[m][s])[HYP] for s in SEEDS]))
    METR = ("macro_auroc", "macro_ap", "hyp_auroc")
    ci = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])] + [int(np.isnan(v).sum())]
    comps = [("scp_main_linear", "frozen_mlp"), ("scp_main_linear", "frozen_linear"), ("scp_main_proto", "frozen_mlp"), ("scp_main_proto", "frozen_linear"),
             ("scp_main_proto", "scp_main_linear"), ("scp_main_linear", "rr_concept_linear"), ("scp_main_proto", "rr_prototype"), ("scp_main_linear", "scp_main_shuffled"),
             ("scp_main_proto", "scp_main_shuffled"), ("scp_main_linear", "scp_main_annotation_pattern"), ("scp_main_proto", "scp_main_annotation_pattern"),
             ("scp_P2_linear", "scp_main_linear"), ("scp_P2_proto", "scp_main_proto"), ("scp_P2_linear", "frozen_mlp"), ("scp_P2_proto", "frozen_mlp"), ("scp_P2_proto", "scp_P2_linear"),
             ("scp_reduced_linear", "scp_main_linear"), ("scp_reduced_proto", "scp_main_proto"), ("scp_reduced_linear", "frozen_mlp")]
    brow = []
    for m in names:
        for k in METR:
            lo, hi, nb = ci(stat[(m, k)])
            brow.append(["model", m, k, pt[(m, k)], lo, hi, nb])
    for a_, b_ in comps:
        for k in METR:
            lo, hi, nb = ci(stat[(a_, k)] - stat[(b_, k)])
            brow.append(["difference", f"{a_} - {b_}", k, pt[(a_, k)] - pt[(b_, k)], lo, hi, nb])
    with open(os.path.join(out, "bootstrap_summary.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "model_or_comparison", "metric", "fold9_point_estimate_seed_mean", "ci95_low", "ci95_high", "invalid_replicates"])
        w.writerows(brow)
    np.savez(os.path.join(out, "predictions_fold9.npz"), ecg_id=ids[va], patient_id=pat, y=Y[va], **{f"{n}_{m}_seed{s}": R[n][s]["probs"][m] for n in configs for m in R[n][42]["probs"] for s in SEEDS})
    np.savez(os.path.join(out, "predicted_concept_probabilities.npz"), ecg_id=ids, **{f"{n}_seed{s}": R[n][s]["Ph"] for n in configs for s in SEEDS},
             **{f"{n}_targets_seed42": R[n][42]["T"] for n in configs})
    for n in configs:
        for s in SEEDS:
            torch.save({"state": R[n][s]["state"], "proto": R[n][s]["proto"], "vocab": R[n][s]["vocab"], "seed": s}, os.path.join(ckd, f"{n}_seed{s}.pt"))
    json.dump({"results": res, "vocab": {n: configs[n][1] for n in configs}, "policy": {n: configs[n][0] for n in configs}, "repro": repro, "seeds": SEEDS,
               "hist": {f"{n}|{s}": {k: {x: v[x] for x in ("best_epoch", "epochs_run", "best_val")} for k, v in R[n][s]["hist"].items()} for n in configs for s in SEEDS},
               "proto": {f"{n}|{s}": R[n][s]["proto"] for n in configs for s in SEEDS}, "pos_weight_diagnosis": pos_weight.tolist(),
               "concept_pos_weight": {f"{n}|{s}": R[n][s]["concept_pw"] for n in configs for s in SEEDS},
               "hyper": {"lr_concept": LR_C, "lr_heads": LR_H, "weight_decay": WD, "batch": BATCH, "max_epochs": MAXEP, "patience": PAT, "K": K, "bootstrap_B": B, "bootstrap_seed": BOOT_SEED,
                         "bootstrap_draws_sha256": sha_file(os.path.join(BOOT, "bootstrap_patient_draws.npz"))},
               "inputs": {"audit_results_sha256": sha_file(os.path.join(aud, "audit_results.json")), "concept_vocabulary_sha256": ares["vocabulary_file_sha256"],
                          "manifest_sha256": ares["inputs"]["manifest_sha256"], "scp_statements_sha256": ares["inputs"]["scp_statements_sha256"],
                          "baseline_run": os.path.relpath(BASE, ROOT), "bootstrap_run": os.path.relpath(BOOT, ROOT), "rr_run": os.path.relpath(RRRUN, ROOT)},
               "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("scripts/ptbxl_scp_bottleneck.py", "src/ecg_concept/scp_concepts.py", "src/ecg_concept/concept_proto.py")},
               "fold10_waveforms_read": 0}, open(os.path.join(out, "results.json"), "x"), indent=1)
    say(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
