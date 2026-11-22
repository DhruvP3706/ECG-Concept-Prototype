r"""Experiment 7: graded RR-variability interventions on sinus-rhythm (SR) and atrial-fibrillation (AFIB) records of the frozen VALIDATION role (frozen original LightECGNetV2; counterfactual sensitivity).
Populations (dataset annotations from the frozen role manifest, never model predictions): SR group = label SR present and no other rhythm/ectopy label; AFIB group = label AFIB present and no other
rhythm/ectopy label (excluded labels: AF (atrial flutter), AT, SVT, SB, ST, SA, AVRT, AVNRT, SAAWR, APB, VPB, VEB, JEB, VPE, VFW, ERV). Beats: frozen Step 4.3 extractor (automatic, not expert-validated).
Interventions (src/ecg_concept/rr_warp.py): first/last beat fixed -> beat count, order, mean RR and total duration preserved; QRS windows, 150 samples of ST-T and 50 samples of P region copied rigidly where they fit.
Targets: RR sample-CV levels derived from the observed SR/AFIB distributions (see `population`); SR records only get levels ABOVE their own CV (+0.01), AFIB records only levels BELOW (-0.01); every record also gets the
'own' level (CV unchanged, sequence replaced = control for the intervention itself). Four RR patterns per level (iid_a, iid_b, smooth AR(1), perm_orig = the record's own RR deviations reordered).
Acceptance = the Experiment 6 morphology/discontinuity rules + achieved CV within max(0.005, 5%) of the target. Rejected transformations are kept in the item table with their reasons.
Rigid P/ST-T copying does NOT reproduce physiological QT/RR coupling. Stages:  population | run <dir> | analyze <dir>
Usage: .venv\Scripts\python.exe -I scripts\xai_graded_rr.py population   (creates results/xai_graded_rr/run_<stamp>)   then   ... run <dir>   then   ... analyze <dir>
"""
import csv
import datetime
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ecg_concept import rr_warp as rw  # noqa: E402
import graded_rr_core as core  # noqa: E402
from graded_rr_core import ACC, MARGIN, PATTERNS, PNAMES, RIGID, SEED  # noqa: E402

xf = xb = None                                                                  # heavy modules (torch) are imported lazily, in the main process only


def _heavy():
    global xf, xb
    if xf is None:
        import xai_beat_attribution as _xb
        import xai_ig_faithfulness as _xf
        xb, xf = _xb, _xf

EXCLUDE = {"AF", "AT", "SVT", "SB", "ST", "SA", "AVRT", "AVNRT", "SAAWR", "APB", "VPB", "VEB", "JEB", "VPE", "VFW", "ERV"}
FOCUS = {"AFIB": "AFIB", "SR": "SR", "SB": "SB", "ST": "ST", "AF": "AF", "SA": "SA", "AT": "AT", "SVT": "SVT", "STTC": "STTC", "TWC": "TWC", "TWO": "TWO"}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def population():
    _heavy()
    params = xb.verify_extractor()
    run = os.path.join(ROOT, "results", "xai_graded_rr", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run)
    log = open(os.path.join(run, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    man = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", "step3_7", "prospective_role_manifest.csv"))) if r["role"] == "validation"]
    assert len(man) == 4437
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(xf.CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    groups = {"SR": [], "AFIB": []}
    excl = {"SR": {}, "AFIB": {}}
    for r in man:
        L = set(r["labels"].split("|"))
        for g in groups:
            if g in L:
                bad = sorted((L & EXCLUDE) | ({"AFIB"} if g == "SR" and "AFIB" in L else set()) | ({"SR"} if g == "AFIB" and "SR" in L else set()))
                if bad:
                    for b_ in bad:
                        excl[g][b_] = excl[g].get(b_, 0) + 1
                else:
                    groups[g].append(r["record_id"])
    n_ann = {g: sum(1 for r in man if g in r["labels"].split("|")) for g in groups}
    say(f"annotated validation records: SR {n_ann['SR']}, AFIB {n_ann['AFIB']}; after exclusions: SR {len(groups['SR'])}, AFIB {len(groups['AFIB'])}; exclusion reasons {excl}")
    ids = sorted(groups["SR"] + groups["AFIB"])
    t0 = time.time()
    with ProcessPoolExecutor(8) as ex:
        out = list(ex.map(core._extract, [(i, os.path.join(xf.DS, rel[i])) for i in ids], chunksize=8))
    say(f"extractor on {len(ids)} records in {time.time() - t0:.0f}s")
    info = {}
    for rid, cons, state, reasons in out:
        g = "SR" if rid in set(groups["SR"]) else "AFIB"
        b = np.array(cons, int)
        usable = len(b) >= 4 and state != "UNAVAILABLE"                      # >= 3 intervals needed to construct a CV-targeted sequence
        info[rid] = {"group": g, "beats": cons, "state": state, "reasons": reasons, "usable": bool(usable), "cv": rw.rr_cv(b) if len(b) >= 3 else None, "mean_rr": float(np.diff(b).mean()) if len(b) >= 2 else None}
    z = np.load(os.path.join(xf.CACHE, "merged.npz"))
    thr = np.load(xf.THR, allow_pickle=False)
    names = [str(c) for c in z["class_names"]]
    pos = {rid: i for i, rid in enumerate(z["ids"])}
    lg = {rid: z["orig_logits"][pos[rid]].astype(float) for rid in ids}
    pred = {rid: (1 / (1 + np.exp(-lg[rid])) > thr) for rid in ids}
    pop = {"n_annotated": n_ann, "exclusion_counts": excl, "groups": {}}
    cvs = {}
    for g in groups:
        recs = groups[g]
        us = [r for r in recs if info[r]["usable"]]
        cv = np.array([info[r]["cv"] for r in us])
        cvs[g] = cv
        own = names.index(g)
        pop["groups"][g] = {"n_records": len(recs), "n_usable_beats": len(us), "states": {s_: sum(info[r]["state"] == s_ for r in recs) for s_ in ("ACCEPTED", "UNCERTAIN", "UNAVAILABLE")},
                            "cv_quantiles_5_25_50_75_90_95": np.percentile(cv, [5, 25, 50, 75, 90, 95]).round(4).tolist(), "cv_mean": float(cv.mean()), "mean_rr_ms_median": float(np.median([info[r]["mean_rr"] * 2 for r in us])),
                            "original_prediction_of_own_label": {"predicted_positive": int(sum(pred[r][own] for r in us)), "of": len(us), "rate": float(np.mean([pred[r][own] for r in us]))},
                            "original_predicted_other_group_label_positive": int(sum(pred[r][names.index("AFIB" if g == "SR" else "SR")] for r in us)),
                            "mean_logit_AFIB": float(np.mean([lg[r][names.index("AFIB")] for r in us])), "mean_logit_SR": float(np.mean([lg[r][names.index("SR")] for r in us]))}
    if min(len(cvs["SR"]), len(cvs["AFIB"])) < 50:
        say("STOP: fewer than 50 usable records in a group; the comparison is not supported")
        json.dump(pop, open(os.path.join(run, "population.json"), "x"), indent=1, default=float)
        sys.exit("insufficient population")
    q = lambda g, p: float(np.percentile(cvs[g], p))
    levels = {"L1_low(SR median)": q("SR", 50), "L2_intermediate(mid of SR p90 and AFIB p10)": 0.5 * (q("SR", 90) + q("AFIB", 10)), "L3_high(AFIB median)": q("AFIB", 50), "L4_higher(AFIB p75)": q("AFIB", 75)}
    levels = {k: round(v, 3) for k, v in levels.items()}
    pop["levels"] = levels
    pop["level_rule"] = "L1 = SR median CV, L2 = mean of SR 90th and AFIB 10th percentile, L3 = AFIB median CV, L4 = AFIB 75th percentile (usable records, original CV)"
    pop["overlap"] = {"share_SR_records_with_cv_above_AFIB_p10": float((cvs["SR"] > q("AFIB", 10)).mean()), "share_AFIB_records_with_cv_below_SR_p90": float((cvs["AFIB"] < q("SR", 90)).mean())}
    json.dump(pop, open(os.path.join(run, "population.json"), "x"), indent=1, default=float)
    json.dump(info, open(os.path.join(run, "records_beats.json"), "x"))
    json.dump({"extractor_params_sha256": params, "seed": SEED, "acceptance": ACC, "rigid": RIGID, "patterns": PNAMES, "margin": MARGIN, "checkpoint_sha256": xf.CKPT_SHA,
               "manifest_sha256": sha_file(os.path.join(ROOT, "results", "step3_7", "prospective_role_manifest.csv"))}, open(os.path.join(run, "config.json"), "x"), indent=1)
    say(f"levels {levels}")
    say(json.dumps(pop["groups"], default=float))
    say(f"overlap {pop['overlap']}")
    print(run)


def run(run_dir):
    _heavy()
    import torch
    from ecg_concept.inference import load_model
    info = json.load(open(os.path.join(run_dir, "records_beats.json")))
    pop = json.load(open(os.path.join(run_dir, "population.json")))
    levels = pop["levels"]
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(xf.CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    use = sorted(r for r, v in info.items() if v["usable"])
    keep = set()
    for g in ("SR", "AFIB"):
        gr = [r for r in use if info[r]["group"] == g]
        med = np.median([info[r]["cv"] for r in gr])
        keep |= set(sorted(gr, key=lambda r: abs(info[r]["cv"] - med))[:2])
    tasks = [(r, info[r]["group"], os.path.join(xf.DS, rel[r]), info[r]["beats"], info[r]["cv"], i, levels, r in keep) for i, r in enumerate(use)]
    log = open(os.path.join(run_dir, "run_log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    assert sha_file(xf.CKPT) == xf.CKPT_SHA
    dev = "cuda"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = load_model(xf.CKPT, dev)
    h0 = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    z = np.load(os.path.join(xf.CACHE, "merged.npz"))
    pos = {rid: i for i, rid in enumerate(z["ids"])}
    items, logits, examples = [], [], {}
    t0 = time.time()
    buf_x, buf_idx = [], []

    def flush():
        if buf_x:
            with torch.no_grad():
                lg = model(torch.from_numpy(np.stack(buf_x)).to(dev)).cpu().numpy().astype(np.float32)
            for k, l in zip(buf_idx, lg):
                items[k]["logit_row"] = len(logits)
                logits.append(l)
            buf_x.clear()
            buf_idx.clear()

    with ProcessPoolExecutor(8) as ex:
        for n_done, res in enumerate(ex.map(core._work, tasks, chunksize=2)):
            for it in res:
                x = it.pop("x", None)
                for k in ("raw_warped", "raw_orig"):
                    if k in it:
                        examples[(it["record_id"], it["level"], it["pattern"], k)] = it.pop(k)
                items.append(it)
                if x is not None:
                    buf_x.append(x)
                    buf_idx.append(len(items) - 1)
            if len(buf_x) >= 64:
                flush()
            if n_done % 100 == 0:
                say(f"records {n_done}/{len(tasks)} ({time.time() - t0:.0f}s); items {len(items)}")
        flush()
    assert h0 == hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest() and sha_file(xf.CKPT) == xf.CKPT_SHA
    L = np.stack(logits)
    # reproduction check of the untransformed records against the stored Step 4.4 logits
    d = [np.abs(L[it["logit_row"]] - z["orig_logits"][pos[it["record_id"]]]).max() for it in items if it["level"] == "orig"]
    say(f"{len(items)} items, {len(L)} model evaluations in {time.time() - t0:.0f}s; 'orig' reproduces stored logits (max abs diff {max(d):.1e}); model hash unchanged")
    assert max(d) < 1e-3
    np.save(os.path.join(run_dir, "logits.npy"), L)
    keys = sorted({k for it in items for k in it if k not in ("new_beats",)})
    with open(os.path.join(run_dir, "items.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: it.get(k, "") for k in keys} for it in items])
    json.dump({f"{it['record_id']}|{it['level']}|{it['pattern']}": it["new_beats"] for it in items}, open(os.path.join(run_dir, "all_beat_sequences.json"), "x"))
    np.savez_compressed(os.path.join(run_dir, "example_signals.npz"), **{"|".join(k): v for k, v in examples.items()})
    say("saved items.csv, logits.npy, all_beat_sequences.json, example_signals.npz")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "population":
        population()
    elif len(sys.argv) > 2 and sys.argv[1] == "run":
        run(os.path.abspath(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == "analyze":
        import xai_graded_rr_analysis as an
        an.analyze(os.path.abspath(sys.argv[2]))
    else:
        sys.exit(__doc__)
