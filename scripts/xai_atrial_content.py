r"""Experiment 9: atrial-CONTENT substitution and re-arrangement (frozen original LightECGNetV2; counterfactual sensitivity, no clinical claim).
Stages:  population    -> Stage-1 feasibility audit + donor bank + deterministic donor selection + run directory results/xai_atrial_content/run_<stamp>
         run <dir>     -> all conditions per recipient (torch-free workers), quality control, frozen-model evaluation
         analyze <dir> -> paired statistics, figures (xai_atrial_content_analysis.py)
Population, beats, atrial-region convention (r-160..r-51, eligibility rule, >=4 eligible beats) and preprocessing are those of Experiment 8 (read-only copies).
Usage: .venv\Scripts\python.exe -I scripts\xai_atrial_content.py population | run <dir> | analyze <dir>"""
import csv
import datetime
import hashlib
import json
import math
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import atrial_core as c0  # noqa: E402
import content_core as cc  # noqa: E402
from ecg_concept import atrial_content as ac  # noqa: E402
from ecg_concept import atrial_ops as ao  # noqa: E402
from ecg_concept.preprocessing import read_raw_mat  # noqa: E402

EXP8 = os.path.join(ROOT, "results", "xai_atrial_activity", "run_20261120_152906")
CKPT = os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")
CKPT_SHA = "7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72"
CACHE = os.path.join(ROOT, "results", "step4_4", "cache")
DS = c0.DS
N_AF, N_SR, N_RND = 5, 3, 3
LAMBDA = 0.02                                                                  # usage penalty per previous use of a donor (spreads the load over the pool)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def population():
    assert sha_file(CKPT) == CKPT_SHA
    run = os.path.join(ROOT, "results", "xai_atrial_content", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(run, "bank"))
    log = open(os.path.join(run, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    cfg8 = json.load(open(os.path.join(EXP8, "config.json")))
    shutil.copy(os.path.join(EXP8, "records_beats.json"), os.path.join(run, "records_beats.json"))
    assert sha_file(os.path.join(run, "records_beats.json")) == cfg8["records_beats_sha256"]
    info = json.load(open(os.path.join(run, "records_beats.json")))
    items8 = list(csv.DictReader(open(os.path.join(EXP8, "items.csv"))))
    flat8 = {it["record_id"]: it["accepted"] == "True" for it in items8 if it["condition"] == "flat"}
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    use = sorted(r for r, v in info.items() if v["usable"])
    pool = {"SR": [], "AFIB": []}
    cov = {}
    arrs = {a: [] for a in (120, 160, 200)}
    idx = {str(a): {} for a in arrs}
    idx["raw"] = {}
    counts = {a: 0 for a in arrs}
    raws, rcount = [], 0
    for r in use:
        g = info[r]["group"]
        b = np.array(info[r]["beats"], int)
        regs160 = ao.regions(b, 160)[0]
        if len(regs160) < c0.MIN_ELIG:
            continue
        x = read_raw_mat(os.path.join(DS, rel[r]))
        pool[g].append(r)
        devs160 = ac.region_devs(x, regs160)
        cov[r] = {"mean_rr": float(np.diff(b).mean()), "rms": ac.rms_lead(devs160)}
        for a1 in arrs:
            regs = ao.regions(b, a1)[0]
            d = np.stack(ac.region_devs(x, regs)).astype(np.float32) if regs else np.zeros((0, 12, a1 - 50), np.float32)
            idx[str(a1)][r] = [counts[a1], counts[a1] + len(d)]
            counts[a1] += len(d)
            arrs[a1].append(d)
        if g == "AFIB":
            rs = np.stack([x[:, a:e] for a, e in regs160]).astype(np.int16)
            idx["raw"][r] = [rcount, rcount + len(rs)]
            rcount += len(rs)
            raws.append(rs)
    for a1 in arrs:
        np.save(os.path.join(run, "bank", f"dev_{a1}.npy"), np.concatenate(arrs[a1]))
    np.save(os.path.join(run, "bank", "raw_160.npy"), np.concatenate(raws))
    json.dump(idx, open(os.path.join(run, "bank", "index.json"), "x"))
    say(f"pools (>=4 eligible beats at 160): SR {len(pool['SR'])}, AFIB {len(pool['AFIB'])}; bank regions {counts}")

    # ---------------- deterministic donor selection (no model output used)
    def dist(r, ds):
        rr = np.array([abs(math.log(cov[d]["mean_rr"] / cov[r]["mean_rr"])) for d in ds])
        am = np.array([np.median(np.abs(np.log((cov[d]["rms"] + 1e-6) / (cov[r]["rms"] + 1e-6)))) for d in ds])
        return rr + am, rr, am

    recip = sorted(pool["SR"] + pool["AFIB"], key=lambda r: zlib.crc32((r + str(c0.SEED)).encode()))
    cap = {"AF": int(math.ceil(1.5 * N_AF * len(recip) / len(pool["AFIB"]))), "SR": int(math.ceil(1.5 * N_SR * len(recip) / len(pool["SR"])))}
    usage = {"AF": {d: 0 for d in pool["AFIB"]}, "SR": {d: 0 for d in pool["SR"]}}
    donors, rows = {}, []
    for r in recip:
        donors[r] = {}
        for cls, src, n in (("AF", pool["AFIB"], N_AF), ("SR", pool["SR"], N_SR)):
            cand = [d for d in src if d != r and usage[cls][d] < cap[cls]]
            tot, rr, am = dist(r, cand)
            score = tot + LAMBDA * np.array([usage[cls][d] for d in cand])
            order = np.argsort(score, kind="stable")[:n]
            donors[r][cls] = []
            for rank, i in enumerate(order):
                usage[cls][cand[i]] += 1
                donors[r][cls].append([cand[i], float(tot[i])])
                rows.append([r, info[r]["group"], cls, rank, cand[i], float(tot[i]), float(rr[i]), float(am[i]), usage[cls][cand[i]]])
        rg = np.random.RandomState(zlib.crc32(("rnd" + r + str(c0.SEED)).encode()))
        cand = [d for d in pool["AFIB"] if d != r]
        pick = rg.choice(len(cand), N_RND, replace=False)
        tot, _, _ = dist(r, [cand[i] for i in pick])
        donors[r]["RND"] = [[cand[i], float(t)] for i, t in zip(pick, tot)]
        for rank, (i, t) in enumerate(zip(pick, tot)):
            rows.append([r, info[r]["group"], "RND", rank, cand[i], float(t), "", "", ""])
    json.dump(donors, open(os.path.join(run, "donors.json"), "x"))
    with open(os.path.join(run, "donor_selection.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recipient", "recipient_group", "donor_class", "rank", "donor", "distance", "rr_term", "amp_term", "usage_after"])
        w.writerows(rows)

    # ---------------- Stage-1 feasibility numbers
    use_af = np.array(list(usage["AF"].values()))
    use_sr = np.array(list(usage["SR"].values()))
    dist_af = np.array([v["AF"][0][1] for v in donors.values()])
    ratio = []
    for r in recip:
        if info[r]["group"] != "SR":
            continue
        d0 = donors[r]["AF"][0][0]
        ratio.append(float(np.median(cov[d0]["rms"] / (cov[r]["rms"] + 1e-6))))
    sr_dev = [np.stack(ac.region_devs(read_raw_mat(os.path.join(DS, rel[r])), ao.regions(np.array(info[r]["beats"], int), 160)[0])) for r in pool["SR"][:60]]
    af_dev = [np.stack(ac.region_devs(read_raw_mat(os.path.join(DS, rel[r])), ao.regions(np.array(info[r]["beats"], int), 160)[0])) for r in pool["AFIB"][:60]]
    bf = lambda L: np.mean([ac.band_fractions(list(a)) for a in L], axis=0).round(3).tolist()
    jump = []
    for r in pool["SR"][:100]:
        x = read_raw_mat(os.path.join(DS, rel[r]))
        regs = ao.regions(np.array(info[r]["beats"], int), 160)[0]
        rs = np.asarray(BANK_raw(run, donors[r]["AF"][0][0]))
        jump.append(float(np.median([np.abs(rs[i % len(rs)][:, 0].astype(float) - x[:, a].astype(float)).mean() for i, (a, e) in enumerate(regs)])))
    audit = {"recipient_pools": {g: len(v) for g, v in pool.items()}, "donor_pool_sources": "validation role only (same population as Experiment 8); donors are other records than the recipient; AFIB donors never overlap with SR recipients",
             "exp8_flat_accepted": {g: int(sum(flat8.get(r, False) for r in pool[g])) for g in pool},
             "donor_usage_AF": {"n_donors": len(use_af), "min": int(use_af.min()), "median": float(np.median(use_af)), "max": int(use_af.max()), "cap": cap["AF"], "donors_used": int((use_af > 0).sum())},
             "donor_usage_SR": {"n_donors": len(use_sr), "min": int(use_sr.min()), "median": float(np.median(use_sr)), "max": int(use_sr.max()), "cap": cap["SR"], "donors_used": int((use_sr > 0).sum())},
             "matched_AF_donor_distance": {"median": float(np.median(dist_af)), "p5": float(np.percentile(dist_af, 5)), "p95": float(np.percentile(dist_af, 95))},
             "SR_recipient_vs_first_AF_donor_region_rms_ratio": {"median": float(np.median(ratio)), "p5": float(np.percentile(ratio, 5)), "p95": float(np.percentile(ratio, 95))},
             "mean_band_energy_fractions_[0-4,4-9,9-20,20-250]Hz": {"SR_recipient_deviation": bf(sr_dev), "AFIB_donor_deviation": bf(af_dev)},
             "raw_substitution_median_end_step_counts_100_SR_recipients": {"median": float(np.median(jump)), "p90": float(np.percentile(jump, 90)), "note": "mean over leads of |donor first sample - recipient first sample|; deviation-based substitution has exactly 0 by construction"},
             "verdict": ("Substitution of donor DEVIATIONS (x - line through region ends) onto the recipient's own baseline line is feasible and defensible: region length identical, region end values unchanged (no step), QRS/RR/non-target samples untouched, "
                         "per-lead scales comparable (see ratio). Raw-sample substitution changes the baseline and creates boundary steps and is run only as a documented variant. The donor window is positional (r-160..r-51), not a verified fibrillatory-wave segment; "
                         "donor segments are called 'AFIB-donor atrial-region content'. Time reversal / shuffling are not physiologically realistic signals.")}
    json.dump(audit, open(os.path.join(run, "stage1_audit.json"), "x"), indent=1, default=float)
    json.dump({"seed": c0.SEED, "checkpoint_sha256": CKPT_SHA, "exp8_run": os.path.relpath(EXP8, ROOT), "region_convention": "r-160..r-51, eligibility: region start >=220 samples after previous R, >=4 eligible beats (Experiment 8)",
               "n_donors": {"AF": N_AF, "SR": N_SR, "RND": N_RND}, "usage_lambda": LAMBDA, "usage_cap": cap, "scale_compat": cc.SCALE_COMPAT, "shuffle_blocks": 5, "sham": "common-phase amplitude-spectrum surrogate of the real modification, per-lead per-region RMS matched, ends zero",
               "rms_min": c0.RMS_MIN, "diff_ratio_max": c0.DIFF_RATIO_MAX, "sham_ratio": c0.SHAM_RATIO, "timing_tol": c0.TIMING_TOL, "factorial_cv": c0.FACTORIAL_CV, "bootstrap": "record-level B=5000 seed 20261120",
               "records_beats_sha256": sha_file(os.path.join(run, "records_beats.json")), "donors_sha256": sha_file(os.path.join(run, "donors.json")), "audit_sha256": sha_file(os.path.join(run, "stage1_audit.json")),
               "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("src/ecg_concept/atrial_content.py", "scripts/content_core.py", "scripts/xai_atrial_content.py", "src/ecg_concept/atrial_ops.py", "scripts/atrial_core.py")}}, open(os.path.join(run, "config.json"), "x"), indent=1)
    say(json.dumps({k: v for k, v in audit.items() if k != "verdict"}, default=float)[:3500])
    print(run)


_RAW = {}


def BANK_raw(run, did):
    if "b" not in _RAW:
        _RAW["b"] = ac.Bank(os.path.join(run, "bank"))
    return _RAW["b"].raw160(did)


def run(run_dir):
    import torch
    from ecg_concept.inference import load_model
    info = json.load(open(os.path.join(run_dir, "records_beats.json")))
    donors = json.load(open(os.path.join(run_dir, "donors.json")))
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    use = sorted(r for r, v in info.items() if v["usable"])
    keep = set()
    for g in ("SR", "AFIB"):
        keep |= set([r for r in use if info[r]["group"] == g and r in donors and len(ao.regions(np.array(info[r]["beats"], int), 160)[0]) >= 8][:2])
    tasks = [(r, info[r]["group"], os.path.join(DS, rel[r]), info[r]["beats"], info[r]["cv"], i, donors.get(r, {}), r in keep) for i, r in enumerate(use)]
    log = open(os.path.join(run_dir, "run_log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    assert sha_file(CKPT) == CKPT_SHA
    dev = "cuda"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = load_model(CKPT, dev)
    h0 = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    z = np.load(os.path.join(CACHE, "merged.npz"))
    pos = {rid: i for i, rid in enumerate(z["ids"])}
    items, logits, examples = [], [], {}
    buf_x, buf_idx = [], []
    t0 = time.time()

    def flush():
        if buf_x:
            with torch.no_grad():
                lg = model(torch.from_numpy(np.stack(buf_x)).to(dev)).cpu().numpy().astype(np.float32)
            for k, l in zip(buf_idx, lg):
                items[k]["logit_row"] = len(logits)
                logits.append(l)
            buf_x.clear()
            buf_idx.clear()

    with ProcessPoolExecutor(8, initializer=cc.init, initargs=(os.path.join(run_dir, "bank"),)) as ex:
        for n_done, res in enumerate(ex.map(cc.work, tasks, chunksize=2)):
            for it in res:
                x = it.pop("x", None)
                for k in ("raw_mod", "raw_orig"):
                    if k in it:
                        examples[f"{it['record_id']}|{it['condition']}|{k}"] = it.pop(k)
                items.append(it)
                if x is not None:
                    buf_x.append(x)
                    buf_idx.append(len(items) - 1)
            if len(buf_x) >= 64:
                flush()
            if n_done % 100 == 0:
                say(f"records {n_done}/{len(tasks)} ({time.time() - t0:.0f}s); items {len(items)}")
        flush()
    assert h0 == hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest() and sha_file(CKPT) == CKPT_SHA
    L = np.stack(logits)
    d = [np.abs(L[it["logit_row"]] - z["orig_logits"][pos[it["record_id"]]]).max() for it in items if it["condition"] == "orig"]
    say(f"{len(items)} items, {len(L)} model evaluations in {time.time() - t0:.0f}s; 'orig' reproduces stored logits (max abs diff {max(d):.1e}); checkpoint hash and weights unchanged")
    assert max(d) < 1e-3
    np.save(os.path.join(run_dir, "logits.npy"), L)
    keys = sorted({k for it in items for k in it})
    with open(os.path.join(run_dir, "items.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: it.get(k, "") for k in keys} for it in items])
    np.savez_compressed(os.path.join(run_dir, "example_signals.npz"), **examples)
    say("saved items.csv, logits.npy, example_signals.npz")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "population":
        population()
    elif len(sys.argv) > 2 and sys.argv[1] == "run":
        run(os.path.abspath(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == "analyze":
        import xai_atrial_content_analysis as an
        an.analyze(os.path.abspath(sys.argv[2]))
    else:
        sys.exit(__doc__)
