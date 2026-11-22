r"""Experiment 10: P-wave-like (phase-locked) atrial patch transplantation from SR donors into AFIB recipients (frozen original LightECGNetV2; counterfactual sensitivity, no clinical claim).
Stages:  population    -> Stage-1 localisation feasibility audit, donor bank, deterministic donor selection, run directory results/xai_pwave_transplant/run_<stamp>
         run <dir>     -> all conditions per AFIB recipient (torch-free workers), quality control, frozen-model evaluation
         analyze <dir> -> paired statistics and figures (xai_pwave_analysis.py)
Population, beats, region convention (r-160..r-51), preprocessing and thresholds are those of Experiments 8/9 (read-only copies / read-only use of the Experiment 9 region bank).
Usage: .venv\Scripts\python.exe -I scripts\xai_pwave_transplant.py population | run <dir> | analyze <dir>"""
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
import pwave_core as pc  # noqa: E402
from ecg_concept import atrial_content as ac  # noqa: E402
from ecg_concept import atrial_ops as ao  # noqa: E402
from ecg_concept import pwave_ops as pw  # noqa: E402

EXP8 = os.path.join(ROOT, "results", "xai_atrial_activity", "run_20261120_152906")
EXP9 = os.path.join(ROOT, "results", "xai_atrial_content", "run_20261120_154832")
CKPT = os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")
CKPT_SHA = "7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72"
CACHE = os.path.join(ROOT, "results", "step4_4", "cache")
DS = c0.DS
N_AF, N_RND = 5, 3
LAG_RANGE = (-88, -72)                                        # donor pool: every offset in {-45..+10} must be feasible for a typical donor
LAMBDA = 0.05


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def population():
    assert sha_file(CKPT) == CKPT_SHA
    run = os.path.join(ROOT, "results", "xai_pwave_transplant", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
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
    idx9 = json.load(open(os.path.join(EXP9, "bank", "index.json")))["160"]
    dev9 = np.load(os.path.join(EXP9, "bank", "dev_160.npy"), mmap_mode="r")
    states = {}
    for r, v in info.items():
        states.setdefault(v["group"], {}).setdefault((v["state"], v["reasons"]), 0)
        states[v["group"]][(v["state"], v["reasons"])] += 1
    states = {g: {f"{k[0]}|{k[1]}": n for k, n in d.items()} for g, d in states.items() if g in ("SR", "AFIB")}

    # ---------------- P-like localisation on SR records (Experiment 9 region bank, read only)
    loc_rows, bank = [], {k: [] for k in ("full", "p40", "p50", "p30", "ctl")}
    index, meta = {}, {}
    total = 0
    sr_all = [r for r in sorted(idx9) if info[r]["group"] == "SR" and idx9[r][1] - idx9[r][0] >= 4]
    templ = {}
    for r in sr_all:
        a, e = idx9[r]
        D = np.asarray(dev9[a:e], float)
        L = pw.localize(D)
        ok_b = L["corr"] >= pw.CORR_MIN
        why = []
        if L["snr"] < pw.SNR_MIN:
            why.append("low_template_snr")
        if ok_b.sum() < pw.MIN_BEATS:
            why.append("too_few_beats_matching_template")
        if not (LAG_RANGE[0] <= L["lag"] <= LAG_RANGE[1]):
            why.append("lag_outside_offset_feasible_range")
        loc_rows.append({"record": r, "lag": L["lag"], "snr": L["snr"], "n_beats": len(D), "n_valid_beats": int(ok_b.sum()), "median_beat_corr": float(np.median(L["corr"])), "mean_rr": float(np.diff(np.array(info[r]["beats"])).mean()), "excluded_reason": ";".join(why)})
        templ[r] = L["template"]
        if why:
            continue
        Dv = D[ok_b]
        for key, fn in (("full", lambda d: d), ("p40", lambda d: pw.patch_from(d, L["lag"], 40)), ("p50", lambda d: pw.patch_from(d, L["lag"], 50)), ("p30", lambda d: pw.patch_from(d, L["lag"], 30)), ("ctl", lambda d: pw.control_from(d, 40))):
            bank[key].append(np.stack([fn(d) for d in Dv]).astype(np.float32))
        index[r] = [total, total + len(Dv)]
        total += len(Dv)
        meta[r] = {"lag": int(L["lag"]), "snr": L["snr"], "n_valid": int(ok_b.sum())}
    for k, v in bank.items():
        np.save(os.path.join(run, "bank", k + ".npy"), np.concatenate(v))
    json.dump(index, open(os.path.join(run, "bank", "index.json"), "x"))
    json.dump(meta, open(os.path.join(run, "bank", "meta.json"), "x"))
    with open(os.path.join(run, "pwave_localization_sr.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(loc_rows[0]))
        w.writeheader()
        w.writerows(loc_rows)
    donors_pool = sorted(meta)
    say(f"SR records with >=4 eligible beats {len(sr_all)}; donors accepted {len(donors_pool)}; beats in bank {total}")

    # ---------------- AFIB records under the same localiser (is there a phase-locked deflection?)
    af_all = [r for r in sorted(idx9) if info[r]["group"] == "AFIB" and idx9[r][1] - idx9[r][0] >= 4]
    af_rows = []
    cov = {}
    for r in af_all:
        a, e = idx9[r]
        D = np.asarray(dev9[a:e], float)
        L = pw.localize(D)
        af_rows.append({"record": r, "lag": L["lag"], "snr": L["snr"], "median_beat_corr": float(np.median(L["corr"]))})
        cov[r] = {"mean_rr": float(np.diff(np.array(info[r]["beats"])).mean()), "rms": ac.rms_lead([d for d in D])}
    for r in donors_pool:
        a, e = idx[r] if False else idx9[r]
        cov[r] = {"mean_rr": float(np.diff(np.array(info[r]["beats"])).mean()), "rms": np.median(np.stack([np.sqrt(np.mean(np.asarray(pb) ** 2, axis=1)) for pb in np.load(os.path.join(run, "bank", "p40.npy"), mmap_mode="r")[index[r][0]:index[r][1]]]), axis=0)}

    # ---------------- deterministic donor selection (no model output)
    def dist(r, ds):
        rr = np.array([abs(math.log(cov[d]["mean_rr"] / cov[r]["mean_rr"])) for d in ds])
        am = np.array([np.median(np.abs(np.log((cov[d]["rms"] + 1e-6) / (cov[r]["rms"] + 1e-6)))) for d in ds])
        return rr + am, rr, am

    recip = sorted(af_all, key=lambda r: zlib.crc32((r + str(c0.SEED)).encode()))
    cap = int(math.ceil(1.5 * N_AF * len(recip) / len(donors_pool)))
    usage = {d: 0 for d in donors_pool}
    donors, rows = {}, []
    for r in recip:
        cand = [d for d in donors_pool if usage[d] < cap]
        tot, rr, am = dist(r, cand)
        score = tot + LAMBDA * np.array([usage[d] for d in cand])
        order = np.argsort(score, kind="stable")[:N_AF]
        donors[r] = {"AF": []}
        for rank, i in enumerate(order):
            usage[cand[i]] += 1
            donors[r]["AF"].append([cand[i], float(tot[i])])
            rows.append([r, "AF", rank, cand[i], float(tot[i]), float(rr[i]), float(am[i]), usage[cand[i]], meta[cand[i]]["lag"], meta[cand[i]]["snr"]])
        rg = np.random.RandomState(zlib.crc32(("rnd" + r + str(c0.SEED)).encode()))
        pick = rg.choice(len(donors_pool), N_RND, replace=False)
        tot, _, _ = dist(r, [donors_pool[i] for i in pick])
        donors[r]["RND"] = [[donors_pool[i], float(t)] for i, t in zip(pick, tot)]
        for rank, (i, t) in enumerate(zip(pick, tot)):
            rows.append([r, "RND", rank, donors_pool[i], float(t), "", "", "", meta[donors_pool[i]]["lag"], meta[donors_pool[i]]["snr"]])
    json.dump(donors, open(os.path.join(run, "donors.json"), "x"))
    with open(os.path.join(run, "donor_selection.csv"), "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recipient", "donor_class", "rank", "donor", "distance", "rr_term", "amp_term", "usage_after", "donor_lag", "donor_snr"])
        w.writerows(rows)

    # ---------------- audit
    snr_sr = np.array([x["snr"] for x in loc_rows])
    lag_sr = np.array([x["lag"] for x in loc_rows])
    snr_af = np.array([x["snr"] for x in af_rows])
    excl = {}
    for x in loc_rows:
        for t in (x["excluded_reason"].split(";") if x["excluded_reason"] else ["accepted"]):
            excl[t] = excl.get(t, 0) + 1
    use_ = np.array(list(usage.values()))
    d1 = np.array([v["AF"][0][1] for v in donors.values()])
    af_elig_rr = [cov[r]["mean_rr"] for r in af_all]
    af_inel_rr = [float(np.diff(np.array(info[r]["beats"])).mean()) for r, v in info.items() if v["group"] == "AFIB" and v["usable"] and r not in idx9 or (v["group"] == "AFIB" and v["usable"] and r in idx9 and idx9[r][1] - idx9[r][0] < 4)]
    audit = {"localisation_method": "template-based phase-locked pre-QRS deflection per SR record (lead-pooled energy peak of the mean region deviation); not an expert-validated P-wave delineation; no packages installed; no existing P-wave delineator in the repository (only R-peak detectors)",
             "definitions": {"template_snr_min": pw.SNR_MIN, "beat_corr_min": pw.CORR_MIN, "min_valid_beats": pw.MIN_BEATS, "donor_lag_range_samples_before_R": LAG_RANGE, "patch_widths_(pre,post)": pw.WIDTHS, "control_window_region_idx": [pw.CTL_START, pw.CTL_START + 40]},
             "annotation_states_exp7": states,
             "SR_records_with_>=4_eligible_beats": len(sr_all), "SR_donors_accepted": len(donors_pool), "SR_exclusion_reasons_(a record can have several)": excl, "beats_in_donor_bank": total,
             "SR_template_snr_percentiles_5_25_50_75_95": np.percentile(snr_sr, [5, 25, 50, 75, 95]).round(2).tolist(), "SR_lag_percentiles_5_25_50_75_95": np.percentile(lag_sr, [5, 25, 50, 75, 95]).tolist(),
             "AFIB_eligible_recipients": len(af_all), "AFIB_template_snr_percentiles_5_25_50_75_95_(same_localiser)": np.percentile(snr_af, [5, 25, 50, 75, 95]).round(2).tolist(), "AFIB_share_snr>=5": float((snr_af >= 5).mean()),
             "donor_usage": {"n_donors": len(use_), "used": int((use_ > 0).sum()), "min": int(use_.min()), "median": float(np.median(use_)), "max": int(use_.max()), "cap": cap},
             "matched_distance_first_donor": {"median": float(np.median(d1)), "p5": float(np.percentile(d1, 5)), "p95": float(np.percentile(d1, 95))},
             "selection_bias_heart_rate_mean_RR_samples": {"AFIB_eligible_median": float(np.median(af_elig_rr)), "AFIB_ineligible_median": float(np.median(af_inel_rr)) if af_inel_rr else None, "n_AFIB_ineligible": len(af_inel_rr),
                                                          "SR_donors_median": float(np.median([cov[d]["mean_rr"] for d in donors_pool]))},
             "verdict": ("A phase-locked, template-consistent pre-QRS deflection can be localised in most SR records (high SNR, lag about 6.4-9.4 samples-hundredths before R). It is called 'P-wave-like'. Its alignment is relative to the R peak of the donor beat; "
                         "PR-interval variability across beats is not modelled; ectopic beats are excluded only through the Experiment 7 annotation-based exclusions and the per-beat template correlation. AFIB records do not show such a deflection under the same localiser.")}
    json.dump(audit, open(os.path.join(run, "stage1_audit.json"), "x"), indent=1, default=float)
    np.savez_compressed(os.path.join(run, "sr_templates.npz"), **{r: templ[r].astype(np.float32) for r in donors_pool[:400]})
    json.dump({"seed": c0.SEED, "checkpoint_sha256": CKPT_SHA, "exp8_run": os.path.relpath(EXP8, ROOT), "exp9_run_region_bank_readonly": os.path.relpath(EXP9, ROOT), "offsets": pc.OFFS, "patch_widths": pw.WIDTHS, "n_donors": {"AF": N_AF, "RND": N_RND},
               "usage_lambda": LAMBDA, "usage_cap": cap, "scale_compat": pc.SCALE_COMPAT, "rr_target_cv_factorial": pc.RR_TARGET_CV, "bootstrap": "record-level B=5000 seed 20261120",
               "records_beats_sha256": sha_file(os.path.join(run, "records_beats.json")), "donors_sha256": sha_file(os.path.join(run, "donors.json")), "audit_sha256": sha_file(os.path.join(run, "stage1_audit.json")),
               "exp9_bank_dev160_sha256": sha_file(os.path.join(EXP9, "bank", "dev_160.npy")),
               "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("src/ecg_concept/pwave_ops.py", "scripts/pwave_core.py", "scripts/xai_pwave_transplant.py", "src/ecg_concept/atrial_content.py", "src/ecg_concept/atrial_ops.py", "scripts/atrial_core.py")}},
              open(os.path.join(run, "config.json"), "x"), indent=1)
    say(json.dumps({k: v for k, v in audit.items() if k not in ("verdict",)}, default=float)[:3800])
    print(run)


def run(run_dir):
    import torch
    from ecg_concept.inference import load_model
    info = json.load(open(os.path.join(run_dir, "records_beats.json")))
    donors = json.load(open(os.path.join(run_dir, "donors.json")))
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    use = sorted(r for r, v in info.items() if v["usable"] and v["group"] == "AFIB")
    keep = set([r for r in use if r in donors and len(ao.regions(np.array(info[r]["beats"], int), 160)[0]) >= 8][:3])
    tasks = [(r, "AFIB", os.path.join(DS, rel[r]), info[r]["beats"], info[r]["cv"], i, donors.get(r, {}), r in keep) for i, r in enumerate(use)]
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

    with ProcessPoolExecutor(8, initializer=pc.init, initargs=(os.path.join(run_dir, "bank"),)) as ex:
        for n_done, res in enumerate(ex.map(pc.work, tasks, chunksize=2)):
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
            if n_done % 40 == 0:
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
        import xai_pwave_analysis as an
        an.analyze(os.path.abspath(sys.argv[2]))
    else:
        sys.exit(__doc__)
