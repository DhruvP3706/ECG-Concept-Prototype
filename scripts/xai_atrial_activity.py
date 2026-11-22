r"""Experiment 8: atrial-activity interventions on SR and AFIB records of the frozen validation role (frozen original LightECGNetV2; counterfactual sensitivity, no clinical claim).
Stages:  population  -> Stage-1 feasibility audit (P-wave localisation, eligibility coverage, f-wave calibration) + creates results/xai_atrial_activity/run_<stamp>
         run <dir>   -> builds every condition per record (torch-free workers), quality control, frozen-model evaluation
         analyze <dir> -> paired statistics and figures (xai_atrial_analysis.py)
Groups, beats and exclusions are those of Experiment 7 (read-only copies of results/xai_graded_rr/run_20261120_145324/records_beats.json). Conditions: see src/ecg_concept/atrial_ops.py and scripts/atrial_core.py.
Usage: .venv\Scripts\python.exe -I scripts\xai_atrial_activity.py population | run <dir> | analyze <dir>"""
import csv
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import atrial_core as core  # noqa: E402
from ecg_concept import atrial_ops as ao  # noqa: E402
from ecg_concept.preprocessing import read_raw_mat  # noqa: E402

EXP7 = os.path.join(ROOT, "results", "xai_graded_rr", "run_20261120_145324")
CKPT = os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")
CKPT_SHA = "7779719e9480a0181cff99933d91dbea9051eeda470e4bd0b20fa33119d79f72"
CACHE = os.path.join(ROOT, "results", "step4_4", "cache")
DS = core.DS


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def population():
    from scipy.signal import butter, sosfiltfilt
    assert sha_file(CKPT) == CKPT_SHA
    run = os.path.join(ROOT, "results", "xai_atrial_activity", "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run)
    log = open(os.path.join(run, "log.txt"), "x", buffering=1)

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")

    info = json.load(open(os.path.join(EXP7, "records_beats.json")))
    shutil.copy(os.path.join(EXP7, "records_beats.json"), os.path.join(run, "records_beats.json"))
    shutil.copy(os.path.join(EXP7, "population.json"), os.path.join(run, "population_exp7_copy.json"))
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    use = {g: sorted(r for r, v in info.items() if v["usable"] and v["group"] == g) for g in ("SR", "AFIB")}
    sos = butter(3, [0.5, 40], btype="band", fs=500, output="sos")
    sos_f = butter(3, [4, 9], btype="band", fs=500, output="sos")
    audit = {"definition": "pre-QRS average over beats with RR >= 460 samples (920 ms), segment [r-220, r-41], endpoints joined by a straight line; SNR = |peak of average| / mean standard error", "groups": {}}
    pooled = {"II": [], "V1": []}
    fw = {"SR": [], "AFIB": []}
    for g in ("SR", "AFIB"):
        rows = {"II": [], "V1": []}
        for r in use[g]:
            raw = read_raw_mat(os.path.join(DS, rel[r]))
            b = np.array(info[r]["beats"], int)
            regs, _ = ao.regions(b, 160)
            if len(regs) >= 4:
                fw[g].append(ao.region_deviation_rms(raw, regs))
            for ln, li in (("II", 1), ("V1", 6)):
                x = sosfiltfilt(sos, raw[li].astype(float))
                xfb = sosfiltfilt(sos_f, raw[li].astype(float))
                segs, tp = [], []
                for i in range(1, len(b)):
                    if b[i] - b[i - 1] >= 460 and b[i] - 220 >= 0:
                        s = x[b[i] - 220:b[i] - 41]
                        s = s - (s[0] + (s[-1] - s[0]) * np.arange(len(s)) / (len(s) - 1))
                        segs.append(s)
                        a0, a1 = b[i - 1] + 191, b[i] - 41
                        if a1 - a0 >= 60:
                            tp.append(float(np.sqrt(np.mean(xfb[a0:a1] ** 2))))
                if len(segs) >= 4:
                    S = np.stack(segs)
                    avg = S.mean(0)
                    se = S.std(0, ddof=1) / np.sqrt(len(S))
                    k = int(np.argmax(np.abs(avg)))
                    rows[ln].append({"snr": float(np.abs(avg[k]) / (se.mean() + 1e-9)), "peak_lag": k - 179, "amp": float(np.abs(avg[k])), "tp_band_rms": float(np.median(tp)) if tp else float("nan"), "avg": avg})
                    if g == "SR" and rows[ln][-1]["snr"] > 4:
                        pooled[ln].append(avg ** 2 / (np.sum(avg ** 2) + 1e-9))
        audit["groups"][g] = {}
        for ln in ("II", "V1"):
            rr_ = rows[ln]
            sn = np.array([x["snr"] for x in rr_])
            audit["groups"][g][ln] = {"records_with_>=4_long_RR_beats": len(rr_), "of": len(use[g]), "snr_median": float(np.median(sn)) if len(sn) else None, "share_snr_gt_4": float((sn > 4).mean()) if len(sn) else None,
                                      "peak_lag_median_samples": float(np.median([x["peak_lag"] for x in rr_ if x["snr"] > 4])) if (sn > 4).any() else None, "band_4_9Hz_rms_in_TP_median_counts": float(np.nanmedian([x["tp_band_rms"] for x in rr_])) if rr_ else None}
    for ln in ("II", "V1"):
        E = np.mean(pooled[ln], axis=0)
        c = np.cumsum(E) / E.sum()
        audit[f"SR_pooled_energy_window_{ln}"] = {"n_records": len(pooled[ln]), "5pct_to_95pct_samples_before_R": [220 - int(np.searchsorted(c, 0.05)), 220 - int(np.searchsorted(c, 0.95))],
                                                  "quartiles_samples_before_R": [220 - int(np.searchsorted(c, q)) for q in (0.25, 0.5, 0.75)]}
    for a1 in (120, 160, 200):
        audit[f"eligibility_a1={a1}"] = {g: {"share_beats_eligible": float(np.mean([len(ao.regions(np.array(info[r]['beats'], int), a1)[0]) / max(len(info[r]['beats']) - 1, 1) for r in use[g]])),
                                              "records_with_>=4_eligible_beats": float(np.mean([len(ao.regions(np.array(info[r]['beats'], int), a1)[0]) >= 4 for r in use[g]]))} for g in use}
    fw_target = np.nanmedian(np.stack(fw["AFIB"]), axis=0)
    audit["fwave_calibration"] = {"AFIB_median_region_deviation_rms_per_lead_counts": fw_target.round(2).tolist(), "SR_median_region_deviation_rms_per_lead_counts": np.nanmedian(np.stack(fw["SR"]), axis=0).round(2).tolist(),
                                  "n_AFIB_records_used": len(fw["AFIB"])}
    audit["verdict"] = ("SR: a coherent pre-QRS atrial deflection is recoverable (high SNR); chosen atrial region = [r-160, r-51] (samples). AFIB: no phase-locked atrial wave (low SNR), the region is positional only. "
                        "Eligibility rule: previous R peak >= 220 samples before the region start (RR >= 380 for the 160-sample region).")
    json.dump(audit, open(os.path.join(run, "stage1_audit.json"), "x"), indent=1, default=float)
    json.dump({"fwave_target_rms_per_lead": fw_target.tolist(), "seed": core.SEED, "checkpoint_sha256": CKPT_SHA, "conditions_atrial": core.ATRIAL, "min_eligible_beats": core.MIN_ELIG, "rms_min": core.RMS_MIN,
               "diff_ratio_max": core.DIFF_RATIO_MAX, "sham_ratio": core.SHAM_RATIO, "timing_tol": core.TIMING_TOL, "factorial_cv": core.FACTORIAL_CV, "rigid_factorial": core.RIGID,
               "exp7_run": os.path.relpath(EXP7, ROOT), "records_beats_sha256": sha_file(os.path.join(run, "records_beats.json")), "audit_sha256": sha_file(os.path.join(run, "stage1_audit.json")),
               "code_sha256": {f: sha_file(os.path.join(ROOT, f)) for f in ("src/ecg_concept/atrial_ops.py", "scripts/atrial_core.py", "scripts/xai_atrial_activity.py")}}, open(os.path.join(run, "config.json"), "x"), indent=1)
    say(json.dumps({k: v for k, v in audit.items() if k != "verdict"}, default=float)[:4000])
    print(run)


def run(run_dir):
    import torch
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from ecg_concept.inference import load_model
    info = json.load(open(os.path.join(run_dir, "records_beats.json")))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    fw = np.array(cfg["fwave_target_rms_per_lead"])
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    use = sorted(r for r, v in info.items() if v["usable"])
    keep = set()
    for g in ("SR", "AFIB"):
        gr = [r for r in use if info[r]["group"] == g and len(ao.regions(np.array(info[r]["beats"], int), 160)[0]) >= 8]
        keep |= set(gr[:2])
    tasks = [(r, info[r]["group"], os.path.join(DS, rel[r]), info[r]["beats"], info[r]["cv"], i, fw, r in keep) for i, r in enumerate(use)]
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

    with ProcessPoolExecutor(8) as ex:
        for n_done, res in enumerate(ex.map(core.work, tasks, chunksize=2)):
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
        import xai_atrial_analysis as an
        an.analyze(os.path.abspath(sys.argv[2]))
    else:
        sys.exit(__doc__)
