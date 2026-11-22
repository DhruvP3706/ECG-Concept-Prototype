"""Step 4.3 bounded training-only pilot of the RR-concept extractor.

Stages (run in this order):
  sample   deterministic sampling of DEV-S / LOCK-S / LOCK-R units from TRAIN-role records; lists saved + hashed BEFORE any waveform is read
  develop  run the extractor on DEV-S only (results to results/step4_3/work/); may be repeated while fixing implementation bugs
  freeze   write extraction_config.json with parameter + source-code hashes (replaces a git commit; no Git is used)
  lock     verify the frozen hashes, then process DEV-S, LOCK-S and LOCK-R once with the frozen code
  report   summaries and engineering checks (work/pilot_summary.json); pilot_report.md is rendered by scripts/step4_3_render_report.py

Only train-role waveforms are read (sampled units plus their train-role duplicate partners). Labels and batch metadata are used for
sampling and for AGGREGATE coverage tables only; they never enter a concept value or mask. No waveform is exported.
Run:  .venv\\Scripts\\python.exe -I scripts\\step4_3_pilot.py <stage>
"""
import csv
import glob
import hashlib
import importlib.util
import io
import json
import os
import platform
import sys
import time
import unittest
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept import rr_concepts as rc  # noqa: E402
from ecg_concept.preprocessing import read_raw_mat  # noqa: E402

OUT = os.path.join(ROOT, "results", "step4_3")
WORK = os.path.join(OUT, "work")
os.makedirs(WORK, exist_ok=True)
DS = glob.glob(os.path.join(ROOT, "data", "raw", "a-large-scale-*"))[0]
ROLE = os.path.join(ROOT, "results", "step3_7", "prospective_role_manifest.csv")
FOLD = os.path.join(ROOT, "results", "step3_7", "original_fold_manifest.csv")
SEED, QUOTA, N_R, N_STRESS = 20261111, 10, 100, 120
LOGF = os.path.join(OUT, "execution_log.txt")
STRATA = ["S01_timestamp_block", "S02_age_missing", "S03_AFIB", "S04_AF_flutter", "S05_SVT_AT", "S06_ectopy", "S07_conduction_wide_QRS",
          "S08_SA", "S09_SB", "S10_ST", "S11_SR", "S12_LVQRSAL"]
SRC_FILES = ["src/ecg_concept/beat_detection.py", "src/ecg_concept/rr_concepts.py", "src/ecg_concept/preprocessing.py",
             "scripts/step4_3_pilot.py", "tests/test_rr_concepts.py"]


def log(*a):
    msg = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def never_overwrite(*paths):
    for p in paths:
        assert not os.path.exists(p), f"refusing to overwrite {p}"


def write_csv(path, rows, cols=None):
    cols = cols or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(r.get(k), float) and np.isnan(r.get(k)) else r.get(k, "")) for k in cols})


def read_manifest_rows():
    return list(csv.DictReader(open(os.path.join(OUT, "pilot_manifest.csv"), encoding="utf-8")))


# ------------------------------------------------------------------ sample
def header_flags(path):
    """(timestamp_present, age_missing) from the .hea of a TRAIN record; used for stratification/coverage only."""
    try:
        L = open(path, encoding="utf-8", errors="replace").read().replace("\r", "").split("\n")
        ts = len(L[0].split()) == 6
        age = next((l.split(":", 1)[1].strip() for l in L if l.startswith("#Age")), None)
        return ts, age is None or age == "0" or age.lower() == "nan"
    except Exception:
        return False, False


def stage_sample():
    never_overwrite(os.path.join(OUT, "pilot_manifest.csv"))
    rows = [r for r in csv.DictReader(open(ROLE)) if r["role"] == "train"]
    assert len(rows) == 35568
    rel = {r["record_id"]: r["dataset_rel_path"] for r in csv.DictReader(open(FOLD))}
    ids = [r["record_id"] for r in rows]
    cache = os.path.join(WORK, "header_flags.csv")
    if os.path.exists(cache):
        flags = {r["record_id"]: (r["ts"] == "True", r["age_missing"] == "True") for r in csv.DictReader(open(cache))}
    else:
        t0 = time.time()
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(lambda i: header_flags(os.path.join(DS, rel[i][:-4] + ".hea")), ids))
        flags = dict(zip(ids, res))
        write_csv(cache, [{"record_id": i, "ts": f[0], "age_missing": f[1]} for i, f in flags.items()])
        log(f"header flags for {len(ids)} train records in {time.time() - t0:.0f}s")
    by = {r["record_id"]: r for r in rows}
    units = defaultdict(list)
    for r in rows:
        units[r["dup_group_id"] if r["within_train_dup_group"] == "True" else r["record_id"]].append(r["record_id"])
    units = {u: sorted(m) for u, m in units.items()}
    assert all(len(m) in (1, 2) for m in units.values())
    key = {u: hashlib.sha256(f"{SEED}|{u}".encode()).hexdigest() for u in units}
    order = sorted(units, key=lambda u: key[u])

    def stratum(u):
        r = by[units[u][0]]
        lab, n = set(r["labels"].split("|")), int(r["record_id"][2:])
        ts, am = flags[r["record_id"]]
        if 8344 <= n <= 9373 and ts:
            return STRATA[0]
        if am:
            return STRATA[1]
        for name, cond in ((STRATA[2], "AFIB" in lab), (STRATA[3], "AF" in lab), (STRATA[4], bool(lab & {"SVT", "AT"})),
                            (STRATA[5], bool(lab & {"APB", "VPB", "JEB", "VEB"})),
                            (STRATA[6], bool(lab & {"RBBB", "LFBBB", "IVB", "WPW", "VPE", "1AVB", "2AVB", "2AVB1", "3AVB", "AVB"})),
                            (STRATA[7], "SA" in lab), (STRATA[8], "SB" in lab), (STRATA[9], "ST" in lab), (STRATA[10], "SR" in lab),
                            (STRATA[11], "LVQRSAL" in lab)):
            if cond:
                return name
        return "S99_other"

    strat = {u: stratum(u) for u in units}
    sets, how, drawn = {"LOCK-R": order[:N_R], "DEV-S": [], "LOCK-S": []}, {}, set(order[:N_R])
    for u in sets["LOCK-R"]:
        how[("LOCK-R", u)] = "srs"
    info = {"eligible_units_per_stratum": dict(Counter(strat.values())), "shortfalls": {}}
    for name in ("DEV-S", "LOCK-S"):
        sel = []
        for s in STRATA:
            take = [u for u in order if strat[u] == s and u not in drawn][:QUOTA]
            if len(take) < QUOTA:
                info["shortfalls"][f"{name}:{s}"] = QUOTA - len(take)
            for u in take:
                how[(name, u)] = "stratum"
            sel += take
            drawn |= set(take)
        need = N_STRESS - len(sel)
        if need > 0:                                           # deterministic fallback: next units by hash order, any stratum
            fill = [u for u in order if u not in drawn][:need]
            for u in fill:
                how[(name, u)] = "fallback"
            sel += fill
            drawn |= set(fill)
        sets[name] = sel
    assert len({u for s in sets.values() for u in s}) == sum(map(len, sets.values())), "sets overlap"
    out = []
    for name in ("DEV-S", "LOCK-S", "LOCK-R"):
        for u in sets[name]:
            for m in units[u]:
                r = by[m]
                out.append({"set": name, "unit_id": u, "record_id": m, "member_role": "metrics" if m == units[u][0] else "partner",
                            "stratum": strat[u], "selection": how[(name, u)], "unit_key": key[u][:16],
                            "n_members": len(units[u]), "rel_path": rel[m], "ts_flag": flags[m][0], "age_missing_flag": flags[m][1],
                            "id_block": "JS08344-09373" if 8344 <= int(m[2:]) <= 9373 else "other", "labels": r["labels"]})
    write_csv(os.path.join(OUT, "pilot_manifest.csv"), out)
    sizes = {n: len(sets[n]) for n in sets}
    info.update(seed=SEED, quota_per_stratum=QUOTA, set_unit_counts=sizes,
                set_record_counts={n: sum(1 for r in out if r["set"] == n) for n in sets},
                lists_sha256=hashlib.sha256("\n".join(sorted(f"{r['set']}|{r['unit_id']}|{r['record_id']}" for r in out)).encode()).hexdigest(),
                manifest_sha256=sha_file(os.path.join(OUT, "pilot_manifest.csv")),
                rules={"unit": "duplicate group if within-train pair else record", "key": "sha256(seed|unit_id) ascending",
                       "draw_order": "LOCK-R (SRS 100 over all train units) -> DEV-S -> LOCK-S",
                       "stratum_priority": STRATA, "shortfall": "fill with next units by hash order from any stratum, selection='fallback'",
                       "metrics_member": "lowest record ID in the unit; partner is processed only for identical-output checks"})
    json.dump(info, open(os.path.join(WORK, "sampling_info.json"), "w"), indent=1)
    log("sampled", sizes, "records", info["set_record_counts"], "shortfalls", info["shortfalls"], "manifest sha256", info["manifest_sha256"][:16])


# ------------------------------------------------------------------ processing
def process_rows(rows, cfg=rc.CONFIG):
    """Extract concepts for manifest rows (reads raw .mat per row). Returns list of result dicts + beat-time rows + raw hashes."""
    res, beats, raw_hash = [], [], {}
    for r in rows:
        path = os.path.join(DS, r["rel_path"])
        try:
            sig = read_raw_mat(path)
            raw_hash[r["record_id"]] = hashlib.sha256(np.ascontiguousarray(sig).tobytes()).hexdigest()
            out = rc.extract_record(sig, cfg, return_beats=True)
            out["read_ok"] = True
        except Exception as e:
            out = rc.extract_record(None, cfg, return_beats=True)
            out["read_ok"] = False
            out["reason_codes"] = "READ_ERROR:" + type(e).__name__
        out.update(record_id=r["record_id"], unit_id=r["unit_id"], set=r["set"], stratum=r["stratum"], member_role=r["member_role"], selection=r["selection"])
        for det, t in out.pop("beats", {}).items():
            beats += [{"record_id": r["record_id"], "set": r["set"], "detector": det, "beat_index": i, "beat_time_s": round(float(x), 5)} for i, x in enumerate(t)]
        res.append(out)
    return res, beats, raw_hash


CONCEPT_COLS = ["record_id", "unit_id", "set", "stratum", "member_role", "selection", "read_ok", "structural_ok", "n_samples", "n_usable_leads", "n_flat_leads",
                "n_clip_leads", "n_nonfinite_leads", "n_beats_A", "n_beats_B", "n_beats_cons", "n_rr_cons", "n_matched", "n_unmatched_A", "n_unmatched_B",
                "match_median_abs_dt_ms", "median_support_A", "median_support_B", "n_edge_excluded_A", "n_edge_excluded_B",
                "rate_mean_bpm_A", "rate_mean_bpm_B", "rate_mean_bpm_cons", "rate_median_bpm_A", "rate_median_bpm_B", "rate_median_bpm_cons",
                "rr_cv_A", "rr_cv_B", "rr_cv_cons", "rr_nrmssd_A", "rr_nrmssd_B", "rr_nrmssd_cons", "rr_mad_succ_A", "rr_mad_succ_B", "rr_mad_succ_cons",
                "rate_state", "irr_state", "reason_codes", "accepted_rate_mean_bpm", "accepted_rate_median_bpm", "accepted_rr_cv", "accepted_rr_nrmssd",
                "accepted_rr_mad_succ", "runtime_s"]


def comparison_rows(res):
    out = []
    for r in res:
        na, nb, m = r["n_beats_A"], r["n_beats_B"], r["n_matched"]
        out.append({"record_id": r["record_id"], "set": r["set"], "stratum": r["stratum"], "member_role": r["member_role"], "tolerance_ms": rc.CONFIG["match_tol_s"] * 1000,
                    "n_beats_A": na, "n_beats_B": nb, "n_matched": m, "n_unmatched_A": r["n_unmatched_A"], "n_unmatched_B": r["n_unmatched_B"],
                    "dice_agreement": (2 * m / (na + nb)) if (na + nb) else float("nan"), "match_median_abs_dt_ms": r["match_median_abs_dt_ms"],
                    "rate_mean_bpm_A": r["rate_mean_bpm_A"], "rate_mean_bpm_B": r["rate_mean_bpm_B"],
                    "abs_rate_mean_diff_bpm": abs(r["rate_mean_bpm_A"] - r["rate_mean_bpm_B"]),
                    "rr_nrmssd_A": r["rr_nrmssd_A"], "rr_nrmssd_B": r["rr_nrmssd_B"], "abs_rr_nrmssd_diff": abs(r["rr_nrmssd_A"] - r["rr_nrmssd_B"])})
    return out


def stage_develop():
    rows = [r for r in read_manifest_rows() if r["set"] == "DEV-S"]
    t0 = time.time()
    res, beats, _ = process_rows(rows)
    path = os.path.join(WORK, f"dev_concepts_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    write_csv(path, res, CONCEPT_COLS)
    st = Counter(r["rate_state"] for r in res if r["member_role"] == "metrics")
    ir = Counter(r["irr_state"] for r in res if r["member_role"] == "metrics")
    reasons = Counter(c for r in res if r["member_role"] == "metrics" for c in r["reason_codes"].split(";") if c)
    log(f"DEV-S processed {len(res)} records in {time.time() - t0:.1f}s -> {path}")
    log("DEV rate states", dict(st), "irr states", dict(ir), "reasons", dict(reasons))
    log("DEV read failures", sum(not r["read_ok"] for r in res))


# ------------------------------------------------------------------ freeze / lock
def code_fingerprint():
    return {f: sha_file(os.path.join(ROOT, f)) for f in SRC_FILES}


def dup_check_rows():
    """All remaining within-train duplicate pairs not in the sampled sets: processed ONLY for identical-output checks (set DUP-CHECK)."""
    rel = {r["record_id"]: r["dataset_rel_path"] for r in csv.DictReader(open(FOLD))}
    sampled = {r["unit_id"] for r in read_manifest_rows()}
    groups = defaultdict(list)
    for r in csv.DictReader(open(ROLE)):
        if r["role"] == "train" and r["within_train_dup_group"] == "True":
            groups[r["dup_group_id"]].append(r["record_id"])
    out = []
    for g, m in sorted(groups.items()):
        if g not in sampled:
            for x in sorted(m):
                out.append({"set": "DUP-CHECK", "unit_id": g, "record_id": x, "member_role": "metrics" if x == min(m) else "partner",
                            "stratum": "dup_check", "selection": "dupcheck", "rel_path": rel[x]})
    return out


def stage_freeze():
    dest = os.path.join(OUT, "extraction_config.json")
    never_overwrite(dest)
    info = json.load(open(os.path.join(WORK, "sampling_info.json")))
    dup = dup_check_rows()
    never_overwrite(os.path.join(WORK, "dup_check_manifest.csv"))
    write_csv(os.path.join(WORK, "dup_check_manifest.csv"), dup)
    cfg = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "frozen_before_processing_lock_sets": True,
           "note": "source-code and parameter hashes replace a git commit (no Git operations were used)",
           "parameters": rc.CONFIG, "parameters_sha256": hashlib.sha256(json.dumps(rc.CONFIG, sort_keys=True).encode()).hexdigest(),
           "source_sha256": code_fingerprint(), "sampling": info,
           "input_sha256": {"role_manifest": sha_file(ROLE), "original_fold_manifest": sha_file(FOLD), "pilot_manifest": sha_file(os.path.join(OUT, "pilot_manifest.csv")),
                            "dup_check_manifest": sha_file(os.path.join(WORK, "dup_check_manifest.csv")),
                            "label_dictionary": sha_file(os.path.join(DS, "ConditionNames_SNOMED-CT.csv"))},
           "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": __import__("scipy").__version__},
           "detectors": {"A": "ADAPT: Pan-Tompkins-style adaptive-threshold energy detector (own implementation, not a verified reproduction)",
                         "B": "TMA: Elgendi-style two-moving-average detector (own implementation, not a verified reproduction)"},
           "fusion": "per-detector cross-lead clustering over leads I, II, V1-V6 (<=50 ms from first member, >=3 leads), then one-to-one detector matching (50 ms)",
           "interpretation": "automatically derived 10-second RR statistics; not conventional HRV; no reference annotations; no accuracy claim"}
    json.dump(cfg, open(dest, "w"), indent=1)
    log("frozen config written", dest, "params sha", cfg["parameters_sha256"][:16])


def stage_lock():
    never_overwrite(*(os.path.join(OUT, f) for f in ("pilot_concepts.csv", "detector_comparison.csv", "pilot_beat_times.csv")))
    frozen = json.load(open(os.path.join(OUT, "extraction_config.json")))
    assert frozen["source_sha256"] == code_fingerprint(), "source changed since freeze; re-freeze is not allowed after lock sets were declared"
    assert frozen["parameters_sha256"] == hashlib.sha256(json.dumps(rc.CONFIG, sort_keys=True).encode()).hexdigest(), "parameters changed since freeze"
    assert sha_file(os.path.join(OUT, "pilot_manifest.csv")) == frozen["input_sha256"]["pilot_manifest"], "manifest changed since freeze"
    assert sha_file(os.path.join(WORK, "dup_check_manifest.csv")) == frozen["input_sha256"]["dup_check_manifest"], "dup-check list changed since freeze"
    rows = read_manifest_rows()
    dup_rows = list(csv.DictReader(open(os.path.join(WORK, "dup_check_manifest.csv"), encoding="utf-8")))
    t0 = time.time()
    res, beats, raw_hash = process_rows(rows + dup_rows)
    log(f"processed {len(res)} records (3 sampled sets + {len(dup_rows)} DUP-CHECK records, frozen code) in {time.time() - t0:.1f}s")
    write_csv(os.path.join(OUT, "pilot_concepts.csv"), res, CONCEPT_COLS)
    write_csv(os.path.join(OUT, "detector_comparison.csv"), comparison_rows(res))
    write_csv(os.path.join(OUT, "pilot_beat_times.csv"), beats, ["record_id", "set", "detector", "beat_index", "beat_time_s"])
    json.dump(raw_hash, open(os.path.join(WORK, "raw_array_hashes.json"), "w"))
    # determinism re-run on 20 DEV-S metric records (not an assessment set)
    dev = [r for r in rows if r["set"] == "DEV-S" and r["member_role"] == "metrics"][:20]
    again, _, _ = process_rows(dev)
    first = {r["record_id"]: r for r in res}
    same = all(all(_eq(first[a["record_id"]][k], a[k]) for k in CONCEPT_COLS if k not in ("runtime_s",)) for a in again)
    json.dump({"determinism_rerun_records": len(dev), "identical": bool(same)}, open(os.path.join(WORK, "determinism.json"), "w"))
    log("determinism re-run on", len(dev), "DEV-S records identical:", same)


def _eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


# ------------------------------------------------------------------ report
def q(v, p=(5, 25, 50, 75, 95)):
    v = np.asarray([x for x in v if x == x], dtype=float)
    return [round(float(np.percentile(v, i)), 3) for i in p] if v.size else None


def stage_report():
    never_overwrite(os.path.join(WORK, "pilot_summary.json"))
    man = {(r["set"], r["record_id"]): r for r in read_manifest_rows()}
    conc = list(csv.DictReader(open(os.path.join(OUT, "pilot_concepts.csv"), encoding="utf-8")))
    cmp_ = list(csv.DictReader(open(os.path.join(OUT, "detector_comparison.csv"), encoding="utf-8")))
    frozen = json.load(open(os.path.join(OUT, "extraction_config.json")))
    det = json.load(open(os.path.join(WORK, "determinism.json")))
    raw_hash = json.load(open(os.path.join(WORK, "raw_array_hashes.json")))
    f = lambda r, k: float(r[k]) if r[k] != "" else float("nan")
    metrics = [r for r in conc if r["member_role"] == "metrics" and r["set"] in ("DEV-S", "LOCK-S", "LOCK-R")]
    partners = [r for r in conc if r["member_role"] == "partner"]
    L = []
    for s in ("DEV-S", "LOCK-S", "LOCK-R"):
        rs = [r for r in metrics if r["set"] == s]
        L.append((s, rs))
    # unit tests
    suite = unittest.defaultTestLoader.discover(os.path.join(ROOT, "tests"), pattern="test_rr_concepts.py")
    buf = io.StringIO()
    tres = unittest.TextTestRunner(stream=buf, verbosity=0).run(suite)
    tests = {"run": tres.testsRun, "failures": len(tres.failures), "errors": len(tres.errors)}
    # duplicate identity
    by_unit = defaultdict(dict)
    for r in conc:
        by_unit[(r["set"], r["unit_id"])][r["member_role"]] = r
    pairs = [(v["metrics"], v["partner"]) for v in by_unit.values() if "partner" in v]
    keys = [k for k in CONCEPT_COLS if k not in ("record_id", "unit_id", "member_role", "runtime_s", "selection")]
    ident = sum(all(a[k] == b[k] for k in keys) and raw_hash[a["record_id"]] == raw_hash[b["record_id"]] for a, b in pairs)
    # states
    def counts(rs, k):
        return dict(Counter(r[k] for r in rs))
    summ = {"tests": tests, "determinism": det, "duplicate_pairs_read": len(pairs), "duplicate_pairs_identical_outputs_and_raw_array": ident,
            "read_failures": sum(r["read_ok"] != "True" for r in conc), "structural_failures": sum(r["structural_ok"] != "True" for r in conc),
            "sets": {}}
    for s, rs in L:
        reasons = Counter(c for r in rs for c in r["reason_codes"].split(";") if c)
        summ["sets"][s] = {"units": len(rs), "rate_state": counts(rs, "rate_state"), "irr_state": counts(rs, "irr_state"),
                           "reasons": dict(reasons), "runtime_s_per_record": q([f(r, "runtime_s") for r in rs]),
                           "n_beats_cons": q([f(r, "n_beats_cons") for r in rs]),
                           "rate_mean_bpm_cons_all": q([f(r, "rate_mean_bpm_cons") for r in rs]),
                           "accepted_rate_mean_bpm": q([f(r, "accepted_rate_mean_bpm") for r in rs]),
                           "accepted_rr_cv": q([f(r, "accepted_rr_cv") for r in rs]),
                           "accepted_rr_nrmssd": q([f(r, "accepted_rr_nrmssd") for r in rs]),
                           "accepted_rr_mad_succ": q([f(r, "accepted_rr_mad_succ") for r in rs])}
    cm = [r for r in cmp_ if r["member_role"] == "metrics"]
    for s, _ in L:
        cs = [r for r in cm if r["set"] == s]
        both = [r for r in cs if f(r, "dice_agreement") == f(r, "dice_agreement")]
        summ["sets"][s]["agreement"] = {"n_with_detections": len(both), "dice_q": q([f(r, "dice_agreement") for r in cs]),
                                       "perfect_agreement_frac": float(np.mean([r["n_unmatched_A"] == "0" and r["n_unmatched_B"] == "0" for r in cs])) if cs else None,
                                       "abs_rate_diff_q": q([f(r, "abs_rate_mean_diff_bpm") for r in cs]), "median_abs_dt_ms_q": q([f(r, "match_median_abs_dt_ms") for r in cs]),
                                       "abs_nrmssd_diff_q": q([f(r, "abs_rr_nrmssd_diff") for r in cs])}
    # coverage by stratum / batch / label (aggregate only)
    cov = defaultdict(lambda: Counter())
    for r in metrics:
        m = man[(r["set"], r["record_id"])]
        for key in (("stratum", r["set"] + ":" + r["stratum"]), ("id_block", m["id_block"]), ("ts_flag", "ts=" + m["ts_flag"]), ("age_missing", "age_missing=" + m["age_missing_flag"])):
            cov[key][r["rate_state"]] += 1
            cov[key]["irr_" + r["irr_state"]] += 1
            cov[key]["n"] += 1
    summ["coverage"] = {f"{k[0]}|{k[1]}": dict(v) for k, v in sorted(cov.items())}
    # failure examples (IDs only)
    ex = defaultdict(list)
    for r in metrics:
        for c in r["reason_codes"].split(";"):
            if c and len(ex[c]) < 5:
                ex[c].append(r["record_id"])
    summ["failure_examples"] = dict(ex)
    # NaN-as-zero guard: every UNAVAILABLE value must be empty
    bad_zero = sum(1 for r in conc if r["rate_state"] == "UNAVAILABLE" and r["accepted_rate_mean_bpm"] not in ("",))
    summ["accepted_values_with_unavailable_state"] = bad_zero
    checks = {"unit_tests_pass": tests["failures"] == 0 and tests["errors"] == 0,
              "no_read_failures": summ["read_failures"] == 0, "determinism_identical": bool(det["identical"]),
              "duplicate_identity_all": ident == len(pairs), "no_value_with_unavailable_state": bad_zero == 0}
    summ["engineering_checks"] = checks
    summ["verdict"] = "ENGINEERING READY" if all(checks.values()) else "NEEDS FIX"
    json.dump(summ, open(os.path.join(WORK, "pilot_summary.json"), "w"), indent=1)
    log("report summary written; verdict", summ["verdict"], checks)


if __name__ == "__main__":
    {"sample": stage_sample, "develop": stage_develop, "freeze": stage_freeze, "lock": stage_lock, "report": stage_report}[sys.argv[1]]()
