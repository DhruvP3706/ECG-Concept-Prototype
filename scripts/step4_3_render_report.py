"""Render results/step4_3/pilot_report.md from saved pilot outputs (no new extraction). Refuses to overwrite."""
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "step4_3")
dest = os.path.join(OUT, "pilot_report.md")
assert not os.path.exists(dest)
S = json.load(open(os.path.join(OUT, "work", "pilot_summary.json")))
C = json.load(open(os.path.join(OUT, "extraction_config.json")))
ST = json.load(open(os.path.join(OUT, "work", "stability_dev.json")))
conc = list(csv.DictReader(open(os.path.join(OUT, "pilot_concepts.csv"), encoding="utf-8")))
man = list(csv.DictReader(open(os.path.join(OUT, "pilot_manifest.csv"), encoding="utf-8")))
f = lambda r, k: float(r[k]) if r[k] != "" else float("nan")
SETS = ("DEV-S", "LOCK-S", "LOCK-R")
met = [r for r in conc if r["member_role"] == "metrics" and r["set"] in SETS]
nonacc = [r for r in met if r["rate_state"] != "ACCEPTED"]
samp = C["sampling"]


def fmt(v, nd=2):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def qrow(name, q):
    return f"| {name} | " + " | ".join(fmt(x, 3) for x in q) + " |" if q else f"| {name} | n/a |"


L = []
A = L.append
A("# Step 4.3 — Minimal RR-concept extraction pipeline: engineering pilot report")
A("")
A(f"**Verdict: {S['verdict']}** (all automated engineering checks passed: {', '.join(k for k, v in S['engineering_checks'].items() if v)}).")
A("")
A("This is an **engineering** pilot on **training-role** records. The extractor works and its limits are documented. There are **no reference annotations**, so no sensitivity, PPV, physiological error or accuracy is computed, and a high detector-agreement rate is **not** evidence of clinical or physiological validity. Outputs are *automatically derived* ventricular-rate values and **10-second RR statistics** (not conventional HRV; not expert-validated). The locked sets are engineering assessment subsets inside the training role; they are not the project's validation or test fold.")
A("")
A("## 1. What now runs")
A("")
A("| Component | File |")
A("|---|---|")
A("| Detectors (ADAPT: Pan–Tompkins-*style*; TMA: Elgendi-*style*), per-lead → cross-lead fusion, one-to-one matching | `src/ecg_concept/beat_detection.py` |")
A("| Concept formulas, quality flags/states/reason codes, record interface `extract_record` / `extract_path` / `extract_many` | `src/ecg_concept/rr_concepts.py` |")
A("| 20 engineering tests (synthetic) | `tests/test_rr_concepts.py` |")
A("| Pilot runner (sample → develop → freeze → lock → report) | `scripts/step4_3_pilot.py` |")
A("| DEV-S perturbation diagnostic; report renderer | `scripts/step4_3_stability.py`, `scripts/step4_3_render_report.py` |")
A("")
A("Both detectors are our own SciPy implementations (no ECG package is installed); they are named after the published methods they take inspiration from, **not verified reproductions**. Parameters are in `extraction_config.json`. Concepts: primary rate = 60/mean(RR); secondary = 60/median(RR); RR coefficient of variation (sample SD); normalised RMS of successive differences; median absolute successive RR difference / median(RR); beat/interval counts; states `ACCEPTED/UNCERTAIN/UNAVAILABLE` with reason codes; missing values are empty/NaN (never 0). Nothing but the signal enters a value or mask (interface test).")
A("")
A("Detection preprocessing is independent of the backbone: raw integer signal, per-lead detector band-pass for timing only (zero phase), polarity-agnostic energy detection, |x| localisation. Leads I, II, V1–V6 (8 non-redundant channels, not independent votes). Beats within 100 ms of a record edge are excluded; ≥3 supporting leads (within 50 ms of the cluster's first member, no chaining); detector outputs matched one-to-one (optimal assignment, 50 ms).")
A("")
A("## 2. Exact commands")
A("")
A("```")
for c in (r".venv\Scripts\python.exe -I tests\test_rr_concepts.py",
          r".venv\Scripts\python.exe -I scripts\step4_3_pilot.py sample",
          r".venv\Scripts\python.exe -I scripts\step4_3_pilot.py develop",
          r".venv\Scripts\python.exe -I scripts\step4_3_pilot.py freeze",
          r".venv\Scripts\python.exe -I scripts\step4_3_pilot.py lock",
          r".venv\Scripts\python.exe -I scripts\step4_3_pilot.py report",
          r".venv\Scripts\python.exe -I scripts\step4_3_stability.py",
          r".venv\Scripts\python.exe -I scripts\step4_3_render_report.py"):
    A(c)
A("```")
A("")
A(f"Tests: {S['tests']['run']} run, {S['tests']['failures']} failures, {S['tests']['errors']} errors (first run had 2 failures that exposed a clipping-plateau false positive on smooth baseline minima, fixed before any pilot record was opened; the other was a test-construction bug).")
A("")
A("## 3. Sample counts and processing success")
A("")
rc_ = samp["set_record_counts"]
A("| Set | Type | Units | Records read | Selection |")
A("|---|---|---|---|---|")
sel = {s: Counter(r["selection"] for r in man if r["set"] == s and r["member_role"] == "metrics") for s in SETS}
A(f"| DEV-S | stratified stress | {samp['set_unit_counts']['DEV-S']} | {rc_['DEV-S']} (incl. 1 duplicate partner) | {dict(sel['DEV-S'])} |")
A(f"| LOCK-S | stratified stress (locked) | {samp['set_unit_counts']['LOCK-S']} | {rc_['LOCK-S']} | {dict(sel['LOCK-S'])} |")
A(f"| LOCK-R | representative simple random sample | {samp['set_unit_counts']['LOCK-R']} | {rc_['LOCK-R']} | {dict(sel['LOCK-R'])} |")
A(f"| DUP-CHECK | remaining within-train duplicate pairs, identical-output check only | {S['duplicate_pairs_read'] - 1} pairs | {2 * (S['duplicate_pairs_read'] - 1)} | not sampled units; never in metrics |")
A("")
A(f"Seed {samp['seed']}; units = duplicate groups or records; key = SHA-256(seed|unit); draw order LOCK-R → DEV-S → LOCK-S; sets disjoint (asserted). Lists were written and hashed (`pilot_manifest.csv` SHA-256 `{samp['manifest_sha256'][:16]}…`, list hash `{samp['lists_sha256'][:16]}…`) **before any waveform was read**. **Shortfall:** stratum S12_LVQRSAL has only {samp['eligible_units_per_stratum'].get('S12_LVQRSAL')} eligible training units under the declared priority order (LVQRSAL records overwhelmingly carry an earlier-priority rhythm label); DEV-S received 5 and LOCK-S 0; the remaining quota ({', '.join(f'{k}: {v}' for k, v in samp['shortfalls'].items())}) was filled by the deterministic fallback (next units by hash order from any stratum, marked `fallback`). The stress sets are **not representative**; only LOCK-R supports population-level statements about coverage.")
A("")
A(f"Processing: {len(conc)} records read from training `.mat` files (frozen code), **{S['read_failures']} read failures, {S['structural_failures']} structural failures**; runtime per record (5/25/50/75/95th percentile, s): {S['sets']['LOCK-R']['runtime_s_per_record']} (LOCK-R). The lock stage took 33 s for {len(conc)} records, so a full extraction over the 35,568 training records is roughly 45 minutes single-threaded (arithmetic estimate; not run).")
A("")
A("## 4. Measurement coverage (states are detector-consensus acceptance, not validity)")
A("")
A("| Set | Units | ACCEPTED | UNCERTAIN | UNAVAILABLE | Reasons present (record counts) |")
A("|---|---|---|---|---|---|")
for s in SETS:
    d = S["sets"][s]
    A(f"| {s} | {d['units']} | {d['rate_state'].get('ACCEPTED', 0)} | {d['rate_state'].get('UNCERTAIN', 0)} | {d['rate_state'].get('UNAVAILABLE', 0)} | {d['reasons']} |")
A("")
A("Rate and irregularity states coincide in every record here (irregularity needs ≥3 RR; every record with a rate had ≥6 consensus beats, i.e. ≥5 RR). Coverage by stratum (ACCEPTED/units per set; full table in `work/pilot_summary.json`):")
A("")
cov = defaultdict(dict)
for k, v in S["coverage"].items():
    kind, name = k.split("|", 1)
    if kind == "stratum":
        st, strat = name.split(":", 1)
        cov[strat][st] = f"{v.get('ACCEPTED', 0)}/{v['n']}"
A("| Stratum | DEV-S | LOCK-S | LOCK-R |")
A("|---|---|---|---|")
for strat in sorted(cov):
    A(f"| {strat} | {cov[strat].get('DEV-S', '-')} | {cov[strat].get('LOCK-S', '-')} | {cov[strat].get('LOCK-R', '-')} |")
A("")
A("Batch metadata (aggregate only; all three sets pooled): " + "; ".join(f"{k.split('|')[1]}: ACCEPTED {v.get('ACCEPTED', 0)}/{v['n']}" for k, v in S["coverage"].items() if k.startswith(("id_block", "age_missing"))) + ". No batch or stratum shows a coverage deficit beyond the isolated failures listed below, but the sample is small (descriptive only).")
A("")
A("## 5. Detector agreement (matching tolerance 50 ms; one-to-one assignment)")
A("")
A("| Set | Records with detections | Perfect beat-set agreement | Dice agreement (5/25/50/75/95th pct) | |ΔRate A−B| bpm (pct) | median |Δt| of matched beats, ms |")
A("|---|---|---|---|---|---|")
for s in SETS:
    a = S["sets"][s]["agreement"]
    A(f"| {s} | {a['n_with_detections']} | {a['perfect_agreement_frac']:.3f} | {a['dice_q']} | {a['abs_rate_diff_q']} | {a['median_abs_dt_ms_q']} |")
A("")
A("**Interpretation limits.** Agreement is between two detectors that share the same band-pass family, the same refractory logic and — crucially — the **same peak-localisation step**, so matched beats have identical timing (|Δt| = 0 ms) and only *detection decisions* can differ. Agreement therefore cannot reveal errors common to both (e.g., a T wave or pacemaker spike detected by both, or a low-amplitude beat missed by both), and it says nothing about accuracy against annotations.")
A("")
A("## 6. Concept-value distributions (ACCEPTED values; percentiles 5/25/50/75/95)")
A("")
A("| Set | Metric | p5 | p25 | p50 | p75 | p95 |")
A("|---|---|---|---|---|---|---|")
for s in SETS:
    d = S["sets"][s]
    for name, key in (("beats (consensus)", "n_beats_cons"), ("rate mean, bpm", "accepted_rate_mean_bpm"), ("RR CV", "accepted_rr_cv"), ("RR NRMSSD", "accepted_rr_nrmssd"), ("RR median-abs-succ-diff / median RR", "accepted_rr_mad_succ")):
        A(qrow(f"{s} · {name}", d[key]).replace("| " + f"{s} · {name}", f"| {s} | {name}", 1))
A("")
by = defaultdict(list)
for r in met:
    by[r["stratum"]].append(r)
A("Median per stress stratum (all sampled units, consensus values; shown only to illustrate that outputs vary directionally with the strata used for sampling — **diagnostic labels are not reference values**, no cutoffs are applied):")
A("")
A("| Stratum | n | median rate (bpm) | median RR CV | median beats |")
A("|---|---|---|---|---|")
for s in sorted(by):
    xs = by[s]
    A(f"| {s} | {len(xs)} | {fmt(np.nanmedian([f(r, 'rate_mean_bpm_cons') for r in xs]), 1)} | {fmt(np.nanmedian([f(r, 'rr_cv_cons') for r in xs]), 3)} | {fmt(np.nanmedian([f(r, 'n_beats_cons') for r in xs]), 0)} |")
A("")
A("## 7. Internal stability (DEV-S only; perturbed copies of the parameters; not accuracy)")
A("")
A(f"{ST['n_records']} DEV-S records re-extracted under 16 single-parameter perturbations (lead support, cluster span, edge zone, detector thresholds/windows, localisation window, matching tolerance). Records whose primary rate moved by > 1 bpm, per variant: " + "; ".join(f"{k}: {v['rate_changed_gt_1bpm']}" for k, v in ST["variants"].items() if v["rate_changed_gt_1bpm"]) + f". The largest single change was {max(v['rate_max_abs_change_bpm'] for v in ST['variants'].values()):.1f} bpm (relaxing lead support to 2 leads). Accepted-status changed in 0–4 of {ST['n_records']} records per variant. Values are stable to reasonable parameter changes on this sample; this does not show they are correct.")
A("")
A("## 8. Duplicate-pair output identity")
A("")
A(f"All **{S['duplicate_pairs_read']}** within-train duplicate pairs (1 sampled pair + {S['duplicate_pairs_read'] - 1} read only for this check, counted separately from sampled units) have byte-identical raw arrays and identical extracted outputs ({S['duplicate_pairs_identical_outputs_and_raw_array']}/{S['duplicate_pairs_read']}). Determinism re-run on {S['determinism']['determinism_rerun_records']} DEV-S records: identical = {S['determinism']['identical']}.")
A("")
A("## 9. Actual failures and non-accepted records (IDs and reasons only)")
A("")
A("| Record | Set | Stratum | Rate state | Reason codes | Beats A/B/consensus | Usable/flat leads |")
A("|---|---|---|---|---|---|---|")
for r in sorted(nonacc, key=lambda x: (x["set"], x["record_id"])):
    A(f"| {r['record_id']} | {r['set']} | {r['stratum']} | {r['rate_state']} | {r['reason_codes']} | {r['n_beats_A']}/{r['n_beats_B']}/{r['n_beats_cons']} | {r['n_usable_leads']}/{r['n_flat_leads']} |")
A("")
A(f"Failure counts across the 340 sampled units: **UNAVAILABLE {sum(r['rate_state'] == 'UNAVAILABLE' for r in met)}** (both with only 2 usable non-redundant leads after 6 flat leads — the rule requires ≥3 usable leads and ≥3 supporting leads, so a record with only leads I and II usable is reported unavailable although a human might still read a rate), **UNCERTAIN {sum(r['rate_state'] == 'UNCERTAIN' for r in met)}** (detector disagreement: one detector reports extra or missing beats; values reported separately in `*_A`, `*_B`, `*_cons` columns but not accepted). Detector disagreement is not automatically an error and mean-vs-median rate differences are not treated as errors. Reason-code examples: {S['failure_examples']}.")
A("")
A("## 10. Earlier plan gates — status")
A("")
A("| Gate | Status |")
A("|---|---|")
A("| G1 trusted reference available | **not assessed — reference unavailable** |")
A("| G2 detection sensitivity/PPV vs reference | **not assessed — reference unavailable** |")
A("| G3 rate agreement vs reference | **not assessed — reference unavailable** |")
A("| G4 coverage | descriptive only: ACCEPTED " + ", ".join(f"{s} {S['sets'][s]['rate_state'].get('ACCEPTED', 0)}/{S['sets'][s]['units']}" for s in SETS) + " (acceptance = detector consensus; the gate wording referred to validated states) — **not claimed as a gate pass** |")
A("| G5 mask-leakage control | **not run** (mask-only diagnostic classifier deferred) |")
A("| G6 irregularity vs reference-derived values | **not assessed — reference unavailable** |")
A("| G7 stratum-based morphology marking | **not adopted** (no morphology is marked unreliable from stratum performance) |")
A("")
A("## 11. Limitations (stated, not resolved)")
A("")
for t in ("No reference annotations: nothing here measures accuracy; agreement and stability are internal consistency only.",
          "Both detectors share filtering family, refractory logic and peak localisation; common-mode errors (T-wave double counting, pacemaker spikes, missed low-amplitude beats, very high rates) are invisible to the agreement check. No noise/SQI layer beyond flat, non-finite, clipping indicators and insufficient detections was implemented.",
          "10-second RR statistics are descriptive (roughly 5–30 RR intervals per record in this sample); not conventional HRV, not NN-based, no sinus-rhythm assumption, and not validated for arrhythmia.",
          "Physical calibration (1 µV vs 4.88 µV per count) and raw-vs-denoised provenance remain unresolved and untested; the pilot uses timing only, which is unaffected by a constant gain.",
          "Clipping and flat-lead rules are engineering conventions (mode fraction ≥ 0.8; plateau of ≥ 4 samples at the lead extreme with steep flanks; int16 limits) with no external validation; clipped leads remain in detection and are only flagged.",
          "Sampling: strata assigned by a fixed priority order starve S12_LVQRSAL; fallbacks fill the quota; stress sets are not representative; LOCK-R is 100 units, so rare failure modes may be unobserved (a 3% record-level failure prevalence is detected with ≈95% probability).",
          "Locked sets lie inside the training role, so the extractor was developed on data the backbone was trained on; this is irrelevant for timing extraction but must be remembered when concepts are later paired with backbone embeddings.",
          "A single configuration was used; the stability diagnostic perturbed parameters on DEV-S only."):
    A("- " + t)
A("")
A("## 12. Reuse and next step")
A("")
A("The extractor accepts a record array or `.mat` path and returns concept values, states, reason codes and provenance; `extract_many` is the hook for processing a role manifest. **Use only `accepted_*` values (state ACCEPTED) for later supervision; keep UNCERTAIN values and masks separate; never impute 0.** Suggested next combined step (not started): frozen extraction over the training role with these exact parameters, backbone embeddings, and an initial validation-only concept/prototype experiment, reporting masks and coverage by label and batch and treating the concepts as automatically derived.")
A("")
A("## 13. Fingerprints (replace a git commit; no Git operations were used)")
A("")
A("| Item | SHA-256 |")
A("|---|---|")
A(f"| parameters (`extraction_config.json`) | `{C['parameters_sha256']}` |")
for k, v in C["source_sha256"].items():
    A(f"| `{k}` | `{v}` |")
for k, v in C["input_sha256"].items():
    A(f"| input: {k} | `{v}` |")
A(f"| sampled lists | `{samp['lists_sha256']}` |")
A(f"| `pilot_manifest.csv` | `{samp['manifest_sha256']}` |")
A("")
A(f"Frozen at {C['frozen_at']} (before the lock sets were processed); software: {C['software']}. Execution log: `execution_log.txt`. Beat times (no waveforms) in `pilot_beat_times.csv`. Intermediate files (header flags, DEV run, hashes) in `work/`.")
A("")
A("## 14. Skipped / not checked")
A("")
for t in ("No accuracy, sensitivity, PPV or physiological-error computation (no references).",
          "No axis, amplitude, low voltage, delineation, PR, QT, ST, T-wave or ectopy classification.",
          "No noise-level SQI, no pacemaker-spike handling, no mask-only leakage control, no full-training-set run, no validation/test records read.",
          "Detector implementations were not compared with reference implementations (none installed).",
          "Calibration and raw/denoised provenance not examined."):
    A("- " + t)
open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
with open(os.path.join(OUT, "execution_log.txt"), "a", encoding="utf-8") as lf:
    lf.write(time.strftime("%Y-%m-%d %H:%M:%S ") + "stability diagnostic + report rendered (scripts/step4_3_stability.py, scripts/step4_3_render_report.py)\n")
print("report written", dest)
