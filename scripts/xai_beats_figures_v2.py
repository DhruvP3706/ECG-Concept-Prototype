r"""Regenerates the Experiment 5 figures from SAVED results with the corrected example layout (title no longer overlaps the beat markers, no blank page area).
Writes <run>/v2/figures/. Run: .venv\Scripts\python.exe -I scripts\xai_beats_figures_v2.py"""
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ecg_concept.preprocessing import load_ecg  # noqa: E402
import xai_attention_analysis as xa  # noqa: E402
import xai_beat_attribution as xb  # noqa: E402
import xai_ig_faithfulness as xf  # noqa: E402

run = sorted(glob.glob(os.path.join(ROOT, "results", "xai_beats", "run_*")))[-1]
pairs = list(csv.DictReader(open(os.path.join(xa.PREV, "pairs.csv"))))
rec = np.array([p["record_id"] for p in pairs])
cls = np.array([p["class"] for p in pairs])
corr = np.array([p["correct"] == "True" for p in pairs])
A0 = np.load(os.path.join(xa.PREV, "attributions.npz"))["attr"].astype(np.float64)
bj = json.load(open(os.path.join(run, "beats.json")))
beats = {r: np.array(v["beat_samples"], int) for r, v in bj.items()}
ext = {r: {"rate_state": v["rate_state"]} for r, v in bj.items()}
usable = np.array([len(beats[r]) >= xb.MIN_BEATS for r in rec])
accepted = np.array([usable[i] and ext[rec[i]]["rate_state"] == "ACCEPTED" for i in range(len(rec))])
rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(xf.CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
X = np.stack([load_ecg(os.path.join(xf.DS, rel[r])) for r in rec])
res = json.load(open(os.path.join(run, "results.json")))
view = os.path.join(run, "v2")
os.makedirs(view)
xb.figures(view, X, A0, rec, cls, corr, usable, accepted, beats, ext, res, None, None, None, None, None, None, None, None)
