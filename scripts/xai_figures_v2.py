"""Corrected figures (v2) regenerated from SAVED results only (no model is run, nothing is recomputed except layout). Fixes found by rendering the SVGs in a browser:
overlapping x tick labels (Experiment 2 lead panels), float-noise tick labels such as 3.5e-09, confidence bands spilling outside axes, text running off the page, excess blank space.
Writes <run>/v2/figures/ for Experiment 3 and Experiment 2; the original figure folders are left untouched.
Run: .venv\\Scripts\\python.exe -I scripts\\xai_figures_v2.py
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ecg_concept.preprocessing import load_ecg  # noqa: E402
import xai_attention_analysis as xa  # noqa: E402
import xai_contrastive_ig as xc  # noqa: E402
import xai_ig_faithfulness as xf  # noqa: E402
import xai_ig_figures as fg  # noqa: E402

C3 = os.path.join(ROOT, "results", "xai_contrastive", "run_20261120_033747")
C2 = os.path.join(ROOT, "results", "xai_attention", "run_20261120_032548")

# ---- monkey-patches (figure layer only)
_orig_axes, _orig_poly, _orig_polygon = fg.axes, fg.Svg.poly, fg.Svg.polygon


def _clip_attr(self):
    return f' clip-path="url(#{self._clip})"' if getattr(self, "_clip", None) else ""


def poly(self, pts, c="#000", sw=1, fill="none", op=1):
    self.add('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="{fill}" stroke="{c}" stroke-width="{sw}" stroke-opacity="{op}" stroke-linejoin="round"{_clip_attr(self)}/>')


def polygon(self, pts, fill, op=0.2):
    self.add('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="{fill}" fill-opacity="{op}" stroke="none"{_clip_attr(self)}/>')


def axes(s, x0, y0, w, h, xlim, ylim, xt, yt, xlabel, ylabel, title=None, xfmt="{:g}", yfmt="{:.2g}"):
    if xlabel == "lead":
        xt = []                                                                             # lead names are drawn by the caller
    yt = np.round(np.asarray(list(yt), float), 2) + 0.0
    cid = f"clip{len(s.el)}"
    s.add(f'<clipPath id="{cid}"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath>')
    r = _orig_axes(s, x0, y0, w, h, xlim, ylim, xt, yt, xlabel, ylabel, title, xfmt, yfmt)
    s._clip = cid
    return r


fg.Svg.poly, fg.Svg.polygon, fg.axes = poly, polygon, axes


def example_figure_v2(path, x, a0, aM, meta):
    W, H = 1400, 1130
    s = fg.Svg(W, H)
    s.text(20, 26, f"{meta['record_id']}: target {meta['class']} vs competitor {meta['competitor']}  (label for {meta['class']}: {'PRESENT' if meta['correct'] else 'ABSENT'})", 15, weight="bold")
    s.text(20, 45, f"{meta['note']}", 11, c="#333")
    s.text(20, 62, f"f_c(x) = {meta['f_c']:.2f}   f_k(x) = {meta['f_k']:.2f}   margin M(x) = {meta['Mx']:.2f}   zero-baseline margin {meta['M0']:.2f}   contrastive IG completeness error {100 * meta['comp_rel']:.1f}%", 11, c="#333")
    L, Rr, top, rowh = 70, 1010, 95, 34
    T = x.shape[1]
    nw = 40
    wM = aM.reshape(12, nw, T // nw).sum((0, 2))
    top4 = np.argsort(-wM)[:4]
    s.text(L, top - 8, "A. Input (filtered, z-scored); shaded = 4 windows with the highest contrastive (margin) attribution", 12, weight="bold")
    for l in range(12):
        yc = top + rowh * (l + 0.5)
        s.line(L, yc, Rr, yc, "#eee", 0.6)
        s.text(L - 8, yc + 4, fg.LEADS[l], 11, "end", weight="bold")
    for wi in top4:
        s.rect(L + wi / nw * (Rr - L), top, (Rr - L) / nw, 12 * rowh, "#D55E00", 0.18)
    n = 1250
    for l in range(12):
        yc = top + rowh * (l + 0.5)
        seg = x[l].astype(float).reshape(n, -1)
        mx, mn = seg.max(1), seg.min(1)
        pts = []
        for i in range(n):
            xx = L + i / n * (Rr - L)
            pts += [(xx, yc - np.clip(mn[i], -4, 4) * rowh * 0.11), (xx, yc - np.clip(mx[i], -4, 4) * rowh * 0.11)]
        s.poly(pts, "#222", 0.8)
    yb = top + 12 * rowh
    for sec in range(11):
        xx = L + sec / 10 * (Rr - L)
        s.line(xx, yb, xx, yb + 4, "#444")
        s.text(xx, yb + 16, f"{sec}", 10, "middle")
    s.text((L + Rr) / 2, yb + 30, "time (s)", 11, "middle")
    b0, bM = a0.reshape(12, 250, 20).sum(2), aM.reshape(12, 250, 20).sum(2)
    vmax = np.percentile(np.abs(np.concatenate([b0.ravel(), bM.ravel()])), 99.5) + 1e-12
    cw = (Rr - L) / 250
    hh = 16
    for pi, (title, blocks) in enumerate(((f"B. Single-class IG for {meta['class']} (logit f_c)", b0), (f"C. Contrastive IG for {meta['class']} vs {meta['competitor']} (margin f_c - f_k)", bM))):
        h0 = yb + 62 + pi * (12 * hh + 40)
        s.text(L, h0 - 6, title + " - red raises, blue lowers; SAME colour scale in B and C", 12, weight="bold")
        for l in range(12):
            s.text(L - 8, h0 + hh * (l + 0.5) + 3, fg.LEADS[l], 9, "end")
            for i in range(250):
                s.rect(L + i * cw, h0 + l * hh, cw + 0.3, hh, fg.div_color(blocks[l, i] / vmax))
        s.rect(L, h0, Rr - L, 12 * hh, "none", stroke="#444")
    yend = yb + 62 + 2 * (12 * hh + 40) - 18
    s.text(L, yend, f"colour scale: +/-{vmax:.3g} logit units per 20-sample block (99.5th percentile over both maps)", 10, c="#333")
    cx0 = 1085
    s.text(cx0, top - 8, "D. Window evidence for the margin", 12, weight="bold")
    pos = np.array([np.maximum(aM[:, w * 125:(w + 1) * 125], 0).sum() for w in range(nw)])
    neg = np.array([np.minimum(aM[:, w * 125:(w + 1) * 125], 0).sum() for w in range(nw)])
    mid, scale = top + 5 * rowh, 5 * rowh * 0.9 / (max(pos.max(), -neg.min()) + 1e-12)
    for w in range(nw):
        xx = cx0 + 6 + w * 7
        s.rect(xx, mid - pos[w] * scale, 5, pos[w] * scale, "#D55E00", 0.9)
        s.rect(xx, mid, 5, -neg[w] * scale, "#0072B2", 0.9)
    s.line(cx0 + 6, mid, cx0 + 6 + nw * 7, mid, "#444", 1)
    for i, t in enumerate(["250 ms windows over 10 s.", "Red (up): evidence for the target over", "the competitor. Blue (down): evidence", "for the competitor.",
                           f"sum positive {pos.sum():.2f}, sum negative {neg.sum():.2f},", f"net {pos.sum() + neg.sum():.2f} (= M(x) - M(0) up to IG error)"]):
        s.text(cx0 + 6, mid + 5 * rowh + 14 + i * 13, t, 9, c="#333")
    ey = yb + 62
    s.text(cx0, ey - 6, "E. Lead sums (grey: single-class; colour: contrastive)", 11, weight="bold")
    ls0, lsM = a0.sum(1), aM.sum(1)
    mxv = max(np.abs(ls0).max(), np.abs(lsM).max()) + 1e-12
    for l in range(12):
        y = ey + 8 + l * 17
        s.text(cx0 + 24, y + 11, fg.LEADS[l], 10, "end")
        for j, (v, col) in enumerate(((ls0[l], "#888"), (lsM[l], "#D55E00" if lsM[l] >= 0 else "#0072B2"))):
            wv = v / mxv * 110
            s.rect(min(cx0 + 150, cx0 + 150 + wv), y + j * 6, abs(wv), 5, col, 0.9)
    s.line(cx0 + 150, ey + 8, cx0 + 150, ey + 8 + 12 * 17, "#444", 1)
    s.text(L, H - 12, "Attribution describes the MODEL's sensitivity along the zero-baseline path; it is not evidence of a physiological cause. No region is labelled with a clinical finding.", 10, c="#555")
    s.save(path)


def main():
    # ---------------- Experiment 3
    z = np.load(os.path.join(C3, "contrastive_attributions.npz"))
    A0 = np.load(os.path.join(xa.PREV, "attributions.npz"))["attr"]
    AM, Mx, M0, comp, flags = z["attr_margin"], z["margin_x"], z["margin_zero"], z["competitor_idx"], z["flagged"]
    rec, cls = z["record_id"], z["cls"]
    names = [str(c) for c in np.load(os.path.join(xf.CACHE, "merged.npz"))["class_names"]]
    pc = list(csv.DictReader(open(os.path.join(C3, "pairs_competitors.csv"))))
    pairs = list(csv.DictReader(open(os.path.join(xa.PREV, "pairs.csv"))))
    cidx = np.array([int(p["class_idx"]) for p in pairs])
    corr = np.array([p["correct"] == "True" for p in pairs])
    crel = np.array([float(r["comp_rel"]) for r in pc])
    LG = np.zeros((len(pc), 45))
    for k, r in enumerate(pc):
        LG[k, cidx[k]], LG[k, comp[k]] = float(r["f_c"]), float(r["f_k"])
    rel = {r["record_id"]: r["rel_path"] for r in csv.DictReader(open(os.path.join(xf.CACHE, "record_index.csv"), encoding="utf-8")) if r["role"] == "validation"}
    X = np.stack([load_ecg(os.path.join(xf.DS, rel[str(r)])) for r in rec])
    curves = {}
    for key in z.files:
        if key.startswith("curve_"):
            g, b, d, o = key[len("curve_"):].split("_", 3)
            curves.setdefault((g, b, d), {})[o] = z[key]
    rows = list(csv.DictReader(open(os.path.join(C3, "gap_by_class.csv"))))
    for r in rows:
        r["raw_gap_vs_random"] = float(r["raw_gap_vs_random"])

    class CL:
        R = len(set(rec.tolist()))

    tr = at_ids = None
    from ecg_concept import attribution as at
    t = at.region_ids("time", 12, 5000, 125)[0]
    sw0 = np.stack([at.region_scores(A0[k].astype(float), t, 40, "signed") for k in range(len(rec))])
    swM = np.stack([at.region_scores(AM[k].astype(float), t, 40, "signed") for k in range(len(rec))])
    pw = np.array([np.corrcoef(sw0[k], swM[k])[0, 1] for k in range(len(rec))])
    xc.example_figure = example_figure_v2
    view = os.path.join(C3, "v2")
    os.makedirs(view)
    xc.figures(view, X, A0, AM, rec, cls, comp, names, pairs, Mx, M0, LG, cidx, crel, flags, corr, pw, curves, None, CL, rows)
    # ---------------- Experiment 2 profile figures (tick overlap fix)
    res = json.load(open(os.path.join(C2, "results.json")))
    zz = np.load(os.path.join(C2, "attention_and_ig.npz"))
    S, cls2, A1 = zz["gates"], zz["cls"], zz["attr_intervened"]
    lead_rid = at.region_ids("lead", 12, 5000, 125)[0]
    sh = lambda A: (lambda ab: ab / ab.sum(1, keepdims=True))(np.stack([at.region_scores(np.abs(A[k]).astype(float), lead_rid, 12, "abs") for k in range(len(A))]))
    view2 = os.path.join(C2, "v2")
    os.makedirs(view2)
    xa.make_figures(view2, res, S, cls2, sh(A0), sh(A1), None, None, None, None)


if __name__ == "__main__":
    main()
