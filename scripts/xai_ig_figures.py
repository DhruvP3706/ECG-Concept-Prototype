"""SVG figures for Experiment 1 (matplotlib is not installed; plain SVG is written directly, no dependency, vector quality).
Reads a results/xai_ig/run_<stamp>/ directory and writes figures into <run>/figures/ (refuses to overwrite).
Example selection rule (fixed in advance, not by attribution pattern): for each of the first 3 classes of the sample the FIRST correct pair in sample order; for the first 3 classes
that have one, the FIRST incorrect pair (model positive, label absent). Regions are shown only as 'high-attribution windows'; no waveform region is given a physiological label.
Run: .venv\\Scripts\\python.exe -I scripts\\xai_ig_figures.py <run_dir>
"""
import csv
import json
import os
import sys

import numpy as np

C = {"pos": "#D55E00", "abs": "#E69F00", "low_abs": "#0072B2", "neg": "#CC79A7", "random": "#555555"}
LBL = {"pos": "highest signed attribution first", "abs": "highest |attribution| first", "low_abs": "lowest |attribution| first (control)", "neg": "most negative first", "random": "random order (mean of 10)"}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.el = w, h, []

    def add(self, s):
        self.el.append(s)

    def line(self, x1, y1, x2, y2, c="#000", sw=1, dash=None, op=1):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{sw}" stroke-opacity="{op}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")

    def rect(self, x, y, w, h, fill, op=1, stroke="none"):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" fill="{fill}" fill-opacity="{op}" stroke="{stroke}"/>')

    def poly(self, pts, c="#000", sw=1, fill="none", op=1):
        self.add('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="{fill}" stroke="{c}" stroke-width="{sw}" stroke-opacity="{op}" stroke-linejoin="round"/>')

    def polygon(self, pts, fill, op=0.2):
        self.add('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="{fill}" fill-opacity="{op}" stroke="none"/>')

    def text(self, x, y, s, size=12, anchor="start", c="#111", weight="normal", rot=None):
        s = str(s).replace("&", "&amp;").replace("<", "&lt;")
        t = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica,Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{c}" font-weight="{weight}"{t}>{s}</text>')

    def save(self, path):
        assert not os.path.exists(path), path
        open(path, "x", encoding="utf-8").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}"><rect width="100%" height="100%" fill="white"/>'
                                                + "".join(self.el) + "</svg>")


def div_color(v):
    v = max(-1.0, min(1.0, v))
    if v >= 0:
        return "#%02x%02x%02x" % (255, int(255 - 190 * v), int(255 - 215 * v))
    v = -v
    return "#%02x%02x%02x" % (int(255 - 215 * v), int(255 - 130 * v), 255)


def seq_color(v):
    v = max(0.0, min(1.0, v))
    return "#%02x%02x%02x" % (int(255 - 200 * v), int(255 - 150 * v), int(255 - 60 * v))


def axes(s, x0, y0, w, h, xlim, ylim, xt, yt, xlabel, ylabel, title=None, xfmt="{:g}", yfmt="{:.2g}"):
    s.rect(x0, y0, w, h, "none", stroke="#444")
    for v in xt:
        x = x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * w
        s.line(x, y0 + h, x, y0 + h + 4, "#444")
        s.text(x, y0 + h + 17, xfmt.format(v), 10, "middle")
    for v in yt:
        y = y0 + h - (v - ylim[0]) / (ylim[1] - ylim[0]) * h
        s.line(x0 - 4, y, x0, y, "#444")
        s.line(x0, y, x0 + w, y, "#ddd", 0.6)
        s.text(x0 - 7, y + 3, yfmt.format(v), 10, "end")
    s.text(x0 + w / 2, y0 + h + 34, xlabel, 11, "middle")
    s.text(x0 - 40, y0 + h / 2, ylabel, 11, "middle", rot=-90)
    if title:
        s.text(x0 + w / 2, y0 - 8, title, 12, "middle", weight="bold")
    return lambda xv, yv: (x0 + (xv - xlim[0]) / (xlim[1] - xlim[0]) * w, y0 + h - (yv - ylim[0]) / (ylim[1] - ylim[0]) * h)


def example_figure(path, x, a, meta, thr_note):
    W, H = 1400, 900
    s = Svg(W, H)
    s.text(20, 28, f"{meta['record_id']}  -  explained class: {meta['class']} (model-predicted; ground-truth label {'PRESENT' if meta['correct'] else 'ABSENT'})", 16, weight="bold")
    p = 1 / (1 + np.exp(-meta["f_x"]))
    s.text(20, 48, f"pre-sigmoid logit {meta['f_x']:.2f} (probability {p:.3f}); zero baseline logit {meta['f_base']:.2f}; IG completeness error {100 * meta['comp_rel']:.2f}% of f(x)-f(baseline). {thr_note}", 11, c="#333")
    L, Rr, top, rowh = 70, 1010, 70, 40
    T = x.shape[1]
    nwin = 40
    wscore = a.reshape(12, nwin, T // nwin).sum((0, 2))
    top4 = np.argsort(-wscore)[:4]
    # A: ECG with highlighted windows
    s.text(L, top - 6, "A. Preprocessed (filtered, z-scored) 12-lead input with the 4 highest-attribution 250 ms windows shaded", 12, weight="bold")
    for l in range(12):
        yc = top + rowh * (l + 0.5)
        s.line(L, yc, Rr, yc, "#eee", 0.6)
        s.text(L - 8, yc + 4, LEADS[l], 11, "end", weight="bold")
    for wi in top4:
        s.rect(L + wi / nwin * (Rr - L), top, (Rr - L) / nwin, 12 * rowh, "#D55E00", 0.18)
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
    # B: heatmap
    h0 = yb + 55
    s.text(L, h0 - 6, "B. Integrated Gradients, signed (red: raises the logit, blue: lowers it), lead x time (20-sample blocks)", 12, weight="bold")
    blocks = a.reshape(12, 250, 20).sum(2)
    vmax = np.percentile(np.abs(blocks), 99.5) + 1e-12
    cw = (Rr - L) / 250
    for l in range(12):
        s.text(L - 8, h0 + rowh * 0.5 * (l + 0.5) + 3, LEADS[l], 9, "end")
        for i in range(250):
            s.rect(L + i * cw, h0 + l * rowh * 0.5, cw + 0.3, rowh * 0.5, div_color(blocks[l, i] / vmax))
    s.rect(L, h0, Rr - L, 12 * rowh * 0.5, "none", stroke="#444")
    s.text(L, h0 + 12 * rowh * 0.5 + 16, f"colour scale: +/-{vmax:.3g} logit units per 20-sample block (99.5th percentile of |block|)", 10, c="#333")
    # C: lead importance
    cx0, cw0 = 1090, 270
    s.text(cx0, top - 6, "C. Lead attribution (sum over time)", 12, weight="bold")
    ls = a.sum(1)
    lab = np.abs(a).sum(1)
    order = np.argsort(-np.abs(ls))
    mxv = np.abs(ls).max() + 1e-12
    for r, l in enumerate(order):
        y = top + r * rowh
        s.text(cx0 + 18, y + 24, LEADS[l], 11, "end", weight="bold")
        mid = cx0 + 40 + (cw0 - 40) / 2
        wv = ls[l] / mxv * (cw0 - 40) / 2
        s.rect(min(mid, mid + wv), y + 10, abs(wv), 18, "#D55E00" if ls[l] >= 0 else "#0072B2", 0.85)
        s.text(cx0 + cw0 + 25, y + 24, f"{ls[l]:+.2f}", 10, "end")
    s.line(cx0 + 40 + (cw0 - 40) / 2, top, cx0 + 40 + (cw0 - 40) / 2, top + 12 * rowh, "#444", 1)
    s.text(cx0 + 40 + (cw0 - 40) / 2, top + 12 * rowh + 16, "signed sum (logit units); ranked by |sum|", 10, "middle")
    s.text(cx0, top + 12 * rowh + 45, f"top lead share of total |IG|: {100 * lab.max() / lab.sum():.0f}% ({LEADS[int(lab.argmax())]})", 11)
    s.text(L, H - 20, "High attribution marks where the MODEL's logit is sensitive along the integration path; it does not establish a physiological cause. Windows are not labelled with any ECG feature.", 10, c="#555")
    s.save(path)


def curve_panel(s, x0, y0, w, h, res, title, direction, ylab):
    fr = np.array(res["fractions"])
    ords = res["orderings"]
    allv = np.concatenate([np.array(ords[o]["mean_curve"]) for o in ords])
    lo, hi = float(min(allv.min(), 0)), float(max(allv.max(), 0))
    pad = (hi - lo) * 0.06 + 1e-9
    f = axes(s, x0, y0, w, h, (0, 1), (lo - pad, hi + pad), [0, 0.25, 0.5, 0.75, 1], np.linspace(lo, hi, 5), "fraction of regions " + ("deleted" if direction == "delete" else "restored"), ylab, title)
    for o in ("random", "low_abs", "neg", "abs", "pos"):
        m, se = np.array(ords[o]["mean_curve"]), np.array(ords[o]["se_curve"])
        s.polygon([f(a, b) for a, b in zip(fr, m + 1.96 * se)] + [f(a, b) for a, b in zip(fr[::-1], (m - 1.96 * se)[::-1])], C[o], 0.15)
        s.poly([f(a, b) for a, b in zip(fr, m)], C[o], 2 if o in ("pos", "random") else 1.3)
    s.line(*f(0, 0), *f(1, 0), c="#888", sw=0.8, dash="3,3")


def main():
    run = os.path.abspath(sys.argv[1])
    out = os.path.join(run, "figures")
    os.makedirs(out)
    res = json.load(open(os.path.join(run, "results.json")))
    cfg = json.load(open(os.path.join(run, "config.json")))
    z = np.load(os.path.join(run, "attributions.npz"))
    A, X = z["attr"], z["x"].astype(np.float32)
    pairs = list(csv.DictReader(open(os.path.join(run, "pairs.csv"))))
    for p in pairs:
        p.update(pair=int(p["pair"]), correct=p["correct"] == "True", f_x=float(p["f_x"]), f_base=float(p["f_base"]), comp_rel=float(p["comp_rel"]))
    classes = cfg["classes"]
    chosen = []
    for c in classes[:3]:
        c_ = [p for p in pairs if p["class"] == c and p["correct"]]
        if c_:
            chosen.append(c_[0])
    seen = []
    for c in classes:
        w_ = [p for p in pairs if p["class"] == c and not p["correct"]]
        if w_ and len(seen) < 3:
            chosen.append(w_[0])
            seen.append(c)
    for p in chosen:
        example_figure(os.path.join(out, f"example_{p['class']}_{'correct' if p['correct'] else 'incorrect'}_{p['record_id']}.svg"), X[p["pair"]], A[p["pair"]].astype(float), p,
                       "Thresholds were selected on this fold.")
    cv = res["curves"]
    # ---- faithfulness curves
    s = Svg(1500, 900)
    s.text(20, 28, "Deletion / insertion of contiguous regions on the model's own logit (mean change over (record, class) pairs, 95% band = 1.96 SE)", 15, weight="bold")
    s.text(20, 48, f"n = {res['n_pairs']} pairs from the frozen validation role; IG zero baseline; deletion replaces regions by the baseline, insertion restores them from a baseline signal", 11, c="#333")
    panels = [("time|zero", "250 ms windows (all leads), zero baseline"), ("cell|zero", "lead x 250 ms cells, zero baseline"), ("time|donor", "250 ms windows, donor-ECG baseline")]
    for j, (k, t) in enumerate(panels):
        curve_panel(s, 90 + j * 480, 100, 380, 300, cv[k + "|delete"], t, "delete", "change in logit from f(x)")
        curve_panel(s, 90 + j * 480, 500, 380, 300, cv[k + "|insert"], "", "insert", "logit above the baseline signal's logit")
    for i, o in enumerate(("pos", "abs", "low_abs", "neg", "random")):
        s.line(130 + i * 280, 862, 160 + i * 280, 862, C[o], 3)
        s.text(166 + i * 280, 866, LBL[o], 10)
    s.text(20, 890, "Deletion: lower = the ordering removes evidence faster. Insertion: higher = the ordering restores evidence faster. Dashed line = no change.", 10, c="#555")
    s.save(os.path.join(out, "faithfulness_curves.svg"))
    # ---- lead-level + class views
    s = Svg(1500, 820)
    s.text(20, 28, "Lead-level behaviour and per-class faithfulness gaps", 15, weight="bold")
    cl = sorted(res["class_lead_share_mean"])
    M = np.array([res["class_lead_share_mean"][c] for c in cl])
    s.text(90, 70, "Mean share of |IG| per lead, by explained class", 12, weight="bold")
    cwid, rh = 40, 30
    for j, ln in enumerate(LEADS):
        s.text(90 + j * cwid + cwid / 2, 90, ln, 10, "middle")
    for i, c in enumerate(cl):
        s.text(80, 100 + i * rh + 19, c, 11, "end")
        for j in range(12):
            s.rect(90 + j * cwid, 100 + i * rh, cwid - 1, rh - 1, seq_color(M[i, j] / max(M.max(), 1e-9)))
            s.text(90 + j * cwid + cwid / 2, 100 + i * rh + 19, f"{M[i, j]:.2f}", 9, "middle", c="#111")
    s.text(90, 100 + len(cl) * rh + 18, f"uniform share would be {1 / 12:.3f}", 10, c="#555")
    lc = cv["lead|zero|delete"]
    curve_panel(s, 640, 100, 380, 300, lc, "single leads removed in IG order (zero baseline)", "delete", "change in logit from f(x)")
    for i, o in enumerate(("pos", "abs", "low_abs", "neg", "random")):
        s.line(640, 450 + i * 18, 670, 450 + i * 18, C[o], 3)
        s.text(676, 454 + i * 18, LBL[o], 10)
    # gaps by class
    rows = list(csv.DictReader(open(os.path.join(run, "gap_by_class.csv"))))
    sel = [r for r in rows if r["gran"] == "time" and r["baseline"] == "zero" and r["direction"] == "delete" and r["ordering"] == "pos" and r["class"] in cl]
    gx0, gy0, gw, gh = 1110, 100, 340, 300
    vals = {r["class"]: float(r["mean_gap_vs_random"]) for r in sel}
    mx = max(vals.values()) * 1.15
    f = axes(s, gx0, gy0, gw, gh, (0, len(cl)), (min(0, min(vals.values())) * 1.2, mx), [], np.linspace(0, mx, 5), "", "AUC gap vs random (logit units)", "Deletion gap by class (time windows)")
    for i, c in enumerate(cl):
        x1, y1 = f(i + 0.15, vals[c])
        x2, y2 = f(i + 0.85, 0)
        s.rect(x1, min(y1, y2), x2 - x1, abs(y1 - y2), C["pos"], 0.85)
        s.text((x1 + x2) / 2, gy0 + gh + 15, c, 9, "middle")
    ll = res["lead_level"]
    ytxt = 520
    for line in [f"Spearman(IG lead score, single-lead removal drop): mean {ll['spearman_IGsigned_vs_single_lead_drop'][0]:.2f}  95% CI [{ll['spearman_IGsigned_vs_single_lead_drop'][1]:.2f}, {ll['spearman_IGsigned_vs_single_lead_drop'][2]:.2f}]",
                 f"IG's top lead = the lead whose removal drops the logit most: {100 * ll['top1_lead_agreement_signed']:.0f}% of pairs (chance {100 * ll['chance_top1']:.0f}%)",
                 f"mean share of |IG| in the single top lead: {100 * ll['mean_top_lead_share_of_abs_IG']:.0f}% (uniform 8%); pairs with a top lead above 30%: {100 * ll['share_pairs_top_lead_over_30pct']:.0f}%",
                 f"removing the best-IG lead alone flips the model's prediction in {100 * ll['share_pairs_where_best_IG_lead_dropped_flips_prediction']:.0f}% of pairs; any single lead flips it in {100 * ll['share_pairs_where_any_single_lead_flips']:.0f}%"]:
        s.text(90, ytxt, line, 12)
        ytxt += 22
    cc = res["concentration"]
    s.text(90, ytxt + 10, f"Temporal concentration: top window holds {100 * cc['mean_top1_window_share_of_40']:.1f}% of |IG| (uniform {100 * cc['uniform_value']:.1f}%); first+last window {100 * cc['mean_edge_share_first_plus_last_window']:.1f}% (uniform {100 * cc['uniform_edge_share']:.1f}%)", 12)
    s.text(90, ytxt + 32, f"Lead-share cosine similarity: within the same class {res['lead_share_cosine_similarity']['within_class_mean']:.3f}, between classes {res['lead_share_cosine_similarity']['between_class_mean']:.3f}", 12)
    s.save(os.path.join(out, "lead_level_and_class_gaps.svg"))
    print(sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
