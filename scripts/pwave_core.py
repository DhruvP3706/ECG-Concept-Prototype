"""Torch-free worker for Experiment 10 (P-wave-like patch transplantation into AFIB recipients). One call = one recipient record."""
import json
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import atrial_core as c0  # noqa: E402
from ecg_concept import atrial_content as ac  # noqa: E402
from ecg_concept import atrial_ops as ao  # noqa: E402
from ecg_concept import pwave_ops as pw  # noqa: E402
from ecg_concept import rr_warp as rw  # noqa: E402
from ecg_concept.preprocessing import preprocess, read_raw_mat  # noqa: E402

SEED = c0.SEED
SCALE_COMPAT = (0.1, 5.0)                                  # median over leads of donor-patch RMS / recipient region-deviation RMS
LEADS = {"II": [1], "V1": [6], "limb": [0, 1, 2, 3, 4, 5], "chest": [6, 7, 8, 9, 10, 11]}
OFFS = [-45, -30, -15, 10]
RR_TARGET_CV = 0.03
B = None


def init(bank_dir):
    global B
    B = {"idx": json.load(open(os.path.join(bank_dir, "index.json"))), "meta": json.load(open(os.path.join(bank_dir, "meta.json")))}
    for k in ("full", "p40", "p50", "p30", "ctl"):
        B[k] = np.load(os.path.join(bank_dir, k + ".npy"), mmap_mode="r")


def mask_of(name):
    if name is None:
        return None
    m = np.zeros(12, bool)
    m[LEADS[name]] = True
    return m


def specs():
    S = []

    def add(name, kind, **kw):
        S.append((name, kind, kw))
    add("flat", "flat")
    for k in range(5):
        add(f"Pal{k}", "patch", cls="AF", k=k, off=0, w=40)
    for o in OFFS:
        for k in range(3):
            add(f"Pd{o:+d}_{k}", "patch", cls="AF", k=k, off=o, w=40)
    for k in range(5):
        add(f"shP{k}", "sham_patch", real=f"Pal{k}")
    for k in range(3):
        add(f"shM{k}", "sham", real=f"Pal{k}")
    for k in range(5):
        add(f"Ctl{k}", "control", cls="AF", k=k, off=0, w=40)
    for k in range(5):
        add(f"Full{k}", "full", cls="AF", k=k)
    for k in range(3):
        add(f"shFull{k}", "sham", real=f"Full{k}")
    for k in range(3):
        add(f"Prnd{k}", "patch", cls="RND", k=k, off=0, w=40)
    for k in range(3):
        add(f"PalW{k}", "patch", cls="AF", k=k, off=0, w=50)
        add(f"PalN{k}", "patch", cls="AF", k=k, off=0, w=30)
    for ln in LEADS:
        add(f"flat_{ln}", "flat", mask=ln)
        add(f"Pal0_{ln}", "patch", cls="AF", k=0, off=0, w=40, mask=ln)
        add(f"shP0_{ln}", "sham_patch", real=f"Pal0_{ln}", mask=ln)
    return S


FACT = ["rr_only", "rr_flat", "rr_Pal0", "rr_Pal1", "rr_Pal2", "rr_shP0"]


def all_names():
    return [s[0] for s in specs()] + FACT


def work(task):
    rid, group, path, beats, cv0, ridx, donors, keep_raw = task
    x0 = read_raw_mat(path)
    b = np.array(beats, int)
    base = {"record_id": rid, "group": group, "cv0": cv0, "mean_rr": float(np.diff(b).mean()), "n_beats": len(b)}
    items = [dict(base, condition="orig", accepted=True, reason="", x=preprocess(x0))]
    if keep_raw:
        items[0]["raw_orig"] = x0
    names = all_names()

    def rng(name):
        return np.random.RandomState((SEED + 1000003 * ridx + 31 * (zlib.crc32(name.encode()) % 1000003 + 100)) % (2 ** 32))

    def reject(name, why, regs=()):
        return dict(base, condition=name, accepted=False, reason=why, n_eligible=len(regs))

    if len(ao.regions(b, 160)[0]) < c0.MIN_ELIG:
        return items + [reject(n, "insufficient_eligible_beats") for n in names]
    cache = {}

    def finish(name, y_int, regs, bts, xref, mask, extra=None, ref_rms=None, ref_y=None):
        it = dict(base, condition=name, n_eligible=len(regs))
        if extra:
            it.update(extra)
        q = ao.qc(xref, y_int, bts, regs)
        rms = ac.masked_rms(xref, y_int, regs, mask)
        it.update(rms_modification=rms, max_region_diff_ratio=q["max_region_diff_ratio"], new_clipped_samples=q["new_clipped_samples"])
        why = []
        if not q["qrs_identical"]:
            why.append("qrs_changed")
        if not q["outside_identical"]:
            why.append("outside_changed")
        if not q["shape_dtype_ok"]:
            why.append("shape")
        if q["max_region_diff_ratio"] > c0.DIFF_RATIO_MAX:
            why.append("discontinuity")
        if not q["amplitude_ok"] or q["new_clipped_samples"] > 0:
            why.append("amplitude")
        if not name.startswith("rr_only") and rms < c0.RMS_MIN:
            why.append("no_change")
        if ref_rms is not None and ref_rms > 0:
            it["rms_ratio_to_real"] = rms / ref_rms
            if not (c0.SHAM_RATIO[0] <= rms / ref_rms <= c0.SHAM_RATIO[1]):
                why.append("magnitude_mismatch")
        if ref_y is not None:
            it["spec_l1_to_real"] = ac.band_l1(xref, ref_y, y_int, regs, mask)
        if extra and extra.get("scale_ratio") is not None and not (SCALE_COMPAT[0] <= extra["scale_ratio"] <= SCALE_COMPAT[1]):
            why.append("scale_incompatible")
        ok_t, shift, n_det = c0._timing(y_int, bts)
        it.update(timing_max_shift=shift, timing_n_detected=n_det)
        if not ok_t:
            why.append("timing_changed")
        it.update(accepted=not why, reason=";".join(why))
        if not why:
            it["x"] = preprocess(y_int)
            if keep_raw:
                it["raw_mod"] = y_int
        return it

    def donor_of(cls, k):
        lst = donors.get(cls, [])
        return lst[k] if k < len(lst) else None

    def make(name, kind, kw, xref=None, bts=None, regs=None):
        xref = x0 if xref is None else xref
        bts = b if bts is None else bts
        regs = ao.regions(bts, 160)[0] if regs is None else regs
        if len(regs) < c0.MIN_ELIG:
            return reject(name, "insufficient_eligible_beats", regs)
        mask = mask_of(kw.get("mask"))
        old = ac.region_devs(xref, regs)
        extra, ref_rms, ref_y, start = {}, None, None, None
        if kind == "flat":
            new = [np.zeros_like(d) for d in old]
        elif kind in ("patch", "control", "full"):
            dd = donor_of(kw["cls"], kw["k"])
            if dd is None:
                return reject(name, "no_donor", regs)
            did, dist = dd
            e = B["idx"].get(did)
            if e is None:
                return reject(name, "donor_no_beats", regs)
            lag = B["meta"][did]["lag"]
            n = e[1] - e[0]
            off = (ridx * 7 + kw["k"] * 3) % 9973
            extra = {"donor": did, "donor_dist": dist, "donor_lag": lag, "offset": kw.get("off", 0), "width": kw.get("w", 0), "donor_snr": B["meta"][did]["snr"]}
            if kind == "full":
                new = [np.asarray(B["full"][e[0] + (off + i) % n], float) for i in range(len(regs))]
                ref = [np.asarray(B["p40"][e[0] + (off + i) % n], float) for i in range(len(regs))]
            else:
                w = kw["w"]
                start = pw.placement(lag, kw["off"], w)
                if start is None:
                    return reject(name, "offset_infeasible", regs)
                key = {40: "p40", 50: "p50", 30: "p30"}[w]
                new, ref = [], []
                for i in range(len(regs)):
                    j = e[0] + (off + i) % n
                    pt = np.asarray(B[key][j], float)
                    if kind == "control":
                        p40 = np.asarray(B["p40"][j], float)
                        ct = np.asarray(B["ctl"][j], float)[:, :w]
                        sc = np.clip(np.sqrt(np.mean(p40 ** 2, 1)) / (np.sqrt(np.mean(ct ** 2, 1)) + 1e-6), 0.2, 5.0)
                        pt = ct * sc[:, None]
                    ref.append(pt)
                    new.append(pw.place(pt, start))
            rr_ = ac.rms_lead(old)
            rp_ = np.median(np.stack([np.sqrt(np.mean(p ** 2, axis=1)) for p in ref]), axis=0)
            extra["scale_ratio"] = float(np.median(rp_ / (rr_ + 1e-6)))
        elif kind == "sham_patch":
            real = cache.get(kw["real"])
            if real is None or real["start"] is None:
                return reject(name, "real_unavailable", regs)
            r = rng(name)
            st, w = real["start"], real["w"]
            new = [pw.place(pw.surrogate_patch(d[:, st:st + w], r), st) for d in real["new"]]
            mask, ref_rms, ref_y, regs = real["mask"], real["rms"], real["y"], real["regs"]
            extra = {k: real[k] for k in ("donor", "donor_dist") if real.get(k) is not None}
        elif kind == "sham":
            real = cache.get(kw["real"])
            if real is None:
                return reject(name, "real_unavailable", regs)
            new = ac.sham_devs(real["old"], real["new"], rng(name))
            mask, ref_rms, ref_y, regs = real["mask"], real["rms"], real["y"], real["regs"]
            extra = {k: real[k] for k in ("donor", "donor_dist") if real.get(k) is not None}
        else:
            raise ValueError(kind)
        y_int = ao.to_int16(ac.with_devs(xref, regs, new, mask))
        it = finish(name, y_int, regs, bts, xref, mask, extra, ref_rms, ref_y)
        if kind not in ("sham", "sham_patch"):
            cache[name] = {"old": old, "new": new, "mask": mask, "rms": it["rms_modification"], "y": y_int, "regs": regs, "donor": extra.get("donor"), "donor_dist": extra.get("donor_dist"), "start": start, "w": kw.get("w")}
        return it

    for name, kind, kw in specs():
        items.append(make(name, kind, kw))

    rg = np.random.RandomState(SEED + 1000003 * ridx + 31 * 23)
    nb, tries, why = rw.target_cv_beats(b, RR_TARGET_CV, "iid", rg)
    if nb is None:
        items += [reject(n, "rr_target_" + why) for n in FACT]
    else:
        x2 = rw.to_int16(rw.warp(x0, b, nb, **c0.RIGID))
        qm = rw.quality(x0, x2, b, nb, pre=170, **c0.RIGID)
        W = c0.WARP_ACC
        wok = (qm["qrs_windows_identical"] and not (qm["post_qrs_corr_median"] < W["post"]) and not (qm["pre_qrs_corr_median"] < W["pre"]) and qm["junction_ratio_max"] <= W["junction"] and qm["p999_abs_diff_ratio"] <= W["p999"])
        if not wok:
            items += [reject(n, "rr_warp_quality") for n in FACT]
        else:
            cvx = {"rr_cv_achieved": rw.rr_cv(nb)}
            items.append(finish("rr_only", x2, [], nb, x2, None, cvx))
            regs2 = ao.regions(nb, 160)[0]
            if len(regs2) < c0.MIN_ELIG:
                items += [reject(n, "insufficient_eligible_beats", regs2) for n in FACT[1:]]
            else:
                for nm, kind, kw in (("rr_flat", "flat", {}), ("rr_Pal0", "patch", {"cls": "AF", "k": 0, "off": 0, "w": 40}), ("rr_Pal1", "patch", {"cls": "AF", "k": 1, "off": 0, "w": 40}),
                                     ("rr_Pal2", "patch", {"cls": "AF", "k": 2, "off": 0, "w": 40}), ("rr_shP0", "sham_patch", {"real": "rr_Pal0"})):
                    items.append(make(nm, kind, kw, xref=x2, bts=nb, regs=regs2))
                    items[-1].update(cvx)
    return items
