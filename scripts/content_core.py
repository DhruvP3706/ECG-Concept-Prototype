"""Torch-free worker for Experiment 9 (atrial-content substitution). One call = one recipient record: builds every condition, applies the quality-control rules, returns accepted preprocessed signals."""
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import atrial_core as c0  # noqa: E402  (Experiment 8 constants, timing check)
from ecg_concept import atrial_content as ac  # noqa: E402
from ecg_concept import atrial_ops as ao  # noqa: E402
from ecg_concept import rr_warp as rw  # noqa: E402
from ecg_concept.preprocessing import preprocess, read_raw_mat  # noqa: E402

SEED = c0.SEED
SCALE_COMPAT = (0.25, 4.0)                                 # median over leads of donor/recipient region RMS must lie in this range
LEADS = {"II": [1], "V1": [6], "limb": [0, 1, 2, 3, 4, 5], "chest": [6, 7, 8, 9, 10, 11]}
BANK = None


def init(bank_dir):
    global BANK
    BANK = ac.Bank(bank_dir)


def mask_of(name):
    if name is None:
        return None
    m = np.zeros(12, bool)
    m[LEADS[name]] = True
    return m


def specs(group):
    S = []

    def add(name, a1, kind, **kw):
        S.append((name, a1, kind, kw))
    add("flat", 160, "flat"); add("rev", 160, "rev"); add("shuf", 160, "shuf")
    for real in ("flat", "rev", "shuf"):
        for t in "ab":
            add(f"sh_{real}_{t}", 160, "sham", real=real)
    for k in range(5):
        add(f"dAF{k}", 160, "donor", cls="AF", k=k)
    for k in range(3):
        add(f"dAFsc{k}", 160, "donor", cls="AF", k=k, scale=True)
    for k in range(5):
        add(f"shAF{k}", 160, "sham", real=f"dAF{k}")
    for k in range(2):
        add(f"rawAF{k}", 160, "raw", cls="AF", k=k)
    for k in range(3):
        add(f"dRND{k}", 160, "donor", cls="RND", k=k)
    for k in range(3):
        add(f"dSR{k}", 160, "donor", cls="SR", k=k)
    for k in range(3):
        add(f"shSR{k}", 160, "sham", real=f"dSR{k}")
    add("flat_s", 120, "flat"); add("flat_l", 200, "flat")
    for k in range(2):
        add(f"dAF{k}_s", 120, "donor", cls="AF", k=k); add(f"dAF{k}_l", 200, "donor", cls="AF", k=k)
    add("shAF0_s", 120, "sham", real="dAF0_s"); add("shAF0_l", 200, "sham", real="dAF0_l")
    for ln in LEADS:
        add(f"flat_{ln}", 160, "flat", mask=ln); add(f"dAF0_{ln}", 160, "donor", cls="AF", k=0, mask=ln); add(f"shAF0_{ln}", 160, "sham", real=f"dAF0_{ln}", mask=ln)
    return S


FACT = ["rr_only", "rr_flat", "rr_dAF0", "rr_dAF1", "rr_dAF2", "rr_shAF0"]


def all_names(group):
    return [s[0] for s in specs(group)] + (FACT if group == "SR" else [])


def work(task):
    rid, group, path, beats, cv0, ridx, donors, keep_raw = task
    x0 = read_raw_mat(path)
    b = np.array(beats, int)
    base = {"record_id": rid, "group": group, "cv0": cv0, "mean_rr": float(np.diff(b).mean()), "n_beats": len(b)}
    items = [dict(base, condition="orig", accepted=True, reason="", x=preprocess(x0))]
    if keep_raw:
        items[0]["raw_orig"] = x0
    names = all_names(group)

    def rng(name):
        return np.random.RandomState((SEED + 1000003 * ridx + 31 * (zlib.crc32(name.encode()) % 1000003 + 100)) % (2 ** 32))

    def reject(name, why, regs=()):
        return dict(base, condition=name, accepted=False, reason=why, n_eligible=len(regs))

    if len(ao.regions(b, 160)[0]) < c0.MIN_ELIG:
        return items + [reject(n, "insufficient_eligible_beats") for n in names]
    cache = {}
    reg_cache = {}

    def regions(a1):
        if a1 not in reg_cache:
            reg_cache[a1] = ao.regions(b, a1)[0]
        return reg_cache[a1]

    def finish(name, y_int, regs, bts, xref, mask, extra=None, ref_rms=None, ref_y=None, check_diff=True):
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

    def donor_for(cls, k):
        lst = donors.get(cls, [])
        return lst[k] if k < len(lst) else None

    def make(name, a1, kind, kw, xref=None, bts=None, regs=None, tag=None):
        xref = x0 if xref is None else xref
        bts = b if bts is None else bts
        regs = regions(a1) if regs is None else regs
        if len(regs) < c0.MIN_ELIG:
            return reject(name, "insufficient_eligible_beats", regs)
        mask = mask_of(kw.get("mask"))
        old = ac.region_devs(xref, regs)
        extra, ref_rms, ref_y, y = {}, None, None, None
        if kind == "flat":
            new = [np.zeros_like(d) for d in old]
        elif kind == "rev":
            new = ac.reverse_devs(old)
        elif kind == "shuf":
            new = ac.shuffle_devs(old, rng(name))
        elif kind in ("donor", "raw"):
            dd = donor_for(kw["cls"], kw["k"])
            if dd is None:
                return reject(name, "no_donor", regs)
            did, dist = dd
            extra = {"donor": did, "donor_dist": dist}
            off = (ridx * 7 + kw["k"] * 3) % 9973
            if kind == "raw":
                rs = BANK.raw160(did)
                if rs is None:
                    return reject(name, "donor_no_region", regs)
                segs = [np.asarray(rs[(off + i) % len(rs)], float) for i in range(len(regs))]
                y = ac.raw_substitute(xref, regs, segs, mask)
                new = None
            else:
                bd = BANK.devs(did, a1)
                if bd is None:
                    return reject(name, "donor_no_region", regs)
                dev = ac.donor_devs(bd, len(regs), off % len(bd))
                rr_, rd_ = ac.rms_lead(old), ac.rms_lead([np.asarray(d) for d in dev])
                extra["scale_ratio"] = float(np.median(rd_ / (rr_ + 1e-6)))
                if kw.get("scale"):
                    sc = np.clip(rr_ / (rd_ + 1e-6), 0.2, 5.0)
                    dev = [d * sc[:, None] for d in dev]
                new = dev
            if kind == "raw":
                rr_ = ac.rms_lead(old)
                extra["scale_ratio"] = float(np.median(ac.rms_lead([np.asarray(s, float) - ao._line(np.asarray(s, float)) for s in segs]) / (rr_ + 1e-6)))
        elif kind == "sham":
            real = cache.get(kw["real"])
            if real is None:
                return reject(name, "real_unavailable", regs)
            new = ac.sham_devs(real["old"], real["new"], rng(name))
            mask = real["mask"]
            ref_rms, ref_y = real["rms"], real["y"]
            regs = real["regs"]
            extra = {"donor": real["donor"], "donor_dist": real["donor_dist"]} if real.get("donor") else {}
        else:
            raise ValueError(kind)
        if y is None:
            y = ac.with_devs(xref, regs, new, mask)
        y_int = ao.to_int16(y)
        it = finish(name, y_int, regs, bts, xref, mask, extra, ref_rms, ref_y)
        if kind != "sham":
            cache[name] = {"old": old, "new": new, "mask": mask, "rms": it["rms_modification"], "y": y_int, "regs": regs, "donor": extra.get("donor"), "donor_dist": extra.get("donor_dist")}
        return it

    for name, a1, kind, kw in specs(group):
        items.append(make(name, a1, kind, kw))

    if group == "SR":                                           # factorial: RR-CV 0.20 (iid, rigid atrial region) then atrial manipulation. Same RNG as Experiment 8 -> rr_only/rr_flat must reproduce it.
        rg = np.random.RandomState(SEED + 1000003 * ridx + 31 * 21)
        nb, tries, why = rw.target_cv_beats(b, c0.FACTORIAL_CV, "iid", rg)
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
                    kw = {}
                    items.append(make("rr_flat", 160, "flat", kw, xref=x2, bts=nb, regs=regs2))
                    items[-1].update(cvx)
                    for k in range(3):
                        items.append(make(f"rr_dAF{k}", 160, "donor", {"cls": "AF", "k": k}, xref=x2, bts=nb, regs=regs2))
                        items[-1].update(cvx)
                    items.append(make("rr_shAF0", 160, "sham", {"real": "rr_dAF0"}, xref=x2, bts=nb, regs=regs2))
                    items[-1].update(cvx)
    return items
