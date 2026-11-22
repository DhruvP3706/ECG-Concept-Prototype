"""WFDB-free reader for the ONE PTB-XL record layout verified on the real files (v1.0.3, 500 Hz):
single `.dat` file, 12 signals, format 16 (signed little-endian int16), signals interleaved sample by sample (sample-major),
no skew / samples-per-frame / byte offset, gain 1000 counts/mV, baseline 0, ADC resolution 16, ADC zero 0, units mV, 5000 samples.

Anything else is rejected loudly. Digital samples are returned unchanged: no calibration, filtering, normalisation or mV conversion.
This module knows nothing about folds; the fold-10 guard lives in `ptbxl.load_signal_native`.
"""
import os
import re

import numpy as np

from .preprocessing import LEAD_NAMES

N_SIG, FS, SIG_LEN, GAIN, BASELINE, ADC_RES, ADC_ZERO, UNITS = 12, 500, 5000, 1000.0, 0, 16, 0, "mV"
_REC = re.compile(r"^(\S+) (\d+) (\d+) (\d+)$")                                     # name n_sig fs sig_len (no timestamp in PTB-XL)
_SIG = re.compile(r"^(\S+) (\S+) ([0-9.]+)\((-?\d+)\)/(\S+) (\d+) (-?\d+) (-?\d+) (-?\d+) (\d+) (\S+)$")


class RecordError(ValueError):
    """Raised for any corrupt, unsupported or inconsistent record."""


def _wrap16(x):
    """Signed 16-bit view of an integer (checksums are sums modulo 65536; headers may print them signed or unsigned)."""
    return (int(x) + 32768) % 65536 - 32768


def parse_header(text, name):
    """Parse header text of record `name` (e.g. '00001_hr'). Returns dict(names, fs, sig_len, gain, baseline, units, init, checksum)."""
    try:
        lines = text.splitlines()
    except AttributeError as e:
        raise RecordError("header is not text") from e
    sig_lines = [l for l in lines[1:] if not l.startswith("#") and l.strip()]
    if not lines or any("\x00" in l or not l.isprintable() for l in lines):
        raise RecordError("header is empty or contains non-printable characters (corrupt)")
    m = _REC.match(lines[0])
    if not m:
        raise RecordError(f"unsupported/malformed record line: {lines[0][:60]!r}")
    if m.group(1) != name:
        raise RecordError(f"record name {m.group(1)!r} != expected {name!r}")
    n_sig, fs, sig_len = int(m.group(2)), int(m.group(3)), int(m.group(4))
    if (n_sig, fs, sig_len) != (N_SIG, FS, SIG_LEN):
        raise RecordError(f"unsupported (n_sig, fs, sig_len) = {(n_sig, fs, sig_len)}, only {(N_SIG, FS, SIG_LEN)} is supported")
    if len(sig_lines) != n_sig:
        raise RecordError(f"{len(sig_lines)} signal lines for {n_sig} signals")
    out = {"names": [], "init": [], "checksum": []}
    for l in sig_lines:
        s = _SIG.match(l)
        if not s:
            raise RecordError(f"unsupported/malformed signal line: {l[:70]!r}")                 # also rejects '16x2', '16:5', '16+24', missing fields
        fname, fmt, gain, base, units, res, zero, init, chk, block, desc = s.groups()
        if fname != name + ".dat":
            raise RecordError(f"signal file {fname!r} != {name + '.dat'!r} (single-file records only)")
        if fmt != "16":
            raise RecordError(f"format {fmt!r} unsupported (only plain format 16)")
        if (float(gain), int(base), units, int(res), int(zero), int(block)) != (GAIN, BASELINE, UNITS, ADC_RES, ADC_ZERO, 0):
            raise RecordError(f"gain/baseline/units/adc_res/adc_zero/blocksize = {(gain, base, units, res, zero, block)} unsupported")
        out["names"].append(desc)
        out["init"].append(int(init))
        out["checksum"].append(int(chk))
    out.update(fs=fs, sig_len=sig_len, gain=GAIN, baseline=BASELINE, units=UNITS, fmt=16, adc_res=ADC_RES)
    return out


def decode_samples(raw, header):
    """Bytes -> (T, n_sig) int16 array. Little-endian, sample-major; checks byte count, each channel's initial value and checksum."""
    n, t = len(header["names"]), header["sig_len"]
    if len(raw) != n * t * 2:
        raise RecordError(f"{len(raw)} bytes, expected {n * t * 2} ({t} samples x {n} signals x 2)")
    d = np.frombuffer(raw, dtype="<i2").reshape(t, n)
    for j, name in enumerate(header["names"]):
        col = d[:, j]
        if int(col[0]) != header["init"][j]:
            raise RecordError(f"{name}: initial value {int(col[0])} != header {header['init'][j]}")
        if _wrap16(int(col.astype(np.int64).sum())) != _wrap16(header["checksum"][j]):
            raise RecordError(f"{name}: checksum {_wrap16(int(col.astype(np.int64).sum()))} != header {_wrap16(header['checksum'][j])}")
    return d.astype(np.int16, copy=False)


def read_record(base_path):
    """`base_path` = path without extension. Returns (signal (12, T) int16 in model lead order, header dict). Lead order is by NAME.
    The header gets `original_lead_names`; the (T, n) -> (12, T) reorder reuses `ptbxl.reorder_leads`' name logic."""
    from .ptbxl import reorder_leads
    name = os.path.basename(base_path)
    try:
        with open(base_path + ".hea", "rb") as f:
            text = f.read().decode("ascii")
        header = parse_header(text, name)
        with open(base_path + ".dat", "rb") as f:
            raw = f.read()
        d = decode_samples(raw, header)
        sig = reorder_leads(d, header["names"], tuple(LEAD_NAMES))
    except UnicodeDecodeError as e:
        raise RecordError(f"{name}: header is not ASCII (corrupt)") from e
    except RecordError as e:
        raise RecordError(f"{name}: {e}") from e
    except ValueError as e:                                                     # unexpected/duplicate/missing lead names
        raise RecordError(f"{name}: {e}") from e
    header["original_lead_names"] = header.pop("names")
    return sig, header
