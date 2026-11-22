"""Persistent PTB-XL fold-10 access guard (Phase 1).

What it does
* Reads the fold-10 file list from the frozen manifest (`benchmarks/ptbxl_v1/manifest_test_reserved.csv`, column `filename_hr`).
* `check(path)` raises PermissionError for any path that names a fold-10 record, unless a valid ONE-SHOT release exists. Every decision (allowed/denied) is appended to a hash-chained JSONL log
  (`results/ptbxl_fold10_access_log.jsonl`); `verify_log()` detects edited/removed lines (not a wholly deleted log).
* `install_hooks()` patches builtins.open, io.open, os.open and numpy.fromfile/memmap in the CURRENT process so that ANY Python code (including libraries) is checked.
* Release needs ALL of: a release file (`results/ptbxl_fold10_release.json`) with approved_by, approval_text, protocol_sha256 (must equal the hash of docs/PROTOCOL_PHASE1.md at that time), config_sha256 and token_sha256;
  the environment variable PTBXL_FOLD10_RELEASE_TOKEN whose SHA-256 equals token_sha256; and no earlier `SESSION_OPENED` event in the log. `open_release_session()` writes SESSION_OPENED (one-shot).

What it cannot do (documented limitation)
* It only protects Python processes that import and install it. Another program (Explorer, PowerShell, a script that does not install the hooks) can still read the files; a deleted log cannot be detected.
* Safer alternatives that need your approval and were NOT applied: (a) move the 2,198 fold-10 record pairs out of the dataset root into a directory with an NTFS deny-read ACL (`icacls ... /deny`),
  (b) keep them in a password-protected archive, (c) store them on a removable drive that is not mounted during development.
"""
import builtins
import csv
import hashlib
import io
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MANIFEST = os.path.join(ROOT, "benchmarks", "ptbxl_v1", "manifest_test_reserved.csv")
DEFAULT_LOG = os.path.join(ROOT, "results", "ptbxl_fold10_access_log.jsonl")
DEFAULT_RELEASE = os.path.join(ROOT, "results", "ptbxl_fold10_release.json")
DEFAULT_PROTOCOL = os.path.join(ROOT, "docs", "PROTOCOL_PHASE1.md")
TOKEN_ENV = "PTBXL_FOLD10_RELEASE_TOKEN"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _key(path):
    """Normalised '<records500>/<dir>/<name>' (extension stripped) or None."""
    p = str(path).replace("\\", "/")
    parts = [x for x in p.split("/") if x]
    if len(parts) < 3:
        return None
    k = "/".join(parts[-3:])
    return os.path.splitext(k)[0].lower() if k.lower().endswith((".dat", ".hea")) else None


class Fold10Guard:
    def __init__(self, manifest=DEFAULT_MANIFEST, log=DEFAULT_LOG, release=DEFAULT_RELEASE, protocol=DEFAULT_PROTOCOL):
        self.log_path, self.release_path, self.protocol_path = log, release, protocol
        with open(manifest, newline="", encoding="utf-8") as f:                       # manifest has no waveform data
            self.keys = {"/".join(r["filename_hr"].replace("\\", "/").split("/")[-3:]).lower() for r in csv.DictReader(f)}
        self._reentry = False

    # ---------------------------------------------------------------- log
    def _lines(self):
        if not os.path.isfile(self.log_path):
            return []
        with _REAL_OPEN(self.log_path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def _append(self, event, **kw):
        lines = self._lines()
        prev = lines[-1]["hash"] if lines else "GENESIS"
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **kw, "prev": prev}
        rec["hash"] = _sha((prev + json.dumps({k: v for k, v in rec.items() if k != "hash"}, sort_keys=True)).encode())
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with _REAL_OPEN(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def verify_log(self):
        prev = "GENESIS"
        for r in self._lines():
            body = {k: v for k, v in r.items() if k != "hash"}
            if r["prev"] != prev or r["hash"] != _sha((prev + json.dumps(body, sort_keys=True)).encode()):
                return False
            prev = r["hash"]
        return True

    def n_allowed_reads(self):
        return sum(1 for r in self._lines() if r["event"] == "READ_ALLOWED")

    # ---------------------------------------------------------------- release
    def _release_valid(self):
        if not os.path.isfile(self.release_path):
            return False, "no release file"
        try:
            rel = json.load(_REAL_OPEN(self.release_path, "r", encoding="utf-8"))
        except Exception as e:                                                          # noqa: BLE001
            return False, f"unreadable release file: {e}"
        for k in ("approved_by", "approval_text", "protocol_sha256", "config_sha256", "token_sha256"):
            if not rel.get(k):
                return False, f"release file lacks {k}"
        if not os.path.isfile(self.protocol_path) or _sha(_REAL_OPEN(self.protocol_path, "rb").read()) != rel["protocol_sha256"]:
            return False, "protocol hash differs from the approved protocol"
        tok = os.environ.get(TOKEN_ENV, "")
        if not tok or _sha(tok.encode()) != rel["token_sha256"]:
            return False, f"environment token {TOKEN_ENV} missing or wrong"
        return True, "ok"

    def open_release_session(self):
        ok, why = self._release_valid()
        if not ok:
            self._append("SESSION_DENIED", reason=why)
            raise PermissionError(f"fold-10 release refused: {why}")
        if any(r["event"] == "SESSION_OPENED" for r in self._lines()):
            self._append("SESSION_DENIED", reason="one-shot release already used")
            raise PermissionError("fold-10 release was already used once")
        self._append("SESSION_OPENED")
        self._session = True

    # ---------------------------------------------------------------- check
    def is_fold10(self, path):
        k = _key(path)
        return k is not None and k in self.keys

    def check(self, path):
        if self._reentry or not self.is_fold10(path):
            return
        self._reentry = True
        try:
            if getattr(self, "_session", False) and self.verify_log():
                self._append("READ_ALLOWED", path=_key(path))
                return
            self._append("READ_DENIED", path=_key(path))
        finally:
            self._reentry = False
        raise PermissionError(f"fold-10 waveform access denied: {_key(path)} (see docs/PROTOCOL_PHASE1.md)")

    # ---------------------------------------------------------------- hooks
    def install_hooks(self):
        g = self

        def guarded_open(file, *a, **k):
            if isinstance(file, (str, bytes, os.PathLike)):
                g.check(os.fsdecode(file))
            return _REAL_OPEN(file, *a, **k)

        def guarded_os_open(path, *a, **k):
            g.check(os.fsdecode(path))
            return _REAL_OS_OPEN(path, *a, **k)

        def guarded_fromfile(file, *a, **k):
            if isinstance(file, (str, os.PathLike)):
                g.check(os.fsdecode(file))
            return _REAL_FROMFILE(file, *a, **k)

        builtins.open = guarded_open
        io.open = guarded_open
        os.open = guarded_os_open
        np.fromfile = guarded_fromfile
        return self

    @staticmethod
    def uninstall_hooks():
        builtins.open, io.open, os.open, np.fromfile = _REAL_OPEN, _REAL_IO_OPEN, _REAL_OS_OPEN, _REAL_FROMFILE


_REAL_OPEN, _REAL_IO_OPEN, _REAL_OS_OPEN, _REAL_FROMFILE = builtins.open, io.open, os.open, np.fromfile


def guarded_read_record(guard, base_path):
    """Convenience wrapper: check the .dat and .hea paths, then use the validated native reader."""
    from .ptbxl_reader import read_record
    guard.check(base_path + ".hea")
    guard.check(base_path + ".dat")
    return read_record(base_path)
