"""Experiment provenance and checkpoint immutability (Phase 1). No Git operations: code identity = SHA-256 of the listed source files."""
import hashlib
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY = os.path.join(ROOT, "docs", "REFERENCE_CHECKPOINTS.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_registry(path=REGISTRY):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_registry(path=REGISTRY, root=ROOT):
    """Raise if any registered reference file is missing or changed. Returns the verified {relpath: sha256}."""
    out = {}
    for rel, sha in load_registry(path)["files"].items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"reference file missing: {rel}")
        got = sha256_file(p)
        if got != sha:
            raise RuntimeError(f"reference file CHANGED: {rel} (expected {sha[:12]}…, got {got[:12]}…)")
        out[rel] = got
    return out


def assert_not_reference_output(out_path, root=ROOT):
    """New checkpoints must never be written onto a registered reference file or into models/pretrained."""
    out = os.path.abspath(out_path)
    if out in {os.path.abspath(os.path.join(root, r)) for r in load_registry()["files"]} or os.path.abspath(os.path.dirname(out)) == os.path.abspath(os.path.join(root, "models", "pretrained")):
        raise PermissionError(f"refusing to write into the reference area: {out_path}")


def write_run_manifest(run_dir, config, seeds, code_files, extra=None):
    """Write run_manifest.json (config, seeds, code hashes, verified reference-checkpoint hashes, environment). Refuses to overwrite."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run_manifest.json")
    if os.path.exists(path):
        raise FileExistsError(path)
    try:
        import torch
        env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception:                                                                    # noqa: BLE001
        env = {}
    man = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": config, "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), "seeds": seeds,
           "code_sha256": {f: sha256_file(os.path.join(ROOT, f)) for f in code_files}, "reference_files_verified": verify_registry(),
           "environment": {"python": sys.version.split()[0], "platform": platform.platform(), **env}, **(extra or {})}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    return man
