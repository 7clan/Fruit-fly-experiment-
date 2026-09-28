"""Reproducible setup for the Digital Drosophila brain (D1).

Subcommands:
  clone         git-clone the third-party model at the PINNED commit
                (no checkout of newer commits, ever, without a new pin)
  verify        verify the pinned commit + SHA-256 of every tracked file
                against setup/THIRD_PARTY_MANIFEST.json (fails closed)
  env           create brain/.venv and install the pinned
                requirements-brain.txt
  make-manifest regenerate the manifest from the current pinned checkout
                (used once, by the maintainer)

Cross-platform (Windows-compatible): pure subprocess + hashlib.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../brain
MODEL_DIR = ROOT / "third_party" / "Drosophila_brain_model"
SETUP = ROOT / "setup"
MANIFEST = SETUP / "THIRD_PARTY_MANIFEST.json"

REPO_URL = "https://github.com/philshiu/Drosophila_brain_model.git"
PINNED_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"  # pinned 2026-09-28


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def git_head():
    out = run(["git", "-C", str(MODEL_DIR), "rev-parse", "HEAD"]).stdout.strip()
    return out


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_clone(args):
    if MODEL_DIR.exists():
        print(f"[clone] already present at {MODEL_DIR} (HEAD {git_head()[:12]})")
        if git_head() != PINNED_COMMIT:
            print(f"[clone] WARNING: HEAD != pinned {PINNED_COMMIT[:12]}")
        return
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[clone] cloning {REPO_URL} ...")
    run(["git", "clone", REPO_URL, str(MODEL_DIR)])
    run(["git", "-C", str(MODEL_DIR), "checkout", PINNED_COMMIT])
    print(f"[clone] checked out pinned commit {PINNED_COMMIT}")


def cmd_verify(args):
    if not MODEL_DIR.is_dir():
        raise SystemExit("FAIL: model not cloned; run `setup_model.py clone`")
    head = git_head()
    ok_commit = head == PINNED_COMMIT
    print(f"[verify] HEAD        {head}")
    print(f"[verify] pinned      {PINNED_COMMIT}  ->",
          "OK" if ok_commit else "MISMATCH")
    manifest = json.loads(MANIFEST.read_text())
    ok_files, bad = True, []
    for rel, want in sorted(manifest["files_sha256"].items()):
        p = MODEL_DIR / rel
        if not p.is_file():
            ok_files, _ = False, bad.append(f"MISSING {rel}")
            continue
        got = sha256(p)
        if got != want:
            ok_files, _ = False, bad.append(f"CHANGED {rel}")
    for b in bad:
        print("[verify]", b)
    print(f"[verify] files       {len(manifest['files_sha256'])} checked ->",
          "OK" if ok_files else "MISMATCH")
    verdict = ok_commit and ok_files
    print("[verify] VERDICT:", "PASS" if verdict else "FAIL")
    if not verdict:
        sys.exit(1)


def cmd_env(args):
    venv = ROOT / ".venv"
    if not venv.exists():
        print("[env] creating venv ...")
        run([sys.executable, "-m", "venv", str(venv)])
    py = venv / (Path("Scripts") / "python.exe" if sys.platform == "win32"
                 else Path("bin") / "python")
    print("[env] installing pinned requirements ...")
    run([str(py), "-m", "pip", "install", "-r", str(SETUP / "requirements-brain.txt")])
    out = run([str(py), "-c",
               "import brian2,numpy,pandas,pyarrow,joblib;"
               "print(brian2.__version__, numpy.__version__, pandas.__version__,"
               " pyarrow.__version__, joblib.__version__)"]).stdout
    print("[env] installed:", out.strip())


def cmd_make_manifest(args):
    if not MODEL_DIR.is_dir():
        raise SystemExit("model not cloned")
    files = {}
    for p in sorted(MODEL_DIR.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            files[str(p.relative_to(MODEL_DIR))] = sha256(p)
    manifest = {
        "repo": REPO_URL,
        "pinned_commit": git_head(),
        "license": "MIT (Philip Shiu and Nico Spiller)",
        "files_sha256": files,
        "includes_flywire": {
            "v630": ["2023_03_23_completeness_630_final.csv",
                     "2023_03_23_connectivity_630_final.parquet"],
            "v783": ["Completeness_783.csv", "Connectivity_783.parquet"],
        },
        "generated": "by brain/setup/setup_model.py make-manifest",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] {len(files)} files -> {MANIFEST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["clone", "verify", "env", "make-manifest"])
    args = ap.parse_args()
    {"clone": cmd_clone, "verify": cmd_verify,
     "env": cmd_env, "make-manifest": cmd_make_manifest}[args.cmd](args)
