"""Rig state, provenance hashing, and shared apparatus files.

The rig directory `phase2/data/rig/` holds machine-written apparatus state:
    calibration.json   camera-scale + arena geometry (from `calibrate`)
    background.png     per-pixel median of the day's passed BLANK recording
                       (the `reference` detector background for fly sessions)
    rig_state.json     PASS references for calibrate/preflight/blank + the
                       config/code hash they were earned under
    preflight_*.json   preflight reports (append-only evidence)
    blank_report_*.json blank-test reports (append-only evidence)

Provenance rule: the composite config hash covers tracking.yaml, gate.yaml,
apparatus_checks.yaml, every flyrec source file, and the git revision. If ANY
of these changes, previously earned preflight/blank PASSes are stale and the
software refuses to record animal sessions until they are re-earned. This is
apparatus-level discipline only — no pre-registered gate threshold lives here.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .config import CONFIG_DIR, DATA_DIR, PHASE2_ROOT

RIG_DIR = DATA_DIR / "rig"
SESSIONS_DIR = DATA_DIR / "sessions"
CALIBRATION_FILE = RIG_DIR / "calibration.json"
BACKGROUND_FILE = RIG_DIR / "background.png"
STATE_FILE = RIG_DIR / "rig_state.json"

LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")


# --------------------------------------------------------------------- time
def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def _parse_utc(s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(s)


def age_hours(stamp_iso: str) -> float:
    """Hours between a UTC ISO stamp and now."""
    dt = _parse_utc(stamp_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds() \
        / 3600.0


# ------------------------------------------------------------------ hashing
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision() -> str:
    """Short rev of HEAD, or 'no-git' (also tagged dirty when tree is dirty)."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PHASE2_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PHASE2_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        if rev:
            return f"{rev}{'-dirty' if dirty else ''}"
    except Exception:
        pass
    return "no-git"


def config_hashes() -> dict:
    """Provenance: config file hashes + flyrec source tree hash + git rev."""
    out = {}
    for name in ("tracking", "gate", "apparatus_checks"):
        p = CONFIG_DIR / f"{name}.yaml"
        out[f"{name}_yaml"] = sha256_file(p) if p.exists() else None
    flyrec = PHASE2_ROOT / "flyrec"
    tree = hashlib.sha256()
    for p in sorted(flyrec.rglob("*.py")):
        tree.update(str(p.relative_to(flyrec)).encode())
        tree.update(b"\0")
        tree.update(hashlib.sha256(p.read_bytes()).digest())
    out["flyrec_tree"] = tree.hexdigest()
    out["git_revision"] = git_revision()
    return out


def composite_config_hash(hashes: dict | None = None) -> str:
    """One digest binding config + code (used to stale-date PASSes)."""
    h = hashes or config_hashes()
    joined = json.dumps(h, sort_keys=True)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


# --------------------------------------------------------------- rig state
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    RIG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def record_pass(kind: str, payload: dict, checks_cfg: dict) -> None:
    """Append a PASS under the current composite hash to rig_state.json."""
    state = load_state()
    state["config_hash"] = composite_config_hash()
    state["config_hashes"] = config_hashes()
    state[kind] = {**payload,
                   "passed_utc": utc_now_iso(),
                   "config_hash": state["config_hash"]}
    save_state(state)


# ------------------------------------------------------------ calibration
def save_calibration(px_per_mm: float, arena: dict, checks: dict,
                     physical_id_mm: float | None = None) -> Path:
    RIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "px_per_mm": float(px_per_mm),
        "arena": arena,
        "created_utc": utc_now_iso(),
        "frame": checks.get("frame", {}),
        "physical_inner_diameter_mm": physical_id_mm,
        "checks": {k: v for k, v in checks.items() if k != "frame"},
        "git_revision": git_revision(),
    }
    CALIBRATION_FILE.write_text(json.dumps(payload, indent=2),
                                encoding="utf-8")
    return CALIBRATION_FILE


def load_calibration() -> dict | None:
    if CALIBRATION_FILE.exists():
        return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    return None


def save_background_median(video_path: Path, out_path: Path | None = None,
                           every_n: int = 5, max_frames: int = 400) -> dict:
    """Per-pixel median of a fly-free (blank) recording -> background.png.

    Subsamples frames (deterministic: every Nth from the start) to bound
    memory; a median over a clean static scene is insensitive to this.
    """
    out_path = out_path or BACKGROUND_FILE
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    stack = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % every_n == 0 and len(stack) < max_frames:
            stack.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        n += 1
    cap.release()
    if not stack:
        raise RuntimeError("no frames read for background")
    bg = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)
    if not cv2.imwrite(str(out_path), bg):
        raise RuntimeError(f"cannot write background image {out_path}")
    return {"file": str(out_path), "sha256": sha256_file(out_path),
            "median_of_frames": len(stack), "video": str(video_path)}


# ------------------------------------------------------------- gating logic
def check_record_readiness(minutes: float, kind: str,
                           checks_cfg: dict) -> tuple[bool, list[str]]:
    """May a recording of this kind be made right now? (fail-closed)

    Returns (ok, problems).
      - baseline/stimulus (animal sessions): need calibration + preflight
        PASS + blank PASS, all fresh and under the current config hash.
      - blank (no-fly apparatus recording): needs calibration + preflight
        PASS (the blank itself is the deeper check).
      - anything else (software validation): not gated here.
    """
    problems: list[str] = []
    max_age = float(checks_cfg.get("general", {}).get("max_age_hours", 24))

    if kind not in ("baseline", "stimulus", "blank"):
        return True, []                      # software validation: not gated

    state = load_state()
    if not state:
        return False, ["no rig state at all - run: calibrate, preflight"
                       + (", blank" if kind != "blank" else "")]
    current = composite_config_hash()
    if state.get("config_hash") != current:
        problems.append(
            "config/code changed since the last preflight/blank PASS "
            f"(state {state.get('config_hash')} != current {current}) - "
            "re-run: preflight" + (", blank" if kind != "blank" else ""))
    if "calibration" not in state:
        problems.append("no rig calibration on file - run: calibrate")
    needed = ["preflight"] if kind == "blank" else ["preflight", "blank"]
    for kind_check in ("preflight", "blank"):
        if kind_check not in needed:
            continue
        entry = state.get(kind_check)
        if not entry:
            problems.append(f"no {kind_check} PASS on record - "
                            f"run: {kind_check}")
            continue
        if age_hours(entry.get("passed_utc", "")) > max_age:
            problems.append(
                f"{kind_check} PASS is older than {max_age:g} h - re-run it")
        if entry.get("config_hash") != current:
            problems.append(f"{kind_check} PASS was earned under a different "
                            "config/code hash - re-run it")
    if kind != "blank" and "blank" in state and not BACKGROUND_FILE.exists():
        problems.append("background.png missing - re-run blank")
    return (not problems), problems


def session_label_ok(label: str, sessions_root: Path | None = None) -> tuple[
        bool, str]:
    """Label must be safe and UNIQUE across existing sessions."""
    if not LABEL_RE.match(label or ""):
        return False, (f"label {label!r} must match [a-z0-9_]+ (max 64 chars)")
    root = sessions_root or SESSIONS_DIR
    if root.exists():
        for d in root.iterdir():
            if d.is_dir() and d.name.endswith(f"_{label}"):
                return False, (f"label {label!r} already used by {d.name} - "
                               "session labels must be unique")
    return True, ""
