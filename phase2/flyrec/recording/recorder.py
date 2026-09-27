"""Live session recorder (runs on the machine attached to the camera).

Writes an immutable session directory:
    data/sessions/<UTC-timestamp>_<label>/
        video.mp4       raw recording (never edited, never overwritten)
        session.json    metadata + provenance + stimulus schedule (if any)

Apparatus discipline (PHASE2-APPARATUS-1.0.0, Amendment 2):
  * camera probe before the session dir is created (openable + fps floor);
  * a session with a live fly (baseline/stimulus) is REFUSED unless the
    preflight AND blank checks passed within 24 h under the SAME config/code
    hash (blank needs the preflight); `--force` overrides are recorded as
    deviations in session.json and flagged by intake;
  * every session records full provenance: session_id, config/code hashes,
    git revision, rig calibration reference, preflight/blank PASS refs,
    and (for fly sessions) the reference-background file + hash;
  * labels must be unique across sessions.

The stimulus is OPEN-LOOP: the schedule is generated in advance from a seed;
the operator (or a hardware hook) switches the LED segments following the
printed/console prompts, and the exact timing is read back from the video
margin markers by the tracker (so manual-timing sloppiness is measured, not
trusted). No closed loop, no game, no automation.
"""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import DATA_DIR, load_checks_cfg, load_tracking_cfg
from ..rig import (CALIBRATION_FILE, RIG_DIR, SESSIONS_DIR, check_record_readiness,
                   composite_config_hash, config_hashes, load_calibration,
                   load_state, session_label_ok, sha256_file, utc_now_iso)

try:
    import cv2
    FOURCC_ORDER = ["mp4v", "XVID", "MJPG"]
except ImportError:  # pragma: no cover
    FOURCC_ORDER = []


def make_stimulus_schedule(session_index: int, n_events: int = 10,
                           seed_base: int = 20260927,
                           on_s: float = 20.0, off_s: float = 40.0,
                           start_s: float = 60.0) -> list[dict]:
    """Seeded side sequence (N/E/S/W), fixed on/off timing (pre-registered)."""
    rng = np.random.default_rng(seed_base + session_index)
    sides = [str(s) for s in rng.choice(["N", "E", "S", "W"], size=n_events)]
    events, t = [], start_s
    for s in sides:
        events.append({"t_on": round(t, 2), "t_off": round(t + on_s, 2),
                       "side": s})
        t += on_s + off_s
    return events


def _probe_camera(cap: cv2.VideoCapture, seconds: float = 2.0) -> tuple[int, float]:
    """Count frames for `seconds` on an OPEN capture -> (n, measured_fps)."""
    t0 = time.monotonic()
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
        if time.monotonic() - t0 >= seconds:
            break
    wall = time.monotonic() - t0
    return n, (n / wall if wall > 0 else 0.0)


def _writer_fps(reported: float, measured: float,
                assume: float) -> tuple[float, str]:
    """Pick the container fps honestly (Amendment 2: recording fidelity).

    Uses the camera-reported rate when it is plausible (within 5% of the
    measured probe rate); otherwise the measured rate; else the configured
    fallback. Returns (fps, note).
    """
    if reported and reported >= 1.0:
        if measured > 0 and abs(reported - measured) / reported > 0.05:
            return round(measured, 2), (
                f"camera reported {reported:.1f} fps but the probe measured "
                f"{measured:.1f} fps - using the MEASURED rate so timestamps "
                "stay truthful")
        return float(reported), ""
    if measured > 0:
        return round(measured, 2), (f"camera reported no usable fps; using "
                                    f"the measured probe rate "
                                    f"{measured:.1f} fps")
    return float(assume), f"using configured assume_fps={assume}"


def record(camera_index: int = 0, minutes: float = 10.0, label: str = "baseline",
           fly_id: str = "", fly_age_days=None, fly_sex: str = "",
           temperature_c=None, humidity_pct=None, px_per_mm=None,
           arena: dict | None = None, stimulus: bool = False,
           session_index: int = 0, out_root: Path | None = None,
           assume_fps: float = 30.0, kind: str | None = None,
           calibration_meta: dict | None = None,
           tracking_cfg: dict | None = None, checks_cfg: dict | None = None,
           force: bool = False, override_reason: str = "",
           operator_notes: str = "") -> Path:
    """Record one session. Headless-safe (no preview window required).

    kind: 'baseline' | 'stimulus' | 'blank' (derived from `stimulus` when
    omitted). baseline/stimulus require fresh preflight+blank PASSes unless
    force=True (recorded as a deviation).
    """
    if not FOURCC_ORDER:
        raise RuntimeError("opencv is required for recording")
    tracking_cfg = tracking_cfg or load_tracking_cfg()
    checks_cfg = checks_cfg or load_checks_cfg()
    out_root = Path(out_root) if out_root else SESSIONS_DIR
    kind = kind or ("stimulus" if stimulus else "baseline")

    # ---- label discipline (unique, filesystem-safe) ----------------------
    ok, msg = session_label_ok(label, out_root)
    if not ok:
        raise RuntimeError(f"invalid label: {msg}")

    # ---- apparatus gating (fail-closed) ----------------------------------
    ok, problems = check_record_readiness(minutes, kind, checks_cfg)
    if not ok and not force:
        raise RuntimeError(
            "APPARATUS GATE - this session is refused:\n  - " +
            "\n  - ".join(problems) +
            "\nRun the required checks (calibrate/preflight/blank) or pass "
            "--force to record anyway (recorded as a protocol deviation).")
    if problems and force:
        print("WARNING - recording with apparatus-gate problems (recorded "
              "as a deviation):")
        for p in problems:
            print(f"  - {p}")

    # ---- calibration (rig store or explicit CLI values) ------------------
    calibration_meta = calibration_meta or load_calibration()
    if px_per_mm is not None:
        calibration_meta = {"px_per_mm": px_per_mm, "arena": arena or {}}
    elif calibration_meta and arena is not None:
        calibration_meta = {**calibration_meta, "arena": arena}
    if not calibration_meta or \
            not calibration_meta.get("px_per_mm") or \
            not calibration_meta.get("arena"):
        raise RuntimeError(
            "no calibration available - run `calibrate` (or pass "
            "--px-per-mm and --arena explicitly). Every session must carry "
            "its calibration.")
    px_per_mm = float(calibration_meta["px_per_mm"])
    arena = dict(calibration_meta["arena"])

    # ---- camera probe (before any files are created) ---------------------
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {camera_index}")
    n_probe, probe_fps = _probe_camera(cap)
    floor = float(checks_cfg.get("preflight", {})
                  .get("probe_fps_floor", 25))
    if probe_fps < floor and not force:
        cap.release()
        raise RuntimeError(
            f"camera probe measured {probe_fps:.1f} fps < floor {floor:.0f} "
            "fps - the pre-registered data requirement is >= 30 fps; fix the "
            "camera (USB bandwidth, exposure settings, resolution) or pass "
            "--force to record anyway (recorded as a deviation)")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    if not reported_fps or reported_fps < 1:
        reported_fps = 0.0
    fps, fps_note = _writer_fps(reported_fps, probe_fps, assume_fps)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    sdir = out_root / f"{ts}_{label}"
    sdir.mkdir(parents=True, exist_ok=False)     # immutable: never overwrite
    session_id = sdir.name

    writer = None
    for f in FOURCC_ORDER:
        fourcc = cv2.VideoWriter_fourcc(*f)
        ext = "mp4" if f == "mp4v" else "avi"
        path = str(sdir / f"video.{ext}")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        cap.release()
        raise RuntimeError("no working video codec found")

    # ---- provenance -------------------------------------------------------
    state = load_state()
    hashes = config_hashes()
    composite = composite_config_hash(hashes)
    calib_ref = None
    if CALIBRATION_FILE.exists():
        calib_ref = {"file": str(CALIBRATION_FILE),
                     "sha256": sha256_file(CALIBRATION_FILE),
                     "px_per_mm": px_per_mm, "arena": arena}
    rig_provenance = {
        "calibration": calib_ref,
        "preflight": state.get("preflight"),
        "blank": state.get("blank"),
    }
    tracking_block = {"mode": "static"}
    if kind in ("baseline", "stimulus"):
        blank = state.get("blank") or {}
        bg_file = blank.get("background")
        if bg_file and Path(bg_file).exists():
            tracking_block = {"mode": "reference", "background_ref": bg_file,
                              "background_ref_sha256":
                                  blank.get("background_sha256")}
        else:
            tracking_block = {"mode": "static",
                              "note": "no blank background available; static "
                                      "mode with the fly present at frame 0 "
                                      "is a known risk (Amendment 2)"}

    events = make_stimulus_schedule(session_index) if stimulus else []
    meta = {
        "session_id": session_id,
        "label": label,
        "kind": kind,
        "fly_id": fly_id, "fly_age_days": fly_age_days, "fly_sex": fly_sex,
        "temperature_c": temperature_c, "humidity_pct": humidity_pct,
        "started_utc": ts,
        "fps": float(fps), "reported_fps": float(reported_fps),
        "measured_probe_fps": round(probe_fps, 2),
        "fps_note": fps_note,
        "width": w, "height": h,
        "duration_s": minutes * 60.0,
        "stimulus_schedule": events,
        "stimulus_seed": (20260927 + session_index) if stimulus else None,
        "calibration": {"px_per_mm": px_per_mm},
        "arena": arena,
        "welfare": "non-invasive observation; benign visual stimuli only",
        "operator_notes": operator_notes,
        "checks_overridden": bool(force and problems),
        "override_reason": override_reason or None,
        "config_hashes": hashes,
        "config_hash": composite,
        "git_revision": hashes.get("git_revision"),
        "rig": rig_provenance,
        "tracking": tracking_block,
    }

    t0 = time.monotonic()
    n = 0
    duration = minutes * 60.0
    print(f"recording {minutes:.1f} min -> {sdir}")
    if events:
        print("Follow the printed stimulus prompts (or wire the LED hook); "
              "exact timing is recovered from the margin markers.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            n += 1
            t = time.monotonic() - t0
            # open-loop stimulus prompts
            for e in events:
                if abs(t - e["t_on"]) < 0.5:
                    print(f"  [stim] {e['side']} ON  (t={t:6.1f}s)")
                if abs(t - e["t_off"]) < 0.5:
                    print(f"  [stim] {e['side']} OFF (t={t:6.1f}s)")
            if n % (int(fps) * 30 if fps >= 1 else 300) == 0:
                print(f"  {t / 60:5.1f} min, {n} frames", flush=True)
            if t >= duration:
                break
    except KeyboardInterrupt:
        print("\nstopped by user (Ctrl-C)")
    finally:
        cap.release()
        writer.release()
        meta["frames_recorded"] = n
        meta["ended_utc"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds")
        (sdir / "session.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"done: {n} frames -> {sdir}")
    return sdir
