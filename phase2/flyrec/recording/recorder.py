"""Live session recorder (runs on the machine attached to the camera).

Writes an immutable session directory:
    data/sessions/<UTC-timestamp>_<label>/
        video.mp4       raw recording (never edited)
        session.json    metadata + stimulus schedule (if any)

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

from ..config import DATA_DIR

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


def record(camera_index: int = 0, minutes: float = 10.0, label: str = "baseline",
           fly_id: str = "", fly_age_days=None, fly_sex: str = "",
           temperature_c=None, humidity_pct=None, px_per_mm=None,
           arena: dict | None = None, stimulus: bool = False,
           session_index: int = 0, out_root: Path | None = None,
           assume_fps: float = 30.0) -> Path:
    """Record one session. Headless-safe (no preview window required)."""
    if not FOURCC_ORDER:
        raise RuntimeError("opencv is required for recording")
    out_root = Path(out_root) if out_root else DATA_DIR / "sessions"
    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    sdir = out_root / f"{ts}_{label}"
    sdir.mkdir(parents=True, exist_ok=False)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {camera_index}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = assume_fps

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

    events = make_stimulus_schedule(session_index) if stimulus else []
    meta = {
        "label": label,
        "kind": "stimulus" if stimulus else "baseline",
        "fly_id": fly_id, "fly_age_days": fly_age_days, "fly_sex": fly_sex,
        "temperature_c": temperature_c, "humidity_pct": humidity_pct,
        "started_utc": ts,
        "fps": float(fps), "width": w, "height": h,
        "duration_s": minutes * 60.0,
        "stimulus_schedule": events,
        "stimulus_seed": (20260927 + session_index) if stimulus else None,
        "calibration": {"px_per_mm": px_per_mm},
        "arena": arena or {},
        "welfare": "non-invasive observation; benign visual stimuli only",
    }
    if px_per_mm is None:
        meta["calibration"] = {"px_per_mm": None,
                               "note": "set px_per_mm before gate evaluation"}

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
            if n % (int(fps) * 30) == 0:
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
