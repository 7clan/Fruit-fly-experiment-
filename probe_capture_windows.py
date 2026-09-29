#!/usr/bin/env python3
"""Passive Windows.Graphics.Capture probe for Gate-5 bring-up.

Captures ONLY the single visible window matching "Roblox". The user must
manually confirm that this is the authorized GPO window before running it.
No keyboard/mouse input is emitted and no screenshots are written to disk.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lab.bus import Bus
from lab.capture.windows_graphics_capture import WindowsGraphicsCaptureAdapter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    bus = Bus()
    cap = WindowsGraphicsCaptureAdapter(bus, target_fps=args.fps)
    target = cap.find_target_window()
    print(f"[capture-probe] target={target['title']!r} hwnd={target['handle']}")
    print("[capture-probe] PASSIVE capture only; no input emission")

    ch = cap.frames
    seen = 0
    last_id = -1
    times = []
    first_shape = None
    t0 = time.perf_counter()
    cap.start(target)
    try:
        while time.perf_counter() - t0 < args.seconds:
            env = ch.newest()
            if env is not None:
                p = env.payload
                fid = int(p["frame_id"])
                if fid != last_id:
                    last_id = fid
                    seen += 1
                    times.append(time.perf_counter())
                    if first_shape is None:
                        img = p["data_ref"]
                        first_shape = list(img.shape)
            if not cap.is_running():
                break
            time.sleep(0.005)
    finally:
        cap.stop()

    elapsed = max(time.perf_counter() - t0, 1e-9)
    gaps_ms = [(b-a)*1000.0 for a, b in zip(times, times[1:])]
    result = {
        "ok": bool(seen > 0 and first_shape is not None),
        "target_title": target["title"],
        "frames_seen": seen,
        "elapsed_s": round(elapsed, 3),
        "observed_fps": round(seen / elapsed, 2),
        "frame_shape": first_shape,
        "copy_count": 1,
        "interframe_p50_ms": round(statistics.median(gaps_ms), 2)
            if gaps_ms else None,
        "interframe_p95_ms": round(sorted(gaps_ms)[
            min(len(gaps_ms)-1, int(0.95 * len(gaps_ms)))], 2)
            if gaps_ms else None,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
