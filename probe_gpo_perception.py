#!/usr/bin/env python3
"""Short passive real-GPO perception probe.

Runs only Windows capture + fast perception. No brain, planner, motor
executor, keyboard, or mouse input. Prints the latest structured
WorldObservation so live detector behavior can be checked quickly.
"""

from __future__ import annotations

import json
import time

from lab.bus import Bus
from lab.capture.base import create_windows_capture
from lab.perception.fast_vision import FastVisionWorker


def main() -> int:
    bus = Bus()
    capture = create_windows_capture(
        bus, window_title_re=r"^Roblox$", target_fps=5.0)
    vision = FastVisionWorker(bus, target_hz=5.0, max_detect_width=640)
    obs = bus.state("world.observation")

    print("[gpo-perception] PASSIVE: capture + fast vision only; no input")
    capture.start()
    vision.start()
    try:
        time.sleep(5.0)
        snap = obs.read()
    finally:
        vision.stop()
        capture.stop()

    if snap is None:
        print(json.dumps({"ok": False, "error": "no observation"}, indent=2))
        return 1

    p = snap.payload
    result = {
        "ok": True,
        "player": p.get("player"),
        "target": p.get("target"),
        "enemies": p.get("enemies"),
        "ui": p.get("ui"),
        "notes": p.get("notes"),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
