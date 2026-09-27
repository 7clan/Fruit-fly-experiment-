#!/usr/bin/env python3
"""Interactive 4-click arena calibration (webcam).

Usage (on the machine with the camera):
    python scripts/calibrate_arena.py [--camera 0]

A snapshot window opens: click the arena corners in this order:
    1. TOP-LEFT   2. TOP-RIGHT   3. BOTTOM-RIGHT   4. BOTTOM-LEFT
The homography is saved to config/calibration.npz and used automatically
when config/tracking.yaml sets calibration.file to that path.

Also edit tracking.yaml -> source.arena_mm to the physical arena size so
mm/s thresholds are physically meaningful.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from flyagent.fly_tracker.calibration import Calibrator
from flyagent.config import PROJECT_ROOT

PROMPTS = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]


def main(args=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "config" / "calibration.npz"))
    ap.add_argument("--arena-mm", nargs=2, type=float, default=[90, 60],
                    help="physical arena size mm (width height)")
    a = ap.parse_args(args)

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        print(f"cannot open camera {a.camera}")
        return 1
    print("warming up camera ... press SPACE to capture the calibration frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            return 1
        cv2.imshow("calibration - press SPACE to capture", frame)
        if (cv2.waitKey(30) & 0xFF) in (27, ord(" ")):
            break
    cap.release()
    cv2.destroyAllWindows()

    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            print(f"  clicked {PROMPTS[len(pts) - 1]} at ({x}, {y})")

    win = "click 4 corners: TL, TR, BR, BL  -  then press SPACE"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = frame.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(disp, PROMPTS[i], (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if len(pts) == 4:
            cv2.polylines(disp, [np.array(pts, np.int32)], True, (0, 200, 0), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            pts.clear()
        if key == 27:
            return 1
        if key == ord(" ") and len(pts) == 4:
            break
    cv2.destroyAllWindows()

    calib = Calibrator.from_points(np.array(pts, np.float32),
                                   arena_mm=tuple(a.arena_mm))
    calib.save(a.out)
    print(f"saved calibration to {a.out}")
    print(f"set config/tracking.yaml -> calibration.file: {a.out}")
    print(f"and source.arena_mm: [{a.arena_mm[0]}, {a.arena_mm[1]}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
