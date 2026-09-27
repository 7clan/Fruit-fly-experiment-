"""Interactive rig calibration for the real-fly apparatus.

Produces `data/rig/calibration.json`:
  * px_per_mm — measured on a printed ruler/reference card lying flat in
    the camera's floor plane (two dot pairs: horizontal AND vertical, so
    camera perpendicularity is checked as scale anisotropy);
  * arena     — circle fitted to >= 3 clicks on the arena wall's INNER
    edge (center_px + radius_px);
  * geometry  — validated against the pre-registered software geometry
    (arena >= 600 px across; marker boxes outside the ROI; radius vs the
    measured physical inner diameter).

The card stays mounted beside the arena as a permanent fixture, enabling a
quick between-day drift re-check (`--recheck-scale`).

Non-interactive helpers (pure math, unit-testable) are separated from the
OpenCV GUI parts, which run on the operator's machine.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from ..rig import (RIG_DIR, load_calibration, record_pass, save_calibration,
                   utc_now_iso)
from .apparatus import geometry_report

RULER_DEFAULT_MM = 50.0


# ------------------------------------------------------------- pure math
def distance_px(p1, p2) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def px_per_mm_from_pair(p1, p2, real_mm: float) -> float:
    if real_mm <= 0:
        raise ValueError("real distance must be > 0")
    return distance_px(p1, p2) / float(real_mm)


def scale_anisotropy(ppm_x: float, ppm_y: float) -> float:
    m = (ppm_x + ppm_y) / 2.0
    return abs(ppm_x - ppm_y) / m if m > 0 else float("inf")


def fit_circle(points) -> tuple[float, float, float, float]:
    """Algebraic (Kasa) least-squares circle fit.

    Returns (cx, cy, r, residual_rms_px). Residual = RMS of
    |sqrt((x-cx)^2+(y-cy)^2) - r| over the clicked points.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("need >= 3 points to fit a circle")
    x, y = pts[:, 0], pts[:, 1]
    # fit x^2 + y^2 = 2a x + 2b y + c  ->  center (a, b)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    s = x ** 2 + y ** 2
    (a, b, c), *_ = np.linalg.lstsq(A, s, rcond=None)
    r = float(np.sqrt(max(c + a * a + b * b, 0.0)))
    resid = np.abs(np.hypot(x - a, y - b) - r)
    return float(a), float(b), r, float(np.sqrt((resid ** 2).mean()))


def calibration_checks(ppm_x: float, ppm_y: float | None,
                       circle: tuple, anisotropy_max: float = 0.02,
                       residual_max_px: float = 3.0) -> dict:
    ok = True
    ppm = ppm_x if ppm_y is None else (ppm_x + ppm_y) / 2.0
    checks: dict = {"px_per_mm_x": round(ppm_x, 4)}
    if ppm_y is not None:
        checks["px_per_mm_y"] = round(ppm_y, 4)
        ani = scale_anisotropy(ppm_x, ppm_y)
        checks["scale_anisotropy"] = round(ani, 5)
        if ani > anisotropy_max:
            ok = False
            checks["anisotropy_error"] = (
                f"horizontal {ppm_x:.3f} vs vertical {ppm_y:.3f} px/mm "
                f"differ by {ani * 100:.1f}% - camera not perpendicular, "
                "card not flat, or misclicks; redo the scale clicks")
    cx, cy, r, rms = circle
    checks["circle_center_px"] = [round(cx, 1), round(cy, 1)]
    checks["circle_radius_px"] = round(r, 1)
    checks["circle_residual_rms_px"] = round(rms, 2)
    if rms > residual_max_px:
        ok = False
        checks["circle_error"] = (f"arena wall clicks do not lie on a circle "
                                  f"(residual {rms:.1f} px) - click the "
                                  "INNER wall edge at evenly spaced points")
    checks["pass"] = ok
    return checks


# ------------------------------------------------------------------ GUI
def _grab_frame(camera_index: int | None, video: str | Path | None):
    if video is not None:
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read a frame from {video}")
        return frame
    cap = cv2.VideoCapture(camera_index or 0)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {camera_index}")
    import time
    for _ in range(30):                      # let auto settings settle
        cap.read()
        time.sleep(0.02)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not grab a frame")
    return frame


def _collect_clicks(frame, title: str, n: int, prompt: str) -> list[tuple]:
    """Show `frame`; collect exactly n left-clicks. u=undo, r=reset, q=abort."""
    points: list[tuple] = []
    state = {"x": None, "y": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["x"], state["y"] = x, y

    win = title
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    print(prompt)
    while len(points) < n:
        disp = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(disp, p, 5, (0, 0, 255), 2)
            cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(disp, f"{len(points)}/{n} clicks - u undo, r reset, q abort",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            cv2.destroyAllWindows()
            raise RuntimeError("calibration aborted by user")
        if k == ord("u") and points:
            points.pop()
        if k == ord("r"):
            points.clear()
        if state["x"] is not None and k in (13, 32):
            points.append((state["x"], state["y"]))
            state["x"] = state["y"] = None
    cv2.destroyAllWindows()
    return points


def _ask(prompt: str, default: str) -> str:
    try:
        v = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        v = ""
    return v or default


# ---------------------------------------------------------------- main
def interactive_calibrate(camera_index: int | None = 0,
                          video: str | Path | None = None,
                          arena_mm: float | None = None) -> dict:
    """Full calibration: scale (ruler pairs) + arena circle + geometry."""
    frame = _grab_frame(camera_index, video)
    h, w = frame.shape[:2]

    # ---- phase A: scale ---------------------------------------------------
    px = _collect_clicks(
        frame, "scale X - click the two ruler dots along X", 2,
        "Click the RULER dot pair along the X (horizontal) axis, in order.")
    real_x = float(_ask("real distance between the X dots in mm",
                        f"{RULER_DEFAULT_MM:g}"))
    ppm_x = px_per_mm_from_pair(px[0], px[1], real_x)

    ppm_y = None
    if _ask("also click a vertical dot pair for the tilt check? (y/n)",
            "y").lower().startswith("y"):
        py = _collect_clicks(
            frame, "scale Y - click the two ruler dots along Y", 2,
            "Click the RULER dot pair along the Y (vertical) axis.")
        real_y = float(_ask("real distance between the Y dots in mm",
                            f"{RULER_DEFAULT_MM:g}"))
        ppm_y = px_per_mm_from_pair(py[0], py[1], real_y)

    # ---- phase B: arena circle --------------------------------------------
    pts = _collect_clicks(
        frame, "arena - click >= 3 points on the INNER wall edge", 5,
        "Click 5+ points on the INNER edge of the arena wall, spread "
        "around the circle (avoid glare spots).")
    circle = fit_circle(pts)

    checks = calibration_checks(ppm_x, ppm_y, circle)
    ppm = ppm_x if ppm_y is None else (ppm_x + ppm_y) / 2.0
    arena = {"type": "circle", "center_px": [round(circle[0], 1),
                                             round(circle[1], 1)],
             "radius_px": round(circle[2], 1)}
    calib = {"px_per_mm": ppm, "arena": arena,
             "physical_inner_diameter_mm": arena_mm}

    from ..config import load_checks_cfg
    checks_cfg = load_checks_cfg()
    geo = geometry_report(w, h, {**calib, "frame": {}}, checks_cfg)
    calib["checks"] = {**checks, "geometry": geo}
    overall = bool(checks["pass"] and geo.get("pass"))
    calib["pass"] = overall

    print(json.dumps(calib, indent=2, default=str))
    if not overall:
        raise RuntimeError(
            "calibration FAILED the built-in checks (see above) - nothing "
            "was saved. Fix the indicated problem and re-run.")
    save_calibration(ppm, arena, {**checks, "geometry": geo,
                                  "frame": {"size": [w, h]}},
                     physical_id_mm=arena_mm)
    record_pass("calibration", {"px_per_mm": round(ppm, 4),
                                 "arena": arena, "frame": [w, h]},
                checks_cfg)
    print(f"saved -> {RIG_DIR / 'calibration.json'}")
    print("Record this in your lab notebook: px_per_mm="
          f"{ppm:.4f}, center={arena['center_px']}, "
          f"radius={arena['radius_px']} px")
    return calib


def verify_calibration() -> dict:
    """Non-interactive verification of the stored calibration file."""
    calib = load_calibration()
    if calib is None:
        return {"pass": False,
                "error": "no calibration on file - run `calibrate` first"}
    frame_size = calib.get("frame", {}).get("size")
    if not frame_size:
        return {"pass": False, "error": "calibration file lacks frame size "
                                        "(old format) - re-run `calibrate`"}
    geo = geometry_report(int(frame_size[0]), int(frame_size[1]), calib,
                          _load_checks_for_geometry())
    ok = bool(geo.get("pass"))
    return {"pass": ok, "calibration": {
        "px_per_mm": calib.get("px_per_mm"), "arena": calib.get("arena"),
        "created_utc": calib.get("created_utc"),
        "physical_inner_diameter_mm": calib.get(
            "physical_inner_diameter_mm")}, "geometry": geo,
        "file": str(RIG_DIR / "calibration.json")}


def recheck_scale(camera_index: int | None = 0,
                  video: str | Path | None = None,
                  tolerance: float = 0.01) -> dict:
    """Quick between-day drift check: re-measure px_per_mm on the ruler."""
    calib = load_calibration()
    if calib is None:
        return {"pass": False,
                "error": "no stored calibration to compare against"}
    frame = _grab_frame(camera_index, video)
    pts = _collect_clicks(frame, "drift check - click the X ruler dots", 2,
                          "Click the ruler's X dot pair (same card, same "
                          "dots as during calibration).")
    real = float(_ask("real distance in mm", f"{RULER_DEFAULT_MM:g}"))
    ppm_now = px_per_mm_from_pair(pts[0], pts[1], real)
    ppm_ref = float(calib["px_per_mm"])
    dev = abs(ppm_now - ppm_ref) / ppm_ref
    report = {"utc": utc_now_iso(), "px_per_mm_now": round(ppm_now, 4),
              "px_per_mm_stored": round(ppm_ref, 4),
              "relative_deviation": round(dev, 5),
              "tolerance": tolerance,
              "pass": dev <= tolerance,
              "note": "" if dev <= tolerance else
              "camera height/focus changed since calibration - re-run the "
              "full `calibrate` and re-run preflight+blank"}
    RIG_DIR.mkdir(parents=True, exist_ok=True)
    path = RIG_DIR / f"recheck_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_file"] = str(path)
    return report
