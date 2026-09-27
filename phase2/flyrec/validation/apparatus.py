"""Pre-session APPARATUS checks: PREFLIGHT (camera/environment) and the
BLANK-ARENA TEST (no-fly recording) — PHASE2-APPARATUS-1.0.0.

These run BEFORE any animal session and FAIL CLOSED: the recorder refuses
fly sessions unless both passed within `general.max_age_hours` under the
current config/code hash (see flyrec/rig.py). Thresholds are registered in
`phase2/config/apparatus_checks.yaml` — separate from, and never overriding,
the pre-registered behavioral gate (config/gate.yaml).

Preflight verifies (live ~20 s capture, no fly, calibration card mounted):
  frame rate, exposure/lighting stability, focus, resolution vs calibration,
  background stability, sensor noise, arena/marker geometry.

Blank test verifies (>= 3 min no-fly recording):
  zero confident false detections, no raw foreground motion, lighting
  stability, no static "fly-like" dark features (reflections/smudges/rim
  arcs), sensor noise, and — with the operator blinking each stimulus LED
  once — that the N/E/S/W margin markers are readable from the video.

A PASSED blank also becomes the stored `reference` detector background
(data/rig/background.png) used for all fly sessions that day (Amendment 2).
"""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import (DATA_DIR, RESULTS_DIR, load_checks_cfg,
                      load_tracking_cfg)
from ..rig import (BACKGROUND_FILE, RIG_DIR, SESSIONS_DIR, composite_config_hash,
                   config_hashes, load_calibration, record_pass,
                   save_background_median, sha256_file, utc_now_iso)
from ..tracking.detector import FlyDetector
from ..tracking.tracker import load_tracks, track_video


def _ts() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------
# geometry helpers (mirror tracking.yaml stimulus_markers / tracker ROI math)
# --------------------------------------------------------------------------
def marker_boxes(frame_w: int, frame_h: int, marker_cfg: dict) -> dict:
    """The four marker rectangles {side: (x, y, w, h)} at the frame margins.

    Same math as StimulusMarkerReader (kept in sync with tracking.yaml).
    """
    mw = int(marker_cfg.get("marker_w_px", 30))
    mh = int(marker_cfg.get("marker_h_px", 14))
    margin = int(marker_cfg.get("margin_px", 18))
    cx, cy = frame_w // 2, frame_h // 2
    return {
        "N": (cx - mw // 2, margin, mw, mh),
        "S": (cx - mw // 2, frame_h - margin - mh, mw, mh),
        "W": (margin, cy - mh // 2, mw, mh),
        "E": (frame_w - margin - mw, cy - mh // 2, mw, mh),
    }


def arena_roi_mask(w: int, h: int, arena: dict, inflate: int) -> np.ndarray:
    """Same ROI construction as the tracker (circle/rect + inflate)."""
    roi = np.zeros((h, w), np.uint8)
    cx, cy = arena["center_px"]
    if arena.get("type", "circle") == "circle":
        cv2.circle(roi, (int(round(cx)), int(round(cy))),
                   int(round(arena["radius_px"])) + inflate, 255, -1)
    else:
        cv2.rectangle(roi,
                      (int(cx - arena["half_w_px"] - inflate),
                       int(cy - arena["half_h_px"] - inflate)),
                      (int(cx + arena["half_w_px"] + inflate),
                       int(cy + arena["half_h_px"] + inflate)), 255, -1)
    return roi


def geometry_report(frame_w: int, frame_h: int, calib: dict,
                    checks_cfg: dict) -> dict:
    """Pure geometric checks of the stored calibration vs the frame size."""
    pf = checks_cfg["preflight"]
    arena = calib["arena"]
    px_per_mm = float(calib["px_per_mm"])
    out: dict = {"px_per_mm": px_per_mm, "arena": arena}
    ok, warns = True, []

    lo, hi = pf["px_per_mm_range"]
    if not (lo <= px_per_mm <= hi):
        ok = False
        out["px_per_mm_error"] = f"outside plausible range [{lo}, {hi}]"

    if arena.get("type", "circle") == "circle":
        r = float(arena["radius_px"])
        cx, cy = [float(v) for v in arena["center_px"]]
        diameter_px = 2.0 * r
        out["arena_diameter_px"] = round(diameter_px, 1)
        if diameter_px < pf["arena_diameter_px_min"]:
            ok = False
            out["arena_diameter_error"] = (
                f"arena spans {diameter_px:.0f} px < required "
                f"{pf['arena_diameter_px_min']} px - raise the camera or use "
                "a wider angle")
        clearances = {
            "N": cy - r, "S": frame_h - (cy + r),
            "W": cx - r, "E": frame_w - (cx + r),
        }
        out["rim_clearance_px"] = {k: round(v, 1)
                                   for k, v in clearances.items()}
        cmin = min(clearances.values())
        if cmin < pf["rim_clearance_px_min"]:
            ok = False
            out["rim_clearance_error"] = (
                f"arena rim only {cmin:.0f} px from a frame edge (need >= "
                f"{pf['rim_clearance_px_min']}): the stimulus marker boxes "
                "would overlap the arena ROI - lower the camera magnification "
                "or raise it")
        elif cmin < pf["rim_clearance_px_recommended"]:
            warns.append(f"rim clearance {cmin:.0f} px < recommended "
                         f"{pf['rim_clearance_px_recommended']} px (stimulus "
                         "LEDs will sit close to the arena wall)")
        # marker boxes must lie fully outside the arena ROI
        mcfg = load_tracking_cfg().get("stimulus_markers", {})
        inflate = int(load_tracking_cfg().get("roi_inflate_px", 8))
        r_inf = r + inflate
        boxes = marker_boxes(frame_w, frame_h, mcfg)
        overlap = []
        for side, (bx, by, bw, bh) in boxes.items():
            nearest_x = max(bx - cx, cx - (bx + bw), 0)
            nearest_y = max(by - cy, cy - (by + bh), 0)
            if math_hypot(nearest_x, nearest_y) < r_inf:
                overlap.append(side)
        out["marker_boxes_outside_arena"] = not overlap
        if overlap:
            ok = False
            out["marker_overlap_error"] = (
                f"marker box(es) {overlap} fall inside the arena ROI - the "
                "stimulus readback would be masked; adjust camera geometry")
        # fitted radius vs measured physical arena
        phys = calib.get("physical_inner_diameter_mm")
        if phys:
            err_mm = abs(r / px_per_mm - float(phys) / 2.0)
            out["radius_vs_physical_mm"] = round(err_mm, 2)
            if err_mm > pf["radius_vs_physical_max_mm"]:
                ok = False
                out["radius_error"] = (
                    f"fitted arena radius {r / px_per_mm:.1f} mm differs from "
                    f"measured physical radius {float(phys) / 2:.1f} mm by "
                    f"{err_mm:.1f} mm - redo the calibration clicks or check "
                    "which circle you clicked (inner wall edge)")
        ani = calib.get("checks", {}).get("scale_anisotropy")
        if ani is not None:
            out["scale_anisotropy_at_calibration"] = round(float(ani), 4)
    else:
        out["note"] = "rectangular arena: geometric checks defined for the " \
                      "circular default; check center/half-extents manually"
    out["pass"], out["warnings"] = ok, warns
    return out


def math_hypot(a: float, b: float) -> float:
    return float(np.hypot(a, b))


# --------------------------------------------------------------------------
# PREFLIGHT
# --------------------------------------------------------------------------
def run_preflight(camera_index: int = 0, seconds: float | None = None,
                  checks_cfg: dict | None = None,
                  tracking_cfg: dict | None = None,
                  calibration: dict | None = None,
                  write_state: bool = True) -> dict:
    """Live camera/environment check. Returns the report dict.

    Scene requirements (printed for the operator): no fly, enclosure closed,
    stimulus LEDs OFF, the printed reference card mounted beside the arena.
    """
    checks_cfg = checks_cfg or load_checks_cfg()
    tracking_cfg = tracking_cfg or load_tracking_cfg()
    pf = checks_cfg["preflight"]
    seconds = float(seconds or pf["capture_seconds"])
    calibration = calibration or load_calibration()
    if calibration is None:
        return {"pass": False,
                "error": "no rig calibration on file - run "
                         "`python phase2/run_phase2.py calibrate` first"}

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"pass": False,
                "error": f"cannot open camera {camera_index}"}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    print("PREFLIGHT: no fly in the arena; enclosure CLOSED; stimulus LEDs "
          "OFF; reference card mounted beside the arena.")
    print(f"capturing {seconds:.0f} s from camera {camera_index} "
          f"({w}x{h}) ...")

    geo = geometry_report(w, h, calibration, checks_cfg)
    roi = arena_roi_mask(w, h, calibration["arena"],
                         int(tracking_cfg.get("roi_inflate_px", 8)))
    roi_f = roi.astype(float)
    n_roi = float(roi_f.sum() / 255.0)
    outside = (roi == 0)

    det_cfg = dict(tracking_cfg.get("detector", {}))
    detector = FlyDetector(det_cfg, px_per_mm=float(calibration["px_per_mm"]))

    warmup_s = float(pf.get("warmup_seconds", 3))
    t0 = time.monotonic()
    n = 0
    mean_gray_roi: list[float] = []       # post-warmup frame means in ROI
    lap_out: list[float] = []             # Laplacian var outside ROI (card)
    consec_med: list[float] = []          # median |frame - prev| in ROI
    sum_px = None
    sumsq_px = None
    prev_gray = None
    fg_frames = 0
    detections = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.monotonic() - t0
        if t >= seconds:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = gray.astype(np.float32)
        if sum_px is None:
            sum_px = np.zeros_like(g)
            sumsq_px = np.zeros_like(g)
        x, y, area, conf = detector.detect(frame, roi_mask=roi)
        if x is not None:
            detections += 1
        post_warmup = t >= warmup_s
        if post_warmup:
            mean_gray_roi.append(float((g * roi_f).sum() / (255.0 * n_roi)))
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_out.append(float(lap[outside].var()))
            if prev_gray is not None:
                d = cv2.absdiff(gray, prev_gray)
                consec_med.append(float(
                    (d.astype(np.float32) * roi_f).sum() / (255.0 * n_roi)))
            sum_px += g
            sumsq_px += g * g
        prev_gray = gray
        n += 1
        if n % 120 == 0:
            print(f"  {t:5.1f} s, {n} frames", flush=True)
    cap.release()
    wall_s = time.monotonic() - t0
    fps = n / wall_s if wall_s > 0 else 0.0

    # ------------------------------------------------------------- checks
    checks: dict = {}
    checks["PF1_frame_rate"] = {
        "measured_fps": round(fps, 2), "reported_fps": round(reported_fps, 2),
        "required": pf["min_fps"], "target": pf["target_fps"],
        "pass": fps >= pf["min_fps"],
        "note": "" if fps >= pf["target_fps"]
        else f"below the preferred {pf['target_fps']} fps (acceptable)",
    }
    checks["PF2_resolution_matches_calibration"] = {
        "frame": [w, h],
        "calibration_frame": calibration.get("frame", {}),
        "pass": ([w, h] == list(calibration.get("frame", {}).get("size", []))
                  if calibration.get("frame", {}).get("size") else True),
        "note": "camera resolution changed since calibration" if
        calibration.get("frame", {}).get("size") and
        [w, h] != list(calibration["frame"]["size"]) else "",
    }
    if mean_gray_roi:
        arr = np.array(mean_gray_roi)
        std = float(arr.std())
        half = len(arr) // 2
        drift = float(abs(arr[:max(half, 1)].mean() - arr[-max(half, 1):].mean()))
        checks["PF3_exposure_lighting_stability"] = {
            "mean_gray_std": round(std, 3), "std_max": pf["mean_gray_std_max"],
            "drift": round(drift, 3), "drift_max": pf["mean_gray_drift_max"],
            "pass": std <= pf["mean_gray_std_max"] and
            drift <= pf["mean_gray_drift_max"],
            "note": "lock exposure/gain (auto-exposure pumps the background "
                    "model) and check the enclosure is light-tight",
        }
        checks["PF4_focus"] = {
            "laplacian_var_outside_arena": round(float(np.median(lap_out)), 1),
            "min": pf["laplacian_var_min"],
            "pass": float(np.median(lap_out)) >= pf["laplacian_var_min"],
            "note": "refocus the lens (reference card pattern must be sharp)",
        }
    if consec_med:
        checks["PF5_background_stability"] = {
            "median_consec_diff": round(float(np.median(consec_med)), 3),
            "max": pf["consec_frame_diff_max"],
            "fly_like_blobs_in_roi": detections,
            "max_allowed": pf["max_fg_frames"],
            "pass": float(np.median(consec_med)) <= pf["consec_frame_diff_max"]
            and detections <= pf["max_fg_frames"],
            "note": "unexpected foreground with no fly: glare, condensation, "
                    "moving object, or a sheet/card inside the arena ROI",
        }
    if sum_px is not None and len(mean_gray_roi) > 1:
        k = len(mean_gray_roi)
        var_px = (sumsq_px / k) - (sum_px / k) ** 2
        std_px = np.sqrt(np.maximum(var_px, 0))
        med = float(np.median(std_px[roi > 0]))
        checks["PF6_sensor_noise"] = {
            "median_temporal_std": round(med, 2),
            "fail_above": pf["temporal_noise_median_max"],
            "warn_above": pf["temporal_noise_warn"],
            "pass": med <= pf["temporal_noise_median_max"],
            "warning": med > pf["temporal_noise_warn"],
        }
    checks["PF7_geometry"] = geo

    warnings = [c.get("note") for c in checks.values()
                if isinstance(c, dict) and c.get("pass") and c.get("note")]
    warnings += [w for w in geo.get("warnings", [])]
    overall = all(c.get("pass", False) for c in checks.values()
                  if isinstance(c, dict))
    report = {
        "kind": "preflight", "utc": utc_now_iso(),
        "camera": {"index": camera_index, "width": w, "height": h,
                   "measured_fps": round(fps, 2),
                   "reported_fps": round(reported_fps, 2)},
        "capture_seconds": round(wall_s, 2), "frames": n,
        "checks": checks, "warnings": warnings, "pass": overall,
        "config_hashes": config_hashes(),
        "config_hash": composite_config_hash(),
    }
    RIG_DIR.mkdir(parents=True, exist_ok=True)
    path = RIG_DIR / f"preflight_{_ts()}.json"
    path.write_text(json.dumps(report, indent=2, default=str),
                    encoding="utf-8")
    report["report_file"] = str(path)
    if overall and write_state:
        record_pass("preflight", {
            "report": str(path), "camera_index": camera_index,
            "measured_fps": round(fps, 2)}, checks_cfg)
    return report


def render_preflight(report: dict) -> str:
    lines = ["=" * 70, "PREFLIGHT (apparatus check - no fly)",
             "=" * 70]
    if "error" in report:
        lines.append(f"ERROR: {report['error']}")
    for k, v in report.get("checks", {}).items():
        if isinstance(v, dict) and "pass" in v:
            lines.append(f"[{'PASS' if v['pass'] else 'FAIL'}] {k}")
    for w in report.get("warnings", []):
        lines.append(f"  warn: {w}")
    lines.append(f"OVERALL: {'PASS' if report.get('pass') else 'FAIL'}")
    lines.append("=" * 70)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# BLANK-ARENA TEST
# --------------------------------------------------------------------------
def analyze_blank(session_dir: Path, tracking_cfg: dict | None = None,
                  checks_cfg: dict | None = None,
                  marker_check: bool | None = None) -> dict:
    """Analyze a recorded no-fly session. All criteria must pass."""
    session_dir = Path(session_dir)
    tracking_cfg = tracking_cfg or load_tracking_cfg()
    checks_cfg = checks_cfg or load_checks_cfg()
    bc = checks_cfg["blank"]
    marker_check = bc.get("marker_check", True) if marker_check is None \
        else marker_check

    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    video = next(p for p in session_dir.glob("video.*"))
    tracks = load_tracks(session_dir / "tracks.csv")

    # ---- pass 1: median background + lighting + temporal noise ----------
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    stack, means = [], []
    sum_px = sumsq_px = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if n % 5 == 0 and len(stack) < 400:
            stack.append(gray.astype(np.uint8))
        means.append(float(gray.mean()))
        if sum_px is None:
            sum_px = np.zeros_like(gray)
            sumsq_px = np.zeros_like(gray)
        else:                                # skip frame 0 (baseline)
            sum_px += gray
            sumsq_px += gray * gray
        n += 1
    cap.release()
    if n == 0:
        raise RuntimeError("no frames in blank video")
    bg = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)

    arena = meta.get("arena") or {}
    roi = arena_roi_mask(int(meta["width"]), int(meta["height"]), arena,
                         int(tracking_cfg.get("roi_inflate_px", 8))) \
        if arena else np.ones(bg.shape, np.uint8) * 255
    # the four stimulus marker rectangles are KNOWN dark features when the
    # LEDs are off; exclude them from the raw-foreground and black-hat masks
    # (on the real rig they sit outside the ROI anyway; this also makes the
    # check robust when the geometry is tight)
    boxes = marker_boxes(int(meta["width"]), int(meta["height"]),
                         tracking_cfg.get("stimulus_markers", {}))
    for bx, by, bw, bh in boxes.values():
        roi[by:by + bh, bx:bx + bw] = 0

    # ---- pass 2: raw foreground vs the full median + static dark spots ---
    cap = cv2.VideoCapture(str(video))
    det_cfg = dict(tracking_cfg.get("detector", {}))
    dark_thr = float(det_cfg.get("dark_threshold", 25))
    min_area = float(det_cfg.get("min_area_px", 12))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_frames = 0
    nf = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.subtract(bg, gray)
        mask = (diff > dark_thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.bitwise_and(mask, roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if any(cv2.contourArea(c) >= min_area for c in contours):
            fg_frames += 1
        nf += 1
    cap.release()

    # static dark features (black-hat on the median, inside the ROI)
    k_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    blackhat = cv2.morphologyEx(bg, cv2.MORPH_BLACKHAT, k_bh)
    bh_mask = ((blackhat > dark_thr).astype(np.uint8) * 255) & roi
    contours, _ = cv2.findContours(bh_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    dark_blobs = [c for c in contours
                  if cv2.contourArea(c) >= bc["static_dark_blob_min_area_px"]]

    # ---- criteria --------------------------------------------------------
    found = tracks["found"]
    conf = tracks["conf"]
    n_conf_det = int(((found) & (conf >= 0.5)).sum())
    n_any_det = int(found.sum())
    mean_arr = np.array(means)
    half = max(len(mean_arr) // 5, 1)
    drift = float(abs(mean_arr[:half].mean() - mean_arr[-half:].mean()))
    std = float(mean_arr.std())
    k = max(n - 1, 1)
    var_px = (sumsq_px / k) - (sum_px / k) ** 2
    noise_med = float(np.median(np.sqrt(np.maximum(var_px, 0))[roi > 0]))
    sides_seen = sorted(set(tracks["stimulus"][tracks["stimulus"] != ""]))

    checks: dict = {
        "B1_confident_false_detections": {
            "n": n_conf_det, "max": bc["confident_detections_max"],
            "pass": n_conf_det <= bc["confident_detections_max"],
            "note": "false positives at tracking confidence: lighting, "
                    "glare, dust, rim reflections, flicker"},
        "B2_any_detections_fraction": {
            "fraction": round(n_any_det / max(nf, 1), 5),
            "max": bc["any_detections_frac_max"],
            "pass": n_any_det / max(nf, 1) <= bc["any_detections_frac_max"]},
        "B3_raw_foreground_motion": {
            "fraction_frames": round(fg_frames / max(nf, 1), 5),
            "max": bc["raw_fg_frames_frac_max"],
            "pass": fg_frames / max(nf, 1) <= bc["raw_fg_frames_frac_max"]},
        "B4_lighting_stability": {
            "mean_gray_std": round(std, 3), "max": bc["mean_gray_std_max"],
            "drift": round(drift, 3), "drift_max": bc["mean_gray_drift_max"],
            "pass": std <= bc["mean_gray_std_max"] and
            drift <= bc["mean_gray_drift_max"],
            "note": "enclosure must stay closed; constant-current lighting "
                    "only (no PWM dimming)"},
        "B5_static_dark_features": {
            "n_blobs": len(dark_blobs),
            "max": bc["static_dark_blobs_max"],
            "pass": len(dark_blobs) <= bc["static_dark_blobs_max"],
            "note": "smudges/shadows/dark rim arcs inside the arena look "
                    "fly-like: clean the floor mat and lid"},
        "B6_sensor_noise": {
            "median_temporal_std": round(noise_med, 2),
            "fail_above": bc["temporal_noise_median_max"],
            "warn_above": bc["temporal_noise_warn"],
            "pass": noise_med <= bc["temporal_noise_median_max"],
            "warning": noise_med > bc["temporal_noise_warn"]},
        "B7_stimulus_marker_channel": {
            "sides_seen": sides_seen, "required": ["E", "N", "S", "W"],
            "checked": bool(marker_check),
            "pass": (not marker_check) or
            set(sides_seen) >= {"N", "E", "S", "W"},
            "note": "blink each stimulus LED once (N,E,S,W) during the blank "
                    "recording following the printed timetable; markers must "
                    "be readable from the video margin"},
    }
    overall = all(c["pass"] for c in checks.values())
    return {
        "kind": "blank", "session": str(session_dir),
        "label": meta.get("label"), "frames": n,
        "duration_s": round(n / float(meta.get("fps") or 30), 1),
        "checks": checks,
        "warnings": [c["note"] for c in checks.values()
                     if c.get("pass") and c.get("note")] +
                    (["sensor noise above warning level"] if noise_med >
                      bc["temporal_noise_warn"] else []),
        "pass": overall,
        "marker_check_skipped": bool(not marker_check),
    }


def next_blank_label(sessions_root: Path | None = None) -> str:
    root = sessions_root or SESSIONS_DIR
    used = []
    if root.exists():
        for d in root.iterdir():
            if d.is_dir() and "_blank_" in d.name:
                try:
                    used.append(int(d.name.rsplit("_blank_", 1)[1].split("_")[0]))
                except (ValueError, IndexError):
                    pass
    return f"blank_{max(used, default=0) + 1:02d}"


def run_blank(camera_index: int = 0, minutes: float | None = None,
              checks_cfg: dict | None = None,
              tracking_cfg: dict | None = None,
              calibration: dict | None = None,
              marker_check: bool = True, force: bool = False) -> dict:
    """Record + track + analyze the no-fly blank; PASS sets the rig state
    (and stores the reference background for fly sessions)."""
    from ..recording.recorder import record

    checks_cfg = checks_cfg or load_checks_cfg()
    tracking_cfg = tracking_cfg or load_tracking_cfg()
    bc = checks_cfg["blank"]
    minutes = float(minutes or bc["minutes"])
    calibration = calibration or load_calibration()

    # ---- gate on a fresh preflight (the blank is the deeper check) -------
    from ..rig import check_record_readiness
    ok, problems = check_record_readiness(minutes, "blank", checks_cfg)
    if not ok and not force:
        return {"pass": False, "error": "preflight gate: " + "; ".join(problems)}

    label = next_blank_label()
    timetable = bc.get("marker_blink", {})
    print("BLANK-ARENA TEST: no fly anywhere in the enclosure; enclosure "
          "CLOSED; arena floor clean; lid on.")
    if marker_check:
        t = float(timetable.get("start_s", 30))
        on = float(timetable.get("on_s", 10))
        gap = float(timetable.get("gap_s", 20))
        print("Follow this LED timetable (switch each stimulus LED on/off "
              "by hand):")
        for side in ("N", "E", "S", "W"):
            print(f"  t={t:5.0f} s: {side} ON for {on:.0f} s")
            t += on + gap
    sdir = record(camera_index=camera_index, minutes=minutes, label=label,
                  kind="blank", calibration_meta=calibration,
                  tracking_cfg=tracking_cfg, checks_cfg=checks_cfg,
                  force=force)
    sdir = Path(sdir)

    print("tracking the blank recording ...")
    from ..tracking.calibration import calibration_from_session
    meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
    track_video(next(sdir.glob("video.*")), sdir / "tracks.csv",
                tracking_cfg, calibration_from_session(meta),
                read_stimulus=True)

    report = analyze_blank(sdir, tracking_cfg, checks_cfg,
                           marker_check=marker_check)
    bg = save_background_median(next(sdir.glob("video.*")))
    report["background"] = bg
    report["config_hashes"] = config_hashes()
    report["config_hash"] = composite_config_hash()
    report["utc"] = utc_now_iso()
    RIG_DIR.mkdir(parents=True, exist_ok=True)
    path = RIG_DIR / f"blank_report_{_ts()}.json"
    path.write_text(json.dumps(report, indent=2, default=str),
                    encoding="utf-8")
    report["report_file"] = str(path)
    if report["pass"]:
        record_pass("blank", {
            "session": sdir.name, "report": str(path),
            "background": str(BACKGROUND_FILE),
            "background_sha256": bg["sha256"],
            "marker_check": bool(marker_check)}, checks_cfg)
    return report


def render_blank(report: dict) -> str:
    lines = ["=" * 70, "BLANK-ARENA TEST (no fly)",
             "=" * 70]
    if "error" in report:
        lines.append(f"ERROR: {report['error']}")
    for k, v in report.get("checks", {}).items():
        lines.append(f"[{'PASS' if v['pass'] else 'FAIL'}] {k}"
                     + (f"  (skipped)" if k.startswith("B7") and
                        not v.get("checked") else ""))
    for w in report.get("warnings", []):
        lines.append(f"  warn: {w}")
    lines.append(f"OVERALL: {'PASS' if report.get('pass') else 'FAIL'}")
    lines.append("=" * 70)
    return "\n".join(lines)
