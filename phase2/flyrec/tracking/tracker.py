"""Track a recorded session: video -> tracks.csv (per-frame position,
confidence, arena-frame mm coordinates, stimulus readback).

tracks.csv columns:
  frame, t_s, x_px, y_px, area_px, confidence, found,
  x_mm, y_mm, stimulus            (stimulus: N|E|S|W or empty)
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration
from .detector import FlyDetector, StimulusMarkerReader

COLUMNS = ["frame", "t_s", "x_px", "y_px", "area_px", "confidence", "found",
           "x_mm", "y_mm", "stimulus"]


def track_video(video_path: str | Path, out_csv: str | Path,
                tracking_cfg: dict, calibration: Calibration,
                read_stimulus: bool = True,
                verbose: bool = False) -> dict:
    """Run the detector over a whole video; write tracks.csv; return stats."""
    video_path, out_csv = Path(video_path), Path(out_csv)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or not np.isfinite(fps):
        fps = float(tracking_cfg.get("assume_fps", 30))
    fps = float(fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det_cfg = dict(tracking_cfg.get("detector", {}))
    marker_cfg = tracking_cfg.get("stimulus_markers", {})
    detector = FlyDetector(det_cfg, px_per_mm=calibration.px_per_mm)
    markers = StimulusMarkerReader(w, h, marker_cfg) if read_stimulus else None

    # arena ROI mask (inflated a few px): margin markers / arena ring can
    # never be mistaken for the animal
    roi = np.zeros((h, w), np.uint8)
    inflate = int(tracking_cfg.get("roi_inflate_px", 8))
    if calibration.kind == "circle":
        cx, cy = calibration.center_px
        cv2.circle(roi, (int(round(cx)), int(round(cy))),
                   int(round(calibration.radius_px)) + inflate, 255, -1)
    else:
        cx, cy = calibration.center_px
        cv2.rectangle(roi,
                      (int(cx - calibration.half_w_px - inflate),
                       int(cy - calibration.half_h_px - inflate)),
                      (int(cx + calibration.half_w_px + inflate),
                       int(cy + calibration.half_h_px + inflate)), 255, -1)

    rows = []
    frame = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t_s = frame / fps
        x, y, area, conf = detector.detect(frame_bgr, roi_mask=roi)
        stim = markers.read(frame_bgr) if markers is not None else None
        if x is None:
            rows.append([frame, f"{t_s:.4f}", "", "", "", f"{conf:.3f}", 0,
                         "", "", stim or ""])
        else:
            xm, ym = calibration.px_to_mm(x, y)
            rows.append([frame, f"{t_s:.4f}", f"{x:.2f}", f"{y:.2f}",
                         f"{area:.1f}", f"{conf:.3f}", 1,
                         f"{xm:.3f}", f"{ym:.3f}", stim or ""])
        frame += 1
        if verbose and frame % 500 == 0:
            print(f"  ... {frame} frames", flush=True)
    cap.release()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(COLUMNS)
        wcsv.writerows(rows)

    found = [r for r in rows if r[6] == 1]
    conf_arr = np.array([float(r[5]) for r in found]) if found else np.array([])
    stats = {
        "video": str(video_path),
        "tracks_csv": str(out_csv),
        "fps": fps,
        "n_frames": frame,
        "n_found": len(found),
        "coverage": (len(found) / frame) if frame else 0.0,
        "coverage_conf05": (float((conf_arr >= 0.5).mean())
                            if len(conf_arr) else 0.0),
        "mean_confidence": float(conf_arr.mean()) if len(conf_arr) else 0.0,
    }
    return stats


# ---------------------------------------------------------------------------
def load_tracks(csv_path: str | Path) -> dict:
    """Load tracks.csv into numpy arrays (missing detections -> NaN)."""
    p = Path(csv_path)
    data = np.genfromtxt(p, delimiter=",", names=True,
                         dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = data.reshape(1)
    frame = np.atleast_1d(data["frame"]).astype(float)
    t = np.atleast_1d(data["t_s"]).astype(float)
    found = np.atleast_1d(data["found"]).astype(int)
    with np.errstate(invalid="ignore"):
        x_px = np.where(found == 1, np.atleast_1d(data["x_px"]).astype(float), np.nan)
        y_px = np.where(found == 1, np.atleast_1d(data["y_px"]).astype(float), np.nan)
        x_mm = np.where(found == 1, np.atleast_1d(data["x_mm"]).astype(float), np.nan)
        y_mm = np.where(found == 1, np.atleast_1d(data["y_mm"]).astype(float), np.nan)
        conf = np.where(found == 1, np.atleast_1d(data["confidence"]).astype(float), 0.0)
    stim_raw = np.atleast_1d(data["stimulus"]).astype(str)
    stim = np.array([("" if str(v).strip().lower() in
                      ("", "nan", "false", "none") else str(v).strip())
                     for v in stim_raw])
    return {"frame": frame, "t": t, "found": found.astype(bool),
            "x_px": x_px, "y_px": y_px, "x_mm": x_mm, "y_mm": y_mm,
            "conf": conf, "stimulus": stim,
            "csv": str(p)}


def stimulus_events_from_tracks(tracks: dict, fps: float) -> list[dict]:
    """Reconstruct ON events (t_on, t_off, side) from the marker column."""
    events = []
    cur_side, t_on, t_last = None, None, None
    for t, s in zip(tracks["t"], tracks["stimulus"]):
        s = (s or "").strip()
        if s != cur_side:
            if cur_side is not None:
                events.append({"t_on": float(t_on), "t_off": float(t_last),
                               "side": cur_side})
            cur_side = s if s else None
            t_on = t if s else None
        t_last = t
    if cur_side is not None:
        events.append({"t_on": float(t_on), "t_off": float(t_last),
                       "side": cur_side})
    # drop degenerate single-frame glitches (< 0.2 s)
    return [e for e in events if e["t_off"] - e["t_on"] >= 0.2]
