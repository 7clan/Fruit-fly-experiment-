"""Trajectory processing: gap interpolation, smoothing, velocity/heading.

Pipeline (constants frozen in gate.yaml:analysis_constants):
  1. interpolate missing detections spanning <= max_gap_s (0.5 s)
  2. 5-frame median filter on x, y
  3. central difference (span 3) on smoothed positions -> velocity
  4. speed, heading (y-up arena frame: 0 rad = RIGHT/+x, pi/2 = FORWARD/+y)
"""
from __future__ import annotations

import numpy as np


def interpolate_gaps(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                     found: np.ndarray, max_gap_s: float = 0.5):
    """Linear interpolation over lost-track gaps shorter than max_gap_s."""
    x, y = x.copy(), y.copy()
    n = len(t)
    i = 0
    interp = np.zeros(n, dtype=bool)
    while i < n:
        if not found[i]:
            j = i
            while j < n and not found[j]:
                j += 1
            # found[i-1] and found[j] bracket the gap (if they exist)
            if 0 < i and j < n and (t[j] - t[i - 1]) <= max_gap_s:
                for k in range(i, j):
                    w = (t[k] - t[i - 1]) / (t[j] - t[i - 1])
                    x[k] = x[i - 1] + w * (x[j] - x[i - 1])
                    y[k] = y[i - 1] + w * (y[j] - y[i - 1])
                    interp[k] = True
            i = j
        else:
            i += 1
    return x, y, interp


def median_filter(a: np.ndarray, k: int = 5) -> np.ndarray:
    """Edge-padded median filter (k odd)."""
    if k <= 1 or len(a) == 0:
        return a.copy()
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    out = np.empty_like(a, dtype=np.float64)
    for i in range(len(a)):
        out[i] = np.median(ap[i:i + k])
    return out


def compute_kinematics(t: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray,
                       median_k: int = 5, vel_span: int = 3):
    """Return dict with smoothed positions, velocity, speed, heading (rad).

    NaN entries (uninterpolated losses) propagate: speed/heading are NaN
    there and downstream stats ignore them.
    """
    xs = median_filter(np.asarray(x_mm, float), median_k)
    ys = median_filter(np.asarray(y_mm, float), median_k)
    dt = np.diff(t)
    dt = np.where(dt > 0, dt, np.nan)
    half = vel_span // 2
    vx = np.full_like(t, np.nan, dtype=float)
    vy = np.full_like(t, np.nan, dtype=float)
    for i in range(half, len(t) - half):
        denom = t[i + half] - t[i - half]
        if denom > 0:
            vx[i] = (xs[i + half] - xs[i - half]) / denom
            vy[i] = (ys[i + half] - ys[i - half]) / denom
    speed = np.hypot(vx, vy)
    heading = np.arctan2(vy, vx)                    # y-up frame
    return {"x": xs, "y": ys, "vx": vx, "vy": vy, "speed": speed,
            "heading": heading}


def path_length(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                valid: np.ndarray) -> float:
    """Total traveled distance over valid samples."""
    d = np.hypot(np.diff(x), np.diff(y))
    ok = valid[1:] & valid[:-1]
    return float(d[ok].sum())
