"""Bouts, pauses, and wall/edge interactions.

Bout detection: hysteresis speed threshold (on 3.0 / off 2.0 mm/s),
gaps <= 0.2 s merged, bouts < 0.25 s dropped. Pauses = inter-bout
intervals >= 0.30 s. Wall metrics: distance to wall, in-band time,
wall-following (in band AND heading within 45 deg of the wall tangent),
wall contact (<= 1 mm).
"""
from __future__ import annotations

import math

import numpy as np

from ..tracking.calibration import Calibration


def find_bouts(t, speed, valid, on=3.0, off=2.0, min_dur=0.25,
               gap_merge=0.2):
    """State-machine bout detection with hysteresis. Returns list of dicts."""
    n = len(t)
    bouts = []
    in_bout = False
    start = 0.0
    last_fast = -1.0
    i = 0
    while i < n:
        v = speed[i] if valid[i] else np.nan
        if not in_bout:
            if np.isfinite(v) and v >= on:
                in_bout = True
                start = t[i]
                last_fast = t[i]
        else:
            if np.isfinite(v) and v >= off:
                last_fast = t[i]
            elif (not np.isfinite(v)) or t[i] - last_fast > gap_merge:
                # end of bout at last_fast
                bouts.append((start, last_fast))
                in_bout = False
        i += 1
    if in_bout:
        bouts.append((start, last_fast))
    # merge bouts separated by <= gap_merge (already implicit) and filter
    out = []
    for (a, b) in bouts:
        if b - a >= min_dur:
            out.append({"t_on": float(a), "t_off": float(b),
                        "duration_s": float(b - a)})
    return out


def find_pauses(t, bouts, min_pause=0.30, t_end=None):
    """Pauses = gaps between consecutive bouts >= min_pause."""
    pauses = []
    prev_end = 0.0
    bounds = [(b["t_on"], b["t_off"]) for b in bouts] + \
             [(t_end if t_end is not None else (t[-1] if len(t) else 0.0), None)]
    for (a, b) in bounds:
        if a - prev_end >= min_pause:
            pauses.append({"t_on": prev_end, "t_off": float(a),
                           "duration_s": float(a - prev_end)})
        if b is not None:
            prev_end = b
    return pauses


def bout_metrics(bouts, t, x, y, heading, valid):
    """Per-bout duration, net displacement, path length, straightness, mean
    heading (circular)."""
    res = []
    for b in bouts:
        sel = (t >= b["t_on"] - 1e-9) & (t <= b["t_off"] + 1e-9) & valid
        if sel.sum() < 2:
            continue
        bx, by, bt = x[sel], y[sel], t[sel]
        net = float(np.hypot(bx[-1] - bx[0], by[-1] - by[0]))
        seg = np.hypot(np.diff(bx), np.diff(by))
        plen = float(seg.sum())
        hs = heading[sel]
        hs = hs[np.isfinite(hs)]
        mean_h = float(np.arctan2(np.mean(np.sin(hs)), np.mean(np.cos(hs)))) \
            if len(hs) else float("nan")
        res.append({
            "duration_s": b["duration_s"],
            "net_displacement_mm": net,
            "path_length_mm": plen,
            "straightness": (net / plen) if plen > 0 else 0.0,
            "mean_heading_rad": mean_h,
        })
    return res


def wall_metrics(t, x, y, heading, valid, calib: Calibration,
                 wall_band_mm=10.0, parallel_deg=45.0,
                 moving_speed=None, contact_mm=1.0):
    """Wall/edge interaction metrics (fractions over VALID frames; following
    and contact additionally require movement)."""
    n = len(t)
    dist = np.full(n, np.nan)
    tangent = np.full(n, np.nan)
    for i in range(n):
        if valid[i]:
            dist[i] = calib.wall_distance_mm(x[i], y[i])
            tangent[i] = calib.wall_tangent_angle(x[i], y[i])
    in_band = np.isfinite(dist) & (dist <= wall_band_mm) & (dist >= -1.0)
    moving = np.isfinite(heading) if moving_speed is None else \
        (np.isfinite(moving_speed) & (moving_speed >= 2.0))
    ang_diff = np.abs(_wrap_pi(heading - tangent))
    ang_diff_alt = np.abs(_wrap_pi(heading - tangent - math.pi))
    parallel = np.minimum(ang_diff, ang_diff_alt) <= math.radians(parallel_deg)
    following = in_band & parallel & moving
    contact = np.isfinite(dist) & (dist <= contact_mm) & moving

    stats = {
        "median_wall_distance_mm": float(np.nanmedian(dist))
        if np.isfinite(dist).any() else None,
        "frac_time_in_band": float(in_band.mean()) if n else None,
        "frac_moving_near_wall": float((in_band & moving).sum() /
                                       max(1, moving.sum())),
        "frac_moving_wall_following": float(following.sum() /
                                            max(1, moving.sum())),
        "wall_contact_rate_per_min": float(contact.sum() /
                                           (t[-1] / 60.0)) if n and t[-1] > 0 else None,
    }
    return stats, dist


def _wrap_pi(a):
    return (np.asarray(a, float) + math.pi) % (2 * math.pi) - math.pi
