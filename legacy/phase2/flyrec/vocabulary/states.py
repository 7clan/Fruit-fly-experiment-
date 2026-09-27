"""Windowing + behavioral-state labels for the action vocabulary.

A window is a non-overlapping `window_s` (1.0 s) block of the session.
  label = PAUSE                 if window mean speed < pause_speed (2.0 mm/s)
        = sector(net displacement angle)  otherwise
          (FORWARD/BACKWARD/LEFT/RIGHT; arena frame, y UP;
           FORWARD = camera-up)
Features (for the LOSO decoder): [mean_speed, dx, dy, disp_mag].
"""
from __future__ import annotations

import math

import numpy as np

from ..analysis.circular import sector_index

CARDINALS = ["FORWARD", "BACKWARD", "LEFT", "RIGHT"]
STATES_A = CARDINALS + ["PAUSE"]


def build_windows(t, x, y, speed, valid, window_s=1.0,
                  pause_speed=2.0, require_frac_valid=0.6):
    """Non-overlapping windows. Returns list of dicts with labels/features."""
    if len(t) == 0:
        return []
    t0 = float(t[0])
    duration = float(t[-1] - t[0])
    n_win = int(math.floor(duration / window_s))
    windows = []
    for k in range(n_win):
        a = t0 + k * window_s
        b = a + window_s
        sel = (t >= a) & (t < b)
        v = sel & valid
        if sel.sum() == 0 or v.sum() / max(1, sel.sum()) < require_frac_valid:
            continue
        tw, xw, yw, sw = t[v], x[v], y[v], speed[v]
        mean_speed = float(np.mean(sw)) if len(sw) else float("nan")
        if not math.isfinite(mean_speed):
            continue
        dx = float(xw[-1] - xw[0]) if len(xw) >= 2 else 0.0
        dy = float(yw[-1] - yw[0]) if len(yw) >= 2 else 0.0
        mag = math.hypot(dx, dy)
        if mean_speed < pause_speed:
            label = "PAUSE"
        elif mag < 1e-9:
            label = "PAUSE"          # no net displacement -> treat as pause
        else:
            label = sector_index(math.atan2(dy, dx))
        windows.append({
            "session_t0": a, "label": label,
            "features": [mean_speed, dx, dy, mag],
        })
    return windows


def occupancy(windows, states=None) -> dict:
    """Fraction of windows per state (all states, zero-filled)."""
    states = states or STATES_A
    n = len(windows)
    c = {s: 0 for s in states}
    for w in windows:
        if w["label"] in c:
            c[w["label"]] += 1
    return {s: (c[s] / n if n else 0.0) for s in states}
