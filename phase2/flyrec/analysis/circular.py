"""Circular statistics for movement direction (headings).

Heading convention: radians in the arena frame with y UP
(0 = RIGHT/+x, pi/2 = FORWARD/+y, +/-pi = LEFT/-x, -pi/2 = BACKWARD/-y).
"""
from __future__ import annotations

import math

import numpy as np


def mean_resultant(heading: np.ndarray) -> tuple[float, float]:
    """(mean angle, mean resultant length R in [0, 1])."""
    h = heading[np.isfinite(heading)]
    if len(h) == 0:
        return float("nan"), 0.0
    s, c = np.mean(np.sin(h)), np.mean(np.cos(h))
    return float(math.atan2(s, c)), float(math.hypot(s, c))


def circular_std(R: float) -> float:
    """Circular standard deviation in degrees (from R)."""
    if not (0 < R <= 1):
        return float("inf") if R == 0 else 0.0
    return float(math.degrees(math.sqrt(-2.0 * math.log(R))))


def heading_autocorr(heading: np.ndarray, t: np.ndarray, lag_s: float,
                     speed: np.ndarray | None = None) -> float:
    """Persistence: Re(mean(exp(i(theta_{t+lag} - theta_t)))) over pairs
    separated by ~lag_s. 1 = same direction, 0 = uncorrelated.

    If `speed` is given, only pairs where BOTH frames are MOVING
    (speed >= 2 mm/s) are used: a heading is undefined while the animal
    stands still.
    """
    h = heading
    ok = np.isfinite(h)
    if speed is not None:
        ok = ok & np.isfinite(speed) & (speed >= 2.0)
    vals = []
    n = len(t)
    for i in range(n):
        j = np.searchsorted(t, t[i] + lag_s)
        if j < n and ok[i] and ok[j] and t[j] - t[i] <= lag_s * 1.5:
            vals.append(h[j] - h[i])
    if not vals:
        return float("nan")
    return float(np.real(np.mean(np.exp(1j * np.array(vals)))))


def heading_histogram(heading: np.ndarray, bins: int = 16):
    """Counts per bin over [-pi, pi)."""
    h = heading[np.isfinite(heading)]
    counts, edges = np.histogram(h, bins=bins, range=(-math.pi, math.pi))
    return counts.tolist(), edges.tolist()


def sector_index(heading_rad: float) -> str:
    """90-degree bins centered on the cardinals (arena frame, y UP).

    FORWARD = pi/2 (+y, camera-up), BACKWARD = -pi/2, RIGHT = 0, LEFT = pi.
    """
    a = (heading_rad + math.pi) % (2 * math.pi) - math.pi   # wrap to [-pi, pi)
    # bin centers at -pi/2 (BACKWARD), 0 (RIGHT), pi/2 (FORWARD), +/-pi (LEFT)
    if a >= math.pi / 4 and a < 3 * math.pi / 4:
        return "FORWARD"
    if a >= -math.pi / 4 and a < math.pi / 4:
        return "RIGHT"
    if a >= -3 * math.pi / 4 and a < -math.pi / 4:
        return "BACKWARD"
    return "LEFT"
