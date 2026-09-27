"""Trajectory buffer: smoothing, velocity, heading, pause detection.

Keeps a short history of calibrated (normalized) positions with timestamps
and produces a per-frame KinSample used by the behavior classifier.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .calibration import Calibrator


@dataclass
class KinSample:
    t: float                 # seconds since experiment start
    x: float                 # normalized [0,1]
    y: float
    vx: float                # normalized units / s
    vy: float
    speed_mm_s: float
    heading_rad: float
    found: bool
    confidence: float = 0.0


class TrajectoryBuffer:
    def __init__(self, maxlen: int = 45, vel_span: int = 4, vel_alpha: float = 0.5):
        self.buf: deque = deque(maxlen=maxlen)
        self.vel_span = max(2, int(vel_span))
        self.vel_alpha = float(vel_alpha)
        self._vx = 0.0
        self._vy = 0.0

    def reset(self) -> None:
        self.buf.clear()
        self._vx = self._vy = 0.0

    def update(self, t: float, xn: float, yn: float, found: bool,
               confidence: float, calib: Calibrator) -> KinSample:
        if not found:
            # hold last known position, decay velocity
            self._vx *= 0.5
            self._vy *= 0.5
            if self.buf:
                last = self.buf[-1]
                return KinSample(t, last[0], last[1], self._vx, self._vy,
                                 calib.speed_mm_s(self._vx, self._vy),
                                 calib.heading_rad(self._vx, self._vy),
                                 found=False, confidence=0.0)
            return KinSample(t, xn, yn, 0.0, 0.0, 0.0, 0.0, found=False)

        self.buf.append((xn, yn, t))
        if len(self.buf) >= self.vel_span:
            x0, y0, t0 = self.buf[-self.vel_span]
            x1, y1, t1 = self.buf[-1]
            dt = max(t1 - t0, 1e-6)
            rvx = (x1 - x0) / dt
            rvy = (y1 - y0) / dt
            a = self.vel_alpha
            self._vx = a * rvx + (1 - a) * self._vx
            self._vy = a * rvy + (1 - a) * self._vy
        return KinSample(t, xn, yn, self._vx, self._vy,
                         calib.speed_mm_s(self._vx, self._vy),
                         calib.heading_rad(self._vx, self._vy),
                         found=True, confidence=confidence)

    @property
    def path_length(self) -> float:
        """Normalized path length accumulated since reset (for diagnostics)."""
        total = 0.0
        for i in range(1, len(self.buf)):
            dx = self.buf[i][0] - self.buf[i - 1][0]
            dy = self.buf[i][1] - self.buf[i - 1][1]
            total += float(np.hypot(dx, dy))
        return total
