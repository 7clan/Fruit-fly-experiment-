"""Background-subtraction fly tracker (non-invasive, markerless).

Pipeline per frame:
  grayscale -> MOG2 foreground -> binary threshold -> morphological open
  -> contour filter by area -> select candidate nearest to last position.

The tracker is masked to the arena ROI so stimulus hardware drawn outside the
arena can never be mistaken for the animal. If the fly pauses for a long time
it may fade into the background model; the decoder handles brief tracking
loss gracefully (`max_lost_frames`) and reacquires when the fly moves again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import cv2
import numpy as np


class TrackedPoint(NamedTuple):
    found: bool
    x: float = 0.0          # frame pixels
    y: float = 0.0
    area: float = 0.0       # px^2 of the fly blob
    confidence: float = 0.0 # area / expected_area, clipped to [0, 1.5]


@dataclass
class TrackerConfig:
    history: int = 300
    var_threshold: float = 32
    min_area: float = 25
    max_area: float = 2500
    expected_area: float = 120   # px^2 of a typical fly blob at calibration
    learning_rate: float = -1.0  # MOG2 default adaptive rate


class FlyTracker:
    def __init__(self, cfg: dict, roi=None):
        valid = {f.name for f in TrackerConfig.__dataclass_fields__.values()}
        c = {k: v for k, v in (cfg or {}).items() if k in valid}
        c = {**TrackerConfig().__dict__, **c}
        self.cfg = TrackerConfig(**c)
        self.roi = roi  # (x, y, w, h) arena interior in frame pixels
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history,
            varThreshold=self.cfg.var_threshold,
            detectShadows=False,
        )
        self.last: Optional[tuple[float, float]] = None
        self.kernel = np.ones((3, 3), np.uint8)
        self._prev_gray: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def update(self, frame_bgr: np.ndarray) -> TrackedPoint:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lr = self.cfg.learning_rate
        if lr is not None and lr > 0:
            lr = float(lr)   # fixed slow rate: a paused fly survives longer
        else:
            lr = -1.0        # MOG2 auto
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            mask = np.zeros_like(gray)
            mask[ry:ry + rh, rx:rx + rw] = 255
            fg = self.bg.apply(gray, learningRate=lr)
            fg = cv2.bitwise_and(fg, mask)
        else:
            fg = self.bg.apply(gray, learningRate=lr)

        _, bw = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, self.kernel)

        pts = self._pick(gray, bw)
        if pts is None:
            # Fallback: frame differencing. MOG2 slowly absorbs nearly-static
            # flies (wall-press wiggle, long pauses); inter-frame motion
            # still reveals them. Helps real footage as much as synthetic.
            # The SAME arena ROI mask is applied - otherwise a moving cue
            # stripe outside the arena would be grabbed instead of the fly.
            if self._prev_gray is not None and self.last is not None:
                diff = cv2.absdiff(gray, self._prev_gray)
                _, bw2 = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
                bw2 = cv2.morphologyEx(bw2, cv2.MORPH_OPEN, self.kernel)
                if self.roi is not None:
                    rx, ry, rw, rh = self.roi
                    m2 = np.zeros_like(bw2)
                    m2[ry:ry + rh, rx:rx + rw] = 255
                    bw2 = cv2.bitwise_and(bw2, m2)
                pts = self._pick(gray, bw2)
        self._prev_gray = gray
        if pts is None:
            return TrackedPoint(found=False)
        x, y, area = pts
        self.last = (x, y)
        conf = float(np.clip(area / max(self.cfg.expected_area, 1.0), 0.0, 1.5))
        return TrackedPoint(found=True, x=x, y=y, area=area, confidence=conf)

    def _pick(self, gray, bw):
        """Return (x, y, area) of the best blob in the binary mask, or None."""
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in contours:
            a = cv2.contourArea(c)
            if self.cfg.min_area <= a <= self.cfg.max_area:
                m = cv2.moments(c)
                if m["m00"] > 0:
                    cands.append((m["m10"] / m["m00"], m["m01"] / m["m00"], a))
        if not cands:
            return None
        if self.last is not None:
            lx, ly = self.last
            return min(cands, key=lambda p: (p[0] - lx) ** 2 + (p[1] - ly) ** 2)
        return max(cands, key=lambda p: p[2])

    def reset_position(self) -> None:
        """Forget the last position (e.g., after long tracking loss)."""
        self.last = None
