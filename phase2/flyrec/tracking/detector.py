"""Fly detection: dark blob on bright background + tracking confidence.

Two background modes (config `tracking.yaml: detector.mode`):
  static — background = per-pixel MEDIAN of the first `bg_frames` frames
           (the recording protocol starts BEFORE the fly is introduced /
           the synthetic videos have an empty lead-in, so those frames are
           fly-free). Fully deterministic; best for a fixed, light-tight rig.
  mog2   — running Gaussian-mixture background (Phase-1 style, tiny learning
           rate); use when illumination drifts slowly.

Detection: dark-blob threshold -> morphology -> contours -> area filter ->
nearest-to-last-position candidate. Confidence combines area plausibility
and jump plausibility (see PHASE2_PROTOCOL: confidence feeds G1 coverage).
"""
from __future__ import annotations

import math

import cv2
import numpy as np


class FlyDetector:
    def __init__(self, cfg: dict, px_per_mm: float = 1.0):
        d = dict(cfg)
        self.mode = d.get("mode", "static")
        self.min_area = float(d.get("min_area_px", 12))
        self.max_area = float(d.get("max_area_px", 400))
        self.dark_threshold = float(d.get("dark_threshold", 25))
        self.bg_frames = int(d.get("bg_frames", 40))
        self.max_jump_mm = float(d.get("max_jump_mm", 30.0))
        self.px_per_mm = float(px_per_mm) if px_per_mm > 0 else 1.0
        # morphological kernels
        self._k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # static background state
        self._bg_stack: list[np.ndarray] = []
        self._bg: np.ndarray | None = None
        # mog2 state
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=int(d.get("mog2_history", 300)),
            varThreshold=float(d.get("mog2_varThreshold", 32)),
            detectShadows=False) if self.mode == "mog2" else None
        self._mog_lr = float(d.get("mog2_learning_rate", 0.0005))
        self.last_xy: tuple[float, float] | None = None
        self.n_frames = 0

    # ------------------------------------------------------------------ api
    def reset(self) -> None:
        """Forget all learned state (background, last position)."""
        self._bg_stack.clear()
        self._bg = None
        if self.mode == "mog2":
            self._mog = cv2.createBackgroundSubtractorMOG2(
                history=self._mog.getHistory(),
                varThreshold=self._mog.getVarThreshold(),
                detectShadows=False)
        self.last_xy = None
        self.n_frames = 0

    def detect(self, frame_bgr: np.ndarray, roi_mask: np.ndarray | None = None):
        """Return (x, y, area_px, confidence) or (None, None, 0.0, 0.0).

        roi_mask: optional uint8 mask (255 = search region). Pass the arena
        interior so margin stimulus markers / the arena ring can never be
        mistaken for the animal.
        """
        self.n_frames += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.mode == "static":
            if self._bg is None:
                self._bg_stack.append(gray.astype(np.uint8))
                if len(self._bg_stack) >= self.bg_frames:
                    stack = np.stack(self._bg_stack, axis=0)
                    self._bg = np.median(stack, axis=0).astype(np.uint8)
                return None, None, 0.0, 0.0          # still building background
            diff = cv2.subtract(self._bg, gray)       # fly darker than bg > 0
            mask = (diff > self.dark_threshold).astype(np.uint8) * 255
        else:  # mog2
            fg = self._mog.apply(gray, learningRate=self._mog_lr)
            mask = fg
            # keep only DARK foreground (a fly on a bright floor)
            if self._bg is None:
                self._bg = self._mog.getBackgroundImage()
                if self._bg is None:
                    return None, None, 0.0, 0.0
            diff = cv2.subtract(self._bg if self._bg is not None else gray, gray)
            mask = cv2.bitwise_and(mask, (diff > 8).astype(np.uint8) * 255)
            if self._mog.getBackgroundImage() is not None:
                self._bg = self._mog.getBackgroundImage()

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close)
        if roi_mask is not None:
            mask = cv2.bitwise_and(mask, roi_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in contours:
            a = cv2.contourArea(c)
            if self.min_area <= a <= self.max_area:
                m = cv2.moments(c)
                if m["m00"] > 0:
                    cands.append((m["m10"] / m["m00"], m["m01"] / m["m00"], a))
        if not cands:
            self.last_xy = None
            return None, None, 0.0, 0.0

        # candidate selection: nearest to last position (largest if unknown)
        if self.last_xy is not None:
            cands.sort(key=lambda c: math_dist(c[:2], self.last_xy))
        else:
            cands.sort(key=lambda c: -c[2])
        x, y, area = cands[0]

        # hard gate: impossible jump -> treat as loss (tracker re-acquires)
        hard_cap_px = 2.5 * self.max_jump_mm * self.px_per_mm
        if self.last_xy is not None and math_dist((x, y), self.last_xy) > hard_cap_px:
            self.last_xy = None
            return None, None, 0.0, 0.0

        conf = self._confidence(x, y, area)
        self.last_xy = (x, y)
        return float(x), float(y), float(area), float(conf)

    def _confidence(self, x: float, y: float, area: float) -> float:
        """0..1 plausibility: area inside range (0.6) + jump plausibility (0.4)."""
        if area < self.min_area:
            c_area = math.exp(-(self.min_area - area) / (0.3 * self.min_area + 1e-9))
        elif area > self.max_area:
            c_area = math.exp(-(area - self.max_area) / (0.3 * self.max_area + 1e-9))
        else:
            c_area = 1.0
        if self.last_xy is None:
            c_jump = 0.5                                     # neutral
        else:
            d_mm = math_dist((x, y), self.last_xy) / self.px_per_mm
            c_jump = math.exp(-d_mm / self.max_jump_mm)
        return float(0.6 * c_area + 0.4 * c_jump)


def math_dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# ---------------------------------------------------------------------------
class StimulusMarkerReader:
    """Reads the open-loop stimulus state back from the video margin.

    Four marker ROIs sit OUTSIDE the arena (so the fly tracker never sees
    them) but INSIDE the camera field of view. Each marker is bright when
    its side's stimulus is ON. Baseline brightness = median of the lead-in
    frames. Returns the active side ('N'|'E'|'S'|'W') or None.
    """

    def __init__(self, frame_w: int, frame_h: int, cfg: dict | None = None):
        cfg = cfg or {}
        mw, mh = int(cfg.get("marker_w_px", 30)), int(cfg.get("marker_h_px", 14))
        margin = int(cfg.get("margin_px", 18))
        self.on_delta = float(cfg.get("on_brightness_delta", 40.0))
        cx, cy = frame_w // 2, frame_h // 2
        def rect(x, y):  # top-left corner -> (x, y, w, h)
            return (x, y, mw, mh)
        self.rois = {
            "N": rect(cx - mw // 2, margin),
            "S": rect(cx - mw // 2, frame_h - margin - mh),
            "W": rect(margin, cy - mh // 2),
            "E": rect(frame_w - margin - mw, cy - mh // 2),
        }
        self._base: dict[str, float] | None = None
        self._samples: list[dict[str, float]] = []
        self._n_base = int(cfg.get("baseline_frames", 40))

    def read(self, frame_bgr: np.ndarray) -> str | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        vals = {s: float(gray[y:y + h, x:x + w].mean())
                for s, (x, y, w, h) in self.rois.items()}
        if self._base is None:
            self._samples.append(vals)
            if len(self._samples) >= self._n_base:
                self._base = {s: float(np.median([v[s] for v in self._samples]))
                              for s in vals}
            return None
        active = [s for s, v in vals.items()
                  if v - self._base[s] > self.on_delta]
        return active[0] if len(active) == 1 else (active[0] if active else None)
