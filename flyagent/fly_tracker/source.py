"""Frame sources for the fly tracker.

All sources expose the same interface, so the tracking pipeline is identical
whether the fly is simulated (pipeline validation without any animal), live on
a webcam, or offline in a recorded video.

The stimulus hook `set_cue(action)` is the closed-loop channel:
  * SyntheticSource  -> positions a dark stripe in the frame margin (the
                        simulated fly is attracted to it with probability
                        cue_bias).
  * WebcamSource     -> Phase 2 hardware hook (LED strip / display command).
                        Currently a logged no-op.

Cue stripes are drawn OUTSIDE the arena rectangle so the tracker (which masks
to the arena ROI) never confuses stimulus hardware for the animal.
"""
from __future__ import annotations

import abc
import math

import cv2
import numpy as np

from ..biological_agent.synthetic_fly import SyntheticFly

# Frame layout: a margin band around the arena rectangle.
MARGIN = 20


class FrameSource(abc.ABC):
    """Provides BGR frames with a monotonically increasing frame counter."""

    def __init__(self, width: int, height: int, warmup_frames: int = 0):
        self.width = int(width)
        self.height = int(height)
        self.frame_index = -1
        self.warmup_frames = int(warmup_frames)

    @abc.abstractmethod
    def read(self):
        """Return (frame_index, frame_bgr); (None, None) at end of stream."""

    def close(self) -> None:
        pass

    def set_cue(self, action: str | None) -> None:
        """Present the benign visual stimulus for `action` (or clear it)."""

    @property
    def arena_rect(self):
        """(x, y, w, h) of the arena interior in frame pixels."""
        return (MARGIN, MARGIN,
                self.width - 2 * MARGIN, self.height - 2 * MARGIN)


class SyntheticSource(FrameSource):
    """Renders a simulated Drosophila into frames -> validates the real CV path."""

    def __init__(self, cfg: dict, fly_cfg: dict, rng: np.random.Generator,
                 warmup_frames: int = 0):
        super().__init__(cfg.get("width", 360), cfg.get("height", 280), warmup_frames)
        self.arena_mm = tuple(cfg.get("arena_mm", [90, 60]))
        self.fly = SyntheticFly(rng, fly_cfg, arena_mm=self.arena_mm)
        self._cue: str | None = None
        self._rng = rng
        self._noise = rng

    def read(self):
        self.frame_index += 1
        during_warmup = self.frame_index < self.warmup_frames
        if not during_warmup:
            dt = 1.0 / 30.0  # fly sim advances at the tracker's frame rate
            self.fly.step(dt)
        return self.frame_index, self._render(with_fly=not during_warmup)

    def set_cue(self, action: str | None) -> None:
        self._cue = action
        self.fly.set_cue(action)

    def true_position(self):
        """Ground-truth normalized position (for pipeline validation only)."""
        return self.fly.position

    def _render(self, with_fly: bool = True):
        h, w = self.height, self.width
        frame = np.full((h, w, 3), 240, dtype=np.uint8)
        noise = self._noise.normal(0, 4.0, (h, w))
        frame = np.clip(frame.astype(np.float32) + noise[..., None], 0, 255)
        frame = frame.astype(np.uint8)

        ax, ay, aw, ah = self.arena_rect
        cv2.rectangle(frame, (ax - 2, ay - 2), (ax + aw + 1, ay + ah + 1),
                      (150, 150, 150), 1)

        # Cue stripe in the margin (outside the tracked ROI).
        if self._cue is not None:
            dark = (70, 70, 70)
            L = 14
            if self._cue == "LEFT":
                cv2.rectangle(frame, (3, ay + ah // 2 - 40), (3 + L, ay + ah // 2 + 40), dark, -1)
            elif self._cue == "RIGHT":
                cv2.rectangle(frame, (w - 3 - L, ay + ah // 2 - 40), (w - 3, ay + ah // 2 + 40), dark, -1)
            elif self._cue == "FORWARD":  # top margin
                cv2.rectangle(frame, (ax + aw // 2 - 40, 3), (ax + aw // 2 + 40, 3 + L), dark, -1)
            elif self._cue == "BACKWARD":  # bottom margin
                cv2.rectangle(frame, (ax + aw // 2 - 40, h - 3 - L), (ax + aw // 2 + 40, h - 3), dark, -1)

        if with_fly:
            xn, yn = self.fly.position
            px = int(ax + xn * aw)
            py = int(ay + yn * ah)
            cv2.circle(frame, (px, py), 6, (60, 60, 60), -1)
            cv2.circle(frame, (px, py), 7, (90, 90, 90), 1)
        return frame


class WebcamSource(FrameSource):
    """Real overhead camera over the behavioral arena (user machine, Phase 2)."""

    def __init__(self, cfg: dict, warmup_frames: int = 0):
        super().__init__(cfg.get("width", 1280), cfg.get("height", 720), warmup_frames)
        self.cap = cv2.VideoCapture(int(cfg.get("camera_index", 0)))
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam; check camera_index")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        self.frame_index += 1
        return self.frame_index, frame

    def set_cue(self, action: str | None) -> None:
        # Phase 2: forward `action` to the stimulus hardware (LED strip /
        # display layer) over serial. Benign visual stimuli only - no heat,
        # no shock, no aversive stimuli (see docs/METHODOLOGY.md, welfare).
        pass

    def close(self) -> None:
        self.cap.release()


class VideoSource(FrameSource):
    """Recorded video of a fly (offline analysis / tracker validation)."""

    def __init__(self, cfg: dict, warmup_frames: int = 0):
        path = cfg.get("video_path", "")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        super().__init__(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640,
                         int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480,
                         warmup_frames)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        self.frame_index += 1
        return self.frame_index, frame

    def close(self) -> None:
        self.cap.release()


def build_source(tracking_cfg: dict, fly_cfg: dict | None = None,
                 rng: np.random.Generator | None = None):
    t = tracking_cfg["source"]["type"]
    warm = int(tracking_cfg["tracker"].get("warmup_frames", 0))
    if t == "synthetic":
        return SyntheticSource(tracking_cfg["source"],
                               fly_cfg or {}, rng or np.random.default_rng(0), warm)
    if t == "webcam":
        return WebcamSource(tracking_cfg["source"], warm)
    if t == "video":
        return VideoSource(tracking_cfg["source"], warm)
    raise ValueError(f"unknown source type: {t}")
