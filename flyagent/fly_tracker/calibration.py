"""Arena calibration: 4-point perspective transform.

`scripts/calibrate_arena.py` lets you click the 4 arena corners
(TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT) on a live camera snapshot.
The resulting homography maps any frame pixel to normalized arena
coordinates [0, 1]^2, which makes zone logic independent of camera angle,
lens distortion (first order) and resolution.

flip_x / flip_y correct camera orientation relative to the game screen axes.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Normalized destination corners, same order as the click order above.
DST = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)


def order_points(pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    return p


class Calibrator:
    def __init__(self, H: np.ndarray | None = None,
                 flip_x: bool = False, flip_y: bool = False,
                 arena_mm: tuple[float, float] = (90.0, 60.0)):
        self.H = H
        self.flip_x = bool(flip_x)
        self.flip_y = bool(flip_y)
        self.arena_mm = (float(arena_mm[0]), float(arena_mm[1]))

    # ------------------------------------------------------------------
    @classmethod
    def from_points(cls, src_px, **kw) -> "Calibrator":
        src = order_points(src_px).astype(np.float32)
        H = cv2.getPerspectiveTransform(src, DST)
        return cls(H=H, **kw)

    @classmethod
    def identity(cls, **kw) -> "Calibrator":
        return cls(H=None, **kw)

    @classmethod
    def from_rect(cls, rect, **kw) -> "Calibrator":
        """Arena occupies a known rectangle in the frame (synthetic source)."""
        x, y, w, h = rect
        src = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                       dtype=np.float32)
        return cls.from_points(src, **kw)

    # ------------------------------------------------------------------
    def to_normalized(self, x_px: float, y_px: float) -> tuple[float, float]:
        if self.H is not None:
            pt = np.array([[[x_px, y_px]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, self.H)[0, 0]
            xn, yn = float(out[0]), float(out[1])
        else:
            xn, yn = float(x_px), float(y_px)
        if self.flip_x:
            xn = 1.0 - xn
        if self.flip_y:
            yn = 1.0 - yn
        return xn, yn

    def speed_mm_s(self, vx_norm: float, vy_norm: float) -> float:
        """Convert normalized-axis velocity components to mm/s ground speed.

        The two normalized axes have different physical scale when the arena
        is not square, so each component is scaled by its own axis length.
        """
        vx_mm = vx_norm * self.arena_mm[0]
        vy_mm = vy_norm * self.arena_mm[1]
        return float(np.hypot(vx_mm, vy_mm))

    def heading_rad(self, vx_norm: float, vy_norm: float) -> float:
        """Physically-correct heading (image coords: y axis points DOWN)."""
        vx_mm = vx_norm * self.arena_mm[0]
        vy_mm = vy_norm * self.arena_mm[1]
        return float(np.arctan2(vy_mm, vx_mm))

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        np.savez(str(path), H=self.H if self.H is not None else np.zeros((3, 3)),
                 flip_x=self.flip_x, flip_y=self.flip_y,
                 arena_mm=np.array(self.arena_mm))

    @classmethod
    def load(cls, path) -> "Calibrator":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        d = np.load(str(p))
        H = d["H"]
        H = None if (H.size == 9 and float(H.sum()) == 0.0) else H
        return cls(H=H, flip_x=bool(d["flip_x"]), flip_y=bool(d["flip_y"]),
                   arena_mm=tuple(d["arena_mm"].tolist()))
