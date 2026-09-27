"""px -> mm calibration + arena geometry (wall distance / wall tangent).

The arena can be circular (default; matches the Phase-2 protocol dish) or
rectangular. All behavioral analysis is done in millimetres with the origin
at the arena CENTER, x to the right, y UP (note: image coordinates have y
down; `px_to_mm` flips y so that downstream math uses a standard right-handed
frame. FORWARD in the action vocabulary means camera-up = +y here).
"""
from __future__ import annotations

import math


class Calibration:
    def __init__(self, px_per_mm: float, arena: dict):
        if px_per_mm <= 0:
            raise ValueError("px_per_mm must be > 0")
        self.px_per_mm = float(px_per_mm)
        self.mm_per_px = 1.0 / self.px_per_mm
        self.kind = arena.get("type", "circle")
        if self.kind == "circle":
            self.center_px = tuple(arena["center_px"])
            self.radius_px = float(arena["radius_px"])
            self.radius_mm = self.radius_px * self.mm_per_px
        elif self.kind == "rect":
            self.center_px = tuple(arena["center_px"])
            self.half_w_px = float(arena["half_w_px"])
            self.half_h_px = float(arena["half_h_px"])
            self.half_w_mm = self.half_w_px * self.mm_per_px
            self.half_h_mm = self.half_h_px * self.mm_per_px
        else:
            raise ValueError(f"unknown arena type {self.kind!r}")

    # ---------------------------------------------------------------- px->mm
    def px_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Image px -> arena-frame mm (origin at arena center, y UP)."""
        cx, cy = self.center_px
        x = (x_px - cx) * self.mm_per_px
        y = -(y_px - cy) * self.mm_per_px      # flip: image y down -> y up
        return x, y

    def mm_to_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        cx, cy = self.center_px
        return (cx + x_mm * self.px_per_mm, cy - y_mm * self.px_per_mm)

    # ------------------------------------------------------------------ wall
    def wall_distance_mm(self, x_mm: float, y_mm: float) -> float:
        """Distance to the nearest wall (>= 0 inside the arena)."""
        if self.kind == "circle":
            r = math.hypot(x_mm, y_mm)
            return self.radius_mm - r
        return min(self.half_w_mm - abs(x_mm), self.half_h_mm - abs(y_mm))

    def wall_tangent_angle(self, x_mm: float, y_mm: float) -> float:
        """Heading (radians, y-up frame) that runs PARALLEL to the nearest
        wall at this position. For a circle: tangent at the angular position
        of the fly. For a rect: direction of the nearest edge."""
        if self.kind == "circle":
            return math.atan2(y_mm, x_mm) + math.pi / 2.0
        dx = self.half_w_mm - abs(x_mm)
        dy = self.half_h_mm - abs(y_mm)
        return 0.0 if dx < dy else math.pi / 2.0   # horizontal or vertical edge

    def angle_to_wall_normal(self, x_mm: float, y_mm: float) -> float:
        """Heading pointing at the nearest wall (outward normal direction)."""
        if self.kind == "circle":
            return math.atan2(y_mm, x_mm)
        dx = self.half_w_mm - abs(x_mm)
        dy = self.half_h_mm - abs(y_mm)
        if dx < dy:
            return 0.0 if x_mm >= 0 else math.pi
        return math.pi / 2.0 if y_mm >= 0 else -math.pi / 2.0

    # ------------------------------------------------------------------- io
    def as_dict(self) -> dict:
        d = {"px_per_mm": self.px_per_mm, "type": self.kind,
             "center_px": list(self.center_px)}
        if self.kind == "circle":
            d["radius_px"] = self.radius_px
            d["radius_mm"] = round(self.radius_mm, 2)
        else:
            d["half_w_px"] = self.half_w_px
            d["half_h_px"] = self.half_h_px
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        if "px_per_mm" in d:
            return cls(d["px_per_mm"], d)
        # allow flat session.json geometry blocks
        return cls(d["px_per_mm"], {k: v for k, v in d.items() if k != "px_per_mm"})


def calibration_from_session(session: dict) -> Calibration:
    """Build a Calibration from a session.json metadata dict."""
    cal = dict(session.get("calibration", {}))
    if "arena" in session:
        cal = {"px_per_mm": session["calibration"]["px_per_mm"],
               **session["arena"]}
    return Calibration.from_dict(cal)
