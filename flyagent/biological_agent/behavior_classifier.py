"""Behavior classifier: kinematics -> zone occupancy + movement state.

Adds the semantic layer on top of raw tracking:
  * which arena zone the fly occupies (configurable grid)
  * MOVING vs PAUSED (speed threshold in mm/s - physical units, so the
    threshold is independent of camera resolution)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..fly_tracker.trajectory import KinSample

_ROW_NAMES = {0: "top", 1: "middle", 2: "bottom"}
_COL_NAMES = {0: "left", 1: "center", 2: "right"}


def zone_name(row: int, col: int) -> str:
    r = _ROW_NAMES.get(row, f"r{row}")
    c = _COL_NAMES.get(col, f"c{col}")
    return f"{r}_{c}"


@dataclass
class BehaviorSample:
    t: float
    x: float
    y: float
    speed_mm_s: float
    heading_rad: float
    state: str            # MOVING | PAUSED | LOST
    zone: str             # zone_name(...) or "unknown"
    found: bool
    confidence: float = 0.0
    extra: dict = field(default_factory=dict)


class BehaviorClassifier:
    def __init__(self, actions_cfg: dict):
        grid = (actions_cfg or {}).get("zone_grid", {"cols": 3, "rows": 3})
        self.cols = max(1, int(grid.get("cols", 3)))
        self.rows = max(1, int(grid.get("rows", 3)))
        self.pause_speed_mm_s = float(
            (actions_cfg or {}).get("pause_speed_mm_s", 8.0))

    def classify(self, kin: KinSample) -> BehaviorSample:
        if not kin.found:
            return BehaviorSample(kin.t, kin.x, kin.y, kin.speed_mm_s,
                                  kin.heading_rad, "LOST", "unknown",
                                  found=False)
        row = min(self.rows - 1, max(0, int(kin.y * self.rows)))
        col = min(self.cols - 1, max(0, int(kin.x * self.cols)))
        state = "PAUSED" if kin.speed_mm_s < self.pause_speed_mm_s else "MOVING"
        return BehaviorSample(kin.t, kin.x, kin.y, kin.speed_mm_s,
                              kin.heading_rad, state, zone_name(row, col),
                              found=True, confidence=kin.confidence)
