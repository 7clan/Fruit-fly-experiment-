"""Live dashboard (optional; requires a display).

Shows side by side:
  [ fly camera | 2D game | status panel ]
with the live decoded action, zone, dwell progress, reward and rolling
success rate - so it is immediately obvious whether the fly is moving
randomly, the computer is controlling, or the fly's behavior is useful.

Headless-safe: if no display is available the dashboard disables itself
with one warning and the experiment continues.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..environment.render import render_game, compose_dashboard


class Dashboard:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._warned = False
        self._history: list[bool] = []

    # ------------------------------------------------------------------
    def show(self, arena, controller, action, reward_total, trial_number) -> int | None:
        if not self.enabled:
            return None
        try:
            fly_frame = getattr(controller, "last_frame", None)
            status_rows = [
                ("trial", str(trial_number)),
                ("t [s]", f"{arena.t:.1f}"),
                ("controller", getattr(controller, "name", "?")),
            ]
            b = getattr(controller, "behavior", None)
            if b is not None:
                status_rows += [
                    ("fly", f"({b.x:.2f},{b.y:.2f})"),
                    ("speed mm/s", f"{b.speed_mm_s:.1f}"),
                    ("state", b.state),
                    ("zone", b.zone),
                ]
                d = getattr(controller, "decoder", None)
                if d is not None:
                    st = d.status()
                    status_rows.append(("dwell", f"{st['zone_dwell_progress']:.0%}"))
            status_rows += [
                ("action", str(action or "-")),
                ("reward", f"{reward_total:.2f}"),
                ("dist", f"{arena.distance_to_target:.0f}px"),
                ("outcome", "-" if arena.reached() is False else "HIT"),
            ]
            game_frame = render_game(arena, {"trial": trial_number,
                                             "action": action,
                                             "reward": reward_total})
            frame = compose_dashboard(game_frame, fly_frame, status_rows)
            cv2.imshow("fly-agent dashboard", frame)
            key = cv2.waitKey(1) & 0xFF
            return key if key != 255 else None
        except Exception:
            if not self._warned:
                print("[dashboard] no display available - continuing headless")
                self._warned = True
            self.enabled = False
            return None

    def close(self) -> None:
        if self.enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
