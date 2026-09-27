"""Action decoder: behavior stream -> game actions.

Two schemes (config/actions.yaml):

zone_dwell (default)
    The fly must remain in one arena zone for `dwell_frames` -> that zone's
    action is emitted. Debounce-safe: momentary crossings emit nothing.
    A sustained PAUSE emits STOP.

motion_direction
    A sustained walking heading (dominant velocity axis) emits the matching
    directional action. Useful when no zone/cue structure is desired.

Both schemes share a cooldown and a tracking-loss tolerance. This module is
deliberately policy-free: it only translates behavior. It never looks at the
game - goal-directedness comes from the computer's cue placement, not from
the decoder.
"""
from __future__ import annotations

import math

from .behavior_classifier import BehaviorSample


class ActionDecoder:
    def __init__(self, actions_cfg: dict):
        cfg = actions_cfg or {}
        self.cfg = cfg
        self.vocabulary = list(cfg.get("action_vocabulary",
                                       ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]))
        self.scheme = cfg.get("scheme", "zone_dwell")
        self.zone_actions: dict[str, str] = dict(cfg.get("zone_actions", {}))
        self.dwell_frames = int(cfg.get("dwell_frames", 5))
        self.pause_dwell_frames = int(cfg.get("pause_dwell_frames", 12))
        self.cooldown_frames = int(cfg.get("cooldown_frames", 8))
        self.max_lost_frames = int(cfg.get("max_lost_frames", 25))
        m = (cfg.get("motion_scheme") or {})
        self.heading_dwell_frames = int(m.get("heading_dwell_frames", 6))
        self.motion_map = dict(m.get("map", {
            "up": "FORWARD", "down": "BACKWARD", "left": "LEFT", "right": "RIGHT"}))

        self._cooldown = 0
        self._cur_zone: str | None = None
        self._zone_frames = 0
        self._pause_frames = 0
        self._lost_frames = 0
        self._cur_bin: str | None = None
        self._bin_frames = 0
        self._last_emitted: str | None = None
        # Outer 20% of each axis counts as an edge band. MUST stay larger
        # than the synthetic attractor geometry (inset 0.12/0.88, rest
        # radius 0.06 -> resting flies sit at 0.82..0.94, inside the band).
        self.edge_band = 0.20

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._cooldown = 0
        self._cur_zone = None
        self._zone_frames = 0
        self._pause_frames = 0
        self._lost_frames = 0
        self._cur_bin = None
        self._bin_frames = 0
        self._last_emitted = None

    def update(self, b: BehaviorSample) -> str | None:
        """Feed one behavior sample; return an emitted action or None."""
        if self._cooldown > 0:
            self._cooldown -= 1

        if not b.found:
            self._lost_frames += 1
            # Brief tracking loss must NOT reset the pause accumulator:
            # intermittent loss at a wall (slow wiggle) used to starve the
            # decoder of emissions forever -> closed-loop deadlock.
            return None
        self._lost_frames = 0

        if self.scheme == "zone_dwell":
            return self._update_zone(b)
        return self._update_motion(b)

    # ------------------------------------------------------------------
    def _emit(self, action: str | None) -> str | None:
        if action is None or action not in self.vocabulary:
            return None
        if self._cooldown > 0:
            return None
        self._cooldown = self.cooldown_frames
        self._last_emitted = action
        return action

    def _update_zone(self, b: BehaviorSample) -> str | None:
        if b.state == "PAUSED":
            self._pause_frames += 1
            self._zone_frames = 0
            self._cur_zone = None
            if self._pause_frames >= self.pause_dwell_frames:
                self._pause_frames = 0          # re-arm so STOP re-fires later
                return self._emit("STOP")
            return None

        self._pause_frames = 0
        if b.zone != self._cur_zone:
            self._cur_zone = b.zone
            self._zone_frames = 1
        else:
            self._zone_frames += 1
        if self._zone_frames >= self.dwell_frames:
            self._zone_frames = 0               # re-arm
            return self._emit(self.zone_actions.get(b.zone))
        return None

    def _update_motion(self, b: BehaviorSample) -> str | None:
        if b.state != "MOVING":
            self._bin_frames = 0
            self._cur_bin = None
            if b.state == "PAUSED":
                self._pause_frames += 1
                if self._pause_frames >= self.pause_dwell_frames:
                    self._pause_frames = 0   # re-arm
                    # POSITIONAL wall-press rule (pure observation, no
                    # memory of past commands, no cue knowledge):
                    #   paused while pressing ONE edge  -> that direction
                    #   paused in the open interior      -> STOP
                    #   paused in a corner (two edges)  -> STOP (ambiguous)
                    bands = []
                    if b.x < self.edge_band:
                        bands.append("LEFT")
                    elif b.x > 1 - self.edge_band:
                        bands.append("RIGHT")
                    if b.y < self.edge_band:
                        bands.append("FORWARD")
                    elif b.y > 1 - self.edge_band:
                        bands.append("BACKWARD")
                    if len(bands) == 1:
                        return self._emit(bands[0])
                    return self._emit("STOP")
            return None
        self._pause_frames = 0
        # dominant velocity axis in IMAGE coordinates (y down)
        if abs(math.cos(b.heading_rad)) >= abs(math.sin(b.heading_rad)):
            axis = "right" if math.cos(b.heading_rad) >= 0 else "left"
        else:
            axis = "down" if math.sin(b.heading_rad) >= 0 else "up"
        if axis == self._cur_bin:
            self._bin_frames += 1
        else:
            self._cur_bin = axis
            self._bin_frames = 1
        if self._bin_frames >= self.heading_dwell_frames:
            self._bin_frames = 0
            return self._emit(self.motion_map.get(axis))
        return None

    # ------------------------------------------------------------------
    def status(self) -> dict:
        """For the live dashboard."""
        return {
            "scheme": self.scheme,
            "zone": self._cur_zone,
            "zone_dwell_progress": self._zone_frames / max(1, self.dwell_frames),
            "pause_progress": self._pause_frames / max(1, self.pause_dwell_frames),
            "cooldown": self._cooldown,
        }
