"""Baseline controllers: random floor + fixed greedy ceiling + keyboard."""
from __future__ import annotations

import numpy as np

from .base import Controller


class RandomController(Controller):
    """Floor baseline: random action every random interval.

    Interval distribution is drawn to roughly match the fly decoder's
    emission cadence (configurable) - otherwise cadence alone would
    confound the comparison (see docs/METHODOLOGY.md).
    """

    name = "random"

    def __init__(self, cfg: dict, rng: np.random.Generator,
                 vocabulary: list[str] | None = None):
        self.cfg = cfg or {}
        self.rng = rng
        self.vocabulary = [a for a in (vocabulary or
                                       ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"])]
        self._timer = 0.0

    def reset(self, ctx: dict) -> None:
        self._timer = 0.0

    def observe(self, game_state: dict, t_ms: int) -> None:
        pass

    def act(self) -> str | None:
        self._timer -= 1.0 / 30.0
        if self._timer > 0:
            return None
        lo, hi = self.cfg.get("interval_s", [0.5, 1.2])
        self._timer = float(self.rng.uniform(lo, hi))
        p_stop = float(self.cfg.get("stop_probability", 0.10))
        if self.rng.random() < p_stop:
            return "STOP"
        return str(self.rng.choice([a for a in self.vocabulary if a != "STOP"]))


class FixedController(Controller):
    """Ceiling baseline: computer directly picks the greedy action.

    Equivalent to a perfect screen-perception + hardcoded policy. Any
    fly-controlled condition should sit BETWEEN random and fixed.
    """

    name = "fixed"

    def __init__(self, cfg: dict):
        self.interval_s = float((cfg or {}).get("interval_s", 0.5))
        self._timer = 0.0

    def reset(self, ctx: dict) -> None:
        self._timer = 0.0

    def observe(self, game_state: dict, t_ms: int) -> None:
        self._desired = game_state["desired_action"]

    def act(self) -> str | None:
        self._timer -= 1.0 / 30.0
        if self._timer > 0:
            return None
        self._timer = self.interval_s
        return self._desired


class KeyboardController(Controller):
    """Human control (Autonomy Level 0) - debugging and calibration.

    Requires a display: run `main.py run --controller keyboard --show`.
    Keys: WASD / arrows, Space = STOP, Q = quit trial.
    """

    name = "keyboard"

    _KEYS = {
        ord("w"): "FORWARD", ord("a"): "LEFT", ord("s"): "BACKWARD",
        ord("d"): "RIGHT", 82: "FORWARD", 81: "LEFT", 84: "BACKWARD",
        83: "RIGHT", ord(" "): "STOP",
    }

    def reset(self, ctx: dict) -> None:
        self._last = None

    def observe(self, game_state: dict, t_ms: int) -> None:
        pass

    def feed_key(self, key: int) -> None:
        if key == ord("q"):
            self._last = "STOP"
        elif key in self._KEYS:
            self._last = self._KEYS[key]

    def act(self) -> str | None:
        a, self._last = self._last, self._last  # persistent like a held key
        return a
