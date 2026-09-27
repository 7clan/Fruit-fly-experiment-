"""Controllers: everything that can drive the game implements this interface.

This is the control-group methodology made architectural:
  RandomController   - floor baseline (A)
  FixedController    - hand-coded greedy ceiling (B)
  FlyController      - biological behavioral input (C)
  KeyboardController - human control, Autonomy Level 0 (debugging)
All run through the SAME trial loop, the SAME metrics and the SAME database,
so any performance difference is attributable to the controller alone.
"""
from __future__ import annotations

import abc


class Controller(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def reset(self, ctx: dict) -> None:
        """Called once per trial. ctx: {rng, game_cfg, trial_index, ...}."""

    @abc.abstractmethod
    def observe(self, game_state: dict, t_ms: int) -> None:
        """Called every frame BEFORE act().

        Computer-side perception + stimulus placement happen here.
        IMPORTANT: the fly never sees the game. The computer decides which
        benign visual cue to present based on game state; the fly only ever
        sees its own arena.
        """

    @abc.abstractmethod
    def act(self) -> str | None:
        """Return an action to apply, or None this frame."""

    def on_trial_end(self, result) -> None:
        pass

    def close(self) -> None:
        pass
