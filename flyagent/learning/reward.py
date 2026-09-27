"""Reward system: sparse AND shaped, both supported (config-driven).

sparse : only final objective
          TARGET_REACHED -> +sparse_success        (e.g. "max level" later)
shaped : per-step potential-based shaping
          distance closed  -> +shaped_per_px * closed
          time passing     -> -time_penalty_per_s * dt
          obstacle blocked -> -obstacle_penalty
Both modes can run simultaneously ("both"); components are logged separately
so an analysis can always reconstruct the sparse-only view.

Nothing here encodes real-game mechanics - in Phase 1 the "objective" is a
2D target, and in later phases the same class receives progression deltas
observed on screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardComponents:
    sparse: float = 0.0
    shaped_progress: float = 0.0
    time_penalty: float = 0.0
    obstacle_penalty: float = 0.0
    total: float = 0.0
    log: list = field(default_factory=list)


class RewardFunction:
    def __init__(self, cfg: dict):
        cfg = cfg or {}
        self.mode = cfg.get("mode", "both")
        self.sparse_success = float(cfg.get("sparse_success", 1.0))
        self.shaped_per_px = float(cfg.get("shaped_per_px", 0.004))
        self.time_penalty_per_s = float(cfg.get("time_penalty_per_s", 0.002))
        self.obstacle_penalty = float(cfg.get("obstacle_penalty", 0.05))
        self._prev_dist: float | None = None
        self.total = 0.0

    def reset(self, dist: float) -> None:
        self._prev_dist = float(dist)
        self.total = 0.0

    def step(self, dist: float, event_kinds: list[str], dt: float,
             t: float) -> RewardComponents:
        c = RewardComponents()
        if self.mode in ("sparse", "both") and "TARGET_REACHED" in event_kinds:
            c.sparse = self.sparse_success
        if self.mode in ("shaped", "both"):
            if self._prev_dist is not None:
                c.shaped_progress = (self._prev_dist - dist) * self.shaped_per_px
            c.time_penalty = -self.time_penalty_per_s * dt
            if "BLOCKED" in event_kinds:
                c.obstacle_penalty = -self.obstacle_penalty
        c.total = c.sparse + c.shaped_progress + c.time_penalty + c.obstacle_penalty
        self.total += c.total
        self._prev_dist = float(dist)
        return c
