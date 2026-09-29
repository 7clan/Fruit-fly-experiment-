"""Engineered value learning [ENGINEERED — NOT fly mushroom-body learning].

FINAL_ARCHITECTURE §24: the practical HYBRID agent may use engineered
learning, e.g. (enemy type × move) success, route outcomes. These values
feed the AbilityResolver and Planner. CLEARLY LABELED: this is not the
biological KC→MBON plasticity (that workstream is frozen at Gate-4 v2).

Model: per-context Laplace-smoothed success estimates.
  context keys: ("enemy", <enemy_type>) or ("route", <route_id>) or ()
  value(context, ability_id) = (n_success + 1) / (n_uses + 2)
Backed by the async MemoryStore (episodes in, values out) — updates are
periodic/event-driven, never on the combat-critical path.
"""

from __future__ import annotations

import threading

from .memory import MemoryStore


class ValueTable:
    """Context-keyed ability/route values with Laplace smoothing."""

    def __init__(self, memory: MemoryStore | None = None):
        self.memory = memory
        self._lock = threading.RLock()   # reentrant: best() calls value()
        self._n: dict[tuple, dict[str, list]] = {}   # ctx -> ability -> [succ, uses]

    def observe(self, context: tuple, ability_id: str, success: bool) -> None:
        with self._lock:
            ctx = self._n.setdefault(context, {})
            s, u = ctx.get(ability_id, [0, 0])
            ctx[ability_id] = [s + int(bool(success)), u + 1]

    def value(self, context: tuple, ability_id: str,
              default: float = 0.5) -> float:
        with self._lock:
            ctx = self._n.get(context, {})
            s, u = ctx.get(ability_id, [0, 0])
            return (s + 1) / (u + 2) if u else default

    def best(self, context: tuple, ability_ids: list) -> tuple:
        """(best_id, best_value) among ability_ids for a context."""
        with self._lock:
            scored = [(a, self.value(context, a)) for a in ability_ids]
        if not scored:
            return "", default_value()
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0]

    def snapshot(self) -> dict:
        with self._lock:
            return {"|".join(map(str, k)) if isinstance(k, tuple) else str(k):
                    {a: {"success": s, "uses": u, "value": round((s + 1) / (u + 2), 4)}
                     for a, (s, u) in ctx.items()}
                    for k, ctx in self._n.items()} | {"ENGINEERED": True}


def default_value() -> float:
    return 0.5
