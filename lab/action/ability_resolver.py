"""AbilityResolver — fly INTENTION → concrete game ability [ENGINEERED].

FINAL_ARCHITECTURE §16/§18 (binding):

  * The fly brain has no neuron for a named GPO move, so an ENGINEERED
    resolver converts a fly-selected INTENTION into an actual available
    game ability.
  * Fly-intention compatibility is a HARD GATE / dominant constraint:
    the resolver may choose BETWEEN abilities compatible with the
    current intention, but must NOT casually override FLY=DEFEND into
    ATTACK_HEAVY. The fly chooses the behavioral category; the resolver
    chooses the game-specific implementation.
  * Selection is numerical/structured and CHEAP — milliseconds, not
    seconds; never slow planning per move. No LLM in the loop.

Scoring (weights frozen in RESOLVER_CONFIG_ID v1; changes require a new
config id + benchmark):
  hard gate     intention ∈ ability.intent_tags, ability available
                (cooldown ready, resource OK)
  + utility     learned utility (engineered value learning)
  + range fit   distance-vs-range match (0..1)
  + recency     penalize just-used ability (spread practice)
  + readiness   observed confidence in the record
Fail-closed: NO compatible+available ability → returns None (the motor
executor idles safely); it NEVER substitutes an incompatible intention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Intention
from .ability_registry import AbilityRecord, AbilityRegistry

CONFIG_ID = "ability-resolver-v1"

_WEIGHTS = {
    "utility": 1.0,
    "range_fit": 1.0,
    "readiness": 0.5,
    "recency": 0.3,
}

_RANGE_ORDER = {"close": 0, "mid": 1, "long": 2, "unknown": 1}


@dataclass
class ResolvedAbility:
    intention: str
    ability_id: str
    score: float
    input_binding: str = ""
    candidates: list = field(default_factory=list)   # [(id, score, why)]
    resolver_config: str = CONFIG_ID
    blocked_reason: str | None = None

    def to_dict(self) -> dict:
        return {"intention": self.intention,
                "ability_id": self.ability_id,
                "input_binding": self.input_binding,
                "score": round(self.score, 4),
                "candidates": [{"id": i, "score": round(s, 4), "why": w}
                               for i, s, w in self.candidates],
                "resolver_config": self.resolver_config,
                "blocked_reason": self.blocked_reason}


def _range_fit(record: AbilityRecord, distance: float | None) -> float:
    """Distance-vs-range compatibility (0..1). distance normalized 0..1
    (1 = touching). unknown range -> neutral 0.5."""
    if distance is None or record.range == "unknown":
        return 0.5
    # ideal distance band per range class (normalized)
    bands = {"close": (0.55, 1.0), "mid": (0.25, 0.7), "long": (0.0, 0.45)}
    lo, hi = bands[record.range]
    if lo <= distance <= hi:
        return 1.0
    # linear falloff outside the band
    span = 0.25
    if distance < lo:
        return max(0.0, 1.0 - (lo - distance) / span)
    return max(0.0, 1.0 - (distance - hi) / span)


class AbilityResolver:
    """Intention → concrete ability. Pure dict/numeric work, no I/O."""

    def __init__(self, registry: AbilityRegistry,
                 weights: dict | None = None,
                 config_id: str = CONFIG_ID):
        self.registry = registry
        self.w = dict(_WEIGHTS)
        if weights:
            self.w.update(weights)
        self.config_id = config_id
        self._last_used: dict[str, float] = {}   # ability_id -> ts_ns
        self.stats = {"resolved": 0, "blocked": 0}

    def resolve(self, intention: Intention,
                distance: float | None = None,
                now_ns: int = 0,
                cooldown_state: dict | None = None) -> ResolvedAbility:
        """Resolve ONE intention. cooldown_state: {ability_id: ready_bool}
        from live observation (overrides registry cooldown fields)."""
        # ---- HARD GATE: compatible intents only (never override category)
        compatible = self.registry.compatible(intention.name)
        scored: list[tuple[str, float, dict]] = []
        for rec in compatible:
            ready = True
            why = {}
            if cooldown_state is not None and rec.ability_id in cooldown_state:
                ready = bool(cooldown_state[rec.ability_id])
                why["cooldown_observed"] = not ready
            if not ready:
                continue
            util = rec.learned_utility
            fit = _range_fit(rec, distance)
            read = rec.confidence
            last = self._last_used.get(rec.ability_id, None)
            recency = 1.0
            if last is not None and now_ns:
                age_s = (now_ns - last) / 1e9
                recency = max(0.0, min(1.0, age_s / 3.0))
            score = (self.w["utility"] * util +
                     self.w["range_fit"] * fit +
                     self.w["readiness"] * read +
                     self.w["recency"] * recency)
            why.update({"utility": round(util, 3), "range_fit": round(fit, 3),
                        "readiness": round(read, 3), "recency": round(recency, 3)})
            scored.append((rec.ability_id, score, why))

        if not scored:
            self.stats["blocked"] += 1
            return ResolvedAbility(intention=intention.name, ability_id="",
                                   score=0.0, input_binding="", candidates=[],
                                   resolver_config=self.config_id,
                                   blocked_reason="no_compatible_available")

        scored.sort(key=lambda t: t[1], reverse=True)
        best_id, best_score, _ = scored[0]
        self._last_used[best_id] = now_ns
        self.stats["resolved"] += 1
        best_record = self.registry.get(best_id)
        return ResolvedAbility(intention=intention.name, ability_id=best_id,
                               score=best_score,
                               input_binding=best_record.input_binding,
                               candidates=[(i, s, w) for i, s, w in scored],
                               resolver_config=self.config_id)
