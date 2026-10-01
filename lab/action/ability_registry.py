"""AbilityRegistry — structured records for equipped GPO moves.

FINAL_ARCHITECTURE §17: each ability carries id, name (if known), input
binding, intent tags, range, startup, recovery, cooldown, resource cost,
damage estimate, mobility effect, defensive property, last result,
learned utility, confidence. Records are populated progressively from
observation + allowed GPO knowledge sources, and STATIC properties are
CACHED — never re-derived from scratch every frame.

Knowledge discipline (docs/GPO_PLAN.md §0): third-party guide facts are
UNVERIFIED until confirmed by direct game observation (VERIFIED /
REJECTED / OUTDATED). Direct current observation wins on conflict.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

INTENT_TAGS = ("STOP", "TURN_LEFT", "TURN_RIGHT", "APPROACH", "RETREAT",
               "EVADE_LEFT", "EVADE_RIGHT", "DEFEND",
               "ATTACK_LIGHT", "ATTACK_HEAVY", "ATTACK_RANGED",
               "SPECIAL", "ESCAPE")

RANGES = ("close", "mid", "long", "unknown")
PROVENANCE_TIERS = ("UNVERIFIED", "VERIFIED", "REJECTED", "OUTDATED")


@dataclass
class AbilityRecord:
    ability_id: str
    name: str = ""                       # if known
    input_binding: str = ""              # e.g. "key:Q" / "mouse:left"
    intent_tags: list = field(default_factory=list)   # compatible intentions
    range: str = "unknown"               # close | mid | long | unknown
    startup_s: float | None = None       # wind-up (observed)
    recovery_s: float | None = None
    cooldown_s: float | None = None
    resource_cost: str = ""              # e.g. "stamina:20" (observed)
    damage_estimate: float | None = None
    mobility_effect: str = ""            # e.g. "dash_forward"
    defensive_property: str = ""         # e.g. "block" / "parry" / "dodge"
    provenance: str = "UNVERIFIED"       # knowledge tier
    learned_utility: float = 0.5         # ENGINEERED value learning output
    confidence: float = 0.0              # 0..1 — how well observed
    last_result: dict = field(default_factory=dict)
    n_uses: int = 0
    n_success: int = 0

    def to_dict(self) -> dict: return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AbilityRecord":
        defaults = {f: cls.__dataclass_fields__[f].default
                    for f in cls.__dataclass_fields__}
        defaults.update({k: v for k, v in d.items() if k in defaults})
        return cls(**defaults)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if not self.ability_id:
            raise ValueError("ability_id required")
        bad = [t for t in self.intent_tags if t not in INTENT_TAGS]
        if bad:
            raise ValueError(f"unknown intent tags {bad}")
        if self.range not in RANGES:
            raise ValueError(f"unknown range {self.range!r}")
        if self.provenance not in PROVENANCE_TIERS:
            raise ValueError(f"unknown provenance {self.provenance!r}")
        for key in ("intent_tags",):
            if not isinstance(getattr(self, key), list):
                raise ValueError(f"{key} must be a list")

    def compatible_with(self, intention_name: str) -> bool:
        return intention_name in self.intent_tags


class AbilityRegistry:
    """In-memory registry with JSON persistence (cache — static ability
    properties are NEVER re-derived per frame). Thread-safe."""

    def __init__(self, cache_path: Path | None = None):
        self.cache_path = Path(cache_path) if cache_path else None
        self._lock = threading.Lock()
        self._abilities: dict[str, AbilityRecord] = {}
        if self.cache_path and self.cache_path.exists():
            self.load()

    # -- CRUD ------------------------------------------------------------------
    def register(self, record: AbilityRecord, overwrite: bool = False) -> None:
        record.validate()
        with self._lock:
            if record.ability_id in self._abilities and not overwrite:
                raise KeyError(f"ability {record.ability_id!r} already "
                               "registered (overwrite=True to replace)")
            self._abilities[record.ability_id] = record

    def register_observed_binding(
            self, ability_id: str, name: str, binding: str,
            intent_tags: list[str], *, range: str = "unknown",
            confidence: float = 0.5, provenance: str = "UNVERIFIED",
            overwrite: bool = True) -> AbilityRecord:
        """Register a move learned from the current GPO HUD/loadout.

        Named fruit/fighting-style/sword moves change with the equipped
        loadout and game updates, so the live system stores the observed
        binding instead of assuming a global hotkey table.
        """
        from .gpo_controls import validate_observed_ability_binding

        binding = validate_observed_ability_binding(binding)
        rec = AbilityRecord(
            ability_id=ability_id,
            name=name,
            input_binding=binding,
            intent_tags=list(intent_tags),
            range=range,
            provenance=provenance,
            confidence=float(max(0.0, min(1.0, confidence))),
        )
        self.register(rec, overwrite=overwrite)
        return rec

    def get(self, ability_id: str) -> AbilityRecord:
        with self._lock:
            return self._abilities[ability_id]

    def all(self) -> list[AbilityRecord]:
        with self._lock:
            return list(self._abilities.values())

    def compatible(self, intention_name: str) -> list[AbilityRecord]:
        with self._lock:
            return [a for a in self._abilities.values()
                    if a.compatible_with(intention_name)]

    def record_result(self, ability_id: str, success: bool,
                      outcome: dict | None = None) -> None:
        """Update last result + use counters (feeds learned utility)."""
        with self._lock:
            a = self._abilities[ability_id]
            a.n_uses += 1
            a.n_success += int(bool(success))
            a.last_result = outcome or {}
            # simple Laplace-smoothed utility baseline; richer engineered
            # value learning lives in lab/world/value.py
            a.learned_utility = round((a.n_success + 1) / (a.n_uses + 2), 4)

    # -- persistence (cache) -----------------------------------------------
    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {k: v.to_dict() for k, v in self._abilities.items()}
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.cache_path)

    def load(self) -> None:
        data = json.loads(self.cache_path.read_text())
        with self._lock:
            self._abilities = {k: AbilityRecord.from_dict(v)
                               for k, v in data.items()}

    def size(self) -> int:
        with self._lock:
            return len(self._abilities)
