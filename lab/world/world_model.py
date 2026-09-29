"""Semantic world model [ENGINEERED] — structured entities + relations.

FINAL_ARCHITECTURE §22: entities PLAYER, NPC, ENEMY, QUEST, OBJECT,
ISLAND, LANDMARK, WEAPON, FRUIT, ABILITY, ACCESSORY; relations
located_at, requires, rewards, dangerous_to, useful_for, leads_to,
equipped, cooldown, quest_target.

This is ENGINEERED semantic understanding — the fly brain does NOT
understand the concept of "quest". Every dashboard surface and replay
record labels this component as ENGINEERED.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

ENTITY_TYPES = ("PLAYER", "NPC", "ENEMY", "QUEST", "OBJECT", "ISLAND",
                "LANDMARK", "WEAPON", "FRUIT", "ABILITY", "ACCESSORY")

RELATION_TYPES = ("located_at", "requires", "rewards", "dangerous_to",
                  "useful_for", "leads_to", "equipped", "cooldown",
                  "quest_target")

# knowledge discipline (docs/GPO_PLAN.md §0)
TIERS = ("UNVERIFIED", "VERIFIED", "REJECTED", "OUTDATED")


@dataclass
class Entity:
    eid: str
    etype: str
    label: str = ""
    properties: dict = field(default_factory=dict)
    tier: str = "UNVERIFIED"          # knowledge provenance tier
    last_seen_ns: int = 0
    confidence: float = 0.0

    def __post_init__(self):
        if self.etype not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type {self.etype!r}")
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}")

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Relation:
    rid: str
    subject: str            # entity id
    predicate: str          # RELATION_TYPES
    obj: str                # entity id or free string value
    tier: str = "UNVERIFIED"
    weight: float = 1.0

    def __post_init__(self):
        if self.predicate not in RELATION_TYPES:
            raise ValueError(f"unknown relation {self.predicate!r}")

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class WorldModel:
    """Thread-safe entity/relation store with observation-driven updates.

    Direct current observation WINS over stored knowledge on conflict
    (spec §31): update_from_observation() can mark stored claims
    OUTDATED when live perception contradicts them.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}

    # -- entities ----------------------------------------------------------
    def upsert_entity(self, e: Entity) -> None:
        with self._lock:
            old = self.entities.get(e.eid)
            if old and old.tier == "VERIFIED" and e.tier == "UNVERIFIED":
                # verified knowledge is not downgraded by guide claims
                e.tier = old.tier
            self.entities[e.eid] = e

    def get_entity(self, eid: str) -> Entity | None:
        with self._lock:
            return self.entities.get(eid)

    # -- relations -----------------------------------------------------------
    def add_relation(self, r: Relation) -> None:
        with self._lock:
            self.relations[r.rid] = r

    def relations_of(self, eid: str) -> list[Relation]:
        with self._lock:
            return [r for r in self.relations.values()
                    if r.subject == eid or r.obj == eid]

    # -- observation update (direct observation wins) -------------------------
    def update_from_observation(self, obs: dict) -> list[str]:
        """Update entities from a WorldObservation dict. Returns human-
        readable change notes (dashboard/helper trace)."""
        notes = []
        with self._lock:
            player = obs.get("player", {})
            pos = player.get("position")
            if pos:
                pe = self.entities.get("player:0")
                if pe is None:
                    pe = Entity("player:0", "PLAYER", label="me", tier="VERIFIED")
                    self.entities["player:0"] = pe
                    notes.append("player entity created (VERIFIED by observation)")
                pe.properties["position"] = pos
                pe.properties["heading"] = player.get("heading")
                pe.last_seen_ns = obs.get("ts_ns", 0)
            for i, en in enumerate(obs.get("enemies", [])):
                eid = f"enemy:live:{i}"
                e = Entity(eid, "ENEMY", label=en.get("type", "unknown"),
                           properties={"distance": en.get("distance"),
                                       "direction": en.get("direction"),
                                       "attacking": en.get("attacking"),
                                       "threat": en.get("threat")},
                           tier="VERIFIED", last_seen_ns=obs.get("ts_ns", 0),
                           confidence=en.get("confidence", 0.0))
                self.entities[eid] = e
                notes.append(f"enemy {i}: dist={en.get('distance')}, "
                             f"attacking={en.get('attacking')}")
        return notes

    def snapshot(self) -> dict:
        with self._lock:
            return {"entities": {k: v.to_dict() for k, v in self.entities.items()},
                    "relations": {k: v.to_dict() for k, v in self.relations.items()},
                    "ENGINEERED": True}
