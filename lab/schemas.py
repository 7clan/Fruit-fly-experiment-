"""Structured schemas for the DigitalFlyLab pipeline.

FINAL_ARCHITECTURE.md §5 (structured world observation), §6 (fly input
channels + intention vector), §12 (compact state vectors — never giant
raw data structures to every component).

All schemas are dataclasses with to_dict()/from_dict() so every bus
envelope is JSON-safe (replay, dashboard, multiprocessing).

TIER labels:
  * WorldObservation / FlyChannels — ENGINEERED perception output.
  * Intention — the decoded output of the BIOLOGICAL brain (DN readouts),
    plus provenance fields recording the neural populations it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# structured world observation (compact typed state)
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    position: Optional[list] = None        # [x, y] px or normalized
    heading: Optional[float] = None        # radians or degrees (units set by vision)
    health: Optional[float] = None         # 0..1 fraction or HP units (label which)
    stamina: Optional[float] = None
    health_units: Optional[str] = None     # "fraction" | "hp" | "unknown"

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> "PlayerState":
        return cls(**{k: d.get(k) for k in
                      ("position", "heading", "health", "stamina", "health_units")})


@dataclass
class TargetState:
    type: str = "unknown"                  # semantic label from vision/heavy
    direction: Optional[float] = None      # bearing relative to player heading
    distance: Optional[float] = None       # normalized 0..1 (0=far, 1=touching)
    confidence: float = 0.0                # 0..1 vision confidence

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> "TargetState":
        return cls(**{k: d.get(k) for k in ("type", "direction", "distance", "confidence")})


@dataclass
class EnemyState:
    direction: Optional[float] = None      # bearing relative to player
    distance: Optional[float] = None       # normalized 0..1
    attacking: bool = False                # attack wind-up / active detected
    threat: float = 0.0                    # 0..1 imminent-threat intensity
    confidence: float = 0.0
    type: str = "unknown"

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> "EnemyState":
        return cls(**{k: d.get(k, cls.__dataclass_fields__[k].default)
                      for k in ("direction", "distance", "attacking",
                                "threat", "confidence", "type")})


@dataclass
class UIState:
    loading: bool = False
    dialogue: bool = False
    menu: bool = False
    combat: bool = False                   # large UI-state transitions (fast tier)

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> "UIState":
        return cls(**{k: bool(d.get(k, False))
                      for k in ("loading", "dialogue", "menu", "combat")})


@dataclass
class AbilityAvailability:
    ability_id: str = ""
    available: bool = False
    cooldown_s: Optional[float] = None
    inferred_range: str = "unknown"        # "close"|"mid"|"long"|"unknown"

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> "AbilityAvailability":
        return cls(**{k: d.get(k, cls.__dataclass_fields__[k].default)
                      for k in ("ability_id", "available", "cooldown_s",
                                "inferred_range")})


@dataclass
class WorldObservation:
    """Compact structured observation consumed by world model + fly
    channel encoder. NEVER pass giant raw detections to every component."""
    ts_ns: int
    frame_ref: Optional[dict] = None       # {"frame_id": int, "capture_ts_ns": int, "w":, "h":}
    source: str = "fast_vision"            # producer label
    player: PlayerState = field(default_factory=PlayerState)
    target: TargetState = field(default_factory=TargetState)
    enemies: list = field(default_factory=list)   # list[EnemyState] (nearest first)
    ui: UIState = field(default_factory=UIState)
    abilities: list = field(default_factory=list) # list[AbilityAvailability]
    notes: dict = field(default_factory=dict)     # producer diagnostics (small!)

    def primary_enemy(self) -> Optional[EnemyState]:
        return self.enemies[0] if self.enemies else None

    def to_dict(self) -> dict:
        return {
            "ts_ns": self.ts_ns,
            "frame_ref": self.frame_ref,
            "source": self.source,
            "player": self.player.to_dict(),
            "target": self.target.to_dict(),
            "enemies": [e.to_dict() for e in self.enemies],
            "ui": self.ui.to_dict(),
            "abilities": [a.to_dict() for a in self.abilities],
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorldObservation":
        return cls(
            ts_ns=int(d["ts_ns"]),
            frame_ref=d.get("frame_ref"),
            source=d.get("source", "fast_vision"),
            player=PlayerState.from_dict(d.get("player", {})),
            target=TargetState.from_dict(d.get("target", {})),
            enemies=[EnemyState.from_dict(e) for e in d.get("enemies", [])],
            ui=UIState.from_dict(d.get("ui", {})),
            abilities=[AbilityAvailability.from_dict(a)
                       for a in d.get("abilities", [])],
            notes=dict(d.get("notes", {})),
        )


# ---------------------------------------------------------------------------
# fly input channels (ENGINEERED: "the helper says WHAT MATTERS",
# never which key to press)
# ---------------------------------------------------------------------------

# Canonical channel names (FINAL_ARCHITECTURE §6)
FLY_CHANNELS = (
    "target_left", "target_right", "target_center", "target_distance",
    "threat_left", "threat_right", "threat_intensity",
    "attack_opportunity", "avoidance_value", "approach_value",
    "goal_relevance", "low_health_context",
)


@dataclass
class FlyChannels:
    """Compact sensory signal vector for the biological brain.

    target_left/right/center — visual target in left/right/central field
    target_distance   — normalized 0..1 (1 = touching)
    threat_left/right — lateralized imminent-threat signals
    threat_intensity  — overall looming/threat level 0..1
    attack_opportunity — enemy in range / wind-down window 0..1
    avoidance_value   — ENGINEERED helper bias for avoidance (context)
    approach_value    — ENGINEERED helper bias for approach (context)
    goal_relevance    — how relevant the current target is to active goal
    low_health_context — 1 when player health is critically low

    The helper contributes ONLY the context/values fields; visual fields
    derive from perception. None of these say which key to press.
    """
    ts_ns: int = 0
    target_left: float = 0.0
    target_right: float = 0.0
    target_center: float = 0.0
    target_distance: float = 0.0
    threat_left: float = 0.0
    threat_right: float = 0.0
    threat_intensity: float = 0.0
    attack_opportunity: float = 0.0
    avoidance_value: float = 0.0
    approach_value: float = 0.0
    goal_relevance: float = 0.5
    low_health_context: float = 0.0

    def as_vector(self) -> list:
        return [getattr(self, c) for c in FLY_CHANNELS]

    def to_dict(self) -> dict:
        d = {c: getattr(self, c) for c in FLY_CHANNELS}
        d["ts_ns"] = self.ts_ns
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FlyChannels":
        kw = {"ts_ns": int(d.get("ts_ns", 0))}
        for c in FLY_CHANNELS:
            kw[c] = float(d.get(c, 0.0))
        return cls(**kw)

    @classmethod
    def from_observation(cls, obs: WorldObservation,
                         goal_relevance: float = 0.5,
                         avoidance_bias: float = 0.0,
                         approach_bias: float = 0.0) -> "FlyChannels":
        """Encode a WorldObservation into fly channels.

        The helper may supply goal_relevance / avoidance_bias /
        approach_bias (context only). Visual signals derive from the
        observation. Threshold/curve parameters are pre-registered in
        lab/perception/fly_channels.py (single implementation point).
        """
        from .perception.fly_channels import encode
        return encode(obs, goal_relevance=goal_relevance,
                      avoidance_bias=avoidance_bias,
                      approach_bias=approach_bias)


# ---------------------------------------------------------------------------
# intention schema (output of the BIOLOGICAL brain via DN readouts)
# ---------------------------------------------------------------------------

# Canonical combat intention set (FINAL_ARCHITECTURE §6; start smaller if
# needed — validation is per-decoder-config)
INTENTIONS = (
    "STOP", "TURN_LEFT", "TURN_RIGHT",
    "APPROACH", "RETREAT",
    "EVADE_LEFT", "EVADE_RIGHT",
    "DEFEND",
    "ATTACK_LIGHT", "ATTACK_HEAVY", "ATTACK_RANGED",
    "SPECIAL", "ESCAPE",
)


@dataclass
class Intention:
    """Decoded behavioral intention from descending-neuron activity.

    Provenance fields record the neural evidence (population rates) so
    the dashboard/replay can show the intention came from the brain, not
    the helper. The helper NEVER sets `name` directly.
    """
    name: str
    confidence: float = 0.0                 # 0..1 decoder confidence
    ts_ns: int = 0
    brain_chunk_id: int = -1                # which simulation chunk produced it
    dn_rates_hz: dict = field(default_factory=dict)  # readout population -> Hz
    decoder_config: str = ""                # frozen decoder config id
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name not in INTENTIONS:
            raise ValueError(f"unknown intention {self.name!r}; "
                             f"canonical set: {INTENTIONS}")

    def to_dict(self) -> dict:
        return {"name": self.name, "confidence": self.confidence,
                "ts_ns": self.ts_ns, "brain_chunk_id": self.brain_chunk_id,
                "dn_rates_hz": dict(self.dn_rates_hz),
                "decoder_config": self.decoder_config,
                "notes": dict(self.notes)}

    @classmethod
    def from_dict(cls, d: dict) -> "Intention":
        return cls(name=d["name"], confidence=float(d.get("confidence", 0.0)),
                   ts_ns=int(d.get("ts_ns", 0)),
                   brain_chunk_id=int(d.get("brain_chunk_id", -1)),
                   dn_rates_hz=dict(d.get("dn_rates_hz", {})),
                   decoder_config=d.get("decoder_config", ""),
                   notes=dict(d.get("notes", {})))
