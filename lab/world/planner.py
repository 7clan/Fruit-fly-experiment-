"""High-level planner / helper [ENGINEERED].

FINAL_ARCHITECTURE §13/§21:
  * The helper handles what the fly cannot represent: semantic concepts,
    quests, world knowledge, long-term goals, equipment comparison,
    route planning, progression planning.
  * It outputs GOALS / CONTEXT / VALUE / RELEVANCE — NOT key-by-key
    scripts. Example: GOAL "defeat enemy", not "press W 0.7 s, press Q".
  * NO LLM IN THE COMBAT LOOP: this planner runs at ~0.5–2 Hz or
    event-driven on its own worker; if it is delayed, the fast
    controller (perception → fly → intention → resolver → executor)
    keeps operating entirely.

Current implementation: an explicit goal stack (no learning, no LLM) —
goals are selected from the world model state by deterministic rules.
Gate-8+ replaces the internals (quest logic, navigation); the interface
below is the stable contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..bus import Bus, StateChannel
from ..worker import Worker


@dataclass
class Goal:
    gid: str
    kind: str                 # "combat" | "travel" | "quest" | "observe" ...
    label: str
    target_eid: str = ""
    priority: float = 0.5
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class PlannerWorker(Worker):
    """Event-driven high-level goal selection (ENGINEERED helper)."""

    name = "planner"
    TOPIC_OBS = "world.observation"
    TOPIC_GOAL = "helper.goal"          # state channel (fly uses relevance only)

    def __init__(self, bus: Bus, target_hz: float = 1.0):
        super().__init__(bus, target_hz=target_hz)
        self.obs: StateChannel = bus.state(self.TOPIC_OBS)
        self.goal_state: StateChannel = bus.state(self.TOPIC_GOAL)
        self.current: Goal | None = None

    def _select_goal(self, obs: dict) -> Goal:
        """Deterministic goal rules (explicit; extend per Gate-8 prereg)."""
        enemies = obs.get("enemies", [])
        if enemies:
            e = enemies[0]
            return Goal(gid="g-combat", kind="combat",
                        label="defeat enemy",
                        target_eid="enemy:live:0",
                        priority=0.8,
                        context={"enemy_type": e.get("type", "unknown"),
                                 "distance": e.get("distance"),
                                 "attacking": e.get("attacking")})
        target = obs.get("target", {})
        if target.get("confidence", 0.0) > 0.3:
            return Goal(gid="g-approach", kind="travel",
                        label="approach objective",
                        target_eid="target:live",
                        priority=0.5,
                        context={"target_type": target.get("type")})
        return Goal(gid="g-observe", kind="observe", label="observe world",
                    priority=0.2, context={})

    def step(self) -> None:
        snap = self.obs.read()
        if snap is None:
            return
        obs = snap.payload
        goal = self._select_goal(obs)
        # relevance of the CURRENT target to the goal (helper says WHAT
        # MATTERS; the fly still decides behavior)
        if goal.kind == "combat":
            relevance = 0.9
            avoidance_bias = 0.6 if obs.get("player", {}).get("health") is not None \
                and obs["player"].get("health_units") == "fraction" \
                and obs["player"]["health"] < 0.3 else 0.25
            approach_bias = 0.4
        elif goal.kind == "travel":
            relevance = 0.6
            avoidance_bias, approach_bias = 0.1, 0.6
        else:
            relevance = 0.3
            avoidance_bias, approach_bias = 0.1, 0.2
        payload = {"ts_ns": self.clock.now_ns(), "goal": goal.to_dict(),
                   "goal_relevance": relevance,
                   "avoidance_bias": avoidance_bias,
                   "approach_bias": approach_bias,
                   "ENGINEERED": True}
        self.current = goal
        self.goal_state.write(payload, ts_ns=payload["ts_ns"])
