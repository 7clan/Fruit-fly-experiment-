"""Low-cost GPO quest/combat supervisor [ENGINEERED].

The canonical fly still supplies locomotor intentions (turn/approach/retreat).
This worker handles game semantics the validated DN decoder does not encode:
quest interaction, obstacle recovery and conservative starter-PvE primitives.

It never writes raw arbitrary bindings. It emits a small semantic command set
which MotorExecutor maps through a hard QuestingBackend allowlist.

The first learning loop is deliberately narrow: choose BLOCK vs EVADE_BACK
after damage and update the engineered ValueTable from subsequent health loss.
This is NOT biological mushroom-body learning.
"""

from __future__ import annotations

from ..bus import Bus, StateChannel
from ..worker import Worker
from ..world.value import ValueTable


class QuestCombatSupervisor(Worker):
    name = "quest_combat_supervisor"

    def __init__(self, bus: Bus, value_table: ValueTable,
                 target_hz: float = 4.0,
                 loadout: str = "default_melee"):
        super().__init__(bus, target_hz=target_hz)
        self.obs: StateChannel = bus.state("world.observation")
        self.brain: StateChannel = bus.state("brain.output")
        self.action_meta: StateChannel = bus.state("action.meta")
        self.command: StateChannel = bus.state("action.command")
        self.state: StateChannel = bus.state("quest.state")
        self.value = value_table
        self.loadout = str(loadout or "default_melee").lower()

        self._command_id = 0
        self._phase = "observe"
        self._last_health = None
        self._last_damage_ns = 0
        self._last_command_ns = 0
        self._last_interact_ns = 0
        self._last_search_ns = 0
        self._last_center_ns = 0
        self._last_sprint_ns = 0
        self._last_block_ns = 0
        self._last_attack_ns = 0
        self._last_e_ns = 0
        self._last_r_ns = 0
        self._target_type = None
        self._best_proximity = None
        self._last_progress_ns = 0
        self._stall_recoveries = 0
        self._pending_defense = None
        self.stats.update({
            "commands": 0,
            "quest_interacts": 0,
            "attacks": 0,
            "blocks": 0,
            "evades": 0,
            "jumps": 0,
            "climbs": 0,
            "searches": 0,
            "camera_recenters": 0,
            "sprints": 0,
            "defense_updates": 0,
        })

    def _armed(self) -> bool:
        env = self.action_meta.read()
        return bool(env and (env.payload or {}).get("autonomy"))

    def _emit(self, name: str, now_ns: int, *, reason: str,
              ttl_s: float = 0.8, **extra) -> None:
        self._command_id += 1
        payload = {
            "command_id": self._command_id,
            "name": str(name).upper(),
            "source": self.name,
            "reason": reason,
            "ts_ns": now_ns,
            "expires_ns": now_ns + int(float(ttl_s) * 1e9),
            "ENGINEERED": True,
            **extra,
        }
        self.command.write(payload, ts_ns=now_ns)
        self._last_command_ns = now_ns
        self.stats["commands"] += 1

    @staticmethod
    def _f(v, default=None):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _brain_intention(self) -> str:
        env = self.brain.read()
        if env is None:
            return "STOP"
        return str(((env.payload or {}).get("intention") or {}).get(
            "name", "STOP"))

    def _choose_defense(self) -> str:
        ctx = ("gpo", "defense", "quest_enemy")
        block_v = self.value.value(ctx, "block")
        evade_v = self.value.value(ctx, "evade_back")
        snap = self.value.snapshot().get("|".join(ctx), {})
        block_n = int((snap.get("block") or {}).get("uses", 0))
        evade_n = int((snap.get("evade_back") or {}).get("uses", 0))
        # Force one observation of each before exploiting the better result.
        if block_n == 0:
            return "BLOCK"
        if evade_n == 0:
            return "EVADE_BACK"
        return "BLOCK" if block_v >= evade_v else "EVADE_BACK"

    def _begin_defense_eval(self, action: str, health: float,
                            now_ns: int) -> None:
        self._pending_defense = {
            "action": action.lower(),
            "health": float(health),
            "ts_ns": now_ns,
        }

    def _evaluate_defense(self, health: float | None, now_ns: int) -> None:
        p = self._pending_defense
        if not p or health is None:
            return
        if now_ns - int(p["ts_ns"]) < int(1.1e9):
            return
        # Success means the selected defense prevented another meaningful
        # health loss during the short follow-up window.
        success = float(health) >= float(p["health"]) - 0.025
        self.value.observe(
            ("gpo", "defense", "quest_enemy"),
            str(p["action"]), bool(success))
        self.stats["defense_updates"] += 1
        self._pending_defense = None

    def _update_progress(self, target_type: str, proximity: float | None,
                         now_ns: int, fly_intention: str) -> str | None:
        if (proximity is None or target_type not in {
                "recommended_quest_waypoint", "quest_marker",
                "quest_enemy_marker"}):
            self._target_type = target_type
            self._best_proximity = None
            self._last_progress_ns = now_ns
            self._stall_recoveries = 0
            return None

        if target_type != self._target_type:
            self._target_type = target_type
            self._best_proximity = proximity
            self._last_progress_ns = now_ns
            self._stall_recoveries = 0
            return None

        if self._best_proximity is None or proximity > self._best_proximity + 0.025:
            self._best_proximity = proximity
            self._last_progress_ns = now_ns
            self._stall_recoveries = 0
            return None

        locomoting = fly_intention in {
            "TURN_LEFT", "TURN_RIGHT", "APPROACH", "RETREAT"}
        if (locomoting and now_ns - self._last_progress_ns > int(3.5e9)
                and now_ns - self._last_command_ns > int(1.0e9)):
            self._last_progress_ns = now_ns
            self._stall_recoveries += 1
            if self._stall_recoveries >= 3:
                return "CLIMB"
            return "JUMP"
        return None

    def _publish_state(self, now_ns: int, target: dict,
                       health, stamina, fly_intention: str) -> None:
        self.state.write({
            "ts_ns": now_ns,
            "phase": self._phase,
            "target_type": target.get("type"),
            "target_proximity": target.get("distance"),
            "fly_intention": fly_intention,
            "health": health,
            "stamina": stamina,
            "loadout": self.loadout,
            "defense_values": {
                "block": round(self.value.value(
                    ("gpo", "defense", "quest_enemy"), "block"), 3),
                "evade_back": round(self.value.value(
                    ("gpo", "defense", "quest_enemy"), "evade_back"), 3),
            },
            "ENGINEERED": True,
        }, ts_ns=now_ns)

    def step(self) -> None:
        env = self.obs.read()
        if env is None:
            return
        obs = env.payload or {}
        now = self.clock.now_ns()
        target = obs.get("target") or {}
        notes = obs.get("notes") or {}
        player = obs.get("player") or {}
        target_type = str(target.get("type") or "none")
        proximity = self._f(target.get("distance"))
        direction = self._f(target.get("direction"))
        yellow_quest = bool(notes.get("quest_marker_detected", False))
        yellow_proximity = self._f(notes.get("quest_marker_proximity"))
        yellow_direction = self._f(notes.get("quest_marker_direction"))
        health = self._f(player.get("health"))
        stamina = self._f(player.get("stamina"))
        fly_intention = self._brain_intention()

        self._evaluate_defense(health, now)

        damage = 0.0
        if health is not None and self._last_health is not None:
            damage = max(0.0, self._last_health - health)
            if damage >= 0.025:
                self._last_damage_ns = now
        if health is not None:
            self._last_health = health

        if not self._armed():
            self._phase = "paused"
            self._publish_state(
                now, target, health, stamina, fly_intention)
            return

        # Critical survival beats quest progress.
        if target_type == "quest_enemy_marker" and health is not None:
            if health <= 0.22 and now - self._last_command_ns > int(0.7e9):
                self._phase = "low_health_escape"
                self._emit("EVADE_BACK", now, reason="low_health")
                self.stats["evades"] += 1
                self._begin_defense_eval("evade_back", health, now)
                self._publish_state(
                    now, target, health, stamina, fly_intention)
                return

            if damage >= 0.035 and now - self._last_command_ns > int(0.45e9):
                choice = self._choose_defense()
                self._phase = "defend"
                self._emit(choice, now, reason=f"health_drop:{damage:.3f}")
                if choice == "BLOCK":
                    self.stats["blocks"] += 1
                    self._last_block_ns = now
                else:
                    self.stats["evades"] += 1
                self._begin_defense_eval(choice.lower(), health, now)
                self._publish_state(
                    now, target, health, stamina, fly_intention)
                return

        # Camera is part of the fly's sensory apparatus, not a locomotor
        # decision. Keep a visible navigation cue near the center so the slow
        # canonical brain receives a stable bearing instead of requiring the
        # user to drag the camera by hand. This never chooses W/A/D/S.
        if (target_type in {
                "recommended_quest_waypoint", "quest_marker",
                "quest_enemy_marker"}
                and direction is not None
                and abs(direction) >= 0.65
                # Do not swing the camera while the canonical decoder is
                # already asking for a turn. On the target laptop a 50 ms
                # biological chunk takes several wall-seconds; aggressive
                # recentering during that interval makes its bearing stale.
                and fly_intention in {"STOP", "APPROACH"}
                and now - self._last_center_ns > int(2.5e9)
                and now - self._last_damage_ns > int(0.65e9)):
            dx = int(max(-70, min(70, direction * 90.0)))
            if abs(dx) >= 14:
                self._emit(
                    "SEARCH_CAMERA", now,
                    reason="recenter_visible_navigation_cue", dx=dx)
                self._last_center_ns = now
                self.stats["camera_recenters"] += 1
                self._phase = "camera_recenter"
                self._publish_state(
                    now, target, health, stamina, fly_intention)
                return

        # Long unobstructed travel can use the game's sprint affordance, but
        # only while the biological decoder is already asking to APPROACH.
        if (target_type == "recommended_quest_waypoint"
                and fly_intention == "APPROACH"
                and proximity is not None and proximity < 0.45
                and now - self._last_sprint_ns > int(3.0e9)
                and now - self._last_damage_ns > int(1.0e9)):
            self._emit("SPRINT", now, reason="far_waypoint_approach")
            self._last_sprint_ns = now
            self.stats["sprints"] += 1
            self._phase = "travel_sprint"
            self._publish_state(
                now, target, health, stamina, fly_intention)
            return

        # Movement primitive recovery remains subordinate to the current
        # biological locomotor intention.
        recovery = self._update_progress(
            target_type, proximity, now, fly_intention)
        if recovery and now - self._last_damage_ns > int(0.8e9):
            self._phase = "obstacle_recovery"
            self._emit(recovery, now, reason="target_progress_stalled")
            if recovery == "JUMP":
                self.stats["jumps"] += 1
            else:
                self.stats["climbs"] += 1
            self._publish_state(
                now, target, health, stamina, fly_intention)
            return

        # Yellow QUEST and the green Recommended Quest circle can be
        # visible at the same time. Fast vision preserves both cues. The
        # green circle remains the biological navigation target, while a
        # sufficiently close/centered yellow cue triggers NPC interaction.
        if (target_type != "quest_enemy_marker" and yellow_quest
                and yellow_proximity is not None
                and yellow_proximity >= 0.72
                and yellow_direction is not None
                and abs(yellow_direction) <= 0.45):
            self._phase = "quest_giver"
            if now - self._last_interact_ns > int(1.6e9):
                self._emit(
                    "INTERACT_QUEST", now,
                    reason="yellow_quest_marker_close_and_centered")
                self._last_interact_ns = now
                self.stats["quest_interacts"] += 1

        elif target_type == "quest_enemy_marker":
            if proximity is not None and proximity >= 0.70:
                self._phase = "combat"
                # Regular block cycling gives the slow whole-brain loop a
                # survivable PvE envelope; perfect-block timing is NOT claimed.
                if (now - self._last_block_ns > int(2.2e9)
                        and now - self._last_command_ns > int(0.45e9)):
                    self._emit("BLOCK", now, reason="enemy_close_guard_cycle")
                    self._last_block_ns = now
                    self.stats["blocks"] += 1
                    if health is not None:
                        self._begin_defense_eval("block", health, now)
                elif now - self._last_attack_ns > int(0.38e9):
                    # Current live HUD + GPO wiki confirm starter Melee E/R.
                    # Use stamina-aware cooldowns, otherwise fall back to M1.
                    if (self.loadout == "default_melee"
                            and (stamina is None or stamina >= 0.30)
                            and now - self._last_e_ns > int(15.0e9)):
                        self._emit(
                            "GUT_PUNCH", now,
                            reason="starter_melee_block_break_ready")
                        self._last_e_ns = now
                    elif (self.loadout == "default_melee"
                            and (stamina is None or stamina >= 0.35)
                            and now - self._last_r_ns > int(20.0e9)):
                        self._emit(
                            "GROUND_SMASH", now,
                            reason="starter_melee_aoe_ready")
                        self._last_r_ns = now
                    else:
                        self._emit(
                            "ATTACK_LIGHT", now,
                            reason="confirmed_quest_enemy_close")
                    self._last_attack_ns = now
                    self.stats["attacks"] += 1
            else:
                self._phase = "hunt_enemy"

        elif target_type == "recommended_quest_waypoint":
            self._phase = "travel"

        else:
            self._phase = "search"
            # If the tracker disappears, slowly scan the camera instead of
            # requiring the user to move it by hand.
            if (now - self._last_search_ns > int(1.5e9)
                    and now - self._last_damage_ns > int(1.0e9)):
                self._emit(
                    "SEARCH_CAMERA", now, reason="no_navigation_target",
                    dx=32)
                self._last_search_ns = now
                self.stats["searches"] += 1

        self._publish_state(now, target, health, stamina, fly_intention)
