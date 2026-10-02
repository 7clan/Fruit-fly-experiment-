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
        self.coach: StateChannel = bus.state("coach.plan")
        self.command: StateChannel = bus.state("action.command")
        self.state: StateChannel = bus.state("quest.state")
        self.value = value_table
        self.loadout = str(loadout or "default_melee").lower()

        self._command_id = 0
        self._last_coach_plan_id = -1
        self._last_coach_seen_plan_id = -1
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
        self._last_steer_ns = 0
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
            "coach_commands": 0,
            "coach_plans_seen": 0,
            "semantic_steers": 0,
            "recovery_dashes": 0,
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

        coach_env = self.coach.read()
        coach_skill = str(
            ((coach_env.payload or {}).get("skill") if coach_env else "")
            or "").upper()
        locomoting = (
            fly_intention in {
                "TURN_LEFT", "TURN_RIGHT", "APPROACH", "RETREAT"}
            or coach_skill == "NAVIGATE_OBJECTIVE")
        # Do not interpret every stall as a climbable wall. The live evidence
        # showed 30 blind CLIMB commands while the avatar was simply pinned
        # against scenery. Use a bounded recovery sequence instead.
        if (locomoting and now_ns - self._last_progress_ns > int(4.0e9)
                and now_ns - self._last_command_ns > int(1.1e9)):
            self._last_progress_ns = now_ns
            self._stall_recoveries += 1
            sequence = ("JUMP", "DASH_LEFT", "DASH_RIGHT", "DASH_BACK")
            idx = min(self._stall_recoveries - 1, len(sequence) - 1)
            return sequence[idx]
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

    def _consume_coach_plan(
            self, now_ns: int, target: dict, notes: dict,
            yellow_quest: bool, yellow_proximity: float | None,
            yellow_direction: float | None) -> bool:
        """Bridge ONE fresh semantic-coach plan into an approved command.

        Immediate damage defense remains rule/reflex driven above this layer.
        The coach never supplies raw keys. Navigation direction normally
        remains the fly's job; EXEC_CONTROL is a bounded verified-control
        escape hatch for explicit game semantics.
        """
        env = self.coach.read()
        if env is None:
            return False
        plan = env.payload or {}
        if plan.get("enabled") is False:
            return False
        try:
            pid = int(plan.get("plan_id", -1))
        except (TypeError, ValueError):
            return False
        if pid < 0 or pid == self._last_coach_plan_id:
            return False

        try:
            confidence = float(plan.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        plan_ts = int(plan.get("ts_ns", env.ts_ns))
        if now_ns - plan_ts > int(20.0e9) or confidence < 0.62:
            self._last_coach_plan_id = pid
            return False

        skill = str(plan.get("skill") or "WAIT").upper()
        target_type = str(target.get("type") or "none")
        proximity = self._f(target.get("distance"))
        if pid != self._last_coach_seen_plan_id:
            self._last_coach_seen_plan_id = pid
            self.stats["coach_plans_seen"] += 1

        name = None
        extra = {}
        if skill in {"TAKE_QUEST", "INTERACT"}:
            if (yellow_quest and yellow_proximity is not None
                    and yellow_proximity >= 0.68
                    and yellow_direction is not None
                    and abs(yellow_direction) <= 0.50):
                name = "INTERACT_QUEST"
        elif skill == "FIGHT_QUEST_TARGET":
            if (target_type == "quest_enemy_marker"
                    and proximity is not None and proximity >= 0.66):
                name = "ATTACK_LIGHT"
        elif skill == "BLOCK" and target_type == "quest_enemy_marker":
            name = "BLOCK"
            extra["hold_s"] = 0.55
        elif skill == "EVADE" and target_type == "quest_enemy_marker":
            name = "EVADE_BACK"
        elif skill == "JUMP":
            name = "JUMP"
        elif skill == "CLIMB":
            name = "CLIMB"
        elif skill == "SPRINT":
            name = "SPRINT"
        elif skill == "GEPPO":
            name = "GEPPO"
        elif skill == "USE_HAKI":
            control_id = str(plan.get("control_id") or "")
            name = (
                "OBSERVATION_HAKI"
                if control_id == "observation_haki"
                else "BUSO_HAKI")
        elif skill == "EQUIP_SLOT":
            control_id = str(plan.get("control_id") or "")
            if control_id.startswith("equip_slot_"):
                slot = control_id.rsplit("_", 1)[-1]
                if slot in set("0123456789"):
                    name = "EQUIP_SLOT"
                    extra["slot"] = slot
        elif skill == "EXEC_CONTROL":
            control_id = str(plan.get("control_id") or "")
            if control_id:
                name = "EXEC_CONTROL"
                extra["control_id"] = control_id
        elif skill == "USE_OBSERVED_ABILITY":
            ability = plan.get("observed_ability") or {}
            binding = str(ability.get("binding") or "").strip().upper()
            label = str(ability.get("label") or "").strip()[:96]
            if binding and label:
                name = "USE_OBSERVED_ABILITY"
                extra["binding"] = f"key:{binding}"
                extra["label"] = label
        elif skill == "GO_AROUND":
            control_id = str(plan.get("control_id") or "")
            if control_id in {"move_left", "move_right", "move_backward"}:
                name = "EXEC_CONTROL"
                extra["control_id"] = control_id
        elif skill == "BOARD_SHIP":
            name = "BOARD_SHIP"
        elif skill == "BACKTRACK":
            # One bounded retreat/dash lets the next visual observation
            # choose a different route without replacing normal fly steering.
            name = "DASH_BACK"
        elif skill in {"UI_CLICK", "BUY_ITEM"}:
            # Buying is still just a verified visible in-game UI click. The
            # coach must point at the actual labeled button in the current
            # screenshot; no blind coordinates or hidden store actions.
            ui = plan.get("ui_click") or {}
            if bool(ui.get("needed")) and confidence >= 0.85:
                try:
                    x = float(ui.get("x_norm"))
                    y = float(ui.get("y_norm"))
                except (TypeError, ValueError):
                    x = y = -1.0
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    name = "UI_CLICK"
                    extra.update({
                        "x_norm": x,
                        "y_norm": y,
                        "confidence": confidence,
                        "ui_context": True,
                        "ui_label": str(ui.get("label") or "")[:96],
                        "purchase_intent": skill == "BUY_ITEM",
                    })

        if not name:
            # Geometry may not be actionable yet (e.g. TAKE_QUEST while the
            # NPC is still several metres away). Keep the plan pending so it
            # can fire once fresh CV reaches the required range, until the
            # plan's normal 20 s expiry.
            return False
        self._last_coach_plan_id = pid
        self._phase = "coach:" + skill.lower()
        self._emit(
            name, now_ns,
            reason=(
                f"semantic_coach:{plan.get('explanation', '')}"[:220]),
            source="semantic_coach",
            coach_plan_id=pid,
            coach_confidence=confidence,
            **extra,
        )
        self.stats["coach_commands"] += 1
        if name == "ATTACK_LIGHT":
            self.stats["attacks"] += 1
            self._last_attack_ns = now_ns
        elif name == "BLOCK":
            self.stats["blocks"] += 1
            self._last_block_ns = now_ns
        elif name == "EVADE_BACK":
            self.stats["evades"] += 1
        elif name == "INTERACT_QUEST":
            self.stats["quest_interacts"] += 1
            self._last_interact_ns = now_ns
        elif name == "JUMP":
            self.stats["jumps"] += 1
        elif name == "CLIMB":
            self.stats["climbs"] += 1
        elif name == "SPRINT":
            self.stats["sprints"] += 1
            self._last_sprint_ns = now_ns
        return True

    def _semantic_navigation_assist(
            self, now_ns: int, target: dict, fly_intention: str,
            yellow_quest: bool, yellow_proximity: float | None,
            yellow_direction: float | None) -> bool:
        """Keep a selected navigation skill aligned to FRESH visual geometry.

        The canonical whole-brain chunk takes ~6-10 wall-seconds on the target
        laptop. Its biological output remains part of the experiment, but a
        stale turn cannot be the only real-time steering signal. The semantic
        coach chooses the goal; this bounded servo only tracks that already
        selected visible target using W/A/D, never the mouse/camera.
        """
        env = self.coach.read()
        if env is None:
            return False
        plan = env.payload or {}
        if str(plan.get("skill") or "").upper() != "NAVIGATE_OBJECTIVE":
            return False
        try:
            confidence = float(plan.get("confidence", 0.0))
            plan_ts = int(plan.get("ts_ns", env.ts_ns))
        except (TypeError, ValueError):
            return False
        if confidence < 0.62 or now_ns - plan_ts > int(65.0e9):
            return False

        target_type = str(target.get("type") or "none")
        card = str(plan.get("skill_card") or "")
        try:
            if (card == "quest_accept" and yellow_quest
                    and yellow_proximity is not None
                    and yellow_direction is not None):
                # Red NPC diamonds can coexist with the yellow quest giver.
                # Follow the semantic goal, not whichever color detector wrote
                # the primary target last.
                target_type = "quest_marker"
                target_conf = 0.90
                direction = float(yellow_direction)
                proximity = float(yellow_proximity)
            else:
                if target_type not in {
                        "recommended_quest_waypoint", "quest_marker",
                        "quest_enemy_marker"}:
                    return False
                target_conf = float(target.get("confidence", 0.0))
                direction = float(target.get("direction"))
                proximity = float(target.get("distance"))
        except (TypeError, ValueError):
            return False
        if target_conf < 0.60:
            return False

        # Interaction/combat takes over at useful range.
        if (yellow_quest and yellow_proximity is not None
                and yellow_direction is not None
                and yellow_proximity >= 0.68
                and abs(yellow_direction) <= 0.50):
            return False
        if target_type == "quest_enemy_marker" and proximity >= 0.66:
            return False

        # Preserve urgent biological safety-like outputs.
        if fly_intention in {"RETREAT", "ESCAPE", "DEFEND"}:
            return False
        if now_ns - self._last_steer_ns < int(0.65e9):
            return False

        self._emit(
            "STEER_TARGET", now_ns,
            reason="fresh_visual_servo_for_selected_navigation_skill",
            source="semantic_navigation_assist",
            direction=direction,
            hold_s=0.58,
            coach_plan_id=plan.get("plan_id"),
            coach_provider=plan.get("provider"),
            fly_intention=fly_intention,
            target_type=target_type,
        )
        self._last_steer_ns = now_ns
        self.stats["semantic_steers"] += 1
        self._phase = "navigate_servo"
        return True

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

        if self._consume_coach_plan(
                now, target, notes, yellow_quest,
                yellow_proximity, yellow_direction):
            self._publish_state(
                now, target, health, stamina, fly_intention)
            return

        if self._semantic_navigation_assist(
                now, target, fly_intention,
                yellow_quest, yellow_proximity, yellow_direction):
            self._publish_state(
                now, target, health, stamina, fly_intention)
            return

        # The autonomous agent never moves the user's camera. Character
        # steering and semantic recovery use W/A/D/S, jump/climb/dash and
        # reobservation only.

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
            elif recovery == "CLIMB":
                self.stats["climbs"] += 1
            else:
                self.stats["recovery_dashes"] += 1
            self._publish_state(
                now, target, health, stamina, fly_intention)
            return

        # Yellow QUEST and the green Recommended Quest circle can be
        # visible at the same time. Fast vision preserves both cues.
        coach_env = self.coach.read()
        coach_skill = str(
            ((coach_env.payload or {}).get("skill")
             if coach_env else "") or "").upper()

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

        elif (target_type == "quest_enemy_marker"
                and (not yellow_quest
                     or coach_skill == "FIGHT_QUEST_TARGET")):
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
            # No objective in view: wait for the semantic coach / next game
            # observation instead of taking over the user's camera.
            self._phase = "await_visible_target"

        self._publish_state(now, target, health, stamina, fly_intention)
