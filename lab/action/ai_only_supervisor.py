"""Execution adapter + perception state for the AI-only GPO branch.

The cloud AI owns gameplay decisions. This worker does not pick quests or
strategies; it (1) publishes stable facts the model needs and (2) realizes the
model's selected skill against fresh screen geometry.

Important separation:
- yellow marker = quest giver
- green marker = recommended travel waypoint
- red quest marker = objective/location cue
- quest_enemy_actor = persistent hostile NPC body associated with that red cue

Only the last item is eligible for melee clicks.
"""

from __future__ import annotations

from collections import deque

from ..bus import Bus, StateChannel
from ..worker import Worker


class AIOnlyAutopilotSupervisor(Worker):
    name = "ai_only_autopilot"

    def __init__(self, bus: Bus, target_hz: float = 8.0):
        super().__init__(bus, target_hz=target_hz)
        self.obs: StateChannel = bus.state("world.observation")
        self.plan: StateChannel = bus.state("coach.plan")
        self.meta: StateChannel = bus.state("action.meta")
        self.command: StateChannel = bus.state("action.command")
        self.state: StateChannel = bus.state("quest.state")

        self._command_id = 0
        self._last_emit_ns = 0
        self._last_one_shot_plan_id = -1
        self._last_log_ns = 0
        self._last_log_sig = None

        # Stable game-state memory derived from perception + actions actually
        # emitted. These are facts for the AI, not hidden gameplay decisions.
        self._quest_status = "unknown"
        self._last_enemy_marker_ns = 0
        self._enemy_marker_since_ns = 0
        self._last_yellow_marker_ns = 0
        self._last_waypoint_ns = 0
        self._last_actor_ns = 0
        self._await_quest_until_ns = 0
        self._last_interact_ns = 0
        self._last_action_ns = 0
        self._last_action_name = None
        self._last_equipped_slot = None

        # Progress/circling monitor. Proximity is the detector contract where
        # larger values mean visually closer.
        self._progress_target = None
        self._best_proximity = None
        self._last_progress_ns = 0
        self._direction_history = deque(maxlen=20)
        self._stuck = False
        self._circling = False

        self.stats.update({
            "commands": 0,
            "navigation_commands": 0,
            "combat_commands": 0,
            "interaction_commands": 0,
            "one_shot_commands": 0,
            "ignored_plans": 0,
            "suppressed_duplicate_logs": 0,
            "quest_interact_suppressed": 0,
            "decision_owner": "cloud_ai",
            "fruit_fly_control": False,
        })

    @staticmethod
    def _f(v, default=None):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _armed(self) -> bool:
        env = self.meta.read()
        return bool(env and (env.payload or {}).get("autonomy"))

    def _emit(self, name: str, now_ns: int, *, reason: str,
              ttl_s: float = 1.4, **extra) -> None:
        self._command_id += 1
        name = str(name).upper()
        self.command.write({
            "command_id": self._command_id,
            "name": name,
            "source": self.name,
            "reason": str(reason)[:240],
            "ts_ns": now_ns,
            "expires_ns": now_ns + int(float(ttl_s) * 1e9),
            "AI_ONLY": True,
            "ENGINEERED": True,
            **extra,
        }, ts_ns=now_ns)
        self._last_emit_ns = now_ns
        self._last_action_ns = now_ns
        self._last_action_name = name
        self.stats["commands"] += 1

        # Persistent servo/attack commands must repeat physically, but the
        # terminal should not print the same objective 20 times.
        sig = (name, extra.get("coach_plan_id"), extra.get("target_type"))
        repeat = name in {"STEER_TARGET", "ATTACK_LIGHT"}
        log_due = (
            sig != self._last_log_sig
            or now_ns - self._last_log_ns >= int(3.5e9)
            or not repeat
        )
        if log_due:
            print(
                f"[AI-ONLY->EXEC] {name} "
                f"plan={extra.get('coach_plan_id', '-')} "
                f"reason={str(reason)[:120]}",
                flush=True,
            )
            self._last_log_sig = sig
            self._last_log_ns = now_ns
        else:
            self.stats["suppressed_duplicate_logs"] += 1

    def _one_shot(self, pid: int) -> bool:
        if pid == self._last_one_shot_plan_id:
            return False
        self._last_one_shot_plan_id = pid
        return True

    def _observe(self, now_ns: int, obs: dict) -> None:
        notes = obs.get("notes") or {}
        target = obs.get("target") or {}
        ttype = str(target.get("type") or "none")
        proximity = self._f(target.get("distance"))
        direction = self._f(target.get("direction"))

        yellow = bool(notes.get("quest_marker_detected"))
        green = bool(notes.get("recommended_waypoint_detected"))
        red = bool(notes.get("quest_enemy_marker_detected"))
        actor = bool(
            notes.get("quest_enemy_actor_visible")
            or ttype == "quest_enemy_actor")

        if yellow:
            self._last_yellow_marker_ns = now_ns
        if green:
            self._last_waypoint_ns = now_ns
        if red:
            if not self._enemy_marker_since_ns:
                self._enemy_marker_since_ns = now_ns
            self._last_enemy_marker_ns = now_ns
        else:
            # Brief detector misses keep quest_status sticky via
            # _last_enemy_marker_ns, but a future close-marker combat fallback
            # must re-establish continuous visual evidence.
            self._enemy_marker_since_ns = 0
        if actor:
            self._last_actor_ns = now_ns

        # Quest status is perception-derived and intentionally sticky across
        # brief occlusion/camera motion.
        if red or actor:
            self._quest_status = "active"
            self._await_quest_until_ns = 0
        elif now_ns < self._await_quest_until_ns:
            self._quest_status = "pending_accept"
        elif (self._last_enemy_marker_ns
              and now_ns - self._last_enemy_marker_ns < int(8.0e9)):
            self._quest_status = "active"
        elif yellow:
            self._quest_status = "available"
        elif green:
            self._quest_status = "travel_to_quest_giver"
        else:
            self._quest_status = "unknown"

        # Progress monitor uses the current sensory target, not the AI prose.
        target_key = ttype if ttype != "none" else None
        if target_key != self._progress_target:
            self._progress_target = target_key
            self._best_proximity = proximity
            self._last_progress_ns = now_ns
            self._direction_history.clear()
            self._stuck = False
            self._circling = False
        elif proximity is not None:
            if (self._best_proximity is None
                    or proximity > self._best_proximity + 0.035):
                self._best_proximity = proximity
                self._last_progress_ns = now_ns
                self._stuck = False
                self._circling = False
            elif (self._last_progress_ns
                  and now_ns - self._last_progress_ns > int(4.5e9)
                  and proximity < 0.72):
                self._stuck = True

        if direction is not None and abs(direction) >= 0.26:
            sign = -1 if direction < 0 else 1
            self._direction_history.append((now_ns, sign))
            cutoff = now_ns - int(5.0e9)
            recent = [
                s for ts, s in self._direction_history if ts >= cutoff]
            flips = sum(
                1 for i in range(1, len(recent))
                if recent[i] != recent[i - 1])
            self._circling = bool(self._stuck and flips >= 4)

    def _publish_state(self, now_ns: int, obs: dict,
                       plan: dict | None = None) -> None:
        target = obs.get("target") or {}
        notes = obs.get("notes") or {}
        actor = notes.get("quest_enemy_actor") or {}
        action_age = (
            round((now_ns - self._last_action_ns) / 1e9, 2)
            if self._last_action_ns else None)
        interact_age = (
            round((now_ns - self._last_interact_ns) / 1e9, 2)
            if self._last_interact_ns else None)
        self.state.write({
            "ts_ns": now_ns,
            "phase": self._quest_status,
            "quest_status": self._quest_status,
            "quest_active": self._quest_status == "active",
            "awaiting_quest_confirmation": (
                now_ns < self._await_quest_until_ns),
            "quest_giver_visible": bool(
                notes.get("quest_marker_detected")),
            "recommended_waypoint_visible": bool(
                notes.get("recommended_waypoint_detected")),
            "quest_enemy_objective_visible": bool(
                notes.get("quest_enemy_marker_detected")),
            "quest_enemy_marker_stable_s": (
                round((now_ns - self._enemy_marker_since_ns) / 1e9, 2)
                if self._enemy_marker_since_ns else 0.0),
            "quest_enemy_actor_visible": bool(
                notes.get("quest_enemy_actor_visible")),
            "quest_enemy_actor": actor,
            "target_type": target.get("type"),
            "target_proximity": target.get("distance"),
            "target_direction": target.get("direction"),
            "stuck": bool(self._stuck),
            "circling": bool(self._circling),
            "last_action": self._last_action_name,
            "last_action_age_s": action_age,
            "last_quest_interact_age_s": interact_age,
            "last_equipped_slot": self._last_equipped_slot,
            "ai_plan_id": (plan or {}).get("plan_id"),
            "ai_skill": (plan or {}).get("skill"),
            "ai_target": (plan or {}).get("target"),
            "decision_owner": "cloud_ai",
            "fruit_fly_control": False,
            "AI_ONLY": True,
        }, ts_ns=now_ns)

    def _steer(self, now_ns: int, *, direction: float, pid: int,
               confidence: float, reason: str, target_type: str,
               hold_s: float = 1.05) -> None:
        if now_ns - self._last_emit_ns < int(0.86e9):
            return
        self._emit(
            "STEER_TARGET", now_ns, reason=reason, ttl_s=1.6,
            direction=float(direction), hold_s=float(hold_s),
            camera_align=True,
            coach_plan_id=pid, coach_confidence=confidence,
            target_type=target_type)
        self.stats["navigation_commands"] += 1

    def step(self) -> None:
        if not self._armed():
            return

        oenv = self.obs.read()
        if oenv is None:
            return
        obs = oenv.payload or {}
        now = self.clock.now_ns()
        self._observe(now, obs)

        penv = self.plan.read()
        if penv is None:
            self._publish_state(now, obs, None)
            return

        plan = penv.payload or {}
        if plan.get("enabled") is False:
            self._publish_state(now, obs, plan)
            return

        try:
            pid = int(plan.get("plan_id", -1))
            pts = int(plan.get("ts_ns", penv.ts_ns))
            confidence = float(plan.get("confidence", 0.0))
        except (TypeError, ValueError):
            self.stats["ignored_plans"] += 1
            self._publish_state(now, obs, plan)
            return
        if pid < 0 or now - pts > int(32.0e9) or confidence < 0.55:
            self.stats["ignored_plans"] += 1
            self._publish_state(now, obs, plan)
            return

        skill = str(plan.get("skill") or "WAIT").upper()
        target = obs.get("target") or {}
        notes = obs.get("notes") or {}
        ttype = str(target.get("type") or "none")
        direction = self._f(target.get("direction"))
        proximity = self._f(target.get("distance"))
        reason = "ai_only:" + str(plan.get("explanation") or skill)

        if skill in {"NAVIGATE_OBJECTIVE", "TRAVEL"}:
            if (ttype in {
                    "recommended_quest_waypoint", "quest_marker",
                    "quest_enemy_marker", "quest_enemy_actor"}
                    and direction is not None and proximity is not None):
                self._steer(
                    now, direction=direction, pid=pid,
                    confidence=confidence, reason=reason,
                    target_type=ttype)

        elif skill in {"TAKE_QUEST", "INTERACT"}:
            # Never retake while perception says an objective is already live.
            if self._quest_status == "active":
                pass
            elif now < self._await_quest_until_ns:
                self.stats["quest_interact_suppressed"] += 1
            else:
                yellow = bool(notes.get("quest_marker_detected", False))
                ydir = self._f(notes.get("quest_marker_direction"))
                yprox = self._f(notes.get("quest_marker_proximity"))
                if yellow and ydir is not None and yprox is not None:
                    # The old 0.70 threshold kept orbiting Robert forever.
                    # At >=0.48 and reasonably centered, T is the useful next
                    # physical primitive. Then wait for dialogue/red objective.
                    if yprox >= 0.48 and abs(ydir) <= 0.72:
                        if (not self._last_interact_ns
                                or now - self._last_interact_ns >= int(4.0e9)):
                            self._emit(
                                "INTERACT_QUEST", now, reason=reason,
                                coach_plan_id=pid,
                                coach_confidence=confidence,
                                target_type="quest_marker")
                            self._last_interact_ns = now
                            self._await_quest_until_ns = now + int(5.5e9)
                            self._quest_status = "pending_accept"
                            self.stats["interaction_commands"] += 1
                    else:
                        self._steer(
                            now, direction=ydir, pid=pid,
                            confidence=confidence, reason=reason,
                            target_type="quest_marker")

        elif skill == "FIGHT_QUEST_TARGET":
            # Never M1 a red objective dot. Require the persistent NPC body
            # associated with the active quest.
            actor = notes.get("quest_enemy_actor") or {}
            actor_visible = bool(
                notes.get("quest_enemy_actor_visible")
                or ttype == "quest_enemy_actor")
            adir = self._f(
                actor.get("direction"),
                direction if ttype == "quest_enemy_actor" else None)
            aprox = self._f(
                actor.get("distance"),
                proximity if ttype == "quest_enemy_actor" else None)
            if actor_visible and adir is not None and aprox is not None:
                if aprox >= 0.43 and abs(adir) <= 0.95:
                    if now - self._last_emit_ns >= int(0.34e9):
                        self._emit(
                            "ATTACK_LIGHT", now, reason=reason,
                            ttl_s=0.8,
                            coach_plan_id=pid,
                            coach_confidence=confidence,
                            target_type="quest_enemy_actor")
                        self.stats["combat_commands"] += 1
                else:
                    self._steer(
                        now, direction=adir, pid=pid,
                        confidence=confidence, reason=reason,
                        target_type="quest_enemy_actor",
                        hold_s=0.92)
            elif (bool(notes.get("quest_enemy_marker_detected"))
                  and direction is not None and proximity is not None):
                # Prefer a resolved body. As a conservative fallback for the
                # game's red dot sitting directly on a quest NPC's head, M1
                # is allowed only when the marker has remained continuously
                # visible, is strongly centered, and is visually very close.
                marker_stable_s = (
                    (now - self._enemy_marker_since_ns) / 1e9
                    if self._enemy_marker_since_ns else 0.0)
                if (proximity >= 0.78 and abs(direction) <= 0.42
                        and marker_stable_s >= 0.80):
                    if now - self._last_emit_ns >= int(0.34e9):
                        self._emit(
                            "ATTACK_LIGHT", now, reason=reason,
                            ttl_s=0.8,
                            coach_plan_id=pid,
                            coach_confidence=confidence,
                            target_type="quest_enemy_marker_close_fallback")
                        self.stats["combat_commands"] += 1
                else:
                    # Objective not yet in verified melee geometry: approach.
                    self._steer(
                        now, direction=direction, pid=pid,
                        confidence=confidence, reason=reason,
                        target_type="quest_enemy_marker")

        elif skill == "BLOCK":
            if self._one_shot(pid):
                self._emit(
                    "BLOCK", now, reason=reason, hold_s=0.65,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["combat_commands"] += 1
                self.stats["one_shot_commands"] += 1

        elif skill == "EVADE":
            if self._one_shot(pid):
                self._emit(
                    "EVADE_BACK", now, reason=reason,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["combat_commands"] += 1
                self.stats["one_shot_commands"] += 1

        elif skill == "JUMP":
            if now - self._last_emit_ns >= int(0.95e9):
                self._emit(
                    "JUMP", now, reason=reason,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["one_shot_commands"] += 1

        elif skill == "CLIMB":
            if now - self._last_emit_ns >= int(1.15e9):
                self._emit(
                    "CLIMB", now, reason=reason, ttl_s=1.8,
                    hold_s=1.15,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["one_shot_commands"] += 1

        elif skill == "GEPPO":
            if now - self._last_emit_ns >= int(0.52e9):
                self._emit(
                    "GEPPO", now, reason=reason,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["one_shot_commands"] += 1

        elif skill == "SPRINT":
            if now - self._last_emit_ns >= int(3.0e9):
                self._emit(
                    "SPRINT", now, reason=reason,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["one_shot_commands"] += 1

        elif skill == "BOARD_SHIP":
            if self._one_shot(pid):
                self._emit(
                    "BOARD_SHIP", now, reason=reason,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["one_shot_commands"] += 1

        elif skill == "BACKTRACK":
            if self._one_shot(pid):
                self._emit(
                    "DASH_BACK", now, reason=reason,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["one_shot_commands"] += 1

        elif skill in {"LOOK_LEFT", "LOOK_RIGHT"}:
            if self._one_shot(pid):
                self._emit(
                    "SEARCH_CAMERA", now, reason=reason,
                    dx=(-105 if skill == "LOOK_LEFT" else 105),
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["one_shot_commands"] += 1

        elif skill == "GO_AROUND":
            if self._one_shot(pid):
                control_id = str(plan.get("control_id") or "")
                if control_id in {
                        "move_left", "move_right", "move_backward",
                        "dash_left", "dash_right", "dash_backward"}:
                    self._emit(
                        "EXEC_CONTROL", now, reason=reason,
                        control_id=control_id,
                        coach_plan_id=pid, coach_confidence=confidence)
                    self.stats["one_shot_commands"] += 1

        elif skill == "USE_HAKI":
            if self._one_shot(pid):
                cid = str(plan.get("control_id") or "")
                name = (
                    "OBSERVATION_HAKI"
                    if cid == "observation_haki"
                    else "BUSO_HAKI")
                self._emit(
                    name, now, reason=reason,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["one_shot_commands"] += 1

        elif skill == "EQUIP_SLOT":
            if self._one_shot(pid):
                cid = str(plan.get("control_id") or "")
                if cid.startswith("equip_slot_"):
                    slot = cid.rsplit("_", 1)[-1]
                    if slot in set("0123456789"):
                        self._emit(
                            "EQUIP_SLOT", now, reason=reason, slot=slot,
                            coach_plan_id=pid,
                            coach_confidence=confidence)
                        self._last_equipped_slot = slot
                        self.stats["one_shot_commands"] += 1

        elif skill == "EXEC_CONTROL":
            if self._one_shot(pid):
                cid = str(plan.get("control_id") or "")
                if cid:
                    self._emit(
                        "EXEC_CONTROL", now, reason=reason,
                        control_id=cid,
                        coach_plan_id=pid,
                        coach_confidence=confidence)
                    if cid.startswith("equip_slot_"):
                        self._last_equipped_slot = cid.rsplit("_", 1)[-1]
                    self.stats["one_shot_commands"] += 1

        elif skill == "USE_OBSERVED_ABILITY":
            if self._one_shot(pid):
                ability = plan.get("observed_ability") or {}
                binding = str(
                    ability.get("binding") or "").strip().upper()
                label = str(ability.get("label") or "").strip()[:96]
                if binding and label:
                    self._emit(
                        "USE_OBSERVED_ABILITY", now, reason=reason,
                        binding=f"key:{binding}", label=label,
                        coach_plan_id=pid,
                        coach_confidence=confidence)
                    self.stats["combat_commands"] += 1
                    self.stats["one_shot_commands"] += 1

        elif skill in {"UI_CLICK", "BUY_ITEM"}:
            if self._one_shot(pid):
                ui = plan.get("ui_click") or {}
                if bool(ui.get("needed")) and confidence >= 0.80:
                    try:
                        x = float(ui.get("x_norm"))
                        y = float(ui.get("y_norm"))
                    except (TypeError, ValueError):
                        x = y = -1.0
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                        self._emit(
                            "UI_CLICK", now, reason=reason,
                            x_norm=x, y_norm=y, confidence=confidence,
                            ui_context=True,
                            ui_label=str(ui.get("label") or "")[:96],
                            purchase_intent=(skill == "BUY_ITEM"),
                            coach_plan_id=pid,
                            coach_confidence=confidence)
                        self.stats["one_shot_commands"] += 1

        # WAIT and REOBSERVE intentionally emit no physical command.
        self._publish_state(now, obs, plan)
