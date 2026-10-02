"""Execution adapter for the AI-only GPO branch.

The cloud AI is the only gameplay decision-maker. This worker never invents a
quest/combat strategy. It only keeps the AI's most recent explicit skill alive
long enough to execute against fresh screen geometry.
"""

from __future__ import annotations

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
        self._last_plan_id = -1
        self._last_emit_ns = 0
        self._last_one_shot_plan_id = -1
        self.stats.update({
            "commands": 0,
            "navigation_commands": 0,
            "combat_commands": 0,
            "interaction_commands": 0,
            "one_shot_commands": 0,
            "ignored_plans": 0,
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
              ttl_s: float = 1.0, **extra) -> None:
        self._command_id += 1
        self.command.write({
            "command_id": self._command_id,
            "name": str(name).upper(),
            "source": self.name,
            "reason": str(reason)[:240],
            "ts_ns": now_ns,
            "expires_ns": now_ns + int(float(ttl_s) * 1e9),
            "AI_ONLY": True,
            "ENGINEERED": True,
            **extra,
        }, ts_ns=now_ns)
        self._last_emit_ns = now_ns
        self.stats["commands"] += 1
        print(
            f"[AI-ONLY->EXEC] {name} "
            f"plan={extra.get('coach_plan_id', '-')} "
            f"reason={str(reason)[:120]}",
            flush=True,
        )

    def _publish(self, now_ns: int, plan: dict, obs: dict) -> None:
        target = obs.get("target") or {}
        self.state.write({
            "ts_ns": now_ns,
            "phase": "ai:" + str(plan.get("skill") or "WAIT").lower(),
            "target_type": target.get("type"),
            "target_proximity": target.get("distance"),
            "ai_plan_id": plan.get("plan_id"),
            "ai_skill": plan.get("skill"),
            "ai_target": plan.get("target"),
            "decision_owner": "cloud_ai",
            "fruit_fly_control": False,
            "AI_ONLY": True,
        }, ts_ns=now_ns)

    def _one_shot(self, pid: int) -> bool:
        if pid == self._last_one_shot_plan_id:
            return False
        self._last_one_shot_plan_id = pid
        return True

    def step(self) -> None:
        if not self._armed():
            return
        oenv = self.obs.read()
        penv = self.plan.read()
        if oenv is None or penv is None:
            return
        obs = oenv.payload or {}
        plan = penv.payload or {}
        if plan.get("enabled") is False:
            return

        now = self.clock.now_ns()
        try:
            pid = int(plan.get("plan_id", -1))
            pts = int(plan.get("ts_ns", penv.ts_ns))
            confidence = float(plan.get("confidence", 0.0))
        except (TypeError, ValueError):
            self.stats["ignored_plans"] += 1
            return
        if pid < 0 or now - pts > int(24.0e9) or confidence < 0.60:
            self.stats["ignored_plans"] += 1
            return

        skill = str(plan.get("skill") or "WAIT").upper()
        target = obs.get("target") or {}
        notes = obs.get("notes") or {}
        ttype = str(target.get("type") or "none")
        direction = self._f(target.get("direction"))
        proximity = self._f(target.get("distance"))
        reason = "ai_only:" + str(plan.get("explanation") or skill)
        self._last_plan_id = pid

        # Long-running skills are repeatedly realized against FRESH geometry.
        if skill in {"NAVIGATE_OBJECTIVE", "TRAVEL"}:
            if (ttype in {"recommended_quest_waypoint", "quest_marker",
                          "quest_enemy_marker"}
                    and direction is not None and proximity is not None
                    and now - self._last_emit_ns >= int(0.52e9)):
                self._emit(
                    "STEER_TARGET", now, reason=reason, ttl_s=0.9,
                    direction=direction, hold_s=0.62,
                    coach_plan_id=pid, coach_confidence=confidence,
                    target_type=ttype)
                self.stats["navigation_commands"] += 1

        elif skill in {"TAKE_QUEST", "INTERACT"}:
            yellow = bool(notes.get("quest_marker_detected", False))
            ydir = self._f(notes.get("quest_marker_direction"))
            yprox = self._f(notes.get("quest_marker_proximity"))
            if yellow and ydir is not None and yprox is not None:
                if yprox >= 0.70 and abs(ydir) <= 0.50:
                    if now - self._last_emit_ns >= int(1.25e9):
                        self._emit(
                            "INTERACT_QUEST", now, reason=reason,
                            coach_plan_id=pid,
                            coach_confidence=confidence)
                        self.stats["interaction_commands"] += 1
                elif now - self._last_emit_ns >= int(0.52e9):
                    self._emit(
                        "STEER_TARGET", now, reason=reason, ttl_s=0.9,
                        direction=ydir, hold_s=0.62,
                        coach_plan_id=pid, coach_confidence=confidence,
                        target_type="quest_marker")
                    self.stats["navigation_commands"] += 1

        elif skill == "FIGHT_QUEST_TARGET":
            # Hard identity gate stays below the AI: the plan may only become
            # physical combat while current CV confirms the quest-enemy marker.
            if (ttype == "quest_enemy_marker"
                    and direction is not None and proximity is not None):
                if proximity >= 0.66:
                    if now - self._last_emit_ns >= int(0.42e9):
                        self._emit(
                            "ATTACK_LIGHT", now, reason=reason,
                            coach_plan_id=pid,
                            coach_confidence=confidence)
                        self.stats["combat_commands"] += 1
                elif now - self._last_emit_ns >= int(0.52e9):
                    self._emit(
                        "STEER_TARGET", now, reason=reason, ttl_s=0.9,
                        direction=direction, hold_s=0.62,
                        coach_plan_id=pid, coach_confidence=confidence,
                        target_type=ttype)
                    self.stats["navigation_commands"] += 1

        elif skill == "BLOCK" and self._one_shot(pid):
            self._emit(
                "BLOCK", now, reason=reason, hold_s=0.60,
                coach_plan_id=pid, coach_confidence=confidence)
            self.stats["combat_commands"] += 1
            self.stats["one_shot_commands"] += 1

        elif skill == "EVADE" and self._one_shot(pid):
            self._emit(
                "EVADE_BACK", now, reason=reason,
                coach_plan_id=pid, coach_confidence=confidence)
            self.stats["combat_commands"] += 1
            self.stats["one_shot_commands"] += 1

        elif skill in {"JUMP", "CLIMB", "SPRINT", "GEPPO", "BOARD_SHIP"}:
            if self._one_shot(pid):
                self._emit(
                    skill, now, reason=reason,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["one_shot_commands"] += 1

        elif skill == "BACKTRACK" and self._one_shot(pid):
            self._emit(
                "DASH_BACK", now, reason=reason,
                coach_plan_id=pid, coach_confidence=confidence)
            self.stats["one_shot_commands"] += 1

        elif skill == "GO_AROUND" and self._one_shot(pid):
            control_id = str(plan.get("control_id") or "")
            if control_id in {"move_left", "move_right", "move_backward",
                              "dash_left", "dash_right", "dash_backward"}:
                self._emit(
                    "EXEC_CONTROL", now, reason=reason,
                    control_id=control_id,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["one_shot_commands"] += 1

        elif skill == "USE_HAKI" and self._one_shot(pid):
            cid = str(plan.get("control_id") or "")
            name = ("OBSERVATION_HAKI"
                    if cid == "observation_haki" else "BUSO_HAKI")
            self._emit(
                name, now, reason=reason,
                coach_plan_id=pid, coach_confidence=confidence)
            self.stats["one_shot_commands"] += 1

        elif skill == "EQUIP_SLOT" and self._one_shot(pid):
            cid = str(plan.get("control_id") or "")
            if cid.startswith("equip_slot_"):
                slot = cid.rsplit("_", 1)[-1]
                if slot in set("0123456789"):
                    self._emit(
                        "EQUIP_SLOT", now, reason=reason, slot=slot,
                        coach_plan_id=pid, coach_confidence=confidence)
                    self.stats["one_shot_commands"] += 1

        elif skill == "EXEC_CONTROL" and self._one_shot(pid):
            cid = str(plan.get("control_id") or "")
            if cid:
                self._emit(
                    "EXEC_CONTROL", now, reason=reason, control_id=cid,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["one_shot_commands"] += 1

        elif skill == "USE_OBSERVED_ABILITY" and self._one_shot(pid):
            ability = plan.get("observed_ability") or {}
            binding = str(ability.get("binding") or "").strip().upper()
            label = str(ability.get("label") or "").strip()[:96]
            if binding and label:
                self._emit(
                    "USE_OBSERVED_ABILITY", now, reason=reason,
                    binding=f"key:{binding}", label=label,
                    coach_plan_id=pid, coach_confidence=confidence)
                self.stats["combat_commands"] += 1
                self.stats["one_shot_commands"] += 1

        elif skill in {"UI_CLICK", "BUY_ITEM"} and self._one_shot(pid):
            ui = plan.get("ui_click") or {}
            if bool(ui.get("needed")) and confidence >= 0.85:
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

        # WAIT and REOBSERVE intentionally emit nothing.
        self._publish(now, plan, obs)
