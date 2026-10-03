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
        self.ai_track: StateChannel = bus.state("ai.visual.track")

        self._command_id = 0
        self._last_emit_ns = 0
        self._last_one_shot_plan_id = -1
        self._last_fight_equip_plan_id = -1
        self._last_combat_bundle_ns = 0
        self._last_log_ns = 0
        self._last_log_sig = None

        # Stable game-state memory derived from perception + actions actually
        # emitted. These are facts for the AI, not hidden gameplay decisions.
        self._quest_status = "unknown"
        self._quest_active_latched = False
        self._quest_completed_until_ns = 0
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
        self._action_epoch = 0
        self._ui_epoch = 0
        self._last_ui_click_ns = 0
        self._last_ui_sig = None
        self._ai_enemy_confirmed_until_ns = 0
        self._repeat_counts = {}
        self._repeat_last_ns = {}

        # Progress/circling monitor. Proximity is the detector contract where
        # larger values mean visually closer.
        self._progress_target = None
        self._best_proximity = None
        self._last_progress_ns = 0
        self._direction_history = deque(maxlen=20)
        self._stuck = False
        self._circling = False
        self._recovery_stage = 0
        self._last_recovery_ns = 0
        self._recovery_target = None

        self.stats.update({
            "commands": 0,
            "navigation_commands": 0,
            "combat_commands": 0,
            "compound_combat_commands": 0,
            "interaction_commands": 0,
            "one_shot_commands": 0,
            "ignored_plans": 0,
            "suppressed_duplicate_logs": 0,
            "quest_interact_suppressed": 0,
            "ui_clicks_suppressed": 0,
            "quest_dialogue_followups": 0,
            "marker_melee_fallbacks": 0,
            "visual_track_steers": 0,
            "visual_track_attacks": 0,
            "micro_recoveries": 0,
            "jump_recoveries": 0,
            "climb_recoveries": 0,
            "backtrack_recoveries": 0,
            "camera_recoveries": 0,
            "combat_recoveries": 0,
            "combat_reacquire_searches": 0,
            "combat_reposition_recoveries": 0,
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
        repeat = name in {
            "STEER_TARGET", "ATTACK_LIGHT",
            "ATTACK_ADVANCE", "COMBAT_BUNDLE"}
        log_due = (
            sig != self._last_log_sig
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

    def _repeat_allowed(
            self, pid: int, skill: str, now_ns: int,
            *, interval_s: float, max_attempts: int) -> tuple[bool, int]:
        """Bounded retry for a persistent AI-selected physical skill."""
        key = (int(pid), str(skill).upper())
        count = int(self._repeat_counts.get(key, 0))
        if count >= int(max_attempts):
            return False, count
        last = int(self._repeat_last_ns.get(key, 0))
        if last and now_ns - last < int(float(interval_s) * 1e9):
            return False, count
        count += 1
        self._repeat_counts[key] = count
        self._repeat_last_ns[key] = int(now_ns)
        # Keep bookkeeping bounded as plans roll forward.
        if len(self._repeat_counts) > 64:
            keep = {k: v for k, v in self._repeat_counts.items()
                    if k[0] >= pid - 8}
            self._repeat_counts = keep
            self._repeat_last_ns = {
                k: v for k, v in self._repeat_last_ns.items()
                if k in keep}
        return True, count

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

        # Quest status is sticky, but a red color cue alone does NOT prove a
        # quest is active. In the 2026-10-03 run a false red/objective cue
        # latched "active" before the cloud had confirmed the HUD. Local cues
        # may confirm acceptance only during the short post-interact window;
        # otherwise high-confidence Gemini perception below owns the latch.
        if self._quest_active_latched:
            self._quest_status = "active"
        elif ((red or actor) and now_ns < self._await_quest_until_ns):
            self._quest_active_latched = True
            self._quest_completed_until_ns = 0
            self._quest_status = "active"
            self._await_quest_until_ns = 0
        elif now_ns < self._quest_completed_until_ns:
            self._quest_status = "completed"
        elif now_ns < self._await_quest_until_ns:
            self._quest_status = "pending_accept"
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
            self._recovery_stage = 0
            self._recovery_target = target_key
        elif proximity is not None:
            # "Not getting closer" is a navigation failure signal, but it is
            # NOT a combat failure signal. Orbiting/guarding around a nearby
            # enemy intentionally keeps distance roughly constant. The old
            # monitor therefore fired generic JUMP/CLIMB/DASH recovery in the
            # middle of healthy fights.
            in_close_combat = bool(
                ttype == "quest_enemy_actor" and proximity >= 0.34)
            if in_close_combat:
                self._best_proximity = proximity
                self._last_progress_ns = now_ns
                self._stuck = False
                self._circling = False
                self._recovery_stage = 0
            elif (self._best_proximity is None
                    or proximity > self._best_proximity + 0.035):
                self._best_proximity = proximity
                self._last_progress_ns = now_ns
                self._stuck = False
                self._circling = False
                self._recovery_stage = 0
            elif (self._last_progress_ns
                  and now_ns - self._last_progress_ns > int(3.2e9)
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
            "quest_active": bool(self._quest_active_latched),
            "quest_active_latched": bool(self._quest_active_latched),
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
            "quest_enemy_actor_recent": bool(
                self._last_actor_ns
                and now_ns - self._last_actor_ns < int(2.2e9)),
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
            "melee_ready_visible": bool(
                ((plan or {}).get("perception") or {})
                .get("melee_ready_visible")),
            "quest_hud_ai_visible": bool(
                ((plan or {}).get("perception") or {})
                .get("quest_hud_visible")),
            "quest_progress_text": str(
                ((plan or {}).get("perception") or {})
                .get("quest_progress_text") or "")[:96],
            "equipped_style_visible": str(
                ((plan or {}).get("perception") or {})
                .get("equipped_style_visible") or "unknown")[:96],
            "action_epoch": int(self._action_epoch),
            "ui_epoch": int(self._ui_epoch),
            "ai_plan_id": (plan or {}).get("plan_id"),
            "ai_skill": (plan or {}).get("skill"),
            "ai_target": (plan or {}).get("target"),
            "ai_visual_target": (plan or {}).get("visual_target"),
            "ai_visual_track": (
                (self.ai_track.read().payload
                 if self.ai_track.read() is not None else None)),
            "decision_owner": "cloud_ai",
            "fruit_fly_control": False,
            "AI_ONLY": True,
        }, ts_ns=now_ns)

    def _steer(self, now_ns: int, *, direction: float, pid: int,
               confidence: float, reason: str, target_type: str,
               hold_s: float = 1.05) -> None:
        if now_ns - self._last_emit_ns < int(0.62e9):
            return
        self._emit(
            "STEER_TARGET", now_ns, reason=reason, ttl_s=1.6,
            direction=float(direction), hold_s=float(hold_s),
            camera_align=True,
            coach_plan_id=pid, coach_confidence=confidence,
            target_type=target_type)
        self.stats["navigation_commands"] += 1

    def _recover_navigation(
            self, now_ns: int, *, pid: int, confidence: float,
            reason: str, direction: float | None,
            target_type: str) -> bool:
        """Fast motor recovery under an already AI-selected navigation goal.

        Gemini takes ~3 s on the target laptop. Waiting for another cloud
        round-trip after detecting no progress makes walls feel fatal. These
        are bounded locomotor primitives only; they do not choose a new goal.
        """
        if self._recovery_target != target_type:
            self._recovery_target = target_type
            self._recovery_stage = 0
        if (self._last_recovery_ns
                and now_ns - self._last_recovery_ns < int(1.00e9)):
            return False

        stage = self._recovery_stage % 4
        if stage == 0:
            name, extra = "JUMP", {}
            self.stats["jump_recoveries"] += 1
        elif stage == 1:
            name, extra = "CLIMB", {"hold_s": 1.10}
            self.stats["climb_recoveries"] += 1
        elif stage == 2:
            name, extra = "DASH_BACK", {}
            self.stats["backtrack_recoveries"] += 1
        else:
            # Search toward the last-known target bearing, but keep cursor
            # confinement/camera drag at the Windows backend boundary.
            dx = 64 if (direction is None or direction >= 0) else -64
            name, extra = "SEARCH_CAMERA", {"dx": dx}
            self.stats["camera_recoveries"] += 1

        self._emit(
            name, now_ns,
            reason=reason + f":micro_recovery_{stage + 1}",
            coach_plan_id=pid, coach_confidence=confidence,
            target_type=target_type, **extra)
        self._last_recovery_ns = now_ns
        self._recovery_stage += 1
        self.stats["micro_recoveries"] += 1
        return True

    def _recover_combat(
            self, now_ns: int, *, pid: int, confidence: float,
            reason: str, direction: float | None,
            target_type: str) -> bool:
        """Reacquire/reposition under an already AI-selected FIGHT goal.

        Combat recovery intentionally excludes generic CLIMB/JUMP. Those are
        navigation skills and caused the agent to look confused while already
        in melee. Explicit cloud CLIMB/JUMP plans still work; this local bridge
        only tries to regain target lock or create a small amount of space.
        """
        if self._recovery_target != "combat:" + str(target_type):
            self._recovery_target = "combat:" + str(target_type)
            self._recovery_stage = 0
        if (self._last_recovery_ns
                and now_ns - self._last_recovery_ns < int(0.90e9)):
            return False

        stage = self._recovery_stage % 3
        if stage == 0:
            dx = 58 if (direction is None or direction >= 0) else -58
            name, extra = "SEARCH_CAMERA", {"dx": dx}
            self.stats["combat_reacquire_searches"] += 1
        elif stage == 1 and direction is not None:
            name, extra = "STEER_TARGET", {
                "direction": float(direction),
                "hold_s": 0.55,
                "camera_align": True,
            }
            self.stats["combat_reposition_recoveries"] += 1
        else:
            name, extra = "DASH_BACK", {}
            self.stats["combat_reposition_recoveries"] += 1

        self._emit(
            name, now_ns,
            reason=reason + f":combat_recovery_{stage + 1}",
            coach_plan_id=pid, coach_confidence=confidence,
            target_type=target_type, **extra)
        self._last_recovery_ns = now_ns
        self._recovery_stage += 1
        self.stats["combat_recoveries"] += 1
        return True

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
        ai_perception = plan.get("perception") or {}
        channels = plan.get("action_channels") or {}
        ch_locomotion = str(
            channels.get("locomotion") or "none").strip().lower()
        ch_offense = str(
            channels.get("offense") or "none").strip().lower()
        ch_defense = str(
            channels.get("defense") or "none").strip().lower()
        ch_camera = str(
            channels.get("camera") or "none").strip().lower()
        ch_equip = str(
            channels.get("equip_control_id") or "").strip()
        ai_visual = plan.get("visual_target") or {}
        ai_visual_kind = str(
            ai_visual.get("kind") or "none").strip().lower()
        ai_visual_conf = self._f(ai_visual.get("confidence"), 0.0) or 0.0
        ai_visual_x = self._f(ai_visual.get("x_norm"))
        ai_visual_melee = bool(ai_visual.get("melee_ready"))
        # Convert AI normalized screen x into the same approximate bearing
        # convention used by the local CV. This target is only trusted for the
        # currently fresh cloud plan and never persisted as identity.
        ai_visual_dir = (
            (float(ai_visual_x) - 0.5) * 2.8
            if ai_visual_x is not None else None)

        # A cloud response can be 1-3 seconds old. If the AI supplied a
        # bounding box, AIVisualTracker follows that SAME AI-selected target
        # locally at ~8 Hz. It does not choose a different target.
        track_env = self.ai_track.read()
        track = (track_env.payload or {}) if track_env is not None else {}
        try:
            track_pid = int(track.get("plan_id", -1))
            track_conf = float(track.get("confidence", 0.0))
        except (TypeError, ValueError):
            track_pid, track_conf = -1, 0.0
        track_kind = str(track.get("kind") or "none").strip().lower()
        track_dir = self._f(track.get("direction"))
        track_prox = self._f(track.get("proximity_hint"))
        track_fresh = bool(
            track_pid == pid
            and track_conf >= 0.42
            and track_kind != "none"
            and track_dir is not None)
        if (track_fresh
                and (ai_visual_kind == "none"
                     or track_kind == ai_visual_kind)):
            ai_visual_kind = track_kind
            ai_visual_dir = track_dir
            ai_visual_conf = max(ai_visual_conf, track_conf)
        ai_qstate = str(
            ai_perception.get("quest_state") or "unknown").lower()
        ai_equipped = str(
            ai_perception.get("equipped_slot_visible") or "unknown").strip()
        if (confidence >= 0.88
                and ai_equipped in set("0123456789")):
            self._last_equipped_slot = ai_equipped
        if confidence >= 0.85:
            # The visible accepted-quest HUD is stronger evidence than color
            # marker flicker. Keep the active latch while Gemini can actually
            # read that HUD, even if the NPC is temporarily behind terrain.
            if bool(ai_perception.get("quest_hud_visible")):
                self._quest_active_latched = True
                self._quest_status = "active"
                self._await_quest_until_ns = 0
            elif ai_qstate == "active":
                self._quest_active_latched = True
                self._quest_status = "active"
                self._await_quest_until_ns = 0
            elif ai_qstate == "completed" and confidence >= 0.90:
                self._quest_active_latched = False
                self._quest_status = "completed"
                self._quest_completed_until_ns = now + int(4.0e9)
                self._await_quest_until_ns = 0
            elif (ai_qstate == "pending_accept"
                  and not self._quest_active_latched):
                self._quest_status = "pending_accept"
            elif (ai_qstate == "available"
                  and not self._quest_active_latched
                  and self._quest_status != "pending_accept"):
                self._quest_status = "available"
            if bool(ai_perception.get("enemy_actor_visible")):
                self._ai_enemy_confirmed_until_ns = (
                    now + int(8.0e9))

        if skill in {"NAVIGATE_OBJECTIVE", "TRAVEL"}:
            # Select geometry according to semantic quest state instead of the
            # generic detector's last-writer-wins target. During an ACTIVE
            # quest, a yellow giver may remain visible behind us; steering to
            # it caused the long circular loops in the 2026-10-03 run.
            nav_type = None
            nav_dir = None
            nav_prox = None

            actor = notes.get("quest_enemy_actor") or {}
            actor_visible = bool(notes.get("quest_enemy_actor_visible"))
            if self._quest_status == "active":
                if actor_visible:
                    nav_type = "quest_enemy_actor"
                    nav_dir = self._f(actor.get("direction"))
                    nav_prox = self._f(actor.get("distance"))
                elif bool(notes.get("quest_enemy_marker_detected")):
                    nav_type = "quest_enemy_marker"
                    nav_dir = self._f(
                        notes.get("quest_enemy_marker_direction"))
                    nav_prox = self._f(
                        notes.get("quest_enemy_marker_proximity"))
                elif bool(notes.get("recommended_waypoint_detected")):
                    nav_type = "recommended_quest_waypoint"
                    nav_dir = self._f(
                        notes.get("recommended_waypoint_direction"))
                    nav_prox = self._f(
                        notes.get("recommended_waypoint_proximity"))
            else:
                if bool(notes.get("quest_marker_detected")):
                    nav_type = "quest_marker"
                    nav_dir = self._f(notes.get("quest_marker_direction"))
                    nav_prox = self._f(notes.get("quest_marker_proximity"))
                elif bool(notes.get("recommended_waypoint_detected")):
                    nav_type = "recommended_quest_waypoint"
                    nav_dir = self._f(
                        notes.get("recommended_waypoint_direction"))
                    nav_prox = self._f(
                        notes.get("recommended_waypoint_proximity"))

            # Generic target is only a fallback when it is semantically
            # compatible with the current quest state.
            if nav_dir is None and direction is not None and proximity is not None:
                allowed_types = (
                    {"quest_enemy_marker", "quest_enemy_actor",
                     "recommended_quest_waypoint"}
                    if self._quest_status == "active"
                    else {"quest_marker", "recommended_quest_waypoint"})
                if ttype in allowed_types:
                    nav_type, nav_dir, nav_prox = (
                        ttype, direction, proximity)

            if self._stuck or self._circling:
                self._recover_navigation(
                    now, pid=pid, confidence=confidence,
                    reason=reason, direction=nav_dir,
                    target_type=nav_type or ttype or "objective")
            elif nav_type and nav_dir is not None and nav_prox is not None:
                self._steer(
                    now, direction=nav_dir, pid=pid,
                    confidence=confidence, reason=reason,
                    target_type=nav_type)
            elif (ai_visual_kind in {
                    "quest_giver", "quest_enemy_actor",
                    "quest_objective", "waypoint"}
                  and ai_visual_conf >= 0.82
                  and ai_visual_dir is not None):
                # The VLM selected this target; local tracking realizes the
                # same goal while the next cloud call is in flight.
                if not (self._quest_status == "active"
                        and ai_visual_kind == "quest_giver"):
                    self._steer(
                        now, direction=ai_visual_dir, pid=pid,
                        confidence=min(confidence, ai_visual_conf),
                        reason=reason,
                        target_type="ai_visual_" + ai_visual_kind,
                        hold_s=0.78)
                    if track_fresh:
                        self.stats["visual_track_steers"] += 1

        elif skill in {"TAKE_QUEST", "INTERACT"}:
            # Never retake while perception says an objective is already live.
            if self._quest_status == "active":
                pass
            elif now < self._await_quest_until_ns:
                self.stats["quest_interact_suppressed"] += 1
            else:
                # If the CURRENT AI frame already sees a quest dialogue
                # confirmation, execute that click in the same macro instead
                # of waiting an entire cloud round-trip for UI_CLICK.
                ui = plan.get("ui_click") or {}
                dialogue = bool(ai_perception.get("dialogue_visible"))
                try:
                    ux = float(ui.get("x_norm", -1.0))
                    uy = float(ui.get("y_norm", -1.0))
                except (TypeError, ValueError):
                    ux = uy = -1.0
                ui_label = str(ui.get("label") or "")[:96]
                ui_low = ui_label.strip().lower()

                # A multimodal response sometimes grounds the button in
                # visual_target but omits duplicate ui_click coordinates.
                # Reuse the SAME AI-selected UI target rather than waiting for
                # a second cloud answer.
                if (dialogue
                        and (not bool(ui.get("needed"))
                             or not (0.0 <= ux <= 1.0
                                     and 0.0 <= uy <= 1.0))
                        and ai_visual_kind == "ui"
                        and ai_visual_conf >= 0.82
                        and ai_visual_x is not None):
                    ux = float(ai_visual_x)
                    try:
                        uy = float(ai_visual.get("y_norm", 0.5))
                    except (TypeError, ValueError):
                        uy = 0.5
                    ui["needed"] = True
                    if not ui_label:
                        ui_label = "AI grounded dialogue button"
                        ui_low = ui_label.lower()

                ui_safe = bool(
                    dialogue
                    and ui.get("needed")
                    and confidence >= 0.80
                    and 0.10 <= ux <= 0.90
                    and 0.30 <= uy <= 0.99
                    and not any(x in ui_low for x in (
                        "quit", "cancel", "decline", "no thanks")))
                if ui_safe:
                    sig = (ui_low, round(ux, 2), round(uy, 2))
                    if (sig != self._last_ui_sig
                            or not self._last_ui_click_ns
                            or now - self._last_ui_click_ns >= int(2.5e9)):
                        self._emit(
                            "UI_CLICK", now, reason=reason,
                            x_norm=ux, y_norm=uy, confidence=confidence,
                            ui_context=True, ui_label=ui_label,
                            purchase_intent=False,
                            coach_plan_id=pid,
                            coach_confidence=confidence)
                        self._last_ui_sig = sig
                        self._last_ui_click_ns = now
                        self._ui_epoch += 1
                        self._action_epoch += 1
                        self._await_quest_until_ns = now + int(5.0e9)
                        self._quest_status = "pending_accept"
                        self.stats["one_shot_commands"] += 1
                        self._publish_state(now, obs, plan)
                        return
                yellow = bool(notes.get("quest_marker_detected", False))
                ydir = self._f(notes.get("quest_marker_direction"))
                yprox = self._f(notes.get("quest_marker_proximity"))
                prompt_seen = bool(
                    ai_perception.get("interaction_prompt_visible"))

                # TAKE_QUEST / INTERACT is already the AI's decision. Once the
                # CURRENT frame visibly shows the interaction prompt, do not
                # waste another cloud round-trip asking the model to repeat
                # control_id=interact; execute the semantic skill immediately.
                direct_ai_interact = bool(
                    prompt_seen and confidence >= 0.82)

                geometry_ready = bool(
                    yellow and ydir is not None and yprox is not None
                    and yprox >= 0.46 and abs(ydir) <= 0.76)

                if direct_ai_interact or geometry_ready:
                    if (not self._last_interact_ns
                            or now - self._last_interact_ns >= int(3.0e9)):
                        self._emit(
                            "INTERACT_QUEST", now, reason=reason,
                            coach_plan_id=pid,
                            coach_confidence=confidence,
                            target_type=(
                                "quest_giver_ai_prompt"
                                if direct_ai_interact
                                else "quest_marker"))
                        self._last_interact_ns = now
                        self._quest_completed_until_ns = 0
                        self._await_quest_until_ns = now + int(5.0e9)
                        self._quest_status = "pending_accept"
                        self._action_epoch += 1
                        self.stats["interaction_commands"] += 1
                elif yellow and ydir is not None and yprox is not None:
                    self._steer(
                        now, direction=ydir, pid=pid,
                        confidence=confidence, reason=reason,
                        target_type="quest_marker")
                elif (ai_visual_kind == "quest_giver"
                      and ai_visual_conf >= 0.82
                      and ai_visual_dir is not None):
                    self._steer(
                        now, direction=ai_visual_dir, pid=pid,
                        confidence=min(confidence, ai_visual_conf),
                        reason=reason,
                        target_type="quest_giver_ai_visual")
                    if track_fresh:
                        self.stats["visual_track_steers"] += 1

        elif skill == "FIGHT_QUEST_TARGET":
            # The cloud AI chose the fight goal. This adapter keeps several
            # AI-selected control channels alive together between 2-4 s cloud
            # replies: locomotion + camera + attack + defense + equipment.
            fight_active = bool(
                self._quest_status == "active"
                or (ai_qstate == "active" and confidence >= 0.82))
            if not fight_active:
                self._publish_state(now, obs, plan)
                return

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

            grounded = False
            target_source = None
            target_dir = None
            target_prox = None

            if actor_visible and adir is not None:
                grounded = True
                target_source = "quest_enemy_actor"
                target_dir = adir
                target_prox = aprox
            elif (ai_visual_kind == "quest_enemy_actor"
                  and ai_visual_conf >= 0.70
                  and ai_visual_dir is not None):
                grounded = True
                target_source = "quest_enemy_actor_ai_visual"
                target_dir = ai_visual_dir
                target_prox = (
                    track_prox if track_fresh and track_prox is not None
                    else (0.70 if ai_visual_melee else 0.36))
            elif bool(notes.get("quest_enemy_marker_detected")):
                # In GPO the red quest dot remains tied to the target through
                # walls. It is valid pursuit identity even when the body is
                # occluded; it is not by itself proof of melee contact.
                target_source = "quest_enemy_marker"
                target_dir = self._f(
                    notes.get("quest_enemy_marker_direction"), direction)
                target_prox = self._f(
                    notes.get("quest_enemy_marker_proximity"), proximity)

            if target_dir is None:
                self._publish_state(now, obs, plan)
                return

            if self._stuck or self._circling:
                if grounded:
                    # Constant distance while orbiting a visible enemy is not
                    # "stuck". Keep the AI-selected combat channels alive.
                    self._stuck = False
                    self._circling = False
                    self._recovery_stage = 0
                else:
                    self._recover_combat(
                        now, pid=pid, confidence=confidence,
                        reason=reason, direction=target_dir,
                        target_type=target_source or "quest_enemy")
                    self._publish_state(now, obs, plan)
                    return

            # AI channels are required by the schema. Keep conservative
            # defaults for old cached plans during branch upgrades.
            locomotion = ch_locomotion
            if locomotion in {"", "none"}:
                locomotion = (
                    "orbit_right"
                    if grounded and target_prox is not None
                    and target_prox >= 0.48
                    else "approach")
            offense = ch_offense or "none"
            if offense == "none" and skill == "FIGHT_QUEST_TARGET":
                offense = "m1"
            defense = ch_defense or "none"
            camera = ch_camera or "track_target"
            if camera == "none":
                camera = "track_target"

            # Red-dot pursuit can attack once it is very close/centered even
            # if a wall briefly hides the body; this is the user's in-game
            # through-wall quest marker, not a generic red UI cue.
            close_marker = bool(
                target_source == "quest_enemy_marker"
                and target_prox is not None
                and target_prox >= 0.70
                and abs(float(target_dir)) <= 0.62
                and self._enemy_marker_since_ns
                and now - self._enemy_marker_since_ns >= int(0.45e9))
            grounded_melee = bool(
                grounded
                and target_prox is not None
                and float(target_prox) >= 0.38
                and abs(float(target_dir)) <= 0.95)
            attack_now = bool(
                offense in {"m1", "observed_ability"}
                and (grounded_melee or close_marker))

            equip_id = ch_equip
            if not equip_id:
                legacy = str(plan.get("control_id") or "")
                if legacy.startswith("equip_slot_"):
                    equip_id = legacy
            equip_slot = ""
            if equip_id.startswith("equip_slot_"):
                slot = equip_id.rsplit("_", 1)[-1]
                if slot in set("0123456789"):
                    # Do not repeatedly toggle an already selected tool.
                    if slot != self._last_equipped_slot:
                        equip_slot = slot
                        self._last_equipped_slot = slot
                        self._action_epoch += 1

            ability = plan.get("observed_ability") or {}
            ability_binding = ""
            ability_label = ""
            if offense == "observed_ability":
                binding = str(
                    ability.get("binding") or "").strip().upper()
                label = str(ability.get("label") or "").strip()[:96]
                if binding and label:
                    ability_binding = "key:" + binding
                    ability_label = label

            if now - self._last_combat_bundle_ns >= int(0.30e9):
                self._emit(
                    "COMBAT_BUNDLE", now, reason=reason, ttl_s=1.0,
                    direction=float(target_dir),
                    proximity=(
                        float(target_prox)
                        if target_prox is not None else 0.0),
                    locomotion=locomotion,
                    offense=offense,
                    defense=defense,
                    camera=camera,
                    attack=attack_now,
                    equip_slot=equip_slot,
                    ability_binding=ability_binding,
                    ability_label=ability_label,
                    coach_plan_id=pid,
                    coach_confidence=confidence,
                    target_type=target_source or "quest_enemy")
                self._last_combat_bundle_ns = now
                self.stats["combat_commands"] += 1
                self.stats["compound_combat_commands"] += 1

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
            ok, attempt = self._repeat_allowed(
                pid, "JUMP", now, interval_s=0.95, max_attempts=3)
            if ok:
                self._emit(
                    "JUMP", now, reason=reason,
                    coach_plan_id=pid, coach_confidence=confidence,
                    skill_attempt=attempt)
                if attempt == 1:
                    self._action_epoch += 1
                self.stats["one_shot_commands"] += 1

        elif skill == "CLIMB":
            ok, attempt = self._repeat_allowed(
                pid, "CLIMB", now, interval_s=1.35, max_attempts=3)
            if ok:
                self._emit(
                    "CLIMB", now, reason=reason, ttl_s=2.0,
                    hold_s=1.20,
                    coach_plan_id=pid, coach_confidence=confidence,
                    skill_attempt=attempt)
                if attempt == 1:
                    self._action_epoch += 1
                self.stats["one_shot_commands"] += 1

        elif skill == "GEPPO":
            ok, attempt = self._repeat_allowed(
                pid, "GEPPO", now, interval_s=0.52, max_attempts=6)
            if ok:
                self._emit(
                    "GEPPO", now, reason=reason,
                    coach_plan_id=pid, coach_confidence=confidence,
                    skill_attempt=attempt)
                self.stats["one_shot_commands"] += 1

        elif skill == "SPRINT":
            if self._one_shot(pid):
                self._emit(
                    "SPRINT", now, reason=reason,
                    coach_plan_id=pid, coach_confidence=confidence)
                self._action_epoch += 1
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
            ok, attempt = self._repeat_allowed(
                pid, skill, now, interval_s=0.90, max_attempts=3)
            if ok:
                self._emit(
                    "SEARCH_CAMERA", now, reason=reason,
                    dx=(-78 if skill == "LOOK_LEFT" else 78),
                    coach_plan_id=pid, coach_confidence=confidence,
                    skill_attempt=attempt)
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
                    if (slot in set("0123456789")
                            and slot != self._last_equipped_slot):
                        self._emit(
                            "EQUIP_SLOT", now, reason=reason, slot=slot,
                            coach_plan_id=pid,
                            coach_confidence=confidence)
                        self._last_equipped_slot = slot
                        self._action_epoch += 1
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
            ui = plan.get("ui_click") or {}
            if (not bool(ui.get("needed"))
                    and ai_visual_kind == "ui"
                    and ai_visual_conf >= 0.82
                    and ai_visual_x is not None):
                ui = dict(ui)
                ui["needed"] = True
                ui["x_norm"] = ai_visual_x
                ui["y_norm"] = ai_visual.get("y_norm", 0.5)
                ui["label"] = (
                    str(ui.get("label") or "")
                    or "AI grounded UI target")
            if bool(ui.get("needed")) and confidence >= 0.80:
                try:
                    x = float(ui.get("x_norm"))
                    y = float(ui.get("y_norm"))
                except (TypeError, ValueError):
                    x = y = -1.0
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    label = str(ui.get("label") or "")[:96]
                    low = label.strip().lower()
                    dialogue = bool(
                        ai_perception.get("dialogue_visible"))
                    ambiguous_negative = (
                        "quit" in low or "cancel" in low
                        or "decline" in low or "no thanks" in low)
                    quest_dialogue_safe = (
                        not dialogue
                        or (0.14 <= x <= 0.86 and 0.42 <= y <= 0.99))
                    positive_quest = (
                        skill == "UI_CLICK"
                        and dialogue
                        and any(word in low for word in (
                            "accept", "alright", "yes",
                            "continue", "...", "okay", "ok")))

                    # Normal UI actions are one-shot. For a clearly grounded
                    # positive quest-dialogue button, allow at most ONE local
                    # follow-up click while the SAME AI-selected UI target is
                    # still being tracked. This removes a whole cloud
                    # round-trip between consecutive "..." dialogue pages but
                    # cannot become an unbounded click loop.
                    if positive_quest:
                        ok, attempt = self._repeat_allowed(
                            pid, "QUEST_DIALOGUE_CLICK", now,
                            interval_s=0.95, max_attempts=2)
                        if attempt > 1:
                            ok = bool(
                                ok and track_fresh
                                and track_kind == "ui"
                                and track_conf >= 0.55)
                            if ok:
                                tx = self._f(track.get("x_norm"))
                                ty = self._f(track.get("y_norm"))
                                if tx is not None and ty is not None:
                                    x, y = float(tx), float(ty)
                                    self.stats[
                                        "quest_dialogue_followups"] += 1
                    else:
                        ok = self._one_shot(pid)

                    sig = (low, round(x, 2), round(y, 2))
                    if ambiguous_negative or not quest_dialogue_safe:
                        self.stats["ui_clicks_suppressed"] += 1
                    elif (sig == self._last_ui_sig
                            and self._last_ui_click_ns
                            and now - self._last_ui_click_ns < int(0.75e9)):
                        self.stats["ui_clicks_suppressed"] += 1
                    elif ok:
                        self._emit(
                            "UI_CLICK", now, reason=reason,
                            x_norm=x, y_norm=y, confidence=confidence,
                            ui_context=True,
                            ui_label=label,
                            purchase_intent=(skill == "BUY_ITEM"),
                            coach_plan_id=pid,
                            coach_confidence=confidence)
                        self._last_ui_sig = sig
                        self._last_ui_click_ns = now
                        self._ui_epoch += 1
                        self._action_epoch += 1
                        if positive_quest:
                            self._await_quest_until_ns = (
                                now + int(5.0e9))
                            if self._quest_status != "active":
                                self._quest_status = "pending_accept"
                        self.stats["one_shot_commands"] += 1

        # WAIT and REOBSERVE intentionally emit no physical command.
        self._publish_state(now, obs, plan)
