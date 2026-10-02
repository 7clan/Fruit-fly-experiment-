"""AI-only Ollama Cloud controller.

This branch deliberately removes the fruit-fly brain from the control path.
The cloud model is the sole gameplay decision-maker. Local CV, the semantic
adapter, and the input backend are execution/safety machinery only.
"""

from __future__ import annotations

import base64
import json
import time

from .ollama_cloud import OllamaCloudCoachWorker
from .semantic_coach import (
    ALLOWED_SKILLS, SemanticCoachWorker, _compact,
)


class AIOnlyOllamaCoachWorker(OllamaCloudCoachWorker):
    """Cloud gameplay controller with the full GPO playbook and live state."""

    name = "ai_only_ollama_coach"

    def __init__(self, bus, target_hz: float = 1.5, model: str | None = None,
                 base_url: str | None = None, timeout_s: float = 30.0,
                 api_key: str | None = None):
        super().__init__(
            bus, target_hz=target_hz, model=model, base_url=base_url,
            timeout_s=timeout_s, api_key=api_key)
        # Keep cloud decisions event-driven. The fast local actuator servo
        # continues an already-selected goal; asking the model every few
        # frames only produced duplicate plans and extra cost.
        # The cloud call itself averaged ~2.8 s on the 2026-10-02 run.
        # Keep decisions event-driven and let the local skill executor react
        # at 8-20 Hz. 2 s does not cause a 2 s cadence because unchanged
        # scenes are still held for the longer refresh interval.
        self.min_call_interval_s = 0.45
        self.unchanged_refresh_s = 20.0
        self.provider = "ollama_cloud_ai_only"
        self.suppress_duplicate_plan_logs = True
        # One frame + compact JSON is enough for this controller. Sending the
        # previous frame doubled vision work while local CV already measures
        # progress/stuck state.
        self.max_output_tokens = 320
        self.stats.update({
            "provider": self.provider,
            "decision_owner": "cloud_ai",
            "fruit_fly_control": False,
            "ai_only": True,
        })

    def _call_gemini(self, prompt: str, image_b64: str | None,
                     previous_image_b64: str | None = None) -> dict:
        # Latency path: one CURRENT frame only. Temporal progress is supplied
        # by quest.state (stuck/circling/action epochs), so a second image is
        # redundant and expensive.
        return super()._call_gemini(
            prompt, image_b64, previous_image_b64=None)

    def _signature(self, obs: dict, quest: dict, brain: dict) -> tuple:
        """Event signature for cloud replanning.

        Screen-space direction/proximity changes are handled by the local
        servo and MUST NOT wake the cloud model. Cloud replanning is reserved
        for semantic transitions: quest state, UI actions, enemy visibility,
        recovery state, or meaningful health change.
        """
        player = obs.get("player") or {}
        notes = obs.get("notes") or {}
        try:
            health_bucket = int(max(
                0.0, min(1.0, float(player.get("health")))) * 5)
        except (TypeError, ValueError):
            health_bucket = -1
        return (
            str(quest.get("quest_status") or quest.get("phase") or "unknown"),
            bool(quest.get("quest_enemy_actor_visible")),
            bool(quest.get("quest_enemy_objective_visible")),
            bool(quest.get("quest_giver_visible")),
            bool(quest.get("recommended_waypoint_visible")),
            bool(quest.get("awaiting_quest_confirmation")),
            bool(quest.get("stuck")),
            bool(quest.get("circling")),
            int(quest.get("action_epoch") or 0),
            int(quest.get("ui_epoch") or 0),
            bool(notes.get("quest_marker_detected")),
            bool(notes.get("quest_enemy_marker_detected")),
            health_bucket,
        )

    def _jpeg_b64(self, frame) -> tuple[str | None, float]:
        """Adaptive cloud frame.

        Navigation/combat need fast scene understanding more than tiny text, so
        use a smaller frame there. Quest dialogue/shop/hotbar interaction needs
        fine UI text, so temporarily send the 960 px frame. This directly cuts
        cloud vision latency without making button coordinates less accurate
        when accuracy matters.
        """
        if frame is None:
            return None, 0.0
        t0 = time.perf_counter()
        try:
            import cv2
            quest = self._state(self.quest_state)
            prev = self._state(self.plan_state)
            prev_skill = str(prev.get("skill") or "").upper()
            detail_mode = bool(
                quest.get("awaiting_quest_confirmation")
                or str(quest.get("quest_status") or "") == "pending_accept"
                or prev_skill in {
                    "TAKE_QUEST", "INTERACT", "UI_CLICK",
                    "BUY_ITEM", "EQUIP_SLOT", "USE_OBSERVED_ABILITY",
                })
            max_w = 960 if detail_mode else 736
            quality = 74 if detail_mode else 64
            img = frame
            h, w = img.shape[:2]
            if w > max_w:
                scale = max_w / float(w)
                img = cv2.resize(
                    img, (max_w, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                return None, 0.0
            data = base64.b64encode(enc.tobytes()).decode("ascii")
            return data, (time.perf_counter() - t0) * 1000.0
        except Exception:
            return None, (time.perf_counter() - t0) * 1000.0

    def _prompt(self, obs: dict, quest: dict, brain: dict, catalog: dict,
                action_meta: dict, previous_plan: dict,
                retrieved_context: str = "") -> str:
        controls = [{
            "control_id": c.get("control_id"),
            "name": c.get("name"),
            "category": c.get("category"),
            "pattern": c.get("pattern"),
            "context": c.get("context"),
            "notes": c.get("notes"),
        } for c in (catalog.get("controls") or []) if c.get("control_id")]

        target = obs.get("target") or {}
        player = obs.get("player") or {}
        notes = obs.get("notes") or {}
        # Avoid the misleading legacy field name "distance": in this CV
        # contract the value is actually PROXIMITY (0=far, 1=close).
        world_for_ai = {
            "player": {
                "health_fraction": player.get("health"),
                "stamina_fraction": player.get("stamina"),
            },
            "target": {
                "type": target.get("type"),
                "direction_radians": target.get("direction"),
                "proximity_0_far_1_close": target.get("distance"),
                "confidence": target.get("confidence"),
            },
            "cues": {
                "yellow_quest_giver": bool(notes.get("quest_marker_detected")),
                "green_recommended_waypoint": bool(
                    notes.get("recommended_waypoint_detected")),
                "red_quest_objective": bool(
                    notes.get("quest_enemy_marker_detected")),
                "quest_enemy_actor_visible": bool(
                    notes.get("quest_enemy_actor_visible")),
                "quest_enemy_actor": (
                    {
                        "track_id": (
                            (notes.get("quest_enemy_actor") or {})
                            .get("track_id")),
                        "direction_radians": (
                            (notes.get("quest_enemy_actor") or {})
                            .get("direction")),
                        "proximity_0_far_1_close": (
                            (notes.get("quest_enemy_actor") or {})
                            .get("distance")),
                        "confidence": (
                            (notes.get("quest_enemy_actor") or {})
                            .get("confidence")),
                    }
                    if notes.get("quest_enemy_actor") else {}
                ),
            },
        }
        situation = {
            "world": _compact(world_for_ai, 2600),
            "quest": _compact(quest, 2200),
            "agent": {
                "enabled": bool(action_meta.get("autonomy")),
                "mode": "AI_ONLY",
                "loadout": action_meta.get("gpo_loadout"),
            },
            "verified_controls": controls,
            "previous_plan": _compact(previous_plan, 1300),
            "persistent_character_profile": _compact(self.profile, 2400),
            "state_contract": {
                "proximity": "0 means far; 1 means close/melee",
                "quest_status": (
                    "sticky perception state; active beats visible quest giver"),
                "quest_enemy_marker": (
                    "objective/location cue; NOT proof an enemy body is in melee"),
                "quest_enemy_actor_visible": (
                    "tracked hostile NPC body associated with active quest"),
                "stuck_or_circling": (
                    "closed-loop progress monitor; recover before repeating navigation"),
                "last_equipped_slot": (
                    "last hotbar slot the agent physically selected"),
                "action_epoch": (
                    "increments after one-shot physical actions so you can "
                    "verify the result on the next frame"),
            },
        }
        situation_text = json.dumps(situation, default=str)
        # Keep the full playbook on disk, but send only relevant sections on
        # each live frame. This cuts thousands of repeated prompt tokens and
        # lowers cloud latency without deleting any available knowledge.
        playbook = self._relevant_playbook(situation_text)
        if not playbook:
            playbook = self.knowledge[:12000]
        if retrieved_context:
            playbook += "\n\nCURRENT GPO WIKI CONTEXT:\n" + retrieved_context[:8000]
        skills = ", ".join(sorted(ALLOWED_SKILLS))

        return f"""You are the SOLE GAMEPLAY CONTROLLER for an autonomous
Grand Piece Online research run. There is NO fruit-fly controller in this
branch. Every gameplay decision comes from you. Local computer vision only
measures the screen, the adapter converts your chosen skill into verified
controls, and F12 remains a human emergency stop.

GOAL:
Play normally and make useful progression. Take appropriate quests, navigate
to objectives, fight quest NPCs, equip the needed hotbar item/style, defend,
use observed abilities, buy ordinary in-game progression items when sensible,
travel by ship when needed, and update the persistent character profile.

CONTROL DISCIPLINE:
- Choose exactly ONE skill per response. Treat that skill as a PERSISTENT
  behavior/macro, not a one-frame keypress. Do not restate the same objective
  because its coordinates moved. The local servo keeps executing the selected
  behavior against fresh CV while this cloud call is sleeping. Replan only
  after a semantic state change, failure/stuck signal, UI page change, or goal
  completion.
- Never output raw keys. EXEC_CONTROL must use a control_id from
  verified_controls.
- Read quest state in this priority order:
    1) quest_status=active -> DO NOT take another quest. Follow/fight its target.
    2) quest_status=pending_accept -> wait for confirmation/dialogue change.
    3) quest_status=completed -> the prior quest ended; locate/choose the next
       appropriate quest instead of continuing to hunt a vanished target.
    4) quest_status=available -> approach the yellow quest giver and interact.
    5) travel_to_quest_giver -> follow the green recommended waypoint.
- A red quest_enemy_marker is an OBJECTIVE LOCATION/DIRECTION marker, not an
  enemy body. Do not claim an enemy is in melee from that marker alone.
  FIGHT_QUEST_TARGET is appropriate when quest_enemy_actor_visible=true; if the
  actor is not visible yet, NAVIGATE_OBJECTIVE toward the red objective.
- SAFEZONE / PROTECTED text is PvP protection in this experiment. It is NOT a
  reason to avoid or postpone fighting quest NPCs. Never invent a need to leave
  the safe zone before PvE.
- TAKE_QUEST: move to the yellow QUEST/! giver. If the CURRENT screenshot
  visibly shows the T/Interact prompt, set control_id="interact"; the actuator
  will press T immediately even if geometric proximity is noisy. Otherwise it
  approaches until ready, presses T once, then waits. If a quest dialogue/Accept/Yes button is visible, choose
  UI_CLICK with its normalized center. Do not repeatedly press T while
  awaiting_quest_confirmation=true.
- FIGHT_QUEST_TARGET: the actuator closes distance on the tracked quest NPC and
  emits M1 clicks once melee geometry is reached. Do not attack ordinary players.
- Whenever you can visually identify the object/person you are acting on, fill
  visual_target with its CURRENT normalized screen center, confidence, AND a
  tight normalized bounding box [x1,y1,x2,y2]. For a quest NPC enemy use
  kind="quest_enemy_actor". For a giver use kind="quest_giver". Set
  melee_ready=true only when the actual NPC body is visibly close enough to
  hit now. The local visual tracker follows YOUR selected box between cloud
  replies; it does not choose a target by itself.
- EQUIPMENT FOR COMBAT: inspect the CURRENT hotbar/hand/ability HUD. Report
  the visibly selected hotbar slot in perception.equipped_slot_visible when
  readable. Set perception.melee_ready_visible=true only when the current
  hand/loadout is visibly ready to M1/use melee moves.
  * If you only need to equip, choose EQUIP_SLOT with verified equip_slot_N.
  * If the quest enemy is already present and you know the needed slot, you
    may choose FIGHT_QUEST_TARGET AND set control_id="equip_slot_N". The local
    macro will equip that exact AI-selected slot once, then continue YOUR
    fight plan immediately without waiting another cloud round-trip.
  state.last_equipped_slot records the last slot physically selected.
- USE_OBSERVED_ABILITY requires BOTH a move label and binding visible on the
  CURRENT HUD. Never guess fruit/style/sword ability keys.
- Verified movement mechanics: SPACE jumps; CTRL climbs while contacting a
  wall/object (the actuator combines forward+CTRL for CLIMB); double-W sprints;
  Q with a direction rolls/dashes; repeated airborne SPACE is GEPPO if unlocked.
- If stuck=true or circling=true, do NOT repeat NAVIGATE_OBJECTIVE unchanged.
  Choose a recovery: JUMP, CLIMB, LOOK_LEFT/RIGHT, GO_AROUND, BACKTRACK, SPRINT,
  or GEPPO when its prerequisite is known.
- BLOCK/EVADE are your decisions; no hidden combat policy chooses them.
- LOOK_LEFT/LOOK_RIGHT are explicit camera actions for searching/recentering.
- UI_CLICK/BUY_ITEM may click clearly visible ordinary in-game quest/menu/shop
  buttons at confidence >= 0.80. Never confirm Robux, premium/gamepass,
  account/security, external-link, or trade UI.
- Use CURRENT screenshot/HUD + structured state as truth over static knowledge.
- If the screenshot does not support a claim, do not invent it. Choose
  REOBSERVE/WAIT instead.
- When useful, request a short GPO wiki query; after retrieval make the final
  decision instead of requesting another lookup.

GPO PLAYBOOK / STRATEGY KNOWLEDGE:
{playbook}

CURRENT STATE:
{situation_text}

Return ONLY JSON with exactly:
{{
  "scene": "short",
  "objective": "short",
  "target": "specific target or none",
  "skill": "ONE OF: {skills}",
  "control_id": "verified control id or empty",
  "observed_ability": {{"binding":"","label":""}},
  "visual_target": {{
    "kind": "none|quest_giver|quest_enemy_actor|quest_objective|waypoint|ui",
    "x_norm": 0.5,
    "y_norm": 0.5,
    "bbox_norm": [0.0, 0.0, 0.0, 0.0],
    "confidence": 0.0,
    "melee_ready": false
  }},
  "ui_click": {{"needed":false,"x_norm":0.0,"y_norm":0.0,"label":""}},
  "confidence": 0.0,
  "explanation": "one short sentence",
  "next_after_success": "short",
  "perception": {{
    "quest_state": "active|available|pending_accept|completed|unknown",
    "dialogue_visible": false,
    "interaction_prompt_visible": false,
    "enemy_actor_visible": false,
    "player_dead": false,
    "safezone_visible": false,
    "equipped_slot_visible": "0-9 or unknown",
    "melee_ready_visible": false
  }},
  "knowledge_query": "short GPO wiki query or empty",
  "memory_updates": []
}}"""


    def _validate_plan(self, raw: dict, catalog: dict) -> dict:
        plan = super()._validate_plan(raw, catalog)
        p = raw.get("perception") or {}
        allowed_quest = {
            "active", "available", "pending_accept", "completed", "unknown"
        }
        qstate = str(p.get("quest_state") or "unknown").strip().lower()
        if qstate not in allowed_quest:
            qstate = "unknown"
        plan["perception"] = {
            "quest_state": qstate,
            "dialogue_visible": bool(p.get("dialogue_visible")),
            "interaction_prompt_visible": bool(
                p.get("interaction_prompt_visible")),
            "enemy_actor_visible": bool(p.get("enemy_actor_visible")),
            "player_dead": bool(p.get("player_dead")),
            "safezone_visible": bool(p.get("safezone_visible")),
            "equipped_slot_visible": (
                str(p.get("equipped_slot_visible") or "unknown").strip()),
            "melee_ready_visible": bool(p.get("melee_ready_visible")),
        }

        vt = raw.get("visual_target") or {}
        allowed_kinds = {
            "none", "quest_giver", "quest_enemy_actor",
            "quest_objective", "waypoint", "ui",
        }
        kind = str(vt.get("kind") or "none").strip().lower()
        if kind not in allowed_kinds:
            kind = "none"
        try:
            vx = max(0.0, min(1.0, float(vt.get("x_norm", 0.5))))
            vy = max(0.0, min(1.0, float(vt.get("y_norm", 0.5))))
            vc = max(0.0, min(1.0, float(vt.get("confidence", 0.0))))
        except (TypeError, ValueError):
            vx, vy, vc = 0.5, 0.5, 0.0

        bbox = vt.get("bbox_norm")
        clean_bbox = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = [
                    max(0.0, min(1.0, float(v))) for v in bbox
                ]
                if x2 - x1 >= 0.025 and y2 - y1 >= 0.035:
                    clean_bbox = [
                        round(x1, 4), round(y1, 4),
                        round(x2, 4), round(y2, 4),
                    ]
            except (TypeError, ValueError):
                clean_bbox = None
        if vc < 0.65:
            kind = "none"
        plan["visual_target"] = {
            "kind": kind,
            "x_norm": round(vx, 4),
            "y_norm": round(vy, 4),
            "bbox_norm": clean_bbox,
            "confidence": round(vc, 4),
            "melee_ready": bool(vt.get("melee_ready")) and vc >= 0.82,
        }
        return plan


class AIOnlyGeminiCoachWorker(AIOnlyOllamaCoachWorker):
    """AI-only controller using Gemini Flash-Lite multimodal transport.

    Reuses the exact AI-only prompt/validation/event cadence above, but sends
    frames to Google's low-latency multimodal endpoint instead of Ollama
    Cloud. This is the preferred path on the target laptop because inference
    stays off-device and the model is substantially smaller/faster than the
    31B Ollama fallback that caused multi-second control lag.
    """

    name = "ai_only_gemini_coach"

    def __init__(self, bus, target_hz: float = 1.5,
                 model: str | None = None,
                 base_url: str | None = None,
                 timeout_s: float = 16.0,
                 api_key: str | None = None):
        # Call the Gemini base initializer directly so Ollama-specific auth,
        # model defaults, and transport are never activated.
        SemanticCoachWorker.__init__(
            self, bus,
            target_hz=target_hz,
            model=model,
            min_call_interval_s=0.45,
            unchanged_refresh_s=20.0,
            timeout_s=timeout_s,
            api_key=api_key,
        )
        # AIOnlyOllamaCoachWorker._prompt() uses the compact relevant-section
        # selector provided by OllamaCloudCoachWorker; initialize the same
        # cached section index without using its network transport.
        self._sections = self._split_sections(self.knowledge)
        self.min_call_interval_s = 0.45
        self.unchanged_refresh_s = 20.0
        self.provider = "gemini_ai_only"
        self.suppress_duplicate_plan_logs = True
        self.max_output_tokens = 320
        self.stats.update({
            "provider": self.provider,
            "model": self.model,
            "decision_owner": "cloud_ai",
            "fruit_fly_control": False,
            "ai_only": True,
            "supports_vision": True,
            "cloud_ai_used": True,
        })

    def _call_gemini(self, prompt: str, image_b64: str | None,
                     previous_image_b64: str | None = None) -> dict:
        # One CURRENT frame only; local CV and quest.state provide temporal
        # feedback between cloud calls.
        return SemanticCoachWorker._call_gemini(
            self, prompt, image_b64, previous_image_b64=None)
