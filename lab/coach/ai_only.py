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
from .semantic_coach import ALLOWED_SKILLS, _compact


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
        self.min_call_interval_s = 4.0
        self.unchanged_refresh_s = 14.0
        self.provider = "ollama_cloud_ai_only"
        self.stats.update({
            "provider": self.provider,
            "decision_owner": "cloud_ai",
            "fruit_fly_control": False,
            "ai_only": True,
        })

    def _signature(self, obs: dict, quest: dict, brain: dict) -> tuple:
        """Coarse event signature for cloud calls.

        Screen-space target coordinates jitter every frame. Treat that as
        actuator feedback, not a reason to ask the cloud model for the same
        objective again.
        """
        target = obs.get("target") or {}
        player = obs.get("player") or {}
        notes = obs.get("notes") or {}
        try:
            prox = float(target.get("distance"))
        except (TypeError, ValueError):
            prox = -1.0
        try:
            direction = float(target.get("direction"))
        except (TypeError, ValueError):
            direction = 0.0
        direction_bucket = (
            "left" if direction < -0.45
            else "right" if direction > 0.45
            else "center")
        proximity_bucket = (
            "near" if prox >= 0.62
            else "mid" if prox >= 0.32
            else "far")
        try:
            health_bucket = int(max(
                0.0, min(1.0, float(player.get("health")))) * 4)
        except (TypeError, ValueError):
            health_bucket = -1
        return (
            str(target.get("type") or "none"),
            direction_bucket,
            proximity_bucket,
            str(quest.get("quest_status") or quest.get("phase") or "unknown"),
            bool(quest.get("quest_enemy_actor_visible")),
            bool(quest.get("awaiting_quest_confirmation")),
            bool(quest.get("stuck")),
            bool(quest.get("circling")),
            bool(notes.get("quest_marker_detected")),
            bool(notes.get("quest_enemy_marker_detected")),
            health_bucket,
        )

    def _jpeg_b64(self, frame) -> tuple[str | None, float]:
        """Higher-detail frame for the sole controller.

        The old 720px/Q58 image was optimized for a helper coach. AI-only must
        read hotbar slots, dialogue buttons, quest text and nearby NPC cues.
        One ~960px JPEG every few seconds is cheap compared with the removed
        Brian2 workload.
        """
        if frame is None:
            return None, 0.0
        t0 = time.perf_counter()
        try:
            import cv2
            img = frame
            h, w = img.shape[:2]
            if w > 960:
                scale = 960.0 / float(w)
                img = cv2.resize(
                    img, (960, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 74])
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

        situation = {
            "world": _compact(obs, 4200),
            "quest": _compact(quest, 1500),
            "agent": {
                "enabled": bool(action_meta.get("autonomy")),
                "mode": "AI_ONLY",
                "loadout": action_meta.get("gpo_loadout"),
            },
            "verified_controls": controls,
            "previous_plan": _compact(previous_plan, 1600),
            "persistent_character_profile": _compact(self.profile, 3200),
            "state_contract": {
                "quest_status": (
                    "perception-derived; active beats any yellow quest giver"),
                "quest_enemy_marker": (
                    "objective/location cue; NOT proof an enemy body is in melee"),
                "quest_enemy_actor_visible": (
                    "tracked hostile NPC body associated with active quest"),
                "stuck_or_circling": (
                    "closed-loop progress monitor; recover before repeating navigation"),
                "last_equipped_slot": (
                    "last hotbar slot the agent physically selected"),
            },
        }
        playbook = self.knowledge[:30000]
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
- Choose exactly ONE skill per response. Do not restate the same objective just
  because its screen coordinates moved slightly. If the previous plan is still
  progressing, keep that skill; the local servo persists it without another
  cloud decision.
- Never output raw keys. EXEC_CONTROL must use a control_id from
  verified_controls.
- Read quest state in this priority order:
    1) quest_status=active -> DO NOT take another quest. Follow/fight its target.
    2) quest_status=pending_accept -> wait for confirmation/dialogue change.
    3) quest_status=available -> approach the yellow quest giver and interact.
    4) travel_to_quest_giver -> follow the green recommended waypoint.
- A red quest_enemy_marker is an OBJECTIVE LOCATION/DIRECTION marker, not an
  enemy body. Do not claim an enemy is in melee from that marker alone.
  FIGHT_QUEST_TARGET is appropriate when quest_enemy_actor_visible=true; if the
  actor is not visible yet, NAVIGATE_OBJECTIVE toward the red objective.
- SAFEZONE / PROTECTED text is PvP protection in this experiment. It is NOT a
  reason to avoid or postpone fighting quest NPCs. Never invent a need to leave
  the safe zone before PvE.
- TAKE_QUEST: move to the yellow QUEST/! giver. When close the actuator presses
  T once, then waits. If a quest dialogue/Accept/Yes button is visible, choose
  UI_CLICK with its normalized center. Do not repeatedly press T while
  awaiting_quest_confirmation=true.
- FIGHT_QUEST_TARGET: the actuator closes distance on the tracked quest NPC and
  emits M1 clicks once melee geometry is reached. Do not attack ordinary players.
- EQUIP_SLOT: use a visible hotbar slot or a verified equip_slot_N control.
  state.last_equipped_slot tells you what the agent last physically selected.
  Equip before combat when the fighting tool/style is not ready.
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
{json.dumps(situation, default=str)}

Return ONLY JSON with exactly:
{{
  "scene": "short",
  "objective": "short",
  "target": "specific target or none",
  "skill": "ONE OF: {skills}",
  "control_id": "verified control id or empty",
  "observed_ability": {{"binding":"","label":""}},
  "ui_click": {{"needed":false,"x_norm":0.0,"y_norm":0.0,"label":""}},
  "confidence": 0.0,
  "explanation": "one short sentence",
  "next_after_success": "short",
  "knowledge_query": "short GPO wiki query or empty",
  "memory_updates": []
}}"""
