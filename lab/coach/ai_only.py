"""AI-only Ollama Cloud controller.

This branch deliberately removes the fruit-fly brain from the control path.
The cloud model is the sole gameplay decision-maker. Local CV, the semantic
adapter, and the input backend are execution/safety machinery only.
"""

from __future__ import annotations

import json

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
        # AI-only mode can react faster because Brian2 is not competing for CPU.
        self.min_call_interval_s = 3.0
        self.unchanged_refresh_s = 9.0
        self.provider = "ollama_cloud_ai_only"
        self.stats.update({
            "provider": self.provider,
            "decision_owner": "cloud_ai",
            "fruit_fly_control": False,
            "ai_only": True,
        })

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
            "previous_plan": _compact(previous_plan, 1200),
            "persistent_character_profile": _compact(self.profile, 3200),
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
- Choose exactly ONE skill per response.
- Never output raw keys. EXEC_CONTROL must use a control_id from
  verified_controls.
- NAVIGATE_OBJECTIVE means "move toward the currently visible quest/objective
  geometry"; the local servo will only realize YOUR navigation decision with
  W/A/D/S. It does not decide goals.
- TAKE_QUEST means approach the visible yellow quest giver and interact when
  close. FIGHT_QUEST_TARGET means approach and repeatedly M1 only while a
  confirmed quest-enemy marker is visible.
- Use EQUIP_SLOT before fighting if the needed fighting tool is not equipped.
- USE_OBSERVED_ABILITY requires a move label AND binding visible on the current
  HUD. Never guess fruit/style/sword ability keys.
- BLOCK/EVADE are explicit decisions; no hidden combat supervisor will decide
  them for you.
- Never attack ordinary players. Only fight a quest-marked NPC or an immediate
  clearly hostile NPC threat.
- Never confirm Robux, premium/gamepass, account/security, external-link, or
  trade confirmations.
- UI_CLICK/BUY_ITEM only for clearly visible ordinary in-game Peli/menu/shop
  buttons at confidence >= 0.85.
- Use the CURRENT screenshot/HUD and structured state as truth when they
  conflict with static knowledge.
- If a route is blocked choose JUMP, CLIMB, GO_AROUND, BACKTRACK, GEPPO or a
  verified movement control yourself.
- If uncertain choose REOBSERVE or WAIT.
- When useful, request a short GPO wiki query; after retrieval, make the final
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
