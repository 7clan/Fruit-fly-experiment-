"""Ollama Cloud Qwen3-VL semantic coach.

Heavy multimodal reasoning runs on Ollama Cloud, so the old laptop keeps
its CPU for Roblox, OpenCV and the canonical Brian2 fly brain.
"""

from __future__ import annotations

import json
import os
import urllib.request

from .semantic_coach import ALLOWED_SKILLS, SemanticCoachWorker, _compact

DEFAULT_OLLAMA_MODEL = "qwen3.5"
DEFAULT_OLLAMA_BASE = "https://ollama.com/api"


class OllamaCloudCoachWorker(SemanticCoachWorker):
    """Low-rate multimodal coach using Ollama Cloud Qwen3-VL."""

    name = "ollama_cloud_semantic_coach"

    def __init__(self, bus, target_hz: float = 1.0, model: str | None = None,
                 base_url: str | None = None, timeout_s: float = 30.0,
                 api_key: str | None = None):
        key = str(api_key or os.getenv("OLLAMA_API_KEY") or "").strip()
        selected = str(model or os.getenv("OLLAMA_CLOUD_MODEL")
                       or DEFAULT_OLLAMA_MODEL).strip()
        super().__init__(bus, target_hz=target_hz, model=selected,
                         min_call_interval_s=8.0,
                         unchanged_refresh_s=30.0,
                         timeout_s=timeout_s, api_key=key)
        self.base_url = str(base_url or os.getenv("OLLAMA_CLOUD_BASE")
                            or DEFAULT_OLLAMA_BASE).rstrip("/")
        self.provider = "ollama_cloud_qwen3_5"
        self.api_key = key
        self.model = selected
        self.supports_vision = True
        self.allow_remote_wiki = True
        self._sections = self._split_sections(self.knowledge)
        self.stats.update({
            "enabled": bool(self.api_key),
            "provider": self.provider,
            "base_url": self.base_url,
            "supports_vision": True,
            "cloud_ai_used": True,
            "disabled_reason": (None if self.api_key
                                else "OLLAMA_API_KEY_missing"),
        })

    @staticmethod
    def _split_sections(text: str) -> list[tuple[str, str]]:
        out = []
        heading = "intro"
        body = []
        for line in str(text or "").splitlines():
            if line.startswith("## "):
                if body:
                    out.append((heading, "\n".join(body).strip()))
                heading = line[3:].strip()
                body = [line]
            else:
                body.append(line)
        if body:
            out.append((heading, "\n".join(body).strip()))
        return out

    def _relevant_playbook(self, situation_text: str) -> str:
        lower = str(situation_text or "").lower()
        wanted = ["core play loop", "verified/common controls",
                  "movement / obstacle reasoning", "pve combat policy",
                  "semantic-coach output discipline"]
        topic_terms = {
            "quest": ("level/progression route", "town of beginnings"),
            "ship": ("ships / sea travel",),
            "fruit": ("devil-fruit planning",),
            "accessor": ("accessories / equipment purchases",),
            "weapon": ("weapons and ranged options",),
            "sword": ("weapons and ranged options",),
            "haki": ("haki plan",),
            "style": ("fighting-style planning",),
            "shop": ("buying / ui interaction policy",),
            "buy": ("buying / ui interaction policy",),
            "stat": ("stats/build planning",),
        }
        for needle, names in topic_terms.items():
            if needle in lower:
                wanted.extend(names)
        selected = []
        seen = set()
        for wanted_name in wanted:
            for heading, body in self._sections:
                h = heading.lower()
                if wanted_name in h and h not in seen:
                    seen.add(h)
                    selected.append(body)
                    break
        return "\n\n".join(selected)[:9000]

    def _prompt(self, obs: dict, quest: dict, brain: dict, catalog: dict,
                action_meta: dict, previous_plan: dict,
                retrieved_context: str = "") -> str:
        controls = [{
            "control_id": c.get("control_id"),
            "name": c.get("name"),
            "category": c.get("category"),
            "pattern": c.get("pattern"),
            "context": c.get("context"),
        } for c in (catalog.get("controls") or []) if c.get("control_id")]
        situation = {
            "world": _compact(obs, 2400),
            "quest": _compact(quest, 1000),
            "fly_brain": _compact(brain, 900),
            "agent": {"enabled": bool(action_meta.get("autonomy")),
                      "loadout": action_meta.get("gpo_loadout")},
            "verified_controls": controls,
            "previous_plan": _compact(previous_plan, 700),
            "character_profile": _compact(self.profile, 1800),
        }
        situation_text = json.dumps(situation, default=str)
        playbook = self._relevant_playbook(situation_text)
        if retrieved_context:
            playbook += "\n\nCURRENT WIKI CONTEXT:\n" + retrieved_context[:6000]
        skills = ", ".join(sorted(ALLOWED_SKILLS))
        return f"""You are the visual semantic coach for a Grand Piece Online
research agent. The CURRENT screenshot is authoritative. Choose exactly ONE
high-level skill. The fruit-fly brain/fresh CV servo owns low-level steering.

CRITICAL COMBAT RULES:
- Combat only against a quest-marked NPC or immediate hostile NPC threat.
- If combat is required but melee/fighting style is not equipped, use
  EQUIP_SLOT with a verified equip_slot_N control first.
- Once the fighting tool/style is equipped and a quest enemy is in melee
  range, use FIGHT_QUEST_TARGET. The local executor converts that into real
  left-mouse clicks (M1). It can also BLOCK, EVADE, and use visible abilities.
- Never use UI_CLICK as an attack. UI_CLICK is only for actual game UI buttons.
- Never attack ordinary players.

OTHER RULES:
- Never output raw keyboard keys or scan codes.
- EXEC_CONTROL control_id must come from verified_controls.
- USE_OBSERVED_ABILITY requires BOTH a clearly visible move label and binding.
- Yellow QUEST giver with no accepted quest -> TAKE_QUEST / INTERACT.
- Green tracker/objective -> NAVIGATE_OBJECTIVE.
- Obstacles -> JUMP, CLIMB, GO_AROUND or BACKTRACK as appropriate.
- Never rotate/drag the user camera.
- Never confirm Robux, gamepass, account/security, external-link, or trade UI.
- Visible ordinary in-game Peli purchases may use BUY_ITEM/UI_CLICK.
- If uncertain -> REOBSERVE or WAIT.

RELEVANT GPO PLAYBOOK:
{playbook}

CURRENT STRUCTURED STATE:
{situation_text}

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

    @staticmethod
    def _json_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "scene": {"type": "string"},
                "objective": {"type": "string"},
                "target": {"type": "string"},
                "skill": {"type": "string", "enum": sorted(ALLOWED_SKILLS)},
                "control_id": {"type": "string"},
                "observed_ability": {"type": "object", "properties": {
                    "binding": {"type": "string"}, "label": {"type": "string"}},
                    "required": ["binding", "label"]},
                "ui_click": {"type": "object", "properties": {
                    "needed": {"type": "boolean"},
                    "x_norm": {"type": "number"}, "y_norm": {"type": "number"},
                    "label": {"type": "string"}},
                    "required": ["needed", "x_norm", "y_norm", "label"]},
                "confidence": {"type": "number"},
                "explanation": {"type": "string"},
                "next_after_success": {"type": "string"},
                "knowledge_query": {"type": "string"},
                "memory_updates": {"type": "array"},
            },
            "required": ["scene", "objective", "target", "skill", "control_id",
                         "observed_ability", "ui_click", "confidence",
                         "explanation", "next_after_success",
                         "knowledge_query", "memory_updates"],
        }

    def _validate_plan(self, raw: dict, catalog: dict) -> dict:
        plan = super()._validate_plan(raw, catalog)
        plan["ai_used"] = True
        plan["cloud_ai_used"] = True
        plan["local_ai_used"] = False
        plan["provider"] = self.provider
        return plan

    def _call_gemini(self, prompt: str, image_b64: str | None,
                     previous_image_b64: str | None = None) -> dict:
        endpoint = self.base_url + "/chat"
        msg = {"role": "user", "content": prompt}
        if image_b64:
            msg["images"] = [image_b64]
        body = {
            "model": self.model,
            "messages": [msg],
            "stream": False,
            "format": self._json_schema(),
            "options": {"temperature": 0.0, "num_predict": 520},
        }
        def send(payload_body: dict) -> dict:
            req = urllib.request.Request(
                endpoint, data=json.dumps(payload_body).encode("utf-8"),
                headers={"Accept": "application/json",
                         "Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}",
                         "User-Agent": "DigitalFlyLab/1.0"},
                method="POST")
            with urllib.request.urlopen(
                    req, timeout=float(self.timeout_s)) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            payload = send(body)
        except urllib.error.HTTPError as exc:
            # Some hosted model revisions may reject schema/options even
            # though ordinary chat + vision works. Keep one plain fallback.
            if int(getattr(exc, "code", 0)) != 400:
                raise
            fallback = {
                "model": self.model,
                "messages": [msg],
                "stream": False,
            }
            payload = send(fallback)

        text = str(((payload.get("message") or {}).get("content")) or "").strip()
        if not text:
            raise RuntimeError("Ollama Cloud returned no message content: "
                               + str(payload)[:500])
        if text.startswith("```"):
            rows = text.splitlines()
            if rows and rows[0].startswith("```"): rows = rows[1:]
            if rows and rows[-1].strip().startswith("```"): rows = rows[:-1]
            text = "\n".join(rows).strip()
        return json.loads(text)