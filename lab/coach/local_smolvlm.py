"""Local SmolVLM2-256M semantic coach.

Runs against a local llama.cpp OpenAI-compatible server. It deliberately uses
one small current screenshot per call, compact state, and offline GPO notes so
the i7-5500U / 8 GB target machine is not asked to run a large second neural
workload continuously.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request

from .semantic_coach import ALLOWED_SKILLS, SemanticCoachWorker, _compact
from .gpo_skills import render_skill_cards, procedural_skill_plan


DEFAULT_LOCAL_MODEL = "smolvlm2-256m"
DEFAULT_LOCAL_URL = "http://127.0.0.1:18080/v1"


class LocalSmolVLMCoachWorker(SemanticCoachWorker):
    """Low-duty-cycle local visual coach using llama.cpp."""

    name = "local_semantic_coach"

    def __init__(
        self,
        bus,
        target_hz: float = 0.5,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 45.0,
    ):
        super().__init__(
            bus,
            target_hz=target_hz,
            model=model or DEFAULT_LOCAL_MODEL,
            min_call_interval_s=30.0,
            unchanged_refresh_s=90.0,
            timeout_s=timeout_s,
            api_key="local-no-key",
        )
        self.base_url = str(base_url or DEFAULT_LOCAL_URL).rstrip("/")
        self.provider = "local_smolvlm2_llamacpp"
        self.allow_remote_wiki = False
        self._sections = self._split_sections(self.knowledge)
        self._last_skill_signature = None
        self._last_skill_publish_ns = 0
        self._vision_backoff_until_ns = 0
        self.stats.update({
            "provider": self.provider,
            "base_url": self.base_url,
            "remote_wiki": False,
            "local_only_inference": True,
            "min_call_interval_s": self.min_call_interval_s,
            "procedural_plans": 0,
            "visual_calls_reserved_for_ui": True,
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
        """Offline keyword retrieval from the full committed playbook."""
        lower = str(situation_text or "").lower()
        always = (
            "core play loop",
            "verified/common controls",
            "movement / obstacle reasoning",
            "pve combat policy",
            "semantic-coach output discipline",
        )
        topic_terms = {
            "quest": ("town of beginnings", "level/progression route"),
            "level": ("level/progression route", "long-term autonomous planner"),
            "island": ("level/progression route", "long-term autonomous planner"),
            "ship": ("ships / sea travel", "current equipment / style catalog"),
            "boat": ("ships / sea travel",),
            "fruit": ("devil-fruit planning",),
            "accessor": (
                "accessories / equipment purchases",
                "current equipment / style catalog",
            ),
            "weapon": (
                "weapons and ranged options",
                "current equipment / style catalog",
            ),
            "sword": (
                "weapons and ranged options",
                "current equipment / style catalog",
            ),
            "gun": ("weapons and ranged options",),
            "haki": ("haki plan",),
            "style": (
                "fighting-style planning",
                "current equipment / style catalog",
            ),
            "shop": ("buying / ui interaction policy",),
            "buy": ("buying / ui interaction policy",),
            "menu": ("buying / ui interaction policy",),
            "stat": ("stats/build planning",),
        }
        wanted = list(always)
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
        return "\n\n".join(selected)[:7200]

    def _jpeg_b64(self, frame):
        if frame is None:
            return None, 0.0
        t0 = time.perf_counter()
        try:
            import cv2
            img = frame
            h, w = img.shape[:2]
            if w > 256:
                scale = 256.0 / float(w)
                img = cv2.resize(
                    img, (256, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 48])
            if not ok:
                return None, 0.0
            return (
                base64.b64encode(enc.tobytes()).decode("ascii"),
                (time.perf_counter() - t0) * 1000.0,
            )
        except Exception:
            return None, (time.perf_counter() - t0) * 1000.0

    def _prompt(
        self, obs: dict, quest: dict, brain: dict,
        catalog: dict, action_meta: dict, previous_plan: dict,
        retrieved_context: str = "",
    ) -> str:
        controls = [
            {
                "control_id": c.get("control_id"),
                "name": c.get("name"),
                "pattern": c.get("pattern"),
            }
            for c in (catalog.get("controls") or [])
            if c.get("control_id")
        ]
        compact_state = {
            "world": _compact(obs, 1050),
            "quest": _compact(quest, 450),
            "fly_intention": (brain.get("intention") or {}).get("name"),
            "loadout": action_meta.get("gpo_loadout"),
            "profile": _compact(self.profile, 550),
            "controls": controls[:18],
            "previous_plan": _compact(previous_plan, 260),
        }
        situation_text = json.dumps(compact_state, default=str)
        # Tiny-model strategy: procedural skill cards carry the action recipe.
        # Only a very small playbook excerpt is included as background.
        knowledge = self._relevant_playbook(situation_text)[:2200]
        skill_cards = render_skill_cards(situation_text, max_cards=3)
        skills = ", ".join(sorted(ALLOWED_SKILLS))

        return f"""You are a SMALL LOCAL VISUAL COACH for Grand Piece Online.
Look at the CURRENT screenshot and compact state. Pick ONE immediate skill.
The fruit-fly brain owns ordinary left/right/approach navigation.

RULES:
- Yellow QUEST NPC/marker and no active quest: TAKE_QUEST/INTERACT.
- Green objective: NAVIGATE_OBJECTIVE; do not invent a direction.
- Red quest enemy: FIGHT_QUEST_TARGET only if clearly the quest NPC.
- Stuck at a low ledge: JUMP. Climbable wall: CLIMB. Otherwise GO_AROUND or BACKTRACK.
- Never attack ordinary players.
- Never move the camera.
- Never guess ownership, fruit, weapon, style, ship, or an ability key.
- USE_OBSERVED_ABILITY only if BOTH move label and key are readable now.
- UI_CLICK/BUY_ITEM only for a clearly visible in-game button at >=0.90 confidence.
- Never click Robux/gamepass/trade/account/external-link confirmations.
- If unsure: REOBSERVE.
- Be conservative. One action only.

PROCEDURAL SKILL CARDS:
{skill_cards}

OFFLINE GPO NOTES:
{knowledge}

STATE:
{situation_text}

Return ONLY compact JSON:
{{
 "scene":"short",
 "objective":"short",
 "target":"short",
 "skill":"one of {skills}",
 "control_id":"",
 "observed_ability":{{"binding":"","label":""}},
 "ui_click":{{"needed":false,"x_norm":0.0,"y_norm":0.0,"label":""}},
 "confidence":0.0,
 "explanation":"one short sentence",
 "next_after_success":"short",
 "knowledge_query":"",
 "memory_updates":[]
}}"""

    def _publish_skill_plan(self, raw: dict, catalog: dict,
                            now_ns: int) -> None:
        raw = dict(raw or {})
        raw.setdefault("control_id", "")
        raw.setdefault("observed_ability", {"binding": "", "label": ""})
        raw.setdefault(
            "ui_click",
            {"needed": False, "x_norm": 0.0, "y_norm": 0.0, "label": ""})
        raw.setdefault("memory_updates", [])
        raw.setdefault("knowledge_query", "")
        plan = self._validate_plan(raw, catalog)
        plan["provider"] = "procedural_skill_router"
        plan["model"] = self.model
        plan["skill_card"] = raw.get("skill_card")
        plan["local_vlm_used"] = False
        plan["ts_ns"] = int(now_ns)
        self.plan_state.write(plan, ts_ns=now_ns)
        self.events.publish(
            {"kind": "coach_plan", "plan": plan, "ts_ns": now_ns},
            ts_ns=now_ns)
        self.stats["plans"] += 1
        self.stats["last_plan"] = {
            "plan_id": plan["plan_id"],
            "scene": plan["scene"],
            "skill": plan["skill"],
            "confidence": plan["confidence"],
            "skill_card": plan.get("skill_card"),
            "local_vlm_used": False,
        }

    def step(self) -> None:
        """Skill-first local coach.

        Obvious quest states use executable procedural skills immediately.
        The 256M visual model is reserved for scenes where reading an actual
        dialog/menu can add information.  If local inference stalls or dies,
        gameplay continues through the skill/supervisor layers.
        """
        now = self.clock.now_ns()
        meta = self._state(self.meta_state)
        if not bool(meta.get("autonomy")):
            self.stats["skips"] += 1
            return

        obs = self._state(self.world_state)
        if not obs:
            self.stats["skips"] += 1
            return
        quest = self._state(self.quest_state)
        catalog = self._state(self.catalog_state)

        raw = procedural_skill_plan(obs, quest)
        if raw is not None:
            target = obs.get("target") or {}
            notes = obs.get("notes") or {}
            signature = (
                str(raw.get("skill") or ""),
                str(raw.get("scene") or ""),
                str(target.get("type") or ""),
                round(float(target.get("distance") or 0.0), 1),
                bool(notes.get("quest_marker_detected")),
            )
            due = (
                signature != self._last_skill_signature
                or now - self._last_skill_publish_ns >= int(8.0e9))
            if due:
                self._publish_skill_plan(raw, catalog, now)
                self._last_skill_signature = signature
                self._last_skill_publish_ns = now
                self.stats["procedural_plans"] = (
                    int(self.stats.get("procedural_plans", 0)) + 1)
            else:
                self.stats["skips"] += 1
            return

        ui = dict(obs.get("ui") or {})
        target = dict(obs.get("target") or {})
        needs_visual_semantics = bool(
            ui.get("dialogue") or ui.get("menu")
        )

        # Unknown outdoor scene: do not burn 45-180 seconds of old-laptop CPU
        # trying to make a 256M VLM rediscover that no objective is visible.
        # The CV/supervisor will continue reacquiring quest markers.
        if not needs_visual_semantics:
            fallback = {
                "scene": "uncertain_no_visible_objective",
                "objective": "wait for a reliable quest/objective cue",
                "target": str(target.get("type") or "none"),
                "skill": "REOBSERVE",
                "confidence": 0.94,
                "explanation": (
                    "No dialog/menu or reliable semantic target requires "
                    "expensive local vision right now."),
                "next_after_success": "resume the matching procedural skill",
                "skill_card": "obstacle_recovery",
            }
            if now - self._last_skill_publish_ns >= int(12.0e9):
                self._publish_skill_plan(fallback, catalog, now)
                self._last_skill_publish_ns = now
                self.stats["procedural_plans"] = (
                    int(self.stats.get("procedural_plans", 0)) + 1)
            else:
                self.stats["skips"] += 1
            return

        if now < self._vision_backoff_until_ns:
            self.stats["skips"] += 1
            return

        before_errors = int(self.stats.get("api_errors", 0))
        try:
            super().step()
        except Exception as exc:
            # A local VLM/server failure must never kill the game-control
            # experiment.  Fall back to the executable skill layer.
            self.stats["api_errors"] = int(
                self.stats.get("api_errors", 0)) + 1
            self.stats["last_error"] = repr(exc)
            self.events.publish({
                "kind": "coach_error",
                "ts_ns": now,
                "model": self.model,
                "error": repr(exc),
                "fallback": "procedural_skills",
            }, ts_ns=now)

        if int(self.stats.get("api_errors", 0)) > before_errors:
            # Back off hard after an expensive failure.  The rest of the
            # agent remains fully operational without the VLM.
            self._vision_backoff_until_ns = now + int(180.0e9)
            fallback = {
                "scene": "local_vision_unavailable",
                "objective": "continue using procedural quest skills",
                "target": str(target.get("type") or "none"),
                "skill": "REOBSERVE",
                "confidence": 0.98,
                "explanation": (
                    "Local vision timed out/unavailable; procedural skills "
                    "remain active."),
                "next_after_success": "retry visual semantics later",
                "skill_card": "obstacle_recovery",
            }
            self._publish_skill_plan(fallback, catalog, now)

    def _call_gemini(
            self, prompt: str, image_b64: str | None,
            previous_image_b64: str | None = None) -> dict:
        # Do not encode the previous frame locally on the 2-core target.
        # Temporal progress is already summarized by CV/state.
        content = [{"type": "text", "text": prompt}]
        if image_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + image_b64,
                },
            })
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 160,
        }
        endpoint = self.base_url + "/chat/completions"

        def send(payload):
            raw = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=raw,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": "Bearer local-no-key",
                },
                method="POST",
            )
            with urllib.request.urlopen(
                    req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))

        payload = send(body)

        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"local SmolVLM returned no choices: {str(payload)[:500]}")
        text = str(
            ((choices[0].get("message") or {}).get("content")) or ""
        ).strip()
        if not text:
            raise RuntimeError(
                f"local SmolVLM returned no text: {str(payload)[:500]}")
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)
