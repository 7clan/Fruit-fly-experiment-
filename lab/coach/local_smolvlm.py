"""Local SmolVLM2-256M semantic coach.

Runs against a local llama.cpp OpenAI-compatible server. It deliberately uses
one small current screenshot per call, compact state, and offline GPO notes so
the i7-5500U / 8 GB target machine is not asked to run a large second neural
workload continuously.
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request

from .semantic_coach import ALLOWED_SKILLS, SemanticCoachWorker, _compact
from .gpo_skills import (
    SKILL_CARDS, render_skill_cards, procedural_skill_plan,
)


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
            min_call_interval_s=90.0,
            unchanged_refresh_s=180.0,
            timeout_s=timeout_s,
            api_key="local-no-key",
        )
        self.base_url = str(base_url or DEFAULT_LOCAL_URL).rstrip("/")
        self.provider = "local_smolvlm2_llamacpp"
        self.allow_remote_wiki = False
        self._sections = self._split_sections(self.knowledge)
        self._last_skill_signature = None
        self._last_skill_publish_ns = 0
        self._last_ai_signature = None
        self._last_ai_call_ns = 0
        self._text_backoff_until_ns = 0
        self._vision_backoff_until_ns = 0
        self.stats.update({
            "provider": self.provider,
            "base_url": self.base_url,
            "remote_wiki": False,
            "local_only_inference": True,
            "min_call_interval_s": self.min_call_interval_s,
            "procedural_plans": 0,
            "text_skill_calls": 0,
            "visual_calls": 0,
            "visual_calls_reserved_for_semantic_uncertainty": True,
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

    def _publish_skill_plan(
            self, raw: dict, catalog: dict, now_ns: int, *,
            provider: str = "procedural_skill_router",
            local_vlm_used: bool = False) -> None:
        raw = dict(raw or {})
        raw.setdefault("control_id", "")
        raw.setdefault("observed_ability", {"binding": "", "label": ""})
        raw.setdefault(
            "ui_click",
            {"needed": False, "x_norm": 0.0, "y_norm": 0.0, "label": ""})
        raw.setdefault("memory_updates", [])
        raw.setdefault("knowledge_query", "")
        raw.setdefault("confidence", 0.80)
        raw.setdefault("scene", "unknown")
        raw.setdefault("objective", "")
        raw.setdefault("target", "none")
        raw.setdefault("skill", "REOBSERVE")
        raw.setdefault("explanation", "")
        raw.setdefault("next_after_success", "")
        plan = self._validate_plan(raw, catalog)
        plan["provider"] = str(provider)
        plan["model"] = self.model
        plan["skill_card"] = raw.get("skill_card")
        plan["local_vlm_used"] = bool(local_vlm_used)
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
            "local_vlm_used": bool(local_vlm_used),
        }

    @staticmethod
    def _card_actions(card_id: str) -> set[str]:
        for card in SKILL_CARDS:
            if card.skill_id == str(card_id or ""):
                return {str(x).upper() for x in card.actions}
        return {"REOBSERVE", "WAIT"}

    def _text_skill_prompt(
            self, candidate: dict, obs: dict, quest: dict,
            catalog: dict) -> str:
        """Ultra-small prompt for the 256M model on the 2-core Windows box.

        The model is not asked to reproduce a JSON object. It only selects
        one bounded skill label; the trusted procedural candidate still owns
        all arguments, target geometry and control details.
        """
        card_id = str(candidate.get("skill_card") or "")
        target = dict(obs.get("target") or {})
        notes = dict(obs.get("notes") or {})
        player = dict(obs.get("player") or {})
        allowed = sorted(self._card_actions(card_id))
        state = {
            "card": card_id,
            "scene": candidate.get("scene"),
            "candidate": candidate.get("skill"),
            "target": target.get("type"),
            "direction": target.get("direction"),
            "distance": target.get("distance"),
            "target_conf": target.get("confidence"),
            "phase": quest.get("phase"),
            "health": player.get("health"),
            "stamina": player.get("stamina"),
            "yellow": bool(notes.get("quest_marker_detected")),
            "yellow_proximity": notes.get("quest_marker_proximity"),
            "yellow_direction": notes.get("quest_marker_direction"),
        }
        return (
            "GPO skill selector. Choose ONE exact label from ALLOWED. "
            "Use the candidate unless state clearly requires another allowed "
            "skill. Never invent a label.\n"
            f"STATE={json.dumps(state, separators=(',', ':'), default=str)}\n"
            f"ALLOWED={'|'.join(allowed)}\n"
            "ANSWER="
        )

    def _call_text_action(
            self, prompt: str, allowed: set[str]) -> str:
        """Return one skill token with minimal generation cost."""
        endpoint = self.base_url + "/chat/completions"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 12,
            "stop": ["\n"],
        }
        raw = json.dumps(body).encode("utf-8")
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
                req, timeout=min(float(self.timeout_s), 25.0)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        choices = payload.get("choices") or []
        text = str(
            (((choices[0] if choices else {}).get("message") or {})
             .get("content")) or ""
        ).strip()
        if not text:
            raise RuntimeError(
                f"local skill selector returned no text: {str(payload)[:300]}")

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                text = str(parsed.get("skill") or "")
            elif isinstance(parsed, str):
                text = parsed
        except Exception:
            pass
        upper = str(text).strip().strip("'\" ").upper()
        if upper in allowed:
            return upper

        found = [
            skill for skill in sorted(allowed, key=len, reverse=True)
            if re.search(r"(?<![A-Z0-9_])" + re.escape(skill)
                         + r"(?![A-Z0-9_])", upper)
        ]
        found = list(dict.fromkeys(found))
        if len(found) == 1:
            return found[0]
        raise ValueError(
            f"local skill selector output {text!r} did not name exactly one "
            f"allowed skill {sorted(allowed)}")

    def _run_text_skill_selector(
            self, candidate: dict, obs: dict, quest: dict,
            catalog: dict, now_ns: int) -> bool:
        prompt = self._text_skill_prompt(candidate, obs, quest, catalog)
        allowed = self._card_actions(candidate.get("skill_card"))
        t0 = time.perf_counter()
        self.stats["calls"] = int(self.stats.get("calls", 0)) + 1
        self.stats["text_skill_calls"] = int(
            self.stats.get("text_skill_calls", 0)) + 1
        try:
            chosen = self._call_text_action(prompt, allowed)
            merged = dict(candidate)
            merged["skill"] = chosen
            merged["explanation"] = (
                f"Local SmolVLM selected {chosen} from "
                f"{sorted(allowed)}.")
            merged["skill_card"] = candidate.get("skill_card")
            self._publish_skill_plan(
                merged, catalog, now_ns,
                provider="local_smolvlm2_text_skill",
                local_vlm_used=True)
            self._last_ai_call_ns = now_ns
            self._last_ai_signature = (
                str(candidate.get("skill_card") or ""),
                str(candidate.get("scene") or ""),
                str((obs.get("target") or {}).get("type") or ""),
            )
            self._last_skill_publish_ns = now_ns
            self._last_skill_signature = self._last_ai_signature
            self.stats["last_text_skill_ms"] = round(
                (time.perf_counter() - t0) * 1000.0, 1)
            return True
        finally:
            self.stats["api_ms_total"] = round(
                float(self.stats.get("api_ms_total", 0.0))
                + (time.perf_counter() - t0) * 1000.0, 3)

    def step(self) -> None:
        """Hybrid local coach.

        The 256M model now ACTUALLY selects high-level skills from compact
        state.  Those text-only calls are cheap enough for the target laptop.
        Procedural cards remain a fail-safe if the tiny model stalls.  Full
        image inference is reserved for genuinely ambiguous semantic scenes.
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
            signature = (
                str(raw.get("skill_card") or ""),
                str(raw.get("scene") or ""),
                str(target.get("type") or ""),
            )
            ai_due = (
                signature != self._last_ai_signature
                or now - self._last_ai_call_ns >= int(90.0e9))
            if ai_due and now >= self._text_backoff_until_ns:
                try:
                    if self._run_text_skill_selector(
                            raw, obs, quest, catalog, now):
                        return
                except Exception as exc:
                    self.stats["api_errors"] = int(
                        self.stats.get("api_errors", 0)) + 1
                    self.stats["last_error"] = repr(exc)
                    self._text_backoff_until_ns = now + int(120.0e9)
                    self.events.publish({
                        "kind": "coach_error",
                        "ts_ns": now,
                        "model": self.model,
                        "error": repr(exc),
                        "fallback": "procedural_skill_router",
                        "call_type": "text_skill",
                    }, ts_ns=now)

            # Cheap deterministic fallback only when the actual local model
            # could not produce a valid skill.
            if signature != self._last_skill_signature:
                self._publish_skill_plan(
                    raw, catalog, now,
                    provider="procedural_skill_fallback",
                    local_vlm_used=False)
                self._last_skill_signature = signature
                self._last_skill_publish_ns = now
                self.stats["procedural_plans"] = (
                    int(self.stats.get("procedural_plans", 0)) + 1)
            else:
                self.stats["skips"] += 1
            return

        ui = dict(obs.get("ui") or {})
        target = dict(obs.get("target") or {})
        player = dict(obs.get("player") or {})
        phase = str(quest.get("phase") or "")
        try:
            health = float(player.get("health"))
        except (TypeError, ValueError):
            health = None

        needs_visual_semantics = bool(
            ui.get("dialogue") or ui.get("menu")
            or phase in {"obstacle_recovery", "stuck"}
            or (health is not None and health <= 0.30)
        )

        if not needs_visual_semantics:
            fallback = {
                "scene": "uncertain_no_visible_objective",
                "objective": "wait for a reliable quest/objective cue",
                "target": str(target.get("type") or "none"),
                "skill": "REOBSERVE",
                "confidence": 0.92,
                "explanation": (
                    "No reliable semantic target is visible; preserve control "
                    "until fast CV reacquires an objective."),
                "next_after_success": "resume the matching skill",
                "skill_card": "obstacle_recovery",
            }
            if self._last_skill_signature != (
                    "obstacle_recovery",
                    "uncertain_no_visible_objective",
                    str(target.get("type") or "")):
                self._publish_skill_plan(
                    fallback, catalog, now,
                    provider="procedural_skill_fallback",
                    local_vlm_used=False)
                self._last_skill_signature = (
                    "obstacle_recovery",
                    "uncertain_no_visible_objective",
                    str(target.get("type") or ""))
                self._last_skill_publish_ns = now
            else:
                self.stats["skips"] += 1
            return

        if now < self._vision_backoff_until_ns:
            self.stats["skips"] += 1
            return

        before_calls = int(self.stats.get("calls", 0))
        before_errors = int(self.stats.get("api_errors", 0))
        try:
            super().step()
        except Exception as exc:
            self.stats["api_errors"] = int(
                self.stats.get("api_errors", 0)) + 1
            self.stats["last_error"] = repr(exc)
            self.events.publish({
                "kind": "coach_error",
                "ts_ns": now,
                "model": self.model,
                "error": repr(exc),
                "fallback": "procedural_skills",
                "call_type": "vision",
            }, ts_ns=now)
        finally:
            if int(self.stats.get("calls", 0)) > before_calls:
                self.stats["visual_calls"] = int(
                    self.stats.get("visual_calls", 0)) + 1

        if int(self.stats.get("api_errors", 0)) > before_errors:
            self._vision_backoff_until_ns = now + int(180.0e9)

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
