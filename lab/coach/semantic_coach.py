"""Gemini Flash-Lite semantic coach [ENGINEERED].

This worker exists above the biological fly brain:

    game frame + compact state -> semantic understanding / high-level skill
    fly brain                  -> low-level orientation/navigation intention
    quest supervisor           -> validates and executes approved skill

The cloud model NEVER emits raw scan codes.  It can only select from the
semantic skill vocabulary and verified GPO control catalog.  Calls are
low-rate/event-triggered so the 2-core target laptop spends CPU on the
canonical Brian2 brain rather than another local neural model.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..clock import SHARED_CLOCK
from ..worker import Worker


DEFAULT_MODEL = "gemini-3.5-flash-lite"

ALLOWED_SKILLS = frozenset({
    "WAIT",
    "TAKE_QUEST",
    "NAVIGATE_OBJECTIVE",
    "FIGHT_QUEST_TARGET",
    "BLOCK",
    "EVADE",
    "JUMP",
    "CLIMB",
    "GO_AROUND",
    "BACKTRACK",
    "SPRINT",
    "GEPPO",
    "INTERACT",
    "USE_HAKI",
    "EQUIP_SLOT",
    "EXEC_CONTROL",
    "BUY_ITEM",
    "UI_CLICK",
    "BOARD_SHIP",
    "TRAVEL",
    "REOBSERVE",
})

# Skills that can be bridged into an immediate semantic command. Navigation
# itself remains owned by the fly brain.
IMMEDIATE_SKILLS = frozenset({
    "TAKE_QUEST", "BLOCK", "EVADE", "JUMP", "CLIMB", "SPRINT", "GEPPO",
    "INTERACT", "USE_HAKI", "EQUIP_SLOT", "EXEC_CONTROL", "UI_CLICK",
    "BOARD_SHIP",
})


def _compact(value: Any, limit: int = 1600) -> Any:
    """Cheap JSON-safe compacting for state sent to the API."""
    try:
        txt = json.dumps(value, default=str, separators=(",", ":"))
    except Exception:
        txt = repr(value)
    if len(txt) <= limit:
        return value
    return {"truncated": txt[:limit]}


class SemanticCoachWorker(Worker):
    """Low-rate multimodal instructor using Gemini Flash-Lite."""

    name = "semantic_coach"

    def __init__(
        self,
        bus,
        target_hz: float = 1.0,
        model: str | None = None,
        min_call_interval_s: float = 6.0,
        unchanged_refresh_s: float = 18.0,
        timeout_s: float = 12.0,
        knowledge_path: Path | None = None,
        api_key: str | None = None,
    ):
        super().__init__(bus, target_hz=target_hz)
        self.frame_state = bus.state("capture.frames.latest")
        self.world_state = bus.state("world.observation")
        self.quest_state = bus.state("quest.state")
        self.brain_state = bus.state("brain.output")
        self.meta_state = bus.state("action.meta")
        self.catalog_state = bus.state("action.control_catalog")
        self.plan_state = bus.state("coach.plan")
        self.events = bus.stream("coach.events", maxsize=64)

        self.model = str(
            model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
        self.api_key = str(
            api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        self.min_call_interval_s = max(2.0, float(min_call_interval_s))
        self.unchanged_refresh_s = max(
            self.min_call_interval_s, float(unchanged_refresh_s))
        self.timeout_s = max(3.0, float(timeout_s))

        root = Path(__file__).resolve().parents[2]
        self.knowledge_path = (
            Path(knowledge_path) if knowledge_path
            else root / "knowledge" / "GPO_SEMANTIC_PLAYBOOK.md")
        try:
            self.knowledge = self.knowledge_path.read_text(
                encoding="utf-8")
        except Exception:
            self.knowledge = ""

        self._last_call_ns = 0
        self._last_signature = None
        self._plan_id = 0
        self._disabled_published = False
        self.stats.update({
            "enabled": bool(self.api_key),
            "model": self.model,
            "calls": 0,
            "plans": 0,
            "skips": 0,
            "api_errors": 0,
            "jpeg_ms_total": 0.0,
            "api_ms_total": 0.0,
            "last_plan": None,
            "disabled_reason": (
                None if self.api_key else "GEMINI_API_KEY_missing"),
        })

    # ------------------------------------------------------------------
    def _state(self, channel) -> dict:
        env = channel.read()
        return dict(env.payload or {}) if env is not None else {}

    def _signature(self, obs: dict, quest: dict, brain: dict) -> tuple:
        target = obs.get("target") or {}
        player = obs.get("player") or {}
        notes = obs.get("notes") or {}
        try:
            health_bucket = round(float(player.get("health", 1.0)) * 10)
        except (TypeError, ValueError):
            health_bucket = -1
        return (
            str(target.get("type") or "none"),
            round(float(target.get("direction") or 0.0), 1),
            round(float(target.get("distance") or 0.0), 1),
            str(quest.get("phase") or "none"),
            str((brain.get("intention") or {}).get("name") or "none"),
            health_bucket,
            bool(notes.get("quest_prompt_visible")),
            bool(notes.get("red_enemy_marker")),
            bool(notes.get("recommended_waypoint_xy")),
        )

    def _jpeg_b64(self, frame) -> tuple[str | None, float]:
        if frame is None:
            return None, 0.0
        t0 = time.perf_counter()
        try:
            import cv2
            img = frame
            h, w = img.shape[:2]
            if w > 480:
                scale = 480.0 / float(w)
                img = cv2.resize(
                    img, (480, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            if not ok:
                return None, 0.0
            data = base64.b64encode(enc.tobytes()).decode("ascii")
            return data, (time.perf_counter() - t0) * 1000.0
        except Exception:
            return None, (time.perf_counter() - t0) * 1000.0

    def _prompt(
        self, obs: dict, quest: dict, brain: dict,
        catalog: dict, action_meta: dict,
    ) -> str:
        controls = [
            {
                "control_id": c.get("control_id"),
                "name": c.get("name"),
                "category": c.get("category"),
                "pattern": c.get("pattern"),
                "context": c.get("context"),
            }
            for c in (catalog.get("controls") or [])
            if c.get("control_id")
        ]
        situation = {
            "world": _compact(obs, 2200),
            "quest": _compact(quest, 1000),
            "fly_brain": _compact(brain, 1000),
            "agent": {
                "enabled": bool(action_meta.get("autonomy")),
                "loadout": action_meta.get("gpo_loadout"),
            },
            "verified_controls": controls,
        }
        return f"""You are the SEMANTIC COACH for a Grand Piece Online
autonomous research agent.  You understand game meaning and choose ONE
high-level next skill.  You do NOT replace the fruit-fly brain's low-level
left/right/approach navigation.

IMPORTANT RULES:
- Never output a raw keyboard key or scan code.
- For EXEC_CONTROL choose control_id only from verified_controls.
- Never attack ordinary players; combat target must be a quest-marked NPC or
  an immediate NPC threat.
- Never autonomously confirm trades, Robux purchases, account/security UI, or
  external links.
- UI_CLICK is allowed only for a clearly visible in-game dialog/menu/shop
  button, confidence >= 0.85, with normalized x/y in the supplied image.
- Never move or drag the camera.
- If uncertain choose REOBSERVE or WAIT.
- Use the live screenshot/HUD as truth when it conflicts with static knowledge.
- Prefer the shortest action that makes measurable progress.
- Explain the situation in one short sentence.

STATIC GPO PLAYBOOK:
{self.knowledge}

CURRENT STATE:
{json.dumps(situation, default=str)}

Return ONLY a JSON object with exactly these fields:
{{
  "scene": "short_scene_label",
  "objective": "what the player should accomplish next",
  "target": "specific visible/known target or none",
  "skill": "ONE OF: {", ".join(sorted(ALLOWED_SKILLS))}",
  "control_id": "verified control id or empty string",
  "ui_click": {{
    "needed": false,
    "x_norm": 0.0,
    "y_norm": 0.0,
    "label": ""
  }},
  "confidence": 0.0,
  "explanation": "one short sentence",
  "next_after_success": "one short next step"
}}"""

    def _call_gemini(self, prompt: str, image_b64: str | None) -> dict:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent")
        parts = [{"text": prompt}]
        if image_b64:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": image_b64,
                }
            })
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.15,
                "maxOutputTokens": 420,
                "responseMimeType": "application/json",
            },
        }
        raw = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=raw,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(
                req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        parts_out = (((payload.get("candidates") or [{}])[0]
                      .get("content") or {}).get("parts") or [])
        text = "".join(str(p.get("text") or "") for p in parts_out)
        if not text:
            raise RuntimeError(
                f"Gemini returned no text: {str(payload)[:400]}")
        return json.loads(text)

    def _validate_plan(self, raw: dict, catalog: dict) -> dict:
        if not isinstance(raw, dict):
            raise ValueError("coach output must be an object")
        skill = str(raw.get("skill") or "REOBSERVE").upper()
        if skill not in ALLOWED_SKILLS:
            skill = "REOBSERVE"

        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        valid_ids = {
            str(c.get("control_id"))
            for c in (catalog.get("controls") or [])
            if c.get("control_id")
        }
        control_id = str(raw.get("control_id") or "")
        if control_id not in valid_ids:
            control_id = ""

        ui = raw.get("ui_click") or {}
        try:
            x = max(0.0, min(1.0, float(ui.get("x_norm", 0.0))))
            y = max(0.0, min(1.0, float(ui.get("y_norm", 0.0))))
        except (TypeError, ValueError):
            x, y = 0.0, 0.0
        ui_needed = bool(ui.get("needed")) and confidence >= 0.85
        if skill == "UI_CLICK" and not ui_needed:
            skill = "REOBSERVE"

        self._plan_id += 1
        return {
            "plan_id": self._plan_id,
            "ts_ns": SHARED_CLOCK.now_ns(),
            "provider": "gemini",
            "model": self.model,
            "scene": str(raw.get("scene") or "unknown")[:96],
            "objective": str(raw.get("objective") or "")[:240],
            "target": str(raw.get("target") or "none")[:160],
            "skill": skill,
            "control_id": control_id,
            "ui_click": {
                "needed": ui_needed,
                "x_norm": round(x, 4),
                "y_norm": round(y, 4),
                "label": str(ui.get("label") or "")[:96],
            },
            "confidence": round(confidence, 4),
            "explanation": str(raw.get("explanation") or "")[:300],
            "next_after_success": str(
                raw.get("next_after_success") or "")[:240],
            "ENGINEERED": True,
        }

    # ------------------------------------------------------------------
    def step(self) -> None:
        now = SHARED_CLOCK.now_ns()
        if not self.api_key:
            if not self._disabled_published:
                payload = {
                    "enabled": False,
                    "reason": "GEMINI_API_KEY_missing",
                    "model": self.model,
                    "ts_ns": now,
                    "ENGINEERED": True,
                }
                self.plan_state.write(payload, ts_ns=now)
                self._disabled_published = True
            self.stats["skips"] += 1
            return

        meta = self._state(self.meta_state)
        # Do not spend cloud calls while the user has the agent paused.
        if not bool(meta.get("autonomy")):
            self.stats["skips"] += 1
            return

        obs = self._state(self.world_state)
        if not obs:
            self.stats["skips"] += 1
            return
        quest = self._state(self.quest_state)
        brain = self._state(self.brain_state)
        catalog = self._state(self.catalog_state)

        sig = self._signature(obs, quest, brain)
        age_s = (
            (now - self._last_call_ns) / 1e9
            if self._last_call_ns else 1e9)
        if age_s < self.min_call_interval_s:
            self.stats["skips"] += 1
            return
        if sig == self._last_signature and age_s < self.unchanged_refresh_s:
            self.stats["skips"] += 1
            return

        frame_env = self.frame_state.read()
        frame = None
        if frame_env is not None:
            frame = (frame_env.payload or {}).get("data_ref")
        image_b64, jpeg_ms = self._jpeg_b64(frame)
        self.stats["jpeg_ms_total"] = round(
            float(self.stats["jpeg_ms_total"]) + jpeg_ms, 3)

        prompt = self._prompt(obs, quest, brain, catalog, meta)
        self._last_call_ns = now
        self._last_signature = sig
        t0 = time.perf_counter()
        try:
            raw = self._call_gemini(prompt, image_b64)
            plan = self._validate_plan(raw, catalog)
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, json.JSONDecodeError, ValueError,
                RuntimeError) as exc:
            self.stats["api_errors"] += 1
            self.stats["last_error"] = repr(exc)
            event = {
                "kind": "coach_error",
                "ts_ns": SHARED_CLOCK.now_ns(),
                "model": self.model,
                "error": repr(exc),
            }
            self.events.publish(event, ts_ns=event["ts_ns"])
            return
        finally:
            self.stats["api_ms_total"] = round(
                float(self.stats["api_ms_total"])
                + (time.perf_counter() - t0) * 1000.0, 3)
            self.stats["calls"] += 1

        self.plan_state.write(plan, ts_ns=plan["ts_ns"])
        self.events.publish(
            {"kind": "coach_plan", "plan": plan, "ts_ns": plan["ts_ns"]},
            ts_ns=plan["ts_ns"])
        self.stats["plans"] += 1
        self.stats["last_plan"] = {
            "plan_id": plan["plan_id"],
            "scene": plan["scene"],
            "skill": plan["skill"],
            "confidence": plan["confidence"],
        }
