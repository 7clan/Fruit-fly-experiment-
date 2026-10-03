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
from .wiki import GPOWikiRetriever


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
    "USE_OBSERVED_ABILITY",
    "BUY_ITEM",
    "UI_CLICK",
    "BOARD_SHIP",
    "TRAVEL",
    "LOOK_LEFT",
    "LOOK_RIGHT",
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
        min_call_interval_s: float = 8.0,
        unchanged_refresh_s: float = 30.0,
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

        self.profile_path = root / "runtime_state" / "gpo_coach_profile.json"
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.profile = json.loads(
                self.profile_path.read_text(encoding="utf-8"))
            if not isinstance(self.profile, dict):
                self.profile = {"claims": {}}
        except Exception:
            self.profile = {"claims": {}}
        self.profile.setdefault("claims", {})

        self.wiki = GPOWikiRetriever(
            root / "runtime_state" / "gpo_wiki_cache.json",
            timeout_s=min(6.0, self.timeout_s))
        self.allow_remote_wiki = True
        self.provider = "gemini"

        self._last_call_ns = 0
        self._last_signature = None
        self.drop_stale_responses = False
        self._previous_image_b64 = None
        self._plan_id = 0
        self._disabled_published = False
        self.stats.update({
            "enabled": bool(self.api_key),
            "model": self.model,
            "calls": 0,
            "plans": 0,
            "skips": 0,
            "api_errors": 0,
            "stale_responses": 0,
            "jpeg_ms_total": 0.0,
            "api_ms_total": 0.0,
            "last_api_ms": None,
            "avg_api_ms": None,
            "last_plan": None,
            "profile_claims": len(self.profile.get("claims", {})),
            "wiki_queries": 0,
            "wiki_hits": 0,
            "wiki_errors": 0,
            "refinement_calls": 0,
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
            if w > 720:
                scale = 720.0 / float(w)
                img = cv2.resize(
                    img, (720, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 58])
            if not ok:
                return None, 0.0
            data = base64.b64encode(enc.tobytes()).decode("ascii")
            return data, (time.perf_counter() - t0) * 1000.0
        except Exception:
            return None, (time.perf_counter() - t0) * 1000.0

    def _prompt(
        self, obs: dict, quest: dict, brain: dict,
        catalog: dict, action_meta: dict,
        previous_plan: dict,
        retrieved_context: str = "",
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
            "previous_coach_plan": _compact(previous_plan, 900),
            "persistent_character_profile": _compact(
                self.profile, 2400),
            "on_demand_wiki_context": (
                retrieved_context[:6500] if retrieved_context else ""),
        }
        return f"""You are the SEMANTIC COACH for a Grand Piece Online
autonomous research agent.  You understand game meaning and choose ONE
high-level next skill.  You do NOT replace the fruit-fly brain's low-level
left/right/approach navigation.

IMPORTANT RULES:
- Never output a raw keyboard key or scan code.
- For EXEC_CONTROL choose control_id only from verified_controls.
- For USE_OBSERVED_ABILITY, the binding AND move label must both be visibly
  readable on the CURRENT HUD. Never guess a fruit/style/sword ability key.
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
  "observed_ability": {
    "binding": "visible HUD key such as E/R/Z/X/C/V/B/N or empty",
    "label": "visible HUD move name or empty"
  },
  "ui_click": {{
    "needed": false,
    "x_norm": 0.0,
    "y_norm": 0.0,
    "label": ""
  }},
  "confidence": 0.0,
  "explanation": "one short sentence",
  "next_after_success": "one short next step",
  "knowledge_query": "short GPO wiki search phrase or empty string",
  "memory_updates": [
    {{
      "key": "one of: level,peli,island,active_quest,fruit,fighting_style,primary_weapon,ship,buso_haki,observation_haki,equipment,hotbar,stat_build",
      "value": "observed value",
      "confidence": 0.0,
      "evidence": "visible or state"
    }}
  ]
}}"""

    def _call_gemini(
            self, prompt: str, image_b64: str | None,
            previous_image_b64: str | None = None) -> dict:
        """Call Google's native generateContent REST API.

        The AI-only branch previously used the OpenAI compatibility surface,
        which returned HTML HTTP 400 on the target machine. Google's docs
        recommend the native API when compatibility is not required.
        """
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent")

        parts = []
        if previous_image_b64:
            parts.extend([
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": previous_image_b64,
                    },
                },
                {
                    "text": (
                        "PREVIOUS GAME FRAME from the prior observation. "
                        "Use it only for temporal context.")
                },
            ])
        if image_b64:
            parts.extend([
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {
                    "text": (
                        "CURRENT GAME FRAME. This is the authoritative "
                        "visual observation.")
                },
            ])
        parts.append({"text": prompt})

        model = str(self.model or "")
        if model.startswith("gemini-2.5"):
            thinking = {"thinkingBudget": 0}
        elif model.startswith("gemini-3.8") or model.startswith("gemini-3.7"):
            thinking = {"thinkingLevel": "low"}
        else:
            # Flash-Lite 3.1/3.5 supports minimal and is optimized for this
            # low-latency controller workload.
            thinking = {"thinkingLevel": "minimal"}

        generation = {
            "maxOutputTokens": int(
                getattr(self, "max_output_tokens", 620)),
            "responseMimeType": "application/json",
            "thinkingConfig": thinking,
        }
        body = {
            "contents": [{
                "role": "user",
                "parts": parts,
            }],
            "generationConfig": generation,
        }

        def send(request_body):
            raw = json.dumps(request_body).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=raw,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                    "x-goog-api-client": "digitalflylab-native/1.0",
                    "User-Agent": "DigitalFlyLab/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(
                    req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            payload = send(body)
        except urllib.error.HTTPError as exc:
            # If a particular generation rejects structured JSON or a thinking
            # field, keep the same native transport and retry with the common
            # generateContent surface. The prompt itself still demands JSON.
            if int(getattr(exc, "code", 0)) != 400:
                raise
            fallback = {
                "contents": body["contents"],
                "generationConfig": {
                    "maxOutputTokens": int(
                        getattr(self, "max_output_tokens", 620)),
                },
            }
            payload = send(fallback)

        candidates = payload.get("candidates") or []
        if not candidates:
            raise RuntimeError(
                f"Gemini returned no candidates: {str(payload)[:600]}")
        out_parts = (
            (candidates[0].get("content") or {}).get("parts") or [])
        text = "".join(
            str(p.get("text") or "")
            for p in out_parts if isinstance(p, dict)
        ).strip()
        if not text:
            raise RuntimeError(
                f"Gemini returned no text: {str(payload)[:600]}")

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)

    _PROFILE_KEYS = frozenset({
        "level", "peli", "island", "active_quest", "fruit",
        "fighting_style", "primary_weapon", "ship", "buso_haki",
        "observation_haki", "equipment", "hotbar", "stat_build",
    })

    def _validated_memory_updates(self, raw_updates) -> list[dict]:
        out = []
        if not isinstance(raw_updates, list):
            return out
        for rec in raw_updates[:16]:
            if not isinstance(rec, dict):
                continue
            key = str(rec.get("key") or "").strip()
            evidence = str(rec.get("evidence") or "").strip().lower()
            try:
                conf = float(rec.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if (key not in self._PROFILE_KEYS
                    or evidence not in {"visible", "state"}
                    or conf < 0.85):
                continue
            value = rec.get("value")
            # Keep persistent claims compact and JSON-safe.
            try:
                encoded = json.dumps(value, default=str)
            except Exception:
                continue
            if len(encoded) > 900:
                continue
            out.append({
                "key": key,
                "value": value,
                "confidence": round(min(1.0, max(0.0, conf)), 4),
                "evidence": evidence,
            })
        return out

    def _apply_memory_updates(self, updates: list[dict], now_ns: int) -> None:
        if not updates:
            return
        claims = self.profile.setdefault("claims", {})
        for rec in updates:
            claims[rec["key"]] = {
                "value": rec["value"],
                "confidence": rec["confidence"],
                "evidence": rec["evidence"],
                "updated_ns": int(now_ns),
            }
        self.profile["updated_ns"] = int(now_ns)
        tmp = self.profile_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.profile, indent=1, default=str),
            encoding="utf-8")
        tmp.replace(self.profile_path)
        self.stats["profile_claims"] = len(claims)

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

        ability = raw.get("observed_ability") or {}
        binding = str(ability.get("binding") or "").strip().upper()
        ability_label = str(ability.get("label") or "").strip()[:96]
        if binding not in {"E", "R", "Z", "X", "C", "V", "B", "N", "Q", "F", "G", "J"}:
            binding = ""
        if skill == "USE_OBSERVED_ABILITY" and (
                not binding or not ability_label or confidence < 0.80):
            skill = "REOBSERVE"

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
            "provider": self.provider,
            "model": self.model,
            "scene": str(raw.get("scene") or "unknown")[:96],
            "objective": str(raw.get("objective") or "")[:240],
            "target": str(raw.get("target") or "none")[:160],
            "skill": skill,
            "control_id": control_id,
            "observed_ability": {
                "binding": binding,
                "label": ability_label,
            },
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
            "knowledge_query": str(
                raw.get("knowledge_query") or "")[:120],
            "memory_updates": self._validated_memory_updates(
                raw.get("memory_updates")),
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

        previous_plan = self._state(self.plan_state)
        prompt = self._prompt(
            obs, quest, brain, catalog, meta, previous_plan)
        previous_image = self._previous_image_b64
        self._last_call_ns = now
        self._last_signature = sig
        t0 = time.perf_counter()
        try:
            raw = self._call_gemini(
                prompt, image_b64, previous_image_b64=previous_image)

            # Rare/detail-heavy questions are looked up on the public GPO
            # MediaWiki API only when the model explicitly asks.  This avoids
            # shipping a giant stale encyclopedia on every cheap vision call.
            query = str(raw.get("knowledge_query") or "").strip()[:120]
            if query and self.allow_remote_wiki:
                self.stats["wiki_queries"] += 1
                try:
                    wiki_context = self.wiki.lookup(query)
                except Exception as wiki_exc:
                    wiki_context = ""
                    self.stats["wiki_errors"] += 1
                    self.events.publish({
                        "kind": "coach_wiki_error",
                        "ts_ns": SHARED_CLOCK.now_ns(),
                        "query": query,
                        "error": repr(wiki_exc),
                    })
                if wiki_context:
                    self.stats["wiki_hits"] += 1
                    refine_prompt = self._prompt(
                        obs, quest, brain, catalog, meta, previous_plan,
                        retrieved_context=wiki_context)
                    refine_prompt += (
                        "\n\nThis is the refinement pass. Use the retrieved "
                        "wiki context if relevant and return the FINAL one-step "
                        "plan. Do not request another lookup.")
                    raw = self._call_gemini(
                        refine_prompt, image_b64,
                        previous_image_b64=previous_image)
                    self.stats["calls"] += 1
                    self.stats["refinement_calls"] += 1

            plan = self._validate_plan(raw, catalog)

            # Cloud inference can take long enough for the physical game state
            # to change (quest accepted, UI opened, target appeared, etc.).
            # AI-only mode opts into dropping a response that was generated
            # for an older semantic state instead of executing stale advice.
            if bool(getattr(self, "drop_stale_responses", False)):
                cur_obs = self._state(self.world_state)
                cur_quest = self._state(self.quest_state)
                cur_brain = self._state(self.brain_state)
                cur_sig = self._signature(cur_obs, cur_quest, cur_brain)
                if cur_sig != sig:
                    self.stats["stale_responses"] += 1
                    self._last_signature = None
                    self.events.publish({
                        "kind": "coach_stale_response_dropped",
                        "ts_ns": SHARED_CLOCK.now_ns(),
                        "model": self.model,
                    })
                    return
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
            if image_b64:
                self._previous_image_b64 = image_b64
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.stats["api_ms_total"] = round(
                float(self.stats["api_ms_total"]) + elapsed_ms, 3)
            self.stats["calls"] += 1
            self.stats["last_api_ms"] = round(elapsed_ms, 1)
            calls = max(1, int(self.stats.get("calls", 1)))
            self.stats["avg_api_ms"] = round(
                float(self.stats["api_ms_total"]) / calls, 1)

        self._apply_memory_updates(
            plan.get("memory_updates") or [], plan["ts_ns"])
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
        why = str(plan.get("explanation") or "").replace("\n", " ")
        if len(why) > 160:
            why = why[:157] + "..."
        plan_log_sig = (
            str(plan.get("skill") or ""),
            str(plan.get("target") or ""),
            why,
        )
        suppress_dupes = bool(
            getattr(self, "suppress_duplicate_plan_logs", False))
        if (not suppress_dupes
                or plan_log_sig != getattr(
                    self, "_last_printed_plan_sig", None)):
            print(
                f"[AI->PLAN] skill={plan.get('skill')} "
                f"target={plan.get('target')} "
                f"source={plan.get('provider')} ai=yes "
                f"why={why}",
                flush=True,
            )
            self._last_printed_plan_sig = plan_log_sig
        else:
            self.stats["suppressed_plan_logs"] = int(
                self.stats.get("suppressed_plan_logs", 0)) + 1
