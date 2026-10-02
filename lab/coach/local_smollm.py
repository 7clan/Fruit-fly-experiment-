"""Ultra-light local SmolLM2-135M semantic coach.

This provider is TEXT ONLY. OpenCV/engineered perception turns the Roblox
screen into compact structured state; SmolLM chooses one bounded high-level
skill from the current skill card. The fruit-fly brain and fresh CV servo keep
ownership of time-critical locomotor steering.

The design is intentionally small for the i7-5500U / 8 GB target machine:
- 135M instruction model, Q4_K_M GGUF through llama.cpp
- no vision encoder / mmproj
- one CPU inference thread in the launcher
- schema-constrained one-field output
- procedural safe fallback if the tiny model stalls or disagrees
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .semantic_coach import SemanticCoachWorker
from .gpo_skills import SKILL_CARDS, procedural_skill_plan


DEFAULT_LOCAL_MODEL = "smollm2-135m"
DEFAULT_LOCAL_URL = "http://127.0.0.1:18080/v1"


class LocalSmolLMCoachWorker(SemanticCoachWorker):
    """Text-only local skill selector for structured GPO state."""

    name = "local_semantic_coach"

    def __init__(
        self,
        bus,
        target_hz: float = 0.5,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 8.0,
    ):
        super().__init__(
            bus,
            target_hz=target_hz,
            model=model or DEFAULT_LOCAL_MODEL,
            min_call_interval_s=45.0,
            unchanged_refresh_s=120.0,
            timeout_s=timeout_s,
            api_key="local-no-key",
        )
        self.base_url = str(base_url or DEFAULT_LOCAL_URL).rstrip("/")
        self.provider = "local_smollm2_135m_llamacpp"
        self.allow_remote_wiki = False
        self._last_ai_signature = None
        self._last_ai_call_ns = 0
        self._last_plan_signature = None
        self._text_backoff_until_ns = 0
        self._last_model_signature = None
        self._last_model_skill = None
        self._same_model_skill_streak = 0
        self.stats.update({
            "provider": self.provider,
            "base_url": self.base_url,
            "local_only_inference": True,
            "text_only": True,
            "vision_encoder": False,
            "model_parameters": "135M",
            "procedural_plans": 0,
            "text_skill_calls": 0,
            "schema_constrained_calls": 0,
        })

    @staticmethod
    def _card_actions(card_id: str) -> set[str]:
        for card in SKILL_CARDS:
            if card.skill_id == str(card_id or ""):
                return {str(x).upper() for x in card.actions}
        return {"WAIT", "REOBSERVE"}

    @staticmethod
    def _f(value):
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return None

    def _skill_prompt(
            self, candidate: dict, obs: dict, quest: dict,
            choices: list[str]) -> str:
        """Tiny index-selection prompt; the model emits one digit only."""
        target = dict(obs.get("target") or {})
        notes = dict(obs.get("notes") or {})
        player = dict(obs.get("player") or {})
        state = {
            "candidate": str(candidate.get("skill") or ""),
            "target": str(target.get("type") or "none"),
            "dir": self._f(target.get("direction")),
            "dist": self._f(target.get("distance")),
            "phase": str(quest.get("phase") or ""),
            "hp": self._f(player.get("health")),
            "yellow": bool(notes.get("quest_marker_detected")),
            "red": bool(notes.get("quest_enemy_marker_detected")
                        or notes.get("red_enemy")),
        }
        mapping = ";".join(
            f"{idx}={skill}" for idx, skill in enumerate(choices))
        return (
            "GPO control. Return the INDEX of the best action. "
            "Prefer candidate unless clearly wrong. Never target players. "
            f"STATE={json.dumps(state, separators=(',', ':'))} "
            f"CHOICES={mapping} INDEX="
        )

    def _call_skill(self, prompt: str, choices: list[str]) -> str:
        """One-character constrained decision via llama.cpp.

        This avoids generating JSON on the old CPU. The user-supplied GBNF
        grammar restricts output to a single valid choice index.
        """
        endpoint = self.base_url + "/chat/completions"
        if not choices or len(choices) > 9:
            raise ValueError(f"unsupported choice count: {len(choices)}")
        grammar = "root ::= " + " | ".join(
            f'\"{idx}\"' for idx in range(len(choices)))
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 2,
            "grammar": grammar,
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
                req, timeout=float(self.timeout_s)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        completions = payload.get("choices") or []
        text = str(
            (((completions[0] if completions else {}).get("message") or {})
             .get("content")) or ""
        ).strip()
        if not text:
            raise RuntimeError(
                f"SmolLM returned no choice: {str(payload)[:300]}")
        try:
            idx = int(text[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"SmolLM returned invalid index {text!r}") from exc
        if idx < 0 or idx >= len(choices):
            raise ValueError(
                f"SmolLM index {idx} outside choices={choices}")
        return choices[idx]

    def _publish(
            self, raw: dict, catalog: dict, now_ns: int, *,
            provider: str, local_ai_used: bool) -> None:
        raw = dict(raw or {})
        raw.setdefault("control_id", "")
        raw.setdefault("observed_ability", {"binding": "", "label": ""})
        raw.setdefault(
            "ui_click",
            {"needed": False, "x_norm": 0.0, "y_norm": 0.0, "label": ""})
        raw.setdefault("memory_updates", [])
        raw.setdefault("knowledge_query", "")
        raw.setdefault("confidence", 0.86)
        raw.setdefault("scene", "structured_state")
        raw.setdefault("objective", "")
        raw.setdefault("target", "none")
        raw.setdefault("skill", "REOBSERVE")
        raw.setdefault("explanation", "")
        raw.setdefault("next_after_success", "")

        plan = self._validate_plan(raw, catalog)
        plan["provider"] = provider
        plan["model"] = self.model
        plan["skill_card"] = raw.get("skill_card")
        plan["local_ai_used"] = bool(local_ai_used)
        # Backward-compatible dashboard field used by current renderer.
        plan["local_vlm_used"] = bool(local_ai_used)
        plan["text_only"] = True
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
            "local_ai_used": bool(local_ai_used),
        }
        # Human-visible low-rate trace of exactly what the semantic coach
        # handed downstream. Useful even if the OpenCV dashboard is lagging.
        why = str(plan.get("explanation") or "").replace("\n", " ")
        if len(why) > 140:
            why = why[:137] + "..."
        print(
            f"[AI->FLY] skill={plan.get('skill')} "
            f"target={plan.get('target')} "
            f"source={provider} "
            f"ai={'yes' if local_ai_used else 'fallback'} "
            f"why={why}",
            flush=True,
        )

    def _select_with_model(
            self, candidate: dict, obs: dict, quest: dict,
            catalog: dict, now_ns: int) -> bool:
        allowed = self._card_actions(candidate.get("skill_card"))
        signature = (
            str(candidate.get("skill_card") or ""),
            str(candidate.get("scene") or ""),
            str((obs.get("target") or {}).get("type") or ""),
        )
        choices = sorted(allowed)
        # If the tiny model repeats the same recovery skill twice without the
        # structured scene changing, force it to try another safe option.
        if (
                signature == self._last_model_signature
                and self._same_model_skill_streak >= 2
                and self._last_model_skill in choices
                and len(choices) > 1):
            choices = [
                x for x in choices if x != self._last_model_skill]
        prompt = self._skill_prompt(candidate, obs, quest, choices)
        t0 = time.perf_counter()
        self.stats["calls"] = int(self.stats.get("calls", 0)) + 1
        self.stats["text_skill_calls"] = int(
            self.stats.get("text_skill_calls", 0)) + 1
        self.stats["schema_constrained_calls"] = int(
            self.stats.get("schema_constrained_calls", 0)) + 1
        try:
            chosen = self._call_skill(prompt, choices)
            if (
                    signature == self._last_model_signature
                    and chosen == self._last_model_skill):
                self._same_model_skill_streak += 1
            else:
                self._same_model_skill_streak = 1
            self._last_model_signature = signature
            self._last_model_skill = chosen
            merged = dict(candidate)
            merged["skill"] = chosen
            merged["explanation"] = (
                f"SmolLM2-135M selected {chosen} from "
                f"{choices}.")
            self._publish(
                merged, catalog, now_ns,
                provider="local_smollm2_135m_skill",
                local_ai_used=True)
            self._last_ai_call_ns = now_ns
            self._last_ai_signature = (
                str(candidate.get("skill_card") or ""),
                str(candidate.get("scene") or ""),
                str((obs.get("target") or {}).get("type") or ""),
            )
            self.stats["last_text_skill_ms"] = round(
                (time.perf_counter() - t0) * 1000.0, 1)
            return True
        finally:
            self.stats["api_ms_total"] = round(
                float(self.stats.get("api_ms_total", 0.0))
                + (time.perf_counter() - t0) * 1000.0, 3)

    def step(self) -> None:
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

        candidate = procedural_skill_plan(obs, quest)
        if candidate is None:
            candidate = {
                "scene": "uncertain_structured_state",
                "objective": "wait for reliable CV/objective state",
                "target": str((obs.get("target") or {}).get("type") or "none"),
                "skill": "REOBSERVE",
                "confidence": 0.95,
                "explanation": (
                    "Structured perception has no safe actionable objective."),
                "next_after_success": "re-evaluate when state changes",
                "skill_card": "obstacle_recovery",
            }

        signature = (
            str(candidate.get("skill_card") or ""),
            str(candidate.get("scene") or ""),
            str((obs.get("target") or {}).get("type") or ""),
        )
        ai_due = (
            signature != self._last_ai_signature
            or now - self._last_ai_call_ns >= int(45.0e9)
        )

        if ai_due and now >= self._text_backoff_until_ns:
            try:
                if self._select_with_model(
                        candidate, obs, quest, catalog, now):
                    self._last_plan_signature = signature
                    return
            except (
                    urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, json.JSONDecodeError, ValueError,
                    RuntimeError) as exc:
                self.stats["api_errors"] = int(
                    self.stats.get("api_errors", 0)) + 1
                self.stats["last_error"] = repr(exc)
                # Back off so Brian2 keeps priority if the tiny model has an
                # unexpected slow path on this old CPU.
                self._text_backoff_until_ns = now + int(90.0e9)
                self.events.publish({
                    "kind": "coach_error",
                    "ts_ns": now,
                    "model": self.model,
                    "error": repr(exc),
                    "fallback": "procedural_skill_router",
                    "call_type": "smollm_text_skill",
                }, ts_ns=now)

        if signature != self._last_plan_signature:
            self._publish(
                candidate, catalog, now,
                provider="procedural_skill_fallback",
                local_ai_used=False)
            self._last_plan_signature = signature
            self.stats["procedural_plans"] = int(
                self.stats.get("procedural_plans", 0)) + 1
        else:
            self.stats["skips"] += 1
