"""Meta Model API semantic coach using Muse Spark.

This is the current hosted Meta inference path.  It replaces the retired
api.llama.com public-preview endpoint.  Muse Spark 1.3 is multimodal and
OpenAI-compatible, so the coach can inspect occasional Roblox frames while
keeping all heavy language/vision inference off the old Windows laptop.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .semantic_coach import SemanticCoachWorker


DEFAULT_META_MODEL = "muse-spark-1.3"
DEFAULT_META_BASE = "https://api.meta.ai/v1"


class MetaModelApiCoachWorker(SemanticCoachWorker):
    """Low-rate multimodal semantic coach using Meta Model API."""

    name = "meta_semantic_coach"

    def __init__(
        self,
        bus,
        target_hz: float = 1.0,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 30.0,
        api_key: str | None = None,
    ):
        key = str(api_key or os.getenv("MODEL_API_KEY") or "").strip()
        selected = str(
            model or os.getenv("META_MODEL") or DEFAULT_META_MODEL
        ).strip()
        super().__init__(
            bus,
            target_hz=target_hz,
            model=selected,
            min_call_interval_s=8.0,
            unchanged_refresh_s=30.0,
            timeout_s=timeout_s,
            api_key=key,
        )
        self.base_url = str(
            base_url or os.getenv("META_MODEL_API_BASE") or DEFAULT_META_BASE
        ).rstrip("/")
        self.provider = "meta_model_api"
        self.api_key = key
        self.model = selected
        self.supports_vision = True
        self.stats.update({
            "enabled": bool(self.api_key),
            "provider": self.provider,
            "base_url": self.base_url,
            "supports_vision": True,
            "disabled_reason": (
                None if self.api_key else "MODEL_API_KEY_missing"),
        })

    def _validate_plan(self, raw: dict, catalog: dict) -> dict:
        plan = super()._validate_plan(raw, catalog)
        plan["ai_used"] = True
        plan["cloud_ai_used"] = True
        plan["provider"] = self.provider
        return plan

    @staticmethod
    def _json_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "scene": {"type": "string"},
                "objective": {"type": "string"},
                "target": {"type": "string"},
                "skill": {"type": "string"},
                "control_id": {"type": "string"},
                "observed_ability": {
                    "type": "object",
                    "properties": {
                        "binding": {"type": "string"},
                        "label": {"type": "string"},
                    },
                    "required": ["binding", "label"],
                    "additionalProperties": False,
                },
                "ui_click": {
                    "type": "object",
                    "properties": {
                        "needed": {"type": "boolean"},
                        "x_norm": {"type": "number"},
                        "y_norm": {"type": "number"},
                        "label": {"type": "string"},
                    },
                    "required": [
                        "needed", "x_norm", "y_norm", "label"
                    ],
                    "additionalProperties": False,
                },
                "confidence": {"type": "number"},
                "explanation": {"type": "string"},
                "next_after_success": {"type": "string"},
                "knowledge_query": {"type": "string"},
                "memory_updates": {"type": "array"},
            },
            "required": [
                "scene", "objective", "target", "skill", "control_id",
                "observed_ability", "ui_click", "confidence",
                "explanation", "next_after_success", "knowledge_query",
                "memory_updates",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _strip_fence(text: str) -> str:
        text = str(text or "").strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            lines = text.splitlines()
            if lines and lines[0].startswith(fence):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith(fence):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _call_gemini(
            self, prompt: str, image_b64: str | None,
            previous_image_b64: str | None = None) -> dict:
        """Compatibility shim: parent worker calls this method name."""
        endpoint = self.base_url + "/chat/completions"

        content = [{"type": "text", "text": prompt}]
        if previous_image_b64:
            content.extend([
                {
                    "type": "text",
                    "text": (
                        "PREVIOUS GAME FRAME. Use only for progress/stuck "
                        "comparison; CURRENT frame is authoritative."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + previous_image_b64,
                    },
                },
            ])
        if image_b64:
            content.extend([
                {
                    "type": "text",
                    "text": "CURRENT GAME FRAME. This is authoritative.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + image_b64,
                    },
                },
            ])

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "reasoning_effort": "none",
            "max_completion_tokens": 560,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "digital_fly_coach_plan",
                    "schema": self._json_schema(),
                },
            },
        }

        def send(payload: dict) -> dict:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "DigitalFlyLab/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(
                    req, timeout=float(self.timeout_s)) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            payload = send(body)
        except urllib.error.HTTPError as exc:
            # Keep a compatibility retry for account/model differences in
            # structured output or reasoning controls.
            if int(getattr(exc, "code", 0)) != 400:
                raise
            fallback = {
                "model": self.model,
                "messages": body["messages"],
                "max_completion_tokens": 560,
            }
            payload = send(fallback)

        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"Meta Model API returned no choices: {str(payload)[:500]}")
        text = str(
            ((choices[0].get("message") or {}).get("content")) or ""
        ).strip()
        if not text:
            raise RuntimeError(
                f"Meta Model API returned no text: {str(payload)[:500]}")
        return json.loads(self._strip_fence(text))
