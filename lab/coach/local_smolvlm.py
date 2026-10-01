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
        timeout_s: float = 120.0,
    ):
        super().__init__(
            bus,
            target_hz=target_hz,
            model=model or DEFAULT_LOCAL_MODEL,
            min_call_interval_s=18.0,
            unchanged_refresh_s=60.0,
            timeout_s=timeout_s,
            api_key="local-no-key",
        )
        self.base_url = str(base_url or DEFAULT_LOCAL_URL).rstrip("/")
        self.provider = "local_smolvlm2_llamacpp"
        self.allow_remote_wiki = False
        self._sections = self._split_sections(self.knowledge)
        self.stats.update({
            "provider": self.provider,
            "base_url": self.base_url,
            "remote_wiki": False,
            "local_only_inference": True,
            "min_call_interval_s": self.min_call_interval_s,
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
            if w > 512:
                scale = 512.0 / float(w)
                img = cv2.resize(
                    img, (512, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 52])
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
            "world": _compact(obs, 1900),
            "quest": _compact(quest, 700),
            "fly_intention": (brain.get("intention") or {}).get("name"),
            "loadout": action_meta.get("gpo_loadout"),
            "profile": _compact(self.profile, 1200),
            "controls": controls,
            "previous_plan": _compact(previous_plan, 500),
        }
        situation_text = json.dumps(compact_state, default=str)
        knowledge = self._relevant_playbook(situation_text)
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
            "temperature": 0.1,
            "max_tokens": 360,
            "response_format": {"type": "json_object"},
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

        try:
            payload = send(body)
        except urllib.error.HTTPError as exc:
            if int(getattr(exc, "code", 0)) != 400:
                raise
            fallback = dict(body)
            fallback.pop("response_format", None)
            payload = send(fallback)

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
        if text.startswith("\`\`\`"):
            lines = text.splitlines()
            if lines and lines[0].startswith("\`\`\`"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("\`\`\`"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)
