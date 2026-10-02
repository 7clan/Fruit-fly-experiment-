"""Preflight for Ollama Cloud semantic coaching.

Uses Ollama's documented OpenAI-compatible endpoint at https://ollama.com/v1.
No model-list request is required; candidates are probed directly so a flaky
/tags response cannot block the fly before Brian2 starts.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


DEFAULT_MODEL = "gemma4:cloud"
DEFAULT_BASE = "https://ollama.com/v1"
# AI-only needs a model that can actually see the current Roblox frame.
# Keep the official Ollama Cloud multimodal alias first. Text-only fallbacks
# remain useful for the hybrid coach, but AI-only runs pass --require-vision.
PREFERRED_MODELS = (
    "gemma4:cloud",
    "gemma4:31b-cloud",
    "gemma4:31b",
    "deepseek-v4.1-flash",
    "gpt-oss:120b",
    "gpt-oss:20b",
    "nemotron-3-nano:30b",
)

# 64x64 solid-red PNG. A 1x1 probe can trip vision encoders even when the
# transport is correct, so use a normal-size image and require the model to
# identify its color. This validates real image understanding, not merely
# acceptance of a multimodal request.
_PROBE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAX0lEQVR4nO3P"
    "QQ0AIBDAMMC/50MEj4ZkVbDtWX87OuBVA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNa"
    "A1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA9oFUoUBf3Xr7AgA"
    "AAAASUVORK5CYII="
)


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return (
            exc.read().decode("utf-8", errors="replace")[:1400]
            .replace("\n", " ").replace("\r", " ")
        )
    except Exception:
        return ""


def _post(base_url: str, key: str, body: dict,
          timeout_s: float) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "DigitalFlyLab/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize_model(model: str) -> str:
    # Do NOT strip :cloud / -cloud. Those aliases select Ollama's hosted
    # multimodal variants; stripping them previously turned a vision-capable
    # request into a different text/degraded endpoint.
    return str(model or "").strip()


def _candidate_models(requested: str) -> list[str]:
    out = []
    for name in (_normalize_model(requested),) + PREFERRED_MODELS:
        name = _normalize_model(name)
        if name and name not in out:
            out.append(name)
    return out


def _extract_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    return str(
        (((choices[0] if choices else {}).get("message") or {})
         .get("content")) or ""
    ).strip()


def _probe_text(base_url: str, key: str, model: str,
                timeout_s: float) -> None:
    payload = _post(
        base_url, key,
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply only: OK"}],
            "temperature": 0.0,
            "max_tokens": 8,
        },
        timeout_s,
    )
    if not _extract_text(payload):
        raise RuntimeError("empty text response")


def _probe_vision(base_url: str, key: str, model: str,
                  timeout_s: float) -> tuple[bool, str]:
    # Ollama's documented OpenAI-compatibility schema expects image_url to be
    # the data-URI STRING itself, not the nested {"url": ...} object accepted
    # by OpenAI's own API. The nested form caused HTTP 500 on Ollama Cloud.
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": (
                        "data:image/png;base64," + _PROBE_PNG_B64
                    ),
                },
                {
                    "type": "text",
                    "text": (
                        "What is the dominant color of the square? "
                        "Reply with one color word only."
                    ),
                },
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 12,
    }
    try:
        payload = _post(base_url, key, body, timeout_s)
        answer = _extract_text(payload).strip().lower()
        ok = "red" in answer
        return ok, (
            "vision=yes verified=red_square"
            if ok else f"vision=unverified answer={answer[:60]!r}"
        )
    except urllib.error.HTTPError as exc:
        if int(exc.code) in {400, 500, 502, 503, 504}:
            return False, (
                f"vision=degraded_http_{exc.code} "
                "fallback=structured_state"
            )
        raise


def probe(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE,
          timeout_s: float = 30.0,
          require_vision: bool = False) -> tuple[bool, str, str | None]:
    key = str(os.getenv("OLLAMA_API_KEY") or "").strip()
    if not key:
        return False, "OLLAMA_API_KEY is missing", None

    errors = []
    for candidate in _candidate_models(model):
        try:
            _probe_text(base_url, key, candidate, timeout_s)
            vision_ok, vision_note = _probe_vision(
                base_url, key, candidate, timeout_s)
            if require_vision and not vision_ok:
                errors.append(f"{candidate}:{vision_note}")
                continue
            fallback = (
                "" if candidate == _normalize_model(model)
                else f" fallback_from={model}")
            return (
                True,
                f"model={candidate} base={base_url} "
                f"text=yes {vision_note}{fallback}",
                candidate,
            )
        except urllib.error.HTTPError as exc:
            errors.append(
                f"{candidate}:HTTP{exc.code}:{_error_detail(exc)[:320]}")
        except Exception as exc:
            errors.append(
                f"{candidate}:{type(exc).__name__}:{exc}")

    return (
        False,
        "no Ollama Cloud coach model succeeded via OpenAI-compatible API. "
        f"attempts={' | '.join(errors)[:1800]}",
        None,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("DigitalFlyLab Ollama Cloud preflight")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument(
        "--require-vision", action="store_true",
        help="fail over until a hosted model accepts image input")
    args = ap.parse_args(argv)
    ok, detail, selected = probe(
        args.model, args.base_url, args.timeout,
        require_vision=bool(args.require_vision))
    print(f"[ollama-cloud-probe] {'OK' if ok else 'FAILED'} {detail}")
    if ok and selected:
        print(f"COACH_MODEL={selected}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
