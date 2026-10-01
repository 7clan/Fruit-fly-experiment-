"""Cheap Gemini semantic-coach preflight.

Uses Gemini's documented OpenAI-compatible REST endpoint with
Authorization: Bearer.  This is important for current AI Studio auth keys:
new AI Studio keys are authorization keys, and the OpenAI-compatible surface
provides a clean, documented bearer-auth path.

No game screenshot is sent.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


DEFAULT_MODEL = "gemini-3.5-flash-lite"
PREFERRED_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.8-flash",
)


def _request(url: str, key: str, *, data: dict | None = None,
             timeout_s: float = 10.0) -> dict:
    raw = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "DigitalFlyLab/1.0",
        },
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        return json.loads(resp.read().decode("utf-8"))


def openai_models(key: str, timeout_s: float = 10.0) -> set[str]:
    payload = _request(
        "https://generativelanguage.googleapis.com/v1beta/openai/models",
        key, timeout_s=timeout_s)
    out = set()
    for rec in payload.get("data") or []:
        mid = str(rec.get("id") or "").strip()
        if mid:
            out.add(mid)
    return out


def choose_model(requested: str | None, available: set[str]) -> str | None:
    requested = str(requested or "").strip()
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend(PREFERRED_MODELS)
    seen = set()
    for model in candidates:
        if model in seen:
            continue
        seen.add(model)
        if not available or model in available:
            return model
    compatible = sorted(
        m for m in available
        if m.startswith("gemini-") and "flash-lite" in m
        and "image" not in m and "tts" not in m and "live" not in m)
    return compatible[-1] if compatible else None


def _probe_chat(model: str, key: str, timeout_s: float) -> bool:
    payload = _request(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        key,
        data={
            "model": model,
            "messages": [
                {"role": "user", "content": "Reply with exactly: OK"}
            ],
            "max_tokens": 16,
        },
        timeout_s=timeout_s,
    )
    choices = payload.get("choices") or []
    if not choices:
        return False
    content = (((choices[0].get("message") or {}).get("content")) or "")
    return bool(str(content).strip())


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")[:1200]
    except Exception:
        raw = ""
    return raw.replace("\r", " ").replace("\n", " ").strip()


def probe(model: str | None = None,
          timeout_s: float = 10.0) -> tuple[bool, str, str | None]:
    key = str(os.getenv("GEMINI_API_KEY") or "").strip()
    requested = str(
        model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
    if not key:
        return False, "GEMINI_API_KEY is missing", None

    # Model listing is helpful but not required. If listing is unavailable,
    # directly probe the requested/preferred models.
    available: set[str] = set()
    list_note = ""
    try:
        available = openai_models(key, timeout_s=timeout_s)
    except urllib.error.HTTPError as exc:
        list_note = f" models_list_http={exc.code}"
    except Exception as exc:
        list_note = f" models_list={type(exc).__name__}"

    candidates = []
    if requested:
        candidates.append(requested)
    if available:
        chosen = choose_model(requested, available)
        if chosen:
            candidates.append(chosen)
    candidates.extend(PREFERRED_MODELS)

    seen = set()
    errors = []
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if available and candidate not in available:
            continue
        try:
            if _probe_chat(candidate, key, timeout_s):
                note = ""
                if candidate != requested:
                    note = f" fallback_from={requested}"
                return (
                    True,
                    f"model={candidate}{note} transport=openai_compat{list_note}",
                    candidate,
                )
            errors.append(f"{candidate}:empty_response")
        except urllib.error.HTTPError as exc:
            errors.append(
                f"{candidate}:HTTP{exc.code}:{_error_detail(exc)[:350]}")
        except Exception as exc:
            errors.append(f"{candidate}:{type(exc).__name__}:{exc}")

    # Useful diagnostic without exposing the key.
    key_kind = (
        "auth_key" if key.startswith("AQ.")
        else "standard_or_unknown_key")
    return (
        False,
        f"no Gemini model succeeded via OpenAI-compatible Bearer auth "
        f"(key_type={key_kind}). Attempts: {' | '.join(errors)[:1800]}",
        None,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("DigitalFlyLab semantic-coach preflight")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    ok, detail, selected = probe(args.model)
    print(f"[coach-probe] {'OK' if ok else 'FAILED'} {detail}")
    if ok and selected:
        print(f"COACH_MODEL={selected}")
        print("COACH_TRANSPORT=openai_compat")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
