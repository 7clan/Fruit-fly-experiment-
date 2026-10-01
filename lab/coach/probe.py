"""Cheap Gemini semantic-coach preflight.

The preflight deliberately runs before the expensive canonical fly brain.
It discovers which Flash-Lite models this exact API key/project can use,
selects a compatible one, then sends a tiny generateContent request.

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
)


def _http_json(url: str, key: str, *, data: dict | None = None,
               timeout_s: float = 10.0) -> dict:
    raw = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        return json.loads(resp.read().decode("utf-8"))


def available_generate_models(key: str, timeout_s: float = 10.0) -> set[str]:
    payload = _http_json(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        key, timeout_s=timeout_s)
    out = set()
    for rec in payload.get("models") or []:
        methods = {str(x).lower() for x in
                   (rec.get("supportedGenerationMethods") or [])}
        if "generatecontent" not in methods:
            continue
        name = str(rec.get("name") or "")
        if name.startswith("models/"):
            name = name[7:]
        if name:
            out.add(name)
        base = str(rec.get("baseModelId") or "")
        if base:
            out.add(base)
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
        if model in available:
            return model
    # Last-resort: prefer any available text multimodal Flash-Lite model.
    compatible = sorted(
        m for m in available
        if m.startswith("gemini-") and "flash-lite" in m
        and "image" not in m and "tts" not in m and "live" not in m)
    return compatible[-1] if compatible else None


def _probe_generate(model: str, key: str, timeout_s: float) -> bool:
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent")
    # Keep this intentionally minimal. Structured-output compatibility is
    # exercised by the runtime; this test only verifies key/model access.
    payload = _http_json(
        endpoint, key,
        data={
            "contents": [{
                "role": "user",
                "parts": [{"text": "Reply with exactly: OK"}],
            }],
        },
        timeout_s=timeout_s)
    parts = (((payload.get("candidates") or [{}])[0]
              .get("content") or {}).get("parts") or [])
    text = "".join(str(p.get("text") or "") for p in parts).strip()
    return bool(text)


def probe(model: str | None = None,
          timeout_s: float = 10.0) -> tuple[bool, str, str | None]:
    key = str(os.getenv("GEMINI_API_KEY") or "").strip()
    requested = str(
        model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
    if not key:
        return False, "GEMINI_API_KEY is missing", None

    try:
        available = available_generate_models(key, timeout_s=timeout_s)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")[:700]
        except Exception:
            detail = ""
        return (
            False,
            f"models.list HTTP {exc.code}: {detail}",
            None,
        )
    except Exception as exc:
        return False, f"models.list {type(exc).__name__}: {exc}", None

    selected = choose_model(requested, available)
    if not selected:
        flash_lite = sorted(
            m for m in available if "flash-lite" in m.lower())
        return (
            False,
            "no compatible Flash-Lite generateContent model is available "
            f"to this API project; visible Flash-Lite models={flash_lite[:12]}",
            None,
        )

    try:
        if not _probe_generate(selected, key, timeout_s):
            return False, f"empty response from model={selected}", selected
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")[:700]
        except Exception:
            detail = ""
        return (
            False,
            f"HTTP {exc.code} model={selected}: {detail}",
            selected,
        )
    except Exception as exc:
        return (
            False,
            f"{type(exc).__name__} model={selected}: {exc}",
            selected,
        )

    note = ""
    if selected != requested:
        note = f" fallback_from={requested}"
    return True, f"model={selected}{note}", selected


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("DigitalFlyLab semantic-coach preflight")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    ok, detail, selected = probe(args.model)
    print(f"[coach-probe] {'OK' if ok else 'FAILED'} {detail}")
    if ok and selected:
        # Stable machine-readable line for the PowerShell launcher.
        print(f"COACH_MODEL={selected}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
