"""Cheap native Gemini semantic-coach preflight.

Uses Google's direct Gemini generateContent REST API with x-goog-api-key.
Google recommends the native API when OpenAI compatibility is not required.
This also avoids the compatibility-layer HTTP 400 failures seen on the target
Windows machine.

No game screenshot is sent except a 1x1 PNG used to verify image input.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_MODEL = "gemini-3.5-flash-lite"
PREFERRED_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.8-flash",
)

TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
    "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def sanitize_api_key(value: str | None) -> str:
    """Normalize common copy/paste forms without ever logging the secret."""
    key = str(value or "").strip()
    # Users often copy KEY=value or a quoted value from setup instructions.
    for prefix in ("GEMINI_API_KEY=", "GOOGLE_API_KEY=", "x-goog-api-key:"):
        if key.lower().startswith(prefix.lower()):
            key = key[len(prefix):].strip()
            break
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1].strip()
    return key.replace("\r", "").replace("\n", "").strip()


def _query_key_url(url: str, key: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query = [(k, v) for k, v in query if k.lower() != "key"]
    query.append(("key", key))
    return urllib.parse.urlunsplit((
        parts.scheme, parts.netloc, parts.path,
        urllib.parse.urlencode(query), parts.fragment))


def _request(url: str, key: str, *, data: dict | None = None,
             timeout_s: float = 10.0) -> dict:
    """Send the exact documented REST shape.

    The target Windows machine repeatedly received an HTML HTTP 400 before
    Gemini parsed the request when the API key was sent as a custom header.
    Google's own multimodal shell example also documents ?key= authentication,
    so retry that transport only for this pre-routing style failure.
    """
    key = sanitize_api_key(key)
    raw = None if data is None else json.dumps(data).encode("utf-8")
    method = "POST" if data is not None else "GET"

    def send(target_url: str, use_header: bool):
        headers = {"Content-Type": "application/json"}
        if use_header:
            headers["x-goog-api-key"] = key
        req = urllib.request.Request(
            target_url, data=raw, headers=headers, method=method)
        with urllib.request.urlopen(
                req, timeout=float(timeout_s)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        return send(url, True)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        # HTML 400 means the request died before Gemini's normal JSON error
        # surface. Retry Google's documented query-key REST form.
        if int(getattr(exc, "code", 0)) != 400 or "<html" not in body.lower():
            raise
    return send(_query_key_url(url, key), False)


def native_models(key: str, timeout_s: float = 10.0) -> set[str]:
    payload = _request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        key, timeout_s=timeout_s)
    out = set()
    for rec in payload.get("models") or []:
        methods = {
            str(x).lower()
            for x in (rec.get("supportedGenerationMethods") or [])
        }
        if "generatecontent" not in methods:
            continue
        name = str(rec.get("name") or "").strip()
        if name.startswith("models/"):
            name = name[7:]
        if name:
            out.add(name)
        base = str(rec.get("baseModelId") or "").strip()
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
        if not available or model in available:
            return model
    compatible = sorted(
        m for m in available
        if m.startswith("gemini-")
        and ("flash-lite" in m or m == "gemini-3.8-flash")
        and "image" not in m
        and "tts" not in m
        and "live" not in m)
    return compatible[-1] if compatible else None


def _extract_text(payload: dict) -> str:
    parts = (((payload.get("candidates") or [{}])[0]
              .get("content") or {}).get("parts") or [])
    return "".join(
        str(p.get("text") or "") for p in parts if isinstance(p, dict)
    ).strip()


def _generate(model: str, key: str, parts: list[dict],
              timeout_s: float) -> dict:
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent")
    return _request(
        endpoint, key,
        data={
            "contents": [{
                "role": "user",
                "parts": parts,
            }],
            "generationConfig": {
                "maxOutputTokens": 24,
            },
        },
        timeout_s=timeout_s)


def _probe_chat(model: str, key: str, timeout_s: float) -> bool:
    payload = _generate(
        model, key,
        [{"text": "Reply with exactly: OK"}],
        timeout_s)
    return bool(_extract_text(payload))


def _probe_vision(model: str, key: str, timeout_s: float) -> bool:
    payload = _generate(
        model, key,
        [
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": TINY_PNG_B64,
                },
            },
            {"text": "Reply with exactly: OK"},
        ],
        timeout_s)
    return bool(_extract_text(payload))


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")[:1200]
    except Exception:
        raw = ""
    return raw.replace("\r", " ").replace("\n", " ").strip()


def probe(model: str | None = None,
          timeout_s: float = 10.0,
          require_vision: bool = False,
          prefer_fastest_vision: bool = False) -> tuple[bool, str, str | None]:
    key = sanitize_api_key(os.getenv("GEMINI_API_KEY"))
    requested = str(
        model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
    if not key:
        return False, "GEMINI_API_KEY is missing", None

    available: set[str] = set()
    list_note = ""
    try:
        available = native_models(key, timeout_s=timeout_s)
    except urllib.error.HTTPError as exc:
        list_note = (
            f" models_list_http={exc.code}:"
            f"{_error_detail(exc)[:220]}")
    except Exception as exc:
        list_note = f" models_list={type(exc).__name__}:{exc}"

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

    if require_vision and prefer_fastest_vision:
        timed = []
        for candidate in candidates:
            candidate = str(candidate or "").strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if available and candidate not in available:
                continue
            try:
                t0 = time.perf_counter()
                if not _probe_vision(candidate, key, timeout_s):
                    errors.append(f"{candidate}:vision_empty_response")
                    continue
                ms = (time.perf_counter() - t0) * 1000.0
                timed.append((ms, candidate))
            except urllib.error.HTTPError as exc:
                errors.append(
                    f"{candidate}:vision_HTTP{exc.code}:"
                    f"{_error_detail(exc)[:320]}")
            except Exception as exc:
                errors.append(
                    f"{candidate}:vision_{type(exc).__name__}:{exc}")
        if timed:
            ms, selected = min(timed, key=lambda x: x[0])
            note = (
                "" if selected == requested
                else f" fallback_from={requested}")
            tested = ",".join(
                f"{m}={round(t,1)}ms" for t, m in sorted(timed))
            return (
                True,
                f"model={selected}{note} transport=native_generate_content "
                f"vision=yes startup_vision_ms={round(ms,1)} "
                f"bench=[{tested}]{list_note}",
                selected,
            )
        seen = set()

    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if available and candidate not in available:
            continue
        try:
            if not _probe_chat(candidate, key, timeout_s):
                errors.append(f"{candidate}:empty_response")
                continue
            vision_note = ""
            if require_vision:
                if not _probe_vision(candidate, key, timeout_s):
                    errors.append(f"{candidate}:vision_empty_response")
                    continue
                vision_note = " vision=yes"
            note = ""
            if candidate != requested:
                note = f" fallback_from={requested}"
            return (
                True,
                f"model={candidate}{note} "
                f"transport=native_generate_content"
                f"{vision_note}{list_note}",
                candidate,
            )
        except urllib.error.HTTPError as exc:
            errors.append(
                f"{candidate}:HTTP{exc.code}:{_error_detail(exc)[:380]}")
        except Exception as exc:
            errors.append(f"{candidate}:{type(exc).__name__}:{exc}")

    key_kind = (
        "google_api_key" if key.startswith("AIza")
        else "auth_key" if key.startswith("AQ.")
        else f"unknown_prefix_len_{len(key)}")
    return (
        False,
        "no Gemini model succeeded via native generateContent "
        f"(key_type={key_kind}). Attempts: {' | '.join(errors)[:2200]}"
        f"{list_note}",
        None,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("DigitalFlyLab Gemini native preflight")
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout", type=float, default=12.0)
    ap.add_argument(
        "--require-vision", action="store_true",
        help="only accept a model that successfully processes image input")
    ap.add_argument(
        "--fastest-vision", action="store_true",
        help="benchmark accessible Flash-Lite models and select the fastest")
    args = ap.parse_args(argv)
    ok, detail, selected = probe(
        args.model,
        timeout_s=args.timeout,
        require_vision=bool(args.require_vision),
        prefer_fastest_vision=bool(args.fastest_vision))
    print(f"[coach-probe] {'OK' if ok else 'FAILED'} {detail}")
    if ok and selected:
        print(f"COACH_MODEL={selected}")
        print("COACH_TRANSPORT=native_generate_content")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
