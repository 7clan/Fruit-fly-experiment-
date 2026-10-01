"""Cheap preflight for the Gemini semantic-coach API.

Runs before the expensive canonical brain initializes.  It verifies only:
- GEMINI_API_KEY is present,
- the selected model is available to this project,
- generateContent JSON mode works.

It never sends a game screenshot.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


DEFAULT_MODEL = "gemini-3.1-flash-lite"


def probe(model: str | None = None, timeout_s: float = 10.0) -> tuple[bool, str]:
    key = str(os.getenv("GEMINI_API_KEY") or "").strip()
    model = str(model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
    if not key:
        return False, "GEMINI_API_KEY is missing"

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent")
    body = {
        "contents": [{
            "role": "user",
            "parts": [{"text": 'Return only this JSON: {"ok": true}'}],
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 24,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        parts = (((payload.get("candidates") or [{}])[0]
                  .get("content") or {}).get("parts") or [])
        txt = "".join(str(p.get("text") or "") for p in parts)
        parsed = json.loads(txt)
        if bool(parsed.get("ok")):
            return True, f"model={model}"
        return False, f"unexpected model response from {model}"
    except urllib.error.HTTPError as exc:
        # Do not print request headers/key.
        try:
            detail = exc.read().decode("utf-8")[:500]
        except Exception:
            detail = ""
        return False, f"HTTP {exc.code} model={model}: {detail}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("DigitalFlyLab semantic-coach preflight")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    ok, detail = probe(args.model)
    print(f"[coach-probe] {'OK' if ok else 'FAILED'} {detail}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
