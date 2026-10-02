"""Preflight for Ollama Cloud Qwen3-VL."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "qwen3-vl:235b-cloud"
DEFAULT_BASE = "https://ollama.com/api"

# 1x1 PNG, only to verify that the selected cloud model accepts image input.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
    "AAAAC0lEQVR42mP8/x8AAusB9Y9Z4N8AAAAASUVORK5CYII="
)

def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:1200].replace("\n", " ")
    except Exception:
        return ""

def probe(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE,
          timeout_s: float = 30.0) -> tuple[bool, str]:
    key = str(os.getenv("OLLAMA_API_KEY") or "").strip()
    if not key:
        return False, "OLLAMA_API_KEY is missing"
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": "Look at this tiny image and return JSON {\"ok\":true}.",
            "images": [_TINY_PNG_B64],
        }],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
        "options": {"temperature": 0.0, "num_predict": 24},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "DigitalFlyLab/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = str(((payload.get("message") or {}).get("content")) or "").strip()
        parsed = json.loads(text)
        if bool(parsed.get("ok")):
            return True, f"model={model} base={base_url} vision=yes"
        return False, f"unexpected response: {text[:300]}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {_error_detail(exc)}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

def main(argv=None) -> int:
    ap = argparse.ArgumentParser("DigitalFlyLab Ollama Cloud preflight")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)
    ok, detail = probe(args.model, args.base_url, args.timeout)
    print(f"[ollama-cloud-probe] {'OK' if ok else 'FAILED'} {detail}")
    if ok:
        print(f"COACH_MODEL={args.model}")
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())