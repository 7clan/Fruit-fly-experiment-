"""Cheap preflight for the local SmolLM2-135M llama.cpp server.

The local coach is text-only. OpenCV/engineered perception owns vision, so
there is no vision tower to warm up or contend with Brian2.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


DEFAULT_URL = "http://127.0.0.1:18080/v1"
DEFAULT_MODEL = "smollm2-135m"


def _get(url: str, timeout_s: float):
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, payload: dict, timeout_s: float):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer local-no-key",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def probe(base_url: str = DEFAULT_URL,
          model: str = DEFAULT_MODEL,
          timeout_s: float = 30.0) -> tuple[bool, str]:
    base_url = str(base_url).rstrip("/")
    try:
        models = _get(base_url + "/models", timeout_s)
        ids = [
            str(x.get("id") or "")
            for x in (models.get("data") or [])
            if x.get("id")
        ]
        if ids and model not in ids:
            return False, f"loaded model mismatch: expected={model} available={ids}"
    except Exception as exc:
        return False, f"models endpoint failed: {type(exc).__name__}: {exc}"

    # Exercise the exact schema-constrained one-field decision used in play.
    try:
        t0 = time.perf_counter()
        payload = _post(
            base_url + "/chat/completions",
            {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": (
                        "GPO controller. Pick one ALLOWED skill. "
                        "STATE={\"candidate\":\"NAVIGATE_OBJECTIVE\","
                        "\"target\":\"recommended_quest_waypoint\"} "
                        "ALLOWED=NAVIGATE_OBJECTIVE|REOBSERVE"
                    ),
                }],
                "temperature": 0.0,
                "max_tokens": 24,
                "response_format": {
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "skill": {
                                "type": "string",
                                "enum": [
                                    "NAVIGATE_OBJECTIVE",
                                    "REOBSERVE",
                                ],
                            },
                        },
                        "required": ["skill"],
                        "additionalProperties": False,
                    },
                },
            },
            timeout_s,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        choices = payload.get("choices") or []
        text = str(
            (((choices[0] if choices else {}).get("message") or {})
             .get("content")) or ""
        ).strip()
        if not text:
            return False, f"text probe returned no text: {str(payload)[:500]}"
        parsed = json.loads(text)
        skill = str(parsed.get("skill") or "").upper()
        if skill not in {"NAVIGATE_OBJECTIVE", "REOBSERVE"}:
            return False, f"selector probe returned unexpected JSON: {text!r}"
        return True, (
            f"model={model} local_url={base_url} "
            f"selector_ms={elapsed_ms:.0f} text_only=yes"
        )
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
        except Exception:
            detail = ""
        return False, f"HTTP {exc.code}: {detail}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("DigitalFlyLab local coach preflight")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)
    ok, detail = probe(args.url, args.model, args.timeout)
    print(f"[local-coach-probe] {'OK' if ok else 'FAILED'} {detail}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
