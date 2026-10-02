"""Preflight for current Meta Model API (Muse Spark)."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


DEFAULT_BASE = "https://api.meta.ai/v1"
PREFERRED = (
    "muse-spark-1.3",
    "muse-spark-1.2",
    "muse-spark-1.1",
)


def _request(
        url: str, key: str, *, data: dict | None = None,
        timeout_s: float = 20.0) -> dict:
    req = urllib.request.Request(
        url,
        data=None if data is None else json.dumps(data).encode("utf-8"),
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


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode(
            "utf-8", errors="replace")[:1200].replace("\n", " ")
    except Exception:
        return ""


def list_models(key: str, base_url: str, timeout_s: float) -> set[str]:
    payload = _request(
        base_url.rstrip("/") + "/models",
        key,
        timeout_s=timeout_s)
    return {
        str(x.get("id") or "").strip()
        for x in (payload.get("data") or [])
        if x.get("id")
    }


def choose_model(
        requested: str | None, available: set[str]) -> str | None:
    requested = str(requested or "").strip()
    candidates = []
    if requested and requested.lower() != "auto":
        candidates.append(requested)
    candidates.extend(PREFERRED)
    for model in candidates:
        if not available or model in available:
            return model
    muse = sorted(x for x in available if x.startswith("muse-spark-"))
    return muse[-1] if muse else None


def _probe_chat(
        key: str, base_url: str, model: str,
        timeout_s: float) -> str:
    payload = _request(
        base_url.rstrip("/") + "/chat/completions",
        key,
        data={
            "model": model,
            "messages": [{
                "role": "user",
                "content": 'Return only this JSON object: {"ok":true}',
            }],
            "reasoning_effort": "none",
            "max_completion_tokens": 32,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "probe",
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        timeout_s=timeout_s,
    )
    choices = payload.get("choices") or []
    return str(
        (((choices[0] if choices else {}).get("message") or {})
         .get("content")) or ""
    ).strip()


def probe(
        requested: str | None = None,
        base_url: str = DEFAULT_BASE,
        timeout_s: float = 20.0,
) -> tuple[bool, str, str | None]:
    key = str(os.getenv("MODEL_API_KEY") or "").strip()
    if not key:
        return False, "MODEL_API_KEY is missing", None

    try:
        available = list_models(key, base_url, timeout_s)
    except urllib.error.HTTPError as exc:
        return (
            False,
            f"models HTTP {exc.code}: {_error_detail(exc)[:900]}",
            None,
        )
    except Exception as exc:
        return False, f"models {type(exc).__name__}: {exc}", None

    selected = choose_model(
        requested or os.getenv("META_MODEL"), available)
    if not selected:
        return (
            False,
            f"no Muse Spark model available; "
            f"models={sorted(available)[:30]}",
            None,
        )

    try:
        text = _probe_chat(key, base_url, selected, timeout_s)
        if not text:
            return False, f"empty response from {selected}", selected
        return (
            True,
            f"model={selected} base={base_url}",
            selected,
        )
    except urllib.error.HTTPError as exc:
        return (
            False,
            f"HTTP {exc.code} model={selected}: "
            f"{_error_detail(exc)[:1000]}",
            selected,
        )
    except Exception as exc:
        return (
            False,
            f"{type(exc).__name__} model={selected}: {exc}",
            selected,
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("DigitalFlyLab Meta Model API preflight")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args(argv)

    ok, detail, selected = probe(
        args.model, args.base_url, args.timeout)
    print(f"[meta-probe] {'OK' if ok else 'FAILED'} {detail}")
    if ok and selected:
        print(f"COACH_MODEL={selected}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
