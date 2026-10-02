"""Preflight for Meta Llama API.

Discovers models exposed to this LLAMA_API_KEY and verifies one small
OpenAI-compatible chat request before the expensive canonical fly brain starts.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


DEFAULT_BASE = "https://api.llama.com/compat/v1"
PREFERRED = (
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Llama-4-Scout-17B-16E-Instruct-FP8",
    "Llama-3.3-70B-Instruct",
)


def _request(
        url: str, key: str, *, data: dict | None = None,
        timeout_s: float = 15.0) -> dict:
    req = urllib.request.Request(
        url,
        data=(
            None if data is None
            else json.dumps(data).encode("utf-8")
        ),
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
            "utf-8", errors="replace")[:1000].replace("\n", " ")
    except Exception:
        return ""


def list_models(
        key: str, base_url: str, timeout_s: float) -> set[str]:
    payload = _request(
        base_url.rstrip("/") + "/models",
        key,
        timeout_s=timeout_s,
    )
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

    # Last-resort Llama 4 model if the account exposes a differently named
    # hosted variant.
    llama4 = sorted(
        x for x in available if "llama-4" in x.lower())
    if llama4:
        return llama4[0]
    instruct = sorted(
        x for x in available if "llama" in x.lower()
        and "instruct" in x.lower())
    return instruct[0] if instruct else None


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
                "content": (
                    'Return only this JSON object: '
                    '{"ok":true}'
                ),
            }],
            "temperature": 0.0,
            "max_completion_tokens": 24,
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
        timeout_s: float = 15.0,
) -> tuple[bool, str, str | None]:
    key = str(os.getenv("LLAMA_API_KEY") or "").strip()
    if not key:
        return False, "LLAMA_API_KEY is missing", None

    available: set[str] = set()
    list_note = ""
    try:
        available = list_models(key, base_url, timeout_s)
    except urllib.error.HTTPError as exc:
        list_note = (
            f" models_list_http={exc.code}:"
            f"{_error_detail(exc)[:300]}")
    except Exception as exc:
        list_note = f" models_list={type(exc).__name__}:{exc}"

    selected = choose_model(
        requested or os.getenv("LLAMA_MODEL"), available)
    if not selected:
        return (
            False,
            f"no usable Llama model discovered; "
            f"available={sorted(available)[:20]}{list_note}",
            None,
        )

    attempts = [selected]
    for model in PREFERRED:
        if model not in attempts and (
                not available or model in available):
            attempts.append(model)

    errors = []
    for model in attempts:
        try:
            text = _probe_chat(
                key, base_url, model, timeout_s)
            if text:
                fallback = (
                    f" fallback_from={requested}"
                    if requested and requested not in {"", "auto"}
                    and requested != model else "")
                return (
                    True,
                    f"model={model}{fallback} "
                    f"base={base_url}{list_note}",
                    model,
                )
            errors.append(f"{model}:empty_response")
        except urllib.error.HTTPError as exc:
            errors.append(
                f"{model}:HTTP{exc.code}:"
                f"{_error_detail(exc)[:400]}")
        except Exception as exc:
            errors.append(
                f"{model}:{type(exc).__name__}:{exc}")

    return (
        False,
        "no Llama API model succeeded. Attempts: "
        + " | ".join(errors)[:1800],
        None,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("DigitalFlyLab Llama API preflight")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args(argv)

    ok, detail, selected = probe(
        args.model, args.base_url, args.timeout)
    print(f"[llama-probe] {'OK' if ok else 'FAILED'} {detail}")
    if ok and selected:
        print(f"COACH_MODEL={selected}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
