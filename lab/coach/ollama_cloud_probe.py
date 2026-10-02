"""Preflight for Ollama Cloud multimodal coaching."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


DEFAULT_MODEL = "gemma4:31b"
DEFAULT_BASE = "https://ollama.com/api"
PREFERRED_MODELS = (
    "gemma4:31b",
    "gemma4",
    "gpt-oss:120b",
    "gpt-oss:20b",
    "nemotron-3-nano:30b",
    "nemotron-3-super",
    "nemotron-3-ultra",
)

# 1x1 PNG only verifies that the selected hosted model accepts image input.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
    "AAAAC0lEQVR42mP8/x8AAusB9Y9Z4N8AAAAASUVORK5CYII="
)


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return (
            exc.read().decode("utf-8", errors="replace")[:1400]
            .replace("\n", " ").replace("\r", " ")
        )
    except Exception:
        return ""


def _request(url: str, key: str, *, data: dict | None = None,
             timeout_s: float = 20.0) -> dict:
    req = urllib.request.Request(
        url,
        data=(None if data is None else json.dumps(data).encode("utf-8")),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "DigitalFlyLab/1.0",
        },
        method=("GET" if data is None else "POST"),
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize_direct_model(model: str) -> str:
    model = str(model or "").strip()
    # :cloud is the local-Ollama routing alias. Direct ollama.com API uses
    # the hosted model name without the cloud suffix.
    if model.endswith("-cloud"):
        model = model[:-6]
    if model.endswith(":cloud"):
        model = model[:-6]
    # Old experiments used Qwen hosted models that are not in this account's
    # free included usage. Prefer the current free multimodal Gemma model.
    if model in {
        "qwen3-vl:235b",
        "qwen3-vl:235b-instruct",
        "qwen3-vl:235b-a22b",
        "qwen3-vl:235b-a22b-instruct",
        "qwen3.5",
        "qwen3.6",
        "qwen3.8",
        "kimi-k3",
    }:
        return "gemma4:31b"
    return model


def _list_models(base_url: str, key: str,
                 timeout_s: float) -> list[str]:
    payload = _request(
        base_url.rstrip("/") + "/tags",
        key, timeout_s=timeout_s)
    names = []
    for rec in payload.get("models") or []:
        name = str(rec.get("model") or rec.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _candidate_models(requested: str, available: list[str]) -> list[str]:
    requested = _normalize_direct_model(requested)
    out = []
    for name in (requested,) + PREFERRED_MODELS:
        name = _normalize_direct_model(name)
        if not name or name in out:
            continue
        if available and name not in available:
            # Ollama tags may include explicit sizes. qwen3.5 can be exposed
            # as qwen3.5 or a concrete qwen3.5:* hosted variant.
            matches = [x for x in available if x == name or x.startswith(name + ":")]
            if matches:
                for x in matches:
                    if x not in out:
                        out.append(x)
            continue
        out.append(name)
    if not out and available:
        # Last resort: prefer any clearly multimodal/current family.
        for prefix in ("gemma4:31b", "gemma4", "gpt-oss:120b",
                       "gpt-oss:20b", "nemotron-3-nano:30b",
                       "nemotron-3-super", "nemotron-3-ultra"):
            for x in available:
                if x == prefix or x.startswith(prefix + ":"):
                    if x not in out:
                        out.append(x)
    return out


def _probe_chat(base_url: str, key: str, model: str,
                timeout_s: float) -> tuple[bool, str]:
    endpoint = base_url.rstrip("/") + "/chat"

    # First verify the account can actually use the model. This is the
    # critical preflight. A vision backend hiccup should not waste the whole
    # Brian2 startup because structured CV state is sufficient as fallback.
    text_body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": "Reply only with: OK",
        }],
        "stream": False,
    }
    text_payload = _request(
        endpoint, key, data=text_body, timeout_s=timeout_s)
    text_reply = str(
        ((text_payload.get("message") or {}).get("content")) or ""
    ).strip()
    if not text_reply:
        return False, f"text access returned empty response: {str(text_payload)[:400]}"

    # Then check image input. If the hosted vision path returns a transient
    # 5xx, keep the model usable in structured-state mode instead of aborting.
    vision_body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": "Look at this image and reply only: OK",
            "images": [_TINY_PNG_B64],
        }],
        "stream": False,
    }
    try:
        payload = _request(
            endpoint, key, data=vision_body, timeout_s=timeout_s)
        vision_reply = str(
            ((payload.get("message") or {}).get("content")) or ""
        ).strip()
        if vision_reply:
            return True, "text=yes vision=yes"
        return True, "text=yes vision=empty_fallback_structured_state"
    except urllib.error.HTTPError as exc:
        detail = _error_detail(exc)[:220]
        if int(exc.code) in {400, 500, 502, 503, 504}:
            return True, (
                f"text=yes vision=degraded_http_{exc.code} "
                f"fallback=structured_state detail={detail}"
            )
        raise


def probe(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE,
          timeout_s: float = 30.0) -> tuple[bool, str, str | None]:
    key = str(os.getenv("OLLAMA_API_KEY") or "").strip()
    if not key:
        return False, "OLLAMA_API_KEY is missing", None

    available = []
    list_note = ""
    try:
        available = _list_models(base_url, key, timeout_s)
        list_note = f" listed={len(available)}"
    except urllib.error.HTTPError as exc:
        list_note = f" tags_http={exc.code}:{_error_detail(exc)[:220]}"
    except Exception as exc:
        list_note = f" tags_error={type(exc).__name__}:{exc}"

    errors = []
    for candidate in _candidate_models(model, available):
        try:
            ok, detail = _probe_chat(
                base_url, key, candidate, timeout_s)
            if ok:
                requested = _normalize_direct_model(model)
                fallback = (
                    "" if candidate == requested
                    else f" fallback_from={model}")
                return (
                    True,
                    f"model={candidate} base={base_url} {detail}"
                    f"{fallback}{list_note}",
                    candidate,
                )
            errors.append(f"{candidate}:{detail}")
        except urllib.error.HTTPError as exc:
            errors.append(
                f"{candidate}:HTTP{exc.code}:{_error_detail(exc)[:320]}")
        except Exception as exc:
            errors.append(
                f"{candidate}:{type(exc).__name__}:{exc}")

    visible = ",".join(available[:20])
    return (
        False,
        "no included hosted coach model succeeded. "
        f"available=[{visible}] attempts={' | '.join(errors)[:1600]}"
        f"{list_note}",
        None,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("DigitalFlyLab Ollama Cloud preflight")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)
    ok, detail, selected = probe(
        args.model, args.base_url, args.timeout)
    print(f"[ollama-cloud-probe] {'OK' if ok else 'FAILED'} {detail}")
    if ok and selected:
        print(f"COACH_MODEL={selected}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
