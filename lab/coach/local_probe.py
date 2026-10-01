"""Preflight for the local SmolVLM llama.cpp server."""

from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request


DEFAULT_URL = "http://127.0.0.1:18080/v1"
DEFAULT_MODEL = "smolvlm2-256m"


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


def _tiny_jpeg_b64() -> str:
    import cv2
    import numpy as np
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[12:36, 12:36] = 255
    ok, enc = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("could not encode local probe image")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def probe(base_url: str = DEFAULT_URL,
          model: str = DEFAULT_MODEL,
          timeout_s: float = 180.0) -> tuple[bool, str]:
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

    try:
        image_b64 = _tiny_jpeg_b64()
        payload = _post(
            base_url + "/chat/completions",
            {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look at the image. Reply only: OK"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64,"
                                    + image_b64
                                ),
                            },
                        },
                    ],
                }],
                "temperature": 0.0,
                "max_tokens": 4,
            },
            timeout_s,
        )
        choices = payload.get("choices") or []
        text = str(
            (((choices[0] if choices else {}).get("message") or {})
             .get("content")) or ""
        ).strip()
        if not text:
            return False, f"vision probe returned no text: {str(payload)[:500]}"
        return True, f"model={model} local_url={base_url}"
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
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args(argv)
    ok, detail = probe(args.url, args.model, args.timeout)
    print(f"[local-coach-probe] {'OK' if ok else 'FAILED'} {detail}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
