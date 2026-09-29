"""JSON-lines server for the canonical brain subprocess.

This process runs under brain/.venv so Brian2/Cython stay isolated from the
Windows capture/dashboard process. stdin/stdout are a tiny RPC transport.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from .runtime import CanonicalBrianRuntime


def _reply(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-ms", type=float, default=50.0)
    ap.add_argument("--codegen", choices=["numpy", "cython"], default="cython")
    args = ap.parse_args(argv)

    runtime = CanonicalBrianRuntime(
        chunk_ms=args.chunk_ms,
        codegen_target=args.codegen,
        quiet=True,
    )

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            op = req.get("op")
            if op == "init":
                runtime.warm_import()
                runtime.init_once()
                _reply({"ok": True, "result": {"pop_sizes": runtime.pop_sizes()}})
            elif op == "prewarm":
                runtime.prewarm(int(req.get("chunks", 2)))
                _reply({"ok": True, "result": {}})
            elif op == "advance":
                rec = runtime.advance_chunk(
                    dict(req.get("rates") or {}),
                    chunk_ms=float(req.get("chunk_ms", args.chunk_ms)),
                )
                _reply({"ok": True, "result": dict(rec)})
            elif op == "pop_sizes":
                _reply({"ok": True, "result": runtime.pop_sizes()})
            elif op == "close":
                runtime.close()
                _reply({"ok": True, "result": {}})
                return 0
            else:
                raise ValueError(f"unknown brain subprocess op {op!r}")
        except Exception as exc:
            _reply({
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=20),
            })
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
