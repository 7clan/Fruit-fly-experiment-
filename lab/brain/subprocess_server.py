"""Socket RPC server for the isolated canonical brain process.

The brain process runs under brain/.venv. RPC uses a localhost-only TCP
connection created by the parent; stdout/stderr are not used as protocol
channels because scientific dependencies may write to them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import sys
import traceback

from .runtime import CanonicalBrianRuntime


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-ms", type=float, default=50.0)
    ap.add_argument("--codegen", choices=["numpy", "cython"], default="cython")
    ap.add_argument("--connect-host", required=True)
    ap.add_argument("--connect-port", type=int, required=True)
    ap.add_argument("--token", required=True)
    args = ap.parse_args(argv)

    sock = socket.create_connection(
        (args.connect_host, args.connect_port), timeout=30.0)
    sock.settimeout(None)
    reader = sock.makefile("r", encoding="utf-8", newline="\n")
    writer = sock.makefile("w", encoding="utf-8", newline="\n",
                           buffering=1)

    def reply(obj: dict) -> None:
        writer.write(json.dumps(
            obj, separators=(",", ":"), default=str) + "\n")
        writer.flush()

    reply({"hello": args.token})

    runtime = CanonicalBrianRuntime(
        chunk_ms=args.chunk_ms,
        codegen_target=args.codegen,
        quiet=True,
    )

    try:
        for raw in reader:
            raw = raw.strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)
                op = req.get("op")
                if op == "init":
                    with contextlib.redirect_stdout(sys.stderr):
                        runtime.warm_import()
                        runtime.init_once()
                        result = {"pop_sizes": runtime.pop_sizes()}
                    reply({"ok": True, "result": result})
                elif op == "prewarm":
                    with contextlib.redirect_stdout(sys.stderr):
                        runtime.prewarm(int(req.get("chunks", 2)))
                    reply({"ok": True, "result": {}})
                elif op == "advance":
                    with contextlib.redirect_stdout(sys.stderr):
                        rec = runtime.advance_chunk(
                            dict(req.get("rates") or {}),
                            chunk_ms=float(
                                req.get("chunk_ms", args.chunk_ms)),
                        )
                    reply({"ok": True, "result": dict(rec)})
                elif op == "pop_sizes":
                    with contextlib.redirect_stdout(sys.stderr):
                        result = runtime.pop_sizes()
                    reply({"ok": True, "result": result})
                elif op == "close":
                    with contextlib.redirect_stdout(sys.stderr):
                        runtime.close()
                    reply({"ok": True, "result": {}})
                    return 0
                else:
                    raise ValueError(
                        f"unknown brain subprocess op {op!r}")
            except Exception as exc:
                reply({
                    "ok": False,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=20),
                })
    finally:
        try:
            runtime.close()
        except Exception:
            pass
        for stream in (reader, writer):
            try:
                stream.close()
            except Exception:
                pass
        try:
            sock.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
