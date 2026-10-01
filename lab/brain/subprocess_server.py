"""Named-pipe RPC server for the isolated canonical brain process.

On Windows this uses multiprocessing.connection with AF_PIPE, so the parent
and brain child communicate through a local Windows named pipe rather than a
TCP socket. This avoids localhost firewall/AV interference while keeping
Brian2/Cython stdout/stderr completely separate from the protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import traceback
from multiprocessing.connection import Client

from .runtime import CanonicalBrianRuntime


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-ms", type=float, default=50.0)
    ap.add_argument("--codegen", choices=["numpy", "cython"], default="cython")
    ap.add_argument("--pipe-address", required=True)
    ap.add_argument("--authkey-hex", required=True)
    args = ap.parse_args(argv)

    conn = Client(
        args.pipe_address,
        family="AF_PIPE",
        authkey=bytes.fromhex(args.authkey_hex),
    )

    runtime = CanonicalBrianRuntime(
        chunk_ms=args.chunk_ms,
        codegen_target=args.codegen,
        quiet=True,
        # Live control needs exact population/whole-brain counts, not every
        # individual spike timestamp. Avoid storing the full event stream.
        record_full_spikes=False,
        enable_checkpoints=False,
    )

    try:
        while True:
            req = conn.recv()
            try:
                op = req.get("op")
                if op == "init":
                    with contextlib.redirect_stdout(sys.stderr):
                        runtime.warm_import()
                        runtime.init_once()
                        result = {"pop_sizes": runtime.pop_sizes()}
                    conn.send({"ok": True, "result": result})
                elif op == "prewarm":
                    with contextlib.redirect_stdout(sys.stderr):
                        runtime.prewarm(int(req.get("chunks", 2)))
                    conn.send({"ok": True, "result": {}})
                elif op == "advance":
                    with contextlib.redirect_stdout(sys.stderr):
                        rec = runtime.advance_chunk(
                            dict(req.get("rates") or {}),
                            chunk_ms=float(
                                req.get("chunk_ms", args.chunk_ms)),
                        )
                    conn.send({"ok": True, "result": dict(rec)})
                elif op == "pop_sizes":
                    with contextlib.redirect_stdout(sys.stderr):
                        result = runtime.pop_sizes()
                    conn.send({"ok": True, "result": result})
                elif op == "close":
                    with contextlib.redirect_stdout(sys.stderr):
                        runtime.close()
                    conn.send({"ok": True, "result": {}})
                    return 0
                else:
                    raise ValueError(
                        f"unknown brain subprocess op {op!r}")
            except Exception as exc:
                conn.send({
                    "ok": False,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=20),
                })
    except (EOFError, BrokenPipeError, OSError):
        return 2
    finally:
        try:
            runtime.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
