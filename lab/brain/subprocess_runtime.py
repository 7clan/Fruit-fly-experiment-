"""Canonical brain runtime transported through a dedicated subprocess.

The Windows capture/perception/dashboard process stays in the main venv.
The canonical Brian2 brain runs under brain/.venv. Parent/child RPC uses a
localhost-only socket so library writes to stdout/stderr cannot corrupt the
protocol.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import threading
from collections import deque
from typing import Optional

from .runtime import AGENT_ROOT, BrainChunkRecord


class CanonicalBrainSubprocessRuntime:
    runtime_label = "canonical_brian"
    transport_label = "subprocess"

    def __init__(self, chunk_ms: float = 50.0,
                 codegen_target: str = "cython",
                 python_exe: str | None = None):
        self.chunk_ms = float(chunk_ms)
        self.codegen_target = codegen_target
        self.python_exe = python_exe
        self._proc: subprocess.Popen | None = None
        self._pop_sizes: dict = {}
        self._stderr_tail = deque(maxlen=120)
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._listener: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._reader = None
        self._writer = None

    def _resolve_python(self) -> str:
        if self.python_exe:
            return self.python_exe
        env = os.environ.get("DIGITALFLYLAB_BRAIN_PY")
        if env:
            return env
        if os.name == "nt":
            candidate = AGENT_ROOT / "brain" / ".venv" / "Scripts" / "python.exe"
        else:
            candidate = AGENT_ROOT / "brain" / ".venv" / "bin" / "python"
        if candidate.exists():
            return str(candidate)
        return sys.executable

    def warm_import(self) -> bool:
        # Brian2 is intentionally imported only inside the child process.
        return True

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr_tail.append(line.rstrip())

    def _close_ipc(self) -> None:
        for stream in (self._reader, self._writer):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        for s in (self._conn, self._listener):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
        self._conn = None
        self._listener = None

    def _ensure_started(self) -> None:
        if (self._proc is not None and self._proc.poll() is None
                and self._conn is not None):
            return

        self._close_ipc()
        py = self._resolve_python()
        token = secrets.token_hex(16)

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(30.0)
        self._listener = listener
        host, port = listener.getsockname()

        cmd = [
            py, "-u", "-m", "lab.brain.subprocess_server",
            "--chunk-ms", str(self.chunk_ms),
            "--codegen", self.codegen_target,
            "--connect-host", str(host),
            "--connect-port", str(port),
            "--token", token,
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(AGENT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="canonical_brain_stderr",
            daemon=True,
        )
        self._stderr_thread.start()

        try:
            conn, _addr = listener.accept()
        except Exception as exc:
            proc = self._proc
            tail = "\n".join(self._stderr_tail)
            if proc is not None and proc.poll() is None:
                proc.terminate()
            self._close_ipc()
            raise RuntimeError(
                "canonical brain subprocess did not connect to local RPC "
                f"socket: {exc}; stderr tail:\n{tail}") from exc

        conn.settimeout(None)
        self._conn = conn
        self._reader = conn.makefile("r", encoding="utf-8", newline="\n")
        self._writer = conn.makefile(
            "w", encoding="utf-8", newline="\n", buffering=1)

        hello = self._reader.readline()
        if not hello:
            tail = "\n".join(self._stderr_tail)
            raise RuntimeError(
                "canonical brain subprocess disconnected during RPC "
                f"handshake; stderr tail:\n{tail}")
        try:
            msg = json.loads(hello)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid canonical brain RPC handshake: {hello!r}") from exc
        if msg.get("hello") != token:
            raise RuntimeError("canonical brain RPC authentication mismatch")

        # One connection only; no additional local clients are accepted.
        try:
            listener.close()
        finally:
            self._listener = None

    def _rpc(self, payload: dict) -> dict:
        with self._lock:
            self._ensure_started()
            proc = self._proc
            if self._writer is None or self._reader is None:
                raise RuntimeError("canonical brain RPC socket unavailable")
            if proc is None or proc.poll() is not None:
                tail = "\n".join(self._stderr_tail)
                raise RuntimeError(
                    "canonical brain subprocess is not running; "
                    f"rc={None if proc is None else proc.returncode}; "
                    f"stderr tail:\n{tail}")

            try:
                self._writer.write(
                    json.dumps(payload, separators=(",", ":")) + "\n")
                self._writer.flush()
                line = self._reader.readline()
            except Exception as exc:
                tail = "\n".join(self._stderr_tail)
                raise RuntimeError(
                    f"canonical brain RPC failed: {exc}; "
                    f"stderr tail:\n{tail}") from exc

            if line == "":
                tail = "\n".join(self._stderr_tail)
                rc = proc.poll()
                raise RuntimeError(
                    "canonical brain subprocess disconnected from RPC; "
                    f"rc={rc}; stderr tail:\n{tail}")
            try:
                reply = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid canonical brain RPC reply: {line!r}") from exc

            if not reply.get("ok"):
                raise RuntimeError(
                    "canonical brain subprocess error: "
                    + str(reply.get("error"))
                    + "\n"
                    + str(reply.get("traceback", "")))
            return reply.get("result") or {}

    def init_once(self) -> None:
        result = self._rpc({"op": "init"})
        self._pop_sizes = dict(result.get("pop_sizes") or {})

    def prewarm(self, prewarm_chunks: int = 2) -> None:
        self._rpc({"op": "prewarm", "chunks": int(prewarm_chunks)})

    def advance_chunk(self, sensory_rates_hz: dict,
                      chunk_ms: Optional[float] = None) -> BrainChunkRecord:
        result = self._rpc({
            "op": "advance",
            "rates": {k: float(v) for k, v in sensory_rates_hz.items()},
            "chunk_ms": float(
                chunk_ms if chunk_ms is not None else self.chunk_ms),
        })
        result["transport"] = self.transport_label
        return BrainChunkRecord(result)

    def pop_sizes(self) -> dict:
        if self._pop_sizes:
            return dict(self._pop_sizes)
        self._pop_sizes = dict(self._rpc({"op": "pop_sizes"}))
        return dict(self._pop_sizes)

    def force_terminate(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        self._close_ipc()

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            self._close_ipc()
            return
        try:
            if proc.poll() is None:
                try:
                    self._rpc({"op": "close"})
                except Exception:
                    pass
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        finally:
            self._close_ipc()
            if proc.stderr is not None:
                try:
                    proc.stderr.close()
                except Exception:
                    pass
            self._proc = None
