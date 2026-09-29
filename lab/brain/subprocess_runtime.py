"""Canonical brain runtime transported through a dedicated subprocess.

Windows live runs use a local AF_PIPE named pipe for RPC. This keeps the
canonical Brian2/Cython process isolated without relying on localhost TCP
(which can be interrupted by Windows firewall/AV software) or stdout pipes
(which scientific libraries may write to).
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
from collections import deque
from multiprocessing.connection import Listener
from typing import Optional

from .runtime import AGENT_ROOT, BrainChunkRecord


class CanonicalBrainTransportError(RuntimeError):
    """Fatal parent/child IPC failure; the brain worker should fail closed."""


class CanonicalBrainSubprocessRuntime:
    runtime_label = "canonical_brian"
    transport_label = "subprocess_pipe"

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
        self._listener = None
        self._conn = None

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
        return True

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr_tail.append(line.rstrip())

    def _close_ipc(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None
        try:
            if self._listener is not None:
                self._listener.close()
        except Exception:
            pass
        self._listener = None

    def _ensure_started(self) -> None:
        if (self._proc is not None and self._proc.poll() is None
                and self._conn is not None):
            return
        if os.name != "nt":
            raise CanonicalBrainTransportError(
                "AF_PIPE brain transport currently requires Windows")

        self._close_ipc()
        py = self._resolve_python()
        authkey = secrets.token_bytes(32)
        address = (
            rf"\\.\pipe\DigitalFlyLabBrain-"
            f"{os.getpid()}-{secrets.token_hex(8)}"
        )
        self._listener = Listener(
            address=address,
            family="AF_PIPE",
            authkey=authkey,
        )

        cmd = [
            py, "-u", "-m", "lab.brain.subprocess_server",
            "--chunk-ms", str(self.chunk_ms),
            "--codegen", self.codegen_target,
            "--pipe-address", address,
            "--authkey-hex", authkey.hex(),
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
            self._conn = self._listener.accept()
        except Exception as exc:
            proc = self._proc
            tail = "\n".join(self._stderr_tail)
            if proc is not None and proc.poll() is None:
                proc.terminate()
            self._close_ipc()
            raise CanonicalBrainTransportError(
                "canonical brain child failed to connect to Windows named "
                f"pipe: {exc}; stderr tail:\n{tail}") from exc
        finally:
            try:
                if self._listener is not None:
                    self._listener.close()
            except Exception:
                pass
            self._listener = None

    def _rpc(self, payload: dict) -> dict:
        with self._lock:
            self._ensure_started()
            proc = self._proc
            if self._conn is None:
                raise CanonicalBrainTransportError(
                    "canonical brain named pipe unavailable")
            if proc is None or proc.poll() is not None:
                tail = "\n".join(self._stderr_tail)
                raise CanonicalBrainTransportError(
                    "canonical brain subprocess is not running; "
                    f"rc={None if proc is None else proc.returncode}; "
                    f"stderr tail:\n{tail}")
            try:
                self._conn.send(payload)
                reply = self._conn.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                tail = "\n".join(self._stderr_tail)
                rc = proc.poll()
                self.force_terminate()
                raise CanonicalBrainTransportError(
                    "canonical brain named-pipe RPC disconnected; "
                    f"rc={rc}; error={exc}; stderr tail:\n{tail}") from exc

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
        self._close_ipc()
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            self._close_ipc()
            return
        try:
            if proc.poll() is None and self._conn is not None:
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
