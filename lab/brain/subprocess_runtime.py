"""Canonical brain runtime transported through a dedicated subprocess.

The Windows capture/perception/dashboard process stays in the main venv.
The canonical Brian2 brain runs under brain/.venv and communicates through
small JSON-lines messages. This matches the intended process isolation much
better than running every worker as a Python thread in one process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
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
        self._stderr_tail = deque(maxlen=80)
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.Lock()

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
        # Useful for development when the current interpreter already has
        # Brian2, but Windows production should normally resolve brain/.venv.
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

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        py = self._resolve_python()
        cmd = [
            py, "-u", "-m", "lab.brain.subprocess_server",
            "--chunk-ms", str(self.chunk_ms),
            "--codegen", self.codegen_target,
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(AGENT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
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

    def _rpc(self, payload: dict) -> dict:
        with self._lock:
            self._ensure_started()
            proc = self._proc
            assert proc is not None
            if proc.stdin is None or proc.stdout is None:
                raise RuntimeError("canonical brain subprocess pipes unavailable")
            if proc.poll() is not None:
                tail = "\n".join(self._stderr_tail)
                raise RuntimeError(
                    f"canonical brain subprocess exited rc={proc.returncode}; "
                    f"stderr tail:\n{tail}")
            proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            proc.stdin.flush()

            # Quiet mode should keep stdout JSON-only. If a dependency writes
            # a stray line anyway, retain it for diagnostics and continue.
            while True:
                line = proc.stdout.readline()
                if line == "":
                    tail = "\n".join(self._stderr_tail)
                    raise RuntimeError(
                        "canonical brain subprocess closed stdout; "
                        f"rc={proc.poll()} stderr tail:\n{tail}")
                line = line.strip()
                if not line:
                    continue
                try:
                    reply = json.loads(line)
                    break
                except json.JSONDecodeError:
                    self._stderr_tail.append("[stdout] " + line)

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
            "chunk_ms": float(chunk_ms if chunk_ms is not None else self.chunk_ms),
        })
        result["transport"] = self.transport_label
        return BrainChunkRecord(result)

    def pop_sizes(self) -> dict:
        if self._pop_sizes:
            return dict(self._pop_sizes)
        self._pop_sizes = dict(self._rpc({"op": "pop_sizes"}))
        return dict(self._pop_sizes)

    def force_terminate(self) -> None:
        """Abort an in-flight child chunk during application shutdown."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def close(self) -> None:
        proc = self._proc
        if proc is None:
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
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
            self._proc = None
