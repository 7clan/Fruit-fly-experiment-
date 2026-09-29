"""DigitalFlyLab application assembly (pipeline orchestrator).

Wires the full pipeline for a session (FINAL_ARCHITECTURE §1/§10):

  capture → fast vision → world observation ─┬→ world model/memory/value
                                              ├→ planner (helper goals)
  heavy vision (OCR) → world semantics       └→ fly channel encoder
                                                    ↓
  fly channels → BRAIN WORKER (init-once, chunked) → intention
                                                    ↓
  (Gate-6+) AbilityResolver → MotorExecutor → input backend
  dashboard (snapshot reader) + replay recorder (async)

Default run mode is PASSIVE (Gate-5): autonomy disabled — the
MotorExecutor runs in SHADOW mode: it logs what it WOULD do, the input
backend is SafeNoop, and no input is ever injected.

The same assembly runs threaded (dev/sandbox/CI) or multiprocess
(Windows live). This skeleton ships the threaded assembly; the Windows
process launcher spawns the same worker classes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import __version__
from .action.motor_executor import MotorExecutor, SafeNoopBackend
from .brain.worker import BrainWorker
from .brain.runtime import CanonicalBrianRuntime
from .bus import Bus, BusMode
from .capture.base import (CaptureAdapter, SyntheticCapture,
                           create_windows_capture)
from .clock import SHARED_CLOCK
from .dashboard.dashboard import (DashboardWorker, TextDashboardRenderer)
from .perception.channel_encoder import FlyChannelEncoder
from .perception.fast_vision import FastVisionWorker
from .perception.heavy_vision import HeavyVisionWorker
from .replay import ReplayRecorder
from .world.memory import MemoryStore
from .world.planner import PlannerWorker
from .world.value import ValueTable

AGENT_ROOT = Path(__file__).resolve().parents[1]


class DigitalFlyLab:
    """Assembled pipeline. Passive by default (Gate-5 discipline)."""

    def __init__(self, capture: CaptureAdapter | None = None,
                 capture_kind: str = "synthetic",
                 runtime_kind: str = "mock", chunk_ms: float = 50.0,
                 session_dir: Path | None = None,
                 mode: BusMode = BusMode.THREADED,
                 autonomy: bool = False,
                 dashboard: bool = True,
                 brain_hz: float = 10.0,
                 fast_hz: float = 24.0, heavy_hz: float = 8.0,
                 executor_hz: float = 40.0, planner_hz: float = 1.0,
                 dashboard_hz: float = 20.0, capture_fps: float = 30.0,
                 runtime=None, brain_codegen: str | None = None):
        self.bus = Bus(mode=mode)
        self.session_dir = Path(session_dir) if session_dir else \
            AGENT_ROOT / "runs" / f"lab_{time.strftime('%Y%m%d_%H%M%S')}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if capture is not None:
            self.capture = capture
        elif capture_kind == "windows":
            self.capture = create_windows_capture(
                self.bus, window_title_re=r"^Roblox$")
        elif capture_kind == "synthetic":
            self.capture = SyntheticCapture(self.bus, fps=capture_fps)
        else:
            raise ValueError(f"unknown capture_kind {capture_kind!r}")
        self.memory = MemoryStore(self.session_dir / "memory")
        self.value = ValueTable(self.memory)
        # workers (order = pipeline flow)
        self.fast_vision = FastVisionWorker(self.bus, target_hz=fast_hz)
        self.heavy_vision = HeavyVisionWorker(self.bus, target_hz=heavy_hz)
        self.planner = PlannerWorker(self.bus, target_hz=planner_hz)
        self.encoder = FlyChannelEncoder(self.bus, target_hz=fast_hz)
        if runtime is None and runtime_kind in ("canonical", "canonical_brian") \
                and brain_codegen is not None:
            runtime = CanonicalBrianRuntime(
                chunk_ms=chunk_ms, codegen_target=brain_codegen)
        self.brain = BrainWorker(self.bus, target_hz=brain_hz,
                                 runtime=runtime, runtime_kind=runtime_kind,
                                 chunk_ms=chunk_ms)
        self.executor = MotorExecutor(self.bus, target_hz=executor_hz,
                                      backend=SafeNoopBackend(),
                                      autonomy_enabled=autonomy)
        self.replay = ReplayRecorder(self.bus, self.session_dir)
        self.dashboard = DashboardWorker(
            self.bus, target_hz=dashboard_hz,
            renderer=TextDashboardRenderer() if dashboard else None) \
            if dashboard else None
        self.workers = [self.fast_vision, self.heavy_vision, self.planner,
                        self.encoder, self.brain, self.executor,
                        self.replay] + ([self.dashboard] if self.dashboard else [])

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> dict:
        # canonical runtime: brian2 MUST be imported in the MAIN thread
        # (it installs a SIGINT handler at import); do it before spawning
        # worker threads. No-op for the mock runtime.
        warm = getattr(self.brain.runtime, "warm_import", None)
        if warm is not None:
            warm()
        meta = {
            "app": "DigitalFlyLab", "version": __version__,
            "mode": "PASSIVE" if not self.executor.autonomy_enabled else "ACTIVE",
            "runtime": self.brain.runtime.runtime_label,
            "chunk_ms": self.brain.chunk_ms,
            "started_wall_ns": SHARED_CLOCK.wall_time_ns(),
            "bus_specs": self.bus.specs(),
        }
        self.replay.write_header(meta)
        self.memory.start()
        self.capture.start()
        for w in self.workers:
            w.start()
        return meta

    def wait_for_brain_ready(self, timeout_s: float = 180.0) -> bool:
        """Wait for the brain worker's one-time initialization/prewarm.

        Canonical whole-brain construction can take tens of seconds on a
        laptop. Timed smoke-test duration must begin AFTER this completes;
        otherwise a nominal 30 s test can end with zero brain chunks even
        though initialization is still progressing normally.
        """
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.brain.decoder is not None:
                return True
            th = getattr(self.brain, "_thread", None)
            if th is not None and not th.is_alive():
                return False
            time.sleep(0.1)
        return False

    def stop(self) -> dict:
        for w in self.workers:
            w.stop()
        self.capture.stop()
        self.memory.stop(persist=True)
        return self.report()

    # -- synthetic drive (dev/test/benchmark dry-run) ------------------------
    def drive_synthetic(self, seconds: float, fps: float | None = None) -> dict:
        """Drive the synthetic capture for `seconds` at fps (threaded dev)."""
        if not isinstance(self.capture, SyntheticCapture):
            raise RuntimeError("drive_synthetic requires SyntheticCapture")
        fps = fps or self.capture.fps
        dt = 1.0 / fps
        t = 0.0
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end and self.capture.is_running():
            self.capture.tick(t)
            t += dt
            time.sleep(dt)
        return self.report()

    # -- reporting ----------------------------------------------------------
    def report(self) -> dict:
        out = {
            "ts_ns": SHARED_CLOCK.now_ns(),
            "capture": {"source": self.capture.name,
                        "published": self.capture.frames.published,
                        "dropped": self.capture.frames.dropped},
            "bus": self.bus.metrics(),
            "workers": {w.name: dict(w.stats) for w in self.workers},
            "memory": self.memory.stats(),
        }
        return out

    def status_line(self) -> str:
        brain = self.brain.current or {}
        intention = (brain.get("intention") or {}).get("name", "-")
        return (f"[lab] frames={self.capture.frames.published} "
                f"brain_chunks={self.brain.stats['steps']} "
                f"runtime={self.brain.runtime.runtime_label} "
                f"intention={intention} "
                f"errors={sum(w.stats['errors'] for w in self.workers)}")

    def write_report_json(self) -> Path:
        p = self.session_dir / "session_report.json"
        p.write_text(json.dumps(self.report(), indent=1, default=str))
        return p


# ---------------------------------------------------------------------------
# self-test (application boot check; deeper diagnostics on Windows)
# ---------------------------------------------------------------------------

def self_test(runtime_kind: str = "mock") -> dict:
    """Boot check: capture → vision → channels → brain(mock) → intention
    → shadow executor → replay, ~3 s, autonomy FORCED OFF."""
    lab = DigitalFlyLab(runtime_kind=runtime_kind, dashboard=False)
    meta = lab.start()
    try:
        lab.drive_synthetic(seconds=2.0, fps=30.0)
    finally:
        rep = lab.stop()
    ok = (rep["capture"]["published"] > 0
          and lab.brain.stats["steps"] > 0
          and sum(w.stats["errors"] for w in lab.workers) == 0)
    return {"ok": bool(ok), "meta": meta, "report": rep,
            "session_dir": str(lab.session_dir)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("DigitalFlyLab (dev entry)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--runtime", default="mock", choices=["mock", "canonical"])
    ap.add_argument("--capture", default="synthetic",
                    choices=["synthetic", "windows"])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--chunk-ms", type=float, default=50.0)
    ap.add_argument("--brain-hz", type=float, default=10.0)
    ap.add_argument("--fast-hz", type=float, default=24.0)
    ap.add_argument("--heavy-hz", type=float, default=8.0)
    ap.add_argument("--brain-codegen", choices=["numpy", "cython"], default=None,
                    help="canonical runtime codegen target for measured comparison")
    ap.add_argument("--dashboard", action="store_true")
    ap.add_argument("--autonomy", action="store_true",
                    help="DANGER: enables input emission (gated in app policy)")
    args = ap.parse_args(argv)

    if args.autonomy:
        raise SystemExit(
            "autonomy cannot be enabled from the dev CLI — it requires a "
            "passed gate (6+); run the Windows app, which enforces this")

    if args.self_test:
        res = self_test(runtime_kind=args.runtime)
        print(json.dumps({k: res[k] for k in ("ok",)},
                         indent=1))
        print("session:", res["session_dir"])
        return 0 if res["ok"] else 1

    lab = DigitalFlyLab(capture_kind=args.capture,
                        runtime_kind=args.runtime, chunk_ms=args.chunk_ms,
                        brain_hz=args.brain_hz, fast_hz=args.fast_hz,
                        heavy_hz=args.heavy_hz, dashboard=args.dashboard,
                        brain_codegen=args.brain_codegen)
    lab.start()
    try:
        if args.runtime == "canonical":
            print("[lab] waiting for canonical brain init/prewarm ...", flush=True)
            if not lab.wait_for_brain_ready(timeout_s=180.0):
                raise RuntimeError(
                    "canonical brain did not become ready within 180 s")
            print(
                f"[lab] canonical brain READY: init={lab.brain.stats.get('init_ms')} ms "
                f"prewarm={lab.brain.stats.get('prewarm_ms')} ms; "
                f"starting {args.seconds}s timed smoke test",
                flush=True,
            )
        if isinstance(lab.capture, SyntheticCapture):
            lab.drive_synthetic(seconds=args.seconds, fps=30.0)
        else:
            time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        rep = lab.stop()
    print(lab.status_line())
    p = lab.write_report_json()
    print("session:", lab.session_dir, "| report:", p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
