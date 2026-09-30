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

The control/perception assembly is threaded today. On Windows live runs,
the canonical Brian2 brain is isolated in its own subprocess; broader
worker-process isolation remains future engineering work.
"""

from __future__ import annotations

import argparse
import json
import signal
import shutil
import threading
import time
from pathlib import Path

from . import __version__
from .action.motor_executor import (
    MotorExecutor, SafeNoopBackend, create_windows_movement_only_backend)
from .brain.worker import BrainWorker
from .brain.runtime import CanonicalBrianRuntime
from .brain.subprocess_runtime import CanonicalBrainSubprocessRuntime
from .bus import Bus, BusMode
from .capture.base import (CaptureAdapter, SyntheticCapture,
                           create_windows_capture)
from .clock import SHARED_CLOCK
from .dashboard.dashboard import (
    DashboardWorker, OpenCVDashboardRenderer, Snapshot, TextDashboardRenderer)
from .evidence import EvidenceRecorder
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
                 movement_only: bool = False,
                 dashboard: bool = True,
                 dashboard_renderer: str = "text",
                 brain_hz: float = 10.0,
                 fast_hz: float = 24.0, heavy_hz: float = 8.0,
                 executor_hz: float = 40.0, planner_hz: float = 1.0,
                 dashboard_hz: float = 20.0, capture_fps: float = 30.0,
                 fast_detect_width: int = 640,
                 runtime=None, brain_codegen: str | None = None,
                 brain_transport: str = "auto"):
        self.bus = Bus(mode=mode)
        self.session_dir = Path(session_dir) if session_dir else \
            AGENT_ROOT / "runs" / f"lab_{time.strftime('%Y%m%d_%H%M%S')}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if capture is not None:
            self.capture = capture
        elif capture_kind == "windows":
            self.capture = create_windows_capture(
                self.bus, window_title_re=r"^Roblox$",
                target_fps=capture_fps)
        elif capture_kind == "synthetic":
            self.capture = SyntheticCapture(self.bus, fps=capture_fps)
        else:
            raise ValueError(f"unknown capture_kind {capture_kind!r}")
        self.memory = MemoryStore(self.session_dir / "memory")
        self.value = ValueTable(self.memory)
        # workers (order = pipeline flow)
        self.fast_vision = FastVisionWorker(
            self.bus, target_hz=fast_hz,
            max_detect_width=int(fast_detect_width))
        self.heavy_vision = HeavyVisionWorker(self.bus, target_hz=heavy_hz)
        self.planner = PlannerWorker(self.bus, target_hz=planner_hz)
        self.encoder = FlyChannelEncoder(self.bus, target_hz=fast_hz)
        if runtime is None and runtime_kind in ("canonical", "canonical_brian"):
            chosen_transport = brain_transport
            if chosen_transport == "auto":
                chosen_transport = (
                    "subprocess" if capture_kind == "windows" else "inprocess")
            if chosen_transport == "subprocess":
                runtime = CanonicalBrainSubprocessRuntime(
                    chunk_ms=chunk_ms,
                    codegen_target=brain_codegen or "cython",
                )
            elif chosen_transport == "inprocess":
                runtime = CanonicalBrianRuntime(
                    chunk_ms=chunk_ms, codegen_target=brain_codegen)
            else:
                raise ValueError(
                    f"unknown brain_transport {brain_transport!r}")
        self.brain = BrainWorker(self.bus, target_hz=brain_hz,
                                 runtime=runtime, runtime_kind=runtime_kind,
                                 chunk_ms=chunk_ms)
        self.autonomy_requested = bool(autonomy and movement_only)
        if self.autonomy_requested:
            backend = create_windows_movement_only_backend()
        else:
            backend = SafeNoopBackend()
        self.executor = MotorExecutor(
            self.bus, target_hz=executor_hz,
            backend=backend,
            # Never emit active input during init/prewarm. Movement autonomy
            # is armed only after brain READY and game-focus succeeds.
            autonomy_enabled=False if self.autonomy_requested else autonomy,
            movement_only=movement_only)
        self.replay = ReplayRecorder(self.bus, self.session_dir)
        self.evidence = (
            EvidenceRecorder(self.bus, self.session_dir, target_hz=4.0)
            if movement_only else None)
        self.dashboard_ui = None
        self.assessment_started_ns = None
        self.assessment_ended_ns = None
        self.abort_requested = threading.Event()
        self.abort_reason = None
        self.wait_end_reason = None
        self._emergency_thread = None
        if dashboard:
            if dashboard_renderer == "opencv":
                # OpenCV HighGUI is substantially more stable on Windows
                # when imshow/waitKey are pumped by the main thread.
                self.dashboard_ui = OpenCVDashboardRenderer(
                    evidence_dir=self.session_dir / "evidence")
                renderer = None
            elif dashboard_renderer == "text":
                renderer = TextDashboardRenderer()
            else:
                raise ValueError(
                    f"unknown dashboard_renderer {dashboard_renderer!r}")
            self.dashboard = DashboardWorker(
                self.bus, target_hz=dashboard_hz, renderer=renderer)
        else:
            self.dashboard = None
        self.workers = [self.fast_vision, self.heavy_vision, self.planner,
                        self.encoder, self.brain, self.executor,
                        self.replay] + ([self.evidence] if self.evidence else []) + (
                            [self.dashboard] if self.dashboard else [])

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
            "mode": ("ACTIVE_REQUESTED_DISARMED"
                     if self.autonomy_requested
                     else ("PASSIVE" if not self.executor.autonomy_enabled
                           else "ACTIVE")),
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

    def render_dashboard_once(self) -> None:
        if self.dashboard_ui is None or self.dashboard is None:
            return
        snap_state = self.dashboard.snapshot_state.read()
        if snap_state is None:
            return
        payload = snap_state.payload
        snap = Snapshot(int(payload["ts_ns"]), dict(payload["columns"]))
        self.dashboard_ui.render(snap, self.capture.latest.read())

    def wait_live(self, seconds: float) -> None:
        self.wait_end_reason = None
        if float(seconds) <= 0:
            while not self.abort_requested.is_set():
                self.render_dashboard_once()
                time.sleep(0.02 if self.dashboard_ui is not None else 0.05)
            self.wait_end_reason = self.abort_reason or "abort"
            return
        deadline_ns = SHARED_CLOCK.now_ns() + int(float(seconds) * 1e9)
        while (SHARED_CLOCK.now_ns() < deadline_ns
               and not self.abort_requested.is_set()):
            self.render_dashboard_once()
            time.sleep(0.02 if self.dashboard_ui is not None else 0.05)
        self.wait_end_reason = (
            self.abort_reason or "abort"
            if self.abort_requested.is_set()
            else "deadline")

    def arm_windows_navigation(self) -> None:
        """Focus the game, arm F12, then enable movement input."""
        if not self.autonomy_requested:
            return
        from .action.windows_input import focus_window, f12_pressed
        target = getattr(self.capture, "_target", None) or {}
        hwnd = int(target.get("handle") or 0)
        if hwnd <= 0:
            raise RuntimeError("cannot arm navigation: captured HWND missing")
        if not focus_window(hwnd):
            raise RuntimeError(
                "cannot arm navigation: failed to focus authorized game window")

        def watch():
            was_down = False
            while not self.abort_requested.is_set():
                down = bool(f12_pressed())
                if down and not was_down:
                    self.executor.emergency_stop(reason="global_f12")
                    self.abort_reason = "global_f12"
                    self.abort_requested.set()
                    break
                was_down = down
                time.sleep(0.05)

        self._emergency_thread = threading.Thread(
            target=watch, name="global_f12_stop", daemon=True)
        self._emergency_thread.start()
        self.executor.set_autonomy(True, reason="brain_ready_game_focused")

    def wait_for_brain_ready(self, timeout_s: float = 180.0) -> bool:
        """Wait for the brain worker's explicit init/prewarm-ready event."""
        deadline_ns = SHARED_CLOCK.now_ns() + int(float(timeout_s) * 1e9)
        while SHARED_CLOCK.now_ns() < deadline_ns:
            self.render_dashboard_once()
            if self.brain.ready_event.wait(timeout=0.1):
                return True
            th = getattr(self.brain, "_thread", None)
            if th is not None and not th.is_alive():
                return False
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
            "assessment": {
                "started_ns": self.assessment_started_ns,
                "wait_end_reason": self.wait_end_reason,
                "abort_reason": self.abort_reason,
                "autonomy_requested": self.autonomy_requested,
                "autonomy_armed": self.executor.autonomy_enabled,
                "ended_ns": self.assessment_ended_ns,
                "elapsed_s": (
                    round((self.assessment_ended_ns - self.assessment_started_ns)
                          / 1e9, 3)
                    if self.assessment_started_ns is not None
                    and self.assessment_ended_ns is not None
                    else None
                ),
            },
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
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="run duration; 0 = run until Ctrl+C")
    ap.add_argument("--capture-fps", type=float, default=30.0)
    ap.add_argument("--chunk-ms", type=float, default=50.0)
    ap.add_argument("--brain-hz", type=float, default=10.0)
    ap.add_argument("--fast-hz", type=float, default=24.0)
    ap.add_argument("--heavy-hz", type=float, default=8.0)
    ap.add_argument("--brain-codegen", choices=["numpy", "cython"], default=None,
                    help="canonical runtime codegen target for measured comparison")
    ap.add_argument("--brain-transport",
                    choices=["auto", "inprocess", "subprocess"],
                    default="auto",
                    help="canonical brain isolation; Windows live defaults to subprocess")
    ap.add_argument("--dashboard", action="store_true")
    ap.add_argument("--dashboard-ui", action="store_true",
                    help="show the OpenCV live fly-brain dashboard window")
    ap.add_argument("--dashboard-hz", type=float, default=5.0)
    ap.add_argument("--fast-detect-width", type=int, default=640,
                    help="max width for fast CV detector; geometry stays normalized")
    ap.add_argument("--autonomy", action="store_true",
                    help="generic autonomy remains blocked; use movement gate launcher")
    ap.add_argument("--movement-only-autonomy", action="store_true",
                    help="Gate-6 only: arm hard-filtered W/A/D movement backend")
    args = ap.parse_args(argv)

    if args.autonomy:
        raise SystemExit(
            "generic autonomy is blocked; use the dedicated movement-only "
            "Gate-6 launcher")
    if args.movement_only_autonomy and args.capture != "windows":
        raise SystemExit("movement-only autonomy requires Windows capture")

    if args.self_test:
        res = self_test(runtime_kind=args.runtime)
        print(json.dumps({k: res[k] for k in ("ok",)},
                         indent=1))
        print("session:", res["session_dir"])
        return 0 if res["ok"] else 1

    lab = DigitalFlyLab(capture_kind=args.capture,
                        capture_fps=args.capture_fps,
                        autonomy=bool(args.movement_only_autonomy),
                        movement_only=bool(args.movement_only_autonomy),
                        runtime_kind=args.runtime, chunk_ms=args.chunk_ms,
                        brain_hz=args.brain_hz, fast_hz=args.fast_hz,
                        heavy_hz=args.heavy_hz,
                        fast_detect_width=args.fast_detect_width,
                        dashboard=(args.dashboard or args.dashboard_ui),
                        dashboard_renderer=("opencv" if args.dashboard_ui else "text"),
                        dashboard_hz=args.dashboard_hz,
                        brain_codegen=args.brain_codegen,
                        brain_transport=args.brain_transport)
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
                + (("starting persistent movement-only run (F12/Ctrl+C to stop)"
                    if args.movement_only_autonomy else
                    "starting persistent passive run (Ctrl+C to stop)")
                   if args.seconds <= 0 else
                   (f"starting {args.seconds}s movement-only test"
                    if args.movement_only_autonomy else
                    f"starting {args.seconds}s timed smoke test")),
                flush=True,
            )
        if args.movement_only_autonomy:
            lab.arm_windows_navigation()
            print("[lab] navigation armed; Roblox focused; F12 = EMERGENCY STOP",
                  flush=True)
        lab.assessment_started_ns = SHARED_CLOCK.now_ns()
        if isinstance(lab.capture, SyntheticCapture):
            lab.drive_synthetic(seconds=args.seconds, fps=30.0)
        else:
            lab.wait_live(args.seconds)
        lab.assessment_ended_ns = SHARED_CLOCK.now_ns()
    except KeyboardInterrupt:
        lab.assessment_ended_ns = SHARED_CLOCK.now_ns()
        pass
    finally:
        # First Ctrl+C requests shutdown. Ignore additional Ctrl+C presses
        # while workers are draining/terminating so cleanup and the session
        # report cannot be interrupted halfway through.
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass
        print("[lab] stopping; please wait for cleanup/report...", flush=True)
        rep = lab.stop()
    print(lab.status_line())
    if (lab.assessment_started_ns is not None
            and lab.assessment_ended_ns is not None):
        print(f"[lab] post_ready_run_s="
              f"{(lab.assessment_ended_ns-lab.assessment_started_ns)/1e9:.1f} "
              f"end_reason={lab.wait_end_reason}")
    p = lab.write_report_json()
    bundle = shutil.make_archive(
        str(lab.session_dir), "zip",
        root_dir=str(lab.session_dir.parent),
        base_dir=lab.session_dir.name,
    )
    print("session:", lab.session_dir, "| report:", p)
    print("bundle:", bundle)
    bad = [
        (w.name, int(w.stats.get("errors", 0)), w.stats.get("last_error"))
        for w in lab.workers
        if int(w.stats.get("errors", 0)) > 0
    ]
    if bad:
        print("[lab] worker errors:")
        for name, count, last_error in bad:
            print(f"  - {name}: errors={count} last_error={last_error}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
