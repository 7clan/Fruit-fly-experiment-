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
from .action.gpo_controls import CORE_CONTROLS, loadout_controls
from .action.motor_executor import (
    MotorExecutor, SafeNoopBackend, create_windows_movement_only_backend,
    create_windows_questing_backend)
from .action.quest_combat_supervisor import QuestCombatSupervisor
from .action.ai_only_supervisor import AIOnlyAutopilotSupervisor
from .brain.worker import BrainWorker
from .brain.runtime import CanonicalBrianRuntime
from .brain.subprocess_runtime import CanonicalBrainSubprocessRuntime
from .bus import Bus, BusMode
from .capture.base import (CaptureAdapter, SyntheticCapture,
                           create_windows_capture)
from .clock import SHARED_CLOCK
from .coach import (
    SemanticCoachWorker, LocalSmolVLMCoachWorker, LocalSmolLMCoachWorker,
    LlamaApiCoachWorker, MetaModelApiCoachWorker, OllamaCloudCoachWorker)
from .coach.ai_only import AIOnlyOllamaCoachWorker
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
                 quest_autonomy: bool = False,
                 ai_only: bool = False,
                 gpo_loadout: str = "default_melee",
                 semantic_coach: bool = False,
                 coach_provider: str = "gemini",
                 coach_model: str | None = None,
                 coach_url: str | None = None,
                 coach_hz: float = 1.0,
                 dashboard: bool = True,
                 dashboard_renderer: str = "text",
                 brain_hz: float = 10.0,
                 fast_hz: float = 24.0, heavy_hz: float = 8.0,
                 executor_hz: float = 40.0, planner_hz: float = 1.0,
                 dashboard_hz: float = 20.0, capture_fps: float = 30.0,
                 capture_downsample: int = 1,
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
                target_fps=capture_fps,
                downsample=capture_downsample)
        elif capture_kind == "synthetic":
            self.capture = SyntheticCapture(self.bus, fps=capture_fps)
        else:
            raise ValueError(f"unknown capture_kind {capture_kind!r}")
        self.memory = MemoryStore(self.session_dir / "memory")
        self.value = ValueTable(self.memory)
        self.value_path = (
            AGENT_ROOT / "runtime_state" / "gpo_engineered_values.json")
        try:
            self.value.load_file(self.value_path)
        except Exception:
            # Corrupt optional learned state must never prevent safe startup.
            pass
        self.action_meta = self.bus.state("action.meta")
        self.control_catalog_state = self.bus.state("action.control_catalog")

        self.quest_autonomy = bool(quest_autonomy)
        self.ai_only = bool(ai_only)
        self.gpo_loadout = str(gpo_loadout or "default_melee")
        active_low_power = bool(
            movement_only or self.quest_autonomy or self.ai_only)

        # LOW-POWER LIVE PROFILE. On the target i7-5500U (2C/4T), the
        # canonical Brian2 subprocess is the dominant workload.
        if active_low_power:
            executor_hz = min(float(executor_hz), 20.0)
            planner_hz = min(float(planner_hz), 0.5)

        # workers (order = pipeline flow)
        self.fast_vision = FastVisionWorker(
            self.bus, target_hz=fast_hz,
            max_detect_width=int(fast_detect_width),
            # Questing uses cheap explicit yellow/green/red objective markers
            # instead of the expensive generic humanoid proposal pass.
            role_detection=not active_low_power)
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
        self.autonomy_requested = bool(
            autonomy and (
                movement_only or self.quest_autonomy or self.ai_only))
        if self.autonomy_requested and (self.quest_autonomy or self.ai_only):
            backend = create_windows_questing_backend()
        elif self.autonomy_requested:
            backend = create_windows_movement_only_backend()
        else:
            backend = SafeNoopBackend()
        self.executor = MotorExecutor(
            self.bus, target_hz=executor_hz,
            backend=backend,
            # Never emit active input during init/prewarm.
            autonomy_enabled=False if self.autonomy_requested else autonomy,
            movement_only=movement_only,
            questing=(self.quest_autonomy or self.ai_only),
            command_only=self.ai_only)

        if self.ai_only:
            if not semantic_coach:
                raise ValueError("AI-only mode requires semantic_coach=True")
            provider = str(coach_provider or "ollama_cloud").strip().lower()
            if provider != "ollama_cloud":
                raise ValueError(
                    "AI-only branch currently requires ollama_cloud")
            self.semantic_coach = AIOnlyOllamaCoachWorker(
                self.bus,
                target_hz=float(coach_hz),
                model=coach_model,
                base_url=coach_url)
        elif self.quest_autonomy and semantic_coach:
            provider = str(coach_provider or "gemini").strip().lower()
            if provider == "local":
                self.semantic_coach = LocalSmolLMCoachWorker(
                    self.bus,
                    target_hz=float(coach_hz),
                    model=coach_model,
                    base_url=coach_url)
            elif provider == "meta":
                self.semantic_coach = MetaModelApiCoachWorker(
                    self.bus,
                    target_hz=float(coach_hz),
                    model=coach_model,
                    base_url=coach_url)
            elif provider == "llama":
                self.semantic_coach = LlamaApiCoachWorker(
                    self.bus,
                    target_hz=float(coach_hz),
                    model=coach_model,
                    base_url=coach_url)
            elif provider == "ollama_cloud":
                self.semantic_coach = OllamaCloudCoachWorker(
                    self.bus,
                    target_hz=float(coach_hz),
                    model=coach_model,
                    base_url=coach_url)
            elif provider == "gemini":
                self.semantic_coach = SemanticCoachWorker(
                    self.bus,
                    target_hz=float(coach_hz),
                    model=coach_model)
            else:
                raise ValueError(
                    f"unknown semantic coach provider {coach_provider!r}")
        else:
            self.semantic_coach = None
        self.ai_only_supervisor = (
            AIOnlyAutopilotSupervisor(self.bus, target_hz=8.0)
            if self.ai_only else None)
        self.quest_supervisor = (
            QuestCombatSupervisor(
                self.bus, self.value, target_hz=4.0,
                loadout=self.gpo_loadout)
            if self.quest_autonomy and not self.ai_only else None)
        self.replay = ReplayRecorder(
            self.bus, self.session_dir,
            target_hz=10.0 if active_low_power else 30.0)
        self.evidence = (
            EvidenceRecorder(
                self.bus, self.session_dir,
                target_hz=1.0, save_raw=False)
            if active_low_power else None)
        self.dashboard_ui = None
        self.assessment_started_ns = None
        self.assessment_ended_ns = None
        self.abort_requested = threading.Event()
        self.abort_reason = None
        self.wait_end_reason = None
        self._emergency_thread = None
        self._last_dashboard_snapshot_ts = None
        self._dashboard_render_errors = 0
        if dashboard:
            if dashboard_renderer == "opencv":
                # OpenCV HighGUI is substantially more stable on Windows
                # when imshow/waitKey are pumped by the main thread.
                self.dashboard_ui = OpenCVDashboardRenderer(
                    evidence_dir=(
                        None if active_low_power
                        else self.session_dir / "evidence"),
                    control_handler=self._handle_dashboard_control,
                    lightweight=active_low_power,
                    preview_interval_s=3.0)
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
        self.workers = [self.fast_vision, self.heavy_vision, self.planner] + (
                            [] if self.ai_only
                            else [self.encoder, self.brain]) + (
                            [self.semantic_coach]
                            if self.semantic_coach else []) + (
                            [self.ai_only_supervisor]
                            if self.ai_only_supervisor else []) + (
                            [self.quest_supervisor]
                            if self.quest_supervisor else []) + [
                        self.executor, self.replay] + (
                            [self.evidence] if self.evidence else []) + (
                            [self.dashboard] if self.dashboard else [])

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> dict:
        # canonical runtime: brian2 MUST be imported in the MAIN thread
        # (it installs a SIGINT handler at import); do it before spawning
        # worker threads. No-op for the mock runtime.
        warm = (
            None if self.ai_only
            else getattr(self.brain.runtime, "warm_import", None))
        if warm is not None:
            warm()
        meta = {
            "app": "DigitalFlyLab", "version": __version__,
            "mode": (
                "AI_ONLY_REQUESTED_DISARMED"
                if self.autonomy_requested and self.ai_only
                else "QUEST_PVE_REQUESTED_DISARMED"
                if self.autonomy_requested and self.quest_autonomy
                else "ACTIVE_REQUESTED_DISARMED"
                if self.autonomy_requested
                else ("PASSIVE" if not self.executor.autonomy_enabled
                      else "ACTIVE")),
            "gpo_mode": (
                "ai_only_v1" if self.ai_only
                else "quest_pve_v1" if self.quest_autonomy
                else "navigation_v1" if self.autonomy_requested
                else "passive"),
            "gpo_loadout": self.gpo_loadout,
            "semantic_coach": {
                "requested": bool(self.semantic_coach is not None),
                "provider": (
                    getattr(self.semantic_coach, "provider", None)
                    if self.semantic_coach else None),
                "model": (
                    self.semantic_coach.model
                    if self.semantic_coach else None),
                "base_url": (
                    getattr(self.semantic_coach, "base_url", None)
                    if self.semantic_coach else None),
                "enabled": bool(self.semantic_coach is not None),
            },
            "runtime": (
                "disabled_ai_only" if self.ai_only
                else self.brain.runtime.runtime_label),
            "chunk_ms": None if self.ai_only else self.brain.chunk_ms,
            "decision_owner": "cloud_ai" if self.ai_only else "hybrid_fly",
            "capture_config": {
                "fps": float(getattr(self.capture, "target_fps",
                                     getattr(self.capture, "fps", 0.0))),
                "downsample": int(getattr(self.capture, "downsample", 1)),
            },
            "low_power_profile": bool(self.autonomy_requested),
            "gpo_control_catalog": {
                "count": len(CORE_CONTROLS) + len(
                    loadout_controls(self.gpo_loadout)),
                "ids": (
                    [x.control_id for x in CORE_CONTROLS]
                    + [x.control_id for x in
                       loadout_controls(self.gpo_loadout)]),
                "dynamic_equipped_moves": True,
                "loadout": self.gpo_loadout,
            },
            "worker_targets_hz": {
                "fast": self.fast_vision.governor.target_hz,
                "fast_role_detection": self.fast_vision.role_detection,
                "heavy": self.heavy_vision.governor.target_hz,
                "planner": self.planner.governor.target_hz,
                "encoder": (
                    None if self.ai_only else self.encoder.governor.target_hz),
                "executor": self.executor.governor.target_hz,
                "coach": (
                    self.semantic_coach.governor.target_hz
                    if self.semantic_coach is not None else None),
                "replay": self.replay.governor.target_hz,
                "evidence": (
                    self.evidence.governor.target_hz
                    if self.evidence is not None else None),
                "dashboard": (
                    self.dashboard.governor.target_hz
                    if self.dashboard is not None else None),
            },
            "started_wall_ns": SHARED_CLOCK.wall_time_ns(),
            "bus_specs": self.bus.specs(),
        }
        self.replay.write_header(meta)
        self.action_meta.write({
            "movement_control_available": self.autonomy_requested,
            "autonomy": self.executor.autonomy_enabled,
            "mode": (
                "ai_only_v1" if self.ai_only
                else "quest_pve_v1" if self.quest_autonomy
                else "navigation_v1" if self.autonomy_requested
                else "passive"),
            "gpo_loadout": self.gpo_loadout,
            "last_control_reason": "startup_disarmed",
            "ts_ns": SHARED_CLOCK.now_ns(),
        })
        controls = list(CORE_CONTROLS) + loadout_controls(self.gpo_loadout)
        self.control_catalog_state.write({
            "controls": [x.to_dict() for x in controls],
            "dynamic_equipped_moves": True,
            "loadout": self.gpo_loadout,
            "ts_ns": SHARED_CLOCK.now_ns(),
        })
        self.memory.start()
        self.capture.start()
        for w in self.workers:
            w.start()
        return meta

    def render_dashboard_once(self) -> None:
        if self.dashboard_ui is None or self.dashboard is None:
            return
        try:
            snap_state = self.dashboard.snapshot_state.read()
            if snap_state is None:
                return
            payload = snap_state.payload
            snap_ts = int(payload["ts_ns"])
            if snap_ts == self._last_dashboard_snapshot_ts:
                # Pump HighGUI events without recomposing the full panel.
                self.dashboard_ui._cv2.waitKey(1)
                return
            self._last_dashboard_snapshot_ts = snap_ts
            snap = Snapshot(snap_ts, dict(payload["columns"]))
            self.dashboard_ui.render(snap, self.capture.latest.read())
            self._dashboard_render_errors = 0
        except Exception as exc:
            # The dashboard is observational only. A HighGUI/render failure
            # must never stop the fly, brain, replay, or input safety thread.
            self._dashboard_render_errors += 1
            if self._dashboard_render_errors <= 3:
                print(
                    f"[lab] dashboard render warning "
                    f"{self._dashboard_render_errors}/3: {exc!r}",
                    flush=True,
                )
            if self._dashboard_render_errors >= 3:
                print(
                    "[lab] dashboard disabled after repeated render errors; "
                    "control continues (F8/F9/F10/F11/F12 still work)",
                    flush=True,
                )
                try:
                    self.dashboard_ui._cv2.destroyWindow(
                        self.dashboard_ui.window_name)
                except Exception:
                    pass
                self.dashboard_ui = None

    def _set_navigation_enabled(self, enabled: bool,
                                reason: str = "dashboard") -> None:
        if not self.autonomy_requested:
            return
        if enabled:
            if (not self.ai_only and not self.brain.ready_event.is_set()):
                print("[lab] movement remains disabled until brain READY",
                      flush=True)
                return
            from .action.windows_input import focus_window
            target = getattr(self.capture, "_target", None) or {}
            hwnd = int(target.get("handle") or 0)
            if hwnd <= 0 or not focus_window(hwnd):
                print("[lab] movement not enabled: could not focus Roblox",
                      flush=True)
                return
            self.executor.set_autonomy(True, reason=reason)
            self.action_meta.write({
                "movement_control_available": True,
                "autonomy": True,
                "mode": (
                    "ai_only_v1" if self.ai_only
                    else "quest_pve_v1" if self.quest_autonomy
                    else "navigation_v1"),
                "gpo_loadout": self.gpo_loadout,
                "last_control_reason": str(reason),
                "ts_ns": SHARED_CLOCK.now_ns(),
            })
            print(
                (f"[lab] AI-ONLY AUTOPILOT ENABLED ({reason})"
                 if self.ai_only else
                 ("[lab] QUEST/PVE AGENT ENABLED "
                  f"({reason})")
                 if self.quest_autonomy else
                 f"[lab] MOVEMENT ENABLED ({reason})"),
                flush=True)
        else:
            self.executor.set_autonomy(False, reason=reason)
            self.action_meta.write({
                "movement_control_available": self.autonomy_requested,
                "autonomy": False,
                "mode": (
                    "ai_only_v1" if self.ai_only
                    else "quest_pve_v1" if self.quest_autonomy
                    else "navigation_v1" if self.autonomy_requested
                    else "passive"),
                "gpo_loadout": self.gpo_loadout,
                "last_control_reason": str(reason),
                "ts_ns": SHARED_CLOCK.now_ns(),
            })
            print(
                (f"[lab] AI-ONLY AUTOPILOT DISABLED ({reason})"
                 if self.ai_only else
                 f"[lab] QUEST/PVE AGENT DISABLED ({reason})"
                 if self.quest_autonomy else
                 f"[lab] MOVEMENT DISABLED ({reason})"),
                flush=True)

    def _handle_dashboard_control(self, command: str) -> None:
        if command == "enable_movement":
            self._set_navigation_enabled(
                True, reason="dashboard_enable")
        elif command == "disable_movement":
            self._set_navigation_enabled(
                False, reason="dashboard_disable")
        elif command == "refocus_game":
            from .action.windows_input import focus_window
            target = getattr(self.capture, "_target", None) or {}
            hwnd = int(target.get("handle") or 0)
            ok = bool(hwnd > 0 and focus_window(hwnd))
            print(
                "[lab] Roblox refocused" if ok
                else "[lab] Roblox refocus FAILED",
                flush=True,
            )
        elif command == "release_keys":
            # One-shot release without ending the run. Autonomy remains in
            # its current enabled/disabled state; a later NEW brain decision
            # may issue movement again if movement is still enabled.
            self.executor.emergency_stop(reason="dashboard_release_keys")
            print("[lab] all held movement keys released", flush=True)
        elif command == "emergency_stop":
            self._set_navigation_enabled(
                False, reason="dashboard_emergency_stop")
            self.abort_reason = "dashboard_emergency_stop"
            self.abort_requested.set()
            print("[lab] EMERGENCY STOP — movement disabled; ending run",
                  flush=True)
        elif command == "end_run":
            self._set_navigation_enabled(False, reason="dashboard_end")
            self.abort_reason = "dashboard_end"
            self.abort_requested.set()

    def wait_live(self, seconds: float) -> None:
        self.wait_end_reason = None
        if float(seconds) <= 0:
            while not self.abort_requested.is_set():
                self.render_dashboard_once()
                time.sleep(0.05)
            self.wait_end_reason = self.abort_reason or "abort"
            return
        deadline_ns = SHARED_CLOCK.now_ns() + int(float(seconds) * 1e9)
        while (SHARED_CLOCK.now_ns() < deadline_ns
               and not self.abort_requested.is_set()):
            self.render_dashboard_once()
            time.sleep(0.05)
        self.wait_end_reason = (
            self.abort_reason or "abort"
            if self.abort_requested.is_set()
            else "deadline")

    def arm_windows_navigation(self) -> None:
        """Focus the game, arm F12, then enable movement input."""
        if not self.autonomy_requested:
            return
        from .action.windows_input import focus_window, function_key_pressed
        target = getattr(self.capture, "_target", None) or {}
        hwnd = int(target.get("handle") or 0)
        if hwnd <= 0:
            raise RuntimeError("cannot arm navigation: captured HWND missing")
        target_setter = getattr(
            self.executor.backend, "set_target_window", None)
        if target_setter is not None:
            target_setter(hwnd)
        focused = bool(focus_window(hwnd))
        if not focused and self.dashboard_ui is None:
            raise RuntimeError(
                "cannot arm navigation: failed to focus authorized game window")
        if not focused:
            # Dashboard runs start disarmed anyway. A transient Windows
            # foreground-lock failure must not throw away a minute-long brain
            # prewarm; ENABLE/REFOCUS will retry focus when the user is ready.
            self.action_meta.write({
                "movement_control_available": True,
                "autonomy": False,
                "mode": (
                    "ai_only_v1" if self.ai_only
                    else "quest_pve_v1" if self.quest_autonomy
                    else "navigation_v1"),
                "gpo_loadout": self.gpo_loadout,
                "last_control_reason": "initial_focus_pending",
                "ts_ns": SHARED_CLOCK.now_ns(),
            })
            print(
                "[lab] Roblox focus pending — run remains safely DISABLED; "
                "use REFOCUS or ENABLE to retry",
                flush=True,
            )

        def watch():
            # Dashboard-independent controls:
            # F8 enable, F9 disable, F10 refocus, F11 release, F12 emergency.
            prev = {n: False for n in range(8, 13)}
            obs_state = self.bus.state("world.observation")
            last_health = None
            while not self.abort_requested.is_set():
                for n in range(8, 13):
                    down = bool(function_key_pressed(n))
                    if down and not prev[n]:
                        if n == 8:
                            self._handle_dashboard_control("enable_movement")
                        elif n == 9:
                            self._handle_dashboard_control("disable_movement")
                        elif n == 10:
                            self._handle_dashboard_control("refocus_game")
                        elif n == 11:
                            self._handle_dashboard_control("release_keys")
                        elif n == 12:
                            self.executor.emergency_stop(reason="global_f12")
                            self.abort_reason = "global_f12"
                            self.abort_requested.set()
                            break
                    prev[n] = down

                # Movement-only safety exception: if a clear health drop is
                # observed, release movement immediately instead of walking
                # deeper into combat. This does not choose a movement key.
                env = obs_state.read()
                if env is not None:
                    player = (env.payload or {}).get("player") or {}
                    health = player.get("health")
                    units = player.get("health_units")
                    try:
                        health = float(health)
                    except (TypeError, ValueError):
                        health = None
                    if health is not None and units == "fraction":
                        if self.quest_autonomy or self.ai_only:
                            # Active quest/AI mode keeps running through ordinary
                            # damage; only the critical-health hard stop lives
                            # in this safety watcher.
                            # Do not disable the agent on every hit.
                            if (health <= 0.12
                                    and self.executor.autonomy_enabled):
                                self._set_navigation_enabled(
                                    False, reason="critical_health_stop")
                                self.action_meta.write({
                                    "movement_control_available": True,
                                    "autonomy": False,
                                    "mode": (
                                        "ai_only_v1" if self.ai_only
                                        else "quest_pve_v1"),
                                    "last_safety_event":
                                        "critical_health_stop",
                                    "last_control_reason":
                                        "critical_health_stop",
                                    "ts_ns": SHARED_CLOCK.now_ns(),
                                })
                                print(
                                    f"[lab] CRITICAL HEALTH STOP: "
                                    f"health={health:.2f}; agent disabled",
                                    flush=True,
                                )
                        elif (last_health is not None
                              and last_health - health >= 0.06
                              and self.executor.autonomy_enabled):
                            self._set_navigation_enabled(
                                False, reason="damage_safety_pause")
                            self.action_meta.write({
                                "movement_control_available": True,
                                "autonomy": False,
                                "mode": "navigation_v1",
                                "last_safety_event": "damage_pause",
                                "last_control_reason": "damage_safety_pause",
                                "ts_ns": SHARED_CLOCK.now_ns(),
                            })
                            print(
                                f"[lab] DAMAGE SAFETY PAUSE: health "
                                f"{last_health:.2f}->{health:.2f}; "
                                "movement disabled",
                                flush=True,
                            )
                        last_health = health
                time.sleep(0.05)

        self._emergency_thread = threading.Thread(
            target=watch, name="global_navigation_controls", daemon=True)
        self._emergency_thread.start()
        # Dashboard-controlled runs start safely paused. The user explicitly
        # clicks ENABLE MOVEMENT; that click re-focuses Roblox before input.
        if self.dashboard_ui is None:
            self._set_navigation_enabled(
                True, reason="brain_ready_game_focused")
        else:
            self.executor.set_autonomy(False, reason="dashboard_start_paused")

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
        try:
            self.value.save_file(self.value_path)
        except Exception as exc:
            print(f"[lab] learned-value save warning: {exc!r}", flush=True)
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
            "input_backend": self.executor.backend.backend_health(),
            "coach_state": (
                (lambda e: (e.payload if e is not None else None))(
                    self.bus.state("coach.plan").read())
                if self.semantic_coach is not None else None),
            "quest_state": (
                (self.bus.state("quest.state").read().payload
                 if self.bus.state("quest.state").read() is not None
                 else None)
                if (self.quest_autonomy or self.ai_only) else None),
            "engineered_values": self.value.snapshot(),
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
        brain = {} if self.ai_only else (self.brain.current or {})
        intention = (brain.get("intention") or {}).get("name", "-")
        qenv = self.bus.state("quest.state").read()
        quest_phase = (
            (qenv.payload or {}).get("phase", "-")
            if qenv is not None else "-")
        cenv = self.bus.state("coach.plan").read()
        coach_skill = (
            (cenv.payload or {}).get("skill", "-")
            if cenv is not None else "-")
        brain_chunks = 0 if self.ai_only else self.brain.stats["steps"]
        runtime_label = (
            "disabled_ai_only" if self.ai_only
            else self.brain.runtime.runtime_label)
        return (f"[lab] frames={self.capture.frames.published} "
                f"brain_chunks={brain_chunks} "
                f"runtime={runtime_label} "
                f"intention={intention} "
                f"quest_phase={quest_phase} "
                f"coach={coach_skill} "
                f"inputs={self.executor.stats.get('inputs_emitted', 0)} "
                f"send_ok={self.executor.backend.backend_health().get('sendinput_successes', 0)} "
                f"send_fail={self.executor.backend.backend_health().get('sendinput_failures', 0)} "
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
    ap.add_argument("--capture-downsample", type=int, default=1,
                    help="Windows capture stride downsample before BGR copy")
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
                    help="Gate-6 only: hard-filtered navigation backend")
    ap.add_argument("--quest-autonomy", action="store_true",
                    help="Gate-7: low-power quest + starter-PvE hybrid agent")
    ap.add_argument("--ai-only-autonomy", action="store_true",
                    help="AI-only branch: cloud AI owns all gameplay decisions; fly brain is not started")
    ap.add_argument("--gpo-loadout", default="default_melee",
                    help="observed equipped GPO loadout profile")
    ap.add_argument("--semantic-coach", action="store_true",
                    help="low-rate multimodal game coach (quest mode)")
    ap.add_argument("--coach-provider",
                    choices=["gemini", "meta", "llama", "ollama_cloud", "local"], default="ollama_cloud",
                    help="semantic coach provider")
    ap.add_argument("--coach-model", default=None,
                    help="provider model id")
    ap.add_argument("--coach-url", default=None,
                    help="local coach OpenAI-compatible base URL")
    ap.add_argument("--coach-hz", type=float, default=1.0,
                    help="coach polling rate; API calls are event/rate gated")
    args = ap.parse_args(argv)

    if args.autonomy:
        raise SystemExit(
            "generic autonomy is blocked; use the dedicated movement-only "
            "Gate-6 launcher")
    active_modes = sum(bool(x) for x in (
        args.movement_only_autonomy,
        args.quest_autonomy,
        args.ai_only_autonomy,
    ))
    if active_modes > 1:
        raise SystemExit(
            "choose exactly one active mode: movement-only, quest-autonomy, "
            "or ai-only-autonomy")
    if ((args.movement_only_autonomy or args.quest_autonomy
         or args.ai_only_autonomy)
            and args.capture != "windows"):
        raise SystemExit("active GPO autonomy requires Windows capture")
    if args.semantic_coach and not (
            args.quest_autonomy or args.ai_only_autonomy):
        raise SystemExit(
            "--semantic-coach requires --quest-autonomy or --ai-only-autonomy")
    if args.ai_only_autonomy and not args.semantic_coach:
        raise SystemExit("--ai-only-autonomy requires --semantic-coach")
    if args.ai_only_autonomy and args.coach_provider != "ollama_cloud":
        raise SystemExit(
            "--ai-only-autonomy currently requires --coach-provider ollama_cloud")
    if args.ai_only_autonomy and args.runtime != "mock":
        raise SystemExit(
            "--ai-only-autonomy deliberately uses --runtime mock; "
            "the fly brain is not started")

    if args.self_test:
        res = self_test(runtime_kind=args.runtime)
        print(json.dumps({k: res[k] for k in ("ok",)},
                         indent=1))
        print("session:", res["session_dir"])
        return 0 if res["ok"] else 1

    lab = DigitalFlyLab(capture_kind=args.capture,
                        capture_fps=args.capture_fps,
                        capture_downsample=args.capture_downsample,
                        autonomy=bool(
                            args.movement_only_autonomy
                            or args.quest_autonomy
                            or args.ai_only_autonomy),
                        movement_only=bool(args.movement_only_autonomy),
                        quest_autonomy=bool(args.quest_autonomy),
                        ai_only=bool(args.ai_only_autonomy),
                        gpo_loadout=args.gpo_loadout,
                        semantic_coach=bool(args.semantic_coach),
                        coach_provider=args.coach_provider,
                        coach_model=args.coach_model,
                        coach_url=args.coach_url,
                        coach_hz=args.coach_hz,
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
        if args.ai_only_autonomy:
            print(
                "[lab] AI-ONLY mode: fruit-fly brain is NOT started; "
                "cloud AI is the sole gameplay decision-maker",
                flush=True,
            )
        elif args.runtime == "canonical":
            print("[lab] waiting for canonical brain init/prewarm ...", flush=True)
            if not lab.wait_for_brain_ready(timeout_s=180.0):
                raise RuntimeError(
                    "canonical brain did not become ready within 180 s")
            print(
                f"[lab] canonical brain READY: init={lab.brain.stats.get('init_ms')} ms "
                f"prewarm={lab.brain.stats.get('prewarm_ms')} ms; "
                + ((
                       "starting persistent quest/PvE run (F12/Ctrl+C to stop)"
                       if args.quest_autonomy else
                       "starting persistent movement-only run (F12/Ctrl+C to stop)"
                       if args.movement_only_autonomy else
                       "starting persistent passive run (Ctrl+C to stop)")
                   if args.seconds <= 0 else
                   (f"starting {args.seconds}s quest/PvE test"
                    if args.quest_autonomy else
                    f"starting {args.seconds}s movement-only test"
                    if args.movement_only_autonomy else
                    f"starting {args.seconds}s timed smoke test")),
                flush=True,
            )
        if (args.movement_only_autonomy or args.quest_autonomy
                or args.ai_only_autonomy):
            lab.arm_windows_navigation()
            if lab.dashboard_ui is not None:
                print(
                    "[lab] AI-ONLY controls ready; AUTOPILOT DISABLED — "
                    "click ENABLE in DigitalFlyLab; F12 = EMERGENCY STOP"
                    if args.ai_only_autonomy else
                    "[lab] quest/PvE controls ready; AGENT DISABLED — "
                    "click ENABLE in DigitalFlyLab; F12 = EMERGENCY STOP"
                    if args.quest_autonomy else
                    "[lab] navigation controls ready; MOVEMENT DISABLED — "
                    "click ENABLE in DigitalFlyLab; F12 = EMERGENCY STOP",
                    flush=True)
            else:
                print(
                    "[lab] AI-ONLY autopilot armed; Roblox focused; "
                    "F12 = EMERGENCY STOP"
                    if args.ai_only_autonomy else
                    "[lab] quest/PvE agent armed; Roblox focused; "
                    "F12 = EMERGENCY STOP"
                    if args.quest_autonomy else
                    "[lab] navigation armed; Roblox focused; "
                    "F12 = EMERGENCY STOP",
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
