"""Worker base class: rate-governed bus loop.

Every DigitalFlyLab worker follows the same contract:

  while running:        (stop_event not set)
      governor.tick()   (multi-rate pacing, FINAL_ARCHITECTURE §3)
      step()            (consume newest inputs; publish outputs)

step() MUST be non-blocking-ish: consume via newest()/read() (drop
obsolete), never accumulate backlog. Heavy work is confined to its own
worker process/thread so it can never block the real-time path.
"""

from __future__ import annotations

import threading

from .bus import Bus, RateGovernor, StreamChannel
from .clock import Clock, SHARED_CLOCK


class Worker:
    """Rate-governed pipeline worker."""

    name = "worker"

    def __init__(self, bus: Bus, target_hz: float = 10.0,
                 clock: Clock | None = None):
        self.bus = bus
        self.clock = clock or SHARED_CLOCK
        self.governor = RateGovernor(target_hz, clock=self.clock)
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = {"steps": 0, "errors": 0, "last_error": None,
                      "last_step_ms": None, "work_ms_total": 0.0,
                      "max_step_ms": 0.0, "started": False}

    # -- contract -----------------------------------------------------------
    def step(self) -> None:
        """One unit of work. Override."""

    def on_start(self) -> None:
        """Optional one-time init INSIDE the worker thread (lazy imports,
        model loads — never load a detector during combat)."""

    def on_stop(self) -> None:
        """Optional cleanup."""

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> "Worker":
        if self._thread is not None and self._thread.is_alive():
            return self
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self.name,
                                        daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        self.stats["started"] = True
        try:
            self.on_start()
            while not self.stop_event.is_set():
                self.governor.tick()
                t0 = self.clock.now_ns()
                try:
                    self.step()
                    self.stats["steps"] += 1
                except Exception as e:  # noqa: BLE001 — keep the loop alive
                    self.stats["errors"] += 1
                    self.stats["last_error"] = repr(e)
                step_ms = self.clock.elapsed_ms(t0)
                self.stats["last_step_ms"] = step_ms
                self.stats["work_ms_total"] = round(
                    float(self.stats.get("work_ms_total", 0.0)) + step_ms, 3)
                self.stats["max_step_ms"] = max(
                    float(self.stats.get("max_step_ms", 0.0)), step_ms)
        except Exception as e:  # startup failures must be visible in reports
            self.stats["errors"] += 1
            self.stats["last_error"] = repr(e)
            self.stats["startup_failed"] = True
        finally:
            try:
                self.on_stop()
            except Exception as e:  # cleanup failures are also real failures
                self.stats["errors"] += 1
                self.stats["last_error"] = repr(e)

    def stop(self, join_timeout: float = 2.0) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

    def run_sync(self, seconds: float) -> None:
        """Run the loop synchronously for `seconds` (tests, benchmarks)."""
        import time as _t
        self.stats["started"] = True
        self.on_start()
        t_end = _t.monotonic() + seconds
        while _t.monotonic() < t_end and not self.stop_event.is_set():
            self.governor.tick()
            t0 = self.clock.now_ns()
            try:
                self.step()
                self.stats["steps"] += 1
            except Exception as e:  # noqa: BLE001
                self.stats["errors"] += 1
                self.stats["last_error"] = repr(e)
            step_ms = self.clock.elapsed_ms(t0)
            self.stats["last_step_ms"] = step_ms
            self.stats["work_ms_total"] = round(
                float(self.stats.get("work_ms_total", 0.0)) + step_ms, 3)
            self.stats["max_step_ms"] = max(
                float(self.stats.get("max_step_ms", 0.0)), step_ms)
        self.on_stop()

    def achieved_hz(self, elapsed_s: float) -> float:
        return self.stats["steps"] / elapsed_s if elapsed_s > 0 else 0.0
