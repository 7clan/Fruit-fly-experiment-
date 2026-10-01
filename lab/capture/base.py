"""Capture adapter interface + registry.

FINAL_ARCHITECTURE.md §1/§9 + docs/WINDOWS_RUNTIME_SPEC.md W4:
  * Prefer Windows.Graphics.Capture for frames from the Roblox window.
  * Roblox stays its own normal Windows process (never embed the renderer).
  * ONE capture stream feeds BOTH computer vision AND the dashboard mirror
    — never capture the same window twice.
  * Minimize unnecessary CPU memory copies where feasible.

This module defines the platform-neutral interface:

  CaptureAdapter.start(window) / stop()
  CaptureAdapter.frame_stream()  -> StreamChannel ("capture.frames")

Frame payload contract (JSON-safe):
  {frame_id, capture_ts_ns, width, height, format, data_ref, copy_count}

`data_ref` is an opaque reference to the pixel buffer (numpy array in
process; shared-memory handle cross-process). Vision consumes it; the
dashboard mirror consumes the SAME frames via its own subscription to
the same channel — one stream, many readers of the same published
envelope (the bus never duplicates the buffer).

Available adapters:
  SyntheticCapture     deterministic synthetic game-like frames (dev,
                       sandbox, tests, benchmark dry-runs) — labeled
                       SYNTHETIC in every frame payload.
  WindowsGraphicsCaptureAdapter  Windows.Graphics.Capture (guarded import;
                       only constructible on Windows with the winsdk
                       package present; self-test documented).
  NullCapture          zero frames (control condition / CI).
"""

from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..bus import Bus, StateChannel, StreamChannel
from ..clock import SHARED_CLOCK


class CaptureError(RuntimeError):
    pass


class CaptureAdapter(ABC):
    """Base class. Subclasses implement platform-specific acquisition."""

    name = "abstract"

    def __init__(self, bus: Bus, channel: str = "capture.frames"):
        self.bus = bus
        self.channel_name = channel
        # Legacy event stream kept for compatibility, but only one frame is
        # retained. All live consumers use the non-destructive latest state.
        # Keeping four 1080p frames alive wastes tens of MB on the 8 GB
        # target laptop with no benefit.
        self.frames: StreamChannel = bus.stream(channel, maxsize=1)
        # Non-destructive latest-frame snapshot for multiple readers.
        # A queue cannot be shared by vision/dashboard as a broadcast:
        # one consumer would drain frames away from the others.
        self.latest: StateChannel = bus.state(channel + ".latest")

    # -- lifecycle ----------------------------------------------------------
    @abstractmethod
    def start(self, target: Optional[dict] = None) -> None:
        """Begin acquiring frames. target: {"kind": "window", "title_re":
        "..."} or similar window descriptor (adapter-specific)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop acquisition and release resources."""

    def is_running(self) -> bool:
        return False

    # -- helpers -------------------------------------------------------------
    def publish_frame(self, frame_id: int, width: int, height: int,
                      data_ref, fmt: str = "bgr", copy_count: int = 0,
                      extra: Optional[dict] = None) -> dict:
        ts = SHARED_CLOCK.now_ns()
        payload = {
            "frame_id": frame_id,
            "capture_ts_ns": ts,
            "width": width,
            "height": height,
            "format": fmt,
            "data_ref": data_ref,
            "copy_count": copy_count,          # honest memory-copy accounting
            "source": self.name,
        }
        if extra:
            payload.update(extra)
        self.frames.publish(payload, ts_ns=ts)
        self.latest.write(payload, ts_ns=ts)
        return payload


class NullCapture(CaptureAdapter):
    """Emits no frames. Used for CI and control conditions."""
    name = "null"

    def start(self, target=None):
        pass

    def stop(self):
        pass


class SyntheticCapture(CaptureAdapter):
    """Deterministic synthetic game-like frame source (SYNTHETIC-labeled).

    Generates simple composited frames (via numpy) containing a player
    marker, a moving target, and optional approaching enemy marker with
    attack wind-up flashes — enough to exercise the perception stack,
    latency instrumentation and dashboard WITHOUT any game or screen.

    Frames are (H, W, 3) uint8 numpy arrays attached as data_ref.
    """

    name = "synthetic"

    def __init__(self, bus: Bus, width: int = 640, height: int = 360,
                 fps: float = 30.0, channel: str = "capture.frames"):
        super().__init__(bus, channel)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._running = False
        self._frame_id = 0

    def start(self, target=None):
        self._running = True

    def stop(self):
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def make_frame(self, t_s: float) -> "object":
        """Render one synthetic frame at time t (deterministic)."""
        import numpy as np
        w, h = self.width, self.height
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (24, 24, 28)                       # dark "game" bg
        # player: centered bottom marker
        cv_y, cv_x = int(h * 0.62), int(w * 0.5)
        img[cv_y - 8:cv_y + 8, cv_x - 8:cv_x + 8] = (60, 200, 60)
        # target: orbits the player slowly (quest marker analogue)
        ang = 0.6 * t_s
        tx = int(cv_x + 0.32 * w * __import__("math").cos(ang))
        ty = int(cv_y - 0.18 * h * abs(__import__("math").sin(ang)))
        img[max(0, ty - 10):ty + 10, max(0, tx - 10):tx + 10] = (60, 180, 255)
        # enemy: approaches from right, flashes when "attacking"
        ex = int(w * (0.95 - 0.05 * ((t_s * 0.35) % 6.0)))
        attacking = ((t_s * 0.35) % 6.0) > 5.0
        ey = int(h * 0.45)
        color = (80, 60, 255) if not attacking else (255, 60, 60)
        img[ey - 12:ey + 12, ex - 12:ex + 12] = color
        return img

    def tick(self, t_s: float) -> dict:
        """Render + publish one frame at simulated time t_s."""
        if not self._running:
            raise CaptureError("synthetic capture not started")
        img = self.make_frame(t_s)
        self._frame_id += 1
        return self.publish_frame(self._frame_id, self.width, self.height,
                                  data_ref=img, fmt="bgr", copy_count=0,
                                  extra={"synthetic": True, "t_s": t_s,
                                         "enemy_attacking": attacking_flag(t_s)})


def attacking_flag(t_s: float) -> bool:
    """Shared synthetic-enemy wind-up schedule (used by ground-truth tests)."""
    return ((t_s * 0.35) % 6.0) > 5.0


# ---------------------------------------------------------------------------
# Windows adapter (guarded)
# ---------------------------------------------------------------------------

def windows_capture_available() -> bool:
    """True on Windows with the winsdk/WinRT capture stack importable."""
    if platform.system() != "Windows":
        return False
    try:
        import windows_capture  # noqa: F401
        return True
    except Exception:
        return False


def create_windows_capture(bus: Bus, channel: str = "capture.frames",
                           window_title_re: str = "Roblox",
                           target_fps: float = 30.0) -> CaptureAdapter:
    """Factory for the Windows.Graphics.Capture adapter.

    WindowsGraphicsCaptureAdapter implementation lives in
    lab/capture/windows_graphics_capture.py and is ONLY importable on
    Windows; this factory fails closed with a clear message otherwise.
    """
    if not windows_capture_available():
        raise CaptureError(
            "Windows.Graphics.Capture unavailable (platform=%s, "
            "windows-capture dependency missing); use SyntheticCapture "
            "for non-Windows development"
            % platform.system())
    from .windows_graphics_capture import WindowsGraphicsCaptureAdapter
    return WindowsGraphicsCaptureAdapter(
        bus, channel=channel, window_title_re=window_title_re,
        target_fps=target_fps)
