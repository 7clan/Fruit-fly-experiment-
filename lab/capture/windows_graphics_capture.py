"""Windows.Graphics.Capture adapter (WINDOWS-ONLY, guarded import).

FINAL_ARCHITECTURE.md §9: prefer Windows.Graphics.Capture for acquiring
frames from the Roblox application window. ONE stream feeds computer
vision AND the dashboard mirror. Minimize unnecessary CPU memory copies
where feasible.

Implementation notes (honest, pre-implementation skeleton):
  * Uses the winsdk (Python/WinRT) projection of
    Windows.Graphics.Capture.GraphicsCaptureItem +
    Direct3D11CaptureFramePool. Frame arrival is an event callback; we
    copy the surface ONCE into a numpy buffer (copy_count=1 per frame)
    and publish it. A zero-copy shared-memory path (sensor fingerprint
    of the D3D texture to a shared handle) is a later optimization —
    benchmark first (benchmark_windows.py), optimize second.
  * Window enumeration: use window title match restricted to the
    authorized target (GPO/Roblox). The app-level policy (ONLY the
    authorized game, docs/GPO_PLAN.md §0) is enforced by the caller
    passing the right window descriptor; this adapter still refuses
    captures of windows whose title matches DigitalFlyLab itself.
  * Needs Windows 10 1903+ for the programmatic API; self-test reports
    the OS build and whether border/window enumeration permission works.

This module is NOT imported by tests/sandbox code; it is only imported
inside create_windows_capture() after availability checks.
"""

from __future__ import annotations

import re
from typing import Optional

from ..bus import Bus
from ..clock import SHARED_CLOCK
from .base import CaptureAdapter, CaptureError


class WindowsGraphicsCaptureAdapter(CaptureAdapter):
    name = "windows_graphics_capture"

    def __init__(self, bus: Bus, channel: str = "capture.frames",
                 window_title_re: str = "Roblox", target_fps: float = 30.0):
        super().__init__(bus, channel)
        self.window_title_re = re.compile(window_title_re, re.IGNORECASE)
        self.target_fps = float(target_fps)
        self._running = False
        self._frame_id = 0
        self._item = None
        self._pool = None
        self._session = None
        self._last_error: Optional[str] = None

    # -- window discovery ------------------------------------------------
    def find_target_window(self) -> dict:
        """Enumerate top-level windows; return descriptor of the first
        title match (excluding DigitalFlyLab's own windows)."""
        try:
            import winsdk.windows.ui.shell as _shell  # type: ignore
        except Exception as e:  # pragma: no cover - windows only
            raise CaptureError(f"winsdk shell enumeration unavailable: {e}")
        found = None
        try:
            items = _shell.FindAllWindows()  # IShellWindow-like helper
        except Exception as e:  # pragma: no cover
            raise CaptureError(f"window enumeration failed: {e}")
        for w in items:
            title = getattr(w, "Title", "") or ""
            if "DigitalFlyLab" in title:
                continue
            if self.window_title_re.search(title):
                found = {"title": title, "handle": getattr(w, "HWND", None)}
                break
        if not found:
            raise CaptureError(
                f"no window matching {self.window_title_re.pattern!r} — "
                "is the authorized GPO target running?")
        return found

    # -- lifecycle ---------------------------------------------------------
    def start(self, target: Optional[dict] = None) -> None:
        """Start capture. target: {"kind":"window","title_re":...} or the
        descriptor returned by find_target_window()."""
        if self._running:
            return
        desc = target or self.find_target_window()
        try:
            import asyncio
            import winsdk.windows.graphics.capture as wgc
            import winsdk.windows.graphics.directx as d3d
            import numpy as np

            async def _create():
                interop = (await wgc.GraphicsCaptureItem
                           .CreateAsync)  # placeholder; see winsdk docs
                return None
            # NOTE: full WinRT item creation from HWND uses
            # GraphicsCaptureItemInterop (statics). winsdk exposes it via
            # winsdk.windows.graphics.capture.GraphicsCaptureItem — the
            # exact interop call is exercised by the Windows self-test
            # (DigitalFlyLab --self-test) and documented there. This
            # skeleton keeps the architecture honest: construction,
            # frame-pool sizing, single numpy copy per frame, publish.
            self._last_error = "windows capture backend pending on-device wiring"
            raise CaptureError(self._last_error)
        except CaptureError:
            raise
        except Exception as e:  # pragma: no cover
            raise CaptureError(f"windows capture start failed: {e}")

    def stop(self) -> None:
        self._running = False
        try:
            if self._session is not None:
                self._session.Close()
        except Exception:
            pass
        self._pool = None
        self._item = None

    def is_running(self) -> bool:
        return self._running
