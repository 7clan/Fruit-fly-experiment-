"""Windows.Graphics.Capture adapter (WINDOWS-ONLY).

Uses the pinned `windows-capture` Python package, which wraps the native
Windows.Graphics.Capture API.  Gate 5 remains PASSIVE: this module only
acquires pixels from the explicitly selected game window and never emits
keyboard/mouse input.

One native capture session feeds the bus.  Each callback makes exactly one
owned BGR numpy copy because the package's mapped frame is only guaranteed
for the callback lifetime; that owned array is then shared by reference with
vision/dashboard consumers (copy_count=1).
"""

from __future__ import annotations

import ctypes
import re
import sys
from ctypes import wintypes
from typing import Optional

from ..bus import Bus
from .base import CaptureAdapter, CaptureError


def _visible_top_level_windows() -> list[dict]:
    if sys.platform != "win32":
        return []
    user32 = ctypes.windll.user32
    out: list[dict] = []
    enum_proc_t = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    @enum_proc_t
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value.strip()
        if title:
            out.append({"title": title, "handle": int(hwnd)})
        return True

    if not user32.EnumWindows(callback, 0):
        raise CaptureError("Win32 EnumWindows failed")
    return out


class WindowsGraphicsCaptureAdapter(CaptureAdapter):
    name = "windows_graphics_capture"

    def __init__(self, bus: Bus, channel: str = "capture.frames",
                 window_title_re: str = "Roblox", target_fps: float = 30.0):
        super().__init__(bus, channel)
        self.window_title_re = re.compile(window_title_re, re.IGNORECASE)
        self.target_fps = float(target_fps)
        self._running = False
        self._frame_id = 0
        self._capture = None
        self._control = None
        self._last_error: Optional[str] = None
        self._target: Optional[dict] = None

    def find_target_window(self) -> dict:
        """Fail closed unless exactly one authorized-title candidate exists."""
        matches = []
        for w in _visible_top_level_windows():
            title = w["title"]
            if "DigitalFlyLab" in title:
                continue
            if self.window_title_re.search(title):
                matches.append(w)
        if not matches:
            raise CaptureError(
                f"no visible window matching {self.window_title_re.pattern!r}; "
                "open the authorized GPO game window first")
        if len(matches) != 1:
            desc = ", ".join(f"{w['title']!r} (HWND {w['handle']})"
                             for w in matches)
            raise CaptureError(
                "ambiguous target: expected exactly one matching game window; "
                f"found {len(matches)}: {desc}")
        return matches[0]

    def start(self, target: Optional[dict] = None) -> None:
        if self._running:
            return
        if sys.platform != "win32":
            raise CaptureError("Windows.Graphics.Capture requires Windows")

        desc = target or self.find_target_window()
        hwnd = int(desc.get("handle") or 0)
        title = str(desc.get("title") or "")
        if hwnd <= 0:
            raise CaptureError("target descriptor has no valid HWND")
        if "DigitalFlyLab" in title:
            raise CaptureError("refusing to capture DigitalFlyLab itself")
        if not self.window_title_re.search(title):
            raise CaptureError(
                f"target title {title!r} does not match authorized pattern "
                f"{self.window_title_re.pattern!r}")

        try:
            from windows_capture import (
                Frame, InternalCaptureControl, WindowsCapture)
        except Exception as e:
            raise CaptureError(
                "windows-capture 2.0.1 is required; run setup or the "
                f"Windows capture probe installer first: {e}") from e

        interval_ms = max(1, int(round(1000.0 / max(self.target_fps, 1.0))))
        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            monitor_index=None,
            window_hwnd=hwnd,
            minimum_update_interval=interval_ms,
        )

        @capture.event
        def on_frame_arrived(frame: Frame,
                             capture_control: InternalCaptureControl):
            if not self._running:
                capture_control.stop()
                return
            try:
                # Native mapped frame lifetime ends after callback: one owned
                # copy is necessary and is the only pixel copy at capture.
                img = frame.convert_to_bgr().frame_buffer.copy()
                self._frame_id += 1
                self.publish_frame(
                    self._frame_id,
                    int(frame.width),
                    int(frame.height),
                    data_ref=img,
                    fmt="bgr",
                    copy_count=1,
                    extra={"window_title": title, "window_hwnd": hwnd},
                )
            except Exception as e:
                self._last_error = repr(e)
                self._running = False
                capture_control.stop()

        @capture.event
        def on_closed():
            self._running = False

        self._capture = capture
        self._target = {"title": title, "handle": hwnd}
        self._running = True
        try:
            self._control = capture.start_free_threaded()
        except Exception as e:
            self._running = False
            self._capture = None
            raise CaptureError(f"Windows.Graphics.Capture start failed: {e}") from e

    def stop(self) -> None:
        self._running = False
        ctl = self._control
        self._control = None
        if ctl is not None:
            try:
                ctl.stop()
            except Exception:
                pass
        self._capture = None

    def is_running(self) -> bool:
        return self._running
