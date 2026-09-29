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
import threading
import time
from ctypes import wintypes
from typing import Optional

from ..bus import Bus
from .base import CaptureAdapter, CaptureError


def _process_image_name(pid: int) -> str:
    """Return the executable basename for a PID, or empty string."""
    if sys.platform != "win32":
        return ""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD, wintypes.BOOL, wintypes.DWORD
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD)
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
    )
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(
                handle, 0, buf, ctypes.byref(size)):
            return ""
        return buf.value.rsplit("\\", 1)[-1]
    finally:
        kernel32.CloseHandle(handle)


def _visible_top_level_windows() -> list[dict]:
    """Enumerate visible titled windows with 64-bit-safe Win32 signatures."""
    if sys.platform != "win32":
        return []

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_proc_t = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    user32.EnumWindows.argtypes = [enum_proc_t, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [
        wintypes.HWND, wintypes.LPWSTR, ctypes.c_int
    ]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND, ctypes.POINTER(wintypes.DWORD)
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    out: list[dict] = []

    @enum_proc_t
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        if user32.GetWindowTextW(hwnd, buf, n + 1) <= 0:
            return True
        title = buf.value.strip()
        if not title:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        out.append({
            "title": title,
            "handle": int(hwnd),
            "pid": int(pid.value),
            "process_name": _process_image_name(int(pid.value)),
        })
        return True

    if not user32.EnumWindows(callback, 0):
        err = ctypes.get_last_error()
        raise CaptureError(f"Win32 EnumWindows failed (error {err})")
    return out


class WindowsGraphicsCaptureAdapter(CaptureAdapter):
    name = "windows_graphics_capture"

    def __init__(self, bus: Bus, channel: str = "capture.frames",
                 window_title_re: str = r"^Roblox$",
                 process_name_re: str = r"^RobloxPlayerBeta(?:\.exe)?$",
                 target_fps: float = 30.0):
        super().__init__(bus, channel)
        self.window_title_re = re.compile(window_title_re, re.IGNORECASE)
        self.process_name_re = re.compile(process_name_re, re.IGNORECASE)
        self.target_fps = float(target_fps)
        self._running = False
        self._frame_id = 0
        self._capture = None
        self._control = None
        self._last_error: Optional[str] = None
        self._target: Optional[dict] = None
        self._last_publish_mono = 0.0
        self._publish_throttle_lock = threading.Lock()

    def find_target_window(self) -> dict:
        """Fail closed unless exactly one authorized-title candidate exists."""
        matches = []
        for w in _visible_top_level_windows():
            title = w["title"]
            proc = w.get("process_name", "")
            if "DigitalFlyLab" in title:
                continue
            if (self.window_title_re.search(title)
                    and self.process_name_re.search(proc)):
                matches.append(w)
        if not matches:
            raise CaptureError(
                "no visible authorized game window found matching "
                f"title={self.window_title_re.pattern!r}, "
                f"process={self.process_name_re.pattern!r}; "
                "open the GPO game window first")
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

        # Older Windows.Graphics.Capture implementations (including
        # Windows 10 builds) may reject attempts to toggle optional session
        # properties such as border/cursor capture.  Passing None leaves
        # those properties at the platform default instead of calling the
        # unsupported setter.  We also leave minimum_update_interval unset;
        # the bus is bounded/latest-wins, so excess frames are dropped rather
        # than queued.
        capture = WindowsCapture(
            cursor_capture=None,
            draw_border=None,
            secondary_window=None,
            dirty_region=None,
            monitor_index=None,
            window_hwnd=hwnd,
            minimum_update_interval=None,
        )

        @capture.event
        def on_frame_arrived(frame: Frame,
                             capture_control: InternalCaptureControl):
            if not self._running:
                capture_control.stop()
                return
            now = time.perf_counter()
            min_dt = 1.0 / max(self.target_fps, 0.1)
            # windows-capture may deliver callbacks concurrently. Make the
            # admission check atomic so target_fps is a real upper bound
            # rather than a racy hint.
            with self._publish_throttle_lock:
                if (self._last_publish_mono
                        and now - self._last_publish_mono < min_dt):
                    return
                self._last_publish_mono = now
            try:
                # Best measured full-resolution path on the target
                # Windows laptop: OpenCV performs the required BGRA->BGR
                # ownership copy in optimized native code.
                import cv2
                img = cv2.cvtColor(
                    frame.frame_buffer, cv2.COLOR_BGRA2BGR)
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
