"""Windows SendInput backend (WINDOWS-ONLY, guarded import).

FINAL_ARCHITECTURE §19 + WINDOWS_RUNTIME_SPEC W5: all emitted actions
are logged; a GLOBAL EMERGENCY STOP HOTKEY releases all held keys
immediately. Uses the Win32 SendInput API via ctypes (no external
dependency): key-down/up with scan codes, relative mouse movement.

Contract notes (wired on the Windows box):
  * the emergency-stop hotkey is registered by the app layer
    (DigitalFlyLab) and calls executor.emergency_stop() — the backend
    itself only performs release_all();
  * every injection passes through MotorExecutor (timestamped, logged,
    autonomy-gated). NO other component may inject input (spec §25).
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import platform

if platform.system() != "Windows":  # pragma: no cover
    raise ImportError("windows_input is Windows-only")

INPUT_KEYBOARD = 1
INPUT_MOUSE = 0
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001

_user32 = ctypes.WinDLL("user32", use_last_error=True)


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]
    _fields_ = [("type", wt.DWORD), ("union", _U)]


_SCAN = {  # virtual key -> scan code (subset used by the basic movement map)
    "W": 0x11, "A": 0x1E, "S": 0x1F, "D": 0x20, "SPACE": 0x39,
    "Q": 0x10, "E": 0x12, "R": 0x13, "F": 0x21, "Z": 0x2C, "X": 0x2D,
    "C": 0x2E, "V": 0x2F, "SHIFT": 0x2A, "CTRL": 0x1D,
}


class WindowsInputBackend:
    name = "windows_sendinput"

    def __init__(self):
        self._down: set[str] = set()

    def key_down(self, code: str) -> None:
        sc = _SCAN.get(code.upper())
        if sc is None:
            return
        inp = _INPUT(type=INPUT_KEYBOARD)
        inp.union.ki = _KEYBDINPUT(0, sc, KEYEVENTF_SCANCODE)
        _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        self._down.add(code.upper())

    def key_up(self, code: str) -> None:
        sc = _SCAN.get(code.upper())
        if sc is None:
            return
        inp = _INPUT(type=INPUT_KEYBOARD)
        inp.union.ki = _KEYBDINPUT(0, sc, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP)
        _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        self._down.discard(code.upper())

    def mouse_move(self, dx: int, dy: int) -> None:
        inp = _INPUT(type=INPUT_MOUSE)
        inp.union.mi = _MOUSEINPUT(int(dx), int(dy), MOUSEEVENTF_MOVE)
        _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

    def release_all(self) -> None:
        for code in list(self._down):
            self.key_up(code)
        self._down.clear()


def focus_window(hwnd: int) -> bool:
    """Bring the authorized target game window to the foreground."""
    try:
        hwnd = int(hwnd)
        if hwnd <= 0:
            return False
        SW_RESTORE = 9
        _user32.ShowWindow(wt.HWND(hwnd), SW_RESTORE)
        return bool(_user32.SetForegroundWindow(wt.HWND(hwnd)))
    except Exception:
        return False


def f12_pressed() -> bool:
    """Global emergency-stop poll usable even while the game has focus."""
    VK_F12 = 0x7B
    try:
        return bool(_user32.GetAsyncKeyState(VK_F12) & 0x8000)
    except Exception:
        return False
