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
import time

if platform.system() != "Windows":  # pragma: no cover
    raise ImportError("windows_input is Windows-only")

INPUT_KEYBOARD = 1
INPUT_MOUSE = 0
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


_ULONG_PTR = ctypes.c_size_t


class _POINT(ctypes.Structure):
    _fields_ = [("x", wt.LONG), ("y", wt.LONG)]


class _RECT(ctypes.Structure):
    _fields_ = [("left", wt.LONG), ("top", wt.LONG),
                ("right", wt.LONG), ("bottom", wt.LONG)]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", _ULONG_PTR)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG),
                ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", _ULONG_PTR)]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wt.DWORD),
                ("wParamL", wt.WORD), ("wParamH", wt.WORD)]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT),
                    ("hi", _HARDWAREINPUT)]
    _fields_ = [("type", wt.DWORD), ("union", _U)]


_EXPECTED_INPUT_SIZE = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
if ctypes.sizeof(_INPUT) != _EXPECTED_INPUT_SIZE:
    raise RuntimeError(
        f"Win32 INPUT ctypes layout mismatch: got {ctypes.sizeof(_INPUT)} "
        f"bytes, expected {_EXPECTED_INPUT_SIZE}")


_user32.SendInput.argtypes = [
    wt.UINT, ctypes.POINTER(_INPUT), ctypes.c_int
]
_user32.SendInput.restype = wt.UINT

_user32.IsWindow.argtypes = [wt.HWND]
_user32.IsWindow.restype = wt.BOOL
_user32.GetForegroundWindow.argtypes = []
_user32.GetForegroundWindow.restype = wt.HWND
_user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
_user32.ShowWindow.restype = wt.BOOL
_user32.SetForegroundWindow.argtypes = [wt.HWND]
_user32.SetForegroundWindow.restype = wt.BOOL
_user32.BringWindowToTop.argtypes = [wt.HWND]
_user32.BringWindowToTop.restype = wt.BOOL
_user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.c_void_p]
_user32.GetWindowThreadProcessId.restype = wt.DWORD
_user32.AttachThreadInput.argtypes = [wt.DWORD, wt.DWORD, wt.BOOL]
_user32.AttachThreadInput.restype = wt.BOOL
_user32.SetWindowPos.argtypes = [
    wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, wt.UINT,
]
_user32.SetWindowPos.restype = wt.BOOL
_user32.SetFocus.argtypes = [wt.HWND]
_user32.SetFocus.restype = wt.HWND
_user32.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(_RECT)]
_user32.GetClientRect.restype = wt.BOOL
_user32.ClientToScreen.argtypes = [wt.HWND, ctypes.POINTER(_POINT)]
_user32.ClientToScreen.restype = wt.BOOL
_user32.GetCursorPos.argtypes = [ctypes.POINTER(_POINT)]
_user32.GetCursorPos.restype = wt.BOOL
_user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
_user32.SetCursorPos.restype = wt.BOOL
_kernel32.GetCurrentThreadId.argtypes = []
_kernel32.GetCurrentThreadId.restype = wt.DWORD


_SCAN = {
    # number row
    "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A, "0": 0x0B,
    # letters
    "Q": 0x10, "W": 0x11, "E": 0x12, "R": 0x13, "T": 0x14,
    "Y": 0x15, "U": 0x16, "I": 0x17, "O": 0x18, "P": 0x19,
    "A": 0x1E, "S": 0x1F, "D": 0x20, "F": 0x21, "G": 0x22,
    "H": 0x23, "J": 0x24, "K": 0x25, "L": 0x26,
    "Z": 0x2C, "X": 0x2D, "C": 0x2E, "V": 0x2F, "B": 0x30,
    "N": 0x31, "M": 0x32,
    # common controls
    "ESC": 0x01, "TAB": 0x0F, "CTRL": 0x1D, "SHIFT": 0x2A,
    "ALT": 0x38, "SPACE": 0x39,
}

class WindowsInputBackend:
    name = "windows_sendinput"

    def __init__(self):
        self._down: set[str] = set()
        self._mouse_down: set[str] = set()
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.last_error = None
        self._target_hwnd = 0

    def _send(self, inp: _INPUT, label: str) -> None:
        ctypes.set_last_error(0)
        self.attempts += 1
        sent = int(_user32.SendInput(
            1, ctypes.byref(inp), ctypes.sizeof(_INPUT)))
        if sent != 1:
            self.failures += 1
            err = int(ctypes.get_last_error())
            self.last_error = (
                f"{label}: SendInput inserted {sent}/1 events; "
                f"GetLastError={err}")
            raise OSError(err or 1, self.last_error)
        self.successes += 1
        self.last_error = None

    def key_down(self, code: str) -> None:
        sc = _SCAN.get(code.upper())
        if sc is None:
            raise ValueError(f"unsupported scan-code key {code!r}")
        inp = _INPUT(type=INPUT_KEYBOARD)
        inp.union.ki = _KEYBDINPUT(0, sc, KEYEVENTF_SCANCODE, 0, 0)
        self._send(inp, f"key_down:{code.upper()}")
        self._down.add(code.upper())

    def key_up(self, code: str) -> None:
        sc = _SCAN.get(code.upper())
        if sc is None:
            raise ValueError(f"unsupported scan-code key {code!r}")
        inp = _INPUT(type=INPUT_KEYBOARD)
        inp.union.ki = _KEYBDINPUT(
            0, sc, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, 0)
        self._send(inp, f"key_up:{code.upper()}")
        self._down.discard(code.upper())

    def mouse_move(self, dx: int, dy: int) -> None:
        inp = _INPUT(type=INPUT_MOUSE)
        inp.union.mi = _MOUSEINPUT(
            int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, 0)
        self._send(inp, f"mouse_move:{int(dx)},{int(dy)}")

    def mouse_down(self, button: str) -> None:
        button = str(button).lower()
        flags = {
            "left": MOUSEEVENTF_LEFTDOWN,
            "right": MOUSEEVENTF_RIGHTDOWN,
            "middle": MOUSEEVENTF_MIDDLEDOWN,
        }
        if button not in flags:
            raise ValueError(f"unsupported mouse button {button!r}")
        inp = _INPUT(type=INPUT_MOUSE)
        inp.union.mi = _MOUSEINPUT(0, 0, 0, flags[button], 0, 0)
        self._send(inp, f"mouse_down:{button}")
        self._mouse_down.add(button)

    def mouse_up(self, button: str) -> None:
        button = str(button).lower()
        flags = {
            "left": MOUSEEVENTF_LEFTUP,
            "right": MOUSEEVENTF_RIGHTUP,
            "middle": MOUSEEVENTF_MIDDLEUP,
        }
        if button not in flags:
            raise ValueError(f"unsupported mouse button {button!r}")
        inp = _INPUT(type=INPUT_MOUSE)
        inp.union.mi = _MOUSEINPUT(0, 0, 0, flags[button], 0, 0)
        self._send(inp, f"mouse_up:{button}")
        self._mouse_down.discard(button)

    def mouse_click(self, button: str) -> None:
        self.mouse_down(button)
        self.mouse_up(button)

    def _client_screen_rect(self):
        hwnd = int(self._target_hwnd or 0)
        if hwnd <= 0 or not bool(_user32.IsWindow(wt.HWND(hwnd))):
            return None
        rect = _RECT()
        if not bool(_user32.GetClientRect(wt.HWND(hwnd), ctypes.byref(rect))):
            return None
        origin = _POINT(0, 0)
        if not bool(_user32.ClientToScreen(
                wt.HWND(hwnd), ctypes.byref(origin))):
            return None
        width = max(1, int(rect.right - rect.left))
        height = max(1, int(rect.bottom - rect.top))
        return (
            int(origin.x), int(origin.y),
            int(origin.x + width - 1), int(origin.y + height - 1),
        )

    def center_cursor_in_target(self) -> bool:
        bounds = self._client_screen_rect()
        if bounds is None:
            return False
        left, top, right, bottom = bounds
        cx = int((left + right) // 2)
        cy = int(top + (bottom - top) * 0.48)
        return bool(_user32.SetCursorPos(cx, cy))

    def camera_drag(self, dx: int, dy: int = 0) -> None:
        """Roblox/GPO camera look without letting the cursor escape Roblox.

        Relative SendInput starts wherever the OS cursor currently sits. On a
        small window that previously pushed the pointer onto the desktop. For
        autonomous camera look, always start from the authorized Roblox client
        center, clamp the drag to a fraction of that client, then recenter.
        """
        bounds = self._client_screen_rect()
        if bounds is not None:
            left, top, right, bottom = bounds
            width = max(1, right - left + 1)
            height = max(1, bottom - top + 1)
            dx = max(-int(width * 0.18), min(int(width * 0.18), int(dx)))
            dy = max(-int(height * 0.14), min(int(height * 0.14), int(dy)))
            self.center_cursor_in_target()
        else:
            dx = max(-140, min(140, int(dx)))
            dy = max(-90, min(90, int(dy)))

        self.mouse_down("right")
        try:
            self.mouse_move(int(dx), int(dy))
        finally:
            self.mouse_up("right")
            # Leave the pointer in a deterministic safe place inside Roblox.
            self.center_cursor_in_target()

    # Compatibility names used by MotorExecutor/InputBackend.
    def mouse_button_down(self, button: str) -> None:
        self.mouse_down(button)

    def mouse_button_up(self, button: str) -> None:
        self.mouse_up(button)

    def set_target_window(self, hwnd: int) -> None:
        hwnd = int(hwnd or 0)
        if hwnd > 0 and bool(_user32.IsWindow(wt.HWND(hwnd))):
            self._target_hwnd = hwnd
        else:
            self._target_hwnd = 0

    def ui_click(self, x_norm: float, y_norm: float,
                 button: str = "left", restore_cursor: bool = True) -> None:
        """Click a normalized point inside the authorized Roblox client.

        This is deliberately separate from camera movement. It is intended
        only for an explicitly identified in-game UI button. The user's
        cursor position is restored immediately after the click.
        """
        hwnd = int(self._target_hwnd or 0)
        if hwnd <= 0 or not bool(_user32.IsWindow(wt.HWND(hwnd))):
            raise RuntimeError("UI click blocked: authorized Roblox HWND missing")
        if int(_user32.GetForegroundWindow() or 0) != hwnd:
            raise RuntimeError("UI click blocked: Roblox is not foreground")

        x_norm = max(0.0, min(1.0, float(x_norm)))
        y_norm = max(0.0, min(1.0, float(y_norm)))
        rect = _RECT()
        if not bool(_user32.GetClientRect(wt.HWND(hwnd), ctypes.byref(rect))):
            raise RuntimeError("UI click blocked: GetClientRect failed")
        origin = _POINT(0, 0)
        if not bool(_user32.ClientToScreen(
                wt.HWND(hwnd), ctypes.byref(origin))):
            raise RuntimeError("UI click blocked: ClientToScreen failed")
        width = max(1, int(rect.right - rect.left))
        height = max(1, int(rect.bottom - rect.top))
        sx = int(origin.x + x_norm * (width - 1))
        sy = int(origin.y + y_norm * (height - 1))

        old = _POINT()
        have_old = bool(_user32.GetCursorPos(ctypes.byref(old)))
        if not bool(_user32.SetCursorPos(sx, sy)):
            raise RuntimeError("UI click blocked: SetCursorPos failed")
        try:
            self.mouse_click(button)
        finally:
            if restore_cursor and have_old:
                bounds = self._client_screen_rect()
                if bounds is not None:
                    left, top, right, bottom = bounds
                    if (left <= int(old.x) <= right
                            and top <= int(old.y) <= bottom):
                        _user32.SetCursorPos(int(old.x), int(old.y))
                    else:
                        self.center_cursor_in_target()
                else:
                    _user32.SetCursorPos(int(old.x), int(old.y))

    def release_all(self) -> None:
        for code in list(self._down):
            self.key_up(code)
        for button in list(self._mouse_down):
            self.mouse_button_up(button)
        self._down.clear()
        self._mouse_down.clear()

    def backend_health(self) -> dict:
        return {
            "backend": self.name,
            "ok": self.failures == 0,
            "sendinput_attempts": self.attempts,
            "sendinput_successes": self.successes,
            "sendinput_failures": self.failures,
            "last_error": self.last_error,
            "held_keys": sorted(self._down),
            "held_mouse_buttons": sorted(self._mouse_down),
            "input_struct_size": ctypes.sizeof(_INPUT),
            "input_struct_expected": _EXPECTED_INPUT_SIZE,
            "target_hwnd": int(self._target_hwnd or 0),
            "ui_click_supported": True,
        }


def focus_window(hwnd: int) -> bool:
    """Bring the authorized target game window to the foreground.

    Windows can reject SetForegroundWindow transiently even when the target
    is valid (foreground-lock rules). Treat an already-foreground window as
    success, retry briefly, then use AttachThreadInput only as a bounded
    fallback. No synthetic Alt/key press is used here.
    """
    try:
        hwnd = int(hwnd)
        if hwnd <= 0 or not bool(_user32.IsWindow(wt.HWND(hwnd))):
            return False

        SW_RESTORE = 9
        HWND_TOP = wt.HWND(0)
        SWP_NOSIZE = 0x0001
        SWP_NOMOVE = 0x0002
        SWP_SHOWWINDOW = 0x0040

        def is_foreground() -> bool:
            return int(_user32.GetForegroundWindow() or 0) == hwnd

        _user32.ShowWindow(wt.HWND(hwnd), SW_RESTORE)
        if is_foreground():
            return True

        for _ in range(3):
            _user32.BringWindowToTop(wt.HWND(hwnd))
            _user32.SetForegroundWindow(wt.HWND(hwnd))
            if is_foreground():
                return True
            time.sleep(0.05)

        # Bounded fallback for foreground-lock situations. Attach only the
        # current and foreground GUI threads and always detach in finally.
        fg = int(_user32.GetForegroundWindow() or 0)
        cur_tid = int(_kernel32.GetCurrentThreadId())
        fg_tid = int(_user32.GetWindowThreadProcessId(
            wt.HWND(fg), None)) if fg else 0
        attached = False
        try:
            if fg_tid and fg_tid != cur_tid:
                attached = bool(_user32.AttachThreadInput(
                    wt.DWORD(cur_tid), wt.DWORD(fg_tid), True))
            _user32.SetWindowPos(
                wt.HWND(hwnd), HWND_TOP, 0, 0, 0, 0,
                SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW)
            _user32.BringWindowToTop(wt.HWND(hwnd))
            _user32.SetForegroundWindow(wt.HWND(hwnd))
            _user32.SetFocus(wt.HWND(hwnd))
        finally:
            if attached:
                _user32.AttachThreadInput(
                    wt.DWORD(cur_tid), wt.DWORD(fg_tid), False)

        return is_foreground()
    except Exception:
        return False


def function_key_pressed(number: int) -> bool:
    """Poll F1..F12 globally, independent of dashboard responsiveness."""
    number = int(number)
    if not 1 <= number <= 12:
        return False
    # VK_F1=0x70 ... VK_F12=0x7B
    vk = 0x6F + number
    try:
        return bool(_user32.GetAsyncKeyState(vk) & 0x8000)
    except Exception:
        return False


def f12_pressed() -> bool:
    return function_key_pressed(12)
