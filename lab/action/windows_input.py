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
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040

_user32 = ctypes.WinDLL("user32", use_last_error=True)


_ULONG_PTR = ctypes.c_size_t


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
    # modifiers / utility
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

    def mouse_button_down(self, button: str) -> None:
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

    def mouse_button_up(self, button: str) -> None:
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
        }


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
