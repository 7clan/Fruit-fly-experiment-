"""Human teacher input recorder for Windows gameplay demonstrations.

Records ONLY a bounded catalog of GPO gameplay controls, and only while the
authorized game window is foreground. It never injects input, records text, or
captures unrelated keyboard activity.

The resulting state is joined with perception by TrajectoryRecorder so later
behavior cloning / DAgger can learn from human corrections.
"""

from __future__ import annotations

import platform

from ..worker import Worker


class TeacherInputRecorder(Worker):
    name = "teacher_input_recorder"

    # Virtual-key codes for the project's verified gameplay controls.
    _VK = {
        "W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44,
        "Q": 0x51, "F": 0x46, "T": 0x54, "E": 0x45, "R": 0x52,
        "Z": 0x5A, "X": 0x58, "C": 0x43, "V": 0x56, "B": 0x42,
        "N": 0x4E, "G": 0x47, "J": 0x4A, "P": 0x50, "M": 0x4D,
        "SPACE": 0x20, "CTRL": 0x11, "SHIFT": 0x10,
        "0": 0x30, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
        "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39,
    }

    def __init__(self, bus, target_hwnd_getter=None,
                 target_hz: float = 20.0):
        super().__init__(bus, target_hz=target_hz)
        self.state = bus.state("teacher.action")
        self.events = bus.stream("teacher.events", maxsize=128)
        self.target_hwnd_getter = target_hwnd_getter or (lambda: 0)
        self._user32 = None
        self._prev = None
        self.stats.update({
            "samples": 0,
            "focused_samples": 0,
            "action_changes": 0,
            "unfocused_samples": 0,
        })

    def on_start(self) -> None:
        if platform.system() != "Windows":
            raise RuntimeError("TeacherInputRecorder requires Windows")
        import ctypes
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self._user32.GetAsyncKeyState.restype = ctypes.c_short
        self._user32.GetForegroundWindow.argtypes = []
        self._user32.GetForegroundWindow.restype = ctypes.c_void_p

    def _down(self, vk: int) -> bool:
        return bool(int(self._user32.GetAsyncKeyState(int(vk))) & 0x8000)

    @staticmethod
    def _labels(keys, left, right):
        out = []
        k = set(keys)
        if "W" in k:
            out.append("move_forward")
        if "S" in k:
            out.append("move_backward")
        if "A" in k:
            out.append("move_left")
        if "D" in k:
            out.append("move_right")
        if "Q" in k:
            out.append("dash_or_evade")
        if "F" in k:
            out.append("block")
        if "SPACE" in k:
            out.append("jump")
        if "CTRL" in k:
            out.append("climb_or_context_ctrl")
        if left:
            out.append("m1")
        if right:
            out.append("camera_drag")
        for ability in ("E", "R", "Z", "X", "C", "V", "B", "N", "G", "J"):
            if ability in k:
                out.append("key_" + ability.lower())
        for slot in "0123456789":
            if slot in k:
                out.append("equip_" + slot)
        return out

    def step(self) -> None:
        hwnd = int(self.target_hwnd_getter() or 0)
        foreground = int(self._user32.GetForegroundWindow() or 0)
        focused = bool(hwnd > 0 and foreground == hwnd)
        keys = []
        left = right = False
        if focused:
            keys = [name for name, vk in self._VK.items() if self._down(vk)]
            left = self._down(0x01)
            right = self._down(0x02)
            self.stats["focused_samples"] += 1
        else:
            self.stats["unfocused_samples"] += 1

        actions = self._labels(keys, left, right) if focused else []
        payload = {
            "ts_ns": self.clock.now_ns(),
            "enabled": True,
            "focused": focused,
            "target_hwnd": hwnd,
            "keys_down": keys,
            "mouse_left": left,
            "mouse_right": right,
            "actions": actions,
            "source": "human_teacher",
            "records_only_verified_game_controls": True,
        }
        self.state.write(payload, ts_ns=payload["ts_ns"])
        sig = (focused, tuple(keys), left, right)
        if sig != self._prev:
            self.events.publish(payload, ts_ns=payload["ts_ns"])
            self.stats["action_changes"] += 1
            self._prev = sig
        self.stats["samples"] += 1
