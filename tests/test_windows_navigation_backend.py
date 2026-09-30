import ctypes
import platform

import pytest


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only ABI")
def test_sendinput_structure_matches_native_windows_size():
    from lab.action.windows_input import _INPUT, _EXPECTED_INPUT_SIZE
    assert ctypes.sizeof(_INPUT) == _EXPECTED_INPUT_SIZE
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert _EXPECTED_INPUT_SIZE == 40


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only backend")
def test_navigation_backend_hard_allows_only_w_a_d():
    from lab.action.motor_executor import MovementOnlyBackend

    class Fake:
        name = "fake"
        def __init__(self): self.events = []
        def key_down(self, k): self.events.append(("down", k))
        def key_up(self, k): self.events.append(("up", k))
        def mouse_move(self, dx, dy): self.events.append(("mouse", dx, dy))
        def release_all(self): self.events.append(("release",))
        def backend_health(self): return {"ok": True}

    fake = Fake()
    b = MovementOnlyBackend(fake)
    for key in ["W", "A", "D", "S", "SPACE", "F"]:
        b.key_down(key)
    b.mouse_move(50, 0)
    assert fake.events == [("down", "W"), ("down", "A"), ("down", "D")]
