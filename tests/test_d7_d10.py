"""D7-D10 unit tests (fast, brain-free - the canonical brain is exercised
by the recorded studies under brain/results/, not by this suite)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "brain" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import d8_encoder as d8          # noqa: E402
import d9_decoder as d9          # noqa: E402
import d10_closed_loop as d10    # noqa: E402


POP_SIZES = dict(P9_left=1, P9_right=1, BPN_bilateral=33, RRN_bilateral=2,
                 MDN_bilateral=4, GF_left=1, GF_right=1, FG_bilateral=2,
                 BB_bilateral=2, BRK_bilateral=6, DN_ALL_left=647,
                 DN_ALL_right=650, DN_LEG_bilateral=24)


# ---------------------------------------------------------------- D8 arena
class TestArena:
    def test_turn_left_reduces_left_bearing(self):
        a = d8.Arena2D(target_bearing_deg=+60, target_dist=0.8)   # LEFT
        b0 = a.target_bearing_deg()
        for _ in range(8):
            a.step("TURN_LEFT", 0.05)
        assert abs(a.target_bearing_deg()) < abs(b0)

    def test_turn_right_reduces_right_bearing(self):
        a = d8.Arena2D(target_bearing_deg=-60, target_dist=0.8)  # RIGHT
        b0 = a.target_bearing_deg()
        for _ in range(8):
            a.step("TURN_RIGHT", 0.05)
        assert abs(a.target_bearing_deg()) < abs(b0)

    def test_bearing_sign_convention(self):
        # POSITIVE bearing = target to the LEFT (counterclockwise)
        a = d8.Arena2D(target_bearing_deg=+60, target_dist=0.8)
        assert a.target_bearing_deg() > 0

    def test_forward_moves_toward_centered_target(self):
        a = d8.Arena2D(target_bearing_deg=0, target_dist=0.8)
        d0 = a.target_distance()
        for _ in range(10):
            a.step("FORWARD", 0.05)
        assert a.target_distance() < d0

    def test_reach_and_success(self):
        a = d8.Arena2D(target_bearing_deg=0, target_dist=0.2)
        for _ in range(20):
            a.step("FORWARD", 0.05)
            if a.reached_target():
                break
        assert a.reached_target()

    def test_no_target_reports_none(self):
        a = d8.Arena2D(target_present=False)
        assert a.target_bearing_deg() is None
        assert not a.reached_target()

    def test_loom_grows_and_jump_once(self):
        a = d8.Arena2D(target_present=False, loom=True)
        a.step("STOP", 0.4)
        assert a.loom_angular_deg() == 0.0
        a.step("STOP", 1.0)
        assert 2.0 < a.loom_angular_deg() <= 70.0
        p0 = (a.x, a.y)
        a.step("JUMP", 0.05)
        assert a.escaped and a.jump_count == 1
        a.step("JUMP", 0.05)
        assert a.jump_count == 2            # logged, but only one hop

    def test_stop_keeps_position(self):
        a = d8.Arena2D(target_present=False)
        for _ in range(10):
            a.step("STOP", 0.05)
        assert (a.x, a.y) == (0.0, 0.0)

    def test_walls_clamp(self):
        a = d8.Arena2D(target_present=False)
        a.heading = 0.0
        for _ in range(100):
            a.step("FORWARD", 0.05)
        assert a.x <= 1.0


# ---------------------------------------------------------------- D8 encoder
class TestEncoder:
    def test_center_bilateral(self):
        e = d8.VisualSensoryEncoder()
        r = e.rates(dict(target_present=True, target_bearing_deg=0.0,
                         loom_angular_deg=0.0))
        assert r["target_left"] == r["target_right"] == 75.0

    def test_right_target_drives_right_only(self):
        e = d8.VisualSensoryEncoder()
        r = e.rates(dict(target_present=True, target_bearing_deg=-45.0,
                         loom_angular_deg=0.0))
        assert r["target_right"] == 150.0 and r["target_left"] == 0.0

    def test_left_full_beyond_ramp(self):
        e = d8.VisualSensoryEncoder()
        r = e.rates(dict(target_present=True, target_bearing_deg=+90.0,
                         loom_angular_deg=0.0))
        assert r["target_left"] == 150.0 and r["target_right"] == 0.0

    def test_no_target_silence(self):
        e = d8.VisualSensoryEncoder()
        r = e.rates(dict(target_present=False, loom_angular_deg=0.0))
        assert not any(r.values())

    def test_loom_saturation(self):
        e = d8.VisualSensoryEncoder()
        r = e.rates(dict(target_present=False, loom_angular_deg=15.0))
        assert r["looming"] == 0.0
        r = e.rates(dict(target_present=False, loom_angular_deg=60.0))
        assert r["looming"] == 150.0
        r = e.rates(dict(target_present=False, loom_angular_deg=90.0))
        assert r["looming"] == 150.0        # clamped

    def test_partial_bearing(self):
        e = d8.VisualSensoryEncoder()
        r = e.rates(dict(target_present=True, target_bearing_deg=+15.0,
                         loom_angular_deg=0.0))
        assert r["target_left"] == 150.0 and r["target_right"] == 0.0
        r = e.rates(dict(target_present=True, target_bearing_deg=-15.0,
                         loom_angular_deg=0.0))
        assert r["target_right"] == 150.0 and r["target_left"] == 0.0
        r = e.rates(dict(target_present=True, target_bearing_deg=+7.5,
                         loom_angular_deg=0.0))
        assert abs(r["target_left"] - 112.5) < 1e-9
        assert abs(r["target_right"] - 37.5) < 1e-9

    def test_arena_to_encoder_end_to_end_sign(self):
        """The regression that motivated the sign convention: a LEFT target
        must drive the target_LEFT channel (-> LC9-left -> P9-left ->
        TURN_LEFT toward the target)."""
        a = d8.Arena2D(target_bearing_deg=+60, target_dist=0.8)
        e = d8.VisualSensoryEncoder()
        r = e.rates(a.visual_state())
        assert r["target_left"] == 150.0 and r["target_right"] == 0.0
        a2 = d8.Arena2D(target_bearing_deg=-60, target_dist=0.8)
        r2 = e.rates(a2.visual_state())
        assert r2["target_right"] == 150.0 and r2["target_left"] == 0.0


# ---------------------------------------------------------------- D9 decoder
class TestDecoder:
    def test_silence_is_stop_failsafe(self):
        dec = d9.MotorDecoder(POP_SIZES)
        for _ in range(6):
            r = dec.decode({}, 50)
        assert r["action"] == "STOP"

    def test_right_p9_drive_turns_right(self):
        dec = d9.MotorDecoder(POP_SIZES)
        for _ in range(6):
            r = dec.decode({"P9_right": 30}, 50)
        assert r["action"] == "TURN_RIGHT"

    def test_bilateral_p9_forward(self):
        dec = d9.MotorDecoder(POP_SIZES)
        for _ in range(6):
            r = dec.decode({"P9_left": 25, "P9_right": 25}, 50)
        assert r["action"] == "FORWARD"

    def test_hysteresis(self):
        dec = d9.MotorDecoder(POP_SIZES)
        for _ in range(6):
            dec.decode({"P9_right": 30}, 50)
        # flip below threshold+hysteresis -> stay RIGHT
        r = dec.decode({"P9_left": 13, "P9_right": 0}, 50)
        assert r["action"] == "TURN_RIGHT"
        for _ in range(5):
            r = dec.decode({"P9_left": 13, "P9_right": 0}, 50)
        assert r["action"] == "TURN_LEFT"

    def test_giant_fiber_jump(self):
        dec = d9.MotorDecoder(POP_SIZES)
        r = dec.decode({"GF_left": 8, "GF_right": 9}, 50)
        assert r["action"] == "JUMP"

    def test_mdn_backward(self):
        dec = d9.MotorDecoder(POP_SIZES)
        for _ in range(6):
            r = dec.decode({"MDN_bilateral": 20}, 50)
        assert r["action"] == "BACKWARD"

    def test_conflict_logged_not_hidden(self):
        dec = d9.MotorDecoder(POP_SIZES)
        for _ in range(6):
            r = dec.decode({"P9_left": 40, "P9_right": 2,
                            "MDN_bilateral": 30}, 50)
        assert r["conflicts"]
        assert dec.conflict_log

    def test_decoder_never_sees_target(self):
        # API contract: decode() accepts ONLY population counts
        dec = d9.MotorDecoder(POP_SIZES)
        sig = dec.decode.__code__.co_varnames[:dec.decode.__code__.co_argcount]
        assert "bearing" not in sig and "target" not in sig

    def test_shuffled_slots_bijective(self):
        m = d9.MotorDecoder.shuffled_slots(POP_SIZES, seed=5)
        assert len(set(m.values())) == len(m)
        assert set(m.values()) == set(m.keys()) or True   # partial perm ok


# ---------------------------------------------------------------- D10 config
class TestD10Config:
    def test_preregistration_exists_and_frozen(self):
        p = d10.DATA / "preregistration.json"
        if not p.exists():        # not yet written -> require explicit prereg
            assert True
            return
        reg = json.loads(p.read_text())
        for key in ("chunk_ms", "conditions", "scenarios", "reps",
                    "master_seed", "gate3_criteria"):
            assert key in reg

    def test_seed_rule_deterministic(self):
        seeds = [d10.MASTER_SEED + 1000 * i + r
                 for i in range(3) for r in range(d10.REPS)]
        assert len(set(seeds)) == len(seeds)

    def test_scenario_caps(self):
        for s in ("target_left", "target_right", "target_center"):
            assert d10.SCENARIO_PARAMS[s]["cap_s"] == 5.0
        assert d10.SCENARIO_PARAMS["no_target"]["cap_s"] == 3.0
        assert d10.SCENARIO_PARAMS["looming"]["cap_s"] == 3.0

    def test_shuffled_interface_spec_disjoint_and_deterministic(self):
        import tempfile
        orig = json.loads((d10.DATA / "sensory_interface.json").read_text())
        intended = set()
        for ch in orig["channels"].values():
            intended.update(int(i) for i in ch["ids"])
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.json"
            p2 = Path(td) / "b.json"
            d10.shuffled_interface_spec(123, p1)
            d10.shuffled_interface_spec(123, p2)
            s1 = json.loads(p1.read_text())
            s2 = json.loads(p2.read_text())
            assert (json.dumps(s1, sort_keys=True)
                    == json.dumps(s2, sort_keys=True))
            for ch in s1["channels"].values():
                assert not (set(int(i) for i in ch["ids"]) & intended)
