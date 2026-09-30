import json
from pathlib import Path
from types import SimpleNamespace

from lab.perception import fly_channels as fc


ROOT = Path(__file__).resolve().parents[1]


def test_live_d8_constants_match_frozen_preregistration():
    prereg = json.loads(
        (ROOT / "brain" / "data" / "d7_d10" /
         "preregistration.json").read_text())
    enc = prereg["encoder"]
    assert fc._D8_TARGET_HZ == enc["r_target_hz"] == 150.0
    assert fc._D8_THETA0_DEG == enc["theta0_deg"] == 15.0
    assert fc._D8_THETA_W_DEG == enc["theta_w_deg"] == 30.0


def test_live_d8_lateral_tuning_matches_reference_geometry():
    left, right, center = fc.lateral_weights(0.0)
    assert (left, right, center) == (0.5, 0.5, 1.0)
    left, right, _ = fc.lateral_weights(-15.0)
    assert (left, right) == (1.0, 0.0)
    left, right, _ = fc.lateral_weights(15.0)
    assert (left, right) == (0.0, 1.0)
    left, right, _ = fc.lateral_weights(-60.0)
    assert (left, right) == (1.0, 0.0)
    left, right, _ = fc.lateral_weights(60.0)
    assert (left, right) == (0.0, 1.0)


def test_live_d8_rate_scale_is_150_hz_not_dn_response_rate():
    ch = SimpleNamespace(
        target_left=1.0, target_right=0.0, threat_intensity=0.0)
    rates = fc.to_sensory_rates(ch)
    assert rates["target_left_hz"] == 150.0
    assert rates["target_right_hz"] == 0.0

    ch = SimpleNamespace(
        target_left=0.5, target_right=0.5, threat_intensity=0.0)
    rates = fc.to_sensory_rates(ch)
    assert rates["target_left_hz"] == 75.0
    assert rates["target_right_hz"] == 75.0
