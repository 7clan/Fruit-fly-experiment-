"""WorldObservation → fly input channel encoding.

TIER: ENGINEERED (perception-side encoding harness). The biological
brain receives these as Poisson drive rates through the VERIFIED D8
sensory interface (lateralized LC9 target channels, looming LPLC2/LC4
threat channel); the canonical brain itself is untouched.

Pre-registered curves (frozen before any live gate; changes require a
new decoder-config id and re-validation):

  * target lateralization: triangular tuning on relative bearing,
    matching D8 encoder rung-1 calibration (left/right fields, center
    overlap) — NOT reinvented here; bearing→(left,right,center) weights.
  * distance: normalized distance d∈[0,1] passes through as
    target_distance (1 = touching), then mapped to rate curves by the
    brain-side D8 encoder, not here.
  * threat: lateralized wind-up/attack detection weighted by proximity;
    overall intensity = max over enemies of proximity × wind-up weight.
  * attack_opportunity: enemy in effective range and NOT in wind-up.
  * low_health_context: 1.0 when health fraction < 0.25 (or HP unknown → 0).

The helper supplies ONLY goal_relevance / avoidance_bias /
approach_bias context values — never which key to press.
"""

from __future__ import annotations

import math

from ..schemas import FlyChannels, WorldObservation

# ---- frozen parameters (decoder-config "fly-channels-v1") ------------------
CONFIG_ID = "fly-channels-v1"

_HALF_FIELD_DEG = 90.0     # bearing magnitude at which a side channel peaks
_CENTER_HALF_WIDTH_DEG = 22.5   # |bearing| below this counts as center
_LOW_HEALTH_FRACTION = 0.25
_THREAT_RANGE_NEAR = 0.55   # normalized distance at/inside which threat ramps
_ATTACK_RANGE_NEAR = 0.45   # normalized distance inside which attack is viable


def _bearing_deg(direction: float | None) -> float | None:
    if direction is None:
        return None
    # direction is a bearing relative to player heading, radians expected
    return math.degrees(direction)


def lateral_weights(bearing_deg: float | None) -> tuple[float, float, float]:
    """Triangular left/right/center weights from relative bearing (deg).

    Matches D8 triangular tuning: full-peak at ±90°, linear ramp from 0°,
    center window ±22.5°. Unknown bearing → all zeros.
    """
    if bearing_deg is None:
        return 0.0, 0.0, 0.0
    b = abs(bearing_deg) % 360.0
    if b > 180.0:
        b = 360.0 - b
    sign = 1.0 if bearing_deg >= 0 else -1.0     # + : right, - : left
    center = max(0.0, 1.0 - b / _CENTER_HALF_WIDTH_DEG) if b <= _CENTER_HALF_WIDTH_DEG else 0.0
    if b <= _CENTER_HALF_WIDTH_DEG:
        # inside center window: split residual to both side channels
        side = 1.0 - center
        left = right = 0.5 * side
    else:
        side = min(1.0, (b - _CENTER_HALF_WIDTH_DEG)
                   / (_HALF_FIELD_DEG - _CENTER_HALF_WIDTH_DEG))
        left = side if sign < 0 else 0.0
        right = side if sign > 0 else 0.0
    return left, right, center


def _threat_terms(enemy) -> tuple[float, float, float, bool]:
    """(threat_left, threat_right, intensity, attack_opportunity) for one enemy."""
    bearing = _bearing_deg(enemy.direction)
    left, right, _ = lateral_weights(bearing)
    dist = enemy.distance if enemy.distance is not None else 1.0
    # proximity ramp inside _THREAT_RANGE_NEAR
    if dist >= 1.0:
        prox = 0.0
    else:
        prox = max(0.0, min(1.0, (_THREAT_RANGE_NEAR - dist) / _THREAT_RANGE_NEAR))
    windup = 1.0 if enemy.attacking else 0.0
    # threat = proximity × (0.35 base + wind-up boost) — lateralized
    amp = prox * (0.35 + 0.65 * windup) * max(0.05, enemy.confidence)
    # attack opportunity: near enough, not in wind-up, confident
    opp = 1.0 if (dist <= _ATTACK_RANGE_NEAR and not enemy.attacking
                  and enemy.confidence >= 0.3) else 0.0
    return amp * left, amp * right, amp, bool(opp)


def encode(obs: WorldObservation,
           goal_relevance: float = 0.5,
           avoidance_bias: float = 0.0,
           approach_bias: float = 0.0) -> FlyChannels:
    """Encode a structured observation into the compact fly channel set."""
    ch = FlyChannels(ts_ns=obs.ts_ns)
    ch.goal_relevance = float(max(0.0, min(1.0, goal_relevance)))

    # --- target (visual object of interest, e.g. quest/loot/waypoint) ---
    tb = _bearing_deg(obs.target.direction)
    tl, tr, tc = lateral_weights(tb)
    ch.target_left = tl * obs.target.confidence
    ch.target_right = tr * obs.target.confidence
    ch.target_center = tc * obs.target.confidence
    if obs.target.distance is not None:
        ch.target_distance = float(max(0.0, min(1.0, obs.target.distance)))

    # --- enemies: accumulate lateralized threat; nearest gives opportunity ---
    tl_acc = tr_acc = 0.0
    intensity = 0.0
    opportunity = 0.0
    for e in obs.enemies:
        l, r, a, opp = _threat_terms(e)
        tl_acc = min(1.0, tl_acc + l)
        tr_acc = min(1.0, tr_acc + r)
        intensity = max(intensity, a)
        if opp and opportunity == 0.0:
            opportunity = 1.0 if e.distance is not None and e.distance <= _ATTACK_RANGE_NEAR else 0.0
    ch.threat_left = tl_acc
    ch.threat_right = tr_acc
    ch.threat_intensity = intensity
    ch.attack_opportunity = opportunity

    # --- health context ---
    if obs.player.health is not None and obs.player.health_units == "fraction":
        ch.low_health_context = 1.0 if obs.player.health < _LOW_HEALTH_FRACTION else 0.0

    # --- helper context (ENGINEERED; never a key choice) ---
    ch.avoidance_value = float(max(0.0, min(1.0, avoidance_bias)))
    ch.approach_value = float(max(0.0, min(1.0, approach_bias)))
    return ch


def to_sensory_rates(channels: FlyChannels) -> dict[str, float]:
    """Map fly channels → D8 sensory-interface Poisson rates (Hz).

    This is the bridge to the VERIFIED biological sensory interface:
      target_left/right → lateralized LC9 samples (D8 rung-1 calibration:
      peak 72 Hz contra-lateral sample; here 0..72 Hz linear in signal)
      threat_intensity   → looming sample (LPLC2+LC4, Gate-2 150 Hz drive
                            scale; 0..150 Hz linear)

    HONESTY NOTE: looming is driven ONLY by perceived threat_intensity —
    the ENGINEERED helper context channels (avoidance_value /
    approach_value / goal_relevance / low_health_context) are recorded in
    every FlyChannels snapshot but are NOT wired into any biological
    sensory entry: no verified neural route exists for them in the D4–D6
    map (same discipline as the D8 optic-flow note: recorded, not
    forced). Wiring any context channel into the brain would require a
    new pre-registered + validated interface step.

    Exact rate curves are the D8 encoder's frozen calibration — this map
    is the compact live-side equivalent and is labeled as such (any live/
    reference divergence must be benchmarked, never silently equated).
    """
    return {
        "target_left_hz": 72.0 * channels.target_left,
        "target_right_hz": 72.0 * channels.target_right,
        "looming_hz": 150.0 * channels.threat_intensity,
    }
