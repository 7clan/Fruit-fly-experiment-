"""Stimulus-response analysis (open-loop, Phase 2 observation only).

Per event: response = mean cos(heading - stimulus_angle) over [0, 3] s
after onset  MINUS  mean over [-8, -1] s before onset — computed ONLY on
frames where the fly is MOVING (speed >= 2 mm/s); a heading while standing
still is noise. Events with too few moving frames in either window are
dropped (reported in n_used).

Session statistic: mean over usable events. Permutation test: 1000 joint
circular shifts of the event-time vector (sides move with their events),
two-sided, seeded -> deterministic p-value.

The stimulus angle is the arena-frame direction pointing AT the active
marker (N marker = FORWARD = pi/2; E = RIGHT = 0; S = -pi/2; W = pi).
"""
from __future__ import annotations

import math

import numpy as np

SIDE_ANGLE = {"N": math.pi / 2, "E": 0.0, "S": -math.pi / 2, "W": math.pi}
MIN_MOVING_SPEED = 2.0     # mm/s; headings below this are not meaningful


def _mean_toward(heading, speed, t, t0, t1, angle):
    sel = (t >= t0) & (t < t1) & np.isfinite(heading) & \
        np.isfinite(speed) & (speed >= MIN_MOVING_SPEED)
    if sel.sum() < 3:
        return np.nan
    return float(np.mean(np.cos(heading[sel] - angle)))


def event_response(heading, speed, t, event, resp=(0.0, 3.0), base=(-8.0, -1.0)):
    ang = SIDE_ANGLE[event["side"]]
    on = event["t_on"]
    r = _mean_toward(heading, speed, t, on + resp[0], on + resp[1], ang)
    b = _mean_toward(heading, speed, t, on + base[0], on + base[1], ang)
    return float(r - b) if (np.isfinite(r) and np.isfinite(b)) else float("nan")


def session_response(heading, speed, t, events, resp=(0.0, 3.0), base=(-8.0, -1.0)):
    per_event = [event_response(heading, speed, t, e, resp, base)
                 for e in events]
    vals = [v for v in per_event if np.isfinite(v)]
    return {
        "n_events": len(events),
        "n_used": len(vals),
        "mean_response": float(np.mean(vals)) if vals else float("nan"),
        "per_event": per_event,
    }


def permutation_test(heading, speed, t, events, iterations=1000,
                     seed=20260927, resp=(0.0, 3.0), base=(-8.0, -1.0)):
    """Circular-shift permutation of event times (shift applied to ALL events
    jointly, wrapping within the session; sides stay attached to events)."""
    obs = session_response(heading, speed, t, events, resp, base)["mean_response"]
    if not events or not np.isfinite(obs):
        return {"observed": obs, "p_value": float("nan"), "iterations": 0}
    rng = np.random.default_rng(seed)
    t_end = float(t[-1]) if len(t) else 0.0
    span = max(t_end - float(events[0]["t_on"]), 1.0)
    count = 0
    used = 0
    for _ in range(iterations):
        off = rng.uniform(0.0, span)
        shifted = []
        for e in events:
            t_on = (e["t_on"] + off) % span
            # keep shifted onsets away from the edges so windows exist
            if 10.0 < t_on < t_end - 6.0:
                shifted.append({"t_on": t_on, "t_off": t_on + 1.0,
                                "side": e["side"]})
        if not shifted:
            continue
        perm = session_response(heading, speed, t, shifted, resp,
                                base)["mean_response"]
        if np.isfinite(perm):
            used += 1
            if abs(perm) >= abs(obs):
                count += 1
    p = count / used if used else float("nan")
    return {"observed": float(obs), "p_value": float(p), "iterations": used}
