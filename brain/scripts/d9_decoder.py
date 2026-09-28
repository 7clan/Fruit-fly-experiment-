"""D9 motor decoder: verified descending-neuron activity -> actions.

The decoder is an EXPLICIT rule set (no learning, no hidden controller):
firing-rate thresholds over a sliding window, left/right differential,
hysteresis, minimum action duration, STOP failsafe, conflict logging.

Readout populations (Gate-2 / D6 map, side-split via FlyWire v783 sides):

  FORWARD      BPN + RRN + P9 (walk command neurons)
  BACKWARD     MDN (Moonwalker Descending Neurons, Bidaye 2014)
  TURN_LEFT    P9_left > P9_right + delta  (P9 = forward + ipsiversive turn,
               Bidaye 2020); fallback source: DN_ALL lateral differential
  TURN_RIGHT   mirror
  STOP         Foxglove + Bluebell walk-OFF (Sapkal 2024) + failsafe
               (no walk population above the minimum rate)
  JUMP/ESCAPE  giant fiber DNp01 (looming escape, King & Wyman 1980)

The decoder NEVER sees target position or any arena state - ONLY neural
population spike counts. The D10 control conditions prove that behavior
depends on this wiring (shuffled readouts must degrade).
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np

WALK_POPS = ("P9_left", "P9_right", "BPN_bilateral", "RRN_bilateral")
STOP_POPS = ("FG_bilateral", "BB_bilateral")


class MotorDecoder:
    """Explicit DN-rate -> action decoder.

    All parameters are pre-registered (preregistration.json) and frozen
    before the D10 closed-loop matrix; the same frozen decoder config is
    used for intact AND shuffled-motor conditions.
    """

    def __init__(self, pop_sizes, params=None):
        self.pop_sizes = dict(pop_sizes)
        p = dict(
            window_ms=300.0,
            theta_jump_hz=25.0,        # giant fiber
            theta_bwd_hz=4.0,          # MDN
            theta_fwd_hz=5.0,          # mean walk-population rate
            theta_diff_hz=8.0,         # P9 (or DN_ALL) left-right differential
            delta_hyst_hz=4.0,         # turn-direction hysteresis
            theta_stop_hz=4.0,         # walk-OFF populations
            theta_min_walk_hz=1.5,     # failsafe: below this -> STOP
            min_action_chunks=2,
            turn_source="P9",          # "P9" (decoder-A) or "DN_ALL" (fallback)
        )
        if params:
            p.update(params)
        self.p = p
        self.window = deque()          # (pop_counts dict, chunk_ms)
        self.window_ms = 0.0
        self.last_action = "STOP"
        self.action_age = 0            # chunks the current action has been held
        self.conflict_log = []

    # ------------------------------------------------------------------
    def _rates(self):
        """Per-population mean firing rate (Hz/neuron) over the window."""
        n_s = self.window_ms / 1000.0
        if n_s <= 0:
            return {}
        out = {}
        for pop, n in self.pop_sizes.items():
            if n <= 0:
                out[pop] = 0.0
                continue
            c = sum(counts.get(pop, 0) for counts, _ in self.window)
            out[pop] = c / (n * n_s)
        return out

    def _push(self, pop_counts, chunk_ms):
        self.window.append((pop_counts, chunk_ms))
        self.window_ms += chunk_ms
        while self.window_ms - self.window[0][1] >= self.p["window_ms"] \
                and len(self.window) > 1:
            counts, ms = self.window.popleft()
            self.window_ms -= ms

    # ------------------------------------------------------------------
    def decode(self, pop_counts, chunk_ms=50.0):
        """One chunk of DN spike counts -> action + full audit record."""
        self._push(pop_counts, chunk_ms)
        rates = self._rates()
        p = self.p

        gf = rates.get("GF_left", 0) + rates.get("GF_right", 0)
        gf = gf / 2.0 if (self.pop_sizes.get("GF_left", 0)
                          + self.pop_sizes.get("GF_right", 0)) > 0 else gf
        mdn = rates.get("MDN_bilateral", 0.0)
        walk = [rates.get(k, 0.0) for k in WALK_POPS]
        walk_mean = float(np.mean(walk)) if walk else 0.0
        stop_off = float(np.mean([rates.get(k, 0.0) for k in STOP_POPS]))
        if p["turn_source"] == "P9":
            diff = rates.get("P9_left", 0.0) - rates.get("P9_right", 0.0)
        else:
            diff = (rates.get("DN_ALL_left", 0.0)
                    - rates.get("DN_ALL_right", 0.0))

        # ---- intents (all evaluated, conflicts logged, not hidden)
        intents = {}
        intents["JUMP"] = gf > p["theta_jump_hz"]
        intents["BACKWARD"] = mdn > p["theta_bwd_hz"]
        intents["FORWARD"] = walk_mean > p["theta_fwd_hz"]
        intents["STOP_WALK_OFF"] = stop_off > p["theta_stop_hz"]
        any_walk = max(walk) if walk else 0.0
        intents["STOP_FAILSAFE"] = (walk_mean < p["theta_min_walk_hz"]
                                    and mdn < p["theta_bwd_hz"]
                                    and gf < p["theta_jump_hz"])

        # turn with hysteresis around the previous direction
        prev_dir = 0
        if self.last_action == "TURN_LEFT":
            prev_dir = 1
        elif self.last_action == "TURN_RIGHT":
            prev_dir = -1
        turn = 0
        if diff > p["theta_diff_hz"] and (prev_dir >= 0
                                          or diff > p["theta_diff_hz"]
                                          + p["delta_hyst_hz"]):
            turn = 1
        elif diff < -p["theta_diff_hz"] and (prev_dir <= 0
                                             or diff < -p["theta_diff_hz"]
                                             - p["delta_hyst_hz"]):
            turn = -1
        intents["TURN_LEFT"] = turn == 1
        intents["TURN_RIGHT"] = turn == -1

        # ---- conflict detection (logged, never hidden)
        active = [k for k, v in intents.items() if v]
        conflicts = []
        if "BACKWARD" in active and ("FORWARD" in active or turn != 0):
            conflicts.append("FORWARD/TURN vs BACKWARD")
        if intents["STOP_WALK_OFF"] and ("FORWARD" in active or turn != 0
                                         or "BACKWARD" in active):
            conflicts.append("walk-OFF vs walk-ON")
        if "JUMP" in active and len(active) > 1:
            conflicts.append("JUMP vs other action")

        # ---- selection (explicit priority; conflicts recorded above)
        if intents["JUMP"]:
            action = "JUMP"
        elif intents["BACKWARD"] and not intents["FORWARD"] and turn == 0:
            action = "BACKWARD"
        elif intents["BACKWARD"] and intents["FORWARD"]:
            # dominance resolution, logged
            action = "BACKWARD" if mdn > walk_mean else (
                "BACKWARD" if turn == 0 and mdn >= walk_mean * 0.5 else
                ("TURN_LEFT" if turn == 1 else
                 "TURN_RIGHT" if turn == -1 else "FORWARD"))
        elif turn != 0:
            action = "TURN_LEFT" if turn == 1 else "TURN_RIGHT"
        elif intents["FORWARD"]:
            action = "FORWARD"
        else:
            action = "STOP"          # failsafe (incl. walk-OFF and silence)

        # ---- minimum action duration (JUMP is instantaneous by nature)
        if (action != self.last_action and action not in ("JUMP",)
                and self.action_age < p["min_action_chunks"]
                and self.last_action not in ("JUMP",)):
            action = self.last_action
        if action == self.last_action:
            self.action_age += 1
        else:
            self.action_age = 1
        self.last_action = action
        if conflicts:
            self.conflict_log.append(dict(action=action, conflicts=conflicts,
                                          t_window_end_ms=self.window_ms))

        return dict(action=action, rates_hz={k: round(v, 2)
                                             for k, v in rates.items()},
                    p9_diff_hz=round(diff, 2), gf_hz=round(gf, 2),
                    mdn_hz=round(mdn, 0), walk_mean_hz=round(walk_mean, 2),
                    intents={k: bool(v) for k, v in intents.items()},
                    conflicts=conflicts)

    # ------------------------------------------------------------------
    @staticmethod
    def shuffled_slots(pop_sizes, seed):
        """D10 control E: permute which population feeds which decoder slot
        (a seeded permutation of the population-name list)."""
        names = sorted(n for n in pop_sizes
                       if n in WALK_POPS + STOP_POPS
                       + ("MDN_bilateral", "GF_left", "GF_right",
                          "DN_ALL_left", "DN_ALL_right", "BRK_bilateral"))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(names))
        return {src: names[int(dst)] for src, dst in zip(names, perm)}


def demo():
    """Synthetic unit demo (no brain): thresholds, hysteresis, failsafe."""
    sizes = dict(P9_left=1, P9_right=1, BPN_bilateral=33, RRN_bilateral=2,
                 MDN_bilateral=4, GF_left=1, GF_right=1, FG_bilateral=2,
                 BB_bilateral=2, BRK_bilateral=6, DN_ALL_left=647,
                 DN_ALL_right=650, DN_LEG_bilateral=24)
    dec = MotorDecoder(sizes)
    # silence -> STOP
    print(dec.decode({}, 50)["action"], "<- expect STOP")
    # strong right P9 drive -> TURN_RIGHT
    for _ in range(6):
        r = dec.decode({"P9_right": 30}, 50)
    print(r["action"], r["p9_diff_hz"], "<- expect TURN_RIGHT")
    # flip to left, but below hysteresis -> stays RIGHT
    r = dec.decode({"P9_left": 12, "P9_right": 2}, 50)
    print(r["action"], "<- expect TURN_RIGHT (hysteresis)")
    for _ in range(4):
        r = dec.decode({"P9_left": 12, "P9_right": 0}, 50)
    print(r["action"], "<- expect TURN_LEFT")
    # giant fiber burst -> JUMP
    r = dec.decode({"GF_left": 8, "GF_right": 9}, 50)
    print(r["action"], "<- expect JUMP")
    # walk-off conflict
    dec2 = MotorDecoder(sizes)
    for _ in range(6):
        r = dec2.decode({"P9_left": 40, "P9_right": 5, "FG_bilateral": 3}, 50)
    print(r["action"], r["conflicts"], "<- expect TURN_LEFT + conflict logged")


if __name__ == "__main__":
    demo()
