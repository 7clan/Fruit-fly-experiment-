"""Intention decoder: descending-neuron population rates → Intention.

TIER: the DECODER IS AN EXPLICIT RULE SET over neural population spike
counts, extending the validated D9 decoder (thresholds, hysteresis,
min-duration, STOP-failsafe, conflict logging) to the live combat
intention vocabulary. The decoder NEVER sees game/arena state — ONLY DN
population rates from a BrainChunkRecord.

Validated intention subset (decoder-config intent-decoder-v1, frozen):

  STOP         FG/BB walk-OFF above theta, or failsafe (no walk pop active)
  TURN_LEFT    P9_left > P9_right + delta   (P9 = forward + ipsiversive
  TURN_RIGHT   mirror                         turn, Bidaye 2020 — the
               SAME lateralization rule as frozen D9 decoder-A)
  APPROACH     mean walk populations above theta with small differential
  RETREAT      MDN above theta (Moonwalker, Bidaye 2014)
  ESCAPE       giant fiber above theta (looming escape, King & Wyman 1980)

Extension path (NOT enabled in v1 — requires new preregistration +
validation before live use): EVADE_LEFT/RIGHT (lateralized looming
route), DEFEND / ATTACK_* (no verified DN correlate in the D4–D6 map
yet; must be wired + validated at Gate-7 stages, never assumed).

Priority (explicit, like D9): ESCAPE > RETREAT > TURN_* > APPROACH > STOP
STOP-failsafe ALWAYS applies when no population is above minimum.

All thresholds carry hysteresis (delta) and minimum duration
(min_chunks) exactly as D9; a conflicting-rule log records ambiguous
windows for replay inspection.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from ..schemas import Intention

CONFIG_ID = "intent-decoder-v1"
EMITTABLE = ("STOP", "TURN_LEFT", "TURN_RIGHT", "APPROACH", "RETREAT",
             "ESCAPE")

# frozen parameters (mirrors of the validated D9 preregistration values)
PARAMS = dict(
    window_ms=300.0,          # sliding rate window
    theta_escape_hz=25.0,     # giant fiber
    theta_retreat_hz=4.0,     # MDN
    theta_walk_hz=5.0,        # mean walk-population rate
    theta_diff_hz=8.0,        # P9 left-right differential
    delta_hyst_hz=4.0,        # turn-direction hysteresis
    theta_stop_hz=4.0,        # walk-OFF populations
    theta_min_walk_hz=1.5,    # failsafe: below -> STOP
    min_action_chunks=2,
)

WALK_POPS = ("P9_left", "P9_right", "BPN_bilateral", "RRN_bilateral")
STOP_POPS = ("FG_bilateral", "BB_bilateral")


class IntentionDecoder:
    """DN population rates → Intention (explicit rules, no learning)."""

    def __init__(self, pop_sizes: dict, params: Optional[dict] = None,
                 decoder_config: str = CONFIG_ID):
        self.pop_sizes = dict(pop_sizes or {p: 20 for p in
                                            WALK_POPS + STOP_POPS +
                                            ("MDN_bilateral", "GF_left")})
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.decoder_config = decoder_config
        self.window: deque = deque()      # (pop_counts, chunk_ms)
        self.window_ms = 0.0
        self.last_name = "STOP"
        self.age_chunks = 0
        self.conflict_log: list = []
        self.chunk_counter = 0

    # ------------------------------------------------------------------
    def _rates(self) -> dict:
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

    def _pop_mean(self, rates: dict, pops: tuple) -> float:
        vals = [rates.get(p, 0.0) for p in pops if p in self.pop_sizes]
        return sum(vals) / len(vals) if vals else 0.0

    # ------------------------------------------------------------------
    def decode(self, chunk: dict) -> Intention:
        """chunk: BrainChunkRecord (pop_counts + chunk_ms used; NOTHING else)."""
        counts = dict(chunk.get("pop_counts", {}))
        cm = float(chunk.get("chunk_ms", 50.0))
        self.chunk_counter += 1

        self.window.append((counts, cm))
        self.window_ms += cm
        while self.window_ms > self.p["window_ms"] and len(self.window) > 1:
            old_counts, old_ms = self.window.popleft()
            self.window_ms -= old_ms

        rates = self._rates()
        gf = max(rates.get("GF_left", 0.0), rates.get("GF_right", 0.0))
        mdn = self._pop_mean(rates, ("MDN_bilateral", "MDN_left", "MDN_right"))
        walk = self._pop_mean(rates, WALK_POPS)
        stop = self._pop_mean(rates, STOP_POPS)
        p9l, p9r = rates.get("P9_left", 0.0), rates.get("P9_right", 0.0)

        name, confidence, notes = self._decide(gf, mdn, walk, stop, p9l, p9r)

        # minimum action duration (like D9): hold the previous intention
        if name == self.last_name:
            self.age_chunks += 1
        elif self.age_chunks < self.p["min_action_chunks"] and self.last_name != "STOP":
            name = self.last_name          # too young to switch (non-STOP)
            self.age_chunks += 1
            notes = notes | {"held_min_duration": True}
        else:
            self.age_chunks = 0

        self.last_name = name
        return Intention(
            name=name,
            confidence=round(min(1.0, confidence), 3),
            ts_ns=chunk.get("ts_ns", 0),
            brain_chunk_id=int(chunk.get("chunk_id", -1)),
            dn_rates_hz={k: round(v, 2) for k, v in rates.items()},
            decoder_config=self.decoder_config,
            notes=notes,
        )

    # ------------------------------------------------------------------
    def _decide(self, gf, mdn, walk, stop, p9l, p9r) -> tuple:
        p = self.p
        conflicts = {}

        if gf >= p["theta_escape_hz"]:
            return "ESCAPE", gf / (2 * p["theta_escape_hz"]), {"rule": "gf"}
        if mdn >= p["theta_retreat_hz"]:
            return "RETREAT", mdn / (2 * p["theta_retreat_hz"]), {"rule": "mdn"}

        turn_l = p9l > p9r + p["delta_hyst_hz"] and p9l >= p["theta_diff_hz"]
        turn_r = p9r > p9l + p["delta_hyst_hz"] and p9r >= p["theta_diff_hz"]
        if turn_l and turn_r:
            conflicts["turn_conflict"] = (round(p9l, 2), round(p9r, 2))
            turn_l = turn_r = False
        if turn_l:
            return ("TURN_LEFT", (p9l - p9r) / p["theta_diff_hz"] / 2,
                    {"rule": "p9_diff"})
        if turn_r:
            return ("TURN_RIGHT", (p9r - p9l) / p["theta_diff_hz"] / 2,
                    {"rule": "p9_diff"})

        if walk >= p["theta_walk_hz"]:
            return "APPROACH", walk / (2 * p["theta_walk_hz"]), {"rule": "walk"}
        if stop >= p["theta_stop_hz"] or walk < p["theta_min_walk_hz"]:
            return "STOP", max(stop / (2 * p["theta_stop_hz"]), 0.5), \
                {"rule": "walk_off" if stop >= p["theta_stop_hz"] else "failsafe"}

        if conflicts:
            self.conflict_log.append({"chunk": self.chunk_counter, **conflicts})
        return "STOP", 0.5, {"rule": "default_failsafe"}

    def reset(self) -> None:
        self.window.clear()
        self.window_ms = 0.0
        self.last_name = "STOP"
        self.age_chunks = 0
        self.conflict_log.clear()
        self.chunk_counter = 0
