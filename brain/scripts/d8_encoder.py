"""D8 sensory encoder: the engineered visual interface between the
artificial 2D environment and the canonical digital Drosophila brain.

Components (all ENGINEERED harness - the brain itself is the untouched
Shiu/FlyWire model):

  build_interface_spec   channel -> flywire IDs. Channels:
      target_left   left-hemisphere LC9 sample   (visual target, left field)
      target_right  right-hemisphere LC9 sample  (visual target, right field)
      looming       Gate-2 looming sample (60 LPLC2 + 20 LC4, verbatim IDs)
  build_readout_spec     motor readout populations, side-split where the
      biology is lateralized (P9_left/P9_right, DN_ALL_left/right, ...).
  Arena2D                artificial 2D visual environment: agent, bright
      target, dark background, optional obstacle, optional looming disc.
  VisualSensoryEncoder   arena visual state -> per-channel Poisson rates
      (pre-registered triangular tuning + looming saturation curve).
  run_probe (CLI)        open-loop validation probes with full logging:
      condition, stimulated IDs, rate, neural response, downstream DN
      response, sensory->motor latency.

Biological basis (D4-D6 map, verified on v783):
  * LC9 is a lateralized visual-projection class (87 left / 92 right) whose
    v783 wiring to descending neurons is strongly ipsilateral
    (LC9_left -> 932/940 synapses onto LEFT DNs, top partner = left P9;
    LC9_right -> 1582/1599 onto RIGHT DNs, top partner = right P9).
  * P9/DNp09 drives forward walking WITH ipsiversive turning (Bidaye 2020).
  * LPLC2/LC4 -> giant fiber (DNp01) is a direct looming-escape route
    (Gate-2: GF at ~110 Hz under this exact sample at 150 Hz).
  * T4/T5 motion entry propagates only 1-2 hops below DN threshold under
    default parameters (Gate-2), so NO optic-flow channel is wired into the
    closed loop yet - recorded, not forced.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402

DATA = bl.ROOT / "data" / "d7_d10"
RESULTS = bl.ROOT / "results" / "d8_encoder"
IDS = bl.ROOT / "data" / "io_map" / "ids"
ANN = bl.ROOT / "data" / "flywire_annotations_v783" / "classification.csv.gz"

SAMPLING_SEED = 20260928          # Gate-2 sampling seed, kept for continuity


# --------------------------------------------------------------------------
# interface + readout specs
# --------------------------------------------------------------------------
def load_ids(pop):
    return json.loads((IDS / f"{pop}.json").read_text())


def _side_map():
    ann = pd.read_csv(ANN, usecols=["root_id", "side"])
    return dict(zip(ann.root_id.astype(int).tolist(), ann.side.tolist()))


def build_interface_spec(n_lc9_per_side=40, seed=SAMPLING_SEED,
                         out=DATA / "sensory_interface.json"):
    """Seeded sampling of entry populations (engineered interface)."""
    DATA.mkdir(parents=True, exist_ok=True)
    side_of = _side_map()
    rng = np.random.default_rng(seed)

    def sample_side(pop, side, n):
        cand = np.array([i for i in load_ids(pop)
                         if side_of.get(int(i)) == side])
        pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
        return sorted(int(x) for x in pick)

    lc9_L = sample_side("LC9", "left", n_lc9_per_side)
    lc9_R = sample_side("LC9", "right", n_lc9_per_side)
    # looming sample = Gate-2 condition verbatim (60 LPLC2 + 20 LC4)
    loom = json.loads(
        (bl.ROOT / "results" / "gate2_io" / "manifest.json").read_text()
    )["conditions"]["loom_LPLC2_LC4"]["exc_ids"]

    spec = dict(
        version=1,
        sampling_seed=seed,
        n_lc9_per_side=n_lc9_per_side,
        interface_weight="w_syn * f_poi (model defaults, mirrors dbm.poi)",
        note="ENGINEERED sensory interface; brain model unmodified",
        channels={
            "target_left": dict(
                population="LC9", side="left", n=len(lc9_L), ids=lc9_L,
                biological_role="visual target in left hemifield -> "
                                "ipsilateral P9 (forward + left turn)"),
            "target_right": dict(
                population="LC9", side="right", n=len(lc9_R), ids=lc9_R,
                biological_role="visual target in right hemifield -> "
                                "ipsilateral P9 (forward + right turn)"),
            "looming": dict(
                population="LPLC2+LC4 (Gate-2 sample)", n=len(loom), ids=loom,
                biological_role="expanding threat -> giant-fiber escape"),
        },
    )
    Path(out).write_text(json.dumps(spec, indent=1))
    print(f"[interface] {len(lc9_L)} LC9-L / {len(lc9_R)} LC9-R / "
          f"{len(loom)} loom slots -> {out}")
    return spec


def build_readout_spec(out=DATA / "motor_readout.json"):
    """Motor readout populations for the D9 decoder, side-split where
    the biology is lateralized (classification.csv.gz 'side' column)."""
    side_of = _side_map()

    def split(pop):
        ids = [int(i) for i in load_ids(pop)]
        L = [i for i in ids if side_of.get(i) == "left"]
        R = [i for i in ids if side_of.get(i) == "right"]
        return L, R

    pops = {}
    for name, pop in [("P9", "DNp09_P9"), ("MDN", "MDN"), ("BPN", "BPN"),
                      ("RRN", "RRN"), ("GF", "DNp01_GF"),
                      ("FG", "FG_CB0890"), ("BB", "BB_DNg60"),
                      ("BRK", "BRK"), ("DN_LEG", "DN_LEG_CLUSTER"),
                      ("DN_ALL", "DN_ALL")]:
        L, R = split(pop)
        pops[f"{name}_left"] = L
        pops[f"{name}_right"] = R
        if name in ("MDN", "BPN", "BRK", "DN_LEG", "DN_ALL"):
            pops[f"{name}_bilateral"] = L + R      # command-like sets
    spec = dict(version=1, populations=pops,
                note="side split via FlyWire v783 classification side")
    Path(out).write_text(json.dumps(spec, indent=1))
    print(f"[readout] {len(pops)} populations -> {out}")
    return spec


def load_readout_pops(path=DATA / "motor_readout.json"):
    return json.loads(Path(path).read_text())["populations"]


# --------------------------------------------------------------------------
# artificial 2D visual environment
# --------------------------------------------------------------------------
def wrap_deg(a):
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


class Arena2D:
    """Artificial 2D visual environment (idealised geometry; no pixels yet -
    the CV/pixel stage is a much later, game-facing step).

    Agent kinematics are PRE-REGISTERED constants (see preregistration.json):
      FORWARD v=0.25 u/s | TURN (arc) v=0.15 u/s, omega=150 deg/s
      BACKWARD v=0.15 u/s | STOP: no motion | JUMP: 0.30 u hop along heading
    """

    V_FWD, V_TURN, V_BWD = 0.25, 0.15, 0.15
    OMEGA_TURN = math.radians(150.0)
    JUMP_HOP = 0.30
    REACH_RADIUS = 0.15
    HALF = 1.0                        # arena is [-1,1]^2

    def __init__(self, target_bearing_deg=60.0, target_dist=0.8,
                 target_present=True, loom=False, obstacle=None,
                 jitter_deg=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.x, self.y = 0.0, 0.0
        self.heading = 0.5 * math.pi                     # facing +y (north)
        self.target_present = target_present
        bearing = target_bearing_deg + (jitter_deg * rng.uniform(-1, 1)
                                        if jitter_deg else 0.0)
        if target_present:
            ang = self.heading + math.radians(bearing)
            self.tx = self.x + target_dist * math.cos(ang)
            self.ty = self.y + target_dist * math.sin(ang)
        else:
            self.tx = self.ty = None
        self.loom = loom
        self.loom_t0 = 0.5                               # s after episode start
        self.loom_rise_s = 1.5
        self.loom_max_deg = 70.0
        self.obstacle = obstacle                          # dict(x,y,r) or None
        self.t = 0.0
        self.escaped = False
        self.jump_count = 0

    # ---- visual state ------------------------------------------------
    def target_bearing_deg(self):
        if not self.target_present:
            return None
        ang = math.atan2(self.ty - self.y, self.tx - self.x)
        b = math.degrees(ang - self.heading)
        return wrap_deg(b)

    def target_distance(self):
        if not self.target_present:
            return None
        return math.hypot(self.tx - self.x, self.ty - self.y)

    def loom_angular_deg(self):
        if not self.loom:
            return 0.0
        if self.t < self.loom_t0:
            return 0.0
        frac = min((self.t - self.loom_t0) / self.loom_rise_s, 1.0)
        return 2.0 + frac * (self.loom_max_deg - 2.0)

    def visual_state(self):
        return dict(
            t=self.t,
            target_present=self.target_present,
            target_bearing_deg=self.target_bearing_deg(),
            # sign convention: bearing = target_angle - heading (wrapped);
            # POSITIVE = target to the LEFT (counterclockwise) of heading
            target_distance=self.target_distance(),
            loom_angular_deg=self.loom_angular_deg(),
        )

    # ---- dynamics ----------------------------------------------------
    def step(self, action, dt):
        a = action.upper()
        if a == "FORWARD":
            self._move(self.V_FWD * dt)
        elif a == "BACKWARD":
            self._move(-self.V_BWD * dt)
        elif a == "TURN_LEFT":
            self.heading += self.OMEGA_TURN * dt
            self._move(self.V_TURN * dt)
        elif a == "TURN_RIGHT":
            self.heading -= self.OMEGA_TURN * dt
            self._move(self.V_TURN * dt)
        elif a == "STOP":
            pass
        elif a == "JUMP":
            if self.jump_count == 0:
                self._move(self.JUMP_HOP)
                self.escaped = True
            self.jump_count += 1
        else:
            raise ValueError(f"unknown action {action}")
        self.t += dt
        self._clamp()

    def _move(self, d):
        self.x += d * math.cos(self.heading)
        self.y += d * math.sin(self.heading)

    def _clamp(self):
        self.x = min(max(self.x, -self.HALF), self.HALF)
        self.y = min(max(self.y, -self.HALF), self.HALF)

    def reached_target(self):
        return (self.target_present
                and self.target_distance() <= self.REACH_RADIUS)


# --------------------------------------------------------------------------
# visual sensory encoder
# --------------------------------------------------------------------------
class VisualSensoryEncoder:
    """Visual state -> sensory-channel Poisson rates (Hz).

    PRE-REGISTERED tuning (preregistration.json). Sign convention: bearing
    b = target_angle - heading, wrapped to (-180, 180], so POSITIVE b =
    target to the LEFT (counterclockwise) of the agent's heading.

      target channels: triangular hemifield tuning -
          rate_L = R * clamp((theta0 + b) / theta_w, 0, 1)
          rate_R = R * clamp((theta0 - b) / theta_w, 0, 1)
        with theta0=15 deg (bilateral centre zone), theta_w=30 deg (ramp).
        b=0 -> both sides at R/2 (bilateral FORWARD); b=+45 (target LEFT)
        -> LEFT channel only; b=-45 (target RIGHT) -> RIGHT channel only.
      looming channel: R_loom * clamp((ang - 15)/45, 0, 1)
    """

    def __init__(self, r_target=150.0, r_loom=150.0, theta0=15.0,
                 theta_w=30.0, loom_onset_deg=15.0, loom_sat_deg=60.0):
        self.r_target = r_target
        self.r_loom = r_loom
        self.theta0 = theta0
        self.theta_w = theta_w
        self.loom_onset_deg = loom_onset_deg
        self.loom_sat_deg = loom_sat_deg

    def rates(self, vis):
        out = {"target_left": 0.0, "target_right": 0.0, "looming": 0.0}
        b = vis.get("target_bearing_deg")
        if vis.get("target_present") and b is not None:
            b = max(min(b, 90.0), -90.0)
            out["target_left"] = self.r_target * max(
                0.0, min(1.0, (self.theta0 + b) / self.theta_w))
            out["target_right"] = self.r_target * max(
                0.0, min(1.0, (self.theta0 - b) / self.theta_w))
        ang = vis.get("loom_angular_deg", 0.0)
        if ang > self.loom_onset_deg:
            out["looming"] = self.r_loom * max(
                0.0, min(1.0, (ang - self.loom_onset_deg)
                         / (self.loom_sat_deg - self.loom_onset_deg)))
        return out

    def describe(self):
        return dict(r_target=self.r_target, r_loom=self.r_loom,
                    theta0_deg=self.theta0, theta_w_deg=self.theta_w,
                    loom_onset_deg=self.loom_onset_deg,
                    loom_sat_deg=self.loom_sat_deg)


# --------------------------------------------------------------------------
# open-loop probes (CLI)
# --------------------------------------------------------------------------
def run_probe(brain, condition, rates_by_phase, readout_pops, chunk_ms=50.0):
    """One open-loop probe: phases of (duration_ms, rates). Full logging."""
    log = dict(condition=condition, phases=[], chunk_ms=chunk_ms)
    iface = brain.iface
    log["stimulated_ids"] = {
        ch: [brain.i2fly[int(i)] for i in iface.targets[iface.slots[ch]]]
        for ch in iface.channel_names}
    t0 = time.perf_counter()
    onset_chunk = None
    first_dn = {}
    dn_names = [n for n in brain.pop_names
                if n.startswith(("P9", "MDN", "BPN", "RRN", "GF", "FG",
                                 "BB", "BRK", "DN_ALL", "DN_LEG"))]
    for dur, rates in rates_by_phase:
        n_chunks = int(round(dur / chunk_ms))
        phase_rec = dict(dur_ms=dur, rates=dict(rates), chunks=[])
        for k in range(n_chunks):
            r = brain.step(rates, chunk_ms=chunk_ms)
            counts = r.get("pop_counts", {})
            if any(rates.values()) and onset_chunk is None:
                onset_chunk = r["t_bio_s"] - chunk_ms / 1000.0
            if onset_chunk is not None:
                for n_ in dn_names:
                    if counts.get(n_, 0) > 0 and n_ not in first_dn:
                        first_dn[n_] = r["t_bio_s"]
            phase_rec["chunks"].append(dict(
                t_bio_s=r["t_bio_s"], n_spikes_new=r["n_spikes_new"],
                pop_counts=counts))
        # steady-state summary over the last half of the phase
        tail = phase_rec["chunks"][max(0, len(phase_rec["chunks"]) // 2):]
        dt_s = len(tail) * chunk_ms / 1000.0
        phase_rec["steady_rates_hz"] = {
            n_: round(sum(c["pop_counts"].get(n_, 0) for c in tail)
                      / (brain.pop_sizes[n_] * dt_s), 2)
            for n_ in brain.pop_names if brain.pop_sizes.get(n_)}
        log["phases"].append(phase_rec)
    log["wall_s"] = round(time.perf_counter() - t0, 2)
    log["first_dn_response_s"] = first_dn
    log["latency_ms"] = ({n_: round((t - onset_chunk) * 1000, 1)
                          for n_, t in first_dn.items()}
                         if onset_chunk is not None else {})
    log["latency_resolution_ms"] = chunk_ms
    log["n_spikes_total"] = sum(
        c["n_spikes_new"] for ph in log["phases"] for c in ph["chunks"])
    log["pop_sizes"] = brain.pop_sizes
    return log


def probe_conditions(ladder_rung=1, seed=11):
    """Pre-registered open-loop probe set for interface calibration.

    Ladder (rungs tested in order; the SMALLEST rung that satisfies the
    pre-registered P9-recruitment criterion is frozen for D10):
      rung 1: 40 LC9/side @ 150 Hz
      rung 2: full per-side LC9 (87/92) @ 150 Hz
      rung 3: full per-side LC9 @ 250 Hz
    Criterion (rung accepted): steady-state ipsilateral P9 rate >= 20 Hz
    AND (P9_ipsi - P9_contra) >= 15 Hz on BOTH sides.
    """
    base = [("baseline", [(1000, {})]),
            ("loom", [(1000, {"looming": 150.0})]),
            ("latency_probe", [(500, {}), (1000, {"target_right": 150.0})])]
    rungs = {1: (40, 150.0), 2: "all", 3: "all@250"}
    conds = dict(base)
    if ladder_rung in (1, 3):
        n, rate = (40, 150.0) if ladder_rung == 1 else (None, 250.0)
        conds.update({
            "target_left": [(1000, {"target_left": rate})],
            "target_right": [(1000, {"target_right": rate})],
            "target_center": [(1000, {"target_left": rate,
                                      "target_right": rate})],
        })
        conds["_ladder"] = dict(n_per_side=n, rate=rate)
    return conds


def main():
    ap = argparse.ArgumentParser(description="D8 sensory encoder tool")
    ap.add_argument("cmd", choices=["build-interface", "build-readout",
                                    "probe"])
    ap.add_argument("--n-lc9-per-side", type=int, default=40)
    ap.add_argument("--rung", type=int, default=1,
                    help="ladder rung: 1=40@150, 2=all@150, 3=all@250")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--chunk-ms", type=float, default=50.0)
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args()

    if args.cmd == "build-interface":
        spec = build_interface_spec(n_lc9_per_side=args.n_lc9_per_side)
        # rung 2/3 override: full per-side LC9
        if args.rung >= 2:
            side_of = _side_map()
            lc9 = [int(i) for i in load_ids("LC9")]
            spec["channels"]["target_left"]["ids"] = sorted(
                i for i in lc9 if side_of.get(i) == "left")
            spec["channels"]["target_right"]["ids"] = sorted(
                i for i in lc9 if side_of.get(i) == "right")
            for ch, side in (("target_left", "left"),
                             ("target_right", "right")):
                spec["channels"][ch]["n"] = len(spec["channels"][ch]["ids"])
                spec["channels"][ch]["ladder_note"] = f"full side (rung {args.rung})"
            DATA.mkdir(parents=True, exist_ok=True)
            out = DATA / f"sensory_interface_rung{args.rung}.json"
            out.write_text(json.dumps(spec, indent=1))
            print(f"[interface] rung {args.rung} -> {out}")
    elif args.cmd == "build-readout":
        build_readout_spec()
    elif args.cmd == "probe":
        rung = args.rung
        iface_path = DATA / ("sensory_interface.json" if rung == 1
                             else f"sensory_interface_rung{rung}.json")
        interface_spec = str(iface_path)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        channel_ids = {ch: d["ids"] for ch, d in
                       json.loads(Path(interface_spec).read_text())["channels"].items()}
        readout_pops = load_readout_pops()
        rate = 150.0 if rung <= 2 else 250.0
        conds = {
            "baseline": [(1000, {})],
            "target_left": [(1000, {"target_left": rate})],
            "target_right": [(1000, {"target_right": rate})],
            "target_center": [(1000, {"target_left": rate,
                                      "target_right": rate})],
            "loom": [(1000, {"looming": 150.0})],
            "latency_probe": [(500, {}), (1000, {"target_right": rate})],
        }
        import d7_runtime as rt
        brain = rt.DualModeBrain("interactive", seed=args.seed,
                                 channel_ids=channel_ids,
                                 readout_pops=readout_pops)
        phases_by_name = dict(conds)
        summary = dict(seed=args.seed, rung=rung, rate_hz=rate,
                       n_neurons=brain.n_neurons,
                       interface=iface_path.name, conditions={})
        cond_names = list(conds.keys())
        for ci, name in enumerate(cond_names):
            brain.reset_episode(args.seed + 1000 + ci)   # deterministic per-condition seed
            log = run_probe(brain, name, phases_by_name[name], readout_pops,
                            chunk_ms=args.chunk_ms)
            (out_dir / f"probe_{name}_rung{rung}.json").write_text(
                json.dumps(log, indent=1))
            key = {k: log["phases"][-1]["steady_rates_hz"].get(k)
                   for k in ("P9_left", "P9_right", "GF_left", "GF_right",
                             "MDN_bilateral", "DN_ALL_left", "DN_ALL_right")}
            summary["conditions"][name] = dict(
                n_spikes=log["n_spikes_total"], latency_ms=log["latency_ms"],
                steady_hz=key)
            print(f"[probe] {name}: spikes={log['n_spikes_total']} "
                  f"P9L={key['P9_left']} P9R={key['P9_right']} "
                  f"lat={log['latency_ms']}")
        bl.save_json(out_dir / f"probe_summary_rung{rung}.json", summary)


if __name__ == "__main__":
    main()
