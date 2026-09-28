"""D11 - reward-modulated learning through the mushroom-body / dopamine
pathway of the canonical digital Drosophila brain (Shiu et al. LIF on
FlyWire v783, unmodified).

THE LEARNING RUNTIME (chunked whole-brain, on top of the Gate-3 closed-loop
runtime - NOT a replacement runtime):

  one subprocess per SESSION (many trials, one network build)
    per trial:  net.restore('ep0') -> seed(trial_seed) -> re-apply plastic
                KC->MBON weights -> closed-loop choice chunks (arena ->
                encoder -> brain 50 ms chunks -> decoder -> action) ->
                US window (sugar pulse via GRN entry) -> eligibility x
                dopamine plasticity update -> next trial

  Learning happens ONLY at the KC->MBON synapses of the canonical connectome
  (62,261 connections / 256,719 synapses, D5-verified) and is gated by the
  measured network activity of the dopaminergic reinforcement populations
  (PAM = reward/appetitive, PPL1 = punishment/aversive; D5-verified, 307/16
  neurons).  There is NO hidden reward->action shortcut anywhere: reward
  enters as GRN sugar stimulation, the brain's own PAM/PPL1 spiking gates
  plasticity, and behaviour is read out only from descending-neuron activity
  by the UNCHANGED Gate-3 decoder (D9).

TIER LABELS (brain/closedloop/__init__.py, docs/MASTER_SPEC.md section 2):
  BIOLOGICAL  connectome KC->MBON / PAM / PPL1 / MBON / APL / DPM wiring;
              MBON valence classes anchored to Aso 2014 / Owald 2015
              (brain/closedloop/populations.py, verified against v783 data).
  MODELED     the dopamine-gated depression rule, the eligibility trace,
              the per-trial decay toward canonical weights (extinction).
  ENGINEERED  the choice arena, the sugar-pulse US delivery schedule, the
              task's rewarded side, trial seeding, metrics.

Stages in this file:
  anatomy()    verify the interrupted-session anatomical probe against the
               restored repo data + new KC->MBON laterality and
               MBON->DN valence wiring tables (connectivity only, no brain).
  probes()     dynamic probes P1-P4 in ONE subprocess (build once, sequential
               resetEpisode schedules): sugar->PAM (P1), visual->KC side
               specificity (P2), KC->MBON weight potency (P3), MBON->DN
               expression route (P4).
  spec build   learning interface (Gate-3 channels + sugar) + learning
               readout (motor + KC/MBON/DAN populations).

Usage:
  python d11_learning.py anatomy
  python d11_learning.py specs
  python d11_learning.py probes [--seed 11]
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

BRAIN = bl.ROOT
DATA = BRAIN / "data" / "d11_d13"
RESULTS = BRAIN / "results"
IDS = BRAIN / "data" / "io_map" / "ids"
ANN = BRAIN / "data" / "flywire_annotations_v783"
CLOSEDLOOP = BRAIN.parent / "brain" / "closedloop"   # repo brain/closedloop

SAMPLING_SEED = 20260928          # Gate-2/3 sampling seed, kept for continuity
PROBE_SEED = 11                   # Gate-1/2/3 probe seed lineage


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def load_ids(pop):
    return [int(x) for x in json.loads((IDS / f"{pop}.json").read_text())]


def side_map():
    ann = pd.read_csv(ANN / "classification.csv.gz",
                      usecols=["root_id", "side"])
    return dict(zip(ann.root_id.astype("int64"), ann.side))


def type_map():
    typ = pd.read_csv(ANN / "fw_and_hemibrain_types.csv.gz",
                      usecols=["root_id", "cell_type", "hemibrain_type"])
    tm = {}
    for rid, ct, ht in zip(typ.root_id.astype("int64"),
                           typ.cell_type.fillna(""),
                           typ.hemibrain_type.fillna("")):
        tm[rid] = f"{ct},{ht}"
    return tm


def mbon_valence_classes():
    """MBON ids -> approach/avoidance/unclassified (BIOLOGICAL tier).

    Uses brain/closedloop/populations.py (Aso 2014 / Owald 2015 anchored
    classification, reapplied from the interrupted session after
    verification).  Returns dict with per-class id lists + sides.
    """
    sys.path.insert(0, str(BRAIN.parent / "brain"))     # repo/brain package
    from closedloop import populations as clp           # noqa: E402
    return clp.mbon_valence_classes()


# --------------------------------------------------------------------------
# KC->MBON synapse scan (shared by anatomy, probes and the learning brain)
# --------------------------------------------------------------------------
def scan_kc_mbon_connections():
    """Stream the canonical v783 connectivity parquet and return the
    KC->MBON synapse rows (BIOLOGICAL tier - these are the connectome's own
    synapses, the substrate of D11 plasticity).

    Returns dict with (all Brian-index space):
      rows          synapse-object row indices (order of df_con rows = order
                    of Synapses.connect, i.e. syn.w[row] is that synapse)
      pre / post    pre-/post-synaptic Brian indices (pre = KC)
      w_mult        'Excitatory x Connectivity' value (signed synapse count;
                    syn.w = w_mult * w_syn)
      fly_pre       KC flywire ids, aligned
      fly_post      MBON flywire ids, aligned
    """
    import pyarrow.parquet as pq

    flyid2i, _ = bl.load_maps("783")
    i2fly = {v: k for k, v in flyid2i.items()}
    kc_i = np.array(sorted(flyid2i[int(f)] for f in load_ids("KC")),
                    dtype=np.int64)
    mbon_i = np.array(sorted(flyid2i[int(f)] for f in load_ids("MBON")),
                      dtype=np.int64)

    rows, pre, post, wm = [], [], [], []
    pf = pq.ParquetFile(bl.VERSIONS["783"]["path_con"])
    row0 = 0
    wcol = "Excitatory x Connectivity"
    for batch in pf.iter_batches(columns=["Presynaptic_Index",
                                          "Postsynaptic_Index", wcol],
                                 batch_size=2_000_000):
        d = batch.to_pandas()
        p = d["Presynaptic_Index"].to_numpy()
        q = d["Postsynaptic_Index"].to_numpy()
        m = np.isin(p, kc_i) & np.isin(q, mbon_i)
        if m.any():
            idx = np.nonzero(m)[0]
            rows.append(row0 + idx)
            pre.append(p[idx])
            post.append(q[idx])
            wm.append(d[wcol].to_numpy()[idx])
        row0 += len(d)
    rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    pre = np.concatenate(pre) if pre else np.array([], dtype=np.int64)
    post = np.concatenate(post) if post else np.array([], dtype=np.int64)
    wm = np.concatenate(wm) if wm else np.array([], dtype=np.float64)
    return dict(rows=rows, pre=pre, post=post, w_mult=wm,
                fly_pre=[int(i2fly[i]) for i in pre],
                fly_post=[int(i2fly[i]) for i in post])


# --------------------------------------------------------------------------
# P0 - anatomy verification against the restored repo data
# --------------------------------------------------------------------------
def anatomy(out=RESULTS / "d11_anatomy" / "verify.json",
            reference=RESULTS / "d11_anatomy" / "probe.json"):
    """Re-derive the interrupted-session anatomical findings on the RESTORED
    repo data and compare. Adds: KC->MBON laterality split and MBON->DN
    valence wiring (needed to design the plasticity rule + expression)."""
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    con_path = bl.VERSIONS["783"]["path_con"]
    flyid2i, df_comp = bl.load_maps("783")
    i2fly = {v: k for k, v in flyid2i.items()}
    sides = side_map()

    kc_ids = load_ids("KC")
    mbon_ids = load_ids("MBON")
    pam_ids = load_ids("PAM")
    ppl1_ids = load_ids("PPL1")
    dn_ids = load_ids("DN_ALL")

    def to_bidx(fids):
        out, missing = [], []
        for f in fids:
            if int(f) in flyid2i:
                out.append(flyid2i[int(f)])
            else:
                missing.append(int(f))
        return out, missing

    kc_i, _m1 = to_bidx(kc_ids)
    mbon_i, _m2 = to_bidx(mbon_ids)
    dn_i, dn_missing = to_bidx(dn_ids)
    pam_n_present = sum(1 for f in pam_ids if int(f) in flyid2i)
    ppl1_n_present = sum(1 for f in ppl1_ids if int(f) in flyid2i)

    val = mbon_valence_classes()
    val_of = {}
    for cls in ("approach", "avoidance", "unclassified"):
        for f in val[cls]:
            if int(f) in flyid2i:
                val_of[flyid2i[int(f)]] = cls

    # ---- KC->MBON rows via the shared scan; MBON->DN streamed separately --
    sc = scan_kc_mbon_connections()
    kc2mb = list(zip(sc["rows"].tolist(), sc["pre"].tolist(),
                     sc["post"].tolist(), sc["w_mult"].tolist()))
    kc_mb_lateral = {"ipsi": 0, "contra": 0}
    mb2dn = []          # (i, j_dn)
    pf = pq.ParquetFile(con_path)
    mbon_i_arr = np.fromiter(mbon_i, dtype=np.int64)
    dn_i_arr = np.fromiter(dn_i, dtype=np.int64)
    for batch in pf.iter_batches(columns=["Presynaptic_Index",
                                          "Postsynaptic_Index"],
                                 batch_size=2_000_000):
        d = batch.to_pandas()
        pre = d["Presynaptic_Index"].to_numpy()
        post = d["Postsynaptic_Index"].to_numpy()
        m2 = np.isin(pre, mbon_i_arr) & np.isin(post, dn_i_arr)
        for r in np.nonzero(m2)[0]:
            mb2dn.append((int(pre[r]), int(post[r])))
    # laterality of KC->MBON (by FlyWire side annotation)
    for _row, i, j, _w in kc2mb:
        si = sides.get(i2fly[i], "?")
        sj = sides.get(i2fly[j], "?")
        if si == sj and si in ("left", "right"):
            kc_mb_lateral["ipsi"] += 1
        elif si != sj and si in ("left", "right") and sj in ("left", "right"):
            kc_mb_lateral["contra"] += 1

    # ---- compare against the interrupted-session probe --------------------
    ref = json.loads(Path(reference).read_text()) if Path(reference).exists() \
        else None
    checks = {}
    if ref:
        checks["sides_kc_n"] = (ref["sides"]["KC"]["n"] == len(kc_ids))
        checks["sides_mbon_n"] = (ref["sides"]["MBON"]["n"] == len(mbon_ids))
        checks["sides_pam_n"] = (ref["sides"]["PAM"]["n"] == len(pam_ids))
        checks["sides_ppl1_n"] = (ref["sides"]["PPL1"]["n"] == len(ppl1_ids))
        checks["mbon_types_n"] = (len(ref["mbon_types"]) == len(mbon_ids))
        n_direct_dn = len({i for i, _ in mb2dn})
        checks["mbon_to_dn_direct_types"] = (
            n_direct_dn == len(ref["mbon_to_dn_direct"]))

    # ---- MBON -> DN wiring by valence class (expression route table) ------
    mb2dn_valence = {"approach": 0, "avoidance": 0, "unclassified": 0}
    mb2dn_sign = {"approach": {"exc": 0, "inh": 0},
                  "avoidance": {"exc": 0, "inh": 0},
                  "unclassified": {"exc": 0, "inh": 0}}
    for i, j in mb2dn:
        v = val_of.get(i, "unclassified")
        mb2dn_valence[v] += 1

    out_d = dict(
        tier="BIOLOGICAL (connectivity only, no simulation)",
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        kc_n=len(kc_i), mbon_n=len(mbon_i), pam_n=len(pam_ids),
        pam_n_in_v783=pam_n_present, ppl1_n=len(ppl1_ids),
        ppl1_n_in_v783=ppl1_n_present, dn_n=len(dn_i),
        dn_missing_from_v783=dn_missing,
        kc_to_mbon=dict(
            n_connections=len(kc2mb),
            n_synapses=float(sum(w for _, _, _, w in kc2mb if w > 0)),
            ipsilateral_conns=kc_mb_lateral["ipsi"],
            contralateral_conns=kc_mb_lateral["contra"],
            note="synapse multiplicity = 'Excitatory x Connectivity' value"),
        mbon_to_dn_direct=dict(
            n_connections=len(mb2dn),
            n_mbon_types_with_dn_targets=len({i for i, _ in mb2dn}),
            by_valence_class=mb2dn_valence),
        checks_against_interrupted_session=checks,
        all_checks_pass=bool(checks) and all(checks.values()),
        wall_s=round(time.perf_counter() - t0, 1),
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(out_d, indent=1))
    print(json.dumps(out_d, indent=1)[:2000])
    return out_d


# --------------------------------------------------------------------------
# interface + readout specs for learning (ENGINEERED, frozen before matrix)
# --------------------------------------------------------------------------
def build_learning_interface_spec(out=DATA / "learning_interface.json"):
    """Gate-3 sensory channels + the sugar US entry (ENGINEERED delivery of
    the task reward through the BIOLOGICAL GRN sugar entry population)."""
    d7_spec = json.loads(
        (BRAIN / "data" / "d7_d10" / "sensory_interface.json").read_text())
    channels = {ch: dict(d) for ch, d in d7_spec["channels"].items()}
    sugar = load_ids("GRN_SUGAR")
    channels["sugar"] = dict(
        population="GRN_SUGAR", side="left", n=len(sugar), ids=sugar,
        biological_role="sugar taste (US) - engineered task reward delivery "
                        "through the biological sugar GRN entry")
    spec = dict(
        version=1, sampling_seed=d7_spec["sampling_seed"],
        interface_weight="w_syn * f_poi (model defaults, mirrors dbm.poi)",
        note="ENGINEERED sensory+reward interface; brain model unmodified; "
             "target_left/right + looming are the frozen Gate-3 channels",
        channels=channels)
    DATA.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(spec, indent=1))
    print(f"[interface] {sorted(channels)} -> {out}")
    return spec


def build_learning_readout_spec(out=DATA / "learning_readout.json"):
    """Motor readout (Gate-3 frozen) + MB learning populations, side-split.

    New populations (BIOLOGICAL, io_map IDs + v783 sides):
      KC_left/KC_right                      eligibility asymmetry readout
      MBON_approach_left/right (Aso 2014)   reward-prediction / valence
      MBON_avoidance_left/right             valence (negative)
      MBON_unclassified                     plasticity-excluded class
      PAM_left/right, PPL1_left/right       dopamine reinforcement rates
      APL_bilateral, DPM_bilateral          MB feedback
    """
    d9_spec = json.loads(
        (BRAIN / "data" / "d7_d10" / "motor_readout.json").read_text())
    pops = {k: list(v) for k, v in d9_spec["populations"].items()}
    sides = side_map()

    def split(pop):
        ids = load_ids(pop)
        L = [i for i in ids if sides.get(i) == "left"]
        R = [i for i in ids if sides.get(i) == "right"]
        U = [i for i in ids if sides.get(i) not in ("left", "right")]
        return L, R, U

    pops["KC_left"], pops["KC_right"], _u = split("KC")
    val = mbon_valence_classes()
    for cls in ("approach", "avoidance", "unclassified"):
        ids = [int(i) for i in val[cls]]
        pops[f"MBON_{cls}_left"] = [i for i in ids if sides.get(i) == "left"]
        pops[f"MBON_{cls}_right"] = [i for i in ids if sides.get(i) == "right"]
        pops[f"MBON_{cls}_unside"] = [i for i in ids
                                      if sides.get(i) not in ("left", "right")]
    pops["PAM_left"], pops["PAM_right"], _u = split("PAM")
    pops["PPL1_left"], pops["PPL1_right"], _u = split("PPL1")
    pops["APL_bilateral"] = load_ids("APL")
    pops["DPM_bilateral"] = load_ids("DPM")
    spec = dict(version=1, populations=pops,
                note="Gate-3 motor readout + MB learning populations "
                     "(io_map IDs, v783 sides, Aso-2014 valence classes)")
    Path(out).write_text(json.dumps(spec, indent=1))
    n = {k: len(v) for k, v in pops.items()}
    print(f"[readout] {len(pops)} populations -> {out}\n  {n}")
    return spec


def load_learning_readout_pops(path=DATA / "learning_readout.json"):
    return json.loads(Path(path).read_text())["populations"]


# --------------------------------------------------------------------------
# D11 choice arena + encoder (ENGINEERED, mirrors Gate-3 Arena2D/encoder)
# --------------------------------------------------------------------------
class ChoiceArena2D:
    """Choice / pairing environment (extends the Gate-3 Arena2D semantics).

    mode="choice"      two visually IDENTICAL targets at +/- bearing, equal
                       distance (the free-choice TEST trial of the D11
                       learning task).
    mode="pair_left"   only the LEFT target present (forced-exposure
                       PAIRING trial; the fly's innate approach behaviour
                       carries it to the target - no choice required).
    mode="pair_right"  only the RIGHT target present.

    The agent starts at the centre facing +y.  Kinematics are the frozen
    Gate-3 constants.  choice commit: |heading - heading0| >= CHOICE_DEG
    (registered side = sign of the heading deviation); timeout: no choice.
    """

    V_FWD, V_TURN, V_BWD = 0.25, 0.15, 0.15
    OMEGA_TURN = math.radians(150.0)
    HALF = 1.0
    CHOICE_DEG = 40.0

    def __init__(self, bearing_deg=50.0, dist=0.8, jitter_deg=8.0, seed=0,
                 mode="choice"):
        rng = np.random.default_rng(seed)
        self.mode = mode
        self.x, self.y = 0.0, 0.0
        self.heading = 0.5 * math.pi
        self.heading0 = self.heading
        self.t = 0.0
        self.jump_count = 0
        self.escaped = False
        b = bearing_deg + jitter_deg * rng.uniform(-1, 1)
        if mode in ("choice", "pair_left"):
            ang = self.heading + math.radians(+b)
            self.lx, self.ly = self.x + dist * math.cos(ang), \
                self.y + dist * math.sin(ang)
        else:
            self.lx, self.ly = None, None
        if mode in ("choice", "pair_right"):
            ang = self.heading + math.radians(-b)
            self.rx, self.ry = self.x + dist * math.cos(ang), \
                self.y + dist * math.sin(ang)
        else:
            self.rx, self.ry = None, None
        self.choice = None          # "left" / "right" / None
        self.choice_t = None

    # ---- visual state ---------------------------------------------------
    def _bearing(self, tx, ty):
        ang = math.atan2(ty - self.y, tx - self.x)
        b = math.degrees(ang - self.heading)
        while b > 180.0:
            b -= 360.0
        while b <= -180.0:
            b += 360.0
        return b

    def visual_state(self):
        return dict(
            t=self.t,
            bearing_left_deg=self._bearing(self.lx, self.ly)
            if self.lx is not None else None,
            bearing_right_deg=self._bearing(self.rx, self.ry)
            if self.rx is not None else None,
            dist_left=(math.hypot(self.lx - self.x, self.ly - self.y)
                       if self.lx is not None else None),
            dist_right=(math.hypot(self.rx - self.x, self.ry - self.y)
                        if self.rx is not None else None),
            loom_angular_deg=0.0)

    def _update_choice(self):
        if self.choice is None:
            dev = math.degrees(self.heading - self.heading0)
            if dev >= self.CHOICE_DEG:
                self.choice, self.choice_t = "left", self.t
            elif dev <= -self.CHOICE_DEG:
                self.choice, self.choice_t = "right", self.t

    # ---- dynamics --------------------------------------------------------
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
                self._move(0.30)
                self.escaped = True
            self.jump_count += 1
        else:
            raise ValueError(f"unknown action {action}")
        self.t += dt
        self.x = min(max(self.x, -self.HALF), self.HALF)
        self.y = min(max(self.y, -self.HALF), self.HALF)
        self._update_choice()

    def _move(self, d):
        self.x += d * math.cos(self.heading)
        self.y += d * math.sin(self.heading)


class ChoiceVisualEncoder:
    """Two-target version of the frozen Gate-3 triangular hemifield encoder.

    Each target contributes rate_L/R = R_side*clamp((theta0 +/- b)/theta_w);
    the two targets' contributions are summed and clipped at the side's max
    (so two targets at +/-b give a symmetric max drive on both channels -
    visually IDENTICAL targets, no side information except through which
    eye sees which target).

    Per-side rates r_target_left/right default to the Gate-3 frozen 150 Hz;
    the D11 bilateral calibration (d11_probes.calibrate_bilateral) may
    supply an asymmetric pair to cancel the frozen brain's innate P9
    asymmetry (applied IDENTICALLY to every condition; A-control checks).
    """

    def __init__(self, r_target=150.0, theta0=15.0, theta_w=30.0,
                 r_target_left=None, r_target_right=None):
        self.r_target = r_target
        self.r_left = float(r_target_left or r_target)
        self.r_right = float(r_target_right or r_target)
        self.theta0 = theta0
        self.theta_w = theta_w

    def rates(self, vis):
        out = {"target_left": 0.0, "target_right": 0.0, "looming": 0.0}
        for key, side in (("bearing_left_deg", "L"),
                          ("bearing_right_deg", "R")):
            b = vis.get(key)
            if b is None:
                continue
            b = max(min(b, 90.0), -90.0)
            if side == "L":
                out["target_left"] += self.r_left * max(
                    0.0, min(1.0, (self.theta0 + b) / self.theta_w))
                out["target_right"] += self.r_left * max(
                    0.0, min(1.0, (self.theta0 - b) / self.theta_w))
            else:
                out["target_left"] += self.r_right * max(
                    0.0, min(1.0, (self.theta0 + b) / self.theta_w))
                out["target_right"] += self.r_right * max(
                    0.0, min(1.0, (self.theta0 - b) / self.theta_w))
        out["target_left"] = min(out["target_left"], self.r_left)
        out["target_right"] = min(out["target_right"], self.r_right)
        return out

    def describe(self):
        return dict(r_target=self.r_target, theta0_deg=self.theta0,
                    theta_w_deg=self.theta_w, n_targets=2)


# --------------------------------------------------------------------------
# MODELED plasticity rule (dopamine-gated depression at KC->MBON)
# --------------------------------------------------------------------------
class MBPlasticity:
    """[MODELED] reward-modulated plasticity at the canonical KC->MBON
    synapses of the connectome.

    Biology being modelled (Owald & Waddell 2015; Aso 2014; Cohn 2015;
    Hige 2015; Davis 2023):
      * appetitive reinforcement (PAM dopamine) DEPRESSES KC->synapses onto
        AVOIDANCE-class MBONs for the KCs active in the CS context
        -> net approach bias for that context;
      * aversive reinforcement (PPL1) DEPRESSES KC->synapses onto
        APPROACH-class MBONs -> net avoidance bias;
      * memory decays back toward baseline (extinction / forgetting).

    Implementation: a per-synapse scale vector s in [s_min, 1] over the
    KC->MBON rows of the canonical connectivity; effective weight =
    canonical_w * s.  The vector lives OUTSIDE the Brian2 checkpoint, so
    net.restore('ep0') never rolls learning back - apply() re-imposes it.
    """

    def __init__(self, kc_mbon_rows, kc_mbon_pre, kc_mbon_post,
                 mbon_valence_of_post, n_kc,
                 eta=0.30, scale_min=0.05, tau_relax_trials=50.0,
                 tau_elig_s=2.0, elig_mode="spikes",
                 pam_gate_hz=5.0, ppl1_gate_hz=5.0, elig_power=2.0):
        self.rows = np.asarray(kc_mbon_rows, dtype=np.int64)
        self.pre = np.asarray(kc_mbon_pre, dtype=np.int64)
        self.post = np.asarray(kc_mbon_post, dtype=np.int64)
        self.post_valence = np.asarray(
            [1 if mbon_valence_of_post.get(int(p)) == "avoidance" else
             (-1 if mbon_valence_of_post.get(int(p)) == "approach" else 0)
             for p in self.post], dtype=np.int8)
        self.n_kc = n_kc
        self.eta = float(eta)
        self.scale_min = float(scale_min)
        self.tau_relax = float(tau_relax_trials)
        self.tau_elig_s = float(tau_elig_s)
        self.elig_mode = elig_mode
        self.pam_gate_hz = float(pam_gate_hz)
        self.ppl1_gate_hz = float(ppl1_gate_hz)
        # [MODELED] induction nonlinearity: eligibility enters the update
        # as (e/max)^elig_power.  elig_power=2 (frozen from pilot evidence:
        # linear attribution contaminated the non-rewarded context with
        # ~15% of the depression, blurring the learned asymmetry) models
        # the calcium-threshold nonlinearity of KC->MBON LTD induction.
        self.elig_power = float(elig_power)
        self.scale = np.ones(len(self.rows), dtype=np.float64)
        # which synapses each rule may touch
        self.mask_avoid = self.post_valence == 1      # PAM-reward target set
        self.mask_appr = self.post_valence == -1      # PPL1-punish target set
        self.total = len(self.rows)

    # ------------------------------------------------------------------
    def relax(self, n_trials=1):
        """[MODELED] passive decay toward canonical (forgetting/extinction)."""
        if self.tau_relax <= 0:
            return
        f = math.exp(-n_trials / self.tau_relax)
        self.scale = 1.0 - (1.0 - self.scale) * f

    def update(self, kc_elig, pam_hz, ppl1_hz, log=None):
        """One trial's plasticity step.

        kc_elig : (n_kc,) eligibility of each KC (spikes, or spikes +
                  subthreshold integral - MODELED trace with tau_elig_s
                  already applied by the caller).
        pam_hz / ppl1_hz : measured mean DAN rates in the US window
                  (BIOLOGICAL readout of the network's own reinforcement
                  activity).
        """
        reward = pam_hz >= self.pam_gate_hz
        punish = ppl1_hz >= self.ppl1_gate_hz
        changed = 0
        if len(kc_elig) and (reward or punish):
            e_max = float(np.max(kc_elig))
            if e_max > 0:
                # per-KC normalized eligibility with induction
                # nonlinearity (MODELED; see __init__)
                e_norm = np.power(
                    np.clip(kc_elig / e_max, 0.0, 1.0), self.elig_power)
                if reward:
                    m = self.mask_avoid
                    tgt = e_norm[self.pre[m]]
                    new = np.maximum(self.scale[m] * (1.0 - self.eta * tgt),
                                     self.scale_min)
                    changed += int(np.sum(new != self.scale[m]))
                    self.scale[m] = new
                if punish:
                    m = self.mask_appr
                    tgt = e_norm[self.pre[m]]
                    new = np.maximum(self.scale[m] * (1.0 - self.eta * tgt),
                                     self.scale_min)
                    changed += int(np.sum(new != self.scale[m]))
                    self.scale[m] = new
        if log is not None:
            log.update(dict(reward_gated=bool(reward), punish_gated=bool(
                punish), n_synapses_changed=changed))
        return changed

    # ------------------------------------------------------------------
    def stats(self):
        s = self.scale
        return dict(
            n_synapses=int(self.total),
            mean_scale=round(float(s.mean()), 6),
            frac_below_canonical=round(float(np.mean(s < 0.999)), 6),
            approach_class_mean=round(float(s[self.mask_appr].mean()), 6)
            if self.mask_appr.any() else None,
            avoidance_class_mean=round(float(s[self.mask_avoid].mean()), 6)
            if self.mask_avoid.any() else None,
            min_scale=round(float(s.min()), 6))

    def state_dict(self):
        return dict(scale=self.scale.tolist())

    def load_state_dict(self, d):
        self.scale = np.asarray(d["scale"], dtype=np.float64)
        assert len(self.scale) == self.total


# --------------------------------------------------------------------------
# eligibility trace helper
# --------------------------------------------------------------------------
class EligibilityTrace:
    """[MODELED] per-KC eligibility: exponential trace over the trial with
    tau_elig_s; fed by per-chunk KC spike counts (spike mode) and/or the
    end-of-chunk subthreshold depolarization max(v - v_0, 0) (volt mode).
    """

    def __init__(self, n_kc, tau_s=2.0, mode="spikes", v_weight=0.0):
        self.n = n_kc
        self.tau = float(tau_s)
        self.mode = mode
        self.v_weight = float(v_weight)
        self.e = np.zeros(n_kc, dtype=np.float64)

    def on_chunk(self, dt_s, kc_spikes=None, kc_v_depolarization=None):
        self.e *= math.exp(-dt_s / self.tau)
        if kc_spikes is not None and "spikes" in self.mode:
            self.e += kc_spikes
        if kc_v_depolarization is not None and "volt" in self.mode:
            self.e += self.v_weight * kc_v_depolarization

    def reset(self):
        self.e[:] = 0.0

    def value(self):
        return self.e.copy()


# --------------------------------------------------------------------------
# the learning brain: Gate-3 DualModeBrain + KC->MBON plasticity hooks
# --------------------------------------------------------------------------
class LearningBrain:
    """Chunked whole-brain runtime WITH reward-modulated KC->MBON
    plasticity, composed on top of the Gate-3 DualModeBrain (d7_runtime).

    Composition (NOT a replacement runtime): the underlying brain is the
    same canonical v783 network built by the pinned Shiu model; this class
    adds

      * the KC->MBON synapse-row map (connectivity scan, done BEFORE the
        network build so peak RAM stays the Gate-3 profile),
      * per-chunk KC spike counts + end-of-chunk KC subthreshold
        depolarization (harness READS of network state - no monitors that
        could perturb dynamics),
      * apply_plastic(scale) - re-imposes the MODELED weight scale onto the
        canonical KC->MBON synapses (call after every reset_episode: the
        'ep0' checkpoint holds pristine canonical weights),
      * verification that syn.w on those rows equals the canonical
        parquet weights (fails closed).

    The sensory interface, LIF dynamics, all other synapses, the spike
    monitor, the D9 decoder and the checkpoint protocol are UNCHANGED from
    Gate 3.
    """

    def __init__(self, seed=PROBE_SEED, channel_ids=None, readout_pops=None,
                 mbon_valence_of_flyid=None):
        import d7_runtime as rt

        # 1) connectivity scan BEFORE the build (RAM discipline)
        self._scan = scan_kc_mbon_connections()
        if mbon_valence_of_flyid is None:
            val = mbon_valence_classes()      # {class: [flywire ids]}
            self._val_of_fly = {}
            for cls in ("approach", "avoidance", "unclassified"):
                for f in val.get(cls, []):
                    self._val_of_fly[int(f)] = cls
        else:
            self._val_of_fly = {int(k): v
                                for k, v in mbon_valence_of_flyid.items()}

        # 2) the canonical Gate-3 brain (build + pristine 'ep0' checkpoint)
        self.brain = rt.DualModeBrain("interactive", seed=seed,
                                      channel_ids=channel_ids,
                                      readout_pops=readout_pops)

        # 3) KC tracking arrays
        kc_ids = load_ids("KC")
        self.kc_flyids = [int(f) for f in kc_ids if int(f) in
                          self.brain.flyid2i]
        self.kc_bidx = np.array([self.brain.flyid2i[f] for f in
                                 self.kc_flyids], dtype=np.int64)
        self.kc_local = np.full(self.brain.n_neurons, -1, dtype=np.int64)
        self.kc_local[self.kc_bidx] = np.arange(len(self.kc_bidx))
        self.kc_mask = np.zeros(self.brain.n_neurons, dtype=bool)
        self.kc_mask[self.kc_bidx] = True
        sides = side_map()
        self.kc_side = np.array(
            [1 if sides.get(f) == "left" else (-1 if sides.get(f) == "right"
                                               else 0)
             for f in self.kc_flyids], dtype=np.int8)
        self.kc_left_local = np.nonzero(self.kc_side == 1)[0]
        self.kc_right_local = np.nonzero(self.kc_side == -1)[0]

        # 4) KC->MBON synapse rows + canonical weights (fail-closed check)
        import brian2 as _b2
        self.kc_rows = self._scan["rows"]
        w_syn = float(self.brain.params["w_syn"] / _b2.volt)   # volts
        self.kc_canon_w = self._scan["w_mult"] * w_syn
        live = np.asarray(self.brain.syn.w[self.kc_rows])
        if not np.allclose(live, self.kc_canon_w, rtol=1e-9, atol=1e-12):
            raise RuntimeError(
                "KC->MBON canonical weight mismatch between parquet and "
                "live Synapses - refusing to run (fail closed)")
        self.kc_post_valence = np.array(
            [self._val_of_fly.get(int(f), "unclassified")
             for f in self._scan["fly_post"]], dtype=object)

        self.v_rest = float(self.brain.params["v_0"] / _b2.volt)
        self._cursor = 0

    # ---------------- delegated Gate-3 API ----------------
    def __getattr__(self, name):
        if name == "brain":
            raise AttributeError(name)          # pre-init guard
        return getattr(self.brain, name)

    def reset_episode(self, seed):
        self.brain.reset_episode(seed)
        self._cursor = 0

    def apply_plastic(self, scale):
        """Impose the MODELED weight scale on the canonical KC->MBON rows."""
        import brian2 as _b2
        arr = self.kc_canon_w * np.asarray(scale)     # volts (float)
        self.brain.syn.w[self.kc_rows] = arr * _b2.volt   # Quantity (V)

    # ---------------- learning-chunk stepping ----------------
    def step_learn(self, rates=None, chunk_ms=50.0):
        """Advance one chunk; returns the Gate-3 chunk record PLUS
        kc_spike_counts (per KC), kc_v_depolarization_mv (per KC),
        kc_side_summary {left,right} spikes + mean depolarization."""
        prev = int(self._cursor)
        rec = self.brain.step(rates, chunk_ms=chunk_ms)
        self._cursor = int(len(np.asarray(self.brain.mon.t[:])))

        i_new = np.asarray(self.brain.mon.i[prev:])
        if len(i_new):
            sel = i_new[self.kc_mask[i_new]]
            kc_counts = np.bincount(self.kc_local[sel],
                                    minlength=len(self.kc_bidx))
        else:
            kc_counts = np.zeros(len(self.kc_bidx), dtype=np.int64)
        v = np.asarray(self.brain.neu.v_[:])[self.kc_bidx]
        dep_mv = np.maximum((v - self.v_rest) * 1e3, 0.0)   # mV above rest

        rec["kc_spikes_total"] = int(kc_counts.sum())
        rec["kc_left_spikes"] = int(kc_counts[self.kc_left_local].sum())
        rec["kc_right_spikes"] = int(kc_counts[self.kc_right_local].sum())
        rec["kc_left_dep_mv"] = round(float(
            dep_mv[self.kc_left_local].mean()), 4)
        rec["kc_right_dep_mv"] = round(float(
            dep_mv[self.kc_right_local].mean()), 4)
        rec["_kc_counts"] = kc_counts           # stripped before logging
        rec["_kc_dep_mv"] = dep_mv
        return rec


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["anatomy", "specs", "probes"])
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    args = ap.parse_args()
    if args.cmd == "anatomy":
        anatomy()
    elif args.cmd == "specs":
        build_learning_interface_spec()
        build_learning_readout_spec()
    elif args.cmd == "probes":
        from d11_probes import run_probes     # separate module (heavy)
        run_probes(seed=args.seed)


if __name__ == "__main__":
    main()
