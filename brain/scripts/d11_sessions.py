"""D11 learning sessions: the reward-modulated learning matrix on the
canonical digital Drosophila brain.

THE TASK (first learning task, pre-registered): LEFT-vs-RIGHT target choice.
Two visually IDENTICAL targets at +/-50 deg; the agent commits by turning
+/-40 deg; committing toward the rewarded side triggers the US (1.0 s PAM
stimulation, the dopamine reward entry).  The frozen brain has an INNATE
right-turn bias under bilateral target drive (v783 LC9->P9 wiring
asymmetry: P9_left 26 Hz vs P9_right 58 Hz at 150 Hz drive - probe P2),
so LEFT-rewarded sessions are the strong test (learned preference must
OVERCOME the innate bias) and RIGHT-rewarded sessions the alignment case.

THE LEARNING PATHWAY (three labelled tiers, brain/closedloop/__init__.py):

  [BIOLOGICAL]  KC->MBON synapses of the connectome (62,261 conns, 77%
                ipsilateral); PAM/PPL1 DAN populations; MBON valence
                classes (Aso 2014 / Owald 2015); APL/DPM.
  [MODELED]     dopamine-gated depression rule (reward=PAM depresses
                KC->avoidance-MBON; punishment=PPL1 depresses
                KC->approach-MBON), eligibility trace (tau_e), per-trial
                relaxation toward canonical weights (extinction/forgetting).
  [ENGINEERED]  choice arena + kinematics; CS delivery into per-side KC
                ensembles (kc_cs_left/right, seeded 200/side) because the
                pinned model's LC9->KC path is anatomically present but
                dynamically silent (probes P2/P3); US delivery as direct
                PAM stimulation (sugar->PAM anatomically present but
                dynamically silent, probe P1; the standard optogenetic-
                style substitute, pre-registered D4-D6 3.3 tier-3a); the
                MBON valence-bias readout (D4-D6 3.3 tier-3c; probe P4:
                MBON->DN wiring does not reach walk command populations).
                Reward NEVER touches action directly: reward enters ONLY
                as DAN stimulation; behaviour is read ONLY from DN/MBON
                spike counts.

CONDITIONS:
  B_plastic          fly reward plasticity (the real thing)
  A_no_plastic       frozen Gate-3 weights, US still delivered (no learning)
  C_shuffled_da      dopamine gates lagged one trial (breaks CS-US pairing)
  D_shuffled_kcmbon  plasticity applied through a seeded permuted
                     eligibility->KC mapping (same depression, wrong
                     KC->MBON correspondence)
  E_rl_baseline      conventional 2-armed-bandit Q-learner on the same
                     schedule - explicitly NON-biological comparison

Usage (each session runs in its own subprocess, one at a time):
  python d11_sessions.py prereg [--beta X --eta Y --tau-relax N]
  python d11_sessions.py pilot   [--trials 12]
  python d11_sessions.py matrix
  python d11_sessions.py rl
  python d11_sessions.py analyze
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402
import d11_learning as d11  # noqa: E402
import d9_decoder as d9  # noqa: E402

DATA = bl.ROOT / "data" / "d11_d13"
RESULTS = bl.ROOT / "results" / "d11_learning"
PY = sys.executable
CHUNK_MS = 50.0
DT = CHUNK_MS / 1000.0

CS_SEED = 20260928
N_CS_PER_SIDE = 200
US_RATE_HZ = 150.0
US_DURATION_S = 1.0
CHOICE_MAX_S = 2.5
POST_S = 0.3
PAM_N = 307
PPL1_N = 16

# ---------------------------------------------------------------------------
# Session plans: FORCED-PAIRING blocks + FREE-CHOICE test trials
# (the design real fly memory assays use - Aso 2014 style: pair the CS with
#  reinforcement in forced exposure, then read out the PREFERENCE in a
#  free two-target choice WITHOUT reinforcement.  This dissolves the
#  exploration problem discovered by the pilots: the frozen brain's
#  bilateral regime is a decision boundary (calibrated balance -> STOP
#  timeouts; any imbalance -> one-sided basin), so instrumental
#  left-vs-right learning from rare exploratory commits is not viable;
#  classical pairing + choice tests is.)
#
# Trial kinds:
#   pair_<side>   single-target forced exposure (innate approach carries
#                 the fly); US (PAM 1 s) fires at commit toward the
#                 presented side when us_on
#   test          two targets, NO US, choice recorded (preference readout)
# ---------------------------------------------------------------------------
PHASES_B_RIGHT = [
    dict(name="acquisition", kind="blocks", blocks=6,
         pair_per_block=4, test_per_block=3, pair_side="right", us_on=True),
    dict(name="extinction", kind="test_only", n=12),
    dict(name="reversal", kind="blocks", blocks=6,
         pair_per_block=4, test_per_block=3, pair_side="left", us_on=True),
]
PHASES_CONTROLS_RIGHT = [
    dict(name="acquisition", kind="blocks", blocks=6,
         pair_per_block=4, test_per_block=3, pair_side="right", us_on=True),
]


def _expand_phases(plan):
    """Phase plan -> explicit trial list [{'kind','pair_side','phase'},...]."""
    trials = []
    for ph in plan:
        if ph["kind"] == "blocks":
            for _ in range(ph["blocks"]):
                for _ in range(ph["pair_per_block"]):
                    trials.append(dict(phase=ph["name"],
                                       kind=f"pair_{ph['pair_side']}",
                                       pair_side=ph["pair_side"],
                                       us_on=ph["us_on"]))
                for _ in range(ph["test_per_block"]):
                    trials.append(dict(phase=ph["name"], kind="test",
                                       pair_side=None, us_on=False))
        elif ph["kind"] == "test_only":
            for _ in range(ph["n"]):
                trials.append(dict(phase=ph["name"], kind="test",
                                   pair_side=None, us_on=False))
        else:
            raise ValueError(ph)
    return trials


# --------------------------------------------------------------------------
# production interface (probe-informed, ENGINEERED entries labelled)
# --------------------------------------------------------------------------
def cs_ensembles(seed=CS_SEED, n_per_side=N_CS_PER_SIDE):
    """Seeded per-side KC ensembles (the CS representation)."""
    sides = d11.side_map()
    kc = d11.load_ids("KC")
    rng = np.random.default_rng(seed)
    out = {}
    for side in ("left", "right"):
        cand = np.array([i for i in kc if sides.get(int(i)) == side])
        pick = rng.choice(cand, size=min(n_per_side, len(cand)),
                          replace=False)
        out[side] = sorted(int(x) for x in pick)
    return out


def build_production_interface_spec(out=DATA / "learning_interface.json"):
    """Gate-3 visual channels + CS ensembles + PAM US entry (+ sugar kept
    on record as the measured-dead biological route).  v2."""
    ens = cs_ensembles()
    old = json.loads((DATA / "learning_interface.json").read_text()) \
        if (DATA / "learning_interface.json").exists() else {"channels": {}}
    keep = {k: old["channels"][k] for k in
            ("target_left", "target_right", "looming")
            if k in old.get("channels", {})}
    spec = dict(
        version=2, sampling_seed=CS_SEED,
        interface_weight="w_syn * f_poi (model defaults, mirrors dbm.poi)",
        note="ENGINEERED learning interface. CS = per-side KC ensembles "
             "(probes P2/P3: LC9->KC anatomically present, dynamically "
             "silent); US = direct PAM stimulation (probe P1: sugar->PAM "
             "anatomically present, dynamically silent; standard "
             "optogenetic-style substitute, D4-D6 3.3 tier-3a). Reward "
             "NEVER touches action directly.",
        channels={
            **keep,
            "kc_cs_left": dict(
                population="KC left-hemisphere ensemble (seeded)", n=200,
                ids=ens["left"],
                biological_role="CS: left-context representation "
                                "(ENGINEERED entry, see note)"),
            "kc_cs_right": dict(
                population="KC right-hemisphere ensemble (seeded)", n=200,
                ids=ens["right"],
                biological_role="CS: right-context representation "
                                "(ENGINEERED entry, see note)"),
            "pam": dict(
                population="PAM", n=307, ids=d11.load_ids("PAM"),
                biological_role="US: dopamine reward entry (ENGINEERED, "
                                "optogenetic-style; D4-D6 3.3 tier-3a)"),
            "sugar": dict(
                population="GRN_SUGAR", n=20, ids=d11.load_ids("GRN_SUGAR"),
                biological_role="biological sugar entry - retained on "
                                "record, measured dynamically dead to PAM "
                                "(probe P1); NOT used as the US"),
        })
    DATA.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(spec, indent=1))
    print(f"[interface] production learning interface v2 -> {out}")
    return spec


class CSEncoder:
    """Fixation-gated CS ensemble rates (ENGINEERED, pre-registered).

    Each side's KC ensemble is driven by that side's target with a foveal
    tuning  rate_side = R_side * clamp(1 - |b_side| / theta_edge, 0, 1),
    theta_edge = 90:

      * symmetric start (|b|=50 for both): both ensembles ON at ~0.5R -
        both contexts present (like both odors at a T-maze junction), so
        the learned MBON valence asymmetry is readable by the decoder
        BEFORE the choice (expression);
      * during approach of one side, that target nears the fovea (|b|->0,
        rate -> R_side) while the other recedes to the periphery (|b|->90+,
        rate -> 0) - so eligibility at US time is dominated by the
        APPROACHED side's context (correct credit assignment).

    R_side defaults to 150; the D11 bilateral calibration may supply an
    asymmetric pair (expressed as PRE-COMMIT interface rates at |b|=50,
    converted here to full-scale R_side = precommit_hz / (1 - 50/90)).
    """

    def __init__(self, r_cs=150.0, theta_edge_deg=90.0,
                 precommit_hz_left=None, precommit_hz_right=None):
        self.theta_edge = theta_edge_deg
        f0 = 1.0 - 50.0 / theta_edge_deg         # tuning at |b|=50
        self.r_left = (float(precommit_hz_left) / f0
                       if precommit_hz_left else float(r_cs))
        self.r_right = (float(precommit_hz_right) / f0
                        if precommit_hz_right else float(r_cs))

    def rates(self, vis):
        out = {"kc_cs_left": 0.0, "kc_cs_right": 0.0}
        b_l = vis.get("bearing_left_deg")
        if b_l is not None:
            out["kc_cs_left"] = self.r_left * max(
                0.0, min(1.0, 1.0 - abs(b_l) / self.theta_edge))
        b_r = vis.get("bearing_right_deg")
        if b_r is not None:
            out["kc_cs_right"] = self.r_right * max(
                0.0, min(1.0, 1.0 - abs(b_r) / self.theta_edge))
        return out


# --------------------------------------------------------------------------
# decoder with the pre-registered MBON valence bias (D4-D6 3.3 tier-3c)
# --------------------------------------------------------------------------
class ValenceBiasDecoder(d9.MotorDecoder):
    """Gate-3 MotorDecoder + ONE addition (all else identical): the turn
    differential receives a valence bias

        bias = beta * [(appr_L - avoid_L) - (appr_R - avoid_R)]

    computed ONLY from MBON population spike counts in the same sliding
    window.  The bias NEVER sees target positions, the rewarded side, or
    the US schedule.  Rationale (probe P4): MBON valence biases approach/
    avoidance behaviour in the animal (Aso 2014), but this model's
    MBON->DN synapses do not reach the walk command neurons dynamically,
    so the harness reads the valence explicitly (pre-registered tier-3c).
    """

    def __init__(self, pop_sizes, params=None, beta=0.0):
        super().__init__(pop_sizes, params)
        self.beta = float(beta)
        self.bias_history = []
        self.valence_history = []
        # D13 condition B: EXTERNAL episodic-memory bias (ENGINEERED
        # harness module - never credited to the brain).  0 in D11.
        self.external_bias_hz = 0.0

    def decode(self, pop_counts, chunk_ms=50.0):
        self._push(pop_counts, chunk_ms)          # parent window logic
        rates = self._rates()
        p = self.p

        gf = rates.get("GF_left", 0) + rates.get("GF_right", 0)
        gf = gf / 2.0 if (self.pop_sizes.get("GF_left", 0)
                          + self.pop_sizes.get("GF_right", 0)) > 0 else gf
        mdn = rates.get("MDN_bilateral", 0.0)
        walk = [rates.get(k, 0.0) for k in d9.WALK_POPS]
        walk_mean = float(np.mean(walk)) if walk else 0.0
        stop_off = float(np.mean([rates.get(k, 0.0) for k in d9.STOP_POPS]))

        # ---- THE ONE ADDITION: MBON valence bias on the turn differential
        appr_l = rates.get("MBON_approach_left", 0.0)
        appr_r = rates.get("MBON_approach_right", 0.0)
        avoid_l = rates.get("MBON_avoidance_left", 0.0)
        avoid_r = rates.get("MBON_avoidance_right", 0.0)
        valence_bias = self.beta * ((appr_l - avoid_l)
                                    - (appr_r - avoid_r))
        # D13 condition B: the EXTERNAL memory bias adds to the turn
        # differential (labelled ENGINEERED; 0 unless explicitly set)
        valence_bias += self.external_bias_hz
        self.bias_history.append(valence_bias)
        self.valence_history.append(dict(appr_l=appr_l, appr_r=appr_r,
                                         avoid_l=avoid_l, avoid_r=avoid_r))

        if p["turn_source"] == "P9":
            diff = (rates.get("P9_left", 0.0)
                    - rates.get("P9_right", 0.0))
        else:
            diff = (rates.get("DN_ALL_left", 0.0)
                    - rates.get("DN_ALL_right", 0.0))
        diff_total = diff + valence_bias

        intents = {}
        intents["JUMP"] = gf > p["theta_jump_hz"]
        intents["BACKWARD"] = mdn > p["theta_bwd_hz"]
        intents["FORWARD"] = walk_mean > p["theta_fwd_hz"]
        intents["STOP_WALK_OFF"] = stop_off > p["theta_stop_hz"]
        intents["STOP_FAILSAFE"] = (walk_mean < p["theta_min_walk_hz"]
                                    and mdn < p["theta_bwd_hz"]
                                    and gf < p["theta_jump_hz"])

        prev_dir = 0
        if self.last_action == "TURN_LEFT":
            prev_dir = 1
        elif self.last_action == "TURN_RIGHT":
            prev_dir = -1
        turn = 0
        if diff_total > p["theta_diff_hz"] and (prev_dir >= 0 or diff_total
                                                > p["theta_diff_hz"]
                                                + p["delta_hyst_hz"]):
            turn = 1
        elif diff_total < -p["theta_diff_hz"] and (prev_dir <= 0 or
                                                   diff_total <
                                                   -p["theta_diff_hz"]
                                                   - p["delta_hyst_hz"]):
            turn = -1
        intents["TURN_LEFT"] = turn == 1
        intents["TURN_RIGHT"] = turn == -1

        active = [k for k, v in intents.items() if v]
        conflicts = []
        if "BACKWARD" in active and ("FORWARD" in active or turn != 0):
            conflicts.append("FORWARD/TURN vs BACKWARD")
        if intents["STOP_WALK_OFF"] and ("FORWARD" in active or turn != 0
                                         or "BACKWARD" in active):
            conflicts.append("walk-OFF vs walk-ON")
        if intents["JUMP"] and len(active) > 1:
            conflicts.append("JUMP vs other action")

        if intents["JUMP"]:
            action = "JUMP"
        elif intents["BACKWARD"] and not intents["FORWARD"] and turn == 0:
            action = "BACKWARD"
        elif intents["BACKWARD"] and intents["FORWARD"]:
            action = "BACKWARD" if mdn > walk_mean else (
                "BACKWARD" if turn == 0 and mdn >= walk_mean * 0.5 else
                ("TURN_LEFT" if turn == 1 else
                 "TURN_RIGHT" if turn == -1 else "FORWARD"))
        elif turn != 0:
            action = "TURN_LEFT" if turn == 1 else "TURN_RIGHT"
        elif intents["FORWARD"]:
            action = "FORWARD"
        else:
            action = "STOP"

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

        return dict(action=action,
                    rates_hz={k: round(v, 2) for k, v in rates.items()},
                    p9_diff_hz=round(diff, 2),
                    valence_bias_hz=round(valence_bias, 2),
                    diff_total_hz=round(diff_total, 2),
                    mbon_valence=dict(appr_l=round(appr_l, 2),
                                      appr_r=round(appr_r, 2),
                                      avoid_l=round(avoid_l, 2),
                                      avoid_r=round(avoid_r, 2)),
                    gf_hz=round(gf, 2), mdn_hz=round(mdn, 0),
                    walk_mean_hz=round(walk_mean, 2),
                    intents={k: bool(v) for k, v in intents.items()},
                    conflicts=conflicts)


# --------------------------------------------------------------------------
# one learning session (runs in its own subprocess; one network build)
# --------------------------------------------------------------------------
def run_session(cfg):
    t_start = time.perf_counter()
    cond = cfg["condition"]
    chunk_ms = cfg.get("chunk_ms", CHUNK_MS)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (DATA / "learning_interface.json").exists() or \
            json.loads((DATA / "learning_interface.json").read_text()
                       )["version"] < 2:
        build_production_interface_spec()
    channel_ids = {ch: d["ids"] for ch, d in
                   json.loads((DATA / "learning_interface.json").read_text()
                              )["channels"].items()}
    readout_pops = d11.load_learning_readout_pops()

    print(f"[session] building LearningBrain ({cond})...", flush=True)
    lb = d11.LearningBrain(seed=cfg["master_seed"],
                           channel_ids=channel_ids,
                           readout_pops=readout_pops)
    print(f"[session] build done ({time.perf_counter()-t_start:.0f}s); "
          f"KC->MBON rows={len(lb.kc_rows)}", flush=True)

    # MBON valence keyed by BRIAN index (post is a Brian index)
    val_by_bidx = {}
    for f, cls in lb._val_of_fly.items():
        if int(f) in lb.brain.flyid2i:
            val_by_bidx[lb.brain.flyid2i[int(f)]] = cls

    pl = d11.MBPlasticity(
        lb.kc_rows, lb.kc_local[lb._scan["pre"]],   # KC LOCAL indices
        lb._scan["post"],                            # Brian indices
        val_by_bidx,
        n_kc=len(lb.kc_bidx),
        eta=cfg.get("eta", 0.30),
        scale_min=cfg.get("scale_min", 0.05),
        tau_relax_trials=cfg.get("tau_relax_trials", 50.0),
        tau_elig_s=cfg.get("tau_elig_s", 2.0),
        elig_mode="spikes",
        pam_gate_hz=cfg.get("pam_gate_hz", 10.0),
        ppl1_gate_hz=cfg.get("ppl1_gate_hz", 10.0),
        elig_power=cfg.get("elig_power", 2.0))

    # ---- bilateral calibration gains (frozen pre-matrix, all conditions)
    calib_path = bl.ROOT / "results" / "d11_probes" / \
        "bilateral_calibration.json"
    cal = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    cal_left_vis = cfg.get("target_left_hz_override",
                           cal.get("target_left_hz_calibrated", 150.0))
    cal_right_vis = 150.0
    # pre-commit CS interface rates (targets at |b|=50): kc_cs_left from
    # calibration; kc_cs_right was the fixed reference (75 Hz)
    cal_left_cs = cal.get("kc_cs_left_hz_calibrated", 75.0)
    cal_right_cs = 75.0

    encoder = d11.ChoiceVisualEncoder(r_target_left=cal_left_vis,
                                      r_target_right=cal_right_vis)
    cs_encoder = CSEncoder(precommit_hz_left=cal_left_cs,
                            precommit_hz_right=cal_right_cs)
    # local KC indices of the CS ensembles (left ++ right, in channel order)
    cs_ensemble_local = np.array(
        [lb.kc_local[lb.brain.flyid2i[int(f)]]
         for f in (channel_ids["kc_cs_left"] + channel_ids["kc_cs_right"])],
        dtype=np.int64)
    dec_params = dict(window_ms=300.0, theta_jump_hz=25.0, theta_bwd_hz=4.0,
                      theta_fwd_hz=5.0, theta_diff_hz=8.0, delta_hyst_hz=4.0,
                      theta_stop_hz=4.0, theta_min_walk_hz=1.5,
                      min_action_chunks=2, turn_source="P9")
    beta = cfg.get("beta", 0.0)
    us_chunks = int(round(US_DURATION_S * 1000 / chunk_ms))

    trials = []
    # fresh trial log per session run (retries must not accumulate)
    (out_dir / "trials.jsonl").unlink(missing_ok=True)
    chunks_fh = (out_dir / "chunks.jsonl").open("w") if cfg.get(
        "save_chunks") else None
    trial_idx = 0
    prev_gate = (0.0, 0.0)          # (pam, ppl1) of previous trial (cond C)

    trial_seq = _expand_phases(cfg["phases"])
    for spec in trial_seq:
        kind, us_on = spec["kind"], spec["us_on"]
        pair_side = spec["pair_side"]
        phase_name = spec["phase"]
        t_trial0 = time.perf_counter()
        seed = cfg["master_seed"] + trial_idx
        lb.reset_episode(seed)
        lb.apply_plastic(pl.scale)
        arena = d11.ChoiceArena2D(seed=seed, mode=("choice" if kind == "test"
                                                   else kind))
        elig = d11.EligibilityTrace(len(lb.kc_bidx),
                                    tau_s=pl.tau_elig_s, mode="spikes")
        decoder = ValenceBiasDecoder(lb.brain.pop_sizes, dec_params,
                                     beta=beta)

        us_remaining = 0
        us_started = False
        pam_us_spikes = 0
        ppl1_us_spikes = 0
        us_bio_s = 0.0
        n_chunks = 0
        outcome = "timeout"
        # D12 internal-state telemetry (choice-window accumulators)
        choice_done = False
        pam_choice_spikes = 0
        ppl1_choice_spikes = 0
        dn_choice_spikes = 0
        choice_bio_s = 0.0

        while True:
            vis = arena.visual_state()
            rates = encoder.rates(vis)
            rates.update(cs_encoder.rates(vis))
            if us_remaining > 0:
                rates["pam"] = US_RATE_HZ
                us_remaining -= 1

            r = lb.step_learn(rates, chunk_ms=chunk_ms)
            elig.on_chunk(DT, kc_spikes=r["_kc_counts"])
            counts = dict(r.get("pop_counts", {}))
            dec = decoder.decode(counts, chunk_ms)
            arena.step(dec["action"], DT)
            n_chunks += 1

            if us_started:
                pam_us_spikes += (counts.get("PAM_left", 0)
                                  + counts.get("PAM_right", 0))
                ppl1_us_spikes += (counts.get("PPL1_left", 0)
                                   + counts.get("PPL1_right", 0))
                us_bio_s += DT
            if not choice_done:
                pam_choice_spikes += (counts.get("PAM_left", 0)
                                      + counts.get("PAM_right", 0))
                ppl1_choice_spikes += (counts.get("PPL1_left", 0)
                                       + counts.get("PPL1_right", 0))
                dn_choice_spikes += (counts.get("DN_ALL_left", 0)
                                     + counts.get("DN_ALL_right", 0))
                choice_bio_s += DT
                if arena.choice is not None:
                    choice_done = True

            if chunks_fh is not None:
                chunks_fh.write(json.dumps(dict(
                    trial=trial_idx, k=n_chunks - 1,
                    t=round(vis["t"], 3), action=dec["action"],
                    heading_deg=round(math.degrees(arena.heading), 1),
                    b_L=(round(vis["bearing_left_deg"], 1)
                         if vis["bearing_left_deg"] is not None else None),
                    b_R=(round(vis["bearing_right_deg"], 1)
                         if vis["bearing_right_deg"] is not None else None),
                    rates={c: round(v, 1) for c, v in rates.items()},
                    p9_diff=dec["p9_diff_hz"],
                    valence_bias=dec["valence_bias_hz"],
                    diff_total=dec["diff_total_hz"],
                    mbon=dec["mbon_valence"],
                    wall_ms=round(r["wall_s"] * 1000, 1))) + "\n")

            # US trigger at choice commit (pairing trials only: the
            # US follows commit toward the PRESENTED side; test trials
            # never deliver the US)
            if (kind != "test" and arena.choice is not None
                    and not us_started and us_on
                    and arena.choice == pair_side):
                us_started = True
                us_remaining = us_chunks

            committed = arena.choice is not None
            if committed:
                outcome = (f"commit_{arena.choice}_rewarded"
                           if us_started else f"commit_{arena.choice}")
            if committed and (not us_started) and (
                    arena.t >= arena.choice_t + POST_S):
                break
            if us_started and us_remaining == 0 and (
                    arena.t >= arena.choice_t + US_DURATION_S + POST_S):
                break
            if arena.t >= CHOICE_MAX_S + (US_DURATION_S
                                          if us_started else 0.0) \
                    + POST_S:
                outcome = "timeout"
                break

        wall_trial = time.perf_counter() - t_trial0
        pam_hz = (pam_us_spikes / (PAM_N * us_bio_s)
                  if us_bio_s > 0 else 0.0)
        ppl1_hz = (ppl1_us_spikes / (PPL1_N * us_bio_s)
                   if us_bio_s > 0 else 0.0)

        # ---- plasticity update (conditions differ) -----------------
        gate_pam, gate_ppl1 = pam_hz, ppl1_hz
        if cond == "C_shuffled_da":
            gate_pam, gate_ppl1 = prev_gate     # lag-1 (breaks pairing)
        prev_gate = (pam_hz, ppl1_hz)

        w_stats_before = pl.stats()
        pl.relax(1)
        up_log = {}
        if cond in ("B_plastic", "C_shuffled_da", "D_shuffled_kcmbon"):
            e = elig.value()
            if cond == "D_shuffled_kcmbon":
                # seeded permuted eligibility->KC attribution: the same
                # amount of depression lands on the WRONG KC->MBON
                # synapses (broken context->synapse mapping)
                perm = np.random.default_rng(
                    cfg["master_seed"] + 1000 + trial_idx
                ).permutation(len(e))
                e = e[perm]
            pl.update(e, gate_pam, gate_ppl1, up_log)
        else:
            up_log = dict(update_skipped=cond)

        # end-of-choice-window MBON valence (mean of last 6 chunks)
        tail = decoder.valence_history[-6:]
        mbon_end = {k: round(float(np.mean([v[k] for v in tail])), 2)
                    for k in ("appr_l", "appr_r", "avoid_l", "avoid_r")} \
            if tail else None

        rec = dict(
            trial=trial_idx, phase=phase_name, seed=seed, kind=kind,
            pair_side=pair_side,
            condition=cond, us_on=us_on,
            choice=arena.choice, outcome=outcome,
            choice_t=arena.choice_t,
            n_chunks=n_chunks, bio_s=round(arena.t, 2),
            wall_s=round(wall_trial, 1),
            pam_us_hz=round(pam_hz, 2), ppl1_us_hz=round(ppl1_hz, 2),
            us_delivered=us_started,
            pam_choice_hz=round(pam_choice_spikes
                                / (PAM_N * max(choice_bio_s, 1e-9)), 3),
            ppl1_choice_hz=round(ppl1_choice_spikes
                                 / (PPL1_N * max(choice_bio_s, 1e-9)), 3),
            dn_all_choice_hz=round(dn_choice_spikes
                                    / (1297 * max(choice_bio_s, 1e-9)), 3),
            valence_bias_end=(round(float(np.mean(
                decoder.bias_history[-6:])), 2)
                if decoder.bias_history else None),
            mbon_valence_end=mbon_end,
            weight_stats=w_stats_before,
            plasticity=up_log,
            kc_elig_left=round(float(np.mean(
                elig.value()[lb.kc_left_local])), 3),
            kc_elig_right=round(float(np.mean(
                elig.value()[lb.kc_right_local])), 3),
            kc_elig_ensemble=[round(float(x), 3) for x in
                              elig.value()[cs_ensemble_local]],
            calibration=dict(target_left_hz=cal_left_vis,
                             kc_cs_left_precommit_hz=cal_left_cs),
        )
        trials.append(rec)
        with (out_dir / "trials.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[trial {trial_idx:3d}] {phase_name:<12s} "
              f"{kind:<10s} choice={str(arena.choice):<5s} "
              f"us={int(us_started)} pam={pam_hz:5.1f} "
              f"bias={rec['valence_bias_end']} "
              f"avoid_mean={pl.stats()['avoidance_class_mean']}",
              flush=True)
        trial_idx += 1

    if chunks_fh is not None:
        chunks_fh.close()

    wall_total = time.perf_counter() - t_start
    summary = summarize_session(trials, cfg)
    summary.update(
        session_wall_s=round(wall_total, 1),
        peak_rss_mb=round(bl.peak_rss_kb() / 1024, 1),
        cpu_user_s=round(os.times()[0], 1),
        n_kc_mbon_synapses=int(len(pl.rows)),
        final_weight_stats=pl.stats(),
        tier_labels=dict(
            substrate="BIOLOGICAL KC->MBON/PAM/PPL1/MBON (canonical "
                      "connectome)",
            rule="MODELED dopamine-gated depression + eligibility + decay",
            task="ENGINEERED choice arena + CS/US entries + MBON valence "
                 "bias readout (learning_interface.json v2 + d11_probes)"),
    )
    (out_dir / "session.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out_dir / "weights.npz", scale=pl.scale)
    print(f"[session] {cond} done in {wall_total:.0f}s -> {out_dir}",
          flush=True)
    return summary


def summarize_session(trials, cfg):
    """Session-level learning metrics (pre-registered).

    Preference is read on FREE-CHOICE TEST trials only; pairing trials
    report the innate approach rate toward the presented target.
    """
    out = dict(condition=cfg["condition"], n_trials=len(trials))
    for phase in sorted({t["phase"] for t in trials}):
        xs = [t for t in trials if t["phase"] == phase]
        tests = [t for t in xs if t.get("kind") == "test"]
        pairs = [t for t in xs if t.get("kind") != "test"]
        decided = [t for t in tests if t["choice"] is not None]
        pair_side = next((t["pair_side"] for t in pairs
                          if t.get("pair_side")), None)

        def p(side, subset):
            return (round(sum(1 for t in subset if t["choice"] == side)
                          / len(subset), 3) if subset else None)

        blk = dict(
            n=len(xs), n_test=len(tests), n_pair=len(pairs),
            pair_side=pair_side,
            p_test_left=p("left", decided),
            p_test_right=p("right", decided),
            p_test_timeout=round(1 - len(decided) / len(tests), 3)
            if tests else None,
            pairing_approach_rate=round(
                sum(1 for t in pairs if t["choice"] == t["pair_side"])
                / len(pairs), 3) if pairs else None,
            n_us=sum(1 for t in xs if t.get("us_delivered")),
            mean_pam_us_hz=round(float(np.mean(
                [t["pam_us_hz"] for t in xs if t.get("us_delivered")]
                or [0])), 1),
        )
        if pair_side and decided:
            half = len(tests) // 2
            blk["p_paired_side_test_first_half"] = p(
                pair_side, [t for t in decided
                            if t["trial"] < tests[0]["trial"] + half])
            blk["p_paired_side_test_second_half"] = p(
                pair_side, [t for t in decided
                            if t["trial"] >= tests[0]["trial"] + half])
        out[f"phase_{phase}"] = blk
    return out


# --------------------------------------------------------------------------
# subprocess driver (one session at a time - RAM discipline)
# --------------------------------------------------------------------------
def spawn_session(cfg, quiet=False):
    cfg_path = Path(cfg["out_dir"]) / "_cfg.json"
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg))
    cmd = [PY, str(Path(__file__).resolve()), "_worker",
           "--cfg", str(cfg_path)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"session worker failed: {cfg['out_dir']}")
    summ = json.loads((Path(cfg["out_dir"]) / "session.json").read_text())
    summ["outer_wall_s"] = round(wall, 1)
    if not quiet:
        print(f"[matrix] {cfg['condition']} {cfg.get('label', '')} "
              f"finished in {wall:.0f}s")
    return summ


def _worker_main(cfg_path):
    cfg = json.loads(Path(cfg_path).read_text())
    run_session(cfg)


# --------------------------------------------------------------------------
# RL baseline E (explicitly NON-biological comparison)
# --------------------------------------------------------------------------
def run_rl_baseline(out_dir=RESULTS / "E_rl_baseline", master_seed=20261001,
                    plan=None, epsilon=0.1, alpha=0.3):
    """Conventional RL on the SAME pairing+test schedule: Q-learning with
    forced reward observation on pairing trials, epsilon-greedy choice on
    test trials (no test-trial updates - exactly what the fly experiences).
    NO brain, NO biological claim - the mandated conventional-RL
    comparison."""
    if plan is None:
        plan = PHASES_B_RIGHT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(master_seed)
    q = {"left": 0.5, "right": 0.5}
    forget = math.exp(-1.0 / 50.0)   # passive decay, tau=50 trials (matched
    #                                   to the fly's MODELED weight decay for
    #                                   a fair forgetting comparison)
    trials = []
    for spec in _expand_phases(plan):
        idx = len(trials)
        if spec["kind"] != "test" and spec["us_on"]:
            # pairing trial: forced reward observation on the presented side
            side = spec["pair_side"]
            r = 1.0
            q[side] += alpha * (r - q[side])
            choice = side                     # forced approach
        else:
            # test trial: free choice, no reward
            if rng.random() < epsilon:
                choice = str(rng.choice(["left", "right"]))
            else:
                choice = max(q, key=q.get)
            r = 0.0
        for k in q:
            q[k] = 0.5 + (q[k] - 0.5) * forget
        trials.append(dict(trial=idx, phase=spec["phase"], kind=spec["kind"],
                           pair_side=spec["pair_side"], us_on=spec["us_on"],
                           choice=choice, reward=r, q_left=q["left"],
                           q_right=q["right"],
                           us_delivered=bool(spec["kind"] != "test"
                                             and spec["us_on"]),
                           pam_us_hz=0.0, ppl1_us_hz=0.0))
    (out_dir / "trials.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trials))
    summ = summarize_session(trials, dict(condition="E_rl_baseline"))
    summ["note"] = ("conventional RL (epsilon-greedy Q-learner with forced "
                    "pairing observations), explicitly NON-biological")
    (out_dir / "session.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items()
                      if k.startswith("phase_")}, indent=1))
    return summ


# --------------------------------------------------------------------------
# pre-registration (frozen BEFORE the matrix)
# --------------------------------------------------------------------------
def write_preregistration(beta=1.5, eta=0.30, tau_relax=50.0,
                          scale_min=0.05, tau_elig_s=2.0, elig_power=2.0,
                          out=DATA / "preregistration.json",
                          pilot_note=None):
    calib_path = bl.ROOT / "results" / "d11_probes" / \
        "bilateral_calibration.json"
    cal = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    reg = dict(
        version=2,
        written_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        brain="canonical Shiu LIF model on FlyWire v783 (unmodified); "
              "chunked interactive runtime (d7_runtime) + LearningBrain "
              "(d11_learning)",
        design=dict(
            paradigm="forced-pairing blocks + free-choice tests (Aso-2014 "
                     "style); pilots showed the bilateral regime is a "
                     "decision boundary (balanced -> STOP timeouts; any "
                     "imbalance -> one-sided basin), so instrumental "
                     "left/right learning from exploratory commits is not "
                     "viable in the frozen brain - pairing decouples "
                     "learning from exploration",
            pairing_trial="single target (presented side), innate approach; "
                          "US = 1.0 s PAM @150 Hz at commit toward the "
                          "presented side",
            test_trial="two identical targets, NO US, choice = first "
                       "+/-40 deg heading commit (2.5 s cap)",
            plan=dict(
                acquisition="6 blocks x (4 pair RIGHT + 3 test)",
                extinction="12 test-only trials (no US anywhere)",
                reversal="6 blocks x (4 pair LEFT + 3 test)")),
        interface=dict(
            visual="Gate-3 channels at nominal 150 Hz both sides (strong "
                   "regime; pilot: innate LEFT test-choice 12/12, no "
                   "timeouts)",
            cs="per-side KC ensembles (200/side, seed 20260928), "
               "fixation-gated foveal tuning R*clamp(1-|b|/90); "
               "kc_cs_left PRE-COMMIT rate 102.3 Hz from the bilateral "
               "valence calibration (kc_cs_right 75 Hz reference) - the "
               "LC9->KC path is dynamically silent (probes P2/P3), and "
               "LC9 drive does not touch MBONs, so this calibration "
               "transfers to the strong visual regime",
            us="direct PAM stimulation (ENGINEERED optogenetic-style "
               "entry; probe P1: sugar->PAM dynamically silent; D4-D6 3.3 "
               "tier-3a)"),
        plasticity=dict(  # MODELED tier
            site="canonical KC->MBON synapses (62,261; connectome's own)",
            rule="reward(PAM gate >=10 Hz) depresses KC->avoidance-MBON; "
                 "punishment(PPL1 gate >=10 Hz) depresses KC->approach-MBON "
                 "(Aso 2014 / Owald 2015 sign rule)",
            eta=eta, scale_min=scale_min,
            tau_elig_s=tau_elig_s,
            elig_power=elig_power,
            elig_power_note="induction nonlinearity (e/max)^power, "
                            "power=2 from pilot cross-contamination "
                            "evidence",
            tau_relax_trials=tau_relax,
            update_timing="relax(1 trial) then update, at trial end"),
        decoder=dict(  # Gate-3 frozen params + tier-3c bias
            base="Gate-3 D9 MotorDecoder params verbatim",
            addition="valence bias beta*[(appr_L-avoid_L)-(appr_R-"
                     "avoid_R)] added to the P9 turn differential "
                     "(D4-D6 3.3 tier-3c; probe P4 evidence: MBON->DN "
                     "wiring does not reach walk command populations)",
            beta=beta,
            beta_note="readout gain frozen from pilot calibration"),
        sessions=dict(
            B=dict(condition="B_plastic", plan=PHASES_B_RIGHT,
                   master_seed=20260929,
                   note="pair RIGHT (against the innate LEFT test "
                        "preference) = the strong acquisition test; "
                        "extinction; reversal to LEFT"),
            A=dict(condition="A_no_plastic",
                   plan=PHASES_CONTROLS_RIGHT, master_seed=20261001,
                   note="frozen Gate-3 weights, US still delivered"),
            C=dict(condition="C_shuffled_da",
                   plan=PHASES_CONTROLS_RIGHT, master_seed=20261002,
                   note="dopamine gates lagged one trial (broken pairing)"),
            D=dict(condition="D_shuffled_kcmbon",
                   plan=PHASES_CONTROLS_RIGHT, master_seed=20261003,
                   note="plasticity through a seeded permuted "
                        "eligibility->KC mapping"),
            E_rl=dict(condition="E_rl_baseline", master_seed=20261001,
                       note="non-biological Q-learner on the same schedule"),
            seed_rule="trial seed = master_seed + trial_index"),
        metrics=["test-trial choice sequence",
                 "P(paired side | decided test trials) per phase + halves",
                 "pairing approach rate", "valence bias (Hz)",
                 "MBON appr/avoid per side", "PAM/PPL1 rates (US window)",
                 "KC eligibility asymmetry", "KC->MBON weight stats",
                 "US count", "wall/RAM/latency"],
        gate4_criteria=dict(
            G4_1="B acquisition: P(paired side | decided) second half > "
                 "first half AND exceeds the matched A control (bootstrap "
                 "95% CI on the difference excludes 0)",
            G4_2="A (no plasticity) shows NO acquisition trend",
            G4_3="C (shuffled DA) fails to acquire or is severely degraded",
            G4_4="D (shuffled KC/MBON) fails to acquire or is severely "
                 "degraded",
            G4_5="extinction: test preference declines toward baseline "
                 "when the US is withheld",
            G4_6="reversal: test preference flips to the other side "
                 "within the reversal phase",
            honesty="if any criterion fails, report FAIL; no post-hoc "
                    "parameter changes after the matrix starts"),
        pilot_note=pilot_note,
    )
    DATA.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(reg, indent=1))
    print(f"[prereg] frozen -> {out}")
    return reg


# --------------------------------------------------------------------------
# matrix + analysis + plots
# --------------------------------------------------------------------------
def run_matrix():
    reg_path = DATA / "preregistration.json"
    if not reg_path.exists():
        raise SystemExit("run `prereg` first (pre-registration must be "
                         "frozen before the matrix)")
    reg = json.loads(reg_path.read_text())
    pl_reg = reg["plasticity"]
    jobs = []
    for label in ("B", "A", "C", "D"):
        s = reg["sessions"][label]
        jobs.append((label, dict(
            condition=s["condition"],
            phases=s["plan"],
            master_seed=s["master_seed"],
            out_dir=str(RESULTS / label),
            beta=reg["decoder"]["beta"], eta=pl_reg["eta"],
            tau_relax_trials=pl_reg["tau_relax_trials"],
            scale_min=pl_reg["scale_min"],
            tau_elig_s=pl_reg["tau_elig_s"],
            elig_power=pl_reg["elig_power"],
            pam_gate_hz=10.0, ppl1_gate_hz=10.0,
            save_chunks=(label == "B"),      # chunk log for one session
            label=label)))
    for label, cfg in jobs:
        print(f"[matrix] === {label} ({cfg['condition']}) ===", flush=True)
        spawn_session(cfg)
    run_rl_baseline()
    analyze()


def analyze(out_dir=RESULTS):
    """Aggregate: test-trial learning curves, contrasts, plots, verdict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _f in ("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
               "/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf",
               "/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf"):
        try:
            fm.fontManager.addfont(_f)
            break
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    sessions = {}
    for label in ("B", "A", "C", "D", "E_rl_baseline"):
        p = Path(out_dir) / label / "trials.jsonl"
        if p.exists():
            sessions[label] = [json.loads(l) for l in
                               p.read_text().splitlines()]

    def pref_curve(trials, phase=None):
        """Test-trial preference FOR THE PHASE'S PAIRED SIDE."""
        xs = [t for t in trials if t.get("kind") == "test"
              and (phase is None or t["phase"] == phase)
              and t["choice"] is not None]
        ks, rs = [], []
        for t in xs:
            # rewarded side = the phase's pairing side (None in extinction
            # -> use the previous phase's side for continuity)
            side = t.get("pair_side")
            if side is None:
                side = {"acquisition": "right", "reversal": "left",
                        "extinction": "right"}.get(t["phase"], "right")
            ks.append(t["trial"])
            rs.append(1.0 if t["choice"] == side else 0.0)
        return ks, rs

    def smooth(rs, w=6):
        return [round(float(np.mean(rs[max(0, i - w + 1):i + 1])), 3)
                for i in range(len(rs))]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for label, tr in sessions.items():
        ks, rs = pref_curve(tr)
        if not ks:
            continue
        style = ("--" if label == "E_rl_baseline" else "-")
        ax.plot(ks, smooth(rs), style, label=label, alpha=0.9)
    ax.axhline(0.5, color="k", lw=0.5, alpha=0.4)
    ax.set(title="Acquisition: test-trial P(paired side | decided), "
                 "6-trial mean", xlabel="trial", ylabel="P(paired side)")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    if "B" in sessions:
        ks, rs = pref_curve(sessions["B"])
        if ks:
            ax.plot(ks, smooth(rs), label="B (full battery)")
        # phase boundaries
        ph_bounds = {}
        for t in sessions["B"]:
            ph_bounds.setdefault(t["phase"], t["trial"])
        for ph, k0 in ph_bounds.items():
            ax.axvline(k0, color="gray", lw=0.8, ls=":")
            ax.text(k0 + 1, 0.02, ph, fontsize=8, rotation=90)
    ax.set_ylim(-0.05, 1.05)
    ax.set(title="B: acquisition -> extinction -> reversal (test trials)",
           xlabel="trial", ylabel="P(paired side)")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    if "B" in sessions:
        tr = sessions["B"]
        xs = [t for t in tr if t.get("mbon_valence_end")]
        ks = [t["trial"] for t in xs]
        ax.plot(ks, [t["mbon_valence_end"]["avoid_r"] for t in xs],
                label="avoid_R (paired side)")
        ax.plot(ks, [t["mbon_valence_end"]["avoid_l"] for t in xs],
                label="avoid_L")
        ax.plot(ks, [t["valence_bias_end"] for t in xs], "k.",
                label="valence bias (Hz)", alpha=0.5)
        ax.set(title="B: MBON rates + decoder bias (MODELED readout)",
               xlabel="trial", ylabel="Hz")
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    if "B" in sessions:
        tr = sessions["B"]
        xs = [t for t in tr if t.get("weight_stats")]
        ks = [t["trial"] for t in xs]
        ax.plot(ks, [t["weight_stats"]["avoidance_class_mean"] for t in xs],
                label="KC->avoid-MBON mean scale")
        ax.plot(ks, [t["weight_stats"]["approach_class_mean"] for t in xs],
                label="KC->appr-MBON mean scale")
        ax.set_ylim(0, 1.02)
        ax.set(title="B: synaptic scales at the canonical KC->MBON site "
                     "(MODELED)", xlabel="trial")
        ax.legend(fontsize=8)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out_dir) / "learning_curves.png", dpi=130)
    print(f"[analyze] plot -> {out_dir / 'learning_curves.png'}")

    # ---------------- statistical contrasts ----------------
    def last_n_tests(trials, phase="acquisition", n=9):
        xs = [t for t in trials if t.get("kind") == "test"
              and t["phase"] == phase and t["choice"] is not None]
        return xs[-n:]

    rng = np.random.default_rng(20260928)
    contrasts = {}
    if "B" in sessions:
        b = last_n_tests(sessions["B"])
        if b:
            b_side = "right"     # acquisition pairing side (pre-registered)
            b_v = np.array([1.0 if t["choice"] == b_side else 0.0
                            for t in b])
            contrasts["B_acq_last9_P_paired"] = round(float(b_v.mean()), 3)
            for ctrl in ("A", "C", "D"):
                if ctrl in sessions:
                    c = last_n_tests(sessions[ctrl])
                    if not c:
                        continue
                    c_v = np.array([1.0 if t["choice"] == b_side else 0.0
                                    for t in c])
                    obs = float(b_v.mean() - c_v.mean())
                    boot = []
                    for _ in range(5000):
                        mb = rng.integers(0, len(b_v), len(b_v))
                        mc = rng.integers(0, len(c_v), len(c_v))
                        boot.append(b_v[mb].mean() - c_v[mc].mean())
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    contrasts[f"B_vs_{ctrl}_last9_diff"] = dict(
                        observed=round(obs, 3),
                        ci95=[round(lo, 3), round(hi, 3)],
                        significant=bool(lo > 0 or hi < 0))
        # extinction + reversal checks
        for ph, side in (("extinction", "right"), ("reversal", "left")):
            xs = [t for t in sessions["B"] if t.get("kind") == "test"
                  and t["phase"] == ph and t["choice"] is not None]
            if xs:
                contrasts[f"B_{ph}_P_{side}"] = round(
                    sum(1 for t in xs if t["choice"] == side) / len(xs), 3)
    verdict = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                   contrasts=contrasts)
    (Path(out_dir) / "analysis.json").write_text(json.dumps(verdict,
                                                            indent=1))
    print(json.dumps(contrasts, indent=1))
    return verdict


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["prereg", "pilot", "matrix", "rl",
                                    "analyze", "_worker"])
    ap.add_argument("--cfg")
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--eta", type=float, default=0.30)
    ap.add_argument("--tau-relax", type=float, default=50.0)
    ap.add_argument("--trials", type=int, default=12)
    args = ap.parse_args()

    if args.cmd == "_worker":
        _worker_main(args.cfg)
    elif args.cmd == "prereg":
        write_preregistration(beta=args.beta, eta=args.eta,
                              tau_relax=args.tau_relax)
    elif args.cmd == "pilot":
        cfg = dict(condition="B_plastic",
                   phases=[dict(name="pilot", kind="blocks", blocks=1,
                                pair_per_block=max(1, args.trials // 3),
                                test_per_block=3, pair_side="right",
                                us_on=True)],
                   master_seed=20260929,
                   out_dir=str(RESULTS / "pilot"),
                   beta=args.beta, eta=args.eta,
                   tau_relax_trials=args.tau_relax, elig_power=2.0,
                   save_chunks=False, label="pilot")
        spawn_session(cfg)
    elif args.cmd == "matrix":
        run_matrix()
    elif args.cmd == "rl":
        run_rl_baseline()
    elif args.cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main()
