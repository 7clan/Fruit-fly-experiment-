"""D13 - multi-step tasks, delayed reinforcement, and the two memory
conditions.

TWO pre-registered experiments (all tier labels as D11):

E1  DELAYED-REINFORCEMENT SURVIVAL (side-preference task):
    the D11 pairing+test design with the US DELAYED 1.5 s after the
    pairing-trial commit.  Can the learned preference still develop when
    reinforcement is delayed?  (The MODELED eligibility trace tau_e=2 s
    must bridge the delay; acquisition is expected to survive, slower.)
    Followed by the D13 RETENTION BATTERY on the same session:
      immediate tests, delayed recall (after 8 no-stimulus filler
      trials), context change (targets at +/-40 instead of +/-50),
      distractor tests (a 0.5 s centre flash before the choice),
      extinction tests.

E2  MULTI-STEP CUE-MEMORY TASK, memory conditions A vs B:
    per trial:  CUE (single target at the cue side, random L/R; the fly
                orients - detection)
             -> CHOICE (cue removed; two goals; the rewarded goal is the
                one on the CUE's side - cue-following is reward-
                maximizing; within-trial working memory required)
             -> APPROACH (chosen goal only; reach)
             -> INTERACT (0.5 s dwell)
             -> DELAY (1.5 s) -> US (PAM 1 s, iff the reached goal was
                the cue-predicted side)
    Condition A (brain-only): the only memory across the cue->choice gap
    is the brain's own state (eligibility trace; the canonical model has
    NO persistent activity - silence is its resting state, Gate-1
    evidence).  Honest expectation: cue-following near chance.
    Condition B (brain + EXTERNAL episodic memory): the harness logs the
    cue side and injects an explicit labelled bias during the CHOICE
    stage.  THE EXTERNAL MEMORY IS ENGINEERING, NOT THE BRAIN - never
    credited to the fly (mandate).

Usage:
  python d13_multistep.py e1          # delayed-US session + battery
  python d13_multistep.py e2          # conditions A and B
  python d13_multistep.py analyze
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
import d11_sessions as ds   # noqa: E402
import d9_decoder as d9     # noqa: E402

DATA = bl.ROOT / "data" / "d11_d13"
RESULTS = bl.ROOT / "results" / "d13_multistep"
PY = sys.executable
CHUNK_MS = 50.0
DT = CHUNK_MS / 1000.0
US_RATE_HZ = 150.0
US_DURATION_S = 1.0
POST_S = 0.3
PAM_N = 307
PPL1_N = 16

# pre-registered D13 parameters
US_DELAY_S = 1.5              # E1 delayed reinforcement
CUE_BEARING_DEG = 35.0
CUE_MAX_S = 1.5
CHOICE_MAX_S = 2.5
APPROACH_MAX_S = 2.5
INTERACT_S = 0.5
REACH_RADIUS = 0.30
DWELL_RADIUS = 0.35
EXT_MEMORY_BIAS_HZ = 12.0     # condition B external-memory bias (ENGINEERED)


# --------------------------------------------------------------------------
# multi-step arena (mutates targets across stages; kinematics = Gate-3)
# --------------------------------------------------------------------------
class MultiStepArena:
    V_FWD, V_TURN, V_BWD = 0.25, 0.15, 0.15
    OMEGA_TURN = math.radians(150.0)
    HALF = 1.0
    COMMIT_DEG = 40.0

    def __init__(self, seed=0, goal_bearing_deg=50.0):
        rng = np.random.default_rng(seed)
        self.x, self.y = 0.0, 0.0
        self.heading = 0.5 * math.pi
        self.heading0 = self.heading
        self.t = 0.0
        self.jump_count = 0
        self.escaped = False
        self.goal_bearing = goal_bearing_deg + 4.0 * rng.uniform(-1, 1)
        self.cue_side = "left" if rng.random() < 0.5 else "right"
        self.targets = {"left": None, "right": None}   # (x, y) or None
        self.stage = "idle"
        self.choice = None
        self.choice_t = None

    # ---- target management ---------------------------------------------
    def _place(self, side, bearing_deg, dist):
        ang = self.heading0 + math.radians(
            bearing_deg if side == "left" else -bearing_deg)
        self.targets[side] = (self.x + dist * math.cos(ang),
                              self.y + dist * math.sin(ang))

    def set_stage(self, stage, cue_side=None, goal_bearing_deg=None):
        self.stage = stage
        self.targets = {"left": None, "right": None}
        if stage == "cue" and cue_side:
            self._place(cue_side, CUE_BEARING_DEG, 0.8)
        elif stage == "choice":
            gb = goal_bearing_deg or self.goal_bearing
            self._place("left", gb, 0.8)
            self._place("right", gb, 0.8)
        elif stage in ("approach", "interact", "delay", "us"):
            # only the CHOSEN goal remains
            if self.choice:
                self.targets[self.choice] = self._chosen_pos

    # ---- visual state ----------------------------------------------------
    def _bearing(self, pos):
        if pos is None:
            return None
        ang = math.atan2(pos[1] - self.y, pos[0] - self.x)
        b = math.degrees(ang - self.heading)
        while b > 180.0:
            b -= 360.0
        while b <= -180.0:
            b += 360.0
        return b

    def visual_state(self):
        tl, tr = self.targets["left"], self.targets["right"]
        return dict(
            t=self.t, stage=self.stage,
            bearing_left_deg=self._bearing(tl),
            bearing_right_deg=self._bearing(tr),
            dist_left=(math.hypot(tl[0] - self.x, tl[1] - self.y)
                       if tl else None),
            dist_right=(math.hypot(tr[0] - self.x, tr[1] - self.y)
                        if tr else None),
            loom_angular_deg=0.0)

    # ---- dynamics ---------------------------------------------------------
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
        self.t += dt
        self.x = min(max(self.x, -self.HALF), self.HALF)
        self.y = min(max(self.y, -self.HALF), self.HALF)
        # commit tracking (heading deviation from trial start)
        if self.choice is None:
            dev = math.degrees(self.heading - self.heading0)
            if dev >= self.COMMIT_DEG:
                self.choice = "left"
                self.choice_t = self.t
            elif dev <= -self.COMMIT_DEG:
                self.choice = "right"
                self.choice_t = self.t

    def _move(self, d):
        self.x += d * math.cos(self.heading)
        self.y += d * math.sin(self.heading)

    def dist_to(self, side):
        pos = self.targets.get(side)
        if pos is None:
            return None
        return math.hypot(pos[0] - self.x, pos[1] - self.y)

    def pin_chosen_goal(self):
        """Freeze the chosen goal's position for the later stages."""
        self._chosen_pos = self.targets.get(self.choice)


# --------------------------------------------------------------------------
# shared build (LearningBrain + plasticity + decoder, D11-frozen params)
# --------------------------------------------------------------------------
def build_learner(master_seed, external_memory=False):
    if not (DATA / "learning_interface.json").exists() or \
            json.loads((DATA / "learning_interface.json").read_text()
                       )["version"] < 2:
        ds.build_production_interface_spec()
    channel_ids = {ch: d["ids"] for ch, d in
                   json.loads((DATA / "learning_interface.json").read_text()
                              )["channels"].items()}
    readout_pops = d11.load_learning_readout_pops()
    lb = d11.LearningBrain(seed=master_seed, channel_ids=channel_ids,
                           readout_pops=readout_pops)
    val_by_bidx = {}
    for f, cls in lb._val_of_fly.items():
        if int(f) in lb.brain.flyid2i:
            val_by_bidx[lb.brain.flyid2i[int(f)]] = cls
    pl = d11.MBPlasticity(
        lb.kc_rows, lb.kc_local[lb._scan["pre"]], lb._scan["post"],
        val_by_bidx, n_kc=len(lb.kc_bidx),
        eta=0.30, scale_min=0.05, tau_relax_trials=50.0, tau_elig_s=2.0,
        elig_mode="spikes", pam_gate_hz=10.0, ppl1_gate_hz=10.0,
        elig_power=2.0)
    cal = json.loads((bl.ROOT / "results" / "d11_probes" /
                      "bilateral_calibration.json").read_text())
    encoder = d11.ChoiceVisualEncoder(
        r_target_left=cal.get("target_left_hz_calibrated", 134.0),
        r_target_right=150.0)
    cs_encoder = ds.CSEncoder(
        precommit_hz_left=cal.get("kc_cs_left_hz_calibrated", 102.3),
        precommit_hz_right=75.0)
    dec_params = dict(window_ms=300.0, theta_jump_hz=25.0, theta_bwd_hz=4.0,
                      theta_fwd_hz=5.0, theta_diff_hz=8.0, delta_hyst_hz=4.0,
                      theta_stop_hz=4.0, theta_min_walk_hz=1.5,
                      min_action_chunks=2, turn_source="P9")
    decoder = ds.ValenceBiasDecoder(lb.brain.pop_sizes, dec_params, beta=1.5)
    return lb, pl, encoder, cs_encoder, decoder, dec_params


def _fresh_decoder(lb, dec_params, beta=1.5):
    return ds.ValenceBiasDecoder(lb.brain.pop_sizes, dec_params, beta=beta)


def _chunk_rates(encoder, cs_encoder, vis):
    rates = encoder.rates(vis)
    rates.update(cs_encoder.rates(vis))
    return rates


def _update_plasticity(pl, elig, pam_hz, ppl1_hz, cond):
    gate_pam, gate_ppl1 = pam_hz, ppl1_hz
    pl.relax(1)
    up_log = {}
    if cond in ("B_plastic", "brain_A", "brain_B_external"):
        pl.update(elig.value(), gate_pam, gate_ppl1, up_log)
    else:
        up_log = dict(update_skipped=cond)
    return up_log


# --------------------------------------------------------------------------
# E1 - delayed reinforcement + retention battery
# --------------------------------------------------------------------------
def run_e1(master_seed=20261101, out_dir=RESULTS / "E1_delayed"):
    """Pairing trials with the US delayed 1.5 s after commit; test trials;
    then the retention battery (immediate / delayed recall / context
    change / distractor / extinction)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    lb, pl, encoder, cs_encoder, decoder, dec_params = build_learner(
        master_seed)
    print(f"[e1] build done {time.perf_counter()-t_start:.0f}s", flush=True)

    (out_dir / "trials.jsonl").unlink(missing_ok=True)
    us_chunks = int(round(US_DURATION_S * 1000 / CHUNK_MS))
    delay_chunks = int(round(US_DELAY_S * 1000 / CHUNK_MS))
    trial_idx = 0
    trials = []

    def run_pair_trial(seed, pair_side, us_on=True):
        """Pairing trial with DELAYED US: commit -> 1.5 s delay -> US."""
        nonlocal trial_idx
        lb.reset_episode(seed)
        lb.apply_plastic(pl.scale)
        arena = d11.ChoiceArena2D(seed=seed, mode=f"pair_{pair_side}")
        elig = d11.EligibilityTrace(len(lb.kc_bidx), tau_s=pl.tau_elig_s,
                                    mode="spikes")
        dec = _fresh_decoder(lb, dec_params)
        state = "run"            # run -> delay -> us -> post
        countdown = 0
        pam_spikes = 0
        us_s = 0.0
        us_done = False
        n_ch = 0
        while True:
            vis = arena.visual_state()
            rates = _chunk_rates(encoder, cs_encoder, vis)
            if state == "us" and countdown > 0:
                rates["pam"] = US_RATE_HZ
            r = lb.step_learn(rates, chunk_ms=CHUNK_MS)
            elig.on_chunk(DT, kc_spikes=r["_kc_counts"])
            counts = dict(r.get("pop_counts", {}))
            dec_o = dec.decode(counts, CHUNK_MS)
            arena.step(dec_o["action"], DT)
            n_ch += 1
            if state == "us":
                pam_spikes += (counts.get("PAM_left", 0)
                               + counts.get("PAM_right", 0))
                us_s += DT
            # ---- explicit state machine (one decrement per chunk) ----
            if state == "run":
                if arena.choice == pair_side:
                    state = "delay"
                    countdown = delay_chunks
                elif arena.t > CHOICE_MAX_S + 1.0:
                    break
            elif state == "delay":
                countdown -= 1
                if countdown <= 0:
                    if us_on:
                        state = "us"
                        countdown = us_chunks
                    else:
                        break
            elif state == "us":
                countdown -= 1
                if countdown <= 0:
                    us_done = True
                    state = "post"
                    countdown = int(POST_S * 1000 / CHUNK_MS)
            elif state == "post":
                countdown -= 1
                if countdown <= 0:
                    break
        pam_hz = pam_spikes / (PAM_N * us_s) if us_s > 0 else 0.0
        up_log = _update_plasticity(pl, elig, pam_hz, 0.0, "B_plastic")
        rec = dict(trial=trial_idx, kind=f"pair_{pair_side}",
                   pair_side=pair_side, choice=arena.choice,
                   us_delivered=us_done, pam_us_hz=round(pam_hz, 2),
                   delay_s=US_DELAY_S, bio_s=round(arena.t, 2),
                   plasticity=up_log,
                   weight_stats=pl.stats())
        _log_trial(out_dir, trials, rec)
        trial_idx += 1
        return rec

    def run_test_trial(seed, label, goal_bearing=50.0, distractor=False):
        """Free-choice test (no US). Optional context change (bearing) or
        a 0.5 s centre distractor flash before the choice window."""
        nonlocal trial_idx
        lb.reset_episode(seed)
        lb.apply_plastic(pl.scale)
        arena = d11.ChoiceArena2D(seed=seed, bearing_deg=goal_bearing)
        elig = d11.EligibilityTrace(len(lb.kc_bidx), tau_s=pl.tau_elig_s,
                                    mode="spikes")
        dec = _fresh_decoder(lb, dec_params)
        distract_chunks = int(round(0.5 * 1000 / CHUNK_MS)) if distractor \
            else 0
        n_ch = 0
        while True:
            vis = arena.visual_state()
            rates = _chunk_rates(encoder, cs_encoder, vis)
            if distract_chunks > 0:
                # centre distractor: BOTH visual channels max for 0.5 s
                rates["target_left"] = encoder.r_left
                rates["target_right"] = encoder.r_right
                distract_chunks -= 1
            r = lb.step_learn(rates, chunk_ms=CHUNK_MS)
            elig.on_chunk(DT, kc_spikes=r["_kc_counts"])
            dec_o = dec.decode(dict(r.get("pop_counts", {})), CHUNK_MS)
            arena.step(dec_o["action"], DT)
            n_ch += 1
            if arena.choice is not None and arena.t >= arena.choice_t \
                    + POST_S:
                break
            if arena.t > CHOICE_MAX_S + POST_S:
                break
        rec = dict(trial=trial_idx, kind="test", test_label=label,
                   choice=arena.choice, us_delivered=False, pam_us_hz=0.0,
                   bio_s=round(arena.t, 2), plasticity={},
                   weight_stats=pl.stats())
        _log_trial(out_dir, trials, rec)
        trial_idx += 1
        return rec

    # ---- schedule (pre-registered) --------------------------------------
    # acquisition: 6 blocks x (4 delayed-US pair RIGHT + 3 test)
    for b in range(6):
        for k in range(4):
            run_pair_trial(master_seed + trial_idx, "right")
        for k in range(3):
            run_test_trial(master_seed + trial_idx, "acquisition")
    # immediate recall already covered; delayed recall: 8 filler trials
    # (no stimuli, brain silent - only the relaxation clock advances)
    for k in range(8):
        _filler(out_dir, trials, lb, pl, master_seed + trial_idx)
    for k in range(6):
        run_test_trial(master_seed + trial_idx, "delayed_recall")
    # context change: goals at +/-40
    for k in range(6):
        run_test_trial(master_seed + trial_idx, "context_change",
                       goal_bearing=40.0)
    # distractor tests
    for k in range(6):
        run_test_trial(master_seed + trial_idx, "distractor",
                       distractor=True)
    # extinction tests
    for k in range(8):
        run_test_trial(master_seed + trial_idx, "extinction")

    summary = _summarize(trials, "E1_delayed_reinforcement_battery")
    summary.update(session_wall_s=round(time.perf_counter() - t_start, 1),
                   peak_rss_mb=round(bl.peak_rss_kb() / 1024, 1))
    (out_dir / "session.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out_dir / "weights.npz", scale=pl.scale)
    print(json.dumps({k: v for k, v in summary.items()
                      if k.startswith("block_")}, indent=1))
    return summary


def _filler(out_dir, trials, lb, pl, seed):
    """No-stimulus filler trial (time gap; brain silent; relax clock
    advances)."""
    lb.reset_episode(seed)
    lb.apply_plastic(pl.scale)
    for _ in range(int(1.0 / DT)):
        lb.step_learn({}, chunk_ms=CHUNK_MS)
    rec = dict(trial=len(trials), kind="filler", choice=None,
               us_delivered=False, pam_us_hz=0.0, bio_s=1.0,
               plasticity={}, weight_stats=pl.stats())
    pl.relax(1)
    _log_trial(out_dir, trials, rec)
    return rec


def _log_trial(out_dir, trials, rec):
    trials.append(rec)
    with (Path(out_dir) / "trials.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[trial {rec['trial']:3d}] {rec['kind']:<12s} "
          f"{str(rec.get('test_label') or ''):<16s} "
          f"choice={str(rec.get('choice')):<6s} "
          f"us={int(rec.get('us_delivered'))}", flush=True)


def _summarize(trials, condition):
    def p_side(label, side):
        xs = [t for t in trials if t.get("test_label") == label
              and t["choice"] is not None]
        if not xs:
            return None
        return round(sum(1 for t in xs if t["choice"] == side) / len(xs), 3)

    out = dict(condition=condition, n_trials=len(trials))
    for label in ("acquisition", "delayed_recall", "context_change",
                  "distractor", "extinction"):
        n = sum(1 for t in trials if t.get("test_label") == label)
        if n:
            out[f"block_{label}"] = dict(
                n=n, p_right=p_side(label, "right"),
                p_left=p_side(label, "left"))
    xs = [t for t in trials if t["kind"] == "pair_right"]
    if xs:
        out["pairing"] = dict(
            n=len(xs),
            approach_rate=round(sum(1 for t in xs
                                    if t["choice"] == "right") / len(xs), 3),
            n_us=sum(1 for t in xs if t["us_delivered"]))
    return out


# --------------------------------------------------------------------------
# E2 - multi-step cue-memory task (conditions A vs B)
# --------------------------------------------------------------------------
def run_e2(condition="brain_A", master_seed=20261201, n_blocks=8,
           out_dir=None):
    """Condition A (brain-only) vs B (brain + EXTERNAL episodic memory).

    The external memory in B is an ENGINEERED harness module: it logs the
    cue side and injects a labelled +12 Hz bias toward the remembered cue
    side during the CHOICE stage.  It is NOT part of the brain and is
    never credited as fly memory (mandate: never credit external memory
    to the fly brain).
    """
    out_dir = Path(out_dir) if out_dir else RESULTS / f"E2_{condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    lb, pl, encoder, cs_encoder, decoder, dec_params = build_learner(
        master_seed)
    print(f"[e2/{condition}] build done "
          f"{time.perf_counter()-t_start:.0f}s", flush=True)
    (out_dir / "trials.jsonl").unlink(missing_ok=True)
    us_chunks = int(round(US_DURATION_S * 1000 / CHUNK_MS))
    trials = []
    trial_idx = 0

    for block in range(n_blocks):
        for k in range(2):                      # 2 multistep trials/block
            seed = master_seed + trial_idx
            lb.reset_episode(seed)
            lb.apply_plastic(pl.scale)
            arena = MultiStepArena(seed=seed)
            elig = d11.EligibilityTrace(len(lb.kc_bidx),
                                        tau_s=pl.tau_elig_s, mode="spikes")
            dec = _fresh_decoder(lb, dec_params)
            cue_side = arena.cue_side
            rewarded_side = cue_side          # cue predicts the reward

            stage_t0 = 0.0
            step_rec = dict(trial=trial_idx, kind="multistep",
                            condition=condition, cue_side=cue_side,
                            rewarded_side=rewarded_side)
            # stage 1: CUE
            arena.set_stage("cue", cue_side=cue_side)
            cue_detected = False
            while arena.t < CUE_MAX_S:
                vis = arena.visual_state()
                rates = _chunk_rates(encoder, cs_encoder, vis)
                r = lb.step_learn(rates, chunk_ms=CHUNK_MS)
                elig.on_chunk(DT, kc_spikes=r["_kc_counts"])
                dec_o = dec.decode(dict(r.get("pop_counts", {})), CHUNK_MS)
                arena.step(dec_o["action"], DT)
                if arena.choice == cue_side:
                    cue_detected = True
                    break
            step_rec["cue_detected"] = cue_detected
            # stage 2: CHOICE (cue removed)
            arena.set_stage("choice")
            arena.choice = None                 # re-arm commit for choice
            arena.choice_t = None
            arena.heading0 = arena.heading      # re-reference the commit
            # detector to the post-cue heading (otherwise the cue-turn
            # instantly re-fires the commit)
            choice = None
            while arena.t < CUE_MAX_S + CHOICE_MAX_S:
                vis = arena.visual_state()
                rates = _chunk_rates(encoder, cs_encoder, vis)
                # condition B: EXTERNAL episodic memory bias (ENGINEERED)
                if condition == "brain_B_external":
                    dec.external_bias_hz = (
                        EXT_MEMORY_BIAS_HZ if cue_side == "left"
                        else -EXT_MEMORY_BIAS_HZ)
                r = lb.step_learn(rates, chunk_ms=CHUNK_MS)
                elig.on_chunk(DT, kc_spikes=r["_kc_counts"])
                dec_o = dec.decode(dict(r.get("pop_counts", {})), CHUNK_MS)
                arena.step(dec_o["action"], DT)
                if arena.choice is not None:
                    choice = arena.choice
                    break
            dec.external_bias_hz = 0.0
            step_rec["choice"] = choice
            # stage 3+4: APPROACH -> INTERACT -> DELAY -> US
            arena.choice = choice
            arena.pin_chosen_goal()
            reached = interacted = us_done = False
            pam_spikes = 0
            us_s = 0.0
            state = "approach"
            countdown = 0
            if choice:
                arena.set_stage("approach")
                t_cap = arena.t + APPROACH_MAX_S + INTERACT_S \
                    + US_DELAY_S + US_DURATION_S + POST_S + 1.0
                while arena.t < t_cap:
                    vis = arena.visual_state()
                    rates = _chunk_rates(encoder, cs_encoder, vis)
                    if state == "us" and countdown > 0:
                        rates["pam"] = US_RATE_HZ
                        countdown -= 1
                    r = lb.step_learn(rates, chunk_ms=CHUNK_MS)
                    elig.on_chunk(DT, kc_spikes=r["_kc_counts"])
                    counts = dict(r.get("pop_counts", {}))
                    dec_o = dec.decode(counts, CHUNK_MS)
                    arena.step(dec_o["action"], DT)
                    if state == "us":
                        pam_spikes += (counts.get("PAM_left", 0)
                                       + counts.get("PAM_right", 0))
                        us_s += DT
                    d = arena.dist_to(choice)
                    if state == "approach" and d is not None \
                            and d <= REACH_RADIUS:
                        reached = True
                        state = "interact"
                        countdown = int(INTERACT_S * 1000 / CHUNK_MS)
                    elif state == "interact":
                        if d is not None and d <= DWELL_RADIUS:
                            countdown -= 1
                            if countdown <= 0:
                                interacted = True
                                state = "delay"
                                countdown = int(US_DELAY_S * 1000
                                                / CHUNK_MS)
                        else:
                            countdown = int(INTERACT_S * 1000 / CHUNK_MS)
                    elif state == "delay":
                        countdown -= 1
                        if countdown <= 0:
                            if choice == rewarded_side:
                                state = "us"
                                countdown = us_chunks
                            else:
                                break
                    elif state == "us" and countdown <= 0:
                        us_done = True
                        break
            pam_hz = pam_spikes / (PAM_N * us_s) if us_s > 0 else 0.0
            up_log = _update_plasticity(pl, elig, pam_hz, 0.0, condition)
            step_rec.update(reached=reached, interacted=interacted,
                            us_delivered=us_done,
                            pam_us_hz=round(pam_hz, 2),
                            bio_s=round(arena.t, 2),
                            cue_followed=(choice == cue_side
                                          if choice else None),
                            plasticity=up_log,
                            weight_stats=pl.stats())
            _log_trial(out_dir, trials, step_rec)
            trial_idx += 1

    xs = [t for t in trials if t["kind"] == "multistep"]
    decided = [t for t in xs if t["choice"] is not None]
    summary = dict(
        condition=condition, n_trials=len(xs),
        cue_detection_rate=round(sum(1 for t in xs if t["cue_detected"])
                                 / len(xs), 3),
        p_choice_decided=round(len(decided) / len(xs), 3),
        cue_following_rate=(round(sum(1 for t in decided
                                      if t["cue_followed"]) / len(decided),
                                  3) if decided else None),
        reward_rate=round(sum(1 for t in xs if t["us_delivered"]) / len(xs),
                          3),
        reach_rate=round(sum(1 for t in xs if t["reached"]) / len(xs), 3),
        note=("condition B's cue-following uses an EXTERNAL episodic "
              "memory module (ENGINEERED harness bias); it is NOT fly "
              "brain memory" if condition == "brain_B_external" else
              "brain-only: within-trial cue memory must be carried by "
              "the brain's own state (eligibility trace; no persistent "
              "activity exists in the canonical model)"),
        session_wall_s=round(time.perf_counter() - t_start, 1),
        peak_rss_mb=round(bl.peak_rss_kb() / 1024, 1))
    (out_dir / "session.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out_dir / "weights.npz", scale=pl.scale)
    print(json.dumps(summary, indent=1))
    return summary


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def analyze(out_dir=RESULTS):
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

    out = {}
    # E1 battery
    e1p = Path(out_dir) / "E1_delayed" / "trials.jsonl"
    if e1p.exists():
        tr = [json.loads(l) for l in e1p.read_text().splitlines()]
        tests = [t for t in tr if t.get("test_label")]
        fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
        labels = ["acquisition", "delayed_recall", "context_change",
                  "distractor", "extinction"]
        vals = []
        for lb_ in labels:
            xs = [t for t in tests if t["test_label"] == lb_
                  and t["choice"] is not None]
            vals.append(sum(1 for t in xs if t["choice"] == "right")
                        / len(xs) if xs else float("nan"))
        ax.bar(range(len(labels)), vals, color="tab:blue", alpha=0.8)
        ax.axhline(0.5, color="k", lw=0.5, ls="--")
        ax.set_xticks(range(len(labels)),
                      [l.replace("_", "\n") for l in labels])
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(right | decided)")
        ax.set_title("E1: delayed-reinforcement (1.5 s) learning + "
                     "retention battery (test trials)")
        fig.savefig(Path(out_dir) / "e1_battery.png", dpi=130)
        out["E1"] = {l: (round(v, 3) if v == v else None)
                     for l, v in zip(labels, vals)}
    # E2 conditions
    for cond in ("brain_A", "brain_B_external"):
        p = Path(out_dir) / f"E2_{cond}" / "session.json"
        if p.exists():
            out[f"E2_{cond}"] = json.loads(p.read_text())
    (Path(out_dir) / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["e1", "e2", "analyze"])
    ap.add_argument("--condition", default="brain_A",
                    choices=["brain_A", "brain_B_external"])
    args = ap.parse_args()
    if args.cmd == "e1":
        run_e1()
    elif args.cmd == "e2":
        run_e2(condition=args.condition)
    else:
        analyze()


if __name__ == "__main__":
    main()
