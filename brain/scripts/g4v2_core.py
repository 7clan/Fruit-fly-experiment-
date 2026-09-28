"""GATE 4 v2 — core trial runners (shared by calibration + sessions).

Implements the pre-registered v2 trial protocols on top of the EXISTING
Gate-3/D11 runtime (composition, not replacement):

  pref trial (5.1)     open-loop preference test: both CS ensembles at
                       fixed rates; route-B aggregate MBON-valence
                       readout + route-C reachable-DN readout + native
                       DN/walk measurements; no motor loop.
  pairing trial (5.2)  classical conditioning: paired-side CS + US
                       (compartment-matched PAM01-11, ENGINEERED
                       OPTOGENETIC-STYLE US ENTRY) in the last 1.0 s;
                       eligibility trace + dopamine-gated plasticity
                       update at trial end.
  filler (5.3), context (5.4), distractor (5.5) pref variants.
  route-A trial (5.6)  the UNCHANGED v1 closed-loop choice trial
                       (Gate-3 decoder, beta=1.5, visual regime 134 Hz).

Everything the brain sees enters through the sensory interface; every
readout is from measured spike counts.  Reward NEVER touches action.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402
import d11_learning as d11  # noqa: E402
import d9_decoder as d9  # noqa: E402

BRAIN = bl.ROOT
G4 = BRAIN / "data" / "gate4_v2"

# ---- frozen protocol timing (GATE4_V2_PREREG.md section 5) --------------
CHUNK_MS = 50.0
DT = CHUNK_MS / 1000.0
PREF_PRE_S = 0.5        # silence
PREF_CS_S = 1.5         # both-CS window
PREF_TAIL_S = 0.3
PREF_WINDOW_LO_S = 0.8  # measurement window [0.8, 2.0] s
PREF_WINDOW_HI_S = 2.0
PAIR_PRE_S = 0.5
PAIR_CS_S = 1.5
PAIR_US_S = 1.0         # US occupies the LAST 1.0 s of the CS window
PAIR_TAIL_S = 0.5
FILLER_S = 1.5
US_RATE_HZ = 150.0
PAM_US_N_DEFAULT = 252
PPL1_N = 16

# plasticity constants (unchanged from the v1-demonstrated rule)
ETA = 0.30
SCALE_MIN = 0.05
TAU_RELAX_TRIALS = 50.0
TAU_ELIG_S = 2.0
ELIG_POWER = 2.0
PAM_GATE_HZ = 10.0
PPL1_GATE_HZ = 10.0


def load_v2_specs(n_cs: int):
    """interface/readout specs with CS ensembles sliced to n_cs/side."""
    iface = json.loads((G4 / "interface_v2.json").read_text())
    readout = json.loads((G4 / "readout_v2.json").read_text())
    channels = {}
    for ch, d in iface["channels"].items():
        ids = list(d["ids"])
        if ch in ("kc_cs_left", "kc_cs_right"):
            ids = ids[:n_cs]        # nested prefix (pre-registered)
        channels[ch] = ids
    return channels, readout["populations"], iface


class V2Runtime:
    """One LearningBrain build + v2 plasticity wiring (corrected classes)."""

    def __init__(self, master_seed, n_cs=200, plasticity=True):
        channels, pops, iface = load_v2_specs(n_cs)
        self.master_seed = int(master_seed)
        self.n_cs = int(n_cs)
        self.iface_meta = iface
        self.channel_ids = channels

        self.lb = d11.LearningBrain(seed=master_seed,
                                    channel_ids=channels,
                                    readout_pops=pops)
        lb = self.lb

        # corrected valence classes (transmitter-anchored; audit output)
        val = json.loads((G4 / "mbon_valence_corrected.json").read_text())
        classes = {int(k): v for k, v in val["corrected_classes"].items()}
        val_by_bidx = {}
        for f, cls in classes.items():
            if int(f) in lb.brain.flyid2i:
                val_by_bidx[lb.brain.flyid2i[int(f)]] = cls

        self.plasticity_on = bool(plasticity)
        self.pl = d11.MBPlasticity(
            lb.kc_rows, lb.kc_local[lb._scan["pre"]],
            lb._scan["post"], val_by_bidx,
            n_kc=len(lb.kc_bidx), eta=ETA, scale_min=SCALE_MIN,
            tau_relax_trials=TAU_RELAX_TRIALS, tau_elig_s=TAU_ELIG_S,
            elig_mode="spikes", pam_gate_hz=PAM_GATE_HZ,
            ppl1_gate_hz=PPL1_GATE_HZ, elig_power=ELIG_POWER)

        # CS-ensemble bookkeeping (local KC indices)
        self.cs_local = {"left": np.array(
            [lb.kc_local[lb.brain.flyid2i[int(f)]]
             for f in channels["kc_cs_left"]], dtype=np.int64),
            "right": np.array(
            [lb.kc_local[lb.brain.flyid2i[int(f)]]
             for f in channels["kc_cs_right"]], dtype=np.int64)}
        self.pam_us_n = len(channels["pam_us"])

        # population sizes for rate computation
        self.pop_sizes = dict(lb.brain.pop_sizes)

    # ------------------------------------------------------------------
    def _rate(self, counts, pop, window_s):
        n = self.pop_sizes.get(pop, 0)
        return (counts[pop] / (n * window_s)) if (n and window_s > 0) \
            else 0.0

    # ------------------------------------------------------------------
    def pref_trial(self, seed, rate_left, rate_right, ctx_scale=1.0,
                   loom_precursor=False, phase="pref", trial_idx=0,
                   plastic_update=False, us_on=False, pair_side=None,
                   lag_gate=None, perm_seed=None):
        """Open-loop preference test (prereg 5.1; variants 5.4/5.5).

        Returns the measurement record.  If plastic_update is True the
        eligibility+gate machinery runs and the dopamine-gated update is
        applied (used only by pairing trials, which call this with
        us_on=True and a single side's rate set to 0 for the other).
        """
        lb = self.lb
        t0 = time.perf_counter()
        lb.reset_episode(seed)
        lb.apply_plastic(self.pl.scale)

        elig = d11.EligibilityTrace(len(lb.kc_bidx), tau_s=TAU_ELIG_S,
                                    mode="spikes")
        n_pre = int(round(PREF_PRE_S / DT))
        n_loom = int(round(0.5 / DT)) if loom_precursor else 0
        n_cs = int(round(PREF_CS_S / DT))
        n_tail = int(round(PREF_TAIL_S / DT))

        loom_rate = 150.0     # frozen Gate-3 looming channel rate
        win_counts = {}
        win_chunks = 0
        pam_us_spikes = 0.0
        ppl1_spikes = 0.0
        us_bio_s = 0.0
        kc_cs_spikes = {"left": 0, "right": 0}

        total = n_pre + n_loom + n_cs + n_tail
        for k in range(total):
            rates = {}
            if loom_precursor and n_pre <= k < n_pre + n_loom:
                rates["looming"] = loom_rate
            if n_pre + n_loom <= k < n_pre + n_loom + n_cs:
                rates["kc_cs_left"] = rate_left * ctx_scale
                rates["kc_cs_right"] = rate_right * ctx_scale
                if us_on and pair_side and k >= n_pre + n_loom + n_cs \
                        - int(round(PAIR_US_S / DT)):
                    rates["pam_us"] = US_RATE_HZ
            r = lb.step_learn(rates, chunk_ms=CHUNK_MS)
            counts = r.get("pop_counts", {})
            elig.on_chunk(DT, kc_spikes=r["_kc_counts"])

            # window: last 1.2 s of the CS window (24 chunks of 50 ms)
            if (n_pre + n_loom + n_cs) - 24 <= k < n_pre + n_loom + n_cs:
                win_chunks += 1
                for p, c in counts.items():
                    win_counts[p] = win_counts.get(p, 0) + c
                kc_cs_spikes["left"] += int(
                    r["_kc_counts"][self.cs_local["left"]].sum())
                kc_cs_spikes["right"] += int(
                    r["_kc_counts"][self.cs_local["right"]].sum())
            if us_on and k >= total - n_tail - int(round(PAIR_US_S / DT)) \
                    and k < total - n_tail:
                pam_us_spikes += counts.get("PAM_US_SET", 0)
                ppl1_spikes += (counts.get("PPL1_left", 0)
                                + counts.get("PPL1_right", 0))
                us_bio_s += DT

        win_s = win_chunks * DT
        # route B: aggregate MBON-valence readout (ENGINEERED, tier-3c)
        appr_l = self._rate(win_counts, "MBON_approach_left", win_s)
        appr_r = self._rate(win_counts, "MBON_approach_right", win_s)
        avoid_l = self._rate(win_counts, "MBON_avoidance_left", win_s)
        avoid_r = self._rate(win_counts, "MBON_avoidance_right", win_s)
        V = (appr_l - avoid_l) - (appr_r - avoid_r)
        choice_b = "left" if V > 0 else ("right" if V < 0 else "undecided")
        # route C: MBON->reachable-DN aggregate (ENGINEERED, tier-3c)
        ca_l = self._rate(win_counts, "DN_apprRoute_left", win_s)
        ca_r = self._rate(win_counts, "DN_apprRoute_right", win_s)
        cv_l = self._rate(win_counts, "DN_avoidRoute_left", win_s)
        cv_r = self._rate(win_counts, "DN_avoidRoute_right", win_s)
        C = (ca_l - cv_l) - (ca_r - cv_r)
        choice_c = "left" if C > 0 else ("right" if C < 0 else "undecided")

        pam_us_hz = pam_us_spikes / (self.pam_us_n * us_bio_s) \
            if us_bio_s > 0 else 0.0
        ppl1_hz = ppl1_spikes / (PPL1_N * us_bio_s) if us_bio_s > 0 else 0.0

        # plasticity (pairing trials; prereg section 6)
        w_stats = self.pl.stats()          # state that produced behavior
        up_log = {}
        if plastic_update and self.plasticity_on:
            gate_pam, gate_ppl1 = (pam_us_hz, ppl1_hz)
            if lag_gate is not None:            # condition C
                gate_pam, gate_ppl1 = lag_gate
            self.pl.relax(1)
            e = elig.value()
            if perm_seed is not None:           # condition D
                perm = np.random.default_rng(
                    perm_seed).permutation(len(e))
                e = e[perm]
            self.pl.update(e, gate_pam, gate_ppl1, up_log)
        elif plastic_update:
            up_log = dict(update_skipped="plasticity_off")

        rec = dict(
            phase=phase, trial=trial_idx, seed=seed, kind="pref",
            pair_side=pair_side, us_on=bool(us_on),
            ctx_scale=ctx_scale, loom_precursor=bool(loom_precursor),
            V=round(float(V), 4), choice_b=choice_b,
            C=round(float(C), 4), choice_c=choice_c,
            mbon=dict(appr_l=round(appr_l, 3), appr_r=round(appr_r, 3),
                      avoid_l=round(avoid_l, 3), avoid_r=round(avoid_r, 3)),
            dn=dict(
                p9_l=self._rate(win_counts, "P9_left", win_s),
                p9_r=self._rate(win_counts, "P9_right", win_s),
                p9_diff=round(self._rate(win_counts, "P9_left", win_s)
                              - self._rate(win_counts, "P9_right", win_s), 3),
                mdn=self._rate(win_counts, "MDN_bilateral", win_s),
                bpn=self._rate(win_counts, "BPN", win_s),
                rrn=self._rate(win_counts, "RRN", win_s),
                dn_all_l=self._rate(win_counts, "DN_ALL_left", win_s),
                dn_all_r=self._rate(win_counts, "DN_ALL_right", win_s),
                appr_route_l=round(ca_l, 3), appr_route_r=round(ca_r, 3),
                avoid_route_l=round(cv_l, 3), avoid_route_r=round(cv_r, 3)),
            kc_cs=dict(left_hz=round(kc_cs_spikes["left"]
                                     / (self.n_cs * win_s), 3),
                       right_hz=round(kc_cs_spikes["right"]
                                      / (self.n_cs * win_s), 3)),
            pam_us_hz=round(float(pam_us_hz), 3),
            ppl1_us_hz=round(float(ppl1_hz), 3),
            elig_left=round(float(np.mean(
                elig.value()[self.lb.kc_left_local])), 4),
            elig_right=round(float(np.mean(
                elig.value()[self.lb.kc_right_local])), 4),
            weight_stats=w_stats,
            plasticity=up_log,
            bio_s=round(total * DT, 2),
            wall_s=round(time.perf_counter() - t0, 1),
        )
        return rec

    # ------------------------------------------------------------------
    def pairing_trial(self, seed, side, rate_side, trial_idx=0,
                      us_on=True, plastic_update=True, lag_gate=None,
                      perm_seed=None):
        """Classical-conditioning pairing trial (prereg 5.2): the paired
        side's CS alone + US in the last 1.0 s; eligibility over the whole
        trial; relax-then-update at the end."""
        rate_l = rate_side if side == "left" else 0.0
        rate_r = rate_side if side == "right" else 0.0
        rec = self.pref_trial(
            seed, rate_l, rate_r, phase="pair", trial_idx=trial_idx,
            plastic_update=plastic_update, us_on=us_on, pair_side=side,
            lag_gate=lag_gate, perm_seed=perm_seed)
        rec["kind"] = "pair"
        rec["elig_ensemble_paired"] = rec.get(
            "elig_left" if side == "left" else "elig_right")
        return rec

    # ------------------------------------------------------------------
    def filler_trial(self, seed, trial_idx=0):
        lb = self.lb
        t0 = time.perf_counter()
        lb.reset_episode(seed)
        lb.apply_plastic(self.pl.scale)
        n = int(round(FILLER_S / DT))
        for _ in range(n):
            lb.step_learn({}, chunk_ms=CHUNK_MS)
        # relaxation continues during fillers (forgetting)
        self.pl.relax(1)
        return dict(phase="retention_filler", trial=trial_idx, seed=seed,
                    kind="filler", bio_s=round(n * DT, 2),
                    wall_s=round(time.perf_counter() - t0, 1),
                    weight_stats=self.pl.stats())

    # ------------------------------------------------------------------
    def routea_trial(self, seed, rate_left, rate_right, trial_idx=0):
        """Closed-loop native-expression trial (prereg 5.6): the UNCHANGED
        v1 choice test - Gate-3 decoder (P9 turn source, frozen thresholds,
        beta=1.5 MBON-valence bias with the CORRECTED classes), visual
        regime target_left=134 Hz, fixation-gated CS ensembles."""
        import d11_sessions as d11s          # v1 frozen module
        lb = self.lb
        t0 = time.perf_counter()
        lb.reset_episode(seed)
        lb.apply_plastic(self.pl.scale)

        arena = d11.ChoiceArena2D(seed=seed, mode="choice")
        encoder = d11.ChoiceVisualEncoder(r_target_left=134.0,
                                          r_target_right=150.0)
        # fixation-gated CS at the v2 frozen rates, passed as the PRE-COMMIT
        # (|b|=50) rates - the direct analogy of the open-loop pref rates
        cs_enc = d11s.CSEncoder(precommit_hz_left=rate_left,
                                precommit_hz_right=rate_right)
        dec_params = dict(window_ms=300.0, theta_jump_hz=25.0,
                          theta_bwd_hz=4.0, theta_fwd_hz=5.0,
                          theta_diff_hz=8.0, delta_hyst_hz=4.0,
                          theta_stop_hz=4.0, theta_min_walk_hz=1.5,
                          min_action_chunks=2, turn_source="P9")
        decoder = d11s.ValenceBiasDecoder(lb.brain.pop_sizes, dec_params,
                                          beta=1.5)
        n_chunks = 0
        outcome = "timeout"
        while True:
            vis = arena.visual_state()
            rates = encoder.rates(vis)
            rates.update(cs_enc.rates(vis))
            r = lb.step_learn(rates, chunk_ms=CHUNK_MS)
            counts = dict(r.get("pop_counts", {}))
            dec = decoder.decode(counts, CHUNK_MS)
            arena.step(dec["action"], DT)
            n_chunks += 1
            if arena.choice is not None:
                outcome = f"commit_{arena.choice}"
                break
            if arena.t >= d11s.CHOICE_MAX_S:
                break
        return dict(phase="routea", trial=trial_idx, seed=seed,
                    kind="routea", choice=arena.choice, outcome=outcome,
                    bio_s=round(arena.t, 2),
                    valence_bias_end=(round(float(np.mean(
                        decoder.bias_history[-6:])), 2)
                        if decoder.bias_history else None),
                    wall_s=round(time.perf_counter() - t0, 1))
