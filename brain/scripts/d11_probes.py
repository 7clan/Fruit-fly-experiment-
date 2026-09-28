"""D11 dynamic probes P1-P5 (ONE subprocess, ONE network build - Track-D
reuse discipline).

Questions these probes answer BEFORE the learning matrix is designed
(answers are recorded under brain/results/d11_probes/ and feed the
pre-registration; no post-hoc tuning after the matrix):

  P1  Does the BIOLOGICAL sugar entry (GRN_SUGAR) recruit the reward DANs
      (PAM) and the rest of the MB circuit in the canonical dynamics?
  P1b Direct-PAM stimulation reference (ENGINEERED optogenetic-style entry)
      - the escalation rung if P1 is weak.
  P2  Does left/right visual drive (LC9) engage the KCs side-specifically
      (spikes or subthreshold depolarization)?  -> eligibility mode choice.
  P2b Direct-KC stimulation reference (does KC spiking drive MBONs at all?
      the KC->MBON gain measurement).
  P3  KC->MBON potency: same visual schedule with ALL KC->MBON weights
      scaled to 0 vs canonical - does silencing the plasticity site change
      MBON / DN / network activity?  (If nothing changes, depression cannot
      be behaviourally expressed in-network and the expression tier must be
      the pre-registered engineered readout.)
  P4  MBON->DN expression route: direct stimulation of approach-class vs
      avoidance-class MBONs (ENGINEERED entry) - do walk/turn/MDN descending
      populations respond differentially?
  P5  Bit-identity/reproducibility of the learning-chunk protocol
      (reset_episode + reseed + apply_plastic(no-op)) - the session
      determinism contract.

Usage:  python d11_probes.py run [--seed 11] [--chunk-ms 50]
"""

from __future__ import annotations

import argparse
import hashlib
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

DATA = bl.ROOT / "data" / "d11_d13"
RESULTS = bl.ROOT / "results" / "d11_probes"
PROBE_SEED = 11
CHUNK_MS = 50.0

# populations whose steady rates we summarize per phase (learning readout)
SUMMARY_POPS = [
    "PAM_left", "PAM_right", "PPL1_left", "PPL1_right",
    "KC_left", "KC_right",
    "MBON_approach_left", "MBON_approach_right",
    "MBON_avoidance_left", "MBON_avoidance_right",
    "MBON_unclassified_left", "MBON_unclassified_right",
    "APL_bilateral", "DPM_bilateral",
    "P9_left", "P9_right", "BPN_bilateral", "RRN_bilateral",
    "MDN_bilateral", "DN_ALL_left", "DN_ALL_right", "GF_left", "GF_right",
]


# --------------------------------------------------------------------------
# probe interface: learning channels + engineered stimulation entries
# --------------------------------------------------------------------------
def build_probe_interface_spec(out=DATA / "probe_interface.json"):
    base = json.loads((DATA / "learning_interface.json").read_text())
    channels = {ch: dict(d) for ch, d in base["channels"].items()}
    val = d11.mbon_valence_classes()
    appr = sorted(int(i) for i in val["approach"])
    avoid = sorted(int(i) for i in val["avoidance"])
    channels["mbon_appr"] = dict(
        population="MBON approach class (Aso 2014)", n=len(appr), ids=appr,
        biological_role="PROBE entry (engineered, optogenetic-style)")
    channels["mbon_avoid"] = dict(
        population="MBON avoidance class (Aso 2014)", n=len(avoid), ids=avoid,
        biological_role="PROBE entry (engineered, optogenetic-style)")
    pam = d11.load_ids("PAM")
    channels["pam"] = dict(
        population="PAM", n=len(pam), ids=pam,
        biological_role="PROBE entry (engineered, optogenetic-style; "
                        "escalation rung for US delivery)")
    rng = np.random.default_rng(20260928)
    kc = d11.load_ids("KC")
    kc_sample = sorted(int(x) for x in rng.choice(kc, size=200, replace=False))
    channels["kc_stim"] = dict(
        population="KC random sample (200, seeded)", n=len(kc_sample),
        ids=kc_sample,
        biological_role="PROBE entry (engineered, optogenetic-style; "
                        "KC->MBON gain measurement)")
    spec = dict(version=1, note="PROBE interface - adds engineered "
                                "stimulation entries to the learning "
                                "interface; production interface has none "
                                "of these",
                channels=channels)
    Path(out).write_text(json.dumps(spec, indent=1))
    print(f"[probe-interface] channels: {sorted(channels)}")
    return spec


# --------------------------------------------------------------------------
# one schedule = one probe run (reset + optional weight scale)
# --------------------------------------------------------------------------
def run_schedule(lb, schedule, seed, chunk_ms=CHUNK_MS, scale=None,
                 label=""):
    """Run one probe schedule on the LearningBrain.

    scale=None -> canonical weights; array -> applied to KC->MBON rows.
    Returns a record with per-phase steady rates, KC side stats, and a
    full-network spike sha256 (for P3/P5 comparisons).
    """
    lb.reset_episode(seed)
    if scale is not None:
        lb.apply_plastic(scale)
    dt = chunk_ms / 1000.0
    rec = dict(label=label, seed=seed, chunk_ms=chunk_ms,
               scale=("canonical" if scale is None else
                      ("zeros" if np.allclose(scale, 0) else "custom")),
               phases=[])
    kc_l_spikes = kc_r_spikes = 0
    kc_l_dep = []
    kc_r_dep = []
    for dur_ms, rates in schedule:
        n = int(round(dur_ms / chunk_ms))
        chunks = []
        for _ in range(n):
            r = lb.step_learn(rates, chunk_ms=chunk_ms)
            chunks.append(dict(
                t_bio_s=r["t_bio_s"], n_spikes_new=r["n_spikes_new"],
                pop_counts=dict(r.get("pop_counts", {})),
                kc_l=r["kc_left_spikes"], kc_r=r["kc_right_spikes"],
                kc_l_dep=r["kc_left_dep_mv"], kc_r_dep=r["kc_right_dep_mv"]))
        tail = chunks[max(0, len(chunks) // 2):]
        dt_s = len(tail) * dt
        steady = {}
        for p in SUMMARY_POPS:
            n_sz = lb.brain.pop_sizes.get(p, 0)
            steady[p] = (round(sum(c["pop_counts"].get(p, 0) for c in tail)
                               / (n_sz * dt_s), 3) if n_sz and dt_s else 0.0)
        steady["KC_left_spikes_s"] = round(sum(c["kc_l"] for c in tail)
                                           / dt_s, 2)
        steady["KC_right_spikes_s"] = round(sum(c["kc_r"] for c in tail)
                                            / dt_s, 2)
        rec["phases"].append(dict(
            dur_ms=dur_ms, rates=dict(rates),
            n_spikes=sum(c["n_spikes_new"] for c in chunks),
            n_active=len({k for c in chunks
                          for k in c["pop_counts"]}),   # pop-level only
            steady_rates_hz=steady))
        kc_l_spikes += sum(c["kc_l"] for c in chunks)
        kc_r_spikes += sum(c["kc_r"] for c in chunks)
        kc_l_dep += [c["kc_l_dep"] for c in chunks]
        kc_r_dep += [c["kc_r_dep"] for c in chunks]
    rec["kc_total"] = dict(left_spikes=kc_l_spikes, right_spikes=kc_r_spikes,
                           left_dep_mv_mean=round(float(np.mean(kc_l_dep)),
                                                  4) if kc_l_dep else None,
                           right_dep_mv_mean=round(float(np.mean(kc_r_dep)),
                                                   4) if kc_r_dep else None)
    # full-network canonical spike hash over the schedule
    spikes = lb.brain.canonical_spikes_text()
    rec["spike_sha256"] = hashlib.sha256(spikes.encode()).hexdigest()[:16]
    rec["n_spikes_total"] = sum(len(v) for v in
                                lb.brain.spike_trains().values())
    return rec


# --------------------------------------------------------------------------
# the probe set
# --------------------------------------------------------------------------
def probe_schedules():
    S = 150.0
    return dict(
        P1_sugar=[
            (500, {}),
            (1000, {"sugar": S}),
            (500, {}),
        ],
        P1b_pam_direct=[
            (500, {}),
            (1000, {"pam": S}),
            (500, {}),
        ],
        P2_visual_kc=[
            (500, {}),
            (1000, {"target_left": S}),
            (500, {}),
            (1000, {"target_right": S}),
            (500, {}),
        ],
        P2b_kc_direct=[
            (500, {}),
            (1000, {"kc_stim": S}),
            (500, {}),
        ],
        P3_visual_kc_zero_weights=[       # same schedule as P2, weights=0
            (500, {}),
            (1000, {"target_left": S}),
            (500, {}),
            (1000, {"target_right": S}),
            (500, {}),
        ],
        P4a_mbon_approach=[
            (500, {}),
            (1000, {"mbon_appr": S}),
            (500, {}),
        ],
        P4b_mbon_avoidance=[
            (500, {}),
            (1000, {"mbon_avoid": S}),
            (500, {}),
        ],
        P5_repro=[
            (500, {}),
            (1000, {"target_left": S}),
            (500, {}),
            (1000, {"target_right": S}),
            (500, {}),
        ],
    )


def run_probes(seed=PROBE_SEED, chunk_ms=CHUNK_MS, out_dir=RESULTS):
    t0 = time.perf_counter()
    d11.build_learning_interface_spec()
    d11.build_learning_readout_spec()
    build_probe_interface_spec()

    channel_ids = {ch: d["ids"] for ch, d in
                   json.loads((DATA / "probe_interface.json").read_text())
                   ["channels"].items()}
    readout_pops = d11.load_learning_readout_pops()

    print("[probes] building LearningBrain (scan + canonical build)...",
          flush=True)
    lb = d11.LearningBrain(seed=seed, channel_ids=channel_ids,
                           readout_pops=readout_pops)
    print(f"[probes] build done in {time.perf_counter()-t0:.0f}s; "
          f"KC->MBON rows={len(lb.kc_rows)}", flush=True)

    sch = probe_schedules()
    zeros = np.zeros(len(lb.kc_rows))
    ones = np.ones(len(lb.kc_rows))

    results = {}

    def run(name, scale=None, s=None):
        rec = run_schedule(lb, sch[name], s or seed, chunk_ms=chunk_ms,
                           scale=scale, label=name)
        results[name] = rec
        # incremental save (crash-safe): each probe record to its own file
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"probe_{name}.json").write_text(
            json.dumps(rec, indent=1))
        print(f"[probe] {name}: spikes={rec['n_spikes_total']} "
              f"sha={rec['spike_sha256']}", flush=True)
        return rec

    run("P1_sugar")
    run("P1b_pam_direct")
    run("P2_visual_kc")
    run("P2b_kc_direct")
    run("P3_visual_kc_zero_weights", scale=zeros)
    run("P4a_mbon_approach")
    run("P4b_mbon_avoidance")
    run("P5_repro", scale=ones)

    # ---------------- analysis ----------------
    def phase(rec, i):
        return rec["phases"][i]

    def st(rec, i, key):
        return rec["phases"][i]["steady_rates_hz"].get(key, 0.0)

    ana = {}
    r1 = results["P1_sugar"]
    ana["P1_sugar_recruits_PAM"] = dict(
        baseline=st(r1, 0, "PAM_left") + st(r1, 0, "PAM_right"),
        sugar=st(r1, 1, "PAM_left") + st(r1, 1, "PAM_right"),
        PAM_left=st(r1, 1, "PAM_left"), PAM_right=st(r1, 1, "PAM_right"),
        PPL1=st(r1, 1, "PPL1_left") + st(r1, 1, "PPL1_right"),
        KC_left=st(r1, 1, "KC_left"), KC_right=st(r1, 1, "KC_right"))
    r1b = results["P1b_pam_direct"]
    ana["P1b_pam_direct"] = dict(
        PAM=st(r1b, 1, "PAM_left") + st(r1b, 1, "PAM_right"),
        MBON_appr=st(r1b, 1, "MBON_approach_left")
        + st(r1b, 1, "MBON_approach_right"),
        MBON_avoid=st(r1b, 1, "MBON_avoidance_left")
        + st(r1b, 1, "MBON_avoidance_right"),
        DNs=st(r1b, 1, "DN_ALL_left") + st(r1b, 1, "DN_ALL_right"))
    r2 = results["P2_visual_kc"]
    ana["P2_visual_kc_side_specificity"] = dict(
        left_drive=dict(kc_left_spikes_s=st(r2, 1, "KC_left_spikes_s"),
                        kc_right_spikes_s=st(r2, 1, "KC_right_spikes_s"),
                        kc_left_dep_mv=phase(r2, 1)["steady_rates_hz"],
                        kc_left_dep_mean=results["P2_visual_kc"]
                        ["kc_total"]["left_dep_mv_mean"]),
        right_drive=dict(kc_left_spikes_s=st(r2, 3, "KC_left_spikes_s"),
                         kc_right_spikes_s=st(r2, 3, "KC_right_spikes_s")),
        kc_left_dep_mean=results["P2_visual_kc"]["kc_total"][
            "left_dep_mv_mean"],
        kc_right_dep_mean=results["P2_visual_kc"]["kc_total"][
            "right_dep_mv_mean"],
        MBON_appr_left=st(r2, 1, "MBON_approach_left"),
        MBON_avoid_left=st(r2, 1, "MBON_avoidance_left"))
    r2b = results["P2b_kc_direct"]
    ana["P2b_kc_direct_mbon_gain"] = dict(
        MBON_appr=st(r2b, 1, "MBON_approach_left")
        + st(r2b, 1, "MBON_approach_right"),
        MBON_avoid=st(r2b, 1, "MBON_avoidance_left")
        + st(r2b, 1, "MBON_avoidance_right"),
        MBON_unclass=st(r2b, 1, "MBON_unclassified_left")
        + st(r2b, 1, "MBON_unclassified_right"),
        DNs=st(r2b, 1, "DN_ALL_left") + st(r2b, 1, "DN_ALL_right"))
    r3 = results["P3_visual_kc_zero_weights"]
    ana["P3_kc_mbon_silencing_impact"] = dict(
        n_spikes_canonical=results["P2_visual_kc"]["n_spikes_total"],
        n_spikes_zero_kc_mbon=r3["n_spikes_total"],
        spike_sha_canonical=results["P2_visual_kc"]["spike_sha256"],
        spike_sha_zero=r3["spike_sha256"],
        differs=(r3["spike_sha256"]
                 != results["P2_visual_kc"]["spike_sha256"]),
        P9_left_canonical=st(r2, 1, "P9_left"),
        P9_left_zero=st(r3, 1, "P9_left"))
    r4a = results["P4a_mbon_approach"]
    r4b = results["P4b_mbon_avoidance"]
    ana["P4_mbon_to_dn_expression"] = dict(
        approach_stim=dict(
            P9_left=st(r4a, 1, "P9_left"), P9_right=st(r4a, 1, "P9_right"),
            BPN=st(r4a, 1, "BPN_bilateral"), RRN=st(r4a, 1, "RRN_bilateral"),
            MDN=st(r4a, 1, "MDN_bilateral"),
            DN_ALL_L=st(r4a, 1, "DN_ALL_left"),
            DN_ALL_R=st(r4a, 1, "DN_ALL_right")),
        avoidance_stim=dict(
            P9_left=st(r4b, 1, "P9_left"), P9_right=st(r4b, 1, "P9_right"),
            BPN=st(r4b, 1, "BPN_bilateral"), RRN=st(r4b, 1, "RRN_bilateral"),
            MDN=st(r4b, 1, "MDN_bilateral"),
            DN_ALL_L=st(r4b, 1, "DN_ALL_left"),
            DN_ALL_R=st(r4b, 1, "DN_ALL_right")),
        baseline=dict(
            P9_left=st(r4a, 0, "P9_left"), P9_right=st(r4a, 0, "P9_right"),
            BPN=st(r4a, 0, "BPN_bilateral"), MDN=st(r4a, 0, "MDN_bilateral")))
    ana["P5_repro_bit_identical"] = dict(
        sha_P2=results["P2_visual_kc"]["spike_sha256"],
        sha_P5=results["P5_repro"]["spike_sha256"],
        identical=(results["P2_visual_kc"]["spike_sha256"]
                   == results["P5_repro"]["spike_sha256"]))

    out = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        seed=seed, chunk_ms=chunk_ms,
        tier_labels=dict(
            P1="BIOLOGICAL entry (GRN sugar) + canonical dynamics",
            P1b="ENGINEERED entry (direct PAM, optogenetic-style)",
            P2="BIOLOGICAL entry (LC9 visual) + canonical dynamics",
            P2b="ENGINEERED entry (direct KC sample)",
            P3="MODELED weight manipulation at the canonical KC->MBON site",
            P4="ENGINEERED entry (direct MBON class stimulation)",
            P5="protocol determinism check"),
        pop_sizes=lb.brain.pop_sizes,
        probes=results,
        analysis=ana,
        wall_s=round(time.perf_counter() - t0, 1),
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir / "probes.json").write_text(json.dumps(out, indent=1))
    # compact summary for the console
    print("\n===== PROBE ANALYSIS =====")
    print(json.dumps(ana, indent=1))
    print(f"[probes] full record -> {out_dir / 'probes.json'} "
          f"({out['wall_s']}s)")
    return out


# --------------------------------------------------------------------------
# P6 - bilateral interface calibration (pre-matrix, from pilot evidence)
# --------------------------------------------------------------------------
def calibrate_bilateral(seed=PROBE_SEED, chunk_ms=CHUNK_MS,
                        out=RESULTS / "bilateral_calibration.json"):
    """The 12-trial pilot showed the frozen brain has an INNATE LEFT-turn
    preference under symmetric bilateral target drive (12/12 left commits;
    nonlinear network response - unilateral probes P2 could not predict
    it), plus an innate CS-valence asymmetry (trial-0 bias -2.5 Hz from the
    symmetric KC ensembles).  The task requires no-initial-preference, so
    the ENGINEERED interface gains are calibrated here (identically for all
    conditions; the A-control verifies neutrality empirically):

      phase 1: sweep target_left gain X (target_right fixed 150) ->
               steady P9_left - P9_right = 0
      phase 2: sweep kc_cs_left gain Y (kc_cs_right fixed 75, visual at
               the phase-1 solution) -> steady MBON valence diff = 0
               [(apprL-avoidL) - (apprR-avoidR) = 0]
    """
    import d11_sessions as dsr
    dsr.build_production_interface_spec()      # v2: with kc_cs_left/right
    d11.build_learning_readout_spec()
    build_probe_interface_spec()
    channel_ids = {ch: d["ids"] for ch, d in
                   json.loads((DATA / "probe_interface.json").read_text())
                   ["channels"].items()}
    readout_pops = d11.load_learning_readout_pops()
    print("[calib] building LearningBrain...", flush=True)
    t0 = time.perf_counter()
    lb = d11.LearningBrain(seed=seed, channel_ids=channel_ids,
                           readout_pops=readout_pops)
    print(f"[calib] build done {time.perf_counter()-t0:.0f}s", flush=True)

    def steady(rates, dur_ms=2000, seed_k=0, n_avg=3):
        """Steady-state rates over the last half of a drive period,
        averaged over n_avg seeded runs (P9 populations are single
        neurons - Poisson noise dominates single runs)."""
        acc = {k: 0.0 for k in SUMMARY_POPS}
        for j in range(n_avg):
            lb.reset_episode(seed + 100 + seed_k * 37 + j)
            n = int(round(dur_ms / chunk_ms))
            chunks = []
            for _ in range(n):
                chunks.append(lb.step_learn(rates, chunk_ms=chunk_ms))
            tail = chunks[len(chunks) // 2:]
            dt_s = len(tail) * chunk_ms / 1000.0
            for p in SUMMARY_POPS:
                n_sz = lb.brain.pop_sizes.get(p, 0)
                acc[p] += (sum(c["pop_counts"].get(p, 0) for c in tail)
                           / (n_sz * dt_s)) if n_sz else 0.0
        return {p: v / n_avg for p, v in acc.items()}

    def p9_diff(rates):
        return rates["P9_left"] - rates["P9_right"]

    def valence_diff(rates):
        return ((rates["MBON_approach_left"]
                 - rates["MBON_avoidance_left"])
                - (rates["MBON_approach_right"]
                   - rates["MBON_avoidance_right"]))

    # ---- phase 1: P9 balance ------------------------------------------
    sweep1 = []
    for x in (150.0, 110.0, 80.0, 55.0, 35.0, 20.0):
        rates = steady({"target_left": x, "target_right": 150.0,
                        "kc_cs_left": 75.0, "kc_cs_right": 75.0},
                       seed_k=int(x))
        d = p9_diff(rates)
        sweep1.append(dict(target_left_hz=x, p9_left=round(rates["P9_left"],
                                                           2),
                           p9_right=round(rates["P9_right"], 2),
                           p9_diff=round(d, 2)))
        print(f"[calib1] target_left={x:5.0f} -> P9_L={rates['P9_left']:.1f}"
              f" P9_R={rates['P9_right']:.1f} diff={d:+.1f}", flush=True)
    # linear interpolation for diff = 0 (largest +ve to most -ve)
    xs = [s["target_left_hz"] for s in sweep1]
    ys = [s["p9_diff"] for s in sweep1]
    x_star = _interp_zero(xs, ys)

    # ---- phase 2: CS valence balance (upward: KC->avoid-MBON gain
    # exceeds KC->appr, so RAISING the left CS drive lowers the valence
    # diff toward zero; pilot + calib evidence) -------------------------
    sweep2 = []
    for y in (75.0, 100.0, 125.0, 150.0, 175.0):
        rates = steady({"target_left": x_star, "target_right": 150.0,
                        "kc_cs_left": y, "kc_cs_right": 75.0},
                       seed_k=1000 + int(y))
        v = valence_diff(rates)
        sweep2.append(dict(kc_cs_left_hz=y,
                           valence_diff=round(v, 2),
                           appr_l=round(rates["MBON_approach_left"], 2),
                           avoid_l=round(rates["MBON_avoidance_left"], 2),
                           appr_r=round(rates["MBON_approach_right"], 2),
                           avoid_r=round(rates["MBON_avoidance_right"], 2)))
        print(f"[calib2] kc_cs_left={y:5.0f} -> valence_diff={v:+.2f}",
              flush=True)
    xs2 = [s["kc_cs_left_hz"] for s in sweep2]
    ys2 = [s["valence_diff"] for s in sweep2]
    y_star = _interp_zero(xs2, ys2)

    # ---- verification at the solution ---------------------------------
    verify = steady({"target_left": x_star, "target_right": 150.0,
                     "kc_cs_left": y_star, "kc_cs_right": 75.0},
                    seed_k=4242, n_avg=4)
    verify_rec = dict(
        p9_left=round(verify["P9_left"], 2),
        p9_right=round(verify["P9_right"], 2),
        p9_diff=round(p9_diff(verify), 2),
        valence_diff=round(valence_diff(verify), 2),
        appr_l=round(verify["MBON_approach_left"], 2),
        avoid_l=round(verify["MBON_avoidance_left"], 2),
        appr_r=round(verify["MBON_approach_right"], 2),
        avoid_r=round(verify["MBON_avoidance_right"], 2))
    print(f"[calib] VERIFY at solution: {verify_rec}", flush=True)

    rec = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"), seed=seed,
        purpose="restore no-initial-preference for the D11 choice task "
                "(pilot evidence: innate left-turn + innate CS-valence "
                "asymmetry); gains applied IDENTICALLY to all conditions",
        phase1_p9_balance=sweep1, target_left_hz_calibrated=round(x_star, 1),
        phase2_cs_valence_balance=sweep2,
        kc_cs_left_hz_calibrated=round(y_star, 1),
        verification_at_solution=verify_rec,
        note="ENGINEERED interface calibration (pre-registered before the "
             "matrix); the A_no_plastic control verifies neutrality.  The "
             "calibrated rates are PRE-COMMIT interface rates (targets at "
             "+/-50 deg); the runtime encoders scale the full tuning "
             "curves by the same ratios.",
        wall_s=round(time.perf_counter() - t0, 1),
    )
    Path(out).write_text(json.dumps(rec, indent=1))
    print(f"[calib] SOLUTION: target_left={x_star:.1f} Hz "
          f"kc_cs_left={y_star:.1f} Hz -> {out}")
    return rec


def _interp_zero(xs, ys):
    """Linear interpolation of the zero crossing (largest y to smallest)."""
    pts = sorted(zip(xs, ys))
    # prefer the bracketing pair around 0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 == 0:
            return x0
        if y0 * y1 < 0:
            return x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    # fall back: nearest to zero
    return max(pts, key=lambda p: -abs(p[1]))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "interface", "calibrate"])
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--chunk-ms", type=float, default=CHUNK_MS)
    args = ap.parse_args()
    if args.cmd == "interface":
        build_probe_interface_spec()
    elif args.cmd == "calibrate":
        calibrate_bilateral(seed=args.seed, chunk_ms=args.chunk_ms)
    else:
        run_probes(seed=args.seed, chunk_ms=args.chunk_ms)


if __name__ == "__main__":
    main()
