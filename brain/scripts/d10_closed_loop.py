"""D10 closed loop: ARTIFICIAL ENVIRONMENT -> SENSORY ENCODER -> DIGITAL
DROSOPHILA -> DESCENDING NEURAL ACTIVITY -> MOTOR DECODER -> AGENT MOVEMENT
-> NEW VISUAL STATE -> repeat.

A REAL closed loop: the environment knows the target location; the sensory
encoder converts it into neural stimulation; the canonical brain processes
it; the motor decoder reads ONLY descending-neuron spike counts; only then
does the agent move. There is NO hidden target_position -> correct_action
shortcut anywhere in the intact condition (the scripted-optimal controller
exists ONLY as an explicit, labelled control B).

FIRST TASK: TARGET APPROACH - agent starts at the centre, target randomly
left / right / centre; the digital fly should receive visual stimulation,
produce neural activity, emit turning/movement, orient toward and approach
the target. NO LEARNING - this stage establishes FIXED sensorimotor control
with the frozen connectome.

Control conditions (pre-registered, docs/ROADMAP D10):
  A random            seeded random actions (no brain)
  B scripted          direct bearing->action optimal controller (no brain)
  C intact            the fixed Drosophila brain closed loop
  D shuffled_sensory  brain runs, but each channel stimulates a WRONG
                      (random, seeded) same-size set of visual neurons
  E shuffled_motor    brain runs intact, but the decoder reads permuted
                      (wrong) readout populations

Scenarios: target_left / target_right / target_center / no_target / looming.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402
import d7_runtime as rt  # noqa: E402
import d8_encoder as d8  # noqa: E402
import d9_decoder as d9  # noqa: E402

DATA = bl.ROOT / "data" / "d7_d10"
RESULTS = bl.ROOT / "results" / "d10_closed_loop"
PY = sys.executable

CONDITIONS = ["random", "scripted", "intact", "shuffled_sensory",
              "shuffled_motor"]
SCENARIOS = ["target_left", "target_right", "target_center", "no_target",
             "looming"]
REPS = 3
MASTER_SEED = 20260928

SCENARIO_PARAMS = {
    # bearing sign: POSITIVE = target to the LEFT of heading (see Arena2D)
    "target_left":   dict(target_bearing_deg=+60.0, target_dist=0.8,
                          target_present=True, cap_s=5.0),
    "target_right":  dict(target_bearing_deg=-60.0, target_dist=0.8,
                          target_present=True, cap_s=5.0),
    "target_center": dict(target_bearing_deg=0.0, target_dist=0.8,
                          target_present=True, cap_s=5.0),
    "no_target":     dict(target_present=False, cap_s=3.0),
    "looming":       dict(target_present=False, loom=True, cap_s=3.0),
}


def write_preregistration(path=DATA / "preregistration.json"):
    """Freeze EVERYTHING measurable before the matrix runs."""
    reg = dict(
        version=1,
        written_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        brain="canonical Shiu LIF model on FlyWire v783 (unmodified), "
              "interactive chunked runtime (d7_runtime.py)",
        chunk_ms=50.0,
        conditions=CONDITIONS,
        scenarios=SCENARIOS,
        reps=REPS,
        master_seed=MASTER_SEED,
        seed_rule="episode seed = master + 1000*episode_index + rep",
        scenario_params=SCENARIO_PARAMS,
        arena=dict(
            half_size=1.0, reach_radius=0.15, v_forward=0.25, v_turn=0.15,
            v_backward=0.15, omega_turn_deg_s=150.0, jump_hop=0.30,
            bearing_jitter_deg=8.0,
            loom=dict(t_on_s=0.5, rise_s=1.5, max_angular_deg=70.0),
        ),
        encoder=dict(
            r_target_hz=150.0, r_loom_hz=150.0, theta0_deg=15.0,
            theta_w_deg=30.0, loom_onset_deg=15.0, loom_sat_deg=60.0,
            note="triangular hemifield tuning; bilateral centre zone"),
        decoder=dict(
            window_ms=300.0, theta_jump_hz=25.0, theta_bwd_hz=4.0,
            theta_fwd_hz=5.0, theta_diff_hz=8.0, delta_hyst_hz=4.0,
            theta_stop_hz=4.0, theta_min_walk_hz=1.5, min_action_chunks=2,
            turn_source="P9",
            note="frozen after D8 calibration; same config for intact and "
                 "shuffled-motor conditions"),
        controls=dict(
            random="uniform over FORWARD/BACKWARD/TURN_LEFT/TURN_RIGHT/STOP",
            shuffled_sensory="each channel drives a wrong same-size random "
                             "subset of LC9 U LPLC2 U LC4 (seeded per episode)",
            shuffled_motor="decoder population slots permuted (seeded per "
                           "episode), brain and interface intact",
        ),
        metrics=["success", "time_to_target", "path_efficiency",
                 "orientation_accuracy_mean_abs_bearing", "wrong_turn_rate",
                 "stop_frequency", "escape_emitted", "escape_latency_ms",
                 "decision_latency_wall_ms", "bio_wall_ratio",
                 "dn_rate_stats", "cpu", "ram"],
        gate3_criteria=dict(
            G3_1="intact closed loop: reproducible neural->motor actions "
                 "that change the environment; target-approach success "
                 "clearly above the random control",
            G3_2="shuffled sensory control degrades vs intact",
            G3_3="shuffled motor control degrades vs intact",
            G3_4="random controller substantially worse than intact if "
                 "meaningful behavior exists",
            honesty="if the fixed brain does not outperform controls, "
                    "report FAIL - no post-hoc tuning to force a pass",
        ),
    )
    DATA.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(reg, indent=1))
    print(f"[prereg] frozen -> {path}")
    return reg


# --------------------------------------------------------------------------
# one closed-loop episode (brain conditions; runs in its own subprocess)
# --------------------------------------------------------------------------
def run_brain_episode(cfg):
    """cfg: dict(condition, scenario, rep, seed, episode_dir, interface,
    motor_shuffle_seed, chunk_ms). Writes JSONL + summary + spikes."""
    scenario = cfg["scenario"]
    sp = SCENARIO_PARAMS[scenario]
    chunk_ms = cfg.get("chunk_ms", 50.0)
    dt = chunk_ms / 1000.0

    interface_spec = json.loads(Path(cfg["interface"]).read_text())
    channel_ids = {ch: d["ids"] for ch, d in
                   interface_spec["channels"].items()}
    readout_pops = d8.load_readout_pops()

    brain = rt.DualModeBrain("interactive", seed=cfg["seed"],
                             channel_ids=channel_ids,
                             readout_pops=readout_pops)
    encoder = d8.VisualSensoryEncoder(**{
        k: v for k, v in dict(
            r_target=150.0, r_loom=150.0, theta0=15.0, theta_w=30.0,
            loom_onset_deg=15.0, loom_sat_deg=60.0).items()})
    dec_params = dict(window_ms=300.0, theta_jump_hz=25.0, theta_bwd_hz=4.0,
                      theta_fwd_hz=5.0, theta_diff_hz=8.0, delta_hyst_hz=4.0,
                      theta_stop_hz=4.0, theta_min_walk_hz=1.5,
                      min_action_chunks=2, turn_source="P9")
    decoder = d9.MotorDecoder(brain.pop_sizes, dec_params)

    motor_map = None
    if cfg["condition"] == "shuffled_motor":
        motor_map = d9.MotorDecoder.shuffled_slots(
            brain.pop_sizes, cfg["motor_shuffle_seed"])

    arena = d8.Arena2D(
        target_bearing_deg=sp.get("target_bearing_deg", 0.0),
        target_dist=sp.get("target_dist", 0.8),
        target_present=sp["target_present"],
        loom=sp.get("loom", False),
        jitter_deg=8.0, seed=cfg["seed"])

    brain.reset_episode(cfg["seed"])
    ep_dir = Path(cfg["episode_dir"])
    ep_dir.mkdir(parents=True, exist_ok=True)
    jsonl = ep_dir / "chunks.jsonl"
    n_chunks = int(sp["cap_s"] * 1000 / chunk_ms)
    t_start_wall = time.perf_counter()
    path_length = 0.0
    px, py = arena.x, arena.y
    outcome = "cap"
    loom_onset_t = None
    escape_t = None
    _exact_rates = []

    with jsonl.open("w") as fh:
        for k in range(n_chunks):
            vis = arena.visual_state()
            rates = encoder.rates(vis)
            _exact_rates.append((vis["t"], rates))
            r = brain.step(rates, chunk_ms=chunk_ms)

            counts = dict(r.get("pop_counts", {}))
            if motor_map is not None:
                counts = {wrong: counts.get(orig, 0)
                          for orig, wrong in motor_map.items()}
            dec = decoder.decode(counts, chunk_ms)
            action = dec["action"]

            if loom_onset_t is None and vis["loom_angular_deg"] > 0:
                loom_onset_t = vis["t"]
            if escape_t is None and action == "JUMP":
                escape_t = vis["t"]

            arena.step(action, dt)
            path_length += math.hypot(arena.x - px, arena.y - py)
            px, py = arena.x, arena.y

            row = dict(k=k, t=round(vis["t"], 3), ax=round(arena.x, 4),
                       ay=round(arena.y, 4),
                       heading_deg=round(math.degrees(arena.heading), 1),
                       action=action,
                       target_present=vis["target_present"],
                       bearing_deg=(None if vis["target_bearing_deg"] is None
                                    else round(vis["target_bearing_deg"], 1)),
                       dist=(None if vis["target_distance"] is None
                             else round(vis["target_distance"], 4)),
                       loom_deg=round(vis["loom_angular_deg"], 1),
                       rates={c: round(v, 1) for c, v in rates.items()},
                       pop_counts=counts, p9_diff=dec["p9_diff_hz"],
                       gf_hz=dec["gf_hz"], walk_hz=dec["walk_mean_hz"],
                       conflicts=dec["conflicts"],
                       wall_ms=round(r["wall_s"] * 1000, 1),
                       rss_mb=round(r["rss_kb"] / 1024, 1))
            fh.write(json.dumps(row) + "\n")

            if arena.reached_target():
                outcome = "success"
                break
            if arena.escaped and escape_t is None:
                escape_t = arena.t
    wall_total = time.perf_counter() - t_start_wall

    # episode summary + metrics
    spikes = brain.spike_trains()
    n_bio_chunks = k + 1
    bio_s = n_bio_chunks * dt
    rows = [json.loads(l) for l in jsonl.read_text().splitlines()]
    target_rows = [r_ for r_ in rows if r_["target_present"]]
    wrong_turns = sum(
        1 for r_ in rows
        if (r_["action"] == "TURN_LEFT" and (r_["bearing_deg"] or 0) < -5)
        or (r_["action"] == "TURN_RIGHT" and (r_["bearing_deg"] or 0) > 5))
    turn_chunks = sum(1 for r_ in rows
                      if r_["action"] in ("TURN_LEFT", "TURN_RIGHT"))
    mean_abs_bearing = (float(np.mean([abs(r_["bearing_deg"])
                                       for r_ in target_rows]))
                        if target_rows else None)
    start_dist = (rows[0]["dist"] if rows and rows[0]["dist"] is not None
                  else None)
    summary = dict(
        condition=cfg["condition"], scenario=scenario, rep=cfg["rep"],
        seed=cfg["seed"], chunk_ms=chunk_ms, outcome=outcome,
        n_chunks=n_bio_chunks, bio_s=round(bio_s, 2),
        wall_s=round(wall_total, 1),
        bio_per_wall=round(bio_s / wall_total, 4),
        success=(outcome == "success"),
        time_to_target=(round(rows[-1]["t"], 2)
                        if outcome == "success" else None),
        path_length=round(path_length, 3),
        path_efficiency=(round(start_dist / path_length, 3)
                         if (start_dist and path_length > 0) else None),
        mean_abs_bearing_deg=(round(mean_abs_bearing, 1)
                              if mean_abs_bearing is not None else None),
        wrong_turn_rate=round(wrong_turns / turn_chunks, 3)
        if turn_chunks else 0.0,
        stop_frequency=round(sum(1 for r_ in rows if r_["action"] == "STOP")
                             / n_bio_chunks, 3),
        escape_emitted=arena.escaped,
        escape_latency_ms=(round((escape_t - loom_onset_t) * 1000, 0)
                           if (escape_t is not None
                               and loom_onset_t is not None) else None),
        n_spikes_total=sum(len(v) for v in spikes.values()),
        n_conflicts=len(decoder.conflict_log),
        conflict_examples=decoder.conflict_log[:5],
        decision_latency_wall_ms=dict(
            mean=round(float(np.mean([r_["wall_ms"] for r_ in rows])), 1),
            p95=round(float(np.percentile([r_["wall_ms"] for r_ in rows],
                                           95)), 1)),
        peak_rss_mb=round(bl.peak_rss_kb() / 1024, 1),
        cpu_user_s=round(os.times()[0], 1),
        motor_shuffle_map=motor_map,
        encoder=encoder.describe(),
        arena_end=dict(x=round(arena.x, 3), y=round(arena.y, 3),
                       escaped=arena.escaped, jumps=arena.jump_count),
    )
    # mean DN rates over the episode
    pops = ["P9_left", "P9_right", "GF_left", "GF_right", "MDN_bilateral",
            "BPN_bilateral", "RRN_bilateral", "DN_ALL_left", "DN_ALL_right"]
    dur = bio_s
    summary["dn_rate_hz"] = {
        p: round(sum(r_["pop_counts"].get(p, 0) for r_ in rows)
                 / (brain.pop_sizes.get(p, 1) * dur), 2) for p in pops}
    (ep_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (ep_dir / "spikes.txt").write_text(brain.canonical_spikes_text())
    # per-chunk rate schedule at FULL precision for exact replay
    (ep_dir / "rate_schedule.json").write_text(json.dumps(
        [[row["t"], rates] for row, (t_, rates) in zip(rows, _exact_rates)]))
    print(f"[episode] {cfg['condition']}/{scenario}/r{cfg['rep']} -> "
          f"{outcome} wall={wall_total:.0f}s spikes={summary['n_spikes_total']}")
    return summary


# --------------------------------------------------------------------------
# no-brain control episodes (A random, B scripted) - in-process, fast
# --------------------------------------------------------------------------
def run_no_brain_episode(cfg):
    scenario = cfg["scenario"]
    sp = SCENARIO_PARAMS[scenario]
    chunk_ms = cfg.get("chunk_ms", 50.0)
    dt = chunk_ms / 1000.0
    rng = np.random.default_rng(cfg["seed"])
    arena = d8.Arena2D(
        target_bearing_deg=sp.get("target_bearing_deg", 0.0),
        target_dist=sp.get("target_dist", 0.8),
        target_present=sp["target_present"], loom=sp.get("loom", False),
        jitter_deg=8.0, seed=cfg["seed"])
    encoder = d8.VisualSensoryEncoder()
    ep_dir = Path(cfg["episode_dir"])
    ep_dir.mkdir(parents=True, exist_ok=True)
    jsonl = ep_dir / "chunks.jsonl"
    n_chunks = int(sp["cap_s"] * 1000 / chunk_ms)
    t0 = time.perf_counter()
    path_length = 0.0
    px, py = arena.x, arena.y
    outcome = "cap"
    loom_onset_t = None
    escape_t = None
    vocab = ["FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]

    with jsonl.open("w") as fh:
        for k in range(n_chunks):
            vis = arena.visual_state()
            rates = encoder.rates(vis)
            if cfg["condition"] == "random":
                action = vocab[int(rng.integers(len(vocab)))]
            else:                                   # scripted optimal
                b = vis["target_bearing_deg"]
                if vis["target_present"] and b is not None and abs(b) > 10:
                    action = "TURN_LEFT" if b > 0 else "TURN_RIGHT"
                else:
                    action = "FORWARD"
            if loom_onset_t is None and vis["loom_angular_deg"] > 0:
                loom_onset_t = vis["t"]
            if escape_t is None and action == "JUMP":
                escape_t = vis["t"]
            arena.step(action, dt)
            path_length += math.hypot(arena.x - px, arena.y - py)
            px, py = arena.x, arena.y
            fh.write(json.dumps(dict(
                k=k, t=round(vis["t"], 3), ax=round(arena.x, 4),
                ay=round(arena.y, 4),
                heading_deg=round(math.degrees(arena.heading), 1),
                action=action, target_present=vis["target_present"],
                bearing_deg=(None if vis["target_bearing_deg"] is None
                             else round(vis["target_bearing_deg"], 1)),
                dist=(None if vis["target_distance"] is None
                      else round(vis["target_distance"], 4)),
                loom_deg=round(vis["loom_angular_deg"], 1),
                rates={c: round(v, 1) for c, v in rates.items()},
                pop_counts={}, conflicts=[], wall_ms=0.0, rss_mb=0.0)) + "\n")
            if arena.reached_target():
                outcome = "success"
                break
    wall_total = time.perf_counter() - t0
    rows = [json.loads(l) for l in jsonl.read_text().splitlines()]
    target_rows = [r_ for r_ in rows if r_["target_present"]]
    wrong_turns = sum(1 for r_ in rows
                      if (r_["action"] == "TURN_LEFT"
                          and (r_["bearing_deg"] or 0) < -5)
                      or (r_["action"] == "TURN_RIGHT"
                          and (r_["bearing_deg"] or 0) > 5))
    turn_chunks = sum(1 for r_ in rows
                      if r_["action"] in ("TURN_LEFT", "TURN_RIGHT"))
    mean_abs_bearing = (float(np.mean([abs(r_["bearing_deg"])
                                       for r_ in target_rows]))
                        if target_rows else None)
    start_dist = rows[0]["dist"] if rows and rows[0]["dist"] is not None else None
    summary = dict(
        condition=cfg["condition"], scenario=scenario, rep=cfg["rep"],
        seed=cfg["seed"], chunk_ms=chunk_ms, outcome=outcome,
        n_chunks=len(rows), bio_s=round(len(rows) * dt, 2),
        wall_s=round(wall_total, 2), bio_per_wall=None,
        success=(outcome == "success"),
        time_to_target=(round(rows[-1]["t"], 2)
                        if outcome == "success" else None),
        path_length=round(path_length, 3),
        path_efficiency=(round(start_dist / path_length, 3)
                         if (start_dist and path_length > 0) else None),
        mean_abs_bearing_deg=(round(mean_abs_bearing, 1)
                              if mean_abs_bearing is not None else None),
        wrong_turn_rate=round(wrong_turns / turn_chunks, 3) if turn_chunks else 0.0,
        stop_frequency=round(sum(1 for r_ in rows if r_["action"] == "STOP")
                             / len(rows), 3),
        escape_emitted=arena.escaped,
        escape_latency_ms=(round((escape_t - loom_onset_t) * 1000, 0)
                           if (escape_t is not None
                               and loom_onset_t is not None) else None),
        n_spikes_total=0, n_conflicts=0,
        decision_latency_wall_ms=dict(mean=0.0, p95=0.0),
        peak_rss_mb=0.0, cpu_user_s=0.0, motor_shuffle_map=None,
        encoder=encoder.describe(),
        arena_end=dict(x=round(arena.x, 3), y=round(arena.y, 3),
                       escaped=arena.escaped, jumps=arena.jump_count),
        dn_rate_hz={})
    (ep_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[episode] {cfg['condition']}/{scenario}/r{cfg['rep']} -> "
          f"{outcome} wall={wall_total:.2f}s")
    return summary


# --------------------------------------------------------------------------
# matrix orchestration
# --------------------------------------------------------------------------
def shuffled_interface_spec(seed, out_path):
    """D control: each channel drives a WRONG same-size random subset of the
    visual entry pool (LC9 U LPLC2 U LC4), excluding the intended sets."""
    spec = json.loads((DATA / "sensory_interface.json").read_text())
    pool = set(d8.load_ids("LC9")) | set(d8.load_ids("LPLC2")) \
        | set(d8.load_ids("LC4"))
    intended = set()
    for ch in spec["channels"].values():
        intended.update(int(i) for i in ch["ids"])
    pool = sorted(pool - intended)
    rng = np.random.default_rng(seed)
    for ch in spec["channels"].values():
        n = len(ch["ids"])
        pick = rng.choice(np.array(pool), size=n, replace=False)
        ch["ids"] = sorted(int(x) for x in pick)
        ch["shuffled_control"] = True
    Path(out_path).write_text(json.dumps(spec, indent=1))
    return out_path


def run_matrix(out_dir=RESULTS, conditions=None, scenarios=None, reps=None,
               chunk_ms=50.0, interface=None):
    import pandas as pd

    conditions = conditions or CONDITIONS
    scenarios = scenarios or SCENARIOS
    reps = reps if reps is not None else REPS
    interface = interface or (DATA / "sensory_interface.json")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_path = DATA / "preregistration.json"
    if not reg_path.exists():
        write_preregistration(reg_path)

    episode_idx = 0
    all_summaries = []
    t_all = time.perf_counter()
    for cond in conditions:
        for scen in scenarios:
            for rep in range(reps):
                seed = MASTER_SEED + 1000 * episode_idx + rep
                ep_dir = out_dir / "episodes" / f"{cond}_{scen}_r{rep}"
                summary_path = ep_dir / "summary.json"
                if summary_path.exists():
                    s = json.loads(summary_path.read_text())
                    all_summaries.append(s)
                    print(f"[skip] {ep_dir.name} exists")
                    episode_idx += 1
                    continue
                cfg = dict(condition=cond, scenario=scen, rep=rep, seed=seed,
                            episode_dir=str(ep_dir), chunk_ms=chunk_ms)
                if cond in ("random", "scripted"):
                    s = run_no_brain_episode(cfg)
                else:
                    iface = interface
                    if cond == "shuffled_sensory":
                        iface = shuffled_interface_spec(
                            seed, ep_dir.parent
                            / f"shuffled_iface_{cond}_{scen}_r{rep}.json")
                        ep_dir.mkdir(parents=True, exist_ok=True)
                    cfg["interface"] = str(iface)
                    cfg["motor_shuffle_seed"] = seed + 7
                    # subprocess per brain episode (Gate-1 memory discipline)
                    cfg_path = ep_dir / "cfg.json"
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    cfg_path.write_text(json.dumps(cfg))
                    cmd = [PY, str(Path(__file__).resolve()),
                           "_episode_worker", "--cfg", str(cfg_path)]
                    t0 = time.perf_counter()
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode != 0:
                        print(proc.stdout[-3000:])
                        print(proc.stderr[-3000:])
                        raise SystemExit(f"episode failed: {cfg}")
                    s = json.loads(summary_path.read_text())
                    s["outer_wall_s"] = round(time.perf_counter() - t0, 1)
                all_summaries.append(s)
                episode_idx += 1
    print(f"[matrix] {len(all_summaries)} episodes in "
          f"{time.perf_counter() - t_all:.0f}s")

    # ---- aggregate
    df = pd.DataFrame([{k: v for k, v in s.items()
                        if k in ("condition", "scenario", "rep", "seed",
                                 "outcome", "success", "time_to_target",
                                 "path_efficiency", "mean_abs_bearing_deg",
                                 "wrong_turn_rate", "stop_frequency",
                                 "escape_emitted", "escape_latency_ms",
                                 "bio_s", "wall_s", "n_spikes_total",
                                 "n_conflicts", "decision_latency_wall_ms",
                                 "dn_rate_hz")}
                       for s in all_summaries])
    for k in ("mean", "p95"):
        df[f"latency_{k}_ms"] = df["decision_latency_wall_ms"].apply(
            lambda d, k=k: (d or {}).get(k))
    df.drop(columns=["decision_latency_wall_ms"]).to_csv(
        out_dir / "experiments.csv", index=False)

    agg = aggregate(all_summaries)
    verdict = evaluate_gate3(agg, df, all_summaries)
    bl.save_json(out_dir / "verdict.json", verdict)
    bl.save_json(out_dir / "aggregates.json", agg)
    print(json.dumps({k: v.get("ok") for k, v in verdict["criteria"].items()},
                     indent=1))
    print("GATE 3 OVERALL:", "PASS" if verdict["overall_pass"] else "FAIL")
    return verdict


def aggregate(summaries):
    """Per (condition, scenario) aggregate metrics."""
    agg = {}
    for cond in CONDITIONS:
        for scen in SCENARIOS:
            rows = [s for s in summaries
                    if s["condition"] == cond and s["scenario"] == scen]
            if not rows:
                continue
            n = len(rows)
            tgt = [s for s in rows if s["scenario"].startswith("target_")]
            agg[f"{cond}|{scen}"] = dict(
                n=n,
                success_rate=round(sum(s["success"] for s in rows) / n, 3),
                mean_time_to_target=(round(float(np.mean(
                    [s["time_to_target"] for s in tgt
                     if s["time_to_target"] is not None])), 2)
                    if any(s["time_to_target"] is not None for s in tgt)
                    else None),
                mean_path_efficiency=(round(float(np.mean(
                    [s["path_efficiency"] for s in tgt
                     if s["path_efficiency"] is not None])), 3)
                    if any(s["path_efficiency"] is not None for s in tgt)
                    else None),
                mean_abs_bearing_deg=(round(float(np.mean(
                    [s["mean_abs_bearing_deg"] for s in tgt
                     if s["mean_abs_bearing_deg"] is not None])), 1)
                    if any(s["mean_abs_bearing_deg"] is not None
                           for s in tgt) else None),
                wrong_turn_rate=round(float(np.mean(
                    [s["wrong_turn_rate"] for s in rows])), 3),
                stop_frequency=round(float(np.mean(
                    [s["stop_frequency"] for s in rows])), 3),
                escape_rate=round(sum(s["escape_emitted"] for s in rows) / n,
                                  3),
                mean_escape_latency_ms=(round(float(np.mean(
                    [s["escape_latency_ms"] for s in rows
                     if s["escape_latency_ms"] is not None])), 0)
                    if any(s["escape_latency_ms"] is not None
                           for s in rows) else None),
                spikes_mean=round(float(np.mean(
                    [s["n_spikes_total"] for s in rows])), 0),
            )
    return agg


def bootstrap_diff(a, b, n_boot=5000, seed=1):
    """Bootstrap CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return None
    diffs = []
    for _ in range(n_boot):
        da = rng.choice(a, size=len(a), replace=True).mean()
        db = rng.choice(b, size=len(b), replace=True).mean()
        diffs.append(da - db)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(diff=round(float(a.mean() - b.mean()), 3),
                ci95=[round(float(lo), 3), round(float(hi), 3)],
                significant=bool(lo > 0 or hi < 0))


def evaluate_gate3(agg, df, summaries):
    """Gate 3 verdict - fails closed, no post-hoc tuning."""
    tgt_scen = [s for s in SCENARIOS if s.startswith("target_")]

    def succ(cond, scen=None):
        key = f"{cond}|{scen}" if scen else None
        if key:
            return agg.get(key, {}).get("success_rate")
        vals = [agg[f"{cond}|{s}"]["success_rate"] for s in tgt_scen
                if f"{cond}|{s}" in agg]
        return round(float(np.mean(vals)), 3) if vals else None

    intact_succ = succ("intact")
    random_succ = succ("random")
    shufS_succ = succ("shuffled_sensory")
    shufM_succ = succ("shuffled_motor")
    scripted_succ = succ("scripted")

    crit = {}
    crit["G3.1_intact_closed_loop"] = dict(
        intact_target_success=intact_succ,
        ok=bool(intact_succ is not None and intact_succ > 0
                and intact_succ > (random_succ or 0)))
    crit["G3.2_shuffled_sensory_degrades"] = dict(
        intact=intact_succ, shuffled_sensory=shufS_succ,
        ok=bool(shufS_succ is not None and intact_succ is not None
                and shufS_succ < intact_succ))
    crit["G3.3_shuffled_motor_degrades"] = dict(
        intact=intact_succ, shuffled_motor=shufM_succ,
        ok=bool(shufM_succ is not None and intact_succ is not None
                and shufM_succ < intact_succ))
    crit["G3.4_random_substantially_worse"] = dict(
        intact=intact_succ, random=random_succ,
        ok=bool(random_succ is not None and intact_succ is not None
                and intact_succ - random_succ >= 0.2))

    # bootstrap contrast on per-episode success (target scenarios, pooled)
    def pool(cond):
        return [1.0 if s["success"] else 0.0 for s in summaries
                if s["condition"] == cond and s["scenario"] in tgt_scen]
    contrasts = {}
    for other in ("random", "shuffled_sensory", "shuffled_motor", "scripted"):
        contrasts[f"intact_vs_{other}"] = bootstrap_diff(
            pool("intact"), pool(other))
    crit["contrasts_bootstrap"] = contrasts
    crit["reference_values"] = dict(scripted=scripted_succ,
                                    looming_escape=dict(
                                        intact=agg.get(
                                            "intact|looming", {}).get(
                                                "escape_rate"),
                                        random=agg.get(
                                            "random|looming", {}).get(
                                                "escape_rate")))
    verdict = dict(
        gate="THE CONNECTOME-DERIVED DIGITAL DROSOPHILA CAN PERCEIVE A SIMPLE "
             "VISUAL TARGET THROUGH THE VERIFIED SENSORY INTERFACE AND "
             "PRODUCE REPRODUCIBLE MOTOR ACTION THAT CHANGES THE ENVIRONMENT "
             "IN CLOSED LOOP",
        criteria=crit,
        overall_pass=all(c.get("ok", False)
                         for k, c in crit.items()
                         if k.startswith("G3.")),
        honesty_note="Verdict computed from the pre-registered criteria "
                     "only; no parameter was tuned after seeing results.",
    )
    return verdict


# --------------------------------------------------------------------------
# replay verification (bit-exact)
# --------------------------------------------------------------------------
def exact_rate_schedule(episode_dir):
    """Reconstruct the EXACT (full float precision) per-chunk rate schedule
    of a recorded episode.

    chunks.jsonl stores rates rounded to 0.1 Hz (compact logs); replaying
    rounded rates would perturb the Poisson drive. The arena + encoder are
    deterministic pure functions of (seed, recorded actions), so the exact
    original rates are rebuilt here and cross-checked against the rounded
    log values (any mismatch raises - never silently accepted)."""
    ep = Path(episode_dir)
    summary = json.loads((ep / "summary.json").read_text())
    rows = [json.loads(l) for l in (ep / "chunks.jsonl").read_text()
            .splitlines() if l.strip()]
    sp = SCENARIO_PARAMS[summary["scenario"]]
    arena = d8.Arena2D(
        target_bearing_deg=sp.get("target_bearing_deg", 0.0),
        target_dist=sp.get("target_dist", 0.8),
        target_present=sp["target_present"],
        loom=sp.get("loom", False),
        jitter_deg=8.0, seed=summary["seed"])
    encoder = d8.VisualSensoryEncoder()
    sched = []
    chunk_ms = summary["chunk_ms"]
    for row in rows:
        vis = arena.visual_state()
        rates = encoder.rates(vis)
        for c, v in rates.items():
            if round(v, 1) != row["rates"][c]:
                raise AssertionError(
                    f"schedule reconstruction mismatch at chunk {row['k']} "
                    f"channel {c}: exact {v} vs logged {row['rates'][c]}")
        # uniform chunk durations (the log stores chunk START times; the
        # first pair must still last one full chunk, not zero)
        sched.append((chunk_ms, rates))
        arena.step(row["action"], chunk_ms / 1000.0)
    return sched


def verify_replay(episode_dir):
    """Re-run one recorded brain episode from its EXACT rate schedule + seed
    and check the neural trajectory is bit-identical (reproducibility)."""
    ep = Path(episode_dir)
    summary = json.loads((ep / "summary.json").read_text())
    sched = exact_rate_schedule(ep)
    iface = DATA / "sensory_interface.json"
    if summary["condition"] == "shuffled_sensory":
        iface = ep.parent / f"shuffled_iface_{summary['condition']}_" \
                            f"{summary['scenario']}_r{summary['rep']}.json"
    channel_ids = {ch: d["ids"] for ch, d in
                   json.loads(Path(iface).read_text())["channels"].items()}
    readout_pops = d8.load_readout_pops()
    brain = rt.DualModeBrain("interactive", seed=summary["seed"],
                             channel_ids=channel_ids,
                             readout_pops=readout_pops)
    brain.reset_episode(summary["seed"])
    chunk_ms = summary["chunk_ms"]
    for t, rates in sched:
        brain.step(rates, chunk_ms=chunk_ms)
    text = brain.canonical_spikes_text()
    orig = (ep / "spikes.txt").read_text()
    identical = text == orig
    out = dict(episode=str(ep), seed=summary["seed"],
               n_chunks=len(sched), identical=identical,
               sha_replay=rt.sha256_text(text)[:16],
               sha_original=rt.sha256_text(orig)[:16])
    (ep / "replay_verification.json").write_text(json.dumps(out, indent=1))
    print(f"[replay] {ep.name}: bit-identical={identical}")
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="D10 closed loop")
    ap.add_argument("cmd", choices=["prereg", "matrix", "_episode_worker",
                                    "verify-replay", "aggregate-only"])
    ap.add_argument("--cfg", default="")
    ap.add_argument("--episode-dir", default="")
    ap.add_argument("--conditions", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--interface", default="")
    args = ap.parse_args()

    if args.cmd == "prereg":
        write_preregistration()
    elif args.cmd == "_episode_worker":
        cfg = json.loads(Path(args.cfg).read_text())
        run_brain_episode(cfg)
    elif args.cmd == "verify-replay":
        verify_replay(args.episode_dir)
    elif args.cmd == "aggregate-only":
        import glob
        summaries = [json.loads(open(p).read()) for p in
                     glob.glob(str(Path(args.out_dir) / "episodes" / "*"
                                   / "summary.json"))]
        agg = aggregate(summaries)
        verdict = evaluate_gate3(agg, None, summaries)
        bl.save_json(Path(args.out_dir) / "verdict.json", verdict)
        bl.save_json(Path(args.out_dir) / "aggregates.json", agg)
        print("GATE 3 OVERALL:", "PASS" if verdict["overall_pass"] else "FAIL")
    elif args.cmd == "matrix":
        run_matrix(out_dir=args.out_dir,
                   conditions=args.conditions.split(",") if args.conditions else None,
                   scenarios=args.scenarios.split(",") if args.scenarios else None,
                   reps=args.reps,
                   interface=args.interface or None)


if __name__ == "__main__":
    main()
