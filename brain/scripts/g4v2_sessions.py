"""GATE 4 v2 — session matrix (battery + controls + reward-side
randomization; GATE4_V2_PREREG.md section 7).

One subprocess per session (one network build; RAM discipline), driven by
a detached orchestrator (start_new_session daemon; the sandbox reaps the
direct process group between calls, so the orchestrator must be detached).

Battery (per prereg 5/7): 6 baseline pref -> 4x(4 pair + 2 pref + 1
routeA) acquisition -> 8 filler + 3 pref retention -> 12 pref extinction
-> 4x(4 pair-OTHER + 2 pref + 1 routeA) reversal -> 3 pref x0.8 + 3 pref
x1.2 context -> 3 pref loom distractor.  94 trials/session.

Sessions (prereg section 7, pre-registered seeds/sides):
  B_R1..B_L3 (B_plastic, sides R/L/R/L/R/L), A_R, A_L (no plasticity),
  C_R (lagged dopamine gates), D_R (permuted eligibility->KC), F_R
  (untrained), E_rl (Q-learner, no brain).

Usage:
  python g4v2_sessions.py launch            # detached orchestrator, all
  python g4v2_sessions.py run --session B_R1
  python g4v2_sessions.py rl                # E baseline (fast, in-process)
  python g4v2_sessions.py status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402

BRAIN = bl.ROOT
G4 = BRAIN / "data" / "gate4_v2"
RESULTS = BRAIN / "results" / "gate4_v2"
PY = sys.executable

SESSIONS = {
    "B_R1": dict(cond="B_plastic", seed=20261101, side="right"),
    "B_L1": dict(cond="B_plastic", seed=20261102, side="left"),
    "B_R2": dict(cond="B_plastic", seed=20261103, side="right"),
    "B_L2": dict(cond="B_plastic", seed=20261104, side="left"),
    "B_R3": dict(cond="B_plastic", seed=20261105, side="right"),
    "B_L3": dict(cond="B_plastic", seed=20261106, side="left"),
    "A_R": dict(cond="A_no_plastic", seed=20261107, side="right"),
    "A_L": dict(cond="A_no_plastic", seed=20261108, side="left"),
    "C_R": dict(cond="C_shuffled_da", seed=20261109, side="right"),
    "D_R": dict(cond="D_shuffled_kcmbon", seed=20261110, side="right"),
    "F_R": dict(cond="F_untrained", seed=20261111, side="right"),
}


def frozen():
    return json.loads((G4 / "FROZEN_PARAMS.json").read_text())


# --------------------------------------------------------------------------
# battery expansion (prereg section 7)
# --------------------------------------------------------------------------
def battery_plan(reward_side):
    other = "left" if reward_side == "right" else "right"
    plan = []
    plan += [dict(phase="baseline", kind="pref") for _ in range(6)]
    for b in range(4):
        plan += [dict(phase="acquisition", kind="pair", side=reward_side)
                 for _ in range(4)]
        plan += [dict(phase="acquisition", kind="pref") for _ in range(2)]
        plan += [dict(phase="acquisition", kind="routea")]
    plan += [dict(phase="retention", kind="filler") for _ in range(8)]
    plan += [dict(phase="retention", kind="pref") for _ in range(3)]
    plan += [dict(phase="extinction", kind="pref") for _ in range(12)]
    for b in range(4):
        plan += [dict(phase="reversal", kind="pair", side=other)
                 for _ in range(4)]
        plan += [dict(phase="reversal", kind="pref") for _ in range(2)]
        plan += [dict(phase="reversal", kind="routea")]
    plan += [dict(phase="context", kind="pref", ctx=0.8) for _ in range(3)]
    plan += [dict(phase="context", kind="pref", ctx=1.2) for _ in range(3)]
    plan += [dict(phase="distractor", kind="pref", loom=True)
             for _ in range(3)]
    return plan


# --------------------------------------------------------------------------
# one session (subprocess)
# --------------------------------------------------------------------------
def run_session(label):
    import g4v2_core as core
    spec = SESSIONS[label]
    cond, master_seed, reward_side = (spec["cond"], spec["seed"],
                                      spec["side"])
    fz = frozen()
    n_cs = fz["n_cs_per_side"]
    rate_l = fz["cs_rate_left_hz"]
    rate_r = fz["cs_rate_right_hz"]
    out_dir = RESULTS / label
    out_dir.mkdir(parents=True, exist_ok=True)

    plasticity = cond in ("B_plastic", "C_shuffled_da", "D_shuffled_kcmbon")
    us_on = cond in ("B_plastic", "A_no_plastic", "C_shuffled_da",
                     "D_shuffled_kcmbon")
    t0 = time.perf_counter()
    print(f"[session {label}] building V2Runtime (n_cs={n_cs}, "
          f"plasticity={plasticity})...", flush=True)
    rt = core.V2Runtime(master_seed=master_seed, n_cs=n_cs,
                        plasticity=plasticity)
    print(f"[session {label}] build done in "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    plan = battery_plan(reward_side)
    trials = []
    prev_gate = (0.0, 0.0)
    (out_dir / "trials.jsonl").unlink(missing_ok=True)
    for idx, tr in enumerate(plan):
        seed_t = master_seed + idx
        kind = tr["kind"]
        if kind == "pref":
            rec = rt.pref_trial(
                seed_t, rate_l, rate_r,
                ctx_scale=tr.get("ctx", 1.0),
                loom_precursor=tr.get("loom", False),
                phase=tr["phase"], trial_idx=idx)
            # per-trial forgetting (plasticity sessions only)
            if plasticity:
                rt.pl.relax(1)
        elif kind == "pair":
            lag = prev_gate if cond == "C_shuffled_da" else None
            perm = (master_seed + 5000 + idx
                    if cond == "D_shuffled_kcmbon" else None)
            rec = rt.pairing_trial(
                seed_t, tr["side"], rate_l if tr["side"] == "left"
                else rate_r, trial_idx=idx, us_on=us_on,
                plastic_update=plasticity, lag_gate=lag,
                perm_seed=perm)
            prev_gate = (rec["pam_us_hz"], rec["ppl1_us_hz"])
        elif kind == "filler":
            rec = rt.filler_trial(seed_t, trial_idx=idx)
        elif kind == "routea":
            rec = rt.routea_trial(seed_t, rate_l, rate_r, trial_idx=idx)
            if plasticity:
                rt.pl.relax(1)      # per-trial forgetting clock
        rec["phase"], rec["trial"] = tr["phase"], idx
        rec["battery_kind"] = kind
        trials.append(rec)
        with (out_dir / "trials.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        # weight snapshots at phase boundaries (A2 contrast evidence)
        if plasticity:
            if tr["phase"] == "acquisition" and \
                    idx + 1 < len(plan) and \
                    plan[idx + 1]["phase"] != "acquisition":
                np.savez_compressed(out_dir / "weights_acq.npz",
                                    scale=rt.pl.scale)
            if tr["phase"] == "extinction" and \
                    idx + 1 < len(plan) and \
                    plan[idx + 1]["phase"] != "extinction":
                np.savez_compressed(out_dir / "weights_ext.npz",
                                    scale=rt.pl.scale)
        vb = rec.get("V", rec.get("valence_bias_end"))
        print(f"[{label} {idx:2d}] {tr['phase']:<11s} {kind:<6s} "
              f"choice_b={rec.get('choice_b', rec.get('choice'))} V={vb} "
              f"pam={rec.get('pam_us_hz', 0)}", flush=True)

    wall = time.perf_counter() - t0
    summ = summarize(trials, label, cond, reward_side)
    summ.update(session_wall_s=round(wall, 1),
                peak_rss_mb=round(bl.peak_rss_kb() / 1024, 1),
                cpu_user_s=round(os.times()[0], 1),
                bio_s_total=round(sum(t.get("bio_s", 0) for t in trials), 1),
                mean_pref_wall_s=round(float(np.mean(
                    [t["wall_s"] for t in trials
                     if t.get("kind") == "pref"])) if any(
                    t.get("kind") == "pref" for t in trials) else 0, 1),
                n_trials=len(trials),
                final_weight_stats=rt.pl.stats(),
                frozen_params=dict(n_cs=n_cs, rate_l=rate_l, rate_r=rate_r),
                plan="GATE4_V2_PREREG.md section 7")
    (out_dir / "session.json").write_text(json.dumps(summ, indent=1))
    np.savez_compressed(out_dir / "weights.npz", scale=rt.pl.scale)
    print(f"[session {label}] done in {wall:.0f}s -> {out_dir}", flush=True)
    return summ


def summarize(trials, label, cond, reward_side):
    """Session-level metrics (pre-registered; preference on route B)."""
    out = dict(label=label, condition=cond, reward_side=reward_side,
               n_trials=len(trials))
    for phase in sorted({t["phase"] for t in trials}):
        xs = [t for t in trials if t["phase"] == phase]
        prefs = [t for t in xs if t.get("kind") == "pref"]
        dec = [t for t in prefs if t.get("choice_b") in ("left", "right")]
        blk = dict(
            n=len(xs), n_pref=len(prefs),
            p_left_b=(round(sum(1 for t in dec
                                if t["choice_b"] == "left") / len(dec), 3)
                      if dec else None),
            p_rewarded_b=(round(sum(1 for t in dec if t["choice_b"]
                                    == reward_side) / len(dec), 3)
                          if dec else None),
            mean_V=round(float(np.mean([t["V"] for t in prefs])), 3)
            if prefs else None,
            mean_C=round(float(np.mean([t["C"] for t in prefs])), 3)
            if prefs else None,
            p_left_c=(round(sum(1 for t in prefs if t.get("choice_c")
                                == "left") / max(1, sum(
                                    1 for t in prefs
                                    if t.get("choice_c") in ("left",
                                                             "right"))), 3)
                      if any(t.get("choice_c") in ("left", "right")
                             for t in prefs) else None),
        )
        pairs = [t for t in xs if t.get("kind") == "pair"]
        if pairs:
            blk.update(n_pair=len(pairs),
                       us_delivered=sum(1 for t in pairs
                                        if t.get("us_delivered")),
                       mean_pam_us_hz=round(float(np.mean(
                           [t["pam_us_hz"] for t in pairs])), 1))
        ra = [t for t in xs if t.get("kind") == "routea"]
        if ra:
            dec_ra = [t for t in ra if t.get("choice")]
            blk.update(routea_n=len(ra),
                       routea_p_left=(round(sum(
                           1 for t in dec_ra if t["choice"] == "left")
                           / len(dec_ra), 3) if dec_ra else None))
        out[f"phase_{phase}"] = blk
    return out


# --------------------------------------------------------------------------
# E_rl baseline (NO brain; explicitly non-biological)
# --------------------------------------------------------------------------
def run_rl(out_dir=RESULTS / "E_rl"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261112)
    for label in ("B_R1", "B_L1"):
        spec = SESSIONS[label]
        plan = battery_plan(spec["side"])
        q = {"left": 0.5, "right": 0.5}
        forget = np.exp(-1.0 / 50.0)
        eps, alpha = 0.1, 0.3
        trials = []
        for idx, tr in enumerate(plan):
            if tr["kind"] == "pair":
                q[tr["side"]] += alpha * (1.0 - q[tr["side"]])
                choice = tr["side"]
            elif tr["kind"] == "pref":
                choice = (str(rng.choice(["left", "right"]))
                          if rng.random() < eps else max(q, key=q.get))
            else:
                choice = None
            for k in q:
                q[k] = 0.5 + (q[k] - 0.5) * forget
            trials.append(dict(trial=idx, phase=tr["phase"],
                               battery_kind=tr["kind"], choice=choice,
                               q_left=q["left"], q_right=q["right"]))
        (out_dir / f"{label}_trials.jsonl").write_text(
            "\n".join(json.dumps(t) for t in trials))
        summ = _rl_summary(trials, label, spec["side"])
        (out_dir / f"{label}_session.json").write_text(
            json.dumps(summ, indent=1))
        print(f"[rl {label}] acq P(rewarded)="
              f"{summ['phase_acquisition']['p_rewarded_b']} baseline="
              f"{summ['phase_baseline']['p_rewarded_b']}")
    return True


def _rl_summary(trials, label, reward_side):
    out = dict(label=label, condition="E_rl", reward_side=reward_side,
               note="conventional epsilon-greedy Q-learner; explicitly "
                    "NON-biological comparison (no brain)")
    for phase in sorted({t["phase"] for t in trials}):
        prefs = [t for t in trials if t["phase"] == phase
                 and t["battery_kind"] == "pref"]
        if prefs:
            out[f"phase_{phase}"] = dict(
                p_rewarded_b=round(sum(1 for t in prefs
                                       if t["choice"] == reward_side)
                                   / len(prefs), 3),
                p_left_b=round(sum(1 for t in prefs
                                   if t["choice"] == "left")
                               / len(prefs), 3))
    return out


# --------------------------------------------------------------------------
# orchestrator (detached; drives sessions one at a time)
# --------------------------------------------------------------------------
def _spawn_session(label, retry=1):
    log = RESULTS / f"_{label}.log"
    cmd = [PY, str(Path(__file__).resolve()), "run", "--session", label]
    t0 = time.perf_counter()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0 and retry > 0:
        print(f"[matrix] {label} FAILED (see {log}); retrying once",
              flush=True)
        return _spawn_session(label, retry=retry - 1)
    wall = time.perf_counter() - t0
    print(f"[matrix] {label} rc={proc.returncode} in {wall:.0f}s",
          flush=True)
    return proc.returncode


def _git_commit_push(msg):
    """Best-effort commit+push of new results (sandbox-reset safety;
    the mandate requires all results pushed before session end)."""
    import subprocess as sp
    repo = BRAIN.parent
    try:
        sp.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        sp.run(["git", "commit", "-q", "-m", msg], cwd=repo,
               capture_output=True)
        sp.run(["git", "push", "origin", "main"], cwd=repo,
               capture_output=True, timeout=120)
        print(f"[matrix] pushed: {msg}", flush=True)
    except Exception as e:            # pragma: no cover - never fatal
        print(f"[matrix] push failed ({e}); continuing", flush=True)


def run_matrix(only=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_rl()                              # E first: instant, no brain
    _git_commit_push("GATE4 v2: E_rl baseline (non-biological comparison)")
    order = list(SESSIONS) if not only else [only]
    for label in order:
        if label == "E_rl":
            continue
        out = RESULTS / label / "session.json"
        if out.exists():
            print(f"[matrix] {label} already complete - skip", flush=True)
            continue
        _spawn_session(label)
        _git_commit_push(
            f"GATE4 v2 matrix: session {label} complete "
            f"({SESSIONS[label]['cond']}, reward "
            f"{SESSIONS[label]['side']})")
    print("[matrix] ALL SESSIONS COMPLETE", flush=True)


def launch_daemon(only=None):
    log = RESULTS / "_matrix_daemon.log"
    RESULTS.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(Path(__file__).resolve()), "_matrix_child"]
    if only:
        cmd += ["--only", ",".join(only)]
    with open(log, "a", buffering=1) as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=str(BRAIN))
    print(f"[launch] matrix orchestrator pid {proc.pid}; log {log}")


def status():
    print(f"{'session':<8} {'state':<10} notes")
    for label in SESSIONS:
        sj = RESULTS / label / "session.json"
        tj = RESULTS / label / "trials.jsonl"
        if sj.exists():
            print(f"{label:<8} DONE")
        elif tj.exists():
            n = len(tj.read_text().splitlines())
            print(f"{label:<8} PARTIAL  {n} trials logged")
        else:
            print(f"{label:<8} PENDING")
    rl = RESULTS / "E_rl" / "B_R1_session.json"
    print(f"{'E_rl':<8} {'DONE' if rl.exists() else 'PENDING'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "launch", "rl", "status",
                                    "_matrix_child"])
    ap.add_argument("--session", type=str, default="")
    ap.add_argument("--only", type=str, default="")
    a = ap.parse_args()
    if a.cmd == "launch":
        launch_daemon(only=a.only.split(",") if a.only else None)
    elif a.cmd == "_matrix_child":
        run_matrix(only=a.only.split(",") if a.only else None)
    elif a.cmd == "run":
        run_session(a.session)
    elif a.cmd == "rl":
        run_rl()
    elif a.cmd == "status":
        status()


if __name__ == "__main__":
    main()
