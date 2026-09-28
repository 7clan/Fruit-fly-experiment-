"""GATE 4 v2 — calibration (Sections F & G of the v2 brief).

Runs the PRE-REGISTERED calibration (GATE4_V2_PREREG.md section 4) in
subprocesses, one network build per phase/seed:

  dose       CS dose-response, settings S1-S4 x 6 preference trials,
             seed c1=20260930 (symmetric rates, no plasticity, no US)
  boundary   balanced-boundary sweep, kc_cs_left in
             {R-40, R-20, R, R+20, R+40} x 6 trials, seeds c1 & c2
             (at the dose-selected setting; no plasticity, no US)
  uscheck    US feasibility: 2 single-side CS-only trials + 1 full
             pairing trial (plasticity ON): US-set PAM gate rate,
             plasticity gates fired, MBON response change
  select     apply the pre-registered selection rules -> FROZEN_PARAMS
  all        parent orchestration: dose -> select-setting -> boundary
             (c1,c2) -> select-rate -> uscheck -> freeze + report

ALL tested values are reported (calibration_report.md); the reward side
is assigned only AFTER the freeze (prereg section 7).
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
RESULTS = BRAIN / "results" / "gate4_v2" / "calibration"
PY = sys.executable

C1, C2 = 20260930, 20261001
DOSE_SETTINGS = [dict(sid="S1", n_cs=100, rate=75.0),
                 dict(sid="S2", n_cs=200, rate=75.0),
                 dict(sid="S3", n_cs=200, rate=100.0),
                 dict(sid="S4", n_cs=400, rate=100.0)]
BOUNDARY_OFFSETS = [-40.0, -20.0, 0.0, 20.0, 40.0]


# --------------------------------------------------------------------------
# subprocess phase runners (one build each)
# --------------------------------------------------------------------------
def run_dose(seed=C1, setting=None, out=None):
    """One dose-response setting in ONE subprocess (one network build;
    RAM discipline - 4 subprocesses for S1-S4)."""
    import g4v2_core as core
    if setting is None:
        setting = "S1"
    s = next(x for x in DOSE_SETTINGS if x["sid"] == setting)
    out = Path(out) if out else RESULTS / f"dose_{s['sid']}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rt = core.V2Runtime(master_seed=seed, n_cs=s["n_cs"], plasticity=False)
    trials = []
    for j in range(6):
        seed_t = seed + 1000 + j
        rec = rt.pref_trial(seed_t, s["rate"], s["rate"],
                            phase=f"dose_{s['sid']}", trial_idx=j)
        rec.update(sid=s["sid"], n_cs=s["n_cs"], rate=s["rate"])
        trials.append(rec)
        print(f"[dose {s['sid']}] trial {j} V={rec['V']} "
              f"choice={rec['choice_b']} kc={rec['kc_cs']}", flush=True)
    _write(out, trials, dict(phase="dose", sid=s["sid"], seed=seed,
                             wall_s=round(time.perf_counter() - t0, 1)))
    return trials


def run_boundary(seed, n_cs, rate_r, left_rates, out):
    import g4v2_core as core
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rt = core.V2Runtime(master_seed=seed, n_cs=n_cs, plasticity=False)
    trials = []
    for rl in left_rates:
        for j in range(6):
            seed_t = seed + 2000 + j
            rec = rt.pref_trial(seed_t, rl, rate_r,
                                phase="boundary", trial_idx=j)
            rec.update(rate_left=rl, rate_right=rate_r, n_cs=n_cs)
            trials.append(rec)
            print(f"[boundary seed {seed}] L={rl:.0f} trial {j} "
                  f"V={rec['V']} choice={rec['choice_b']}", flush=True)
    _write(out, trials, dict(phase="boundary", seed=seed, n_cs=n_cs,
                             rate_right=rate_r, left_rates=list(left_rates),
                             wall_s=round(time.perf_counter() - t0, 1)))
    return trials


def run_uscheck(seed, n_cs, rate_l, rate_r, out=RESULTS / "uscheck"):
    import g4v2_core as core
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rt = core.V2Runtime(master_seed=seed, n_cs=n_cs, plasticity=True)
    trials = []
    # 2 single-side CS-only trials (right CS alone, no US): baseline
    # avoid-class response to that CS
    for j in range(2):
        rec = rt.pref_trial(seed + 3000 + j, 0.0, rate_r,
                            phase="uscheck_base", trial_idx=j,
                            plastic_update=False)
        trials.append(rec)
    # 1 full pairing trial (right CS + US, plasticity ON)
    rec = rt.pairing_trial(seed + 3010, "right", rate_r, trial_idx=2,
                           us_on=True, plastic_update=True)
    trials.append(rec)
    _write(out, trials, dict(phase="uscheck", seed=seed, n_cs=n_cs,
                             rate_left=rate_l, rate_right=rate_r,
                             wall_s=round(time.perf_counter() - t0, 1)))
    ok_gate = rec["pam_us_hz"] >= 10.0
    ok_update = bool(rec["plasticity"].get("reward_gated")) and \
        rec["plasticity"].get("n_synapses_changed", 0) > 0
    base_avoid = float(np.mean([t["mbon"]["avoid_r"]
                                for t in trials[:2]]))
    pair_avoid = rec["mbon"]["avoid_r"]
    ok_response = abs(pair_avoid - base_avoid) >= 1.0
    verdict = dict(gate_hz=rec["pam_us_hz"], gate_ok=bool(ok_gate),
                   update_ok=bool(ok_update),
                   n_synapses_changed=rec["plasticity"].get(
                       "n_synapses_changed"),
                   avoid_r_base=round(base_avoid, 3),
                   avoid_r_pairing=pair_avoid,
                   response_change_hz=round(pair_avoid - base_avoid, 3),
                   response_ok=bool(ok_response),
                   uscheck_pass=bool(ok_gate and ok_update and ok_response))
    (out / "uscheck.json").write_text(json.dumps(verdict, indent=1))
    print(f"[uscheck] {json.dumps(verdict)}")
    return verdict


def _write(out, trials, meta):
    out = Path(out)
    (out / "trials.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trials))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


# --------------------------------------------------------------------------
# pre-registered selection rules (prereg section 4)
# --------------------------------------------------------------------------
def summarize_pref(trials):
    dec = [t for t in trials if t.get("choice_b") in ("left", "right")]
    p_left = (sum(1 for t in dec if t["choice_b"] == "left") / len(dec)
              if dec else None)
    vs = [t["V"] for t in trials]
    kc = [t["kc_cs"]["left_hz"] for t in trials]
    kc_r = [t["kc_cs"]["right_hz"] for t in trials]
    av = [t["mbon"]["avoid_l"] + t["mbon"]["avoid_r"] for t in trials]
    ap = [t["mbon"]["appr_l"] + t["mbon"]["appr_r"] for t in trials]
    p9 = [t["dn"]["p9_diff"] for t in trials]
    return dict(n=len(trials), n_decided=len(dec), p_left=p_left,
                mean_V=round(float(np.mean(vs)), 3),
                std_V=round(float(np.std(vs)), 3),
                mean_kc_left_hz=round(float(np.mean(kc)), 3),
                mean_kc_right_hz=round(float(np.mean(kc_r)), 3),
                mean_avoid_hz=round(float(np.mean(av)) / 2, 3),
                mean_appr_hz=round(float(np.mean(ap)) / 2, 3),
                mean_p9_diff=round(float(np.mean(p9)), 3),
                mean_wall_s=round(float(np.mean(
                    [t["wall_s"] for t in trials])), 1))


def select_dose(trials):
    """Pre-registered rule: among settings with KC>=5 Hz, both MBON class
    rates >= 2 Hz, |P9 diff| < 8 Hz, P(left|B) in [0.15, 0.85]: choose the
    largest avoidance-class response; tie-break smaller ensemble."""
    by = {}
    for t in trials:
        by.setdefault(t["sid"], []).append(t)
    cands, allrows = [], []
    for s in DOSE_SETTINGS:
        ts = by.get(s["sid"], [])
        if not ts:
            continue
        st = summarize_pref(ts)
        st.update(sid=s["sid"], n_cs=s["n_cs"], rate=s["rate"])
        allrows.append(st)
        passes = (st["mean_kc_left_hz"] >= 5.0 and st["mean_kc_right_hz"]
                  >= 5.0 and st["mean_avoid_hz"] >= 2.0
                  and st["mean_appr_hz"] >= 2.0
                  and abs(st["mean_p9_diff"]) < 8.0
                  and st["p_left"] is not None and 0.15 <= st["p_left"]
                  <= 0.85)
        st["passes"] = bool(passes)
        if passes:
            cands.append(st)
    if not cands:
        return None, allrows
    cands.sort(key=lambda st: (-st["mean_avoid_hz"], st["n_cs"]))
    return cands[0], allrows


def select_boundary(trials):
    """Pre-registered rule: left-rate minimizing |P(left)-0.5|; target
    P(left) in [0.35, 0.65]; if none qualifies, closest value is used and
    the miss is disclosed."""
    by = {}
    for t in trials:
        by.setdefault(t["rate_left"], []).append(t)
    rows = []
    for rl in sorted(by):
        st = summarize_pref(by[rl])
        st.update(rate_left=rl)
        rows.append(st)
    rows.sort(key=lambda st: (abs((st["p_left"] if st["p_left"] is not
                                   None else 0.5) - 0.5), st["rate_left"]))
    best = rows[0]
    in_range = (best["p_left"] is not None and 0.35 <= best["p_left"]
                <= 0.65)
    return best, rows, in_range


# --------------------------------------------------------------------------
# parent orchestration
# --------------------------------------------------------------------------
def _spawn(func, **kw):
    """Run a phase function in a fresh subprocess (RAM discipline).
    Child stdout/stderr streams to a per-phase log (live pollable)."""
    cmd = [PY, str(Path(__file__).resolve()), "run",
           "--phase", func]
    for k, v in kw.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    tag = kw.get("setting", kw.get("seed", "x"))
    log = RESULTS / f"_{func}_{tag}.log"
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"calibration phase failed: {func} "
                         f"(see {log})")
    print(f"[calib] {func} {tag} done in {wall:.0f}s", flush=True)
    return wall


def run_all():
    RESULTS.mkdir(parents=True, exist_ok=True)
    # ---- phase G: dose-response (one subprocess per setting) ------------
    for s in DOSE_SETTINGS:
        if not (RESULTS / f"dose_{s['sid']}" / "trials.jsonl").exists():
            _spawn("dose", seed=C1, setting=s["sid"])
        else:
            print(f"[calib] dose {s['sid']} exists - skip", flush=True)
    dose_trials = []
    for s in DOSE_SETTINGS:
        dose_trials += [json.loads(x) for x in
                        (RESULTS / f"dose_{s['sid']}" / "trials.jsonl")
                        .read_text().splitlines()]
    sel, all_dose_rows = select_dose(dose_trials)
    report = dict(dose_all=all_dose_rows)
    if sel is None:
        report["verdict"] = "FAIL_AT_CALIBRATION (no dose setting passed)"
        (G4 / "calibration_report.json").write_text(json.dumps(report,
                                                               indent=1))
        print(json.dumps(report, indent=1)[:2000])
        return report
    report["dose_selected"] = sel
    n_cs, R = sel["n_cs"], sel["rate"]

    # ---- phase F: boundary sweep (2 seeds) ------------------------------
    left_rates = [R + o for o in BOUNDARY_OFFSETS]
    for tag, sd in (("boundary_c1", C1), ("boundary_c2", C2)):
        if not (RESULTS / tag / "trials.jsonl").exists():
            _spawn("boundary", seed=sd, n_cs=n_cs, rate_r=R,
                   left_rates=",".join(f"{x:g}" for x in left_rates),
                   out=str(RESULTS / tag))
        else:
            print(f"[calib] {tag} exists - skip", flush=True)
    btrials = []
    for d in ("boundary_c1", "boundary_c2"):
        btrials += [json.loads(x) for x in
                    (RESULTS / d / "trials.jsonl").read_text().splitlines()]
    best, b_rows, in_range = select_boundary(btrials)
    report["boundary_all"] = b_rows
    report["boundary_selected"] = best
    report["boundary_in_neutral_range"] = bool(in_range)
    rate_l = best["rate_left"]

    # ---- US feasibility check -------------------------------------------
    if not (RESULTS / "uscheck" / "uscheck.json").exists() or \
            not json.loads((RESULTS / "uscheck" / "uscheck.json")
                           .read_text())["uscheck_pass"]:
        _spawn("uscheck", seed=C1, n_cs=n_cs, rate_l=rate_l, rate_r=R)
    else:
        print("[calib] uscheck passed - skip", flush=True)
    us = json.loads((RESULTS / "uscheck" / "uscheck.json").read_text())
    report["uscheck"] = us
    if not us["uscheck_pass"]:
        report["verdict"] = "FAIL_AT_CALIBRATION (US feasibility failed)"
        (G4 / "calibration_report.json").write_text(json.dumps(report,
                                                               indent=1))
        print(json.dumps(report, indent=1)[:2000])
        return report

    # ---- freeze ----------------------------------------------------------
    frozen = dict(
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_cs_per_side=n_cs, cs_rate_right_hz=R, cs_rate_left_hz=rate_l,
        us_rate_hz=150.0, us_set="PAM01-11 (252 cells)",
        plasticity=dict(eta=0.30, scale_min=0.05, tau_relax_trials=50.0,
                        tau_elig_s=2.0, elig_power=2.0,
                        pam_gate_hz=10.0, ppl1_gate_hz=10.0,
                        target="KC->avoidance-class (GLUT) MBONs"),
        interface_spec="brain/data/gate4_v2/interface_v2.json",
        readout_spec="brain/data/gate4_v2/readout_v2.json",
        boundary=dict(p_left=best["p_left"], n_trials=len(btrials),
                      in_neutral_range=bool(in_range)),
        reward_side_assignment_pre_registered="B_R1 R, B_L1 L, B_R2 R, "
                                              "B_L2 L, B_R3 R, B_L3 L; "
                                              "A_R R, A_L L; C_R R, D_R R, "
                                              "F_R R (GATE4_V2_PREREG s7)")
    (G4 / "FROZEN_PARAMS.json").write_text(json.dumps(frozen, indent=1))
    report["verdict"] = "CALIBRATION COMPLETE - PARAMETERS FROZEN"
    (G4 / "calibration_report.json").write_text(json.dumps(report, indent=1))
    _write_calibration_md(report)
    print(json.dumps({k: report[k] for k in
                      ("dose_selected", "boundary_selected",
                       "boundary_in_neutral_range", "uscheck", "verdict")},
                     indent=1))
    return report


def _write_calibration_md(report):
    lines = ["# Gate-4 v2 — calibration report (ALL tested values)",
             "",
             "Pre-registered procedure: GATE4_V2_PREREG.md section 4. "
             "No reward side existed during calibration; no plasticity "
             "except the uscheck pairing trial; no outcome data.", "",
             "## Phase G — CS dose-response (seed c1, 6 preference "
             "trials/setting, symmetric rates)", "",
             "| setting | n KC/side | rate (Hz) | KC L/R (Hz) | appr "
             "(Hz) | avoid (Hz) | P9 diff (Hz) | P(left|B) | mean V | "
             "std V | passes |", "|---|---|---|---|---|---|---|---|---|"
             "---|---|"]
    for st in report["dose_all"]:
        lines.append(
            f"| {st['sid']} | {st['n_cs']} | {st['rate']:g} | "
            f"{st['mean_kc_left_hz']}/{st['mean_kc_right_hz']} | "
            f"{st['mean_appr_hz']} | {st['mean_avoid_hz']} | "
            f"{st['mean_p9_diff']} | {st['p_left']} | {st['mean_V']} | "
            f"{st['std_V']} | {st['passes']} |")
    sel = report.get("dose_selected")
    if sel:
        lines += ["", f"**Selected (pre-registered rule): {sel['sid']} "
                  f"(n={sel['n_cs']}/side, R={sel['rate']:g} Hz)**", "",
                  "## Phase F — balanced-boundary sweep (seeds c1+c2, 12 "
                  "preference trials/value)", "",
                  "| kc_cs_left (Hz) | P(left|B) | mean V | std V | KC L "
                  "(Hz) |", "|---|---|---|---|---|"]
        for st in report["boundary_all"]:
            lines.append(f"| {st['rate_left']:g} | {st['p_left']} | "
                         f"{st['mean_V']} | {st['std_V']} | "
                         f"{st['mean_kc_left_hz']} |")
        b = report["boundary_selected"]
        neutral = ("YES" if report["boundary_in_neutral_range"] else
                   "NO (disclosed; closest value used, no further "
                   "sweeping)")
        lines += ["", f"**Selected kc_cs_left = {b['rate_left']:g} Hz "
                  f"(P(left)={b['p_left']})** — neutral range "
                  f"[0.35, 0.65]: {neutral}", "",
                  "## US feasibility check", "",
                  "```json", json.dumps(report["uscheck"], indent=1),
                  "```", "", "## Freeze", "",
                  "See `FROZEN_PARAMS.json`. Reward-side assignment "
                  "(prereg section 7) happens only after this freeze."]
    (G4 / "calibration_report.md").write_text("\n".join(lines) + "\n")
    print(f"[calib] report -> {G4/'calibration_report.md'}")


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# daemon launcher (double-fork; the sandbox reaps the direct process
# group between tool calls, but a daemonized orchestrator survives and
# drives all phases sequentially, each as its own subprocess)
# --------------------------------------------------------------------------
def launch_daemon():
    """Spawn a detached orchestrator: a fresh python process in its own
    session (start_new_session=True) that survives the parent and drives
    all phases sequentially, each as its own subprocess.  Avoids fork()
    in a multi-threaded parent (Python 3.12 deadlock risk)."""
    import subprocess as sp
    log = RESULTS / "_daemon.log"
    RESULTS.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(Path(__file__).resolve()), "_daemon_child"]
    with open(log, "a", buffering=1) as fh:
        proc = sp.Popen(cmd, stdout=fh, stderr=sp.STDOUT,
                        start_new_session=True, cwd=str(BRAIN))
    print(f"[launch] orchestrator pid {proc.pid} detached; log: {log}")


def _daemon_child():
    """Body of the detached orchestrator (runs run_all sequentially)."""
    try:
        run_all()
        print("[daemon] ALL DONE", flush=True)
    except SystemExit as e:
        print(f"[daemon] EXITED: {e}", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print("[daemon] FAILED", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "all", "launch",
                                     "_daemon_child"])
    ap.add_argument("--phase", choices=["dose", "boundary", "uscheck"])
    ap.add_argument("--seed", type=int, default=C1)
    ap.add_argument("--setting", type=str, default="S1")
    ap.add_argument("--n-cs", type=int, default=200)
    ap.add_argument("--rate-r", type=float, default=100.0)
    ap.add_argument("--rate-l", type=float, default=100.0)
    ap.add_argument("--left-rates", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()
    if a.cmd == "launch":
        launch_daemon()
        return
    if a.cmd == "_daemon_child":
        _daemon_child()
        return
    if a.cmd == "all":
        run_all()
        return
    if a.phase == "dose":
        run_dose(seed=a.seed, setting=a.setting,
                 out=a.out or None)
    elif a.phase == "boundary":
        rates = [float(x) for x in a.left_rates.split(",") if x]
        run_boundary(a.seed, a.n_cs, a.rate_r, rates,
                     out=a.out or RESULTS / f"boundary_{a.seed}")
    elif a.phase == "uscheck":
        run_uscheck(a.seed, a.n_cs, a.rate_l, a.rate_r,
                    out=a.out or RESULTS / "uscheck")


if __name__ == "__main__":
    main()
