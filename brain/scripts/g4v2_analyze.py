"""GATE 4 v2 — analysis + verdicts (GATE4_V2_PREREG.md section 8).

Computes the TWO INDEPENDENT pre-registered verdicts from the session
matrix (fail-closed):

  G4A NEURAL ASSOCIATIVE LEARNING   (A1 eligibility, A2 cue-specific
     synaptic change, A4 learned valence signal; A3 descriptive)
  G4B BEHAVIORAL EXPRESSION         (B1 pooled route-B preference shift,
     B2 both reward directions, B3 control causality)

Everything else (extinction decline, reversal flip, context/distractor
robustness, route A/C, E_rl) is DESCRIPTIVE and labelled as such.

Usage:  python g4v2_analyze.py            -> verdicts.json + report data
"""
from __future__ import annotations

import json
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
OUT = RESULTS / "analysis"

B_LABELS = ["B_R1", "B_L1", "B_R2", "B_L2", "B_R3", "B_L3"]
NBOOT = 10000
RNG_SEED = 20261201


def load_trials(label):
    p = RESULTS / label / "trials.jsonl"
    if not p.exists():
        return None
    return [json.loads(x) for x in p.read_text().splitlines()]


def prefs(trials, phase):
    return [t for t in trials if t["phase"] == phase
            and t.get("battery_kind") == "pref"]


def decided(ts):
    return [t for t in ts if t.get("choice_b") in ("left", "right")]


def p_rewarded(ts, side):
    d = decided(ts)
    return (sum(1 for t in d if t["choice_b"] == side) / len(d)
            if d else float("nan"))


def v_norm(trials, phase, side):
    """V sign-normalized so positive = toward the rewarded side."""
    sgn = 1.0 if side == "left" else -1.0
    return [sgn * t["V"] for t in prefs(trials, phase)]


def boot_ci_mean(values, weights=None, n=NBOOT, seed=RNG_SEED):
    """Bootstrap CI of a mean. values: list of scalars OR list of lists
    (hierarchical: resample outer with replacement, pool inner)."""
    rng = np.random.default_rng(seed)
    values = [np.atleast_1d(v) for v in values]
    flat = np.concatenate(values)
    outer = len(values)
    means = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, outer, size=outer)
        pool = np.concatenate([values[j] for j in pick]) if outer > 1 \
            else flat
        means[i] = pool.mean() if len(pool) else np.nan
    obs = flat.mean()
    return dict(observed=round(float(obs), 4),
                ci95=[round(float(np.percentile(means, 2.5)), 4),
                      round(float(np.percentile(means, 97.5)), 4)],
                n_outer=outer, n_total=int(len(flat)))


def boot_ci_diff_below(valueSets_a, valueSets_b, n=NBOOT, seed=RNG_SEED):
    """P(a_mean - b_mean < 0) and CI of the difference (hierarchical)."""
    rng = np.random.default_rng(seed + 1)
    a = [np.atleast_1d(v) for v in valueSets_a]
    b = [np.atleast_1d(v) for v in valueSets_b]
    diffs = np.empty(n)
    for i in range(n):
        pa = np.concatenate([a[j] for j in rng.integers(
            0, len(a), size=len(a))]) if len(a) > 1 else np.concatenate(a)
        pb = np.concatenate([b[j] for j in rng.integers(
            0, len(b), size=len(b))]) if len(b) > 1 else np.concatenate(b)
        diffs[i] = pa.mean() - pb.mean()
    return dict(ci95=[round(float(np.percentile(diffs, 2.5)), 4),
                      round(float(np.percentile(diffs, 97.5)), 4)],
                p_below=round(float(np.mean(diffs < 0)), 4))


# --------------------------------------------------------------------------
# A2: cue-specific synaptic change (needs the KC->MBON row map)
# --------------------------------------------------------------------------
def a2_contrast():
    import d11_learning as d11
    fz = json.loads((G4 / "FROZEN_PARAMS.json").read_text())
    n_cs = fz["n_cs_per_side"]
    iface = json.loads((G4 / "interface_v2.json").read_text())
    ens = {"left": set(iface["channels"]["kc_cs_left"]["ids"][:n_cs]),
           "right": set(iface["channels"]["kc_cs_right"]["ids"][:n_cs])}
    val = json.loads((G4 / "mbon_valence_corrected.json").read_text())
    avoid_ids = {int(k) for k, v in val["corrected_classes"].items()
                 if v == "avoidance"}

    cache = RESULTS / "_kc_mbon_rows.npz"
    if cache.exists():
        z = np.load(cache)
        rows, pre, post = z["rows"], z["pre"], z["post"]
    else:
        sc = d11.scan_kc_mbon_connections()
        rows, pre, post = sc["rows"], sc["pre"], sc["post"]
        np.savez(cache, rows=rows, pre=pre, post=post)
    # fly ids of pre (KC) and post (MBON) per row
    flyid2i, _ = bl.load_maps("783")
    i2fly = {v: k for k, v in flyid2i.items()}
    pre_fly = np.array([int(i2fly[i]) for i in pre])
    post_fly = np.array([int(i2fly[i]) for i in post])

    per_session = {}
    for label in B_LABELS:
        wf = RESULTS / label / "weights_acq.npz"
        if not wf.exists():
            continue
        side = json.loads((RESULTS / label / "session.json")
                          .read_text())["reward_side"]
        other = "left" if side == "right" else "right"
        scale = np.load(wf)["scale"]
        m_avoid = np.isin(post_fly, list(avoid_ids))
        m_p = np.isin(pre_fly, list(ens[side])) & m_avoid
        m_u = np.isin(pre_fly, list(ens[other])) & m_avoid
        per_session[label] = dict(
            reward_side=side,
            mean_paired=float(scale[m_p].mean()) if m_p.any() else None,
            mean_unpaired=float(scale[m_u].mean()) if m_u.any() else None,
            n_paired=int(m_p.sum()), n_unpaired=int(m_u.sum()),
            diff=float(scale[m_u].mean() - scale[m_p].mean())
            if (m_p.any() and m_u.any()) else None)
    return per_session


# --------------------------------------------------------------------------
def analyze():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               n_boot=NBOOT, rng_seed=RNG_SEED)

    B = {l: load_trials(l) for l in B_LABELS}
    CTRL = {l: load_trials(l) for l in ("A_R", "A_L", "C_R", "D_R", "F_R")}
    sides = {l: ("right" if "_R" in l else "left") for l in B}

    # ---------------- G4A ----------------------------------------------
    # A1 eligibility lateralization on pairing trials
    a1_fracs = []
    for l, tr in B.items():
        if tr is None:
            continue
        pairs = [t for t in tr if t.get("battery_kind") == "pair"
                 and t.get("us_on")]
        ok = 0
        for t in pairs:
            e_p = t.get("elig_left" if t["pair_side"] == "left"
                        else "elig_right")
            e_u = t.get("elig_right" if t["pair_side"] == "left"
                        else "elig_left")
            if e_p is not None and e_u is not None and e_p > 2 * max(
                    e_u, 1e-9):
                ok += 1
        a1_fracs.append(ok / len(pairs) if pairs else np.nan)
    a1_frac_mean = float(np.nanmean(a1_fracs)) if a1_fracs else 0.0
    out["A1_eligibility"] = dict(
        per_session=[round(f, 3) for f in a1_fracs],
        mean_fraction=round(a1_frac_mean, 4),
        gate=a1_frac_mean >= 0.90)

    # A2 cue-specific synaptic change
    a2 = a2_contrast()
    diffs = [s["diff"] for s in a2.values() if s["diff"] is not None]
    a2_ci = boot_ci_mean(diffs) if diffs else dict(observed=None,
                                                   ci95=[None, None])
    out["A2_weights"] = dict(per_session=a2,
                             pooled=boot_ci_mean(diffs) if diffs else None,
                             gate=bool(diffs and a2_ci["ci95"][0] > 0))

    # A3 (descriptive): avoidance-class response to paired CS
    a3 = {}
    for l, tr in B.items():
        if tr is None:
            continue
        side = sides[l]
        base = prefs(tr, "baseline")
        acq = prefs(tr, "acquisition")
        av_key = "avoid_l" if side == "left" else "avoid_r"
        a3[l] = dict(
            baseline=round(float(np.mean([t["mbon"][av_key]
                                          for t in base])), 3),
            acquisition=round(float(np.mean([t["mbon"][av_key]
                                             for t in acq])), 3))
        a3[l]["change"] = round(a3[l]["acquisition"] - a3[l]["baseline"], 3)
    out["A3_mbon_response_descriptive"] = a3

    # A4 learned valence signal shift (toward rewarded side)
    a4_sets = []
    for l, tr in B.items():
        if tr is None:
            continue
        base = v_norm(tr, "baseline", sides[l])
        acq = v_norm(tr, "acquisition", sides[l])
        a4_sets.append(np.array(acq) - np.mean(base))
    a4_ci = boot_ci_mean(a4_sets) if a4_sets else dict(observed=None,
                                                       ci95=[None, None])
    out["A4_valence_signal"] = dict(
        per_session={l: round(float(np.mean(s)), 4)
                     for l, s in zip([l for l in B if B[l] is not None],
                                     a4_sets)},
        pooled=a4_ci,
        gate=bool(a4_sets and a4_ci["ci95"][0] > 0))

    out["G4A_verdict"] = "PASS" if (out["A1_eligibility"]["gate"]
                                    and out["A2_weights"]["gate"]
                                    and out["A4_valence_signal"]["gate"]) \
        else "FAIL"

    # ---------------- G4B ----------------------------------------------
    def shift(conds):
        sets, per = [], {}
        for l, tr in conds.items():
            if tr is None:
                continue
            side = sides.get(l, "right" if "_R" in l else "left")
            base = p_rewarded(prefs(tr, "baseline"), side)
            acq = p_rewarded(prefs(tr, "acquisition"), side)
            per[l] = round(acq - base, 3)
            sets.append(np.array([acq - base]))
        return per, (boot_ci_mean(sets) if sets else
                     dict(observed=None, ci95=[None, None]))

    b_per, b_ci = shift(B)
    out["B1_preference_shift"] = dict(per_session=b_per, pooled=b_ci)
    b1_gate = bool(b_ci["observed"] is not None and b_ci["observed"] > 0
                   and b_ci["ci95"][0] > 0)

    r_mean = [v for l, v in b_per.items() if sides[l] == "right"]
    l_mean = [v for l, v in b_per.items() if sides[l] == "left"]
    out["B2_both_directions"] = dict(
        right_rewarded=dict(per=r_mean, mean=round(float(np.mean(r_mean)),
                                                   3) if r_mean else None),
        left_rewarded=dict(per=l_mean, mean=round(float(np.mean(l_mean)),
                                                  3) if l_mean else None))
    b2_gate = bool(r_mean and l_mean and np.mean(r_mean) > 0
                   and np.mean(l_mean) > 0)

    ctrl_out = {}
    b3_ok = True
    for l, tr in CTRL.items():
        if tr is None:
            b3_ok = False
            continue
        per, ci = shift({l: tr})
        ctrl_out[l] = dict(per_session=per, pooled=ci)
        # control must NOT show a significant positive shift like B
        if ci["observed"] is not None and ci["ci95"][0] > 0:
            b3_ok = False          # control reproduces the effect -> FAIL
    # C and D specifically: also pass if significantly below B
    for l in ("C_R", "D_R"):
        if l in ctrl_out and B:
            ci_b = b_ci
            ci_c = ctrl_out[l]["pooled"]
            if ci_c["observed"] is not None and ci_b["observed"] is not None:
                d = ci_b["observed"] - ci_c["observed"]
                ctrl_out[l]["vs_B_diff"] = round(d, 3)
    out["B3_controls"] = ctrl_out
    out["B3_gate"] = bool(b3_ok)

    out["G4B_verdict"] = "PASS" if (b1_gate and b2_gate and b3_ok) \
        else "FAIL"
    out["OVERALL_GATE4"] = "PASS" if out["G4B_verdict"] == "PASS" \
        else "FAIL"

    # ---------------- descriptive battery -------------------------------
    desc = {}
    for l, tr in B.items():
        if tr is None:
            continue
        side = sides[l]
        other = "left" if side == "right" else "right"
        ext = prefs(tr, "extinction")
        rev = prefs(tr, "reversal")
        ctx08 = [t for t in prefs(tr, "context") if t["ctx_scale"] == 0.8]
        ctx12 = [t for t in prefs(tr, "context") if t["ctx_scale"] == 1.2]
        dist = [t for t in prefs(tr, "distractor")]
        ra = [t for t in tr if t.get("battery_kind") == "routea"]
        desc[l] = dict(
            extinction=dict(
                first3=p_rewarded(ext[:3], side),
                last3=p_rewarded(ext[-3:], side)),
            reversal=dict(
                p_new_side=p_rewarded(rev, other)),
            context=dict(
                p_rewarded_x08=p_rewarded(ctx08, side),
                p_rewarded_x12=p_rewarded(ctx12, side)),
            distractor=dict(p_rewarded=p_rewarded(dist, side)),
            route_a=dict(
                n=len(ra),
                n_decided=sum(1 for t in ra if t.get("choice")),
                p_left=round(sum(1 for t in ra if t.get("choice") == "left")
                             / max(1, sum(1 for t in ra
                                          if t.get("choice"))), 3)),
        )
    out["descriptive_battery"] = desc

    # E_rl
    rl = {}
    for l in ("B_R1", "B_L1"):
        p = RESULTS / "E_rl" / f"{l}_session.json"
        if p.exists():
            rl[l] = json.loads(p.read_text())
    out["E_rl_descriptive"] = rl

    # ---------------- performance --------------------------------------
    perf = {}
    for label in list(B_LABELS) + list(CTRL):
        p = RESULTS / label / "session.json"
        if p.exists():
            s = json.loads(p.read_text())
            perf[label] = {k: s.get(k) for k in (
                "session_wall_s", "peak_rss_mb", "cpu_user_s",
                "bio_s_total", "mean_pref_wall_s", "n_trials")}
    out["performance"] = perf
    out["wall_s"] = round(time.perf_counter() - t0, 1)

    (OUT / "verdicts.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in (
        "A1_eligibility", "A2_weights", "A4_valence_signal",
        "G4A_verdict", "B1_preference_shift", "B2_both_directions",
        "B3_gate", "G4B_verdict", "OVERALL_GATE4")}, indent=1)[:4000])
    return out


if __name__ == "__main__":
    analyze()


# --------------------------------------------------------------------------
# report rendering (data sections for GATE4_V2_REPORT.md)
# --------------------------------------------------------------------------
def render_report():
    """Render the analysis into report-ready markdown sections."""
    v = json.loads((OUT / "verdicts.json").read_text())
    fz = json.loads((G4 / "FROZEN_PARAMS.json").read_text())
    L = []
    a = L.append
    a("## Verdicts (pre-registered criteria, fail-closed)\n")
    a(f"- **G4A — NEURAL ASSOCIATIVE LEARNING: "
      f"{v['G4A_verdict']}**")
    a(f"- **G4B — BEHAVIOURAL EXPRESSION: {v['G4B_verdict']}**")
    a(f"- **OVERALL GATE 4 (v2): {v['OVERALL_GATE4']}**\n")
    a("## G4A evidence\n")
    a1 = v["A1_eligibility"]
    a(f"- A1 eligibility lateralization: mean fraction of pairing trials "
      f"with paired-side eligibility > 2x unpaired = "
      f"**{a1['mean_fraction']}** (gate >= 0.90: "
      f"{'PASS' if a1['gate'] else 'FAIL'}); per session "
      f"{a1['per_session']}")
    a2 = v["A2_weights"]
    a(f"- A2 cue-specific synaptic change (paired- vs unpaired-context "
      f"KC->GLUT-MBON scales): pooled "
      f"{a2['pooled']['observed'] if a2['pooled'] else None} "
      f"CI95 {a2['pooled']['ci95'] if a2['pooled'] else None} "
      f"({'PASS' if a2['gate'] else 'FAIL'})")
    a3 = v["A3_mbon_response_descriptive"]
    a(f"- A3 (descriptive) avoidance-class MBON response to the paired "
      f"CS, baseline -> acquisition: "
      + "; ".join(f"{k}: {d['baseline']}->{d['acquisition']} "
                  f"({d['change']:+})" for k, d in a3.items()))
    a4 = v["A4_valence_signal"]
    a(f"- A4 learned valence signal (toward-rewarded, Hz): pooled "
      f"{a4['pooled']['observed']} CI95 {a4['pooled']['ci95']} "
      f"({'PASS' if a4['gate'] else 'FAIL'}); per session "
      f"{a4['per_session']}\n")
    a("## G4B evidence\n")
    b1 = v["B1_preference_shift"]
    a(f"- B1 route-B preference shift toward the rewarded side (pooled "
      f"6 sessions): {b1['pooled']['observed']} "
      f"CI95 {b1['pooled']['ci95']}; per session {b1['per_session']}")
    b2 = v["B2_both_directions"]
    a(f"- B2 both directions: RIGHT-rewarded mean "
      f"{b2['right_rewarded']['mean']} {b2['right_rewarded']['per']}; "
      f"LEFT-rewarded mean {b2['left_rewarded']['mean']} "
      f"{b2['left_rewarded']['per']}")
    a(f"- B3 controls (pooled shifts, CI must overlap 0):")
    for l, d in v["B3_controls"].items():
        a(f"  - {l}: {d['pooled']['observed']} CI95 "
          f"{d['pooled']['ci95']}"
          + (f" (vs B: {d.get('vs_B_diff')})" if 'vs_B_diff' in d else ""))
    a("")
    a("## Descriptive battery (B sessions)\n")
    a("| session | extinction first3 -> last3 (rewarded side) | reversal "
      "P(new side) | context x0.8 / x1.2 | distractor | route-A p(left) |")
    a("|---|---|---|---|---|---|")
    for l, d in v["descriptive_battery"].items():
        a(f"| {l} | {d['extinction']['first3']} -> "
          f"{d['extinction']['last3']} | {d['reversal']['p_new_side']} | "
          f"{d['context']['p_rewarded_x08']} / "
          f"{d['context']['p_rewarded_x12']} | "
          f"{d['distractor']['p_rewarded']} | "
          f"{d['route_a']['p_left']} ({d['route_a']['n_decided']}/"
          f"{d['route_a']['n']}) |")
    a("")
    rl = v.get("E_rl_descriptive", {})
    for l, d in rl.items():
        a(f"- E_rl {l} (NON-biological Q-learner): baseline "
          f"{d['phase_baseline']['p_rewarded_b']} -> acquisition "
          f"{d['phase_acquisition']['p_rewarded_b']}")
    a("\n## Performance (Section N)\n")
    a("| session | wall s | bio s | peak RSS MB | CPU s | mean pref "
      "wall s | trials |")
    a("|---|---|---|---|---|---|---|")
    for l, p in v["performance"].items():
        if p.get("session_wall_s") is None:
            continue
        a(f"| {l} | {p['session_wall_s']} | {p['bio_s_total']} | "
          f"{p['peak_rss_mb']} | {p['cpu_user_s']} | "
          f"{p['mean_pref_wall_s']} | {p['n_trials']} |")
    a("")
    a(f"Frozen calibration: n_cs={fz['n_cs_per_side']}/side, "
      f"kc_cs_right={fz['cs_rate_right_hz']} Hz, "
      f"kc_cs_left={fz['cs_rate_left_hz']} Hz "
      f"(boundary P(left)={fz['boundary']['p_left']}), "
      f"US={fz['us_set']} @ {fz['us_rate_hz']} Hz.")
    (OUT / "report_sections.md").write_text("\n".join(L) + "\n")
    print(f"[analyze] report sections -> {OUT/'report_sections.md'}")
    return "\n".join(L)


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "report":
        render_report()
    else:
        analyze()
