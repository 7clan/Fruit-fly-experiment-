"""Control-group orchestration + statistics for the Phase-1 demonstration.

Conditions (all through the SAME trial loop and metrics):
  random            floor: uniform random actions
  fixed             ceiling: computer picks the greedy direction directly
  fly_random        fly-controlled, cue presented but fly ignores it (bias 0)
  fly_cue           fly-controlled, goal-directed cue, taxis fly
  fly_cue_shuffled  same fly, cue position randomized every ~2 s
                    -> isolates the closed-loop goal signal from fly taxis

Key statistics: bootstrap 95% CI of the success-rate DIFFERENCE between
conditions. If (fly_cue - random) and (fly_cue - fly_cue_shuffled) exclude
zero, the apparatus can detect a biological-behavior contribution - the
Phase-1 graduation gate.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from .runner import run_experiment


def bootstrap_diff_ci(success_a, success_b, iters: int, rng) -> dict:
    """Bootstrap CI for mean(success_a) - mean(success_b)."""
    a = np.asarray([1 if s else 0 for s in success_a], dtype=float)
    b = np.asarray([1 if s else 0 for s in success_b], dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return {"diff": None, "ci_low": None, "ci_high": None, "p_le_0": None}
    diffs = np.empty(iters)
    for i in range(iters):
        ra = rng.choice(a, n_a, replace=True)
        rb = rng.choice(b, n_b, replace=True)
        diffs[i] = ra.mean() - rb.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": float(a.mean() - b.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_le_0": float((diffs <= 0).mean()),
    }


COMPARISON_PAIRS = [
    ("fly_cue", "random"),
    ("fly_cue", "fly_random"),
    ("fly_cue", "fly_cue_shuffled"),
    ("fixed", "random"),
]


def write_comparison(summaries: list[dict], out_dir, iters: int = 10000,
                     seed: int = 12345) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    by_cond = {s["condition"]: s for s in summaries}
    stats = {}
    for a, b in COMPARISON_PAIRS:
        if a in by_cond and b in by_cond:
            sa = by_cond[a]["per_trial"]
            sb = by_cond[b]["per_trial"]
            stats[f"{a}__vs__{b}"] = bootstrap_diff_ci(
                [r["outcome"] == "SUCCESS" for r in sa],
                [r["outcome"] == "SUCCESS" for r in sb], iters, rng)

    comparison = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "conditions": {s["condition"]: s["aggregate"] for s in summaries},
        "bootstrap_success_rate_differences": stats,
        "gate_phase1": {
            "description": "Apparatus validated if fly_cue - random AND "
                           "fly_cue - fly_cue_shuffled CIs exclude 0",
            "passed": None,
        },
    }
    gate = stats.get("fly_cue__vs__random")
    gate2 = stats.get("fly_cue__vs__fly_cue_shuffled")
    if gate and gate2 and gate["diff"] is not None and gate2["diff"] is not None:
        comparison["gate_phase1"]["passed"] = bool(
            gate["ci_low"] > 0 and gate2["ci_low"] > 0)

    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8")

    # ----------------------------------------------------------- text table
    lines = []
    header = (f"{'condition':<18}{'n':>4}{'succ%':>8}{'med TTT':>9}"
              f"{'med pathEff':>13}{'meanAct':>9}{'meanReward':>12}")
    lines.append(header)
    lines.append("-" * len(header))
    for s in summaries:
        a = s["aggregate"]
        ttt = a["median_time_to_target_s"]
        eff = a["median_path_efficiency"]
        ttt_s = f"{ttt:.1f}s" if ttt is not None else "--"
        eff_s = f"{eff:.2f}" if eff is not None else "--"
        lines.append(
            f"{s['condition']:<18}{s['trials']:>4}"
            f"{100 * a['success_rate']:>7.1f}%"
            f"{ttt_s:>9}{eff_s:>13}"
            f"{a['mean_actions']:>9.1f}{a['mean_reward']:>12.2f}")
    lines.append("")
    lines.append("Bootstrap 95% CI of success-rate differences:")
    for k, v in stats.items():
        if v["diff"] is None:
            continue
        stars = "  *" if (v["ci_low"] is not None and v["ci_low"] > 0) else ""
        lines.append(f"  {k:<42} diff={v['diff']:+.3f}  "
                     f"CI[{v['ci_low']:+.3f}, {v['ci_high']:+.3f}] "
                     f"p(diff<=0)={v['p_le_0']:.4f}{stars}")
    gate = comparison["gate_phase1"]
    lines.append("")
    lines.append(f"PHASE-1 GATE: {'PASSED' if gate['passed'] else 'NOT PASSED'}"
                 f" - {gate['description']}")
    table = "\n".join(lines)
    (out_dir / "comparison.txt").write_text(table, encoding="utf-8")

    from ..visualization.plots import comparison_figure
    comparison_figure(summaries, out_dir / "comparison.png")

    comparison["table"] = table
    comparison["out_dir"] = str(out_dir)
    return comparison


def run_demo(cfg: dict, out_base: str, seed: int = 20260927,
             trials: int | None = None, dashboard=None) -> dict:
    """Run every demo condition sequentially and produce the comparison."""
    exp = cfg["experiment"]
    trials = int(trials or exp["demo"]["trials_per_condition"])
    summaries = []
    for i, cond in enumerate(exp["demo"]["conditions"]):
        print(f"\n=== condition {i + 1}/{len(exp['demo']['conditions'])}: "
              f"{cond} ({trials} trials) ===", flush=True)
        summaries.append(run_experiment(cond, cfg, out_base, seed + 1000 * i,
                                        trials, notes="phase1 demo",
                                        dashboard=dashboard))
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_base) / f"demo_{ts}"
    comparison = write_comparison(
        summaries, out_dir,
        iters=int(exp["demo"].get("bootstrap_iterations", 10000)), seed=seed)
    print("\n" + comparison["table"])
    return comparison
