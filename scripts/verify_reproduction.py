#!/usr/bin/env python3
"""Verify a Phase-1 reproduction run against the frozen original results.

Compares, for each of the 5 demo conditions:
  1. every per-trial metric (outcome, durations, path metrics, reward, ...)
  2. every aggregate statistic in summary.json / comparison.json
  3. every bootstrap contrast (diff, CI bounds, p-value)

Materiality criterion (from PHASE1_BASELINE.md section 8): the reproduction
is IDENTICAL only if all per-trial metrics and all bootstrap statistics match
exactly (same seeds, same code => bitwise-equal floats). ANY mismatch is
material: STOP and diagnose. Exit code 0 = identical, 1 = different.

Usage:
  python scripts/verify_reproduction.py \
      --original-demo runs/demo_20260927-122233 \
      --new-demo    runs/demo_<new-timestamp> \
      --original-runs runs/20260927-121624-527412_random \
                      runs/20260927-121624-718101_fixed \
                      runs/20260927-121624-751794_fly_random \
                      runs/20260927-121907-419806_fly_cue \
                      runs/20260927-122026-804472_fly_cue_shuffled \
      --new-runs runs/<t1>_random runs/<t2>_fixed runs/<t3>_fly_random \
                 runs/<t4>_fly_cue runs/<t5>_fly_cue_shuffled
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Fields compared exactly per trial (metrics only; run_dir/experiment_id are
# directory names that legitimately differ between runs).
TRIAL_FIELDS = [
    "trial_number", "started_at_ms", "duration_s", "outcome",
    "time_to_target_s", "path_length", "straight_dist", "path_efficiency",
    "n_actions", "n_action_changes", "mean_inter_action_s", "reward_total",
    "target_x", "target_y", "start_x", "start_y",
]
AGG_FIELDS = [
    "success_rate", "median_time_to_target_s", "median_path_efficiency",
    "mean_actions", "mean_action_changes", "mean_inter_action_s",
    "mean_reward",
]
BOOT_FIELDS = ["diff", "ci_low", "ci_high", "p_le_0"]


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def by_condition(run_dirs: list[str]) -> dict[str, dict]:
    out = {}
    for d in run_dirs:
        s = load_json(Path(d) / "summary.json")
        cond = s["condition"]
        if cond in out:
            raise SystemExit(f"duplicate condition {cond!r} in run list")
        out[cond] = s
    return out


def cmp_val(a, b) -> tuple[bool, float]:
    """Return (equal, magnitude_of_difference)."""
    if a == b:
        return True, 0.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            return False, abs(float(a) - float(b))
        except (TypeError, ValueError):
            return False, float("inf")
    return False, float("inf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original-demo", required=True)
    ap.add_argument("--new-demo", required=True)
    ap.add_argument("--original-runs", nargs="+", required=True)
    ap.add_argument("--new-runs", nargs="+", required=True)
    args = ap.parse_args()

    orig = by_condition(args.original_runs)
    new = by_condition(args.new_runs)

    problems: list[str] = []
    report: list[str] = []

    # ---------------------------------------------------------- per-trial
    report.append("=" * 78)
    report.append(f"{'condition':<18}{'trials':>7}{'mismatches':>11}"
                  f"{'max |diff|':>14}  verdict")
    report.append("-" * 78)
    for cond in sorted(orig):
        if cond not in new:
            problems.append(f"condition {cond!r} missing from new runs")
            continue
        pt_o, pt_n = orig[cond]["per_trial"], new[cond]["per_trial"]
        if len(pt_o) != len(pt_n):
            problems.append(f"{cond}: trial count {len(pt_o)} vs {len(pt_n)}")
            continue
        mism, maxd = 0, 0.0
        for i, (ro, rn) in enumerate(zip(pt_o, pt_n)):
            for f in TRIAL_FIELDS:
                eq, d = cmp_val(ro.get(f), rn.get(f))
                if not eq:
                    mism += 1
                    maxd = max(maxd, d)
                    problems.append(
                        f"{cond} trial {i + 1} field {f}: "
                        f"{ro.get(f)!r} != {rn.get(f)!r}")
        report.append(f"{cond:<18}{len(pt_n):>7}{mism:>11}{maxd:>14.3g}  "
                      f"{'IDENTICAL' if mism == 0 else 'DIFFERENT'}")

        # ------------------------------------------------------ aggregates
        for f in AGG_FIELDS:
            eq, d = cmp_val(orig[cond]["aggregate"].get(f),
                            new[cond]["aggregate"].get(f))
            if not eq:
                problems.append(f"{cond} aggregate {f}: "
                                f"{orig[cond]['aggregate'].get(f)!r} != "
                                f"{new[cond]['aggregate'].get(f)!r}")

    # ------------------------------------------------------- bootstrap CIs
    co, cn = load_json(Path(args.original_demo) / "comparison.json"), \
        load_json(Path(args.new_demo) / "comparison.json")
    report.append("-" * 78)
    report.append("Bootstrap contrasts (10,000 iterations, seed 20260927):")
    for key, vo in co["bootstrap_success_rate_differences"].items():
        vn = cn["bootstrap_success_rate_differences"].get(key)
        if vn is None:
            problems.append(f"bootstrap contrast {key} missing in new demo")
            continue
        line_ok = True
        for f in BOOT_FIELDS:
            eq, d = cmp_val(vo.get(f), vn.get(f))
            if not eq:
                line_ok = False
                problems.append(f"bootstrap {key} {f}: {vo.get(f)!r} != "
                                f"{vn.get(f)!r}")
        report.append(f"  {key:<42} "
                      f"{'IDENTICAL' if line_ok else 'DIFFERENT'}")
    go, gn = co["gate_phase1"]["passed"], cn["gate_phase1"]["passed"]
    report.append(f"  {'gate_phase1.passed':<42} "
                  f"{go} -> {gn}  {'IDENTICAL' if go == gn else 'DIFFERENT'}")
    if go != gn:
        problems.append(f"gate outcome changed: {go} -> {gn}")

    # ---------------------------------------------------------------- out
    report.append("=" * 78)
    if problems:
        report.append(f"VERDICT: MATERIALLY DIFFERENT "
                      f"({len(problems)} mismatched item(s); first 10 below)")
        report.extend(f"  ! {p}" for p in problems[:10])
        verdict = 1
    else:
        report.append("VERDICT: IDENTICAL - reproduction matches the frozen "
                      "baseline exactly (all per-trial metrics, all "
                      "aggregates, all bootstrap contrasts, gate outcome).")
        verdict = 0

    text = "\n".join(report)
    print(text)
    out = Path(args.new_demo) / "reproduction_verification.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwritten: {out}")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
