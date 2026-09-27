#!/usr/bin/env python3
"""FlyAgent command line interface.

Commands
--------
demo                     Run the full Phase-1 control-group demonstration
                         (5 conditions, comparison table, plots, bootstrap CIs).
run --controller NAME    Run one condition (random | fixed | fly | keyboard).
report --runs DIR ...    Compare existing runs (re-generates table + plots).
calibrate                Interactive 4-click arena calibration (webcam).

Every run writes an immutable timestamped directory under runs/ containing
manifest.json, experiment.db (SQLite) and summary.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flyagent.config import load_all_config, deep_merge, PROJECT_ROOT
from flyagent.experiments import run_experiment, run_demo, write_comparison
from flyagent.visualization.dashboard import Dashboard


def _out_base(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg["experiment"]["run"].get("out_dir", "runs")


def cmd_demo(args) -> int:
    cfg = load_all_config()
    if args.trials:
        cfg["experiment"]["demo"]["trials_per_condition"] = args.trials
    if args.scenario:
        cfg["experiment"]["game"]["scenario"] = args.scenario
    if args.timeout:
        cfg["experiment"]["game"]["timeout_s"] = args.timeout
    dash = Dashboard(enabled=args.show)
    comparison = run_demo(cfg, str(_out_base(cfg)), seed=args.seed,
                          trials=args.trials, dashboard=dash)
    dash.close()
    print(f"\ncomparison written to: {comparison['out_dir']}")
    return 0


def cmd_run(args) -> int:
    cfg = load_all_config()
    overrides = {}
    if args.scenario:
        overrides = deep_merge(overrides, {
            "experiment": {"game": {"scenario": args.scenario}}})
    if args.timeout:
        overrides = deep_merge(overrides, {
            "experiment": {"game": {"timeout_s": args.timeout}}})
    if args.source:
        if args.source != "synthetic" and args.controller not in ("fly", "keyboard"):
            print("--source only applies to fly controllers")
            return 2
        overrides = deep_merge(overrides, {
            "tracking": {"source": {"type": args.source}}})
    if args.video:
        overrides = deep_merge(overrides, {
            "tracking": {"source": {"type": "video", "video_path": args.video}}})
    if args.camera is not None:
        overrides = deep_merge(overrides, {
            "tracking": {"source": {"type": "webcam",
                                    "camera_index": args.camera}}})
    if args.cue_mode or args.cue_bias is not None:
        variant = {}
        if args.cue_mode:
            variant["cue_mode"] = args.cue_mode
        if args.cue_bias is not None:
            variant["cue_bias"] = args.cue_bias
        overrides["_fly_variant"] = variant
    cfg = deep_merge(cfg, overrides)

    condition = args.condition or args.controller
    dash = Dashboard(enabled=args.show)
    summary = run_experiment(condition, cfg, str(_out_base(cfg)),
                             seed=args.seed, trials=args.trials,
                             notes=args.notes, dashboard=dash)
    dash.close()
    print(f"\nrun directory: {summary['run_dir']}")
    print(f"success rate:  {summary['aggregate']['success_rate']:.1%}")
    return 0


def cmd_report(args) -> int:
    import json
    summaries = []
    for r in args.runs:
        p = Path(r) / "summary.json"
        if not p.exists():
            print(f"skip (no summary.json): {r}")
            continue
        summaries.append(json.loads(p.read_text()))
    if len(summaries) < 2:
        print("need at least two runs to compare")
        return 2
    out = Path(args.out) if args.out else Path(args.runs[0]).parent / "report"
    comp = write_comparison(summaries, out, seed=args.seed)
    print(comp["table"])
    print(f"\nreport written to: {comp['out_dir']}")
    return 0


def cmd_calibrate(args) -> int:
    from scripts.calibrate_arena import main as cal_main
    return cal_main(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="flyagent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo", help="run the Phase-1 control-group demo")
    p.add_argument("--trials", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--scenario", choices=["open_field", "obstacles"])
    p.add_argument("--timeout", type=float)
    p.add_argument("--show", action="store_true",
                   help="live dashboard (requires a display)")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("run", help="run a single condition")
    p.add_argument("--controller", required=True,
                   choices=["random", "fixed", "fly", "keyboard"])
    p.add_argument("--condition", help="override condition label "
                                        "(e.g. fly_random)")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scenario", choices=["open_field", "obstacles"])
    p.add_argument("--timeout", type=float)
    p.add_argument("--source", choices=["synthetic", "webcam", "video"])
    p.add_argument("--video", help="path to a recorded fly video")
    p.add_argument("--camera", type=int, help="webcam index")
    p.add_argument("--cue-mode", choices=["goal", "shuffled", "none"])
    p.add_argument("--cue-bias", type=float,
                   help="synthetic fly taxis strength 0..1")
    p.add_argument("--show", action="store_true")
    p.add_argument("--notes", default="")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="compare existing runs")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out")
    p.add_argument("--seed", type=int, default=12345)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("calibrate", help="interactive arena calibration")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", default=str(PROJECT_ROOT / "config" / "calibration.npz"))
    p.set_defaults(fn=cmd_calibrate)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
