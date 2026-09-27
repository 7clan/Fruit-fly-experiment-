#!/usr/bin/env python3
"""Phase-2 command line interface.

Commands
--------
validate      Run the FULL pipeline validation on synthetic ground-truth
              videos (software acceptance SA1-SA12). No fly involved; the
              synthetic videos are a CODE test, not evidence about flies.
calibrate     Interactive rig calibration (ruler + arena circle clicks)
              -> data/rig/calibration.json. --verify / --recheck-scale for
              non-interactive checks.
preflight     Camera/environment check BEFORE any animal session (fps,
              exposure stability, focus, geometry, noise). Writes a PASS
              into the rig state.
blank         Blank-arena test (no fly, >= 3 min): records, tracks, and
              verifies zero false detections + marker channel; PASS stores
              the reference background for fly sessions.
record        Record a live session from the camera (user machine).
              Refuses fly sessions unless preflight+blank passed today.
              --stimulus generates the pre-registered open-loop schedule.
track         Run the tracker over a session's video -> tracks.csv (uses
              the session's recorded tracking recipe, incl. reference mode).
annotate      Manual annotation tool (G2 accuracy; needs a display) or
              --from-gt (software validation only).
analyze       Full behavioral analysis of one or more session dirs
              (also writes per_frame.csv - the derived per-frame record).
intake        FIRST step when recordings come back: file integrity +
              protocol compliance verification (no analysis, no tuning).
gate          Evaluate the pre-registered PHASE2-GATE-1.0.0 on an analysis
              directory. FAILS CLOSED if annotations are missing.
backup        Immutable, content-addressed backup of all session dirs.

Examples
--------
    python phase2/run_phase2.py validate
    python phase2/run_phase2.py calibrate --camera 0 --arena-mm 95
    python phase2/run_phase2.py preflight --camera 0
    python phase2/run_phase2.py blank --camera 0
    python phase2/run_phase2.py record --minutes 10 --label baseline_01 \
        --fly-id F01 --age 5 --sex M --temperature 23 --humidity 45
    python phase2/run_phase2.py record --minutes 10 --label stimulus_01 \
        --stimulus --session-index 0 --fly-id F01 --age 5 --sex M \
        --temperature 23 --humidity 45
    python phase2/run_phase2.py track data/sessions/<dir>
    python phase2/run_phase2.py annotate data/sessions/<dir> --n 50
    python phase2/run_phase2.py intake data/sessions
    python phase2/run_phase2.py analyze data/sessions/A data/sessions/B \
        --out results/first_analysis
    python phase2/run_phase2.py gate --analysis results/first_analysis \
        --annotations data/sessions/<a>/annotations.csv ...
    python phase2/run_phase2.py backup --to /mnt/usb-drive
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _setup_path():
    """Make `flyrec` importable when invoked as a script."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def main(argv=None) -> int:
    _setup_path()
    from flyrec.config import (DATA_DIR, RESULTS_DIR, analysis_constants,
                               load_gate_cfg, load_tracking_cfg)

    ap = argparse.ArgumentParser(
        prog="phase2", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="pipeline validation on synthetic GT")
    p.add_argument("--out", default=None)
    p.add_argument("--no-regenerate", action="store_true")
    p.add_argument("--duration", type=float, default=60.0)

    p = sub.add_parser("calibrate", help="interactive rig calibration")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None,
                   help="grab the calibration frame from a video instead")
    p.add_argument("--arena-mm", type=float, default=None,
                   help="measured physical arena inner diameter in mm")
    p.add_argument("--verify", action="store_true",
                   help="non-interactive check of the stored calibration")
    p.add_argument("--recheck-scale", action="store_true",
                   help="quick 2-click drift check against the stored scale")

    p = sub.add_parser("preflight", help="camera/environment check (no fly)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seconds", type=float, default=None)

    p = sub.add_parser("blank", help="blank-arena test (no fly)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--skip-marker-check", action="store_true",
                   help="do not require the N/E/S/W LED blink check "
                        "(recorded as skipped - only for debugging)")
    p.add_argument("--force", action="store_true",
                   help="record even without a preflight PASS (deviation)")

    p = sub.add_parser("record", help="record a live session (camera)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--label", default="baseline")
    p.add_argument("--fly-id", default="")
    p.add_argument("--age", type=float, default=None)
    p.add_argument("--sex", default="")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--humidity", type=float, default=None)
    p.add_argument("--px-per-mm", type=float, default=None)
    p.add_argument("--arena", default=None, help="JSON geometry")
    p.add_argument("--stimulus", action="store_true")
    p.add_argument("--session-index", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="override the apparatus gate (recorded as a "
                        "protocol deviation and flagged by intake)")
    p.add_argument("--override-reason", default="",
                   help="why --force was used (recorded in session.json)")
    p.add_argument("--operator-notes", default="",
                   help="free-text notes recorded in session.json")

    p = sub.add_parser("track", help="video -> tracks.csv")
    p.add_argument("session_dir")
    p.add_argument("--video", default=None)

    p = sub.add_parser("annotate", help="manual annotation (interactive)")
    p.add_argument("session_dir")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--from-gt", action="store_true",
                   help="SOFTWARE VALIDATION ONLY: GT-derived annotations")

    p = sub.add_parser("intake", help="integrity + compliance verification "
                                      "(FIRST step when data comes back)")
    p.add_argument("sessions", nargs="+",
                   help="session dirs and/or a sessions root to scan")
    p.add_argument("--out", default=None)

    p = sub.add_parser("analyze", help="behavioral statistics")
    p.add_argument("session_dirs", nargs="+")
    p.add_argument("--out", default=None)
    p.add_argument("--stimulus-events", default=None,
                   help="optional JSON file with events per session")

    p = sub.add_parser("gate", help="evaluate PHASE2-GATE-1.0.0")
    p.add_argument("--analysis", required=True,
                   help="analysis output dir (from `analyze`)")
    p.add_argument("--annotations", nargs="*", default=None,
                   help="annotations.csv files (required for G2)")
    p.add_argument("--subject", default="real-fly")

    p = sub.add_parser("backup", help="immutable session backup")
    p.add_argument("--to", required=True,
                   help="backup target directory (e.g. a USB drive mount)")
    p.add_argument("--sessions", default=None,
                   help="sessions root (default: phase2/data/sessions)")

    args = ap.parse_args(argv)

    if args.cmd == "validate":
        from flyrec.validation.validate import validate
        res = validate(out_dir=Path(args.out) if args.out else None,
                       regenerate=not args.no_regenerate,
                       duration_s=args.duration)
        print(json.dumps({k: v["pass"] for k, v in res["criteria"].items()},
                         indent=2))
        print("OVERALL:", "PASS" if res["overall_pass"] else "FAIL")
        return 0 if res["overall_pass"] else 1

    if args.cmd == "calibrate":
        from flyrec.validation.calibrate_tool import (interactive_calibrate,
                                                    recheck_scale,
                                                    verify_calibration)
        if args.verify:
            rep = verify_calibration()
            print(json.dumps(rep, indent=2, default=str))
            return 0 if rep.get("pass") else 1
        if args.recheck_scale:
            rep = recheck_scale(camera_index=args.camera, video=args.video)
            print(json.dumps(rep, indent=2, default=str))
            return 0 if rep.get("pass") else 1
        interactive_calibrate(camera_index=args.camera, video=args.video,
                              arena_mm=args.arena_mm)
        return 0

    if args.cmd == "preflight":
        from flyrec.validation.apparatus import render_preflight, run_preflight
        rep = run_preflight(camera_index=args.camera, seconds=args.seconds)
        print(render_preflight(rep))
        if not rep.get("pass"):
            print("report:", rep.get("report_file") or rep.get("error"))
        return 0 if rep.get("pass") else 1

    if args.cmd == "blank":
        from flyrec.validation.apparatus import render_blank, run_blank
        rep = run_blank(camera_index=args.camera, minutes=args.minutes,
                        marker_check=not args.skip_marker_check,
                        force=args.force)
        print(render_blank(rep))
        if not rep.get("pass"):
            print("report:", rep.get("report_file") or rep.get("error"))
        return 0 if rep.get("pass") else 1

    if args.cmd == "record":
        from flyrec.recording.recorder import record
        arena = json.loads(args.arena) if args.arena else None
        sdir = record(camera_index=args.camera, minutes=args.minutes,
                      label=args.label, fly_id=args.fly_id,
                      fly_age_days=args.age, fly_sex=args.sex,
                      temperature_c=args.temperature,
                      humidity_pct=args.humidity, px_per_mm=args.px_per_mm,
                      arena=arena, stimulus=args.stimulus,
                      session_index=args.session_index, force=args.force,
                      override_reason=args.override_reason,
                      operator_notes=args.operator_notes)
        print(f"session dir: {sdir}")
        return 0

    if args.cmd == "track":
        from flyrec.tracking.tracker import track_video
        from flyrec.tracking.calibration import calibration_from_session
        from flyrec.rig import sha256_file
        sdir = Path(args.session_dir)
        meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
        calib = calibration_from_session(meta)
        video = Path(args.video) if args.video else \
            next(p for p in sdir.glob("video.*"))
        tcfg = load_tracking_cfg()
        # the session's recorded tracking recipe (reference mode + the
        # blank-derived background for real-fly sessions) overrides the
        # generic config, with provenance verification
        sess_tr = meta.get("tracking") or {}
        if sess_tr.get("mode") == "reference":
            bg = Path(sess_tr.get("background_ref", ""))
            if not bg.exists():
                print(f"ERROR: background reference missing: {bg}")
                return 1
            got = sha256_file(bg)
            want = sess_tr.get("background_ref_sha256")
            if want and got != want:
                print(f"ERROR: background reference hash mismatch\n"
                      f"  expected {want}\n  found    {got}\n"
                      "  (the rig state changed after this session - "
                      "re-run the blank test and keep the original "
                      "background file archived with the session)")
                return 1
            det = dict(tcfg.get("detector", {}))
            det["mode"] = "reference"
            det["reference_file"] = str(bg)
            tcfg = {**tcfg, "detector": det}
        stats = track_video(video, sdir / "tracks.csv", tcfg,
                            calib)
        print(json.dumps(stats, indent=2))
        return 0

    if args.cmd == "annotate":
        from flyrec.tracking.annotate import annotate_from_gt, \
            annotate_interactive
        if args.from_gt:
            annotate_from_gt(Path(args.session_dir), n=args.n, seed=args.seed)
        else:
            annotate_interactive(Path(args.session_dir), n=args.n,
                                 seed=args.seed)
        return 0

    if args.cmd == "analyze":
        from flyrec.analysis.statistics import analyze_session, \
            summarize_cross_session
        from flyrec.figures import plot_session
        from flyrec.vocabulary.states import STATES_A
        gate_cfg = load_gate_cfg()
        consts = analysis_constants(gate_cfg)
        tcfg = load_tracking_cfg()
        out = Path(args.out) if args.out else RESULTS_DIR / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        events_override = {}
        if args.stimulus_events:
            events_override = json.loads(Path(args.stimulus_events)
                                         .read_text(encoding="utf-8"))
        analyses = []
        for sd in args.session_dirs:
            print(f"analyzing {sd} ...")
            ev = events_override.get(Path(sd).name)
            a = analyze_session(sd, consts, tracking_cfg=tcfg,
                                stimulus_events=ev)
            plot_session(a, out / f"{a['label']}_overview.png")
            analyses.append(a)
        summary = summarize_cross_session(analyses, STATES_A)
        payload = {
            "sessions": [{k: v for k, v in a.items() if k != "windows"}
                         for a in analyses],
            "cross_session": summary,
        }
        (out / "behavioral_statistics.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
        # keep windows for the gate step
        (out / "_windows_cache.json").write_text(
            json.dumps({a["label"]: a["windows"] for a in analyses}),
            encoding="utf-8")
        print(f"analysis written to {out}")
        return 0

    if args.cmd == "intake":
        from flyrec.validation.intake import intake, render_intake
        rep = intake(args.sessions,
                     out_dir=Path(args.out) if args.out else None)
        print(render_intake(rep))
        if rep.get("verdict") == "FAIL":
            return 1
        return 0

    if args.cmd == "gate":
        from flyrec.gate.evaluate import evaluate_gate, load_annotations, \
            annotation_errors_mm
        from flyrec.tracking.calibration import calibration_from_session
        adir = Path(args.analysis)
        payload = json.loads((adir / "behavioral_statistics.json")
                             .read_text(encoding="utf-8"))
        windows = json.loads((adir / "_windows_cache.json")
                             .read_text(encoding="utf-8"))
        analyses = []
        for s in payload["sessions"]:
            a = dict(s)
            a["windows"] = windows.get(a["label"], [])
            analyses.append(a)
        errs = None
        if args.annotations:
            session_dirs = {a["label"]: Path(a["session_dir"])
                            for a in analyses}
            calibs = {}
            for a in analyses:
                meta = json.loads((Path(a["session_dir"]) / "session.json")
                                  .read_text(encoding="utf-8"))
                calibs[a["label"]] = calibration_from_session(meta)
            ann = load_annotations(args.annotations)
            errs = annotation_errors_mm(ann, session_dirs, calibs)
            if len(errs) < 150:
                print(f"WARNING: only {len(errs)} usable annotations "
                      f"(pre-registered minimum 150)")
        report = evaluate_gate(analyses, annotations=errs,
                               subject=args.subject, out_dir=adir)
        from flyrec.gate.evaluate import render_text
        print(render_text(report))
        return 0 if report["passed"] else 1

    if args.cmd == "backup":
        from flyrec.data_mgmt.backup import backup, render_backup
        rep = backup(args.to,
                     sessions_root=Path(args.sessions)
                     if args.sessions else None)
        print(render_backup(rep))
        return 0 if rep.get("ok") else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
