"""End-to-end pipeline validation on synthetic ground-truth videos.

Runs the ENTIRE Phase-2 apparatus (tracking -> kinematics -> behaviors ->
circular stats -> stimulus analysis -> vocabulary ladder -> gate) against
known ground truth and checks the pre-registered SOFTWARE ACCEPTANCE
criteria SA1-SA11 (PHASE2_GATE_PREREGISTRATION.md section 8).

THIS VALIDATES CODE ONLY. The synthetic videos are a software test pattern;
they are NOT evidence about real Drosophila behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, RESULTS_DIR, analysis_constants, load_gate_cfg, \
    load_tracking_cfg
from ..gate.evaluate import evaluate_gate
from ..tracking.calibration import calibration_from_session
from ..tracking.tracker import load_tracks, track_video
from ..tracking.annotate import annotate_from_gt
from ..analysis.statistics import analyze_session, summarize_cross_session
from .synthetic_video import generate_validation_suite

import cv2


def _gt(session_dir: Path) -> dict:
    gt = np.genfromtxt(Path(session_dir) / "ground_truth.csv", delimiter=",",
                       names=True, dtype=None, encoding="utf-8")
    if gt.ndim == 0:
        gt = gt.reshape(1)
    return gt


def _to_float(a):
    return np.array([np.nan if v in ("", "nan") else float(v) for v in a],
                    dtype=float)


def validate(out_dir: Path | None = None, regenerate: bool = True,
             duration_s: float = 60.0) -> dict:
    gate_cfg = load_gate_cfg()
    consts = analysis_constants(gate_cfg)
    tcfg = load_tracking_cfg()
    sa = gate_cfg["software_acceptance"]
    from ..tracking.calibration import calibration_from_session
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR / "phase2_pipeline_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_root = DATA_DIR / "validation"
    if regenerate or not (val_root / "baseline_01").exists():
        print("generating synthetic ground-truth sessions ...")
        session_dirs = generate_validation_suite(val_root, duration_s)
    else:
        session_dirs = [val_root / d for d in
                        ("baseline_01", "baseline_02", "baseline_03",
                         "stimulus_01", "stimulus_02", "blank_01")]

    results = {"purpose": "SOFTWARE VALIDATION ONLY - validates the "
                          "measurement code against synthetic ground truth; "
                          "NOT evidence about real flies",
               "criteria": {}, "sessions": {}}

    # ------------------------------------------------ track every session
    for sdir in session_dirs:
        sdir = Path(sdir)
        meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
        calib = calibration_from_session(meta)
        video = next(sdir.glob("video.*"))
        print(f"tracking {sdir.name} ...", flush=True)
        tstats = track_video(video, sdir / "tracks.csv", tcfg, calib,
                             read_stimulus=True)
        results["sessions"][sdir.name] = tstats

    # ---------------------------------------------- per-session SA checks
    per_session = {}
    for sdir in session_dirs:
        sdir = Path(sdir)
        name = sdir.name
        if name == "blank_01":
            tr = load_tracks(sdir / "tracks.csv")
            n_det = int(tr["found"].sum())
            results["criteria"]["SA11_blank_zero_detections"] = {
                "n_detections": n_det,
                "pass": n_det <= int(sa["SA11_blank_max_detections"]),
            }
            continue
        gt = _gt(sdir)
        tr = load_tracks(sdir / "tracks.csv")
        present = gt["present"].astype(int) == 1
        gp = np.stack([_to_float(gt["x_px"]), _to_float(gt["y_px"])], axis=1)
        det = tr["found"]
        # coverage over fly-present frames at confidence >= 0.5 (G1 semantics)
        cov = float(((det) & (tr["conf"] >= 0.5))[present].mean())
        # position error where both GT and detection exist
        both = present & det
        err_px = np.hypot(tr["x_px"][both] - gp[both, 0],
                          tr["y_px"][both] - gp[both, 1])
        err_mm = err_px / 5.0          # PX_PER_MM of the test pattern
        per_session[name] = {"coverage": cov, "err_mm": err_mm,
                             "gt": gt, "tracks": tr}

    # SA1 / SA2
    covs = {k: v["coverage"] for k, v in per_session.items()}
    results["criteria"]["SA1_coverage"] = {
        "observed": {k: round(v, 4) for k, v in covs.items()},
        "min": round(min(covs.values()), 4) if covs else None,
        "pass": bool(covs) and min(covs.values()) >= float(sa["SA1_min_coverage"]),
    }
    all_err = np.concatenate([v["err_mm"] for v in per_session.values()]) \
        if per_session else np.array([])
    results["criteria"]["SA2_position_error"] = {
        "median_mm": float(np.median(all_err)) if len(all_err) else None,
        "p95_mm": float(np.percentile(all_err, 95)) if len(all_err) else None,
        "pass": bool(len(all_err) and
                     np.median(all_err) <= float(sa["SA2_median_err_max_mm"]) and
                     np.percentile(all_err, 95) <= float(sa["SA2_p95_err_max_mm"])),
    }

    # -------------------------------------------- behavioral metric checks
    # GT reference metrics are computed by pushing the GROUND-TRUTH positions
    # through the IDENTICAL analysis code path (same smoothing, same bout
    # logic, same wall/circular metrics). SA3-SA7 therefore measure how much
    # TRACKING ERROR distorts the metrics - not definitional drift between
    # two implementations.
    from ..analysis.behaviors import find_bouts, find_pauses, wall_metrics
    from ..analysis.circular import mean_resultant as _circ_mr
    from ..analysis.trajectory import compute_kinematics, interpolate_gaps

    analyses, gt_stats = [], {}
    for sdir in session_dirs:
        sdir = Path(sdir)
        if sdir.name == "blank_01":
            continue
        a = analyze_session(sdir, consts, tracking_cfg=tcfg)
        analyses.append(a)
        gt = _gt(sdir)
        present = gt["present"].astype(int) == 1
        t_all = gt["t_s"].astype(float)
        x_all, y_all = _to_float(gt["x_mm"]), _to_float(gt["y_mm"])
        first = int(np.argmax(present)) if present.any() else 0
        t_g, x_g, y_g = t_all[first:], x_all[first:], y_all[first:]
        f_g = present[first:]
        calib = calibration_from_session(
            json.loads((sdir / "session.json").read_text(encoding="utf-8")))
        x_gs, y_gs, _ = interpolate_gaps(t_g, x_g, y_g, f_g, max_gap_s=0.5)
        kin_g = compute_kinematics(t_g, x_gs, y_gs)
        valid_g = f_g & np.isfinite(x_gs) & np.isfinite(y_gs)
        bouts_g = find_bouts(t_g, kin_g["speed"], valid_g, on=3.0, off=2.0,
                             min_dur=0.25, gap_merge=0.2)
        pauses_g = find_pauses(t_g, bouts_g, min_pause=0.30,
                               t_end=float(t_g[-1]) if len(t_g) else 0.0)
        dur_g = (t_g[-1] / 60.0) if len(t_g) else 0.0
        wstats_g, _ = wall_metrics(t_g, kin_g["x"], kin_g["y"],
                                   kin_g["heading"], valid_g, calib,
                                   wall_band_mm=10.0, parallel_deg=45.0,
                                   moving_speed=kin_g["speed"])
        h_g = kin_g["heading"][valid_g & np.isfinite(kin_g["heading"]) &
                               (kin_g["speed"] >= 2.0)]
        _, R_g = _circ_mr(h_g)
        gt_stats[sdir.name] = {
            "mean_speed": float(np.nanmean(kin_g["speed"][valid_g]))
            if valid_g.any() else None,
            "pause_rate": (len(pauses_g) / dur_g) if dur_g else None,
            "median_bout_s": float(np.median(
                [b["duration_s"] for b in bouts_g])) if bouts_g else None,
            "wall_frac_moving": wstats_g["frac_moving_wall_following"],
            "resultant": R_g,
        }

    # SA3 mean speed
    sp_ok = all(abs(a["movement"]["speed_mean_mm_s"] -
                    gt_stats[a["label"]]["mean_speed"]) <=
                sa["SA3_speed_rel_tol"] * gt_stats[a["label"]]["mean_speed"]
                for a in analyses if a["movement"]["speed_mean_mm_s"])
    results["criteria"]["SA3_speed"] = {
        "tracked_vs_gt": {a["label"]: [a["movement"]["speed_mean_mm_s"],
                                       round(gt_stats[a["label"]]["mean_speed"], 3)]
                          for a in analyses},
        "pass": sp_ok,
    }
    # SA4 pause frequency
    pf = {a["label"]: (a["pauses"]["rate_per_min"],
                       gt_stats[a["label"]]["pause_rate"]) for a in analyses}
    results["criteria"]["SA4_pause_frequency"] = {
        "tracked_vs_gt": {k: [round(v[0], 3), round(v[1], 3)]
                          for k, v in pf.items()},
        "pass": all(abs(v[0] - v[1]) <= sa["SA4_pause_freq_rel_tol"] * v[1]
                    for v in pf.values() if v[0] is not None and v[1] > 0),
    }
    # SA5 median bout duration
    bd = {a["label"]: (a["bouts"]["median_duration_s"],
                       gt_stats[a["label"]]["median_bout_s"]) for a in analyses}
    results["criteria"]["SA5_bout_duration"] = {
        "tracked_vs_gt": {k: [round(v[0], 3), round(v[1], 3)]
                          for k, v in bd.items()},
        "pass": all(v[0] is not None and v[1] is not None and
                    abs(v[0] - v[1]) <= sa["SA5_bout_dur_rel_tol"] * v[1]
                    for v in bd.values()),
    }
    # SA6 wall-following
    wf = {a["label"]: (a["wall"]["frac_moving_wall_following"],
                       gt_stats[a["label"]]["wall_frac_moving"])
          for a in analyses}
    results["criteria"]["SA6_wall_following"] = {
        "tracked_vs_gt": {k: [round(v[0], 4), round(v[1], 4)]
                          for k, v in wf.items()},
        "pass": all(v[0] is not None and
                    abs(v[0] - v[1]) <= sa["SA6_wall_follow_abs_tol"]
                    for v in wf.values()),
    }
    # SA7 mean resultant length
    rl = {a["label"]: (a["direction"]["mean_resultant_length"],
                       gt_stats[a["label"]]["resultant"])
          for a in analyses}
    results["criteria"]["SA7_resultant_length"] = {
        "tracked_vs_gt": {k: [round(v[0], 4), round(v[1], 4)]
                          for k, v in rl.items()},
        "pass": all(v[0] is not None and v[1] is not None and
                    abs(v[0] - v[1]) <= sa["SA7_resultant_len_abs_tol"]
                    for v in rl.values()),
    }
    # SA8 stimulus detection / baseline false positives
    stim_labels = [a["label"] for a in analyses if a["kind"] == "stimulus"]
    base_labels = [a["label"] for a in analyses if a["kind"] == "baseline"]
    stim_by = {a["label"]: a for a in analyses}
    n_stim_det = sum(1 for l in stim_labels
                     if stim_by[l]["stimulus"] and stim_by[l]["stimulus"]["responds"])
    n_base_det = sum(1 for l in base_labels
                     if stim_by[l]["stimulus"] and stim_by[l]["stimulus"]["responds"])
    results["criteria"]["SA8_stimulus"] = {
        "p_values": {a["label"]: (a["stimulus"]["permutation_p"]
                                  if a["stimulus"] else None) for a in analyses},
        "n_stimulus_detected": n_stim_det,
        "n_baseline_false_positive": n_base_det,
        "pass": bool(n_stim_det >= int(sa["SA8_stimulus_sessions_detect_min"]) and
                     n_base_det <= len(base_labels) -
                     int(sa["SA8_baseline_no_detect_min"])),
    }
    # SA9 vocabulary selection
    from ..vocabulary.selection import evaluate_ladder
    windows = {a["label"]: a["windows"] for a in analyses if a["kind"] != "blank"}
    ladder = evaluate_ladder(windows)
    results["criteria"]["SA9_vocabulary"] = {
        "selected": ladder["selected_level"],
        "expected": sa["SA9_expected_level"],
        "pass": ladder["selected_level"] == sa["SA9_expected_level"],
    }

    # SA10 gate machinery + negative control
    # GT-derived annotations for 3 baseline sessions
    for a in analyses:
        if a["kind"] == "baseline":
            annotate_from_gt(Path(a["session_dir"]), n=50, seed=20260927)
    from ..gate.evaluate import annotation_errors_mm
    ann_rows = []
    for a in analyses:
        p = Path(a["session_dir"]) / "annotations.csv"
        if p.exists():
            arr = np.genfromtxt(p, delimiter=",", names=True, dtype=None,
                                encoding="utf-8")
            if arr.ndim == 0:
                arr = arr.reshape(1)
            for r in arr:
                ann_rows.append((a["label"], float(r["frame"]),
                                 float(r["x_px"]), float(r["y_px"])))
    calibs = {a["label"]: calibration_from_session(
        json.loads((Path(a["session_dir"]) / "session.json")
                   .read_text(encoding="utf-8"))) for a in analyses}
    errs = annotation_errors_mm(
        ann_rows, {a["label"]: Path(a["session_dir"]) for a in analyses}, calibs)
    base_analyses = [a for a in analyses if a["kind"] == "baseline"]
    gate_report = evaluate_gate(base_analyses, annotations=errs,
                                gate_cfg=gate_cfg, subject="synthetic")
    gate_clean_pass = gate_report["passed"]
    # negative control: 20% seeded frame drops in a COPY of session 1
    nc_dir = out_dir / "negative_control"
    nc_dir.mkdir(parents=True, exist_ok=True)
    src = Path(base_analyses[0]["session_dir"])
    import shutil
    for f in ("tracks.csv", "session.json"):
        shutil.copy(src / f, nc_dir / f)
    nc_an = analyze_session(nc_dir, consts, tracking_cfg=tcfg,
                            corrupt_drop_fraction=float(
                                sa["SA10_negative_control_drop_fraction"]))
    nc_report = evaluate_gate([nc_an], gate_cfg=gate_cfg,
                              subject="negative-control")
    nc_g1_fail = (not nc_report["criteria"]["G1_tracking_coverage"]["pass"]) and \
                 nc_report["criteria"]["G1_tracking_coverage"]["diagnosis"] is not None
    results["criteria"]["SA10_gate_machinery"] = {
        "clean_gate_pass": gate_clean_pass,
        "negative_control_G1_failed_with_diagnosis": nc_g1_fail,
        "pass": bool(gate_clean_pass and nc_g1_fail),
        "clean_gate_summary": {k: v["pass"] for k, v in
                               gate_report["criteria"].items()},
    }

    # ------------------------------------------------------------- summary
    results["gate_on_synthetic"] = {
        "G1": gate_report["criteria"]["G1_tracking_coverage"],
        "G2": gate_report["criteria"]["G2_tracking_accuracy"],
        "G3": gate_report["criteria"]["G3_state_occupancy"],
        "G4": gate_report["criteria"]["G4_decoder_reliability"],
        "vocabulary": gate_report["vocabulary"],
    }
    results["cross_session_summary"] = summarize_cross_session(
        analyses, ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "PAUSE"])
    # drop non-serializable window objects before dumping
    for a in analyses:
        a.pop("windows", None)
    results["session_analyses"] = analyses
    results["overall_pass"] = all(
        c["pass"] for c in results["criteria"].values())

    (out_dir / "software_validation_report.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out_dir / "software_validation_report.txt").write_text(
        _render(results), encoding="utf-8")
    return results


def _render(r: dict) -> str:
    lines = ["=" * 74,
             "PHASE-2 SOFTWARE VALIDATION (synthetic ground truth — CODE ONLY,",
             "NOT evidence about real flies)",
             "=" * 74]
    for k, v in r["criteria"].items():
        lines.append(f"[{'PASS' if v['pass'] else 'FAIL'}] {k}")
    lines.append("-" * 74)
    verdict = ("ALL SOFTWARE ACCEPTANCE CRITERIA PASSED"
               if r["overall_pass"] else "VALIDATION FAILED")
    lines.append(f"OVERALL: {verdict}")
    lines.append("=" * 74)
    return "\n".join(lines)
