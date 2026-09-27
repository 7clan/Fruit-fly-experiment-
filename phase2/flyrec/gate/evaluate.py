"""PHASE2-GATE-1.0.0 evaluator (pre-registered; see config/gate.yaml).

Consumes:
  - per-session analyses (flyrec.analysis.statistics.analyze_session)
  - vocabulary ladder evidence (flyrec.vocabulary.selection.evaluate_ladder)
  - manual annotations (frame, x_px, y_px) per session [for G2]

Produces gate_report.json + gate_report.txt with PASS/FAIL per criterion,
the overall verdict, and the pre-registered diagnosis for each failure.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..config import load_gate_cfg
from ..vocabulary.selection import evaluate_ladder
from ..vocabulary.states import STATES_A


def load_annotations(paths: list[str | Path]) -> list[tuple[str, float, float, float]]:
    """annotations.csv rows: session_label, frame, x_px, y_px."""
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        arr = np.genfromtxt(p, delimiter=",", names=True, dtype=None,
                            encoding="utf-8")
        if arr.ndim == 0:
            arr = arr.reshape(1)
        for r in arr:
            rows.append((str(r["session"]), float(r["frame"]),
                         float(r["x_px"]), float(r["y_px"])))
    return rows


def annotation_errors_mm(annotations, session_dirs: dict[str, Path],
                         calibs: dict[str, object]) -> list[float]:
    """Error (mm) between each annotation and the tracked position."""
    from ..tracking.tracker import load_tracks
    errs = []
    cache = {}
    for (label, frame, ax, ay) in annotations:
        if label not in session_dirs:
            continue
        if label not in cache:
            cache[label] = load_tracks(session_dirs[label] / "tracks.csv")
        tr = cache[label]
        i = int(round(frame))
        if i >= len(tr["frame"]) or not tr["found"][i]:
            continue
        tx, ty = tr["x_px"][i], tr["y_px"][i]
        cal = calibs.get(label)
        d_px = math.hypot(ax - tx, ay - ty)
        if cal is not None:
            errs.append(d_px * cal.mm_per_px)
        else:
            errs.append(d_px)
    return errs


def evaluate_gate(session_analyses: list[dict], annotations=None,
                  gate_cfg: dict | None = None, out_dir: Path | None = None,
                  subject: str = "real-fly") -> dict:
    """Evaluate G1-G4 (+G5 secondary). session_analyses from analyze_session."""
    g = gate_cfg or load_gate_cfg()
    c = g["criteria"]
    ac = g["analysis_constants"]
    req = g["data_requirements"]

    report: dict = {
        "gate_id": g["gate_id"],
        "subject": subject,
        "n_sessions": len(session_analyses),
        "criteria": {},
        "passed": None,
        "notes": [],
    }

    # ------------------------------------------------------------------ G1
    covs = {a["label"]: a["tracking"]["coverage_conf05"] for a in session_analyses}
    min_cov = float(c["G1_tracking_coverage"]["min_coverage"])
    g1_pass = bool(session_analyses) and all(v >= min_cov for v in covs.values())
    report["criteria"]["G1_tracking_coverage"] = {
        "requirement": f"coverage(conf>=0.5) >= {min_cov} in EVERY session",
        "observed": covs, "min_observed": min(covs.values()) if covs else None,
        "pass": g1_pass,
        "diagnosis": None if g1_pass else c["G1_tracking_coverage"]["diagnosis"],
    }

    # ------------------------------------------------------------------ G2
    g2 = report["criteria"]["G2_tracking_accuracy"] = {
        "requirement": f"median annotation error <= "
                       f"{c['G2_tracking_accuracy']['median_error_max_mm']} mm "
                       f"over >= {req['annotation_frames_min']} frames",
        "n_annotated": 0, "median_error_mm": None, "pass": False,
        "diagnosis": c["G2_tracking_accuracy"]["diagnosis"],
    }
    if annotations:
        errs = annotations["errors_mm"] if isinstance(annotations, dict) \
            else annotations
        g2["n_annotated"] = len(errs)
        if errs:
            g2["median_error_mm"] = float(np.median(errs))
        g2["pass"] = bool(len(errs) >= req["annotation_frames_min"] and
                          g2["median_error_mm"] is not None and
                          g2["median_error_mm"] <=
                          float(c["G2_tracking_accuracy"]["median_error_max_mm"]))
        if g2["pass"]:
            g2["diagnosis"] = None
    else:
        g2["pass"] = False
        g2["note"] = "no annotations provided — criterion NOT evaluated " \
                     "(fails closed until annotation is done)"

    # ------------------------------------------------- vocabulary ladder G3/G4
    per_session_windows = {a["label"]: a["windows"] for a in session_analyses}
    ladder = evaluate_ladder(
        per_session_windows,
        occupancy_min=float(c["G3_state_occupancy"]["min_window_fraction_per_session"]),
        stability_band=tuple(c["G3_state_occupancy"]["stability_band"]),
        loso_min=float(c["G4_decoder_reliability"]["loso_accuracy_min"]))
    selected = ladder["selected_level"]
    sel_ev = next((e for e in ladder["evidence"] if e["level"] == selected), None)
    report["vocabulary"] = {
        "selected_level": selected,
        "n_states": ladder["n_states_selected"],
        "states": sel_ev["states"] if sel_ev else [],
        "construction": ladder["construction"],
        "directional_control_possible": ladder["directional_control_possible"],
        "minimum_for_directional_control":
            g["vocabulary_ladder"]["minimum_for_directional_control"],
    }
    report["criteria"]["G3_state_occupancy"] = {
        "requirement": "every selected-vocabulary state occupied >= "
                       f"{c['G3_state_occupancy']['min_window_fraction_per_session']}"
                       " in every session, within "
                       f"{c['G3_state_occupancy']['stability_band']} stability band",
        "pass": bool(sel_ev["G3_pass"]) if sel_ev else False,
        "diagnosis": None if (sel_ev and sel_ev["G3_pass"]) else
        c["G3_state_occupancy"]["diagnosis_occupancy"],
    }
    report["criteria"]["G4_decoder_reliability"] = {
        "requirement": f"LOSO nearest-centroid accuracy >= "
                       f"{c['G4_decoder_reliability']['loso_accuracy_min']}",
        "pass": bool(sel_ev["G4_pass"]) if sel_ev else False,
        "loso_accuracy": sel_ev["G4_loso"].get("accuracy") if sel_ev else None,
        "loso_chance": sel_ev["G4_loso"].get("chance") if sel_ev else None,
        "diagnosis": None if (sel_ev and sel_ev["G4_pass"]) else
        c["G4_decoder_reliability"]["diagnosis"],
    }

    # ------------------------------------------------------------------ G5
    g5cfg = g["secondary_G5_stimulus_response"]
    stim_sessions = [a for a in session_analyses if a.get("stimulus")]
    responders = [a["label"] for a in stim_sessions if a["stimulus"]["responds"]]
    g5_pass = len(responders) >= int(g5cfg["sessions_responding_required"])
    report["criteria"]["G5_stimulus_response_secondary"] = {
        "requirement": f"permutation p < {g5cfg['alpha']} in >= "
                       f"{g5cfg['sessions_responding_required']} stimulus "
                       "sessions (SECONDARY — informative, not gating)",
        "observed": {a["label"]: (a["stimulus"]["permutation_p"]
                                  if a["stimulus"] else None)
                     for a in session_analyses},
        "n_stimulus_sessions": len(stim_sessions),
        "responders": responders,
        "pass": g5_pass,
        "gating": False,
        "diagnosis": None if g5_pass else g5cfg["diagnosis"],
    }

    # ---------------------------------------------------------------- total
    gating = ["G1_tracking_coverage", "G2_tracking_accuracy",
              "G3_state_occupancy", "G4_decoder_reliability"]
    report["passed"] = all(report["criteria"][k]["pass"] for k in gating)
    report["gating_criteria"] = gating

    # data-requirement warnings (do not auto-fail; reported honestly)
    if len(session_analyses) < req["min_baseline_sessions"]:
        report["notes"].append(
            f"only {len(session_analyses)} sessions provided "
            f"(pre-registered minimum {req['min_baseline_sessions']})")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gate_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        (out_dir / "gate_report.txt").write_text(render_text(report),
                                                 encoding="utf-8")
        (out_dir / "vocabulary_evidence.json").write_text(
            json.dumps({k: v for k, v in ladder.items()}, indent=2,
                       default=str), encoding="utf-8")
    report["ladder_evidence"] = ladder
    return report


def render_text(r: dict) -> str:
    lines = [f"{'=' * 74}",
             f"PHASE-2 GATE REPORT — {r['gate_id']}  (subject: {r['subject']})",
             f"{'=' * 74}",
             f"sessions analyzed : {r['n_sessions']}",
             f"vocabulary        : {r['vocabulary']['selected_level']} "
             f"({r['vocabulary']['n_states']} states: "
             f"{', '.join(r['vocabulary']['states'])})",
             f"directional control possible "
             f"(>= {r['vocabulary']['minimum_for_directional_control']}): "
             f"{r['vocabulary']['directional_control_possible']}",
             "-" * 74]
    for k, v in r["criteria"].items():
        tag = "PASS" if v["pass"] else "FAIL"
        lines.append(f"[{tag}] {k}")
        lines.append(f"        {v['requirement']}")
        if v.get("diagnosis"):
            lines.append(f"        diagnosis: {v['diagnosis']}")
    for n in r.get("notes", []):
        lines.append(f"note: {n}")
    lines.append("-" * 74)
    lines.append(f"OVERALL (G1-G4, all must pass): "
                 f"{'PASSED' if r['passed'] else 'FAILED'}")
    lines.append(f"(G5 is secondary/informative and does not gate.)")
    lines.append("=" * 74)
    return "\n".join(lines)
