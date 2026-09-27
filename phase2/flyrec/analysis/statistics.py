"""Master behavioral statistics for one session (and cross-session summary).

Computes everything PHASE-2 requires to measure:
  tracking percentage & confidence, movement speed, direction distribution,
  pause frequency, movement-bout duration, directional persistence,
  wall-following behavior, stimulus response, trial-to-trial variability
  (cross-session, see analysis_variability), action-state occupancy.
"""
from __future__ import annotations

import math

import numpy as np

from ..tracking.calibration import Calibration
from ..tracking.tracker import load_tracks
from .behaviors import (bout_metrics, find_bouts, find_pauses, wall_metrics)
from .circular import (circular_std, heading_autocorr, heading_histogram,
                       mean_resultant)
from .stimulus import permutation_test, session_response
from .trajectory import compute_kinematics, interpolate_gaps
from . import stimulus as stimulus_mod
from ..vocabulary.states import STATES_A, build_windows, occupancy


def analyze_session(session_dir, gate_constants: dict,
                    calibration: Calibration | None = None,
                    tracking_cfg: dict | None = None,
                    stimulus_events: list | None = None,
                    session_meta: dict | None = None,
                    corrupt_drop_fraction: float = 0.0,
                    seed: int = 20260927) -> dict:
    """Full analysis of one session directory (tracks.csv + session.json).

    corrupt_drop_fraction: SA10 negative control — randomly mark this
    fraction of detections as lost BEFORE analysis (seeded, deterministic).
    """
    from ..tracking.calibration import calibration_from_session
    import json
    from pathlib import Path

    sdir = Path(session_dir)
    tracks = load_tracks(sdir / "tracks.csv")
    meta = {}
    if (sdir / "session.json").exists():
        meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
    if session_meta:
        meta.update(session_meta)
    calib = calibration or calibration_from_session(meta)

    # optional negative-control corruption (SA10)
    found = tracks["found"].copy()
    if corrupt_drop_fraction > 0:
        rng = np.random.default_rng(seed)
        idx = np.where(found)[0]
        drop = rng.choice(idx, size=int(round(corrupt_drop_fraction * len(idx))),
                          replace=False)
        found[drop] = False
        for i in drop:
            tracks["x_mm"][i] = np.nan
            tracks["y_mm"][i] = np.nan
        tracks["found"] = found
    conf = np.where(found, tracks["conf"], 0.0)

    # ---- trim to the subject's first appearance ----------------------------
    # The recording protocol starts BEFORE the fly is introduced (the static
    # background needs fly-free lead-in frames). Frames before the first
    # detection are not "lost tracking" and must not count against coverage
    # or create a phantom leading pause. The lead-in duration is reported.
    first = int(np.argmax(found)) if found.any() else 0
    if not found.any():
        first = 0
    lead_in_frames = first
    lead_in_s = float(tracks["t"][first]) if first else 0.0
    t = tracks["t"][first:]
    x_raw, y_raw = tracks["x_mm"][first:], tracks["y_mm"][first:]
    found_s = found[first:]
    conf_s = conf[first:]

    # ---- tracking quality (over subject-present analysis frames) ----------
    n_frames = len(t)
    coverage = float(found_s.mean()) if n_frames else 0.0
    coverage_conf05 = float((conf_s >= 0.5).mean()) if n_frames else 0.0

    # ---- trajectory
    x, y, interp = interpolate_gaps(
        t, x_raw, y_raw, found_s,
        max_gap_s=gate_constants.get("max_gap_s", 0.5))
    kin = compute_kinematics(
        t, x, y,
        median_k=int(gate_constants.get("median_filter_frames", 5)),
        vel_span=int(gate_constants.get("velocity_span_frames", 3)))
    valid = found_s | interp
    valid = valid & np.isfinite(x) & np.isfinite(y)
    speed, heading = kin["speed"], kin["heading"]

    # ---- movement / bouts / pauses
    on = float(gate_constants.get("bout_on_speed_mm_s", 3.0))
    off = float(gate_constants.get("bout_off_speed_mm_s", 2.0))
    bouts = find_bouts(t, speed, valid, on=on, off=off,
                       min_dur=float(gate_constants.get("min_bout_s", 0.25)),
                       gap_merge=float(gate_constants.get("bout_gap_merge_s", 0.2)))
    pauses = find_pauses(t, bouts,
                         min_pause=float(gate_constants.get("min_pause_s", 0.30)),
                         t_end=float(t[-1]) if n_frames else 0.0)
    bmets = bout_metrics(bouts, t, kin["x"], kin["y"], heading, valid)
    dur_min = (t[-1] / 60.0) if n_frames and t[-1] > 0 else 0.0

    sp = speed[valid & np.isfinite(speed)]
    mov = sp[sp >= 2.0]
    movement = {
        "speed_mean_mm_s": float(sp.mean()) if len(sp) else None,
        "speed_median_mm_s": float(np.median(sp)) if len(sp) else None,
        "speed_p90_mm_s": float(np.percentile(sp, 90)) if len(sp) else None,
        "frac_frames_moving": float(len(mov) / max(1, len(sp))),
        "moving_speed_mean_mm_s": float(mov.mean()) if len(mov) else None,
    }
    bout_durs = [b["duration_s"] for b in bmets]
    pause_durs = [p["duration_s"] for p in pauses]
    bouts_stats = {
        "n_bouts": len(bmets),
        "rate_per_min": len(bmets) / dur_min if dur_min else None,
        "median_duration_s": float(np.median(bout_durs)) if bout_durs else None,
        "mean_duration_s": float(np.mean(bout_durs)) if bout_durs else None,
        "median_straightness": float(np.median(
            [b["straightness"] for b in bmets])) if bmets else None,
    }
    pauses_stats = {
        "n_pauses": len(pauses),
        "rate_per_min": len(pauses) / dur_min if dur_min else None,
        "median_duration_s": float(np.median(pause_durs)) if pause_durs else None,
    }

    # ---- direction distribution + persistence
    # heading is UNDEFINED while the animal stands still: all heading
    # statistics use MOVING frames only (speed >= 2 mm/s), the same
    # convention as the stimulus-response module.
    mov_mask = valid & np.isfinite(heading) & np.isfinite(speed) & (speed >= 2.0)
    h = heading[mov_mask]
    mean_ang, R = mean_resultant(h)
    counts, edges = heading_histogram(h, bins=16)
    direction = {
        "mean_angle_deg": math.degrees(mean_ang) if math.isfinite(mean_ang) else None,
        "mean_resultant_length": R,
        "circular_std_deg": circular_std(R) if R > 0 else None,
        "histogram_16": counts,
        "n_moving_frames": int(mov_mask.sum()),
    }
    persistence = {
        "mean_resultant_length": R,
        "heading_autocorr_1s": heading_autocorr(heading, t, 1.0, speed=speed),
        "median_bout_straightness": bouts_stats["median_straightness"],
    }

    # ---- wall interactions
    wstats, wall_dist = wall_metrics(
        t, kin["x"], kin["y"], heading, valid, calib,
        wall_band_mm=float(gate_constants.get("wall_band_mm", 10.0)),
        parallel_deg=float(gate_constants.get("wall_parallel_angle_deg", 45.0)),
        moving_speed=speed)

    # ---- stimulus response (if events provided or read from tracks)
    if stimulus_events is None:
        from ..tracking.tracker import stimulus_events_from_tracks
        stimulus_events = stimulus_events_from_tracks(tracks, 30.0)
        # only events after the subject's first appearance are meaningful
        if lead_in_s > 0:
            stimulus_events = [e for e in stimulus_events
                               if e["t_on"] >= lead_in_s - 0.1]
    stim = None
    if stimulus_events:
        resp = session_response(heading, speed, t, stimulus_events)
        perm = permutation_test(
            heading, speed, t, stimulus_events,
            iterations=int(gate_constants.get("permutation_iterations", 1000)),
            seed=int(gate_constants.get("permutation_seed", 20260927)))
        stim = {
            "n_events": len(stimulus_events),
            "events": stimulus_events,
            "mean_response": resp["mean_response"],
            "permutation_p": perm["p_value"],
            "responds": bool(np.isfinite(perm["p_value"]) and perm["p_value"] < 0.05),
        }

    # ---- action-state windows (L_A labels; ladder re-labels later)
    windows = build_windows(
        t, kin["x"], kin["y"], speed, valid,
        window_s=float(gate_constants.get("window_s", 1.0)),
        pause_speed=float(gate_constants.get("pause_speed_mm_s", 2.0)))
    occ = occupancy(windows, STATES_A)

    return {
        "session_dir": str(sdir),
        "label": meta.get("label", sdir.name),
        "kind": meta.get("kind", "baseline"),
        "fps": meta.get("fps"),
        "duration_s": float(t[-1]) if n_frames else 0.0,
        "tracking": {
            "n_frames": int(n_frames),
            "n_frames_total_incl_lead_in": int(len(tracks["t"])),
            "lead_in_frames": int(lead_in_frames),
            "lead_in_s": round(lead_in_s, 3),
            "n_found": int(found_s.sum()),
            "coverage": coverage,
            "coverage_conf05": coverage_conf05,
            "mean_confidence": float(np.mean(conf_s[found_s])) if found_s.any() else None,
            "frac_interpolated": float(interp.mean()) if n_frames else 0.0,
        },
        "movement": movement,
        "bouts": bouts_stats,
        "pauses": pauses_stats,
        "direction": direction,
        "persistence": persistence,
        "wall": wstats,
        "stimulus": stim,
        "windows": windows,          # consumed by vocabulary/gate (not JSON-dumped)
        "state_occupancy_L_A": occ,
    }


def summarize_cross_session(session_analyses: list[dict],
                            variability_states: list[str]) -> dict:
    """Cross-session summary incl. variability of state occupancy."""
    from .variability import session_variability
    occ_by_session = {a["label"]: a["state_occupancy_L_A"]
                      for a in session_analyses}
    var = session_variability(occ_by_session, variability_states)
    def agg(key_path):
        vals = []
        for a in session_analyses:
            v = a
            for k in key_path:
                v = v[k] if isinstance(v, dict) and v is not None else None
            if isinstance(v, (int, float)) and v is not None:
                vals.append(float(v))
        return {"mean": float(np.mean(vals)) if vals else None,
                "min": float(np.min(vals)) if vals else None,
                "max": float(np.max(vals)) if vals else None,
                "n": len(vals)}
    return {
        "n_sessions": len(session_analyses),
        "coverage_conf05": agg(["tracking", "coverage_conf05"]),
        "speed_mean_mm_s": agg(["movement", "speed_mean_mm_s"]),
        "pause_rate_per_min": agg(["pauses", "rate_per_min"]),
        "bout_median_duration_s": agg(["bouts", "median_duration_s"]),
        "mean_resultant_length": agg(["direction", "mean_resultant_length"]),
        "wall_following_fraction": agg(["wall", "frac_moving_wall_following"]),
        "state_variability": var,
    }
