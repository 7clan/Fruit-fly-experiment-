"""Session INTAKE — the fixed verification step that runs FIRST when
recordings come back, BEFORE any tracking/analysis/gate evaluation and
BEFORE any discussion of modifications (protocol-fixed order):

    1. verify file integrity
    2. verify protocol compliance (pre-registered data requirements)
    3. (only then, outside this module) track -> annotate -> analyze -> gate

This module deliberately performs NO behavioral analysis and computes NO
gate criterion, so that PASS/FAIL reporting cannot be influenced by looking
at the animal's behavior first.

Writes results/intake_<ts>/intake_manifest.json (SHA-256 of every raw and
derived file — the tamper-evident snapshot for later copies) plus a
human-readable report. Verdict: PASS | PENDING (steps missing, none failed)
| FAIL.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from ..config import RESULTS_DIR, load_checks_cfg, load_gate_cfg
from ..recording.recorder import make_stimulus_schedule
from ..rig import sha256_file

WELFARE_MAX_SESSION_MIN = 30.0


def _find_session_dirs(args: list[Path]) -> list[Path]:
    dirs = []
    for a in args:
        a = Path(a)
        if a.is_dir() and next(a.glob("video.*"), None) is not None:
            dirs.append(a)
        elif a.is_dir():                       # a root: scan one level down
            for d in sorted(a.iterdir()):
                if d.is_dir() and next(d.glob("video.*"), None) is not None:
                    dirs.append(d)
    # de-duplicate, keep order
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _session_integrity(sdir: Path, checks_cfg: dict) -> dict:
    rep: dict = {"session_dir": str(sdir), "problems": [], "status": "OK"}
    meta = None
    sj = sdir / "session.json"
    if not sj.exists():
        rep["problems"].append("session.json missing (recorder crashed or "
                               "directory is not a session)")
        rep["status"] = "FAIL"
        return rep
    try:
        meta = json.loads(sj.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        rep["problems"].append(f"session.json not parseable: {e}")
        rep["status"] = "FAIL"
        return rep
    rep["meta"] = {k: meta.get(k) for k in (
        "session_id", "label", "kind", "started_utc", "fps",
        "measured_probe_fps", "duration_s", "fly_id", "fly_age_days",
        "fly_sex", "temperature_c", "humidity_pct", "config_hash",
        "git_revision", "checks_overridden", "operator_notes")}

    video = next(sdir.glob("video.*"), None)
    if video is None:
        rep["problems"].append("no video file")
        rep["status"] = "FAIL"
        return rep
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        rep["problems"].append(f"video not openable: {video.name}")
        rep["status"] = "FAIL"
        return rep
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    fps = float(meta.get("fps") or 0)
    dur = float(meta.get("duration_s") or 0)
    min_completion = float(checks_cfg.get("intake", {})
                           .get("min_frame_completion", 0.90))
    if fps > 0 and dur > 0 and n_frames < min_completion * fps * dur:
        rep["problems"].append(
            f"video has {n_frames} frames < {min_completion:.0%} of the "
            f"expected {fps * dur:.0f} (dropped frames / early stop)")
    if vw and vh and (vw != meta.get("width") or vh != meta.get("height")):
        rep["problems"].append(
            f"video is {vw}x{vh} but session.json says "
            f"{meta.get('width')}x{meta.get('height')}")
    if meta.get("kind") in ("baseline", "stimulus"):
        for field in ("fly_id", "temperature_c", "humidity_pct",
                      "started_utc"):
            if meta.get(field) in (None, ""):
                rep["problems"].append(f"missing session metadata: {field}")
        cal = meta.get("calibration") or {}
        if not cal.get("px_per_mm") or not meta.get("arena"):
            rep["problems"].append("missing calibration (px_per_mm/arena)")
        if not meta.get("config_hashes"):
            rep["problems"].append("missing provenance (config_hashes)")
    if dur > (WELFARE_MAX_SESSION_MIN * 60.0 + 60.0):
        rep["problems"].append(
            f"session length {dur / 60:.1f} min exceeds the 30-min welfare "
            "cap (protocol section 2)")
    if rep["problems"]:
        rep["status"] = "FAIL"
    rep["video_frames"] = n_frames
    rep["tracks_csv"] = (sdir / "tracks.csv").exists()
    rep["annotations_csv"] = (sdir / "annotations.csv").exists()
    return rep


def _compliance(reports: list[dict], sessions_root: Path,
                gate_cfg: dict) -> dict:
    req = gate_cfg.get("data_requirements", {})
    metas = [r.get("meta") or {} for r in reports
             if r.get("status") != "FAIL"]
    animal = [m for m in metas if m.get("kind") in ("baseline", "stimulus")]

    def good(m) -> bool:
        fps = float(m.get("fps") or 0)
        dur = float(m.get("duration_s") or 0)
        return fps >= float(req.get("min_fps", 30)) and \
            dur >= float(req.get("min_session_minutes", 10)) * 60.0

    baselines = [m for m in animal if m.get("kind") == "baseline" and good(m)]
    stimuli = [m for m in animal if m.get("kind") == "stimulus" and good(m)]
    days = sorted({str(m.get("started_utc", ""))[:8] for m in animal})

    comp: dict = {}
    comp["baseline_sessions"] = {
        "required": int(req.get("min_baseline_sessions", 3)),
        "observed": len(baselines),
        "status": "PASS" if len(baselines) >=
        int(req.get("min_baseline_sessions", 3)) else "PENDING",
        "detail": f"{len(baselines)} baseline sessions >= "
                  f"{req.get('min_session_minutes', 10)} min at >= "
                  f"{req.get('min_fps', 30)} fps"}
    comp["stimulus_sessions"] = {
        "required": int(req.get("min_stimulus_sessions", 3)),
        "observed": len(stimuli),
        "status": "PASS" if len(stimuli) >=
        int(req.get("min_stimulus_sessions", 3)) else "PENDING"}
    comp["recording_days"] = {
        "required": int(req.get("min_recording_days", 2)),
        "observed": len(days), "days": days,
        "status": "PASS" if len(days) >=
        int(req.get("min_recording_days", 2)) else "PENDING"}

    # annotations (for G2) — PENDING until the annotation step is run
    n_ann = 0
    n_ann_sessions = 0
    for r in reports:
        p = Path(r["session_dir"])
        ap = p / "annotations.csv"
        if ap.exists():
            try:
                arr = np.genfromtxt(ap, delimiter=",", names=True,
                                    dtype=None, encoding="utf-8")
                n_ann += int(arr.size) if arr.size else 0
                n_ann_sessions += 1
            except Exception:
                pass
    comp["manual_annotations"] = {
        "required": int(req.get("annotation_frames_min", 150)),
        "observed": n_ann, "annotated_sessions": n_ann_sessions,
        "status": ("PASS" if n_ann >= int(req.get("annotation_frames_min", 150))
                   else "PENDING"),
        "detail": "run `annotate` on 3 sessions (50 frames each) if PENDING"}

    # stimulus schedules must equal the seeded pre-registered schedule
    sched_ok, sched_bad = [], []
    for m in stimuli:
        seed = m.get("stimulus_seed")
        sched = m.get("stimulus_schedule")
        if seed is None or not isinstance(sched, list):
            sched_bad.append(m.get("label"))
            continue
        expected = make_stimulus_schedule(int(seed) - 20260927)
        if sched != expected:
            sched_bad.append(m.get("label"))
        else:
            sched_ok.append(m.get("label"))
    comp["stimulus_schedule_matches_preregistration"] = {
        "ok": sched_ok, "mismatched": sched_bad,
        "status": "PASS" if stimuli and not sched_bad else
        ("PENDING" if not stimuli else "FAIL")}

    # provenance consistency
    hashes = {m.get("config_hash") for m in animal}
    comp["config_hash_consistency"] = {
        "observed": sorted(str(h) for h in hashes),
        "status": "PASS" if len(hashes) <= 1 else "DEVIATION",
        "detail": "sessions were recorded under different code/config "
                  "states" if len(hashes) > 1 else ""}
    overridden = [m.get("label") for m in animal if m.get("checks_overridden")]
    comp["no_overridden_apparatus_checks"] = {
        "overridden": overridden,
        "status": "PASS" if not overridden else "DEVIATION",
        "detail": "sessions recorded with --force against the apparatus gate"
                  if overridden else ""}
    static_fallback = [m.get("label") for m in animal
                       if (m.get("tracking") or {}).get("mode") != "reference"]
    comp["reference_background_tracking"] = {
        "sessions_not_using_reference_mode": static_fallback,
        "status": "PASS" if not static_fallback else
        ("DEVIATION" if animal else "PENDING"),
        "detail": "baseline/stimulus sessions should be tracked against the "
                  "blank-derived reference background (Amendment 2)"
                  if static_fallback else ""}
    return comp


def intake(session_args: list, out_dir: Path | None = None) -> dict:
    checks_cfg = load_checks_cfg()
    gate_cfg = load_gate_cfg()
    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR / f"intake_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = _find_session_dirs([Path(p) for p in session_args])
    if not dirs:
        return {"verdict": "FAIL", "error": "no session directories found "
                                            "in the given paths"}

    reports = []
    manifest = {}
    for d in dirs:
        rep = _session_integrity(d, checks_cfg)
        files = {}
        for f in sorted(d.iterdir()):
            if f.is_file():
                files[f.name] = sha256_file(f)
        manifest[d.name] = files
        rep["file_hashes"] = files
        reports.append(rep)

    sessions_root = dirs[0].parent
    compliance = _compliance(reports, sessions_root, gate_cfg)

    statuses = [c["status"] for c in compliance.values()]
    verdict = ("FAIL" if any(r["status"] == "FAIL" for r in reports) or
               "FAIL" in statuses else
               "PENDING" if "PENDING" in statuses or
               any(not r.get("tracks_csv") for r in reports) else
               "DEVIATION" if "DEVIATION" in statuses else "PASS")

    result = {
        "utc": ts,
        "purpose": "integrity + protocol compliance verification only - no "
                   "behavioral analysis, no gate evaluation, no tuning",
        "sessions": [{k: v for k, v in r.items() if k != "file_hashes"}
                     for r in reports],
        "compliance": compliance,
        "verdict": verdict,
    }
    (out_dir / "intake_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "intake_report.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["out_dir"] = str(out_dir)
    result["manifest_path"] = str(out_dir / "intake_manifest.json")
    return result


def render_intake(report: dict) -> str:
    lines = ["=" * 70,
             "SESSION INTAKE - integrity + protocol compliance (no analysis)",
             "=" * 70]
    if "error" in report:
        lines.append(f"ERROR: {report['error']}")
        return "\n".join(lines)
    for r in report["sessions"]:
        status = r.get("status", "?")
        meta = r.get("meta") or {}
        lines.append(f"[{status}] {Path(r['session_dir']).name} "
                     f"({meta.get('kind', '?')}, "
                     f"{r.get('video_frames', '?')} frames, "
                     f"fps={meta.get('fps', '?')})")
        for p in r.get("problems", []):
            lines.append(f"    - {p}")
        if not r.get("tracks_csv"):
            lines.append("    - tracks.csv not yet generated (run `track`)")
    lines.append("-" * 70)
    for k, c in report["compliance"].items():
        lines.append(f"[{c['status']}] {k}: {c.get('detail', '')}"
                     .rstrip())
    lines.append("-" * 70)
    lines.append(f"VERDICT: {report['verdict']}")
    lines.append("=" * 70)
    return "\n".join(lines)
