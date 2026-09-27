"""Phase-2 apparatus-readiness tests (offline; no camera, no fly).

Covers the code added for the physical experiment (PHASE2-APPARATUS-1.0.0 /
Amendment 2): rig provenance + gating, calibration math, reference-mode
detection, blank-arena analysis (clean + injected defects), per-frame
export, intake verification, and immutable backup.

Synthetic videos here are SOFTWARE TEST PATTERNS - they validate code
paths only and are NOT evidence about real Drosophila behavior.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PHASE2 = Path(__file__).resolve().parent.parent / "phase2"
sys.path.insert(0, str(PHASE2))

from flyrec import rig  # noqa: E402
from flyrec.config import load_checks_cfg, load_tracking_cfg  # noqa: E402
from flyrec.tracking.calibration import calibration_from_session  # noqa: E402
from flyrec.tracking.tracker import track_video  # noqa: E402
from flyrec.validation import synthetic_video as sv  # noqa: E402
from flyrec.validation.apparatus import analyze_blank, geometry_report  # noqa: E402
from flyrec.validation.calibrate_tool import (calibration_checks,  # noqa: E402
                                              fit_circle,
                                              px_per_mm_from_pair,
                                              scale_anisotropy)
from flyrec.analysis.statistics import analyze_session  # noqa: E402
from flyrec.config import analysis_constants, load_gate_cfg  # noqa: E402


# ------------------------------------------------------------------ fixtures
@pytest.fixture()
def rig_paths(tmp_path, monkeypatch):
    """Redirect all rig state paths to a temp dir."""
    rig_dir = tmp_path / "rig"
    monkeypatch.setattr(rig, "RIG_DIR", rig_dir)
    monkeypatch.setattr(rig, "STATE_FILE", rig_dir / "rig_state.json")
    monkeypatch.setattr(rig, "CALIBRATION_FILE", rig_dir / "calibration.json")
    monkeypatch.setattr(rig, "BACKGROUND_FILE", rig_dir / "background.png")
    monkeypatch.setattr(rig, "SESSIONS_DIR", tmp_path / "sessions")
    return rig_dir


# --------------------------------------------------------------- provenance
def test_config_hash_deterministic_and_sensitive(tmp_path):
    h1 = rig.composite_config_hash()
    h2 = rig.composite_config_hash()
    assert h1 == h2 and len(h1) == 16
    hashes = rig.config_hashes()
    assert hashes["gate_yaml"] and hashes["tracking_yaml"]
    # a config edit must change the composite hash
    cfg = rig.CONFIG_DIR / "tracking.yaml"
    backup = cfg.read_bytes()
    try:
        cfg.write_bytes(backup + b"\n# sensitivity probe\n")
        assert rig.composite_config_hash() != h1
    finally:
        cfg.write_bytes(backup)
    assert rig.composite_config_hash() == h1


def test_label_rules():
    ok, _ = rig.session_label_ok("baseline_01", None)
    assert ok
    for bad in ("Baseline", "base line", "../etc", "", "x" * 65, "a-b"):
        ok, msg = rig.session_label_ok(bad, None)
        assert not ok, bad


def test_label_uniqueness(tmp_path):
    root = tmp_path / "sessions"
    (root / "20260101T000000Z_baseline_01").mkdir(parents=True)
    ok, msg = rig.session_label_ok("baseline_01", root)
    assert not ok and "already used" in msg
    ok, _ = rig.session_label_ok("baseline_02", root)
    assert ok


def _state_with_passes(rig_paths, kinds, ages_h=0.0):
    """Write a rig state with fresh PASS entries for the given kinds."""
    import datetime
    state = {}
    stamp = (datetime.datetime.now(datetime.timezone.utc) -
             datetime.timedelta(hours=ages_h)).isoformat(timespec="seconds")
    for k in kinds:
        state[k] = {"passed_utc": stamp,
                    "config_hash": rig.composite_config_hash()}
    state["config_hash"] = rig.composite_config_hash()
    rig_paths.mkdir(parents=True, exist_ok=True)
    (rig_paths / "rig_state.json").write_text(json.dumps(state))
    return state


def test_record_gating(rig_paths, tmp_path):
    cfg = load_checks_cfg()
    # no state at all -> refuse everything
    ok, problems = rig.check_record_readiness(10, "baseline", cfg)
    assert not ok
    ok, _ = rig.check_record_readiness(3, "blank", cfg)
    assert not ok
    # calibration only -> still refuse
    _state_with_passes(rig_paths, ["calibration"])
    ok, _ = rig.check_record_readiness(10, "baseline", cfg)
    assert not ok
    # calibration + preflight -> blank allowed, baseline still refused
    _state_with_passes(rig_paths, ["calibration", "preflight"])
    ok, _ = rig.check_record_readiness(3, "blank", cfg)
    assert ok
    ok, _ = rig.check_record_readiness(10, "baseline", cfg)
    assert not ok
    # + blank + background file -> baseline allowed
    _state_with_passes(rig_paths, ["calibration", "preflight", "blank"])
    (rig_paths / "background.png").write_bytes(b"fake")
    ok, _ = rig.check_record_readiness(10, "baseline", cfg)
    assert ok
    ok, _ = rig.check_record_readiness(10, "stimulus", cfg)
    assert ok
    # stale passes (> 24 h) -> refuse again
    _state_with_passes(rig_paths, ["calibration", "preflight", "blank"],
                       ages_h=25.0)
    (rig_paths / "background.png").write_bytes(b"fake")
    ok, _ = rig.check_record_readiness(10, "baseline", cfg)
    assert not ok
    # software validation kinds are never gated
    ok, _ = rig.check_record_readiness(1, "validation", cfg)
    assert ok


# --------------------------------------------------------------- calibration
def test_circle_fit_exact_and_noisy():
    cx, cy, r = 240.0, 180.0, 150.0
    ang = np.linspace(0, 2 * np.pi, 7)[:-1]
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in ang]
    fx, fy, fr, rms = fit_circle(pts)
    assert abs(fx - cx) < 1e-6 and abs(fr - r) < 1e-6 and rms < 1e-6
    rng = np.random.default_rng(0)
    noisy = [(x + rng.normal(0, 1.0), y + rng.normal(0, 1.0)) for x, y in pts]
    _, _, r2, rms2 = fit_circle(noisy)
    assert abs(r2 - r) < 2.0 and rms2 < 2.0


def test_calibration_checks_anisotropy_and_residual():
    ppm_x = px_per_mm_from_pair((100, 100), (430, 100), 50.0)
    assert abs(ppm_x - 6.6) < 0.01
    assert scale_anisotropy(6.6, 6.6) == 0.0
    # ~5% tilt -> must fail
    c = calibration_checks(6.6, 6.95, (240, 180, 150, 0.5))
    assert not c["pass"] and "anisotropy_error" in c
    # sloppy circle clicks -> must fail
    c = calibration_checks(6.6, 6.6, (240, 180, 150, 9.0))
    assert not c["pass"] and "circle_error" in c
    # clean -> pass
    c = calibration_checks(6.6, 6.62, (240, 180, 150, 0.4))
    assert c["pass"]


def test_geometry_report_accepts_and_rejects():
    cfg = load_checks_cfg()
    # 1280x960, arena 620 px across, centered -> OK
    good = {"px_per_mm": 6.5,
            "arena": {"type": "circle", "center_px": [640, 480],
                      "radius_px": 310},
            "physical_inner_diameter_mm": 95.0}
    rep = geometry_report(1280, 960, good, cfg)
    assert rep["pass"], rep
    # 720p vertical: arena 600 px -> rim clearance only 60 px = at the limit,
    # and a 660 px arena -> FAIL
    bad = {"px_per_mm": 6.5,
           "arena": {"type": "circle", "center_px": [640, 360],
                     "radius_px": 330},
           "physical_inner_diameter_mm": 95.0}
    rep = geometry_report(1280, 720, bad, cfg)
    assert not rep["pass"] and "rim" in str(rep.get("rim_clearance_error"))
    # too small in frame
    small = {"px_per_mm": 3.0,
             "arena": {"type": "circle", "center_px": [640, 480],
                       "radius_px": 250},
             "physical_inner_diameter_mm": 165.0}
    rep = geometry_report(1280, 960, small, cfg)
    assert not rep["pass"] and "arena spans" in str(
        rep.get("arena_diameter_error"))
    # physical mismatch: fitted radius 310 px / 6.5 = 47.7 mm vs 47.5 ok;
    # a wrong px_per_mm (8.0) -> 38.75 mm vs 47.5 -> fail
    wrong_scale = {**good, "px_per_mm": 8.0}
    rep = geometry_report(1280, 960, wrong_scale, cfg)
    assert not rep["pass"] and "radius" in str(rep.get("radius_error"))


# ------------------------------------------------- reference-mode detection
def test_reference_mode_tracks_fly_from_frame0(tmp_path):
    """Mini-SA12: no lead-in + background from a fly-free clip."""
    root = tmp_path / "refmode"
    bg_path = sv.generate_reference_background(root / "bg.png",
                                               duration_s=3.0)
    sdir = sv.generate_session(root / "s1", "s1", "baseline", 20260927,
                               duration_s=12.0, lead_in_s=0.0)
    tcfg = load_tracking_cfg()
    det = dict(tcfg["detector"])
    det["mode"] = "reference"
    det["reference_file"] = str(bg_path)
    meta = json.loads((sdir / "session.json").read_text())
    track_video(sdir / "video.mp4", sdir / "tracks.csv",
                {**tcfg, "detector": det},
                calibration_from_session(meta), read_stimulus=False)
    from flyrec.tracking.tracker import load_tracks
    tr = load_tracks(sdir / "tracks.csv")
    gt = np.genfromtxt(sdir / "ground_truth.csv", delimiter=",", names=True,
                       dtype=None, encoding="utf-8")
    present = gt["present"].astype(int) == 1
    covered = (tr["found"] & (tr["conf"] >= 0.5))[present].mean()
    both = present & tr["found"]
    err = np.hypot(tr["x_px"][both] - gt["x_px"][both],
                   tr["y_px"][both] - gt["y_px"][both]) / sv.PX_PER_MM
    assert covered >= 0.98, covered
    assert np.median(err) <= 1.0, np.median(err)


def test_reference_mode_requires_background():
    from flyrec.tracking.detector import FlyDetector
    det = FlyDetector({"mode": "reference"}, px_per_mm=5.0)
    frame = np.full((100, 100, 3), 200, np.uint8)
    with pytest.raises(RuntimeError, match="reference_file"):
        det.detect(frame)


# --------------------------------------------------------- blank-arena test
def _track_static(sdir):
    tcfg = load_tracking_cfg()
    meta = json.loads((sdir / "session.json").read_text())
    track_video(sdir / "video.mp4", sdir / "tracks.csv", tcfg,
                calibration_from_session(meta), read_stimulus=True)


def test_blank_clean_passes(tmp_path):
    sdir = sv.generate_blank_session(tmp_path / "blank", "blank_t",
                                     duration_s=12.0)
    _track_static(sdir)
    rep = analyze_blank(sdir, marker_check=False)
    assert rep["pass"], rep["checks"]
    assert rep["checks"]["B1_confident_false_detections"]["n"] == 0


def _read_all_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.copy())
    cap.release()
    return frames


def _write_frames(path, frames):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), sv.FPS,
                        (sv.W, sv.H))
    for f in frames:
        w.write(f)
    w.release()


def test_blank_transient_blob_fails(tmp_path):
    """A moving dark blob (a 'fly') in some frames must FAIL the blank."""
    sdir = sv.generate_blank_session(tmp_path / "blank", "blank_t",
                                     duration_s=12.0)
    frames = _read_all_frames(sdir / "video.mp4")
    rng = np.random.default_rng(7)
    x = 220
    for i in range(150, 220):                     # ~2.3 s of walking blob
        x += int(rng.integers(2, 6))
        cv2.ellipse(frames[i], (x, 190), (12, 6), 0, 0, 360, (45,) * 3, -1)
    _write_frames(sdir / "video.mp4", frames)
    _track_static(sdir)
    rep = analyze_blank(sdir, marker_check=False)
    assert not rep["pass"]
    assert not rep["checks"]["B1_confident_false_detections"]["pass"]


def test_blank_static_dark_spot_fails_B5(tmp_path):
    """A STATIC smudge hides in the background (B1 passes) but B5 must
    catch it - the exact case the detector cannot see."""
    sdir = sv.generate_blank_session(tmp_path / "blank", "blank_t",
                                     duration_s=12.0)
    frames = _read_all_frames(sdir / "video.mp4")
    for i in range(len(frames)):
        cv2.circle(frames[i], (255, 190), 7, (120,) * 3, -1)   # dark smudge
    _write_frames(sdir / "video.mp4", frames)
    _track_static(sdir)
    rep = analyze_blank(sdir, marker_check=False)
    assert rep["checks"]["B1_confident_false_detections"]["pass"], \
        "static smudge should NOT create detections (it joins the bg)"
    assert not rep["checks"]["B5_static_dark_features"]["pass"]


def test_blank_marker_check_requires_all_sides(tmp_path):
    sdir = sv.generate_blank_session(tmp_path / "blank", "blank_t",
                                     duration_s=12.0)
    _track_static(sdir)
    rep = analyze_blank(sdir, marker_check=True)
    assert not rep["checks"]["B7_stimulus_marker_channel"]["pass"]
    assert rep["checks"]["B7_stimulus_marker_channel"]["sides_seen"] == []


# ------------------------------------------------------------ per-frame export
def test_per_frame_export(tmp_path):
    sdir = sv.generate_session(tmp_path / "s", "s", "baseline", 20260927,
                               duration_s=15.0)
    _track_static(sdir)
    consts = analysis_constants(load_gate_cfg())
    analyze_session(sdir, consts, tracking_cfg=load_tracking_cfg())
    p = sdir / "per_frame.csv"
    assert p.exists()
    import csv
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    cols = ["session_id", "frame", "t_s", "x_mm", "y_mm", "speed_mm_s",
            "heading_deg", "pause", "wall_in_band", "wall_following",
            "stimulus", "found", "confidence", "interpolated"]
    assert list(rows[0].keys()) == cols
    # frames are continuous and indexed from the lead-in end
    frames = [int(r["frame"]) for r in rows]
    assert frames == list(range(frames[0], frames[0] + len(frames)))
    # speed/heading present on moving frames
    n_speed = sum(1 for r in rows if r["speed_mm_s"])
    assert n_speed > 0.5 * len(rows)


# ------------------------------------------------------------------- intake
def _fake_session(root, name, meta_over=None, n_frames=90, fps=30,
                  with_annotations=0):
    d = root / name
    d.mkdir(parents=True)
    w = cv2.VideoWriter(str(d / "video.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps, (480, 360))
    base = np.full((360, 480, 3), 200, np.uint8)
    for _ in range(n_frames):
        w.write(base)
    w.release()
    meta = {
        "session_id": name, "label": name, "kind": "baseline",
        "fly_id": "F01", "fly_age_days": 5, "fly_sex": "M",
        "temperature_c": 23, "humidity_pct": 45,
        "started_utc": "20260928T090000Z", "fps": fps, "width": 480,
        "height": 360, "duration_s": n_frames / fps,
        "stimulus_schedule": [], "stimulus_seed": None,
        "calibration": {"px_per_mm": 6.5},
        "arena": {"type": "circle", "center_px": [240, 180],
                  "radius_px": 150},
        "config_hashes": {"gate_yaml": "x"}, "config_hash": "abc",
        "git_revision": "test", "checks_overridden": False,
        "tracking": {"mode": "reference", "background_ref": "bg.png",
                     "background_ref_sha256": "deadbeef"},
    }
    meta.update(meta_over or {})
    (d / "session.json").write_text(json.dumps(meta, indent=2))
    if with_annotations:
        with open(d / "annotations.csv", "w", newline="",
                  encoding="utf-8") as f:
            f.write("session,frame,x_px,y_px\n")
            for i in range(with_annotations):
                f.write(f"{name},{i},100,100\n")
    return d


def _meta_report(name, kind, day, fps=30, minutes=10.0, label=None,
                 seed=None, schedule=None, overridden=False,
                 tracking_mode="reference"):
    return {"session_dir": f"/x/{name}", "status": "OK", "tracks_csv": True,
            "meta": {"label": label or name, "kind": kind,
                     "started_utc": f"{day}T090000Z", "fps": fps,
                     "duration_s": minutes * 60.0, "fly_id": "F01",
                     "config_hash": "abc", "checks_overridden": overridden,
                     "stimulus_seed": seed, "stimulus_schedule": schedule,
                     "tracking": {"mode": tracking_mode}}}


def test_intake_verdicts(tmp_path):
    from flyrec.validation.intake import (_compliance, intake, render_intake)
    from flyrec.recording.recorder import make_stimulus_schedule

    # ---- compliance logic (meta level, no videos needed) ----------------
    reports = [
        _meta_report("d1_baseline_01", "baseline", "20260928"),
        _meta_report("d1_baseline_02", "baseline", "20260928"),
        _meta_report("d2_baseline_03", "baseline", "20260929"),
        _meta_report("d2_stimulus_01", "stimulus", "20260929",
                     seed=20260927, schedule=make_stimulus_schedule(0)),
    ]
    ann = {"d1_baseline_01": 50, "d1_baseline_02": 50, "d2_baseline_03": 50}
    # patch in annotation presence
    for r in reports:
        r["session_dir"] = r["session_dir"]  # unchanged
    comp = _compliance(reports, None, load_gate_cfg())
    # annotations: simulate present annotations on the three baselines
    # (the _compliance helper reads reports' annotated flags only via files,
    #  so exercise the count through the file-based path below instead)
    assert comp["baseline_sessions"]["status"] == "PASS"
    assert comp["recording_days"]["status"] == "PASS"
    assert comp["stimulus_schedule_matches_preregistration"]["status"] \
        == "PASS"
    assert comp["config_hash_consistency"]["status"] == "PASS"
    assert comp["no_overridden_apparatus_checks"]["status"] == "PASS"
    assert comp["reference_background_tracking"]["status"] == "PASS"
    # only 1 of 3 stimulus sessions -> PENDING
    assert comp["stimulus_sessions"]["status"] == "PENDING"
    # a tampered schedule -> FAIL
    reports.append(_meta_report("d3_stimulus_02", "stimulus", "20260930",
                                seed=20260928,
                                schedule=[{"t_on": 61.0, "t_off": 81.0,
                                           "side": "N"}]))
    comp = _compliance(reports, None, load_gate_cfg())
    assert comp["stimulus_schedule_matches_preregistration"]["status"] \
        == "FAIL"
    # overridden checks + mixed hashes -> deviations
    reports[-1]["meta"]["checks_overridden"] = True
    reports[-1]["meta"]["config_hash"] = "zzz"
    comp = _compliance(reports, None, load_gate_cfg())
    assert comp["no_overridden_apparatus_checks"]["status"] == "DEVIATION"
    assert comp["config_hash_consistency"]["status"] == "DEVIATION"

    # ---- end-to-end intake (real files, small videos) --------------------
    root = tmp_path / "sessions"
    s1 = _fake_session(root, "d1_baseline_01", with_annotations=50)
    s2 = _fake_session(root, "d1_baseline_02",
                       {"started_utc": "20260929T090000Z"},
                       with_annotations=50)
    rep = intake([str(root)], out_dir=tmp_path / "intake_out")
    assert render_intake(rep)
    # short (3 s) sessions: integrity OK, compliance PENDING -> not FAIL
    assert rep["verdict"] == "PENDING"
    # a session whose video has far too few frames for its claimed
    # duration -> integrity FAIL
    _fake_session(root, "d3_broken", n_frames=10,
                  meta_over={"duration_s": 3.0})
    rep = intake([str(root / "d3_broken")], out_dir=tmp_path / "intake_out2")
    assert rep["verdict"] == "FAIL"
    assert any("frames" in p for r in rep["sessions"]
               for p in r.get("problems", []))
    # (schedule tampering is covered at the meta level above; a full
    #  10-min-video end-to-end is exercised by the SA validation suite)


# ------------------------------------------------------------------- backup
def test_backup_roundtrip_and_immutability(tmp_path):
    from flyrec.data_mgmt.backup import backup, render_backup
    root = tmp_path / "sessions"
    s1 = _fake_session(root, "s1")
    s2 = _fake_session(root, "s2", meta_over={"kind": "blank"})
    to = tmp_path / "usb"
    rep = backup(to, sessions_root=root)
    assert rep["ok"] and set(rep["backed_up"]) == {"s1", "s2"}
    assert len(rep["verified"]) == 4            # video+json per session
    # second run: everything already up to date
    rep = backup(to, sessions_root=root)
    assert rep["ok"] and rep["skipped"] == ["s1", "s2"] and not rep["backed_up"]
    # mutating a RAW file after backup must raise the immutability alarm
    (s1 / "session.json").write_text("{}")
    rep = backup(to, sessions_root=root)
    assert not rep["ok"]
    assert any("s1" in e and "immutable" in e for e in rep["errors"])
    text = render_backup(rep)
    assert "ERROR" in text
