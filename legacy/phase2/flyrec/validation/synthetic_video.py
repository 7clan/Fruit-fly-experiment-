"""Synthetic ground-truth session generator — PIPELINE VALIDATION ONLY.

Renders videos of a moving dark blob (a SOFTWARE TEST PATTERN with bouts,
pauses, wall interactions and optional stimulus taxis) plus per-frame
ground-truth positions. These videos validate the TRACKING AND ANALYSIS
CODE against known truth. They are NOT a model of real Drosophila behavior
and must NEVER be cited as evidence about real flies (PHASE2_GATE
PREREGISTRATION section 8).

Deterministic: every session is a pure function of its seed.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..config import DATA_DIR

# render geometry ( SOFTWARE TEST PATTERN parameters )
W, H = 480, 360
FPS = 30
PX_PER_MM = 5.0
ARENA = {"type": "circle", "center_px": [W // 2, H // 2], "radius_px": 150}
R_MAX_MM = 28.5          # keep the blob clear of the rim
FLY_AXES_PX = (12, 6)    # ~2.4 x 1.2 mm at 5 px/mm
LEAD_IN_S = 2.0          # empty arena while the static background builds
NOISE_SIGMA = 3.5
BG_LEVEL = 210
RING_LEVEL = 235         # lighter than background -> invisible to the
                         # dark-blob detector, visible to humans
MARKER_ON, MARKER_OFF = 255, 90


CARDINALS = {"RIGHT": 0.0, "FORWARD": math.pi / 2, "LEFT": math.pi,
             "BACKWARD": -math.pi / 2}


def _nearest_cardinal(angle: float) -> float:
    best, bd = 0.0, 1e9
    for a in CARDINALS.values():
        d = abs((angle - a + math.pi) % (2 * math.pi) - math.pi)
        if d < bd:
            best, bd = a, d
    return best


class TestPatternFly:
    """Bout/pause walker with wall interaction + optional stimulus taxis.

    SOFTWARE TEST PATTERN - fixed parameters, no biological claim. Bout
    headings are concentrated near the four CARDINALS (with small noise) so
    the pattern genuinely produces five distinguishable behavioral states
    (what SA9/L_A requires the ladder to detect); wall-follow segments use
    tangential headings (arbitrary angles) so the wall metric has signal.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.x, self.y = 0.0, 0.0            # mm, arena frame (y UP)
        self.heading = float(rng.uniform(-math.pi, math.pi))
        self.mode = "walk"
        self.t_mode_end = float(rng.uniform(1.0, 3.0))
        self.speed = float(rng.uniform(8, 18))
        self.wall_follow_until = -1.0
        self._was_stim = False

    def _new_bout_heading(self, stim_angle: float | None) -> float:
        if stim_angle is not None:
            return stim_angle + float(self.rng.normal(0.0, math.radians(15)))
        card = list(CARDINALS.values())[int(self.rng.integers(0, 4))]
        return card + float(self.rng.normal(0.0, math.radians(15)))

    def step(self, dt: float, t: float, stim_angle: float | None):
        rng = self.rng
        if self.mode == "pause" and stim_angle is not None and \
                self.t_mode_end - t > 0.3:
            # arousal: a pause that started BEFORE the stimulus ends quickly
            self.t_mode_end = t + 0.25
        if self._was_stim and stim_angle is None and self.mode == "walk":
            # stimulus just ended: the current bout is cut short so heading
            # persistence does not linger past the OFF transition
            self.t_mode_end = min(self.t_mode_end, t + 0.3)
        self._was_stim = stim_angle is not None
        if t >= self.t_mode_end:
            if self.mode == "walk":
                self.mode = "pause"
                if stim_angle is not None:      # stimulus arousal: brief stops
                    self.t_mode_end = t + float(rng.uniform(0.2, 0.5))
                else:
                    self.t_mode_end = t + float(rng.uniform(0.5, 1.5))
            else:
                self.mode = "walk"
                # short bouts OFF-stimulus (baseline heading averages over
                # many independent samples); long bouts ON-stimulus
                if stim_angle is not None:
                    dur = rng.uniform(1.5, 3.0)
                else:
                    dur = rng.uniform(0.5, 1.5)
                self.t_mode_end = t + float(dur)
                self.speed = float(rng.uniform(8, 18)) * \
                    (1.4 if stim_angle is not None else 1.0)
                self.heading = self._new_bout_heading(stim_angle)
        if self.mode != "walk":
            return
        # heading dynamics: mild persistent noise + stimulus taxis pull
        turn = float(rng.normal(0.0, 0.25)) * dt * 4.0
        if stim_angle is not None:
            # during stimulus the pattern orients to it (walls ignored)
            self.wall_follow_until = -1.0
            err = (stim_angle - self.heading + math.pi) % (2 * math.pi) - math.pi
            turn += 6.0 * err * dt
        if t < self.wall_follow_until:
            # wall-following: steer along the tangent
            tang = math.atan2(self.y, self.x) + math.pi / 2.0
            err = (tang - self.heading + math.pi) % (2 * math.pi) - math.pi
            turn += 4.0 * err * dt
        self.heading = (self.heading + turn + math.pi) % (2 * math.pi) - math.pi
        nx = self.x + math.cos(self.heading) * self.speed * dt
        ny = self.y + math.sin(self.heading) * self.speed * dt
        r = math.hypot(nx, ny)
        if r > R_MAX_MM:
            if stim_angle is not None:
                # press toward the stimulus WITHOUT jitter: hold position
                # exactly (measured speed -> 0, so response windows keep
                # only the clean approach phase)
                return
            # wall interaction: 35% wall-following segment, 65% bounce to
            # the nearest cardinal (keeps the pattern mostly 5-state)
            if self.rng.random() < 0.35:
                self.wall_follow_until = t + float(self.rng.uniform(1.0, 3.0))
                tang = math.atan2(ny, nx) + math.pi / 2.0
                self.heading = tang + float(self.rng.normal(0, 0.2))
            else:
                inward = _nearest_cardinal(math.atan2(-ny, -nx))
                self.heading = inward + float(self.rng.normal(0, 0.25))
            k = (R_MAX_MM - 0.5) / max(r, 1e-9)
            nx, ny = nx * k, ny * k
        self.x, self.y = nx, ny


def render_frame(t: float, fly: TestPatternFly | None, rng: np.random.Generator,
                 marker_side: str | None, base: np.ndarray) -> np.ndarray:
    # spatially correlated noise (half-res -> upscale), slow illumination drift
    small = rng.normal(0.0, NOISE_SIGMA, (H // 2, W // 2))
    noise = cv2.resize(small.astype(np.float32), (W, H),
                       interpolation=cv2.INTER_LINEAR)
    drift = 3.0 * math.sin(2 * math.pi * t / 40.0)
    frame = np.clip(base + noise + drift, 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # arena ring (LIGHTER than background -> invisible to dark-blob detector)
    cv2.circle(frame, (W // 2, H // 2), ARENA["radius_px"],
               (RING_LEVEL,) * 3, 2)
    # stimulus margin markers (OUTSIDE the arena, bright when active)
    mw, mh, margin = 30, 14, 18
    cx, cy = W // 2, H // 2
    rois = {"N": (cx - mw // 2, margin), "S": (cx - mw // 2, H - margin - mh),
            "W": (margin, cy - mh // 2), "E": (W - margin - mw, cy - mh // 2)}
    for side, (x, y) in rois.items():
        val = MARKER_ON if side == marker_side else MARKER_OFF
        cv2.rectangle(frame, (x, y), (x + mw, y + mh), (val,) * 3, -1)
    # the animal (dark ellipse, rotated to heading; y-up -> image angle -h)
    if fly is not None:
        px, py = cx + fly.x * PX_PER_MM, cy - fly.y * PX_PER_MM
        jx = float(rng.uniform(0.9, 1.1)) * FLY_AXES_PX[0]
        jy = float(rng.uniform(0.9, 1.1)) * FLY_AXES_PX[1]
        cv2.ellipse(frame, (int(round(px)), int(round(py))),
                    (int(jx), int(jy)),
                    math.degrees(-fly.heading), 0, 360, (45, 45, 45), -1)
    return frame


def active_marker(t: float, events: list[dict]) -> str | None:
    for e in events:
        if e["t_on"] <= t < e["t_off"]:
            return e["side"]
    return None


def stim_angle_of(side: str | None) -> float | None:
    if side is None:
        return None
    return {"N": math.pi / 2, "E": 0.0, "S": -math.pi / 2, "W": math.pi}[side]


def generate_session(out_dir: Path, label: str, kind: str, seed: int,
                     duration_s: float = 60.0, stimulus: bool = False,
                     session_index: int = 0,
                     lead_in_s: float | None = None) -> Path:
    """Render one ground-truth session into out_dir (label name).

    lead_in_s: seconds of fly-free arena before the subject appears. Default
    LEAD_IN_S (the software-test structure SA1-SA11 were registered on).
    Set 0.0 to emulate the REAL-RIG situation (fly present from the first
    frame) for the reference-background mode check SA12.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lead_in_s = LEAD_IN_S if lead_in_s is None else float(lead_in_s)
    rng = np.random.default_rng(seed)
    events = []
    if stimulus:
        # IRREGULAR onsets (software-test design choice): regular grids let
        # joint circular-shift permutations land back on the real stimulus
        # grid (inflated permutation tail), and gaps < 13 s (response 8 s
        # baseline + 5 s ON) would let baseline windows overlap the previous
        # ON period. The REAL protocol keeps its own regular 60-s-spaced
        # schedule in 10-min sessions.
        rng_sched = np.random.default_rng(20260927 + session_index)
        n_ev = 8 if duration_s >= 115.0 else 4
        onsets = [8.0, 24.0, 39.0, 55.0, 70.0, 86.0, 101.0, 114.0][:n_ev]
        sides = [str(s) for s in rng_sched.choice(["N", "E", "S", "W"],
                                                  size=n_ev)]
        events = [{"t_on": o, "t_off": o + 5.0, "side": s}
                  for o, s in zip(onsets, sides)]
    n_frames = int(round(duration_s * FPS))
    lead = int(lead_in_s * FPS)

    fly = TestPatternFly(np.random.default_rng(seed + 999))
    gt_rows = []

    # fourcc preference: mp4v (good compression), fallback MJPG
    writer, video_path = None, None
    for f, ext in (("mp4v", "mp4"), ("MJPG", "avi")):
        p = out_dir / f"video.{ext}"
        wtr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*f), FPS,
                              (W, H))
        if wtr.isOpened():
            writer, video_path = wtr, p
            break
        wtr.release()
    if writer is None:
        raise RuntimeError("no video writer available")

    base = np.full((H, W), BG_LEVEL, np.float32)
    for i in range(n_frames):
        t = i / FPS
        present = i >= lead
        cur_fly = fly if present else None
        if present:
            fly.step(1.0 / FPS, t, stim_angle_of(active_marker(t, events)))
        side = active_marker(t, events)
        frame = render_frame(t, cur_fly, rng, side, base)
        writer.write(frame)
        if present:
            px = W / 2 + fly.x * PX_PER_MM
            py = H / 2 - fly.y * PX_PER_MM
            gt_rows.append([i, f"{t:.4f}", f"{px:.2f}", f"{py:.2f}",
                            f"{fly.x:.4f}", f"{fly.y:.4f}", 1,
                            1 if fly.mode == "walk" else 0])
        else:
            gt_rows.append([i, f"{t:.4f}", "", "", "", "", 0, 0])
    writer.release()

    with open(out_dir / "ground_truth.csv", "w", newline="",
              encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "t_s", "x_px", "y_px", "x_mm", "y_mm",
                       "present", "moving"])
        wcsv.writerows(gt_rows)

    meta = {
        "label": label,
        "kind": kind,
        "purpose": "SOFTWARE VALIDATION ONLY - synthetic test pattern; "
                   "NOT a model of real Drosophila behavior and NOT "
                   "evidence about real flies",
        "seed": seed,
        "fps": FPS, "width": W, "height": H,
        "duration_s": duration_s, "lead_in_s": lead_in_s,
        "calibration": {"px_per_mm": PX_PER_MM},
        "arena": ARENA,
        "stimulus_schedule": events,
        "test_pattern": {
            "bout_s": [0.5, 1.5], "pause_s": [0.5, 1.5],
            "stimulus_bout_s": [1.5, 3.0], "stimulus_pause_s": [0.2, 0.5],
            "speed_mm_s": [8, 18], "turn_noise_rad_s": 0.25,
            "bout_headings": "cardinal mixture + N(0, 15 deg) "
                             "(5-state test pattern)",
            "wall_follow_prob": 0.35, "stimulus_taxis_gain": 6.0,
            "stimulus_arousal": "pre-existing pauses cut at onset; speed "
                                "x1.4; wall-following suspended while ON",
        },
    }
    (out_dir / "session.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out_dir


def generate_reference_background(out_path: Path, seed: int = 424243,
                                 duration_s: float = 8.0) -> Path:
    """Fly-free clip -> per-pixel median background image (SA12).

    Mirrors the REAL-RIG flow exactly: a no-fly recording under identical
    lighting becomes the stored `reference` detector background.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    stack = []
    for i in range(int(duration_s * FPS)):
        frame = render_frame(i / FPS, None, rng, None,
                             np.full((H, W), BG_LEVEL, np.float32))
        stack.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    bg = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)
    if not cv2.imwrite(str(out_path), bg):
        raise RuntimeError(f"cannot write {out_path}")
    return out_path


def generate_blank_session(out_dir: Path, label: str = "blank_01",
                           duration_s: float = 20.0,
                           seed: int = 424242) -> Path:
    """Fly-free arena recording (SA11 / blank-arena test pattern)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(out_dir / "video.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    base = np.full((H, W), BG_LEVEL, np.float32)
    n = int(duration_s * FPS)
    for i in range(n):
        writer.write(render_frame(i / FPS, None, rng, None, base))
    writer.release()
    (out_dir / "session.json").write_text(json.dumps({
        "label": label, "kind": "blank", "fps": FPS, "width": W,
        "height": H, "duration_s": duration_s,
        "calibration": {"px_per_mm": PX_PER_MM}, "arena": ARENA,
        "purpose": "blank-arena test pattern - no fly present"},
        indent=2), encoding="utf-8")
    return out_dir


def generate_validation_suite(root: Path | None = None,
                              duration_s: float = 60.0) -> list[Path]:
    """3 baseline + 2 stimulus sessions + 1 blank (SA11)."""
    root = Path(root) if root else DATA_DIR / "validation"
    specs = [
        ("baseline_01", "baseline", 20260927, False, 0),
        ("baseline_02", "baseline", 20260928, False, 0),
        ("baseline_03", "baseline", 20260929, False, 0),
        ("stimulus_01", "stimulus", 20260930, True, 0),
        ("stimulus_02", "stimulus", 20260931, True, 1),
    ]
    dirs = []
    for label, kind, seed, stim, idx in specs:
        print(f"  generating {label} (seed {seed}, {duration_s:.0f}s)...",
              flush=True)
        dirs.append(generate_session(root / label, label, kind, seed,
                                     duration_s=duration_s, stimulus=stim,
                                     session_index=idx))
    # blank: no fly at all
    dirs.append(generate_blank_session(root / "blank_01", "blank_01",
                                       duration_s=20.0))
    return dirs
