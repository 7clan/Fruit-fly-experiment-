"""Manual annotation tool for gate criterion G2 (tracking accuracy).

Interactive mode (requires a display, runs on the USER's machine):
    python phase2/run_phase2.py annotate <session_dir> --n 50 --seed 20260927
    -> shows each sampled frame; click the fly center; 'n' = skip;
       'q' = save and quit. Writes annotations.csv.

Ground-truth mode (software validation only):
    python phase2/run_phase2.py annotate <session_dir> --from-gt --n 150
    -> samples frames from ground_truth.csv and writes exact positions.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


def sample_frames(n: int, n_frames: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    n = min(n, n_frames)
    return sorted(int(i) for i in rng.choice(n_frames, size=n, replace=False))


def annotate_interactive(session_dir: Path, n: int = 50, seed: int = 20260927,
                         start_at: int = 0) -> Path:
    """Click the fly center on sampled frames. Requires a display."""
    sdir = Path(session_dir)
    video = next((p for p in sdir.glob("video.*") if p.suffix in
                  (".mp4", ".avi")), None)
    if video is None:
        raise FileNotFoundError(f"no video in {sdir}")
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = sample_frames(n, total - start_at, seed)
    clicks: list[tuple] = []
    state = {"x": None, "y": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["x"], state["y"] = x, y

    win = "annotate - click fly center, n=skip, q=quit&saves"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    for i, f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f + start_at)
        ok, frame = cap.read()
        if not ok:
            continue
        state["x"] = state["y"] = None
        while True:
            disp = frame.copy()
            cv2.putText(disp, f"frame {f} ({i + 1}/{len(frames)})", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(win, disp)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                cv2.destroyAllWindows()
                cap.release()
                return _write(sdir, clicks)
            if k == ord("n"):
                break
            if state["x"] is not None and k == 13 or \
               (state["x"] is not None and k == 32):
                clicks.append((f, state["x"], state["y"]))
                break
    cv2.destroyAllWindows()
    cap.release()
    return _write(sdir, clicks)


def _session_label(sdir: Path) -> str:
    if (sdir / "session.json").exists():
        try:
            return json.loads((sdir / "session.json").read_text(
                encoding="utf-8")).get("label", sdir.name)
        except Exception:
            pass
    return sdir.name


def _write(sdir: Path, clicks: list[tuple]) -> Path:
    out = sdir / "annotations.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session", "frame", "x_px", "y_px"])
        label = _session_label(sdir)
        for (fr, x, y) in clicks:
            w.writerow([label, fr, x, y])
    print(f"wrote {len(clicks)} annotations -> {out}")
    return out


def annotate_from_gt(session_dir: Path, n: int = 150,
                     seed: int = 20260927) -> Path:
    """SOFTWARE VALIDATION ONLY: annotations sampled from ground truth."""
    sdir = Path(session_dir)
    gt = np.genfromtxt(sdir / "ground_truth.csv", delimiter=",", names=True,
                       dtype=None, encoding="utf-8")
    if gt.ndim == 0:
        gt = gt.reshape(1)
    present = np.where(gt["present"].astype(int) == 1)[0]
    rng = np.random.default_rng(seed)
    n = min(n, len(present))
    idx = sorted(int(i) for i in rng.choice(present, size=n, replace=False))
    out = sdir / "annotations.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session", "frame", "x_px", "y_px"])
        label = _session_label(sdir)
        for i in idx:
            w.writerow([label, int(gt["frame"][i]), f"{gt['x_px'][i]:.2f}",
                        f"{gt['y_px'][i]:.2f}"])
    print(f"wrote {n} GT-derived annotations -> {out}")
    return out


def annotate_repeatability(session_dir: Path, n: int = 20,
                           seed: int = 7) -> Path:
    """Re-annotate a subsample twice for the human self-consistency check."""
    return annotate_interactive(session_dir, n=n, seed=seed)
