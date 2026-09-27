"""Run-directory management: immutable, timestamped, self-describing.

Every experiment run produces:
  runs/<timestamp>_<condition>/
      manifest.json    - who/what/when: software version, full config
                         snapshot, seed, host info (reproducibility)
      experiment.db    - SQLite (never overwritten; unique directory)
      frames/          - optional raw JPEG dumps (raw data, off by default)
      summary.json     - aggregate results + per-trial metrics
"""
from __future__ import annotations

import datetime
import json
import platform
import sys
from pathlib import Path

from ..version import __version__


def new_run_dir(base: str | Path, label: str) -> Path:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = base / f"{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "frames").mkdir(exist_ok=True)
    return run_dir


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.frames_dir = self.run_dir / "frames"
        self.db_path = self.run_dir / "experiment.db"
        self.summary_path = self.run_dir / "summary.json"
        self.manifest_path = self.run_dir / "manifest.json"
        self._frame_counter = 0

    def write_manifest(self, condition: str, controller: str, seed: int,
                       config: dict, notes: str = "") -> None:
        manifest = {
            "condition": condition,
            "controller": controller,
            "seed": int(seed),
            "software_version": __version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "created_at": datetime.datetime.now().isoformat(
                timespec="seconds"),
            "config_snapshot": config,
            "notes": notes,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    def dump_frame(self, frame, trial: int, t_ms: int) -> None:
        """Optional raw-data dump (save_frames: true)."""
        self._frame_counter += 1
        p = self.frames_dir / f"trial{trial:04d}_{t_ms:07d}.jpg"
        import cv2
        cv2.imwrite(str(p), frame)

    def write_summary(self, summary: dict) -> None:
        self.summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")
