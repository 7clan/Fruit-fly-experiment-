#!/usr/bin/env python3
"""Passive GPO frame sampler for perception development."""

from __future__ import annotations
import argparse, json, time, zipfile
from pathlib import Path
import cv2
from lab.bus import Bus
from lab.capture.base import create_windows_capture

ROOT = Path(__file__).resolve().parent

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--capture-fps", type=float, default=5.0)
    ap.add_argument("--save-fps", type=float, default=1.0)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "runs" / f"gpo_sample_{stamp}"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    bus = Bus()
    cap = create_windows_capture(bus, window_title_re=r"^Roblox$", target_fps=args.capture_fps)
    ch = cap.frames

    saved = 0
    last_saved = 0.0
    t0 = time.perf_counter()
    cap.start()
    try:
        while time.perf_counter() - t0 < args.seconds:
            env = ch.newest()
            if env is None:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            if last_saved and now - last_saved < 1.0 / max(args.save_fps, 0.1):
                continue
            last_saved = now
            img = env.payload.get("data_ref")
            if img is None:
                continue
            saved += 1
            cv2.imwrite(str(frames_dir / f"frame_{saved:04d}.jpg"), img)
    finally:
        cap.stop()

    summary = {
        "ok": saved > 0,
        "frames_saved": saved,
        "seconds": args.seconds,
        "capture_fps_cap": args.capture_fps,
        "save_fps_target": args.save_fps,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out_dir))

    print(json.dumps(summary, indent=2))
    print("UPLOAD:", zip_path)
    return 0 if saved else 1

if __name__ == "__main__":
    raise SystemExit(main())
