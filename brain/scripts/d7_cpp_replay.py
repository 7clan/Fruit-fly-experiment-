"""D7 cpp-standalone replay: re-run a RECORDED closed-loop episode schedule
through the C++ standalone device (the fastest full-brain route) and verify
bit-identity with the interactive runtime run.

Why this matters (D7 INTERACTIVE MODE design finding):
  * Brian2's cpp_standalone device compiles the WHOLE run schedule at build
    time - it cannot host a closed loop (rates cannot depend on runtime
    readouts). It is therefore an OFFLINE accelerator/verifier, not an
    interactive engine.
  * But a COMPLETED episode is a fixed piecewise-constant rate schedule +
    seed - exactly what standalone can replay through a TimedArray-driven
    PoissonGroup (same slot order, same Bernoulli rand()<rates*dt draws,
    and brian2 standalone vendors numpy's randomkit RNG, so streams match
    the numpy runtime - proven at Gate 2, m5 == m1 sha256).

Usage:
  python d7_cpp_replay.py --episode-dir <ep> [--build-dir dir]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402
import d7_runtime as rt  # noqa: E402
import d8_encoder as d8  # noqa: E402
import d10_closed_loop as d10  # noqa: E402

OUT = bl.ROOT / "results" / "d7_runtime"


def replay_episode_cpp(episode_dir, build_dir, seed=None, chunk_ms=50.0):
    import brian2 as b2

    ep = Path(episode_dir)
    summary = json.loads((ep / "summary.json").read_text())
    # EXACT full-precision schedule: uniform (dur_ms, rates) per chunk
    # (reconstructed + cross-checked against the rounded log)
    schedule = d10.exact_rate_schedule(ep)
    total_ms = sum(d for d, _ in schedule)
    seed = seed if seed is not None else summary["seed"]

    # interface: shuffled-sensory episodes used a per-episode spec
    iface = d8.DATA / "sensory_interface.json"
    if summary["condition"] == "shuffled_sensory":
        iface = ep.parent / (f"shuffled_iface_{summary['condition']}_"
                             f"{summary['scenario']}_r{summary['rep']}.json")
    channel_ids = {ch: d["ids"] for ch, d in
                   json.loads(Path(iface).read_text())["channels"].items()}

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.parent.mkdir(parents=True, exist_ok=True)

    b2.set_device("cpp_standalone", directory=str(build_dir),
                  with_output=False, build_on_run=False)
    t0 = time.perf_counter()
    brain = rt.DualModeBrain("reference", seed=seed,
                             channel_ids=channel_ids,
                             schedule=schedule, grid_ms=chunk_ms)
    res = brain.run_reference(total_ms)
    b2.device.build(directory=str(build_dir), compile=True, run=True,
                    debug=False)
    total_wall = time.perf_counter() - t0
    # binary-only timing (steady-state, what a laptop would pay per episode)
    bin_main = build_dir / "main"
    t0 = time.perf_counter()
    subprocess.run([str(bin_main)], cwd=str(build_dir),
                   capture_output=True, text=True)
    bin_wall = time.perf_counter() - t0

    text = brain.canonical_spikes_text()
    orig = (ep / "spikes.txt").read_text()
    identical = text == orig
    out = dict(
        episode=str(ep), seed=seed, schedule_chunks=len(schedule),
        total_bio_ms=total_ms,
        cpp_codegen_compile_run_s=round(total_wall, 1),
        binary_only_run_s=round(bin_wall, 1),
        bio_s_per_wall_s_binary=round(total_ms / 1000.0 / bin_wall, 4),
        child_peak_rss_mb=rt.child_peak_rss_mb(),
        identical_to_interactive=identical,
        sha_replay=rt.sha256_text(text)[:16],
        sha_original=rt.sha256_text(orig)[:16],
    )
    (ep / "cpp_replay.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--build-dir", default=str(OUT / "cpp_replay_project"))
    args = ap.parse_args()
    replay_episode_cpp(args.episode_dir, Path(args.build_dir))


if __name__ == "__main__":
    main()
