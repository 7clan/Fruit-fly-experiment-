"""Performance-study trial runner (one mode, one trial, fresh process).

Modes:
  full_reference   numpy codegen, full 138,639-neuron connectome (reference)
  cython           brian2 cython codegen, full connectome
  cpp_standalone   brian2 C++ standalone device, full connectome
  subcircuit       numpy, filtered task-relevant ROI connectome (built by
                   perf_study.py; uses --comp/--con path overrides)
  sparse_active    numpy, filtered data-driven ROI (active set + 1 hop)

Standard workload everywhere: the Gate-2 looming stimulus (80 neurons:
60 LPLC2 + 20 LC4, sampling seed 20260928), trial seed 11, 1 s, 150 Hz.
"""

import argparse
import json
import resource
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["full_reference", "cython", "cpp_standalone",
                             "subcircuit", "sparse_active"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--spikes-out", required=True)
    ap.add_argument("--rec-out", required=True)
    ap.add_argument("--comp", default="")      # filtered completeness csv
    ap.add_argument("--con", default="")       # filtered connectivity parquet
    args = ap.parse_args()

    gate2 = bl.ROOT / "results" / "gate2_io" / "manifest.json"
    manifest = json.loads(gate2.read_text())
    loom = manifest["conditions"]["loom_LPLC2_LC4"]["exc_ids"]

    codegen = None
    device = None
    device_dir = None
    override = None
    if args.mode == "cython":
        codegen = "cython"
    elif args.mode == "cpp_standalone":
        device = "cpp_standalone"
        device_dir = bl.ROOT / "results" / "perf_study" / "cpp_project"
    elif args.mode in ("subcircuit", "sparse_active"):
        assert args.comp and args.con, "filtered connectome paths required"
        override = (args.comp, args.con)

    rec = bl.run_seeded_trial(
        seed=args.seed, version="783", exc_fly_ids=loom,
        t_run_ms=1000, r_poi_hz=150,
        codegen=codegen, device=device, device_dir=device_dir,
        path_override=override)
    rec["mode"] = args.mode

    # child-process peak RSS (standalone binary) + total wall
    rec["child_peak_rss_kb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    rec["mode_total_wall_s"] = rec["wall_build_s"] + rec["wall_run_s"]

    # canonical spikes + flywire-keyed counts for fidelity comparison
    lines = []
    for bi in sorted(rec["spikes"]):
        lines.append(f"{bi}:" + ",".join(repr(t) for t in rec["spikes"][bi]))
    Path(args.spikes_out).write_text("\n".join(lines))
    spikes = rec.pop("spikes")
    rec["active_flywire_ids"] = sorted(
        int(k) for k in spikes.keys())
    bl.save_json(Path(args.rec_out), rec)
    print(json.dumps({k: rec[k] for k in
                      ("mode", "n_neurons_total", "n_active_neurons", "n_spikes",
                       "wall_build_s", "wall_run_s", "peak_rss_kb",
                       "child_peak_rss_kb")}, default=str))


if __name__ == "__main__":
    main()
