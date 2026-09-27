"""D3: benchmark the whole-brain simulation (canonical v783).

Every timed trial runs in a fresh subprocess (whole-brain peak RSS is
~3 GB). This measures what the application will actually experience:
per-process initialization (interpreter + data load + network build) and
steady-state simulation speed (bio-seconds per wall-second, codegen cache
warm). Portable: also runs on the user's Windows laptop (the same
trial_runner works there; resource/proc metrics degrade gracefully).

Results -> benchmark_783.{json,md}.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402

VENV_PY = bl.ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32"
                                else "bin/python")


def measure_data_load(version):
    """Data-load timing + counts in a THROWAWAY subprocess.

    Memory discipline: loading the 15M-row parquet in THIS parent would
    pin ~1.4 GB in the pyarrow allocator (not returnable via malloc_trim)
    and the parent would be OOM-killed while trial children run.
    """
    code = (
        "import json, time, pandas as pd\n"
        f"t0 = time.perf_counter()\n"
        f"comp = pd.read_csv({str(bl.VERSIONS[version]['path_comp'])!r}, index_col=0)\n"
        f"con = pd.read_parquet({str(bl.VERSIONS[version]['path_con'])!r})\n"
        "print(json.dumps({'load_s': round(time.perf_counter()-t0, 2),"
        " 'neurons': int(len(comp)), 'connections': int(len(con))}))"
    )
    r = subprocess.run([str(VENV_PY), "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"data-load measurement FAILED: {r.stderr[-500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="783", choices=list(bl.VERSIONS))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--t-run-ms", type=int, default=1000)
    ap.add_argument("--out", default=str(bl.RESULTS / "benchmark"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from importlib.metadata import version as pkg_version

    machine = bl.cpu_info()
    machine["platform"] = platform.platform()
    machine["python"] = sys.version.split()[0]
    machine["brian2"] = pkg_version("brian2")
    machine["numpy"] = pkg_version("numpy")
    machine["pandas"] = pkg_version("pandas")

    # ---- data-load timing + counts (throwaway subprocess; parent stays light)
    dl = measure_data_load(args.version)
    t_load = dl["load_s"]
    counts = {"neurons": dl["neurons"], "connections": dl["connections"],
              "connectome": bl.VERSIONS[args.version]["label"]}
    bl.trim_memory()

    # ---- timed trials, one fresh process each
    trials = []
    env = {**os.environ, "MALLOC_ARENA_MAX": "2"}
    for k in range(args.trials):
        seed = 500 + k
        spikes = out / f"bench_spikes_seed{seed}.txt"
        recp = out / f"bench_trial_seed{seed}.json"
        cmd = [str(VENV_PY), str(HERE / "trial_runner.py"),
               "--seed", str(seed), "--t-run-ms", str(args.t_run_ms),
               "--r-poi-hz", "150", "--exc", "sugar",
               "--spikes-out", str(spikes), "--rec-out", str(recp)]
        print(f">>> benchmark trial {k + 1}/{args.trials} (fresh process)...",
              flush=True)
        t0 = time.perf_counter()
        r = subprocess.run(cmd, env=env)
        wall_total = time.perf_counter() - t0
        if r.returncode != 0:
            raise SystemExit(f"benchmark trial seed={seed} FAILED")
        spikes.unlink()                 # bulky; rec keeps the metrics
        rec = json.loads(recp.read_text())
        rec["wall_total_process_s"] = round(wall_total, 2)
        rec.pop("missing_exc2", None)
        trials.append(rec)
        print(f"    build={rec['wall_build_s']}s run={rec['wall_run_s']}s "
              f"total={wall_total:.1f}s spikes={rec['n_spikes']} "
              f"active={rec['n_active_neurons']} "
              f"peakRSS={rec['peak_rss_kb'] / 1024:.0f}MB", flush=True)

    sim_speed = [(t["t_run_ms"] / 1000.0) / t["wall_run_s"] for t in trials]
    res = {
        "machine": machine,
        "counts": counts,
        "data_load_s": round(t_load, 2),
        "trials": trials,
        "init_per_fresh_process_s": {
            "mean": round(float(np.mean([t["wall_total_process_s"] for t in trials])), 2),
            "note": "interpreter + imports + data load + network build + run",
        },
        "network_build_s": [t["wall_build_s"] for t in trials],
        "run_wall_s": [t["wall_run_s"] for t in trials],
        "sim_speed_bio_s_per_wall_s": {
            "mean": round(float(np.mean(sim_speed)), 5),
            "min": round(float(np.min(sim_speed)), 5),
            "max": round(float(np.max(sim_speed)), 5),
            "note": "stimulated network (sugar set, 150 Hz), codegen cache warm",
        },
        "spikes_per_trial": [t["n_spikes"] for t in trials],
        "active_neurons_per_trial": [t["n_active_neurons"] for t in trials],
        "peak_rss_mb": {
            "mean": round(float(np.mean([t["peak_rss_kb"] for t in trials])) / 1024, 1),
            "max": round(float(np.max([t["peak_rss_kb"] for t in trials])) / 1024, 1),
        },
        "memory_note": "single whole-brain network per process; serial trials "
                       "only on machines with < 8 GB RAM",
    }
    bl.save_json(out / f"benchmark_{args.version}.json", res)

    md = [
        f"# D3 benchmark — {counts['connectome']}",
        "",
        f"- Machine: {machine.get('model', 'unknown')} — {machine['cores']} cores, "
        f"{machine.get('mem_total_mb', '?')} MB RAM ({machine['platform']})",
        f"- Stack: Python {machine['python']}, Brian2 {machine['brian2']}, "
        f"NumPy {machine['numpy']}, pandas {machine['pandas']}",
        f"- Neurons: **{counts['neurons']:,}**",
        f"- Connections (synapse rows): **{counts['connections']:,}**",
        f"- Data load (pandas): {res['data_load_s']} s",
        f"- Network build per process: {res['network_build_s']} s",
        f"- Full init per fresh process (mean): **{res['init_per_fresh_process_s']['mean']} s**",
        f"- Peak RSS per process: **{res['peak_rss_mb']['mean']} MB** "
        f"(max {res['peak_rss_mb']['max']} MB)",
        f"- Simulation speed: **{res['sim_speed_bio_s_per_wall_s']['mean']} "
        f"bio-s per wall-s** ({res['sim_speed_bio_s_per_wall_s']['note']})",
        f"- Spikes/trial: {res['spikes_per_trial']}",
        f"- Active neurons/trial: {res['active_neurons_per_trial']}",
    ]
    (out / f"benchmark_{args.version}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
