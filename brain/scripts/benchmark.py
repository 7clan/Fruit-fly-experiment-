"""D3: benchmark the whole-brain simulation (canonical v783).

Measures: RAM (peak RSS), CPU, initialization time (data load + network
build), simulation speed (bio-seconds per wall-second), neuron count,
connection count, active neurons and spikes per trial.

Portable: also runs on the user's Windows laptop later (psutil optional;
resource/proc are Linux fallbacks). Results -> benchmark.json + .md.
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="783", choices=list(bl.VERSIONS))
    ap.add_argument("--warm-trials", type=int, default=3)
    ap.add_argument("--t-run-ms", type=int, default=1000)
    ap.add_argument("--out", default=str(bl.RESULTS / "benchmark"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import brian2

    machine = bl.cpu_info()
    machine["platform"] = platform.platform()
    machine["python"] = sys.version.split()[0]
    machine["brian2"] = brian2.__version__
    machine["numpy"] = np.__version__
    machine["pandas"] = pd.__version__

    # ---- data load timing
    t0 = time.perf_counter()
    df_comp = pd.read_csv(bl.VERSIONS[args.version]["path_comp"], index_col=0)
    df_con = pd.read_parquet(bl.VERSIONS[args.version]["path_con"])
    t_load = time.perf_counter() - t0
    rss_after_load = bl.rss_kb()

    counts = {
        "neurons": int(len(df_comp)),
        "connections": int(len(df_con)),
        "connectome": bl.VERSIONS[args.version]["label"],
    }

    # ---- build timing (network construction, codegen cache warm from the
    #      earlier runs; cold-compile note recorded honestly)
    import model as dbm
    from brian2 import Hz, ms
    build_times = []
    for k in range(3):
        t0 = time.perf_counter()
        params = dict(dbm.default_params)
        params["t_run"] = args.t_run_ms * ms
        params["r_poi"] = 150 * Hz
        neu, syn, spk_mon = dbm.create_model(
            bl.VERSIONS[args.version]["path_comp"],
            bl.VERSIONS[args.version]["path_con"], params)
        build_times.append(time.perf_counter() - t0)
        del neu, syn, spk_mon
    t_build_first, t_build_mean = build_times[0], float(np.mean(build_times[1:]) or build_times[0])
    rss_after_build = bl.rss_kb()

    # ---- warm trials (timed)
    trials = []
    for k in range(args.warm_trials):
        rec = bl.run_seeded_trial(seed=500 + k, version=args.version,
                                  exc_fly_ids=bl.SUGAR_IDS,
                                  t_run_ms=args.t_run_ms, r_poi_hz=150)
        trials.append(bl.public_rec(rec))
        print(f"  trial {k}: build={rec['wall_build_s']}s run={rec['wall_run_s']}s "
              f"spikes={rec['n_spikes']} active={rec['n_active_neurons']}", flush=True)

    steady = trials[1:] or trials
    sim_speed = [ (t["t_run_ms"] / 1000.0) / t["wall_run_s"] for t in steady ]
    res = {
        "machine": machine,
        "counts": counts,
        "timing": {
            "data_load_s": round(t_load, 2),
            "network_build_first_s": round(t_build_first, 2),
            "network_build_warm_mean_s": round(t_build_mean, 2),
            "note": "codegen cache warm; first-ever cold compile was observed "
                    "once during setup (see brain/README.md)",
        },
        "memory": {
            "rss_after_load_mb": round(rss_after_load / 1024, 1),
            "rss_after_build_mb": round(rss_after_build / 1024, 1),
            "peak_rss_mb": round(max(t["peak_rss_kb"] for t in trials) / 1024, 1),
        },
        "trials": trials,
        "sim_speed_bio_s_per_wall_s": {
            "mean": round(float(np.mean(sim_speed)), 5),
            "min": round(float(np.min(sim_speed)), 5),
            "max": round(float(np.max(sim_speed)), 5),
        },
        "spikes_per_trial": [t["n_spikes"] for t in trials],
        "active_neurons_per_trial": [t["n_active_neurons"] for t in trials],
    }
    bl.save_json(out / f"benchmark_{args.version}.json", res)

    md = [
        f"# D3 benchmark — {counts['connectome']}",
        "",
        f"- Machine: {machine.get('model', 'unknown')} — "
        f"{machine['cores']} cores, {machine.get('mem_total_mb', '?')} MB RAM "
        f"({machine['platform']})",
        f"- Stack: Python {machine['python']}, Brian2 {machine['brian2']}, "
        f"NumPy {machine['numpy']}, pandas {machine['pandas']}",
        f"- Neurons: **{counts['neurons']:,}**",
        f"- Connections (synapse rows): **{counts['connections']:,}**",
        f"- Data load: {res['timing']['data_load_s']} s",
        f"- Network build: first {res['timing']['network_build_first_s']} s, "
        f"warm {res['timing']['network_build_warm_mean_s']} s",
        f"- Peak RSS: **{res['memory']['peak_rss_mb']} MB**",
        f"- Simulation speed: **{res['sim_speed_bio_s_per_wall_s']['mean']} "
        f"bio-s per wall-s** (stimulated network, {args.t_run_ms} ms trials)",
        f"- Spikes/trial: {res['spikes_per_trial']}; "
        f"active neurons/trial: {res['active_neurons_per_trial']}",
    ]
    (out / f"benchmark_{args.version}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
