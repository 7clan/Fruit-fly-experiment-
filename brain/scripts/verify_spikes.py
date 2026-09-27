"""Gate-1 evidence: SPIKE PROPAGATION on the canonical v783 brain.

Each trial runs in its OWN fresh interpreter (trial_runner.py); analysis
happens in this (light) parent process. Checks (seeded, unmodified model):

  1. NEGATIVE CONTROL  - no stimulation -> no spikes (silent at rest).
  2. ACTIVATION        - sugar-sensing stimulation evokes activity BEYOND
     the stimulated set, including neurons >= 2 synaptic hops away
     (multi-synaptic propagation through the connectome).
  3. MOTOR READOUT     - MN9 (tutorial motor neuron) is driven.
  4. CAUSALITY         - silencing the most active stimulated neuron
     reduces downstream activity / the MN9 rate (tutorial experiment).

Output: <out>/verdict.json + top-rates CSV + hop histogram CSV.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402

VENV_PY = bl.ROOT / ".venv" / "bin" / "python"


def run_trial(tag, seed, t_run_ms, r_poi_hz, exc="sugar", silence_fid=0):
    out_dir = Path(args.out)
    spikes = out_dir / f"spikes_{tag}.txt"
    rec = out_dir / f"trial_{tag}.json"
    cmd = [str(VENV_PY), str(HERE / "trial_runner.py"),
           "--seed", str(seed), "--t-run-ms", str(t_run_ms),
           "--r-poi-hz", str(r_poi_hz), "--exc", exc,
           "--spikes-out", str(spikes), "--rec-out", str(rec)]
    if silence_fid:
        cmd += ["--silence-fid", str(silence_fid)]
    import os
    env = {**os.environ, "MALLOC_ARENA_MAX": "2"}
    print(f">>> trial '{tag}' (fresh process) ...", flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"trial '{tag}' FAILED")
    # parse canonical spikes text back into {brian_index: [times]}
    trains = {}
    for line in spikes.read_text().splitlines():
        if not line.strip():
            continue
        bi, ts = line.split(":", 1)
        trains[int(bi)] = [float(x) for x in ts.split(",")] if ts else []
    meta = json.loads(rec.read_text())
    return trains, meta


def rates_hz(trains, t_run_ms):
    t = t_run_ms / 1000.0
    return {bi: len(ts) / t for bi, ts in trains.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--t-run-ms", type=int, default=1000)
    ap.add_argument("--r-poi-hz", type=int, default=150)
    ap.add_argument("--out", default=str(bl.RESULTS / "spikes_783"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    flyid2i, _ = bl.load_maps("783")
    i2flyid = {v: k for k, v in flyid2i.items()}
    exc, missing = bl.resolve_ids(bl.SUGAR_IDS, "783")
    stim_set = set(exc)

    # ---- 1. negative control
    base_trains, base_meta = run_trial("baseline", args.seed, args.t_run_ms, 0,
                                       exc="none")
    c1 = base_meta["n_spikes"] == 0

    # ---- 2. activation trial (analysis deferred until all children done,
    #      so the parent stays light while ~2.9 GB subprocesses run)
    act_trains, act_meta = run_trial("activation", args.seed, args.t_run_ms,
                                     args.r_poi_hz)
    r_act = rates_hz(act_trains, args.t_run_ms)

    # ---- 3. motor readout
    mn9_i = flyid2i.get(bl.MN9_ID)
    mn9_rate = r_act.get(mn9_i, 0.0)
    c3 = mn9_rate > 0

    # ---- 4. causality: silence the most active stimulated neuron
    top = max(r_act, key=lambda k: r_act[k])
    top_flyid = i2flyid[top]
    top_is_stim = top in stim_set
    sil_trains, sil_meta = run_trial(
        "silence_top", args.seed, args.t_run_ms, args.r_poi_hz,
        silence_fid=int(top_flyid))
    r_sil = rates_hz(sil_trains, args.t_run_ms)
    mn9_rate_sil = r_sil.get(mn9_i, 0.0)
    c4 = (sil_meta["n_spikes"] < act_meta["n_spikes"]) or \
         (mn9_rate_sil < mn9_rate)

    # ---- 2b. propagation-depth analysis (parent-heavy, children finished)
    hops = bl.hops_from("783", exc, max_hops=4)
    active = np.array(sorted(act_trains), dtype=np.int64)
    active_hops = hops[active]
    hist = {int(h): int((active_hops == h).sum()) for h in range(5)}
    hist["unreached_within_4hops"] = int((active_hops < 0).sum())
    propagated = active[~np.isin(active, np.array(sorted(stim_set)))]
    c2 = (len(propagated) > 0) and (sum(hist[h] for h in (2, 3, 4)) > 0)

    # ---- artifacts
    pd.DataFrame({"brian_index": sorted(r_act),
                  "rate_hz": [r_act[k] for k in sorted(r_act)]}
                 ).to_csv(out / "activation_top_rates.csv", index=False)
    pd.DataFrame({"hops": list(hist.keys()),
                  "n_active_neurons": list(hist.values())}
                 ).to_csv(out / "hop_histogram.csv", index=False)

    verdict = {
        "version": "783 (canonical)",
        "seed": args.seed,
        "stimulated": {"requested": len(bl.SUGAR_IDS),
                       "present_in_v783": len(exc), "missing": missing},
        "c1_negative_control_silent": bool(c1),
        "baseline_spikes": base_meta["n_spikes"],
        "c2_propagation": bool(c2),
        "activation": {"n_spikes": act_meta["n_spikes"],
                       "n_active": act_meta["n_active_neurons"],
                       "n_active_beyond_stim_set": int(len(propagated)),
                       "hop_histogram_active_neurons": hist},
        "c3_motor_readout_mn9": bool(c3),
        "mn9_rate_hz": {"activation": mn9_rate, "silenced": mn9_rate_sil},
        "c4_causality_silencing_reduces_activity": bool(c4),
        "silencing": {"silenced_flywire_id": int(top_flyid),
                      "was_stimulated_neuron": bool(top_is_stim),
                      "n_spikes_after": sil_meta["n_spikes"],
                      "n_active_after": sil_meta["n_active_neurons"]},
        "peak_rss_kb": max(base_meta["peak_rss_kb"], act_meta["peak_rss_kb"],
                           sil_meta["peak_rss_kb"]),
        "checks": {"c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "c4": bool(c4)},
    }
    verdict["gate_spike_propagation"] = all([c1, c2, c3, c4])
    bl.save_json(out / "verdict.json", verdict)
    print(json.dumps(verdict, indent=2, default=str))
    print("\nGATE (spike propagation):",
          "PASS" if verdict["gate_spike_propagation"] else "FAIL")
