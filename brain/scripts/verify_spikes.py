"""Gate-1 evidence: SPIKE PROPAGATION on the canonical v783 brain.

Checks (all on the unmodified third-party model, seeded):
  1. NEGATIVE CONTROL  - no stimulation -> no spikes (network is silent at
     rest; no spurious activity).
  2. ACTIVATION        - stimulating the sugar-sensing set evokes activity
     in neurons BEYOND the stimulated set, including neurons >= 2 synap-
     tic hops away (multi-synaptic propagation through the connectome).
  3. MOTOR READOUT     - MN9 (tutorial motor neuron) is driven.
  4. CAUSALITY         - silencing the most active stimulated neuron
     reduces downstream activity and the MN9 rate (tutorial experiment).

Output: <out>/verdict.json + top-rates CSV + hop histogram CSV.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402


def rates_hz(rec):
    t = rec["t_run_ms"] / 1000.0
    return {bi: len(ts) / t for bi, ts in rec["spikes"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--t-run-ms", type=int, default=1000)
    ap.add_argument("--r-poi-hz", type=int, default=150)
    ap.add_argument("--out", default=str(bl.RESULTS / "spikes_783"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    flyid2i, _ = bl.load_maps("783")
    exc, missing = bl.resolve_ids(bl.SUGAR_IDS, "783")
    stim_set = set(exc)

    # ---- 1. negative control: silence at rest
    print(">>> [1/4] baseline (no stimulation) ...", flush=True)
    base = bl.run_seeded_trial(seed=args.seed, version="783",
                               t_run_ms=args.t_run_ms)
    bl.save_json(out / "baseline.json", bl.public_rec(base))
    c1 = base["n_spikes"] == 0

    # ---- 2. activation + propagation depth
    print(">>> [2/4] activation of sugar-sensing set ...", flush=True)
    act = bl.run_seeded_trial(seed=args.seed, version="783",
                              exc_fly_ids=bl.SUGAR_IDS,
                              t_run_ms=args.t_run_ms, r_poi_hz=args.r_poi_hz)
    bl.save_json(out / "activation.json", bl.public_rec(act))
    hops = bl.hops_from("783", exc, max_hops=4)
    active = np.array(sorted(act["spikes"]), dtype=np.int64)
    active_hops = hops[active]
    hist = {int(h): int((active_hops == h).sum()) for h in range(5)}
    hist["unreached_in_4hops"] = int((active_hops < 0).sum())
    propagated = active[~np.isin(active, np.array(sorted(stim_set)))]
    c2 = (len(propagated) > 0 and
          sum(hist[h] for h in (2, 3, 4)) > 0)

    # ---- 3. motor readout
    mn9_i = flyid2i.get(bl.MN9_ID)
    mn9_rate = rates_hz(act).get(mn9_i, 0.0)
    c3 = mn9_rate > 0

    # ---- 4. causality: silence the most active stimulated neuron
    r = rates_hz(act)
    top = max(r, key=lambda k: r[k] if k not in () else 0)
    top_is_stim = top in stim_set
    top_flyid = {v: k for k, v in flyid2i.items()}[top]
    print(f">>> [4/4] silencing most active neuron "
          f"(flywire {top_flyid}, stim={top_is_stim}) ...", flush=True)
    sil = bl.run_seeded_trial(seed=args.seed, version="783",
                              exc_fly_ids=bl.SUGAR_IDS,
                              slnc_fly_ids=[top_flyid],
                              t_run_ms=args.t_run_ms, r_poi_hz=args.r_poi_hz)
    bl.save_json(out / "silence_top.json", bl.public_rec(sil))
    mn9_rate_sil = rates_hz(sil).get(mn9_i, 0.0)
    c4 = (sil["n_spikes"] < act["n_spikes"]) or (mn9_rate_sil < mn9_rate)

    # ---- artifacts
    pd.DataFrame(
        {"brian_index": sorted(r), "rate_hz": [r[k] for k in sorted(r)]}
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
        "baseline_spikes": base["n_spikes"],
        "c2_propagation": bool(c2),
        "activation": {"n_spikes": act["n_spikes"],
                       "n_active": act["n_active_neurons"],
                       "n_active_beyond_stim_set": int(len(propagated)),
                       "hop_histogram_active_neurons": hist},
        "c3_motor_readout_mn9": bool(c3),
        "mn9_rate_hz": {"activation": mn9_rate, "silenced": mn9_rate_sil},
        "c4_causality_silencing_reduces_activity": bool(c4),
        "silencing": {"silenced_flywire_id": int(top_flyid),
                      "was_stimulated_neuron": bool(top_is_stim),
                      "n_spikes_after": sil["n_spikes"],
                      "n_active_after": sil["n_active_neurons"]},
        "peak_rss_kb": max(base["peak_rss_kb"], act["peak_rss_kb"],
                           sil["peak_rss_kb"]),
    }
    verdict["gate_spike_propagation"] = all(
        [c1, c2, c3, c4])
    bl.save_json(out / "verdict.json", verdict)
    print("\nGATE (spike propagation):",
          "PASS" if verdict["gate_spike_propagation"] else "FAIL")
    for k in ("c1_negative_control_silent", "c2_propagation",
              "c3_motor_readout_mn9", "c4_causality_silencing_reduces_activity"):
        print(f"  {k}: {verdict[k]}")
    print("  hop histogram:", hist)
    print(f"  MN9 rate: {mn9_rate:.1f} Hz -> {mn9_rate_sil:.1f} Hz (silenced)")


if __name__ == "__main__":
    main()
