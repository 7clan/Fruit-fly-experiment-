"""D1: reproduce the shipped example.ipynb tutorial (FlyWire v630, as the
paper shipped it) with reduced n_run and n_proc for this machine.

Faithful to the notebook's experiment sequence:
  1. sugarR          : activate the 21 sugar-sensing neurons (default r_poi)
  2. sugarR_100Hz    : same, r_poi = 100 Hz
  3. sugarR-silence-N: silence the single most active neuron, observe MN9

Documented deviations (machine constraints, this box: 2 cores / ~4 GB RAM):
  - n_run   = 3 (notebook: 30)
  - n_proc  = 1 (notebook default: -1, all cores)
  - silencing top-1 instead of top-3 most active neurons
  - their run_exp is UNSEEDED (Poisson): tutorial reproduction demonstrates
    FUNCTIONING, not bit-reproducibility (that is proven separately by
    verify_reproducibility.py on the canonical v783 brain).
"""

import json
import sys
from pathlib import Path

import pandas as pd
from brian2 import Hz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "third_party" / "Drosophila_brain_model"))

import brainlib as bl  # noqa: E402

N_RUN = 3
OUT = bl.RESULTS / "example_630"

config = {
    "path_res": str(OUT),
    "path_comp": str(bl.VERSIONS["630"]["path_comp"]),
    "path_con": str(bl.VERSIONS["630"]["path_con"]),
    "n_proc": 1,
}

from model import run_exp, default_params  # noqa: E402  (third-party, unmodified)
import utils as utl  # noqa: E402

params = dict(default_params)
params["n_run"] = N_RUN
params100 = dict(params)
params100["r_poi"] = 100 * Hz


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- experiment 1: activation at default rate (150 Hz in the code)
    run_exp(exp_name="sugarR", neu_exc=bl.SUGAR_IDS, params=params, **config)

    # ---- experiment 2: activation at 100 Hz
    run_exp(exp_name="sugarR_100Hz", neu_exc=bl.SUGAR_IDS, params=params100, **config)

    # ---- rates + readouts
    df_spike = utl.load_exps([str(OUT / "sugarR.parquet"),
                              str(OUT / "sugarR_100Hz.parquet")])
    flyid2name = {f: f"sugar_{i+1}" for i, f in enumerate(bl.SUGAR_IDS)}
    df_rate, df_rate_std = utl.get_rate(df_spike, t_run=params["t_run"],
                                        n_run=N_RUN, flyid2name=flyid2name)
    df_rate.to_csv(OUT / "rates.csv")

    mn9 = bl.MN9_ID
    mn9_rates = {c: float(df_rate.loc[mn9, c]) for c in df_rate.columns
                 if c != "name"} if mn9 in df_rate.index else {}

    # ---- experiment 3: silence the single most active neuron (100 Hz run)
    df_sorted = df_rate.sort_values("sugarR_100Hz", ascending=False)
    top_id = int(df_sorted.index[0])
    run_exp(exp_name=f"sugarR-silence-{top_id}", neu_exc=bl.SUGAR_IDS,
            neu_slnc=[top_id], params=params100, **config)

    df_spike_s = utl.load_exps([str(OUT / f"sugarR-silence-{top_id}.parquet")])
    df_rate_s, _ = utl.get_rate(df_spike_s, t_run=params["t_run"], n_run=N_RUN,
                                flyid2name=flyid2name)
    mn9_rate_silenced = (float(df_rate_s.loc[mn9, df_rate_s.columns[-1]])
                         if mn9 in df_rate_s.index else 0.0)

    active = {
        "sugarR": int((df_rate["sugarR"] > 0).sum()),
        "sugarR_100Hz": int((df_rate["sugarR_100Hz"] > 0).sum()),
        f"sugarR-silence-{top_id}": int((df_rate_s.iloc[:, -1] > 0).sum()),
    }
    summary = {
        "version": "630 (as shipped for the paper)",
        "n_run": N_RUN,
        "n_proc": 1,
        "r_poi_hz": {"sugarR": 150, "sugarR_100Hz": 100},
        "silenced_top_neuron": top_id,
        "is_stimulated_neuron": top_id in bl.SUGAR_IDS,
        "total_spikes": int(len(df_spike) + len(df_spike_s)),
        "active_neurons": active,
        "mn9_rate_hz": {**mn9_rates,
                        f"sugarR-silence-{top_id}": mn9_rate_silenced},
        "notebook_expectation_check": {
            "active_neurons_scale_~400": active["sugarR"],
            "mn9_responds": (mn9_rates.get("sugarR", 0) > 0),
        },
        "deviations_from_notebook": [
            "n_run=3 instead of 30 (machine: 2 cores / 4 GB RAM)",
            "n_proc=1 instead of -1 (memory limit)",
            "silenced top-1 instead of top-3 most active neurons",
            "run_exp is unseeded (published behavior); reproducibility is "
            "established separately by verify_reproducibility.py (v783)",
        ],
    }
    bl.save_json(OUT / "tutorial_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
