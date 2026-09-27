"""D1: reproduce the shipped example.ipynb tutorial (FlyWire v630, as the
paper shipped it) with reduced n_run for this machine.

Faithful to the notebook's experiment sequence:
  1. sugarR          : activate the 21 sugar-sensing neurons (code default r_poi=150Hz)
  2. sugarR_100Hz    : same, r_poi = 100 Hz
  3. sugarR-silence-N: silence the single most active neuron, observe MN9

Documented deviations (machine constraints, this box: 2 cores / ~4 GB RAM):
  - n_run   = 3 (notebook: 30)
  - n_proc  = 1 (notebook default: -1)
  - silencing top-1 instead of top-3 most active neurons
  - each experiment runs in its OWN process (a single process holding
    several sequential Brian2 whole-brain rebuilds was OOM-killed here;
    the model's run_exp skips existing outputs, so stages resume cleanly)
  - their run_exp is UNSEEDED (Poisson): tutorial reproduction demonstrates
    FUNCTIONING, not bit-reproducibility (proven separately on v783 by
    verify_reproducibility.py).

Usage:
  python scripts/run_example.py --stage exp1     # sugarR
  python scripts/run_example.py --stage exp2     # sugarR_100Hz
  python scripts/run_example.py --stage exp3     # silence top-1 (from exp2 rates)
  python scripts/run_example.py --stage summarize
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "third_party" / "Drosophila_brain_model"))

import brainlib as bl  # noqa: E402

N_RUN = 3
OUT = bl.RESULTS / "example_630"
VENV_PY = bl.ROOT / ".venv" / "bin" / "python"

config = {
    "path_res": str(OUT),
    "path_comp": str(bl.VERSIONS["630"]["path_comp"]),
    "path_con": str(bl.VERSIONS["630"]["path_con"]),
    "n_proc": 1,
}


def py_env():
    import os
    env = dict(os.environ)
    env["MALLOC_ARENA_MAX"] = "2"
    return env


def run_exp_subprocess(exp_name, params_mods, neu_slnc=None):
    """Run exactly ONE run_exp call in a fresh interpreter (memory safety)."""
    code = f"""
import sys
sys.path.insert(0, {str(HERE.parent / 'third_party' / 'Drosophila_brain_model')!r})
from brian2 import Hz
from model import run_exp, default_params
params = dict(default_params)
params['n_run'] = {N_RUN}
{chr(10).join(f"params[{k!r}] = {v}" for k, v in params_mods.items())}
run_exp(exp_name={exp_name!r}, neu_exc={bl.SUGAR_IDS!r},
        neu_slnc={neu_slnc or []!r}, params=params, **{config!r})
"""
    print(f">>> [subprocess] experiment {exp_name} ...", flush=True)
    r = subprocess.run([str(VENV_PY), "-c", code], env=py_env())
    if r.returncode != 0:
        raise SystemExit(f"experiment {exp_name} FAILED (exit {r.returncode})")


def stage_exp1():
    run_exp_subprocess("sugarR", {})


def stage_exp2():
    run_exp_subprocess("sugarR_100Hz", {"r_poi": "100 * Hz"})


def stage_exp3():
    top_id = compute_rates()["_silence_target"]
    print(f">>> silencing top active neuron: {top_id}")
    run_exp_subprocess(f"sugarR-silence-{top_id}", {"r_poi": "100 * Hz"},
                       neu_slnc=[top_id])


def compute_rates():
    """Rates + readouts from the finished parquets (pandas only, no brian2)."""
    import pandas as pd
    import utils as utl  # third-party (pandas-only module)
    flyid2name = {f: f"sugar_{i+1}" for i, f in enumerate(bl.SUGAR_IDS)}
    df_spike = utl.load_exps([str(OUT / "sugarR.parquet"),
                              str(OUT / "sugarR_100Hz.parquet")])
    df_rate, df_rate_std = utl.get_rate(df_spike, t_run=1.0, n_run=N_RUN,
                                        flyid2name=flyid2name)
    df_rate.to_csv(OUT / "rates.csv")
    mn9 = bl.MN9_ID
    mn9_rates = ({c: float(df_rate.loc[mn9, c]) for c in ("sugarR", "sugarR_100Hz")}
                 if mn9 in df_rate.index else {})
    df_sorted = df_rate.sort_values("sugarR_100Hz", ascending=False)
    top_id = int(df_sorted.index[0])
    return {"mn9_rates": mn9_rates, "_silence_target": top_id,
            "active_sugarR": int((df_rate["sugarR"] > 0).sum()),
            "active_100Hz": int((df_rate["sugarR_100Hz"] > 0).sum())}


def stage_summarize():
    import pandas as pd
    import utils as utl
    base = compute_rates()
    top_id = base["_silence_target"]
    p = OUT / f"sugarR-silence-{top_id}.parquet"
    if p.is_file():
        df_s = utl.load_exps([str(p)])
        df_rate_s, _ = utl.get_rate(df_s, t_run=1.0, n_run=N_RUN)
        col = [c for c in df_rate_s.columns if c != "name"][-1]
        mn9_sil = (float(df_rate_s.loc[bl.MN9_ID, col])
                   if bl.MN9_ID in df_rate_s.index else 0.0)
        active_sil = int((df_rate_s[col] > 0).sum())
        spikes_sil = int(len(df_s))
    else:
        mn9_sil, active_sil, spikes_sil = None, None, None

    summary = {
        "version": "630 (as shipped for the paper)",
        "n_run": N_RUN,
        "n_proc": 1,
        "r_poi_hz": {"sugarR": 150, "sugarR_100Hz": 100},
        "silenced_top_neuron": top_id,
        "is_stimulated_neuron": top_id in bl.SUGAR_IDS,
        "active_neurons": {"sugarR": base["active_sugarR"],
                           "sugarR_100Hz": base["active_100Hz"],
                           "silenced": active_sil},
        "mn9_rate_hz": {**base["mn9_rates"], "silenced": mn9_sil},
        "notebook_expectation_check": {
            "active_neurons_order_of_magnitude_~400": base["active_sugarR"],
            "mn9_responds": base["mn9_rates"].get("sugarR", 0) > 0,
            "silencing_changes_mn9": (mn9_sil is not None and
                                      abs(mn9_sil - base["mn9_rates"].get("sugarR_100Hz", 0)) > 1e-9),
        },
        "deviations_from_notebook": [
            "n_run=3 instead of 30 (machine: 2 cores / 4 GB RAM)",
            "n_proc=1 instead of -1 (memory limit)",
            "silenced top-1 instead of top-3 most active neurons",
            "one experiment per process (OOM otherwise)",
            "run_exp is unseeded (published behavior); reproducibility is "
            "established separately by verify_reproducibility.py (v783)",
        ],
    }
    bl.save_json(OUT / "tutorial_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["exp1", "exp2", "exp3", "summarize"])
    args = ap.parse_args()
    {"exp1": stage_exp1, "exp2": stage_exp2,
     "exp3": stage_exp3, "summarize": stage_summarize}[args.stage]()
