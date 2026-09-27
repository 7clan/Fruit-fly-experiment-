"""Smoke test: one short seeded trial of the canonical v783 brain.

Purpose: feasibility check on THIS machine (2 cores, ~4 GB RAM) before the
full D1-D3 runs. Not itself gate evidence.
"""

import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import brainlib as bl

t0 = time.time()
rec = bl.run_seeded_trial(
    seed=1234,
    version="783",
    exc_fly_ids=bl.SUGAR_IDS,
    t_run_ms=200,
    r_poi_hz=150,
)
total = time.time() - t0
print(f"total wall      : {total:8.1f} s")
print(f"build wall      : {rec['wall_build_s']:8.1f} s")
print(f"run wall        : {rec['wall_run_s']:8.1f} s  (for {rec['t_run_ms']} ms bio-time)")
print(f"neurons         : {rec['n_neurons_total']}")
print(f"stim effective  : {rec['n_exc']}  (missing: {rec['missing_exc']})")
print(f"spikes          : {rec['n_spikes']} from {rec['n_active_neurons']} active neurons")
print(f"peak RSS        : {rec['peak_rss_kb']/1024/1024:.2f} GB")
