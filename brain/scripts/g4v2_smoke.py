"""One-shot smoke test of the FULL v2 session code path on a truncated
battery (every trial kind + snapshots + summarize + weights), into a
scratch dir. Deleted after inspection; NOT part of the matrix.

Usage: python g4v2_smoke.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g4v2_core as core
import g4v2_sessions as gs

OUT = Path(HERE.parent / "results" / "gate4_v2" / "_smoke")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fz = gs.frozen()
    print("frozen:", fz["n_cs_per_side"], fz["cs_rate_left_hz"],
          fz["cs_rate_right_hz"], flush=True)
    t0 = time.perf_counter()
    rt = core.V2Runtime(master_seed=999, n_cs=fz["n_cs_per_side"],
                        plasticity=True)
    print(f"build {time.perf_counter()-t0:.0f}s", flush=True)
    rl, rr = fz["cs_rate_left_hz"], fz["cs_rate_right_hz"]
    trials = []
    prev_gate = (0.0, 0.0)
    plan = [
        ("baseline", "pref", {}), ("baseline", "pref", {}),
        ("acquisition", "pair", {"side": "right"}),
        ("acquisition", "pair", {"side": "right"}),
        ("acquisition", "pref", {}),
        ("acquisition", "routea", {}),
        ("retention", "filler", {}),
        ("retention", "pref", {}),
        ("extinction", "pref", {}),
        ("reversal", "pair", {"side": "left"}),
        ("reversal", "pref", {}),
        ("context", "pref", {"ctx": 0.8}),
        ("distractor", "pref", {"loom": True}),
    ]
    for idx, (phase, kind, kw) in enumerate(plan):
        seed = 999 + idx
        if kind == "pref":
            rec = rt.pref_trial(seed, rl, rr, phase=phase, trial_idx=idx,
                                ctx_scale=kw.get("ctx", 1.0),
                                loom_precursor=kw.get("loom", False))
            rt.pl.relax(1)
        elif kind == "pair":
            rec = rt.pairing_trial(seed, kw["side"],
                                   rl if kw["side"] == "left" else rr,
                                   trial_idx=idx, us_on=True,
                                   plastic_update=True)
            prev_gate = (rec["pam_us_hz"], rec["ppl1_us_hz"])
        elif kind == "filler":
            rec = rt.filler_trial(seed, trial_idx=idx)
        elif kind == "routea":
            rec = rt.routea_trial(seed, rl, rr, trial_idx=idx)
            rt.pl.relax(1)
        rec["phase"], rec["trial"], rec["battery_kind"] = phase, idx, kind
        trials.append(rec)
        print(f"[{idx:2d}] {phase:<11} {kind:<6} choice_b="
              f"{rec.get('choice_b', rec.get('choice'))} V="
              f"{rec.get('V', rec.get('valence_bias_end'))} pam="
              f"{rec.get('pam_us_hz', 0)} elig="
              f"{rec.get('elig_left')}/{rec.get('elig_right')}",
              flush=True)
    summ = gs.summarize(trials, "_smoke", "B_plastic", "right")
    np.savez_compressed(OUT / "weights.npz", scale=rt.pl.scale)
    (OUT / "session.json").write_text(json.dumps(summ, indent=1))
    (OUT / "trials.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trials))
    print("final weight stats:", json.dumps(rt.pl.stats()))
    print("summary keys:", sorted(summ))
    print(f"SMOKE OK total {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
