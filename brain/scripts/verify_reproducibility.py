"""Gate-1 evidence: REPRODUCIBLE neural activity on the canonical v783 brain.

Protocol (engineered harness around the unmodified third-party model):
  * for each seed S in --seeds, run --reps seeded trials (in-process);
    trials with the same seed must produce BITWISE-IDENTICAL spike trains
    (same neurons, same times, same order);
  * different seeds must produce DIFFERENT trains (sensitivity: the
    comparison is not trivially empty);
  * cross-process reproducibility: run this script twice with different
    --out dirs and compare the canonical spike-file hashes
    (verify_hash.py or `sha256sum`).

Output: <out>/spikes_seed<S>_rep<R>.json (canonical form),
        <out>/verdict.json (PASS/FAIL + metrics).
"""

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402


def canonical_spikes(rec):
    """Deterministic serialization of a spike record (for hashing)."""
    lines = []
    for bi in sorted(rec["spikes"]):
        ts = ",".join(repr(t) for t in rec["spikes"][bi])
        lines.append(f"{bi}:{ts}")
    return "\n".join(lines)


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--t-run-ms", type=int, default=1000)
    ap.add_argument("--r-poi-hz", type=int, default=150)
    ap.add_argument("--out", default=str(bl.RESULTS / "repro_783"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = {}   # (seed, rep) -> record
    for rep in range(args.reps):
        for seed in args.seeds:
            print(f">>> trial seed={seed} rep={rep} ...", flush=True)
            rec = bl.run_seeded_trial(
                seed=seed, version="783", exc_fly_ids=bl.SUGAR_IDS,
                t_run_ms=args.t_run_ms, r_poi_hz=args.r_poi_hz)
            recs[(seed, rep)] = rec
            text = canonical_spikes(rec)
            (out / f"spikes_seed{seed}_rep{rep}.json").write_text(text)
            bl.save_json(out / f"trial_seed{seed}_rep{rep}.json", bl.public_rec(rec))

    # ---- checks
    in_proc = {}
    for seed in args.seeds:
        a = canonical_spikes(recs[(seed, 0)])
        b = canonical_spikes(recs[(seed, 1)])
        in_proc[seed] = {
            "identical": a == b,
            "sha256_rep0": sha256_text(a),
            "sha256_rep1": sha256_text(b),
            "n_active": recs[(seed, 0)]["n_active_neurons"],
            "n_spikes": recs[(seed, 0)]["n_spikes"],
        }

    sensitivity = None
    if len(args.seeds) >= 2:
        a = canonical_spikes(recs[(args.seeds[0], 0)])
        b = canonical_spikes(recs[(args.seeds[1], 0)])
        sensitivity = {
            "seeds": args.seeds,
            "differ": a != b,
            "n_active_a": recs[(args.seeds[0], 0)]["n_active_neurons"],
            "n_active_b": recs[(args.seeds[1], 0)]["n_active_neurons"],
        }

    verdict = {
        "version": "783 (canonical)",
        "params": {"t_run_ms": args.t_run_ms, "r_poi_hz": args.r_poi_hz,
                   "stimulus": "sugar-sensing set (20 of 21 present in v783)"},
        "in_process_deterministic": all(v["identical"] for v in in_proc.values()),
        "per_seed": in_proc,
        "seed_sensitivity": sensitivity,
        "peak_rss_kb": max(r["peak_rss_kb"] for r in recs.values()),
        "timings_s": {f"seed{s}_rep{r}": {"build": recs[(s, r)]["wall_build_s"],
                                          "run": recs[(s, r)]["wall_run_s"]}
                      for (s, r) in recs},
    }
    verdict["gate_reproducible_activity"] = bool(
        verdict["in_process_deterministic"] and
        (sensitivity is None or sensitivity["differ"]))
    bl.save_json(out / "verdict.json", verdict)
    print("\nGATE (reproducible activity):",
          "PASS" if verdict["gate_reproducible_activity"] else "FAIL")
    for s, v in in_proc.items():
        print(f"  seed {s}: identical={v['identical']} "
              f"({v['n_spikes']} spikes, {v['n_active']} active) "
              f"sha={v['sha256_rep0'][:16]}")
    if sensitivity:
        print(f"  sensitivity: seeds differ={sensitivity['differ']}")


if __name__ == "__main__":
    main()
