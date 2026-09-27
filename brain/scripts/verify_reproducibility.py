"""Gate-1 evidence: REPRODUCIBLE neural activity on the canonical v783 brain.

Every trial runs in its OWN fresh interpreter (trial_runner.py) — whole-
brain rebuilds peak near 3 GB. Determinism is therefore tested ACROSS
PROCESSES, which is stronger than in-process repetition:

  * seed S, rep 0  (process A)  vs
    seed S, rep 1  (process B)  -> canonical spike text must be
                                    BYTE-IDENTICAL (same hash)
  * different seeds must DIFFER (sensitivity: comparison not trivial)
  * --second-invocation mode re-runs one seed and compares against the
    first invocation's files (script-level reproducibility)

Output: <out>/spikes_seed<S>_rep<R>.txt (canonical), trial JSONs,
<out>/verdict.json.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402

VENV_PY = bl.ROOT / ".venv" / "bin" / "python"


def run_trial(seed, out, t_run_ms, r_poi_hz, tag):
    spikes = out / f"spikes_seed{seed}_{tag}.txt"
    rec = out / f"trial_seed{seed}_{tag}.json"
    cmd = [str(VENV_PY), str(HERE / "trial_runner.py"),
           "--seed", str(seed), "--t-run-ms", str(t_run_ms),
           "--r-poi-hz", str(r_poi_hz), "--exc", "sugar",
           "--spikes-out", str(spikes), "--rec-out", str(rec)]
    import os
    env = {**os.environ, "MALLOC_ARENA_MAX": "2"}
    print(f">>> trial seed={seed} tag={tag} (fresh process) ...", flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"trial seed={seed} tag={tag} FAILED")
    text = spikes.read_text()
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(),
            "size": len(text), "spikes_file": spikes.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--t-run-ms", type=int, default=1000)
    ap.add_argument("--r-poi-hz", type=int, default=150)
    ap.add_argument("--out", default=str(bl.RESULTS / "repro_783"))
    ap.add_argument("--compare-against", default="",
                    help="dir of a previous invocation; compare seed[0] hashes")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = {}
    for rep in range(args.reps):
        for seed in args.seeds:
            files[(seed, rep)] = run_trial(seed, out, args.t_run_ms,
                                           args.r_poi_hz, f"rep{rep}")

    per_seed = {}
    for seed in args.seeds:
        hashes = {rep: files[(seed, rep)]["sha256"]
                  for rep in range(args.reps)}
        identical = len(set(hashes.values())) == 1
        per_seed[seed] = {"identical_across_processes": identical,
                          "sha256": hashes[0], "hashes": hashes}

    sensitivity = None
    if len(args.seeds) >= 2:
        sensitivity = {
            "seeds": args.seeds,
            "differ": (per_seed[args.seeds[0]]["sha256"] !=
                       per_seed[args.seeds[1]]["sha256"]),
        }

    cross_invocation = None
    if args.compare_against.strip():
        prev = Path(args.compare_against.strip()) / f"spikes_seed{args.seeds[0]}_rep0.txt"
        if prev.is_file():
            prev_hash = hashlib.sha256(prev.read_text().encode()).hexdigest()
            cross_invocation = {
                "previous_dir": args.compare_against.strip(),
                "seed": args.seeds[0],
                "identical": prev_hash == per_seed[args.seeds[0]]["sha256"],
            }

    verdict = {
        "version": "783 (canonical)",
        "params": {"t_run_ms": args.t_run_ms, "r_poi_hz": args.r_poi_hz,
                   "stimulus": "sugar-sensing set (20 of 21 present in v783)",
                   "trials_per_seed": args.reps},
        "identical_across_processes": all(
            v["identical_across_processes"] for v in per_seed.values()),
        "per_seed": per_seed,
        "seed_sensitivity": sensitivity,
        "cross_invocation": cross_invocation,
        "engineered_note": "seeding is an engineered harness addition "
                           "(brian2.seed + numpy seed); the published model "
                           "code is unmodified and unseeded",
    }
    verdict["gate_reproducible_activity"] = bool(
        verdict["identical_across_processes"]
        and (sensitivity is None or sensitivity["differ"])
        and (cross_invocation is None or cross_invocation["identical"]))
    bl.save_json(out / "verdict.json", verdict)
    print("\nGATE (reproducible activity):",
          "PASS" if verdict["gate_reproducible_activity"] else "FAIL")
    for s, v in per_seed.items():
        print(f"  seed {s}: identical_across_processes="
              f"{v['identical_across_processes']} sha={v['sha256'][:16]}")
    if sensitivity:
        print(f"  sensitivity (seeds differ): {sensitivity['differ']}")
    if cross_invocation:
        print(f"  cross-invocation identical: {cross_invocation['identical']}")


if __name__ == "__main__":
    main()
