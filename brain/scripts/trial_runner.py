"""Run ONE seeded trial of the canonical v783 brain in a fresh process.

Invoked (as a subprocess) by verify_reproducibility.py / verify_spikes.py
so that each trial gets a clean process — the whole-brain network peaks
near 3 GB and repeated in-process rebuilds OOM on small machines.

Writes: --spikes-out (canonical text, hashable) + --rec-out (trial JSON).
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--t-run-ms", type=int, default=1000)
    ap.add_argument("--r-poi-hz", type=int, default=150)
    ap.add_argument("--exc", default="sugar", choices=["sugar", "none"])
    ap.add_argument("--silence-fid", type=int, default=0)
    ap.add_argument("--spikes-out", required=True)
    ap.add_argument("--rec-out", required=True)
    args = ap.parse_args()

    exc_fids = bl.SUGAR_IDS if args.exc == "sugar" else []
    slnc_fids = [args.silence_fid] if args.silence_fid else []

    rec = bl.run_seeded_trial(
        seed=args.seed, version="783", exc_fly_ids=exc_fids,
        slnc_fly_ids=slnc_fids, t_run_ms=args.t_run_ms,
        r_poi_hz=args.r_poi_hz)

    lines = []
    for bi in sorted(rec["spikes"]):
        lines.append(f"{bi}:" + ",".join(repr(t) for t in rec["spikes"][bi]))
    Path(args.spikes_out).write_text("\n".join(lines))
    bl.save_json(Path(args.rec_out), bl.public_rec(rec))
    print(json.dumps(bl.public_rec(rec), default=str))


if __name__ == "__main__":
    main()
