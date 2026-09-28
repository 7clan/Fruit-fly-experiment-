"""Run ONE seeded trial with arbitrary stimulation/silencing ID sets.

Generic companion to verify_io_path.py (Gate-2 evidence): the exc/silence
neurons are passed as JSON files of FlyWire root IDs (from the D4-D6 map),
instead of the tutorial sugar set used by trial_runner.py.

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
    ap.add_argument("--exc-json", default="")       # file with list of flywire ids
    ap.add_argument("--silence-json", default="")   # file with list of flywire ids
    ap.add_argument("--spikes-out", required=True)
    ap.add_argument("--rec-out", required=True)
    args = ap.parse_args()

    exc_fids = json.loads(Path(args.exc_json).read_text()) if args.exc_json else []
    slnc_fids = json.loads(Path(args.silence_json).read_text()) if args.silence_json else []

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
