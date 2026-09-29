#!/usr/bin/env python3
"""Windows hardware benchmark CLI (FINAL_ARCHITECTURE §8, binding).

Run on the target Windows laptop BEFORE selecting final live rates:

    py -3 benchmark_windows.py              # full: hardware + pipeline + canonical brain
    py -3 benchmark_windows.py --no-canonical   # skip brian2 (fast)
    py -3 benchmark_windows.py --seconds 10     # longer pipeline soak

Canonical-brain benchmarks require the brain venv interpreter:

    brain\\.venv\\Scripts\\python.exe benchmark_windows.py

Outputs:
    WINDOWS_HARDWARE_REPORT.md    human-readable measured report
    config/live_settings.json     auto-chosen conservative live settings

Numbers are MEASURED, never fabricated. Non-Windows platforms are
labeled as such (sandbox/dev numbers are NOT Windows numbers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lab.benchmark.windows_hardware import (  # noqa: E402
    run_all, write_report, LIVE_SETTINGS_PATH)

REPORT_PATH = ROOT / "WINDOWS_HARDWARE_REPORT.md"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-canonical", action="store_true",
                    help="skip the canonical-brain chunk benchmarks")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="pipeline soak duration (s)")
    ap.add_argument("--json", action="store_true",
                    help="also dump the raw report JSON")
    args = ap.parse_args(argv)

    report = run_all(with_canonical=not args.no_canonical,
                     seconds=args.seconds)
    write_report(report, REPORT_PATH)
    if args.json:
        REPORT_PATH.with_suffix(".raw.json").write_text(
            json.dumps(report, indent=1, default=str))
    print(f"[benchmark] report     -> {REPORT_PATH}")
    print(f"[benchmark] settings   -> {LIVE_SETTINGS_PATH}")
    ls = report["live_settings"]
    latency_label = ("estimated_canonical_s2a_p95"
                     if "estimated_canonical_s2a_p95_ms" in ls
                     else "measured_s2a_p95")
    print(f"[benchmark] brain_hz={ls['brain_hz']} chunk_ms={ls['brain_chunk_ms']} "
          f"{latency_label}={ls['measured_s2a_p95_ms']}ms "
          f"target_met={ls['s2a_target_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
