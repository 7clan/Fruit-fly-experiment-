"""Export collected trajectories into offline training datasets.

This script does not call any external training service. It converts the
persistent trajectory JSONL files into three explicit datasets:

- teacher_bc.jsonl: state -> human action examples for local behavior cloning
- gemini_sft_candidates.jsonl: positive-outcome state -> cloud plan examples
- rlft_episodes.jsonl: state/action/reward records for reward-based tuning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact_state(row):
    return {
        "world": row.get("world") or {},
        "quest": row.get("quest") or {},
        "ai_track": row.get("ai_track") or {},
        "action_meta": row.get("action_meta") or {},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runtime_state/training")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root)
    traj_dir = root / "trajectories"
    out_dir = Path(args.out) if args.out else root / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_path = out_dir / "teacher_bc.jsonl"
    sft_path = out_dir / "gemini_sft_candidates.jsonl"
    rlft_path = out_dir / "rlft_episodes.jsonl"

    teacher_n = sft_n = rlft_n = 0
    with open(teacher_path, "w", encoding="utf-8") as teacher_fh, \
         open(sft_path, "w", encoding="utf-8") as sft_fh, \
         open(rlft_path, "w", encoding="utf-8") as rlft_fh:
        for path in sorted(traj_dir.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                reward = row.get("reward") or {}
                reward_total = float(reward.get("total") or 0.0)

                teacher = row.get("teacher_action") or {}
                if teacher.get("focused"):
                    teacher_fh.write(json.dumps({
                        "session_id": row.get("session_id"),
                        "ts_ns": row.get("ts_ns"),
                        "input": compact_state(row),
                        "target_actions": teacher.get("actions") or [],
                        "keys_down": teacher.get("keys_down"),
                        "mouse_left": teacher.get("mouse_left"),
                        "mouse_right": teacher.get("mouse_right"),
                    }, default=str) + "\n")
                    teacher_n += 1

                plan = row.get("plan") or {}
                if plan and reward_total > 0:
                    sft_fh.write(json.dumps({
                        "session_id": row.get("session_id"),
                        "ts_ns": row.get("ts_ns"),
                        "input": compact_state(row),
                        "target_plan": plan,
                        "observed_reward": reward,
                    }, default=str) + "\n")
                    sft_n += 1

                if plan or (row.get("command") or {}):
                    rlft_fh.write(json.dumps({
                        "session_id": row.get("session_id"),
                        "ts_ns": row.get("ts_ns"),
                        "input": compact_state(row),
                        "plan": plan,
                        "command": row.get("command") or {},
                        "reward": reward,
                    }, default=str) + "\n")
                    rlft_n += 1

    print(json.dumps({
        "teacher_bc_examples": teacher_n,
        "gemini_sft_candidates": sft_n,
        "rlft_records": rlft_n,
        "teacher_bc": str(teacher_path),
        "gemini_sft_candidates": str(sft_path),
        "rlft_episodes": str(rlft_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
