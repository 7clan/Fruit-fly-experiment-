"""Mine stable combat-control priors from human teacher trajectories.

The result is descriptive imitation data, not a reward signal. Missing actions
mean "not demonstrated enough", never "bad action". Priors are only activated
when minimum coverage gates are met.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _median(values):
    return round(float(statistics.median(values)), 4) if values else None


def _percentile(values, q: float):
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return round(xs[0], 4)
    pos = (len(xs) - 1) * float(q)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    frac = pos - lo
    return round(xs[lo] * (1.0 - frac) + xs[hi] * frac, 4)


def mine_teacher_priors(trajectory_paths) -> dict:
    onset_counts = Counter()
    sample_counts = Counter()
    episode_durations = defaultdict(list)
    m1_combo_intervals = []
    ability_onsets = Counter()
    focused_samples = 0
    session_count = 0
    total_duration_s = 0.0

    for path in [Path(p) for p in trajectory_paths]:
        if not path.exists():
            continue
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            teacher = row.get("teacher_action") or {}
            if not teacher.get("enabled"):
                continue
            rows.append(row)
        if not rows:
            continue
        rows.sort(key=lambda r: int(r.get("ts_ns") or 0))
        session_count += 1
        if len(rows) >= 2:
            total_duration_s += max(
                0.0,
                (int(rows[-1].get("ts_ns") or 0)
                 - int(rows[0].get("ts_ns") or 0)) / 1e9,
            )

        prev = set()
        starts = {}
        last_m1_onset = None
        last_ts_s = None

        for row in rows:
            teacher = row.get("teacher_action") or {}
            ts_s = int(row.get("ts_ns") or 0) / 1e9
            last_ts_s = ts_s
            focused = bool(teacher.get("focused"))
            acts = set(teacher.get("actions") or []) if focused else set()
            if focused:
                focused_samples += 1
                sample_counts.update(acts)

            for action in prev - acts:
                st = starts.pop(action, None)
                if st is not None and ts_s >= st:
                    episode_durations[action].append(ts_s - st)

            for action in acts - prev:
                onset_counts[action] += 1
                starts[action] = ts_s
                if action.startswith("key_"):
                    ability_onsets[action] += 1
                if action == "m1":
                    if last_m1_onset is not None:
                        dt = ts_s - last_m1_onset
                        # Consecutive M1 presses within one local combo/burst.
                        if 0.08 <= dt <= 0.65:
                            m1_combo_intervals.append(dt)
                    last_m1_onset = ts_s
            prev = acts

        if last_ts_s is not None:
            for action, st in list(starts.items()):
                if last_ts_s >= st:
                    episode_durations[action].append(last_ts_s - st)

    move_actions = (
        "move_forward", "move_backward", "move_left", "move_right")
    attack_ready = bool(
        onset_counts["m1"] >= 100 and len(m1_combo_intervals) >= 80)
    movement_ready = bool(
        all(onset_counts[a] >= 20 for a in move_actions))
    defense_ready = bool(
        onset_counts["block"] >= 20
        and onset_counts["dash_or_evade"] >= 20)
    ability_ready = bool(sum(ability_onsets.values()) >= 20)

    result = {
        "schema_version": 1,
        "source": "human_teacher_demonstrations",
        "descriptive_not_reward": True,
        "sessions": int(session_count),
        "focused_samples": int(focused_samples),
        "total_duration_s": round(float(total_duration_s), 2),
        "readiness": {
            "attack_timing": attack_ready,
            "movement": movement_ready,
            "defense": defense_ready,
            "abilities": ability_ready,
        },
        "coverage": {
            "onsets": dict(sorted(onset_counts.items())),
            "samples": dict(sorted(sample_counts.items())),
            "ability_onsets": dict(sorted(ability_onsets.items())),
        },
        "timing": {
            "m1_inter_onset_median_s": _median(m1_combo_intervals),
            "m1_inter_onset_p90_s": _percentile(m1_combo_intervals, 0.90),
            "m1_hold_median_s": _median(episode_durations["m1"]),
            "movement_hold_median_s": {
                a: _median(episode_durations[a]) for a in move_actions
            },
            "ability_hold_median_s": {
                a: _median(episode_durations[a])
                for a in sorted(ability_onsets)
            },
        },
        "warnings": [],
    }
    if not defense_ready:
        result["warnings"].append(
            "Defense is not training-ready: collect >=20 BLOCK and >=20 "
            "dash/evade onsets before learning defensive policy.")
    if not attack_ready:
        result["warnings"].append(
            "Attack timing is not training-ready: collect more M1 bursts.")
    if not movement_ready:
        result["warnings"].append(
            "Movement is not training-ready in every direction.")
    return result


def rebuild_teacher_priors(training_root: Path) -> dict:
    root = Path(training_root)
    paths = sorted((root / "trajectories").glob("*.jsonl"))
    result = mine_teacher_priors(paths)
    out = root / "teacher_priors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(out)
    return result


def load_teacher_priors(training_root: Path) -> dict:
    path = Path(training_root) / "teacher_priors.json"
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}
