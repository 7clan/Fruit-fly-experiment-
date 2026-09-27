"""Experiment runner: N trials of one condition, fully logged."""
from __future__ import annotations

import datetime
import numpy as np

from ..config import deep_merge
from ..data.database import Database
from ..data.logger import RunLogger, new_run_dir
from ..environment.arena2d import Arena2D
from ..learning.reward import RewardFunction
from ..controller import build_controller
from ..version import __version__
from .trial import run_trial

# condition name -> (controller kind, fly variant overrides)
CONDITIONS = {
    "random":             {"controller": "random"},
    "fixed":              {"controller": "fixed"},
    "fly":                {"controller": "fly", "variant": {"cue_mode": "goal"}},
    "fly_random":         {"controller": "fly",
                           "variant": {"cue_mode": "goal", "cue_bias": 0.0}},
    "fly_cue":            {"controller": "fly", "variant": {"cue_mode": "goal"}},
    "fly_cue_shuffled":   {"controller": "fly", "variant": {"cue_mode": "shuffled"}},
    "fly_no_cue":         {"controller": "fly", "variant": {"cue_mode": "none"}},
}


def run_experiment(condition: str, cfg: dict, out_base: str,
                   seed: int, trials: int | None = None,
                   notes: str = "", dashboard=None) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; known: {list(CONDITIONS)}")
    spec = CONDITIONS[condition]
    exp = cfg["experiment"]
    trials = int(trials or exp["demo"]["trials_per_condition"])
    fps = int(exp["run"]["fps"])

    run_dir = new_run_dir(out_base, condition)
    logger = RunLogger(run_dir)
    db = Database(logger.db_path)
    rng = np.random.default_rng(seed)

    # fly variant overrides ride along in the config copy
    run_cfg = deep_merge(cfg, {"_fly_variant": spec.get("variant", {})})
    controller, bundle = build_controller(spec["controller"], run_cfg, rng)

    experiment_id = run_dir.name
    db.insert_experiment(
        experiment_id=experiment_id,
        started_at=datetime.datetime.now().isoformat(timespec="seconds"),
        controller=controller.name, condition=condition,
        config={k: v for k, v in run_cfg.items() if not k.startswith("_")},
        software_version=__version__, seed=seed,
        subject_id="synthetic", session_index=0, phase="na",
        scenario=exp["game"].get("scenario", "open_field"), notes=notes)
    logger.write_manifest(condition, controller.name, seed,
                          {k: v for k, v in run_cfg.items()
                           if not k.startswith("_")}, notes)

    arena = Arena2D(exp["game"], rng)
    reward_fn = RewardFunction(exp["reward"])
    trial_cfg = exp.get("trial", {})

    results = []
    for i in range(1, trials + 1):
        r = run_trial(i, controller, arena, reward_fn, fps,
                      int(trial_cfg.get("log_stride", 3)), db, experiment_id,
                      dashboard=dashboard,
                      save_frames=bool(trial_cfg.get("save_frames", False)),
                      run_logger=logger)
        results.append(r)
        marker = "OK " if r["outcome"] == "SUCCESS" else "-- "
        print(f"  [{condition}] trial {i:3d}/{trials} {marker}"
              f"t={r['duration_s']:5.1f}s actions={r['n_actions']:3d} "
              f"reward={r['reward_total']:6.2f}", flush=True)

    # ------------------------------------------------------------- aggregate
    succ = [r for r in results if r["outcome"] == "SUCCESS"]
    summary = {
        "experiment_id": experiment_id,
        "condition": condition,
        "controller": controller.name,
        "seed": int(seed),
        "trials": trials,
        "run_dir": str(run_dir),
        "aggregate": {
            "success_rate": len(succ) / trials,
            "median_time_to_target_s": float(np.median([r["time_to_target_s"] for r in succ])) if succ else None,
            "median_path_efficiency": float(np.median([r["path_efficiency"] for r in succ])) if succ else None,
            "mean_actions": float(np.mean([r["n_actions"] for r in results])),
            "mean_action_changes": float(np.mean([r["n_action_changes"] for r in results])),
            "mean_inter_action_s": float(np.mean([r["mean_inter_action_s"] for r in results])) if results else None,
            "mean_reward": float(np.mean([r["reward_total"] for r in results])),
        },
        "apparatus": {},
        "per_trial": results,
    }
    if bundle is not None and hasattr(controller, "lost_rate"):
        summary["apparatus"]["fly_frames"] = controller.n_frames
        summary["apparatus"]["tracking_lost_rate"] = round(controller.lost_rate, 4)
    logger.write_summary(summary)
    db.close()
    if bundle is not None:
        bundle.close()
    return summary
