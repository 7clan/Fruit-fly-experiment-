"""One trial = one episode of the 2D game driven by one controller.

Frame loop (identical for every controller - control-group validity):
    observe() -> act() -> set_action() -> physics step -> reward -> log
"""
from __future__ import annotations

from ..environment.arena2d import Arena2D
from ..learning.reward import RewardFunction
from ..controller.base import Controller
from ..controller.fly_controller import FlyController
from ..data.database import Database


def run_trial(trial_number: int, controller: Controller, arena: Arena2D,
              reward_fn: RewardFunction, fps: int, log_stride: int,
              db: Database, experiment_id: str, dashboard=None,
              save_frames: bool = False, run_logger=None) -> dict:
    dt = 1.0 / fps
    arena.reset()
    reward_fn.reset(arena.distance_to_target)
    controller.reset({"rng": arena.rng, "trial_index": trial_number})

    outcome = None
    reward_total = 0.0
    frame = 0
    buffered_events: list[tuple[int, str, dict]] = []
    buffered_frames: list[tuple[int, dict | None, tuple, float, float]] = []
    prev_lost = False
    fly_ctrl = isinstance(controller, FlyController)

    while True:
        t_ms = int(frame * 1000.0 / fps)
        state = arena.state()

        controller.observe(state, t_ms)
        action = controller.act()

        n0 = len(arena.events)          # BEFORE set_action: ACTION events
        if action is not None:          # must be captured for the DB too
            arena.set_action(action)

        arena.step(dt)
        new_events = arena.events[n0:]
        kinds = [e.kind for e in new_events]

        comps = reward_fn.step(arena.distance_to_target, kinds, dt, arena.t)
        reward_total += comps.total

        for e in new_events:
            buffered_events.append((t_ms, e.kind, e.payload))

        # fly-side tracking-loss transition (apparatus diagnostics)
        if fly_ctrl:
            lost = controller.behavior is not None and not controller.behavior.found
            if lost and not prev_lost:
                buffered_events.append((t_ms, "TRACK_LOST", {}))
            prev_lost = lost

        fly_snapshot = None
        if fly_ctrl and controller.behavior is not None:
            b = controller.behavior
            fly_snapshot = {
                "x": b.x, "y": b.y, "speed": b.speed_mm_s,
                "heading": b.heading_rad, "state": b.state, "zone": b.zone,
                "action": action, "confidence": b.confidence,
            }
        elif action is not None:
            fly_snapshot = {"action": action}

        if frame % max(1, log_stride) == 0:
            buffered_frames.append((t_ms, fly_snapshot, arena.player,
                                    arena.distance_to_target, comps.total))

        if save_frames and run_logger is not None:
            from ..environment.render import render_game
            run_logger.dump_frame(render_game(arena), trial_number, t_ms)

        if dashboard is not None and dashboard.enabled:
            key = dashboard.show(arena, controller, action, reward_total,
                                 trial_number)
            if key is not None and getattr(controller, "name", "") == "keyboard":
                controller.feed_key(key)

        if arena.reached():
            outcome = "SUCCESS"
            break
        if arena.timed_out():
            outcome = "TIMEOUT"
            break
        frame += 1

    result = {
        "trial_number": trial_number,
        "started_at_ms": 0,
        "duration_s": round(arena.t, 3),
        "outcome": outcome,
        "time_to_target_s": round(arena.t, 3) if outcome == "SUCCESS" else None,
        "path_length": round(arena.path_length, 1),
        "straight_dist": round(arena.straight_dist, 1),
        "path_efficiency": round(arena.path_efficiency, 4),
        "n_actions": arena.n_actions,
        "n_action_changes": arena.n_action_changes,
        "mean_inter_action_s": round(arena.mean_inter_action_s, 4),
        "reward_total": round(reward_total + reward_fn.total * 0.0, 4),
        "target_x": round(arena.target[0], 1), "target_y": round(arena.target[1], 1),
        "start_x": round(arena.start[0], 1), "start_y": round(arena.start[1], 1),
    }
    # NOTE reward_fn.total accumulates the same components already summed
    # into reward_total via comps.total; use the running sum for clarity.
    result["reward_total"] = round(reward_total, 4)

    trial_id = db.insert_trial(experiment_id, result)
    for (t_ms, kind, payload) in buffered_events:
        db.insert_event(trial_id, t_ms, kind, payload)
    for (t_ms, fly_snapshot, player, dist, rew) in buffered_frames:
        db.insert_frame(trial_id, t_ms, fly_snapshot, player, dist, rew)
    db.commit()

    controller.on_trial_end(result)
    return result
