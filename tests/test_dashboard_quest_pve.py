import numpy as np

from lab.dashboard.dashboard import OpenCVDashboardRenderer, Snapshot


def test_opencv_quest_panel_reads_quest_state_and_command():
    renderer = OpenCVDashboardRenderer(
        width=1200, height=810, lightweight=True)
    snap = Snapshot(1, {
        "world.observation": {
            "player": {"health": 1.0, "stamina": 1.0},
            "target": {"type": "quest_marker"},
            "notes": {},
        },
        "brain.output": {},
        "fly.channels": {},
        "helper.goal": {"goal": {"label": "approach objective"}},
        "action.selected": {},
        "brain.meta": {"ready": False},
        "action.meta": {
            "mode": "quest_pve_v1",
            "movement_control_available": True,
            "autonomy": False,
        },
        "quest.state": {
            "phase": "quest_giver",
            "defense_values": {"block": 0.5, "evade_back": 0.5},
        },
        "action.command": {"name": "INTERACT_QUEST"},
    })
    canvas = renderer._compose(snap, None)
    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (810, 1200, 3)
