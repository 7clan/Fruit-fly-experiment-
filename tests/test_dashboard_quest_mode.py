from lab.dashboard.dashboard import OpenCVDashboardRenderer, Snapshot


def test_quest_mode_opencv_compose_reads_quest_state_and_command():
    r = OpenCVDashboardRenderer(width=900, height=810, lightweight=True)
    snap = Snapshot(1, {
        "world.observation": {
            "notes": {"quest_enemy_marker_detected": True},
            "player": {"health": 0.9, "stamina": 0.8},
            "target": {"type": "quest_marker"},
        },
        "brain.output": {},
        "fly.channels": {},
        "helper.goal": {"goal": {"label": "starter PvE"}},
        "action.selected": {"autonomy": False},
        "brain.meta": {"ready": True},
        "action.meta": {
            "mode": "quest_pve_v1",
            "movement_control_available": True,
            "autonomy": False,
        },
        "quest.state": {
            "phase": "travel_to_enemy",
            "defense_values": {"block": 0.6, "evade_back": 0.4},
        },
        "action.command": {"name": "block"},
    })
    canvas = r._compose(snap, None)
    assert canvas.shape == (810, 900, 3)
