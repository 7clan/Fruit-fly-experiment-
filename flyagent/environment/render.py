"""OpenCV renderer for the 2D arena + combined dashboard frame.

Rendering is OPTIONAL: in headless experiment runs nothing is rendered,
which keeps the Phase-1 demo fast. With --show (user machine with a
display) the dashboard displays:
  [ fly camera view | 2D game | status panel ].
"""
from __future__ import annotations

import cv2
import numpy as np


def render_game(arena, info: dict | None = None) -> np.ndarray:
    w, h = arena.w, arena.h
    frame = np.full((h, w, 3), 38, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)

    if arena.scenario == "obstacles":
        for (ox, oy, ow, oh) in arena.obstacles:
            cv2.rectangle(frame, (int(ox), int(oy)),
                          (int(ox + ow), int(oy + oh)), (80, 80, 95), -1)

    tx, ty = int(arena.target[0]), int(arena.target[1])
    cv2.circle(frame, (tx, ty), int(arena.target_radius) + 6, (90, 80, 20), 1)
    cv2.circle(frame, (tx, ty), int(arena.target_radius), (60, 190, 240), -1)

    px, py = int(arena.player[0]), int(arena.player[1])
    cv2.circle(frame, (px, py), int(arena.player_radius), (230, 170, 60), -1)
    cv2.circle(frame, (px, py), int(arena.player_radius) + 2, (150, 110, 30), 1)

    info = info or {}
    lines = [
        f"trial {info.get('trial', '?')}   t {arena.t:5.1f}s",
        f"action {info.get('action', '-'):8s} reward {info.get('reward', 0.0):6.2f}",
        f"dist {arena.distance_to_target:6.0f}px",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def render_status_panel(rows: list[tuple[str, str]], width: int = 360,
                        height: int = 280) -> np.ndarray:
    frame = np.full((height, width, 3), 24, dtype=np.uint8)
    for i, (k, v) in enumerate(rows):
        y = 24 + 20 * i
        if y > height - 8:
            break
        cv2.putText(frame, f"{k:<14}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (140, 200, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, str(v), (130, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return frame


def compose_dashboard(game_frame, fly_frame, status_rows) -> np.ndarray:
    def _fit(img, h):
        if img is None:
            return np.full((h, 10, 3), 24, dtype=np.uint8)
        if img.shape[0] != h:
            s = h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * s), h))
        return img

    h = 480
    g = _fit(game_frame, h)
    f = _fit(fly_frame, h)
    s = render_status_panel(status_rows, width=max(240, 900 - g.shape[1] - f.shape[1]), height=h)
    return np.hstack([f, g, s])
