"""Trial-to-trial (session-to-session) variability of behavioral states.

Metrics: pairwise Jensen-Shannon divergence (base 2, normalized to [0, 1])
between sessions' state-occupancy distributions, and per-state occupancy
range across sessions.
"""
from __future__ import annotations

import numpy as np


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2) between two distributions."""
    p = np.asarray(p, float) + 1e-12
    q = np.asarray(q, float) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def session_variability(occupancy_by_session: dict[str, dict],
                        states: list[str]) -> dict:
    """occupancy_by_session: session -> state -> fraction."""
    sessions = sorted(occupancy_by_session)
    mat = np.array([[occupancy_by_session[s].get(st, 0.0) for st in states]
                    for s in sessions]) if sessions else np.zeros((0, len(states)))
    out = {
        "n_sessions": len(sessions),
        "states": states,
        "occupancy_range": {},
        "jsd_matrix": {},
        "mean_pairwise_jsd": None,
        "max_pairwise_jsd": None,
    }
    if len(sessions) < 2:
        return out
    for j, st in enumerate(states):
        col = mat[:, j]
        out["occupancy_range"][st] = {
            "min": float(col.min()), "max": float(col.max()),
            "mean": float(col.mean()), "std": float(col.std()),
        }
    ds = []
    for i, a in enumerate(sessions):
        out["jsd_matrix"][a] = {}
        for b in sessions:
            d = jsd(mat[i], mat[sessions.index(b)]) if a < b else \
                out["jsd_matrix"].get(b, {}).get(a)
            if a < b:
                out["jsd_matrix"][a][b] = d
                ds.append(d)
            else:
                out["jsd_matrix"][a][b] = d if d is not None else 0.0 \
                    if a == b else out["jsd_matrix"].get(b, {}).get(a, 0.0)
    out["mean_pairwise_jsd"] = float(np.mean(ds)) if ds else None
    out["max_pairwise_jsd"] = float(np.max(ds)) if ds else None
    return out
