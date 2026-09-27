"""Pre-registered action-vocabulary reduction ladder + LOSO decoder.

Ladder (PHASE2_GATE_PREREGISTRATION.md section 4):
  L_A (5) FORWARD/BACKWARD/LEFT/RIGHT/PAUSE
  L_B (4) drop the cardinal with the lowest across-session mean occupancy;
          windows of the dropped cardinal are reassigned to the angularly
          nearest remaining cardinal (tie -> clockwise neighbor)
  L_C (3) the axis (horizontal LEFT/RIGHT or vertical FORWARD/BACKWARD) with
          the higher total occupancy + PAUSE; every moving window assigned
          by the sign of its displacement on the chosen axis
  L_D (2) MOVE / PAUSE
  L_E (1) MOVE

Selection: the LARGEST level whose states all satisfy G3 (occupancy >= 0.05
per session AND within the [0.5, 2.0] stability band) and G4 (leave-one-
session-out nearest-centroid accuracy >= 0.80; waived for L_E).
Fully deterministic.
"""
from __future__ import annotations

import math

import numpy as np

from .states import CARDINALS, STATES_A

AXIS = {  # cardinal -> axis sign
    "RIGHT": ("h", +1), "LEFT": ("h", -1),
    "FORWARD": ("v", +1), "BACKWARD": ("v", -1),
}
ANG = {"RIGHT": 0.0, "FORWARD": math.pi / 2, "LEFT": math.pi,
       "BACKWARD": -math.pi / 2}
CLOCKWISE = {"FORWARD": "RIGHT", "RIGHT": "BACKWARD",
             "BACKWARD": "LEFT", "LEFT": "FORWARD"}


# ------------------------------------------------------------------ ladder
def level_labels(level: str, windows: list[dict], drop_cardinal: str | None,
                 axis: str | None) -> list[str]:
    """Relabel window labels to the given level (deterministic)."""
    labs = [w["label"] for w in windows]
    if level == "L_A":
        return labs
    if level == "L_B":
        remaining = [c for c in CARDINALS if c != drop_cardinal]
        return [_nearest_cardinal(l, remaining) if l == drop_cardinal else l
                for l in labs]
    if level == "L_C":
        pair = ["LEFT", "RIGHT"] if axis == "h" else ["FORWARD", "BACKWARD"]
        out = []
        for w, l in zip(windows, labs):
            if l == "PAUSE":
                out.append("PAUSE")
                continue
            _, dx, dy, _ = w["features"]
            if axis == "h":
                out.append("RIGHT" if dx > 0 else
                           ("LEFT" if dx < 0 else
                            (l if l in pair else "LEFT")))
            else:
                out.append("FORWARD" if dy > 0 else
                           ("BACKWARD" if dy < 0 else
                            (l if l in pair else "FORWARD")))
        return out
    if level == "L_D":
        return ["MOVE" if l != "PAUSE" else "PAUSE" for l in labs]
    if level == "L_E":
        return ["MOVE"] * len(labs)
    raise ValueError(level)


def _nearest_cardinal(label: str, remaining: list[str]) -> str:
    if label in remaining:
        return label
    best, best_d = None, 1e9
    for c in remaining:
        d = abs(_wrap(ANG[c] - ANG[label]))
        if d < best_d - 1e-12 or (abs(d - best_d) <= 1e-12 and
                                  CLOCKWISE[label] == c):
            best, best_d = c, d
    return best or remaining[0]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------- LOSO
def nearest_centroid_loso(per_session_windows: dict[str, list[dict]],
                          labels_per_session: dict[str, list[str]]) -> dict:
    """Leave-one-session-out nearest-centroid accuracy (standardized
    features; deterministic ties broken by sorted class name)."""
    sessions = sorted(per_session_windows)
    if len(sessions) < 2:
        return {"accuracy": None, "chance": None, "per_class": {},
                "note": "needs >= 2 sessions"}
    X = {s: np.array([w["features"] for w in per_session_windows[s]],
                     dtype=float) for s in sessions}
    y = {s: np.array(labels_per_session[s], dtype=object) for s in sessions}
    correct = {s: 0 for s in sessions}
    total = 0
    per_class = {}
    for test in sessions:
        train = [s for s in sessions if s != test]
        Xtr = np.vstack([X[s] for s in train])
        ytr = np.concatenate([y[s] for s in train])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        classes = sorted(set(ytr.tolist()))
        cents = {c: ((Xtr[ytr == c] - mu) / sd).mean(0) for c in classes}
        Xte = (X[test] - mu) / sd
        for xi, yi in zip(Xte, y[test]):
            dists = [(float(np.sum((xi - cents[c]) ** 2)), c) for c in classes]
            dists.sort(key=lambda dc: (dc[0], dc[1]))
            pred = dists[0][1]
            total += 1
            key = str(yi)
            per_class.setdefault(key, {"n": 0, "correct": 0})
            per_class[key]["n"] += 1
            if pred == yi:
                correct[test] += 1
                per_class[key]["correct"] += 1
    acc = sum(correct.values()) / max(1, total)
    all_y = np.concatenate([y[s] for s in sessions])
    vals, counts = np.unique(all_y.astype(str), return_counts=True)
    chance = float(counts.max() / counts.sum())
    return {"accuracy": float(acc), "chance": chance,
            "per_class": {k: {"n": v["n"],
                              "accuracy": v["correct"] / max(1, v["n"])}
                          for k, v in sorted(per_class.items())}}


# --------------------------------------------------------------- selection
def evaluate_ladder(per_session_windows: dict[str, list[dict]],
                    occupancy_min=0.05, stability_band=(0.5, 2.0),
                    loso_min=0.80) -> dict:
    """Evaluate L_A..L_E; select the largest passing level (deterministic)."""
    sessions = sorted(per_session_windows)
    # across-session occupancy for construction decisions
    all_windows = [w for s in sessions for w in per_session_windows[s]]
    n_all = len(all_windows)
    mean_occ = {}
    for s in STATES_A:
        mean_occ[s] = sum(1 for w in all_windows if w["label"] == s) / max(1, n_all)
    card_means = {c: mean_occ[c] for c in CARDINALS}
    drop_cardinal = min(sorted(card_means), key=lambda c: card_means[c])
    axis_totals = {"h": mean_occ["LEFT"] + mean_occ["RIGHT"],
                   "v": mean_occ["FORWARD"] + mean_occ["BACKWARD"]}
    axis = "h" if axis_totals["h"] >= axis_totals["v"] else "v"

    levels = ["L_A", "L_B", "L_C", "L_D", "L_E"]
    evidence = []
    selected = None
    for lvl in levels:
        labels = {s: level_labels(lvl, per_session_windows[s],
                                  drop_cardinal, axis) for s in sessions}
        states = sorted(set(labels[sessions[0]])) if sessions else []
        n_states = len(states)
        # G3: occupancy per session + stability band
        occ = {}
        g3 = True
        for s in sessions:
            labs = labels[s]
            n = len(labs)
            occ[s] = {st: labs.count(st) / max(1, n) for st in states}
        for st in states:
            vals = [occ[s].get(st, 0.0) for s in sessions]
            m = float(np.mean(vals)) if vals else 0.0
            for v in vals:
                if v < occupancy_min or not (stability_band[0] * m <= v
                                             <= stability_band[1] * m):
                    g3 = False
        # G4: LOSO (waived for single-class levels)
        if n_states >= 2:
            loso = nearest_centroid_loso(per_session_windows, labels)
            g4 = loso["accuracy"] is not None and loso["accuracy"] >= loso_min
        else:
            loso = {"accuracy": None, "chance": None, "per_class": {}}
            g4 = True  # L_E: waived (trivial decoder)
        passed = g3 and g4
        evidence.append({
            "level": lvl, "n_states": n_states, "states": states,
            "occupancy_by_session": occ,
            "occupancy_min_observed": (min([occ[s][st] for s in sessions
                                            for st in states])
                                       if sessions and states else None),
            "G3_pass": g3, "G4_pass": g4, "G4_loso": loso,
            "pass": passed,
        })
        if passed and selected is None:
            selected = lvl
    return {
        "selected_level": selected,
        "n_states_selected": next((e["n_states"] for e in evidence
                                   if e["level"] == selected), None),
        "construction": {"dropped_cardinal_L_B": drop_cardinal,
                         "L_C_axis": axis},
        "evidence": evidence,
        "minimum_for_directional_control": "L_C",
        "directional_control_possible": selected in
                                        ("L_A", "L_B", "L_C") and selected is not None,
    }
