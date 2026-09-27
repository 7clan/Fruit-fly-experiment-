"""Figures for Phase-2 sessions and validation (matplotlib, headless)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _style():
    plt.rcParams.update({
        "figure.constrained_layout.use": True,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    })


def plot_session(an: dict, out_png: Path, tracks: dict | None = None) -> Path:
    """Overview figure for one session analysis dict (from analyze_session)."""
    _style()
    import json
    sdir = Path(an["session_dir"])
    from .tracking.tracker import load_tracks
    tr = load_tracks(sdir / "tracks.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    fig.suptitle(f"{an['label']} ({an['kind']}) — coverage "
                 f"{an['tracking']['coverage_conf05']:.1%}")

    # 1. trajectory (arena frame, mm)
    ax = axes[0, 0]
    ax.plot(tr["x_mm"], tr["y_mm"], lw=0.5, color="#348ABD", alpha=0.8)
    ax.plot(tr["x_mm"][0], tr["y_mm"][0], "g^", ms=6, label="start")
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(30 * np.cos(th), 30 * np.sin(th), "k--", lw=0.8)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_aspect("equal"); ax.legend(fontsize=7)
    ax.set_title("trajectory (arena frame)")

    # 2. speed distribution
    ax = axes[0, 1]
    sp = tr["x_mm"] * np.nan
    from .analysis.trajectory import compute_kinematics, interpolate_gaps
    x, y, _ = interpolate_gaps(tr["t"], tr["x_mm"], tr["y_mm"], tr["found"])
    kin = compute_kinematics(tr["t"], x, y)
    sp = kin["speed"][np.isfinite(kin["speed"])]
    ax.hist(sp, bins=40, color="#A60628", alpha=0.8)
    ax.axvline(2.0, color="k", ls=":", lw=1)
    ax.set_xlabel("speed (mm/s)"); ax.set_title("speed distribution")

    # 3. heading rose
    ax = fig.add_subplot(2, 3, 3, projection="polar")
    axes[0, 2].remove()
    h = kin["heading"][(np.isfinite(kin["heading"])) & (kin["speed"] >= 2.0)]
    counts, edges = np.histogram(h, bins=24, range=(-math.pi, math.pi))
    ax.bar(np.convolve(edges, [0.5, 0.5], "valid"), counts,
           width=2 * math.pi / 24 * 0.9, color="#7A68A6", alpha=0.8)
    ax.set_theta_zero_location("E")
    ax.set_title("heading (0°=RIGHT, 90°=FORWARD)", fontsize=8)

    # 4. wall distance over time
    ax = axes[1, 0]
    from .tracking.calibration import calibration_from_session
    meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
    cal = calibration_from_session(meta)
    valid = tr["found"]
    dist = [cal.wall_distance_mm(a, b) if f else np.nan
            for a, b, f in zip(tr["x_mm"], tr["y_mm"], valid)]
    ax.plot(tr["t"], dist, lw=0.5, color="#188487")
    ax.axhline(10.0, color="k", ls=":", lw=1)
    ax.set_xlabel("t (s)"); ax.set_ylabel("wall distance (mm)")
    ax.set_title("wall distance (band 10 mm dashed)")

    # 5. state occupancy
    ax = axes[1, 1]
    occ = an["state_occupancy_L_A"]
    states = list(occ)
    ax.bar(states, [occ[s] for s in states], color="#467821", alpha=0.85)
    ax.axhline(0.05, color="r", ls=":", lw=1)
    ax.set_ylabel("window occupancy"); ax.set_title("action-state occupancy")
    ax.tick_params(axis="x", rotation=45)

    # 6. stimulus response (if any)
    ax = axes[1, 2]
    if an.get("stimulus"):
        ev = an["stimulus"]["events"]
        xs = range(1, len(ev) + 1)
        ax.bar(xs, [e.get("t_off", 0) - e["t_on"] for e in ev],
               bottom=[e["t_on"] for e in ev], color="#CF4457", alpha=0.6)
        ax.set_xlabel("event #"); ax.set_ylabel("t (s)")
        ax.set_title(f"stimulus events (p={an['stimulus']['permutation_p']:.3f})")
    else:
        ax.text(0.5, 0.5, "no stimulus (baseline)", ha="center", va="center")
        ax.set_axis_off()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def plot_ladder(ladder: dict, out_png: Path) -> Path:
    """Occupancy + LOSO per ladder level."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    levels = [e["level"] for e in ladder["evidence"]]
    loso = [e["G4_loso"].get("accuracy") or 0 for e in ladder["evidence"]]
    n_states = [e["n_states"] for e in ladder["evidence"]]
    sel = ladder["selected_level"]
    colors = ["#467821" if e["pass"] else "#CF4457"
              for e in ladder["evidence"]]
    axes[0].bar(levels, n_states, color=colors, alpha=0.85)
    axes[0].set_ylabel("states in vocabulary")
    axes[0].set_title(f"reduction ladder (selected: {sel})")
    axes[1].bar(levels, loso, color=colors, alpha=0.85)
    axes[1].axhline(0.80, color="k", ls=":", lw=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("LOSO accuracy")
    axes[1].set_title("decoder reliability (dashed = 0.80 gate)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png
