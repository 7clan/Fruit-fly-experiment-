"""Learning-curve plots (PNG, headless - matplotlib Agg backend).

Panels:
  1. trial number vs rolling success rate (the learning curve)
  2. trial number vs time-to-target (successes only)
  3. trial number vs path efficiency (successes only)
  4. action changes per trial
All in the user's language (English) and saved to file - never shown live.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COLORS = {
    "random": "#9ca3af",
    "fixed": "#059669",
    "fly_random": "#f59e0b",
    "fly_cue": "#2563eb",
    "fly_cue_shuffled": "#db2777",
    "fly": "#2563eb",
    "fly_no_cue": "#a16207",
    "keyboard": "#7c3aed",
}


def _color(cond: str) -> str:
    return COLORS.get(cond, "#64748b")


def _rolling(x: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        out[i] = x[lo:i + 1].mean()
    return out


def comparison_figure(summaries: list[dict], out_png, rolling_window: int = 10):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()
    handles = []

    for s in summaries:
        cond = s["condition"]
        c = _color(cond)
        trials = np.arange(1, len(s["per_trial"]) + 1)
        succ = np.array([1.0 if r["outcome"] == "SUCCESS" else 0.0
                         for r in s["per_trial"]])
        ttt = [r["time_to_target_s"] for r in s["per_trial"]
               if r["time_to_target_s"] is not None]
        eff = [r["path_efficiency"] for r in s["per_trial"]
               if r["time_to_target_s"] is not None]
        changes = [r["n_action_changes"] for r in s["per_trial"]]

        ax1.plot(trials, _rolling(succ, rolling_window), color=c, lw=2)
        ax1.scatter(trials, succ, color=c, s=8, alpha=0.35)
        if ttt:
            ax2.scatter(np.arange(1, len(ttt) + 1), ttt, color=c, s=16, alpha=0.7)
            ax2.axhline(np.median(ttt), color=c, lw=1.2, ls="--", alpha=0.8)
        if eff:
            ax3.scatter(np.arange(1, len(eff) + 1), eff, color=c, s=16, alpha=0.7)
            ax3.axhline(np.median(eff), color=c, lw=1.2, ls="--", alpha=0.8)
        ax4.scatter(trials, changes, color=c, s=14, alpha=0.6)
        ax4.axhline(np.mean(changes), color=c, lw=1.2, ls="--", alpha=0.8)
        handles.append(Line2D([0], [0], color=c, lw=2.5, label=cond))

    ax1.set_xlabel("Trial number")
    ax1.set_ylabel(f"Rolling success rate (window {rolling_window})")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.25)
    ax2.set_xlabel("Trial number (successes only)")
    ax2.set_ylabel("Time to target [s]")
    ax2.grid(alpha=0.25)
    ax3.set_xlabel("Trial number (successes only)")
    ax3.set_ylabel("Path efficiency (1 = straight line)")
    ax3.set_ylim(0, 1.05)
    ax3.grid(alpha=0.25)
    ax4.set_xlabel("Trial number")
    ax4.set_ylabel("Action changes per trial")
    ax4.grid(alpha=0.25)

    fig.legend(handles=handles, loc="outside right upper", title="Condition")
    fig.suptitle("Phase 1 - control-group comparison (2D target-reaching task)",
                 fontsize=13)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def per_run_curves(run_dir) -> Path:
    """Regenerate curves for a single run from its summary.json."""
    import json
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    out = run_dir / "learning_curves.png"
    comparison_figure([summary], out)
    return out
