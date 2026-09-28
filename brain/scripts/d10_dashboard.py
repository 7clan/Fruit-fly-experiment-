"""D10 basic dashboard: WATCH the digital fly control the artificial
environment.

Two modes:
  --replay <episode_dir>   render a recorded episode (chunks.jsonl) to an
                           MP4 (arena LEFT, neural activity RIGHT, status
                           BOTTOM). No brain needed - pure replay.
  --live                   run one intact closed-loop episode in-process
                           with live figure updates (for the user's laptop;
                           needs the canonical brain + ~3 GB RAM).

This is the BASIC dashboard (D10) - the full Windows dashboard is a later
step (D14/D15) per docs/WINDOWS_RUNTIME_SPEC.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brainlib as bl  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation as _anim
from matplotlib.patches import Circle, FancyArrow
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ACT_COLORS = {"FORWARD": "#2a9d8f", "TURN_LEFT": "#e9c46a",
              "TURN_RIGHT": "#f4a261", "BACKWARD": "#e76f51",
              "STOP": "#6c757d", "JUMP": "#9b5de5"}


def load_episode(episode_dir):
    ep = Path(episode_dir)
    rows = [json.loads(l) for l in (ep / "chunks.jsonl").read_text()
            .splitlines() if l.strip()]
    summary = json.loads((ep / "summary.json").read_text())
    return rows, summary


def render_episode(episode_dir, out_mp4=None, fps=5, max_frames=None):
    rows, summary = load_episode(episode_dir)
    if max_frames:
        rows = rows[:max_frames]
    n = len(rows)
    ep_name = f"{summary['condition']} / {summary['scenario']} / rep {summary['rep']}"

    fig = plt.figure(figsize=(13.5, 7.6), dpi=100,
                     constrained_layout=False)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.16],
                          width_ratios=[1.0, 1.25],
                          left=0.05, right=0.985, top=0.90, bottom=0.075,
                          hspace=0.42, wspace=0.16)
    ax_arena = fig.add_subplot(gs[:, 0])
    ax_sens = fig.add_subplot(gs[0, 1])
    ax_dn = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[2, :])

    # ---- static arena decor
    ax_arena.set_xlim(-1.15, 1.15)
    ax_arena.set_ylim(-1.15, 1.15)
    ax_arena.set_aspect("equal")
    ax_arena.set_facecolor("#14161c")
    for s in ("top", "right", "left", "bottom"):
        ax_arena.spines[s].set_visible(False)
    ax_arena.set_xticks([])
    ax_arena.set_yticks([])
    wall = plt.Rectangle((-1, -1), 2, 2, fill=False, ec="#4a4f5a", lw=2)
    ax_arena.add_patch(wall)

    # precompute full-episode series
    ts = [r["t"] for r in rows]
    trail_x = [r["ax"] for r in rows]
    trail_y = [r["ay"] for r in rows]
    rates_L = [r["rates"].get("target_left", 0) for r in rows]
    rates_R = [r["rates"].get("target_right", 0) for r in rows]
    rates_loom = [r["rates"].get("looming", 0) for r in rows]
    dn_pops = ["P9_left", "P9_right", "GF_left", "GF_right",
               "MDN_bilateral", "BPN_bilateral"]
    dn_series = {p: [r["pop_counts"].get(p, 0) for r in rows]
                 for p in dn_pops}
    actions = [r["action"] for r in rows]
    lat = [r["wall_ms"] for r in rows]

    target_xy = None
    if summary["scenario"].startswith("target_"):
        # reconstruct target position from first row geometry
        first = rows[0]
        b = math.radians(first["bearing_deg"] or 0.0)
        hd = math.radians(first["heading_deg"])
        d = first["dist"] if first["dist"] else 0.8
        ang = hd + b
        target_xy = (first["ax"] + d * math.cos(ang),
                     first["ay"] + d * math.sin(ang))

    ax_sens.set_title("sensory encoder  (Poisson drive per channel)",
                      fontsize=10, loc="left", color="#eeeeee")
    ax_sens.set_ylim(-8, 175)
    ax_sens.set_xlim(0, max(ts[-1], 0.1))
    ax_sens.set_ylabel("Hz", fontsize=8)
    ax_sens.tick_params(labelsize=7, colors="#cccccc")
    ax_sens.set_facecolor("#1b1e26")
    for s in ax_sens.spines.values():
        s.set_color("#3a3f4a")

    ax_dn.set_title("descending-neuron activity -> decoded action",
                    fontsize=10, loc="left", color="#eeeeee")
    ax_dn.set_xlim(0, max(ts[-1], 0.1))
    ax_dn.set_facecolor("#1b1e26")
    for s in ax_dn.spines.values():
        s.set_color("#3a3f4a")
    ax_dn.set_ylabel("spikes / 50 ms chunk", fontsize=8)
    ax_dn.tick_params(labelsize=7, colors="#cccccc")
    dn_lines = {}
    dn_colors = dict(P9_left="#ffd166", P9_right="#ef8354", GF_left="#c77dff",
                     GF_right="#e0aaff", MDN_bilateral="#ef476f",
                     BPN_bilateral="#06d6a0")
    offs = dict(P9_left=5.2, P9_right=4.2, GF_left=3.2, GF_right=2.2,
                MDN_bilateral=1.2, BPN_bilateral=0.2)
    for p in dn_pops:
        (ln,) = ax_dn.plot([], [], lw=1.6, label=p, color=dn_colors[p])
        dn_lines[p] = ln
    ax_dn.set_yticks([offs[p] + 0.4 for p in dn_pops])
    ax_dn.set_yticklabels(dn_pops, fontsize=6.5, color="#cccccc")

    ax_bar.axis("off")

    fig.patch.set_facecolor("#0d0f14")
    fig.suptitle(f"D10 closed loop - digital Drosophila -  {ep_name}",
                 color="white", fontsize=12, x=0.05, ha="left")

    # static full-history sensory lines (drawn once)
    ax_sens.plot(ts, rates_L, c="#ffd166", lw=1.6, label="target_left (LC9-L)")
    ax_sens.plot(ts, rates_R, c="#ef8354", lw=1.6, label="target_right (LC9-R)")
    ax_sens.plot(ts, rates_loom, c="#ef233c", lw=1.6, label="looming (LPLC2/LC4)")
    ax_sens.legend(fontsize=6, ncol=3, loc="upper right", framealpha=0.2,
                   labelcolor="#cccccc")
    sens_marker, = ax_sens.plot([], [], "o", ms=4, c="#ffd166")
    sens_marker2, = ax_sens.plot([], [], "o", ms=4, c="#ef8354")
    sens_marker3, = ax_sens.plot([], [], "o", ms=4, c="#ef233c")

    frames = []
    out_mp4 = out_mp4 or (Path(episode_dir) / "dashboard.mp4")
    writer = _anim.FFMpegWriter(fps=fps, bitrate=2400,
                                 metadata=dict(artist="fly-agent D10"))
    with writer.saving(fig, str(out_mp4), dpi=100):
        for k in range(n):
            _render_frame(fig, ax_arena, ax_sens, ax_dn, ax_bar, rows, k,
                          target_xy, ts, trail_x, trail_y, rates_L, rates_R,
                          rates_loom, dn_series, dn_lines, dn_pops, offs,
                          dn_colors, actions, lat, summary, n, sens_marker,
                          sens_marker2, sens_marker3)
            writer.grab_frame()
    plt.close(fig)
    print(f"[dashboard] {n} frames -> {out_mp4}")
    return out_mp4


def _render_frame(fig, ax_arena, ax_sens, ax_dn, ax_bar, rows, k, target_xy,
                  ts, trail_x, trail_y, rates_L, rates_R, rates_loom,
                  dn_series, dn_lines, dn_pops, offs, dn_colors, actions,
                  lat, summary, n, sens_marker, sens_marker2, sens_marker3):
        for coll in list(ax_arena.collections + ax_arena.lines):
            if getattr(coll, "_dyn", False):
                coll.remove()
        for patch in list(ax_arena.patches):
            if getattr(patch, "_dyn", False):
                patch.remove()
        for patch in list(ax_dn.patches):
            if getattr(patch, "_dyn", False):
                patch.remove()
        # trail
        ln, = ax_arena.plot(trail_x[:k + 1], trail_y[:k + 1], "-", c="#8ecae6",
                            lw=1.4, alpha=0.8)
        ln._dyn = True
        # agent arrow
        hd = math.radians(rows[k]["heading_deg"])
        arr = FancyArrow(rows[k]["ax"], rows[k]["ay"],
                         0.10 * math.cos(hd), 0.10 * math.sin(hd),
                         width=0.035, head_width=0.09, head_length=0.07,
                         fc="#e0e1dd", ec="none")
        arr._dyn = True
        ax_arena.add_patch(arr)
        if target_xy is not None and rows[k]["target_present"]:
            c = Circle(target_xy, 0.08, fc="#ffd60a", ec="none", alpha=0.95)
            c._dyn = True
            ax_arena.add_patch(c)
        if rows[k]["loom_deg"] > 0:
            c = Circle((0, 1.0), 0.16 + rows[k]["loom_deg"] / 70 * 0.55,
                       fill=False, ec="#ef233c", lw=2.5, alpha=0.9)
            c._dyn = True
            ax_arena.add_patch(c)
            t = ax_arena.text(0, 0.55, "LOOMING", color="#ef233c",
                              ha="center", fontsize=9, fontweight="bold")
            t._dyn = True

        # sensory panel (static full-history lines are pre-drawn; only the
        # leading marker moves)
        sens_marker.set_data([ts[k]], [rates_L[k]])
        sens_marker2.set_data([ts[k]], [rates_R[k]])
        sens_marker3.set_data([ts[k]], [rates_loom[k]])

        # DN panel
        for p in dn_pops:
            dn_lines[p].set_data(ts[:k + 1],
                                 [v + offs[p] for v in dn_series[p][:k + 1]])

        # action strip inside DN panel
        for j in range(min(k + 1, n)):
            span = ax_dn.axvspan(ts[j], ts[j] + 0.05,
                                 color=ACT_COLORS.get(actions[j], "#555"),
                                 alpha=0.28, lw=0)
            span._dyn = True

        # bottom bar
        ax_bar.clear()
        ax_bar.axis("off")
        lat_mean = float(np.mean(lat[:k + 1]))
        bio_wall = ((rows[k]["t"] or 0.05)
                    / max(sum(lat[:k + 1]) / 1000.0, 1e-9))
        txt = (f"target: {summary['scenario']:<13}  "
               f"action: {actions[k]:<10}  "
               f"chunk latency: {lat[k]:.0f} ms (mean {lat_mean:.0f})  "
               f"bio/wall: {bio_wall:.3f}  "
               f"reward: [placeholder - learning disabled in D10]  "
               f"conflicts: {sum(len(r['conflicts']) for r in rows[:k+1])}")
        ax_bar.text(0.0, 0.55, txt, fontsize=9.2, color="#dddddd",
                    family="DejaVu Sans", va="center")
        ax_bar.add_patch(plt.Rectangle((0, 0), 1.0, 0.14,
                                       color=ACT_COLORS.get(actions[k],
                                                            "#555")))
        fig.canvas.draw()


def condition_summary_figure(results_dir, out_png):
    """Comparison figure across conditions (success rates + key metrics)."""
    import glob
    import matplotlib
    summaries = []
    for p in glob.glob(str(Path(results_dir) / "episodes" / "*"
                           / "summary.json")):
        summaries.append(json.loads(open(p).read()))
    conds = ["scripted", "intact", "shuffled_sensory", "shuffled_motor",
             "random"]
    scen = ["target_left", "target_right", "target_center", "no_target",
            "looming"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                             constrained_layout=True)
    # success rate
    ax = axes[0]
    w = 0.16
    xs = np.arange(len(scen))
    for ci, c in enumerate(conds):
        vals = []
        for s in scen:
            rows = [x for x in summaries
                    if x["condition"] == c and x["scenario"] == s]
            vals.append(100 * sum(r["success"] for r in rows) / len(rows)
                        if rows else 0)
        ax.bar(xs + (ci - 2) * w, vals, w, label=c)
    ax.set_xticks(xs)
    ax.set_xticklabels(scen, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("target-reach success %")
    ax.set_title("A. success rate (target scenarios)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    # escape rate (looming)
    ax = axes[1]
    vals, lats = [], []
    for c in conds:
        rows = [x for x in summaries
                if x["condition"] == c and x["scenario"] == "looming"]
        vals.append(100 * sum(r["escape_emitted"] for r in rows) / len(rows)
                    if rows else 0)
        ll = [r["escape_latency_ms"] for r in rows
              if r["escape_latency_ms"] is not None]
        lats.append(float(np.mean(ll)) if ll else 0)
    bars = ax.bar(conds, vals, color="#9b5de5", alpha=0.8)
    for b, l in zip(bars, lats):
        if l:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                    f"{l:.0f} ms", ha="center", fontsize=8)
    ax.set_ylabel("escape (JUMP) rate %")
    ax.set_title("B. looming escape")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    # stop frequency
    ax = axes[2]
    for c in conds:
        rows = [x for x in summaries if x["condition"] == c]
        if rows:
            ax.scatter([c] * len(rows),
                       [r["stop_frequency"] for r in rows], s=18,
                       alpha=0.7)
            ax.hlines(float(np.mean([r["stop_frequency"] for r in rows])),
                      -0.3 + list(conds).index(c),
                      0.3 + list(conds).index(c), lw=2)
    ax.set_ylabel("STOP frequency (per chunk)")
    ax.set_title("C. STOP frequency (all scenarios)")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("D10 control conditions - does behavior depend on the "
                 "intact biological pathway?", fontsize=11)
    fig.savefig(out_png, dpi=130)
    print(f"[dashboard] summary -> {out_png}")
    return out_png


def live_episode(condition="intact", scenario="target_left", seed=None,
                 chunk_ms=50.0):
    """Live dashboard: run one closed-loop episode with live rendering
    (user's laptop; needs the canonical brain)."""
    matplotlib.use("TkAgg")            # interactive backend
    import matplotlib.pyplot as plt
    plt.ion()
    import d10_closed_loop as d10
    import d7_runtime as rt
    import d8_encoder as d8
    import d9_decoder as d9

    if seed is None:
        seed = 20260928
    sp = d10.SCENARIO_PARAMS[scenario]
    channel_ids = {ch: d["ids"] for ch, d in
                   json.loads((d10.DATA / "sensory_interface.json")
                              .read_text())["channels"].items()}
    readout_pops = d8.load_readout_pops()
    brain = rt.DualModeBrain("interactive", seed=seed,
                             channel_ids=channel_ids,
                             readout_pops=readout_pops)
    encoder = d8.VisualSensoryEncoder()
    decoder = d9.MotorDecoder(brain.pop_sizes)
    arena = d8.Arena2D(target_bearing_deg=sp.get("target_bearing_deg", 0.0),
                       target_dist=sp.get("target_dist", 0.8),
                       target_present=sp["target_present"],
                       loom=sp.get("loom", False), seed=seed)
    brain.reset_episode(seed)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 5.5))
    axa.set_xlim(-1.15, 1.15); axa.set_ylim(-1.15, 1.15)
    axa.set_aspect("equal")
    trail_x, trail_y = [], []
    n_chunks = int(sp["cap_s"] * 1000 / chunk_ms)
    for k in range(n_chunks):
        vis = arena.visual_state()
        rates = encoder.rates(vis)
        r = brain.step(rates, chunk_ms=chunk_ms)
        dec = decoder.decode(r.get("pop_counts", {}), chunk_ms)
        arena.step(dec["action"], chunk_ms / 1000.0)
        trail_x.append(arena.x); trail_y.append(arena.y)
        axa.clear()
        axa.plot(trail_x, trail_y, "-o", ms=2, c="#8ecae6")
        axa.plot(arena.x, arena.y, "w>", ms=8)
        if arena.target_present:
            axa.plot([arena.tx], [arena.ty], "*", ms=16, c="#ffd60a")
        axa.set_title(f"{dec['action']}  P9diff={dec['p9_diff_hz']:.0f}Hz "
                      f"GF={dec['gf_hz']:.0f}Hz")
        axb.clear()
        pc = r.get("pop_counts", {})
        pops = ["P9_left", "P9_right", "GF_left", "GF_right", "MDN_bilateral"]
        axb.bar(pops, [pc.get(p, 0) for p in pops], color="#06d6a0")
        axb.set_ylabel("spikes in chunk")
        plt.pause(0.01)
        if arena.reached_target():
            print("TARGET REACHED at t =", arena.t)
            break
    plt.ioff()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="D10 dashboard")
    ap.add_argument("cmd", choices=["replay", "summary", "live"])
    ap.add_argument("--episode-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--results-dir", default=str(
        bl.ROOT / "results" / "d10_closed_loop"))
    ap.add_argument("--condition", default="intact")
    ap.add_argument("--scenario", default="target_left")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "replay":
        render_episode(args.episode_dir, args.out or None, fps=args.fps,
                       max_frames=args.max_frames or None)
    elif args.cmd == "summary":
        condition_summary_figure(args.results_dir, args.out or str(
            Path(args.results_dir) / "condition_summary.png"))
    elif args.cmd == "live":
        live_episode(condition=args.condition, scenario=args.scenario,
                     seed=args.seed or None)


if __name__ == "__main__":
    main()
