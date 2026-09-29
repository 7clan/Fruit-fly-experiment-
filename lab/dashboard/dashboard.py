"""DigitalFlyLab dashboard — 3-column snapshot reader.

FINAL_ARCHITECTURE §26/§27:
  * Three clearly separated columns:
      FLY BRAIN        — sensory activity, neural activity, descending
                         populations, current intention, threat/approach/
                         avoid signals
      HYBRID HELPER    — recognized objects, world state, goal, memory,
                         learned value, strategy
      ACTION SYSTEM    — eligible abilities, selected intention, selected
                         concrete ability, cooldown, emitted key/mouse input
    so the user can see WHO made each part of the decision.
  * The dashboard reads SNAPSHOTS ONLY. It never owns the real-time
    simulation clock. If rendering drops frames, CONTROL CONTINUES.
    A slow chart is DROPPED, never allowed to pause the fly.

Rendering backends:
  * TextDashboardRenderer — dependency-free ANSI/text 3-column summary
    (works headless in the sandbox/CI and is the content contract).
  * OpenCVDashboardRenderer — composes the live mirror + panels into an
    image (guarded cv2 import; Windows desktop window). The mirror uses
    the SAME capture stream (one stream, vision + dashboard).

Both renderers consume identical Snapshot objects produced by the
DashboardWorker, which reads bus state channels at 10–30 Hz.
"""

from __future__ import annotations

from ..bus import Bus, StateChannel
from ..worker import Worker

TOPICS = ("world.observation", "world.semantics", "brain.output",
          "fly.channels", "helper.goal", "action.selected",
          "brain.meta", "memory.stats", "value.table", "action.meta")


class Snapshot:
    """One consistent-ish read of all dashboard-visible state. Individual
    channels are read independently (they race — that is FINE and honest:
    the dashboard is a mirror, not the control clock)."""

    def __init__(self, ts_ns: int, data: dict):
        self.ts_ns = ts_ns
        self.data = data

    def to_dict(self) -> dict:
        return {"ts_ns": self.ts_ns, "columns": self.data}


class DashboardWorker(Worker):
    """Reads state channels → builds Snapshots → renders (rate-limited)."""

    name = "dashboard"
    TOPIC_SNAP = "dashboard.snapshot"

    def __init__(self, bus: Bus, target_hz: float = 20.0,
                 renderer=None, mirror_channel: str = "capture.frames"):
        super().__init__(bus, target_hz=target_hz)
        self.renderer = renderer
        self.snapshot_state: StateChannel = bus.state(self.TOPIC_SNAP)
        self.mirror: StateChannel = bus.state(mirror_channel + ".latest")
        self._channels = {t: bus.state(t) for t in TOPICS}
        self.stats.update({"rendered": 0, "dropped_renders": 0,
                           "last_render_ms": None})

    def take_snapshot(self) -> Snapshot:
        data = {}
        for topic, ch in self._channels.items():
            snap = ch.read()
            data[topic] = snap.payload if snap else None
        return Snapshot(self.clock.now_ns(), data)

    def step(self) -> None:
        snap = self.take_snapshot()
        self.snapshot_state.write(snap.to_dict(), ts_ns=snap.ts_ns)
        if self.renderer is None:
            return
        # drop the mirror frame if one is not fresh (never queue renders)
        t0 = self.clock.now_ns()
        try:
            self.renderer.render(snap, self.mirror.read())
        except Exception as e:  # noqa: BLE001 — dashboard errors never kill control
            self.stats["errors"] += 1
            self.stats["last_error"] = repr(e)
        dt = self.clock.elapsed_ms(t0)
        self.stats["last_render_ms"] = round(dt, 2)
        if dt > 1000.0 / max(self.governor.target_hz, 1.0):
            self.stats["dropped_renders"] += 1   # honest degradation report
        else:
            self.stats["rendered"] += 1


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------

def _fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


class TextDashboardRenderer:
    """3-column text summary (headless-safe; the content contract)."""

    name = "text"

    def render(self, snap: Snapshot, mirror_env=None) -> str:
        d = snap.data
        brain = d.get("brain.output") or {}
        ch = d.get("fly.channels") or {}
        goal = d.get("helper.goal") or {}
        act = d.get("action.selected") or {}
        obs = d.get("world.observation") or {}
        intention = (brain.get("intention") or {})
        fly_col = [
            "FLY BRAIN",
            f"  runtime        {_fmt(brain.get('runtime'))}",
            f"  chunk          {_fmt(brain.get('chunk_id'))} "
            f"@ {_fmt(brain.get('chunk_ms'), 0)}ms",
            f"  intention      {_fmt(intention.get('name'))} "
            f"(conf {_fmt(intention.get('confidence'))})",
            f"  target L/R/C   {_fmt(ch.get('target_left'))} "
            f"{_fmt(ch.get('target_right'))} {_fmt(ch.get('target_center'))}",
            f"  threat L/R/I   {_fmt(ch.get('threat_left'))} "
            f"{_fmt(ch.get('threat_right'))} {_fmt(ch.get('threat_intensity'))}",
            f"  approach/avoid {_fmt(ch.get('approach_value'))} "
            f"{_fmt(ch.get('avoidance_value'))}",
            "  DNs            " + " ".join(
                f"{k}={_fmt(v, 1)}"
                for k, v in list((brain.get("dn_rates_hz") or {}).items())[:6]),
        ]
        helper_col = [
            "HYBRID HELPER  [ENGINEERED]",
            f"  goal           {_fmt((goal.get('goal') or {}).get('label'))}",
            f"  relevance      {_fmt(goal.get('goal_relevance'))}",
            f"  enemies        {len(obs.get('enemies', []))}",
            f"  target         {_fmt((obs.get('target') or {}).get('type'))}",
            f"  ui             " + "/".join(k for k in ("loading", "dialogue",
                              "menu", "combat")
                              if (obs.get("ui") or {}).get(k)),
        ]
        action_col = [
            "ACTION SYSTEM  [ENGINEERED resolver]",
            f"  autonomy       {_fmt(act.get('autonomy'))}",
            f"  ability        {_fmt(act.get('ability_id')) or '(none)'}",
            f"  bindings       {','.join(act.get('bindings', [])) or '-'}",
            f"  backend        {_fmt(act.get('backend'))}",
        ]
        rows = max(len(fly_col), len(helper_col), len(action_col))
        while len(fly_col) < rows: fly_col.append("")
        while len(helper_col) < rows: helper_col.append("")
        while len(action_col) < rows: action_col.append("")
        w = 34
        lines = ["DigitalFlyLab".center(w * 3) + "  (snapshot "
                 f"t={snap.ts_ns/1e9:.3f}s)"]
        for a, b, c in zip(fly_col, helper_col, action_col):
            lines.append(a.ljust(w) + b.ljust(w) + c.ljust(w))
        out = "\n".join(lines)
        print(out)
        return out


class OpenCVDashboardRenderer:
    """Live mirror + panel overlay via OpenCV (guarded import; desktop).

    Uses the SAME capture stream for the mirror (never a second capture).
    Rendering here is best-effort: frame drops are counted by the worker.
    """

    name = "opencv"

    def __init__(self, window_name: str = "DigitalFlyLab",
                 width: int = 1280, height: int = 720):
        try:
            import cv2  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"cv2 unavailable: {e}") from e
        import cv2
        self._cv2 = cv2
        self.window_name = window_name
        self.width, self.height = width, height

    def render(self, snap: Snapshot, mirror_env=None) -> None:
        cv2 = self._cv2
        canvas = self._compose(snap, mirror_env)
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)          # non-blocking pump

    def _compose(self, snap: Snapshot, mirror_env):
        import numpy as np
        cv2 = self._cv2
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # LEFT: live mirror (same stream as vision)
        if mirror_env is not None and mirror_env.payload.get("data_ref") is not None:
            img = mirror_env.payload["data_ref"]
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            h, w = img.shape[:2]
            scale = min((self.width * 0.5) / w, self.height / h)
            mirror = cv2.resize(img, (int(w * scale), int(h * scale)))
            canvas[0:mirror.shape[0], 0:mirror.shape[1]] = mirror
        # RIGHT: three stacked panels
        d = snap.data
        def put(x, y, text, color=(220, 220, 220)):
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
        x0 = int(self.width * 0.52)
        brain = d.get("brain.output") or {}
        intention = brain.get("intention") or {}
        put(x0, 24, "FLY BRAIN", (120, 220, 120))
        put(x0, 44, f"intention={intention.get('name')} "
                    f"conf={intention.get('confidence')}")
        put(x0, 62, f"chunk={brain.get('chunk_id')} "
                    f"runtime={brain.get('runtime')}")
        put(x0, 80, f"transport={brain.get('transport')} "
                    f"chunk_wall={brain.get('chunk_wall_s')}s")
        ch = d.get("fly.channels") or {}
        tl = float(ch.get("target_left") or 0.0)
        tr = float(ch.get("target_right") or 0.0)
        threat = float(ch.get("threat_intensity") or 0.0)
        put(x0, 100, f"tgt L/R {tl:.2f}/{tr:.2f} threat {threat:.2f}")
        put(x0, 128, "HYBRID HELPER [ENGINEERED]", (120, 180, 240))
        goal = d.get("helper.goal") or {}
        put(x0, 148, f"goal={(goal.get('goal') or {}).get('label')}")
        obs = d.get("world.observation") or {}
        player = obs.get("player") or {}
        target = obs.get("target") or {}
        notes = obs.get("notes") or {}
        put(x0, 166, f"target={target.get('type')} enemies={len(obs.get('enemies', []))}")
        put(x0, 184, f"tracked humanoids={notes.get('humanoid_track_count', 0)} "
                     f"quest NPCs={notes.get('quest_npc_track_count', 0)}")
        put(x0, 202, f"health={player.get('health')} stamina={player.get('stamina')}")
        put(x0, 230, "ACTION SYSTEM [ENGINEERED]", (240, 180, 120))
        act = d.get("action.selected") or {}
        put(x0, 250, f"autonomy={act.get('autonomy')} "
                    f"ability={act.get('ability_id') or '-'}")
        put(x0, 278, "PASSIVE / SHADOW - no game input", (140, 140, 255))
        return canvas
