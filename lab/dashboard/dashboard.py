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

from collections import deque
import time

from ..bus import Bus, StateChannel
from ..worker import Worker

TOPICS = ("world.observation", "world.semantics", "brain.output",
          "fly.channels", "helper.goal", "action.selected",
          "brain.meta", "memory.stats", "value.table", "action.meta",
          "quest.state", "coach.plan", "action.command")


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
        brain_meta = d.get("brain.meta") or {}
        action_meta = d.get("action.meta") or {}
        quest_state = d.get("quest.state") or {}
        coach = d.get("coach.plan") or {}
        command = d.get("action.command") or {}
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
            f"  quest phase    {_fmt(quest_state.get('phase'))}",
            f"  AI -> fly      {_fmt(coach.get('skill'))} "
            f"(conf {_fmt(coach.get('confidence'))})",
            f"  AI source      {_fmt(coach.get('provider'))}",
            f"  AI target      {_fmt(coach.get('target'))}",
            f"  AI why         {_fmt(coach.get('explanation'))}",
            f"  executor cmd   {_fmt(command.get('name'))}",
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
    """Live scientific view of the fly's actual readouts.

    This renderer intentionally distinguishes:
      * real canonical whole-brain chunk activity
      * verified sensory rates delivered to the canonical model
      * descending-neuron population readouts used by the decoder
      * engineered perception/helper/action state

    The node links below visualize the DECODER READOUT FLOW, not anatomical
    connectome edges. The full 138k-neuron connectome is not redrawn every
    frame because doing so would be both unreadable and expensive on the
    target laptop.
    """

    name = "opencv"

    def __init__(self, window_name: str = "DigitalFlyLab",
                 width: int = 1440, height: int = 810,
                 evidence_dir=None, control_handler=None,
                 lightweight: bool = False,
                 preview_interval_s: float = 3.0):
        try:
            import cv2  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"cv2 unavailable: {e}") from e
        import cv2
        self._cv2 = cv2
        self.window_name = window_name
        self.width, self.height = width, height
        self._history = deque(maxlen=90)
        self._last_chunk_id = None
        self._evidence_dir = evidence_dir
        self._control_handler = control_handler
        self.lightweight = bool(lightweight)
        self.preview_interval_s = max(0.25, float(preview_interval_s))
        self._preview_cache = None
        self._preview_updated_at = 0.0
        self._mouse_ready = False
        self._button_rects = {}
        if self._evidence_dir is not None:
            from pathlib import Path
            self._evidence_dir = Path(self._evidence_dir)
            self._evidence_dir.mkdir(parents=True, exist_ok=True)

    def render(self, snap: Snapshot, mirror_env=None) -> None:
        cv2 = self._cv2
        if not self._mouse_ready:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self._on_mouse)
            self._mouse_ready = True
        canvas = self._compose(snap, mirror_env)
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    @staticmethod
    def _inside(rect, x, y) -> bool:
        if rect is None:
            return False
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        if event != self._cv2.EVENT_LBUTTONUP or self._control_handler is None:
            return
        try:
            for command, rect in list(self._button_rects.items()):
                if self._inside(rect, x, y):
                    self._control_handler(command)
                    break
        except Exception:
            # UI controls must never crash the live loop.
            pass

    def _button(self, canvas, rect, label, fill, text_color=(245, 245, 245)):
        cv2 = self._cv2
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (210, 210, 210), 1)
        scale = 0.42
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        max_w = max(12, x2 - x1 - 10)
        if tw > max_w:
            scale = max(0.26, scale * max_w / max(tw, 1))
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        tx = x1 + max(4, (x2 - x1 - tw) // 2)
        ty = y1 + max(th + 4, (y2 - y1 + th) // 2)
        self._put(canvas, tx, ty, label, text_color, scale=scale, thickness=1)

    @staticmethod
    def _safe_float(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _put(self, canvas, x, y, text, color=(220, 220, 220),
             scale=0.46, thickness=1):
        self._cv2.putText(
            canvas, str(text), (int(x), int(y)),
            self._cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
            self._cv2.LINE_AA)

    def _bar(self, canvas, x, y, w, label, value, vmax,
             color=(120, 220, 120)):
        cv2 = self._cv2
        value = max(0.0, self._safe_float(value))
        vmax = max(1e-9, float(vmax))
        frac = min(1.0, value / vmax)
        self._put(canvas, x, y - 4, f"{label}  {value:.1f}", (210, 210, 210),
                  scale=0.38)
        cv2.rectangle(canvas, (x, y), (x + w, y + 10), (70, 70, 70), 1)
        if frac > 0:
            cv2.rectangle(
                canvas, (x + 1, y + 1),
                (x + max(1, int((w - 2) * frac)), y + 9),
                color, -1)

    def _activity_node(self, canvas, x, y, label, rate, max_rate=30.0,
                       color=(120, 220, 120)):
        cv2 = self._cv2
        rate = max(0.0, self._safe_float(rate))
        frac = min(1.0, rate / max(1e-9, max_rate))
        radius = 12 + int(8 * frac)
        base = tuple(int(55 + frac * max(0, c - 55)) for c in color)
        cv2.circle(canvas, (x, y), radius, base, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, (180, 180, 180), 1, cv2.LINE_AA)
        self._put(canvas, x + 24, y + 4, f"{label} {rate:.1f} Hz",
                  (215, 215, 215), scale=0.36)

    def _save_evidence_frame(self, mirror_env, chunk_id, obs=None):
        """Save one raw game frame per completed brain chunk.

        This is deliberately low-rate (the canonical brain is slow), so it
        provides visual evidence for replay analysis without turning the run
        into a video recorder or adding meaningful capture load.
        """
        if self._evidence_dir is None or mirror_env is None:
            return
        try:
            img = mirror_env.payload.get("data_ref")
            if img is None:
                return
            if img.ndim == 3 and img.shape[2] == 4:
                img = self._cv2.cvtColor(img, self._cv2.COLOR_BGRA2BGR)
            path = self._evidence_dir / f"brain_chunk_{int(chunk_id):06d}.jpg"
            self._cv2.imwrite(
                str(path), img,
                [int(self._cv2.IMWRITE_JPEG_QUALITY), 82])

            # Also persist a review copy with the exact perception tracks
            # visible. Raw evidence remains untouched above.
            annotated = img.copy()
            notes = (obs or {}).get("notes") or {}
            for tr in notes.get("entity_tracks") or []:
                box = tr.get("bbox") or []
                if len(box) != 4:
                    continue
                h, w = annotated.shape[:2]
                x1, y1, x2, y2 = box
                p1 = (int(x1 * w), int(y1 * h))
                p2 = (int(x2 * w), int(y2 * h))
                kind = str(tr.get("kind", "unknown"))
                if kind == "quest_npc":
                    color = (0, 220, 255)
                elif kind == "hostile_candidate":
                    color = (80, 80, 255)
                else:
                    color = (255, 180, 70)
                self._cv2.rectangle(annotated, p1, p2, color, 2)
                self._cv2.putText(
                    annotated,
                    f"T{tr.get('track_id')} {kind}",
                    (p1[0], max(16, p1[1] - 5)),
                    self._cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    self._cv2.LINE_AA,
                )
            annotated_path = self._evidence_dir / (
                f"brain_chunk_{int(chunk_id):06d}_annotated.jpg")
            self._cv2.imwrite(
                str(annotated_path), annotated,
                [int(self._cv2.IMWRITE_JPEG_QUALITY), 82])
        except Exception:
            # Evidence capture must never affect the live pipeline.
            pass

    def _draw_game(self, canvas, mirror_env, obs, game_w):
        cv2 = self._cv2
        if mirror_env is None or mirror_env.payload.get("data_ref") is None:
            self._put(canvas, 24, 44, "WAITING FOR GAME FRAME...",
                      (120, 180, 240), scale=0.55)
            return

        img = mirror_env.payload["data_ref"]
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]
        preview_w = min(game_w, 480) if self.lightweight else game_w
        scale = min(preview_w / w, self.height / h)
        out_w, out_h = int(w * scale), int(h * scale)

        now = time.monotonic()
        refresh = (
            self._preview_cache is None
            or not self.lightweight
            or (now - self._preview_updated_at) >= self.preview_interval_s
            or self._preview_cache.shape[:2] != (out_h, out_w)
        )
        if refresh:
            self._preview_cache = cv2.resize(
                img, (out_w, out_h), interpolation=cv2.INTER_AREA)
            self._preview_updated_at = now
        mirror = self._preview_cache
        canvas[0:out_h, 0:out_w] = mirror
        if self.lightweight:
            self._put(
                canvas, 14, out_h + 24,
                "LOW-LOAD PREVIEW (control does not depend on dashboard FPS)",
                (150, 150, 150), scale=0.34)

        notes = (obs or {}).get("notes") or {}
        # Show the exact navigation cue selected by fast vision so the
        # user can verify that the fly is following the real GPO waypoint.
        wp = notes.get("recommended_waypoint_xy")
        dshape = notes.get("detect_shape")
        if (wp and dshape and len(dshape) >= 2
                and float(dshape[1]) > 0 and float(dshape[0]) > 0):
            px = int(float(wp[0]) / float(dshape[1]) * out_w)
            py = int(float(wp[1]) / float(dshape[0]) * out_h)
            cv2.circle(canvas, (px, py), 16, (0, 255, 0), 2)
            self._put(
                canvas, max(4, px - 74), max(16, py - 20),
                "NAV WAYPOINT", (0, 255, 0), scale=0.34)

        tracks = notes.get("entity_tracks") or []
        for tr in tracks:
            box = tr.get("bbox") or []
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            p1 = (int(x1 * out_w), int(y1 * out_h))
            p2 = (int(x2 * out_w), int(y2 * out_h))
            kind = str(tr.get("kind", "unknown"))
            if kind == "quest_npc":
                color = (0, 220, 255)
            elif kind == "hostile_candidate":
                color = (80, 80, 255)
            else:
                color = (255, 180, 70)
            cv2.rectangle(canvas, p1, p2, color, 1)
            self._put(
                canvas, p1[0], max(14, p1[1] - 5),
                f"T{tr.get('track_id')} {kind}", color, scale=0.32)

        border_w = out_w if self.lightweight else game_w
        border_h = out_h if self.lightweight else self.height
        cv2.rectangle(canvas, (0, 0), (border_w - 1, border_h - 1),
                      (65, 65, 65), 1)
        self._put(canvas, 14, self.height - 16,
                  "GAME / PERCEPTION VIEW  [ENGINEERED CV]",
                  (180, 180, 180), scale=0.40)

    def _draw_decoder_flow(self, canvas, x0, y0, rates):
        cv2 = self._cv2
        # These lines are semantic decoder-flow links, not anatomical edges.
        sx = x0 + 25
        mx = x0 + 205
        ix = x0 + 415

        positions = {
            "target_left": (sx, y0 + 34),
            "target_right": (sx, y0 + 92),
            "looming": (sx, y0 + 150),
            "P9_left": (mx, y0 + 24),
            "P9_right": (mx, y0 + 76),
            "MDN": (mx, y0 + 128),
            "GF": (mx, y0 + 180),
            "STOP": (mx, y0 + 232),
            "INTENT": (ix, y0 + 124),
        }

        for a, b in (
            ("target_left", "P9_left"),
            ("target_right", "P9_right"),
            ("looming", "GF"),
            ("P9_left", "INTENT"),
            ("P9_right", "INTENT"),
            ("MDN", "INTENT"),
            ("GF", "INTENT"),
            ("STOP", "INTENT"),
        ):
            cv2.line(canvas, positions[a], positions[b], (62, 62, 62), 1,
                     cv2.LINE_AA)

        self._activity_node(
            canvas, *positions["target_left"], "sens L",
            rates.get("_sens_left", 0.0), max_rate=150.0,
            color=(170, 210, 100))
        self._activity_node(
            canvas, *positions["target_right"], "sens R",
            rates.get("_sens_right", 0.0), max_rate=150.0,
            color=(170, 210, 100))
        self._activity_node(
            canvas, *positions["looming"], "loom",
            rates.get("_loom", 0.0), max_rate=150.0,
            color=(90, 170, 255))

        self._activity_node(
            canvas, *positions["P9_left"], "P9-L",
            rates.get("P9_left", 0.0), max_rate=30.0)
        self._activity_node(
            canvas, *positions["P9_right"], "P9-R",
            rates.get("P9_right", 0.0), max_rate=30.0)
        mdn = max(
            self._safe_float(rates.get("MDN_bilateral")),
            self._safe_float(rates.get("MDN_left")),
            self._safe_float(rates.get("MDN_right")),
        )
        gf = max(
            self._safe_float(rates.get("GF_left")),
            self._safe_float(rates.get("GF_right")),
        )
        stop = max(
            self._safe_float(rates.get("FG_bilateral")),
            self._safe_float(rates.get("BB_bilateral")),
            self._safe_float(rates.get("FG_left")),
            self._safe_float(rates.get("FG_right")),
        )
        self._activity_node(canvas, *positions["MDN"], "MDN", mdn,
                            max_rate=12.0, color=(255, 170, 90))
        self._activity_node(canvas, *positions["GF"], "GF", gf,
                            max_rate=35.0, color=(80, 80, 255))
        self._activity_node(canvas, *positions["STOP"], "FG/BB", stop,
                            max_rate=12.0, color=(180, 130, 240))

        cv2.circle(canvas, positions["INTENT"], 27, (50, 50, 50), -1,
                   cv2.LINE_AA)
        cv2.circle(canvas, positions["INTENT"], 27, (200, 200, 200), 1,
                   cv2.LINE_AA)
        self._put(canvas, positions["INTENT"][0] - 22,
                  positions["INTENT"][1] + 4, "INTENT",
                  (230, 230, 230), scale=0.30)

    def _draw_history(self, canvas, x, y, w, h):
        cv2 = self._cv2
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 55, 55), 1)
        if not self._history:
            self._put(canvas, x + 8, y + 22, "No brain chunks yet",
                      (150, 150, 150), scale=0.38)
            return
        vals = list(self._history)
        # Fixed-width bars keep the first few chunks readable. Previously
        # thickness scaled with spacing, producing huge circles when only a
        # handful of slow canonical chunks had completed.
        bar_w = 7
        gap = 3
        capacity = max(1, w // (bar_w + gap))
        shown = vals[-capacity:]
        colors = {
            "STOP": (150, 150, 150),
            "TURN_LEFT": (220, 180, 80),
            "TURN_RIGHT": (220, 180, 80),
            "APPROACH": (100, 220, 100),
            "RETREAT": (80, 170, 255),
            "ESCAPE": (80, 80, 255),
        }
        for i, item in enumerate(shown):
            name = item.get("intention") or "?"
            active = self._safe_float(item.get("active"))
            bar_h = min(h - 22, max(3, int(3 + active / 4.0)))
            xx = x + 4 + i * (bar_w + gap)
            yy = y + h - 4
            cv2.rectangle(
                canvas, (xx, yy - bar_h), (xx + bar_w, yy),
                colors.get(name, (180, 180, 180)), -1)
        self._put(canvas, x + 8, y + 17,
                  "WHOLE-BRAIN ACTIVITY HISTORY (active neurons/chunk)",
                  (175, 175, 175), scale=0.34)

    def _compose(self, snap: Snapshot, mirror_env):
        import numpy as np
        cv2 = self._cv2
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        d = snap.data
        brain = d.get("brain.output") or {}
        obs = d.get("world.observation") or {}
        ch = d.get("fly.channels") or {}
        goal = d.get("helper.goal") or {}
        act = d.get("action.selected") or {}
        brain_meta = d.get("brain.meta") or {}
        action_meta = d.get("action.meta") or {}
        quest_state = d.get("quest.state") or {}
        coach = d.get("coach.plan") or {}
        command = d.get("action.command") or {}

        game_w = int(self.width * 0.56)
        self._draw_game(canvas, mirror_env, obs, game_w)

        x0 = game_w + 18
        panel_w = self.width - x0 - 14

        chunk_id = brain.get("chunk_id")
        if chunk_id is not None and chunk_id != self._last_chunk_id:
            self._last_chunk_id = chunk_id
            intention = (brain.get("intention") or {}).get("name")
            self._history.append({
                "chunk": chunk_id,
                "intention": intention,
                "active": brain.get("n_active_new", 0),
                "spikes": brain.get("n_spikes_new", 0),
            })
            self._save_evidence_frame(mirror_env, chunk_id, obs)

        self._put(canvas, x0, 28, "DIGITAL DROSOPHILA — LIVE NEURAL ACTIVITY",
                  (120, 230, 120), scale=0.54, thickness=1)

        if not brain:
            self._put(canvas, x0, 58,
                      "CANONICAL BRAIN INITIALIZING / PREWARMING...",
                      (90, 190, 255), scale=0.46)
            self._put(canvas, x0, 80,
                      "Game perception is live; neural output appears after READY.",
                      (165, 165, 165), scale=0.36)
        else:
            intention = brain.get("intention") or {}
            self._put(
                canvas, x0, 58,
                f"INTENTION: {intention.get('name')}   "
                f"confidence={intention.get('confidence')}",
                (230, 230, 230), scale=0.50, thickness=1)
            self._put(
                canvas, x0, 80,
                f"chunk={brain.get('chunk_id')}  bio_t={self._safe_float(brain.get('t_bio_s')):.3f}s  "
                f"wall/chunk={self._safe_float(brain.get('chunk_wall_s')):.2f}s  "
                f"transport={brain.get('transport')}",
                (165, 165, 165), scale=0.34)
            scope = brain.get("instrumentation_scope", "")
            scope_tag = (
                " [count-only]" if scope == "whole_brain_counts_only" else "")
            self._put(
                canvas, x0, 100,
                f"whole brain{scope_tag}: spikes={brain.get('n_spikes_new', 0)}  "
                f"active neurons={brain.get('n_active_new', 0)} / 138639",
                (165, 215, 165), scale=0.38)

        sensory = brain.get("sensory_rates_hz") or {}
        self._bar(canvas, x0, 126, 170, "D8 target_left Hz",
                  sensory.get("target_left", 0.0), 150.0,
                  color=(170, 210, 100))
        self._bar(canvas, x0 + 190, 126, 170, "D8 target_right Hz",
                  sensory.get("target_right", 0.0), 150.0,
                  color=(170, 210, 100))
        self._bar(canvas, x0 + 380, 126, min(160, panel_w - 380),
                  "D8 looming Hz", sensory.get("looming", 0.0), 150.0,
                  color=(90, 170, 255))

        self._put(canvas, x0, 162,
                  "DECODER READOUT FLOW  (lines are NOT connectome edges)",
                  (150, 150, 150), scale=0.34)

        intention = brain.get("intention") or {}
        dn = dict(intention.get("dn_rates_hz") or brain.get("dn_rates_hz") or {})
        dn["_sens_left"] = sensory.get("target_left", 0.0)
        dn["_sens_right"] = sensory.get("target_right", 0.0)
        dn["_loom"] = sensory.get("looming", 0.0)
        self._draw_decoder_flow(canvas, x0, 176, dn)

        ybars = 465
        self._put(canvas, x0, ybars - 12,
                  "DESCENDING POPULATION FIRING RATES",
                  (175, 210, 175), scale=0.38)
        key_pops = [
            ("P9_left", 30.0), ("P9_right", 30.0),
            ("BPN_bilateral", 20.0), ("RRN_bilateral", 20.0),
            ("MDN_bilateral", 12.0), ("GF_left", 35.0),
            ("GF_right", 35.0), ("FG_bilateral", 12.0),
            ("BB_bilateral", 12.0),
        ]
        for i, (name, vmax) in enumerate(key_pops):
            col = i % 3
            row = i // 3
            self._bar(canvas, x0 + col * 185, ybars + row * 31,
                      160, name, dn.get(name, 0.0), vmax)

        sample = brain.get("active_flywire_ids_sample") or []
        if sample:
            short_ids = " ".join(str(x)[-6:] for x in sample[:10])
            self._put(canvas, x0, 570,
                      "spiking FlyWire ID sample: " + short_ids,
                      (150, 150, 190), scale=0.31)
        else:
            self._put(canvas, x0, 570,
                      "spiking FlyWire ID sample: waiting for brain chunk",
                      (120, 120, 140), scale=0.31)

        self._draw_history(canvas, x0, 590, min(panel_w, 540), 64)

        notes = obs.get("notes") or {}
        player = obs.get("player") or {}
        target = obs.get("target") or {}
        mode = str(action_meta.get("mode") or "passive")
        if coach:
            if coach.get("enabled") is False:
                coach_line = (
                    "AI -> FLY: DISABLED - "
                    f"{coach.get('reason', 'coach unavailable')}")
                coach_why = ""
            else:
                card = coach.get("skill_card")
                provider = coach.get("provider")
                used = coach.get("ai_used")
                if used is None:
                    used = (
                        coach.get("local_ai_used")
                        if coach.get("local_ai_used") is not None
                        else coach.get("local_vlm_used"))
                ai_tag = (
                    "AI=YES" if used is True
                    else "AI=FALLBACK" if used is False
                    else "AI=?")
                coach_line = (
                    f"AI -> FLY: {coach.get('skill') or '-'}  "
                    f"target={coach.get('target') or '-'}  {ai_tag}"
                    + (f"  card={card}" if card else ""))
                why = str(coach.get("explanation") or "")
                if len(why) > 82:
                    why = why[:79] + "..."
                coach_why = (
                    f"WHY: {why or '-'}"
                    + (f"  via={provider}" if provider else ""))
            self._put(
                canvas, x0, 662, coach_line,
                (190, 170, 245), scale=0.31)
            if coach_why:
                self._put(
                    canvas, x0, 679, coach_why,
                    (175, 155, 225), scale=0.29)
        self._put(
            canvas, x0, 697,
            f"PERCEPTION -> AI: target={target.get('type')}  "
            f"red_enemy={notes.get('quest_enemy_marker_detected', False)}",
            (120, 180, 240), scale=0.31)
        self._put(
            canvas, x0, 714,
            f"FLY/EXECUTOR: intention={intention.get('name') or '-'}  "
            f"command={command.get('name') or '-'}  "
            f"health={player.get('health')}",
            (120, 180, 240), scale=0.31)
        if mode == "quest_pve_v1":
            dv = quest_state.get("defense_values") or {}
            self._put(
                canvas, x0, 731,
                f"QUEST/PVE: phase={quest_state.get('phase')}  "
                f"goal={(goal.get('goal') or {}).get('label')}  "
                f"block={dv.get('block')} evade={dv.get('evade_back')}",
                (110, 205, 235), scale=0.29)

        active = bool(action_meta.get(
            "autonomy", act.get("autonomy", False)))
        ready = bool(brain_meta.get("ready", False))
        controls_available = bool(
            action_meta.get("movement_control_available", False))

        if not controls_available:
            state_text = "AGENT: PASSIVE MODE"
            state_color = (150, 150, 150)
        elif not ready:
            state_text = "AGENT: LOCKED — BRAIN NOT READY"
            state_color = (80, 190, 255)
        elif mode == "quest_pve_v1":
            reason = str(action_meta.get("last_control_reason") or "")
            state_text = ("QUEST/PVE AGENT: ENABLED" if active
                          else "QUEST/PVE AGENT: DISABLED")
            if reason and not active:
                state_text += f" [{reason}]"
            state_color = ((100, 230, 100) if active
                           else (120, 180, 255))
        else:
            reason = str(action_meta.get("last_control_reason") or "")
            state_text = ("MOVEMENT: ENABLED" if active
                          else "MOVEMENT: DISABLED")
            if reason and not active:
                state_text += f" [{reason}]"
            state_color = ((100, 230, 100) if active
                           else (120, 180, 255))
        self._put(canvas, x0, 747, state_text, state_color,
                  scale=0.37, thickness=1)

        # Explicit controls instead of a single toggle. These are all
        # safety/run controls; none injects an unapproved game action.
        labels = [
            ("enable_movement", "ENABLE", (55, 145, 55)),
            ("disable_movement", "DISABLE", (60, 60, 190)),
            ("refocus_game", "REFOCUS", (110, 95, 55)),
            ("release_keys", "RELEASE KEYS", (115, 75, 45)),
            ("end_run", "END RUN", (85, 85, 85)),
            ("emergency_stop", "EMERGENCY STOP", (55, 55, 185)),
        ]
        self._button_rects = {}
        bx = x0
        by = 760
        bw = 88
        gap = 6
        for idx, (command, label, color) in enumerate(labels):
            rect = (bx + idx * (bw + gap), by,
                    bx + idx * (bw + gap) + bw, 800)
            enabled = True
            if command == "enable_movement":
                enabled = bool(controls_available and ready and not active)
            elif command == "disable_movement":
                enabled = bool(controls_available and active)
            elif command in ("release_keys", "refocus_game"):
                enabled = bool(controls_available and ready)
            if enabled:
                self._button_rects[command] = rect
                self._button(canvas, rect, label, color)
            else:
                self._button(canvas, rect, label, (65, 65, 65),
                             text_color=(150, 150, 150))

        bh = act.get("backend_health") or {}
        self._put(
            canvas, x0 + 405, 742,
            f"SendInput ok={bh.get('sendinput_successes', 0)} "
            f"fail={bh.get('sendinput_failures', 0)}",
            (160, 190, 160) if not bh.get("sendinput_failures")
            else (100, 100, 255), scale=0.29)
        self._put(
            canvas, x0 + 405, 758,
            "F12 = EMERGENCY STOP",
            (120, 120, 255), scale=0.29)
        return canvas

