"""Simulated Drosophila: burst-pause random walk with optional cue taxis.

PURPOSE (scientifically important): this model is the "known ground truth"
fly used to validate the measurement apparatus WITHOUT an animal.

  cue_bias = 0.0  -> pure random walk (fly ignores the stimulus).
                      The fly-controlled system must then perform at the
                      RANDOM floor. If it performed better, the apparatus
                      (not the fly) would be generating the behavior.
  cue_bias > 0    -> taxis toward the cue. The cue is modeled as a POINT
                      ATTRACTOR at the center of the cued edge (a fly walks
                      to the stimulus and rests at it, as real flies do at
                      a preferred stripe/edge). Direction-pull cues were
                      tested and rejected: they park flies in corners where
                      no wall-press intent can be decoded.

Locomotion model: alternating WALK / PAUSE bouts (typical of freely walking
Drosophila), heading diffusion during walking, position clamped near walls
(wall-following; flies do not bounce off walls).

NO claim is made that this model equals a real fly; it exists so that the
tracking -> decoding -> control -> measurement chain is testable end-to-end
before any animal is involved.
"""
from __future__ import annotations

import math

import numpy as np

# Attractor points: center of each edge, inset from the wall. The inset
# (0.12) is inside the decoder's edge band (0.15), so a resting fly is
# unambiguously "pressing" that edge.
ATTRACTOR = {
    "LEFT": (0.12, 0.50),
    "RIGHT": (0.88, 0.50),
    "FORWARD": (0.50, 0.12),
    "BACKWARD": (0.50, 0.88),
}
# Geometry contract with the decoder's edge band:
#   attractor inset (0.12/0.88) - REST_RADIUS must be INSIDE the band
#   (edge_band = 0.20 in action_decoder): resting flies always sit at
#   x/y in [0.82, 0.94], which is unambiguously inside [0.80, 1.00].
REST_RADIUS = 0.06      # within this normalized distance the fly rests


def _ang_diff(target: float, current: float) -> float:
    """Shortest signed angular difference target - current in [-pi, pi]."""
    return (target - current + math.pi) % (2 * math.pi) - math.pi


class SyntheticFly:
    def __init__(self, rng: np.random.Generator, cfg: dict,
                 arena_mm: tuple[float, float] = (90.0, 60.0)):
        self.rng = rng
        self.cfg = dict(cfg or {})
        self.arena_mm = arena_mm
        self.position = (rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65))
        self.heading = float(rng.uniform(-math.pi, math.pi))
        self.mode = self.cfg.get("start_mode", "walk")
        self._new_bout()
        self.cue: str | None = None

    # ------------------------------------------------------------------
    def set_cue(self, action: str | None) -> None:
        self.cue = action

    def _new_bout(self) -> None:
        if self.mode == "walk":
            lo, hi = self.cfg.get("pause_duration_s", [0.2, 1.0])
            self._timer = float(self.rng.uniform(lo, hi))
            self.mode = "pause"
        else:
            lo, hi = self.cfg.get("walk_duration_s", [0.5, 2.0])
            self._timer = float(self.rng.uniform(lo, hi))
            self.mode = "walk"
            lo, hi = self.cfg.get("walk_speed_mm_s", [8.0, 26.0])
            self._speed = float(self.rng.uniform(lo, hi))

    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        pull = 4.0  # rad/s max taxis turn rate (calibrated 2026-09)
        self._timer -= dt
        if self._timer <= 0:
            self._new_bout()

        x, y = self.position
        mm_w, mm_h = self.arena_mm
        bias = float(self.cfg.get("cue_bias", 0.0))

        # --- resting at the attractor (inspection / preening behavior) ---
        if self.cue is not None and bias > 0:
            ax, ay = ATTRACTOR[self.cue]
            d = math.hypot(ax - x, ay - y)
            if d < REST_RADIUS:
                # tiny jitter only (a preening fly) - stays within the band,
                # below the MOVING speed threshold -> decodable as a
                # wall-press pause at the cued edge
                jx = float(self.rng.normal(0, 0.0015))
                jy = float(self.rng.normal(0, 0.0015))
                self.position = (min(max(x + jx, 0.05), 0.95),
                                 min(max(y + jy, 0.05), 0.95))
                return

        if self.mode == "pause":
            return

        # --- walking ---
        sigma = float(self.cfg.get("turn_rate_sigma", 1.6))
        self.heading += float(self.rng.normal(0, sigma * math.sqrt(dt)))

        if self.cue is not None and bias > 0:
            ax, ay = ATTRACTOR[self.cue]
            dx_mm = (ax - x) * mm_w
            dy_mm = (ay - y) * mm_h
            target = math.atan2(dy_mm, dx_mm)
            d_ang = _ang_diff(target, self.heading)
            if abs(d_ang) > 0.17:
                self.heading += math.copysign(bias * pull * dt, d_ang)
            else:
                self.heading += d_ang  # snap into the small deadzone

        vx = math.cos(self.heading) * self._speed / mm_w  # norm/s
        vy = math.sin(self.heading) * self._speed / mm_h
        x += vx * dt
        y += vy * dt

        # Wall contact: clamp position only (no heading reflection).
        # Clamp band is 0.05..0.95 (= ~16 px from the arena edge at
        # 320 px width): pressing closer would clip the fly's blob at
        # the tracker ROI boundary and cause tracking loss in corners.
        # Cue-less flies grind along the band until heading noise turns
        # them away (~2 s) - realistic wall-following.
        x = min(max(x, 0.05), 0.95)
        y = min(max(y, 0.05), 0.95)
        self.position = (x, y)
