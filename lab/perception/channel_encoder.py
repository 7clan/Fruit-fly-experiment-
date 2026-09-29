"""FlyChannelEncoder worker: observation + helper context → fly.channels.

This is the junction where the ENGINEERED helper contributes ONLY
context (goal_relevance / avoidance_bias / approach_bias — WHAT MATTERS,
never which key to press) and perception's structured observation is
encoded into the compact sensory channels (lab/perception/fly_channels).

Runs at fast-vision rate (20–30 Hz) — it is ON the real-time path and
is therefore O(1) dict work only.
"""

from __future__ import annotations

from ..bus import Bus, StateChannel
from ..schemas import FlyChannels, WorldObservation
from ..worker import Worker


class FlyChannelEncoder(Worker):
    name = "fly_channel_encoder"
    TOPIC_OBS = "world.observation"
    TOPIC_GOAL = "helper.goal"
    TOPIC_OUT = "fly.channels"

    def __init__(self, bus: Bus, target_hz: float = 24.0):
        super().__init__(bus, target_hz=target_hz)
        self.obs: StateChannel = bus.state(self.TOPIC_OBS)
        self.goal: StateChannel = bus.state(self.TOPIC_GOAL)
        self.out: StateChannel = bus.state(self.TOPIC_OUT)

    def step(self) -> None:
        obs_snap = self.obs.read()
        if obs_snap is None:
            return
        obs = WorldObservation.from_dict(obs_snap.payload)
        goal_snap = self.goal.read()
        if goal_snap is not None:
            relevance = goal_snap.payload.get("goal_relevance", 0.5)
            avoidance = goal_snap.payload.get("avoidance_bias", 0.0)
            approach = goal_snap.payload.get("approach_bias", 0.0)
        else:
            relevance, avoidance, approach = 0.5, 0.0, 0.0
        ch: FlyChannels = FlyChannels.from_observation(
            obs, goal_relevance=relevance, avoidance_bias=avoidance,
            approach_bias=approach)
        self.out.write(ch.to_dict(), ts_ns=self.clock.now_ns())
