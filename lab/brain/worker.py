"""Brain worker: the live digital-Drosophila decision loop (PROCESS 4).

FINAL_ARCHITECTURE §3/§4/§7: init once → prewarm → run continuously.
The worker owns ONE runtime instance for its whole lifetime; each loop
iteration:

  1. reads the LATEST FlyChannels (state channel — never a backlog)
  2. converts to VERIFIED D8 sensory rates (target_left / target_right /
     looming)
  3. advances ONE simulation chunk (25/50/100 ms — benchmark-chosen)
  4. decodes the Intention from DN population rates (IntentionDecoder —
     neural evidence only)
  5. publishes brain.output state + brain.events stream (replay)

The worker does NOT pace itself to wall-clock by default — it advances
the next chunk as soon as the previous completes, because on weak
hardware the canonical brain runs slower than real time (measured, never
hidden: stats carry achieved_hz vs bio-s/wall-s). The MotorExecutor's
held-key state covers inter-tick continuity (spec §19).

Mode B (BIOLOGICAL_PLASTICITY_FLY) is NOT implemented live: the frozen
Gate-4 v2 plasticity system is an offline experimental record; using it
live would require a new pre-registration. Fail closed.
"""

from __future__ import annotations

from typing import Optional

from ..bus import Bus, StateChannel, StreamChannel
from ..modes import AgentMode
from ..worker import Worker
from .intent import CONFIG_ID, IntentionDecoder
from .runtime import (BrainChunkRecord, CanonicalBrianRuntime,
                      MockBrainRuntime, create_runtime)


class BrainWorker(Worker):
    name = "brain_worker"
    TOPIC_IN = "fly.channels"          # state channel: FlyChannels dict
    TOPIC_OUT = "brain.output"         # state channel: latest intention+chunk
    TOPIC_EVT = "brain.events"         # stream: every chunk (replay)

    def __init__(self, bus: Bus, target_hz: float = 10.0,
                 runtime: Optional[object] = None,
                 runtime_kind: str = "mock",
                 chunk_ms: float = 50.0,
                 decoder_params: Optional[dict] = None,
                 agent_mode: AgentMode = AgentMode.HYBRID_FLY,
                 prewarm_chunks: int = 2):
        super().__init__(bus, target_hz=target_hz)
        if agent_mode is AgentMode.BIOLOGICAL_PLASTICITY_FLY:
            raise RuntimeError(
                "Mode B (biological plasticity) is FROZEN as an offline "
                "Gate-4 v2 record — live use requires a new pre-registered "
                "model-side iteration, not this worker")
        self.channels_in: StateChannel = bus.state(self.TOPIC_IN)
        self.out: StateChannel = bus.state(self.TOPIC_OUT)
        self.events: StreamChannel = bus.stream(self.TOPIC_EVT, maxsize=128)
        self.chunk_ms = float(chunk_ms)
        self.runtime = runtime if runtime is not None else \
            create_runtime(runtime_kind, chunk_ms=self.chunk_ms)
        self.decoder = None
        self._decoder_params = decoder_params or {}
        self._prewarm_chunks = prewarm_chunks
        self.current: Optional[dict] = None   # last published output
        self.bio_to_wall = 0.0                # bio-s per wall-s (honest rate)

    # -- lifecycle ------------------------------------------------------------
    def on_start(self) -> None:
        t0 = self.clock.now_ns()
        self.runtime.init_once()
        init_ms = self.clock.elapsed_ms(t0)
        t1 = self.clock.now_ns()
        self.runtime.prewarm(self._prewarm_chunks)
        prewarm_ms = self.clock.elapsed_ms(t1)
        self.decoder = IntentionDecoder(self.runtime.pop_sizes(),
                                        params=self._decoder_params)
        self.stats["init_ms"] = round(init_ms, 1)
        self.stats["prewarm_ms"] = round(prewarm_ms, 1)
        self.stats["runtime"] = self.runtime.runtime_label
        self.stats["runtime_transport"] = getattr(
            self.runtime, "transport_label", "inprocess")
        self.stats["decoder_config"] = self.decoder.decoder_config
        # seed the input channel with silence so the first chunk is valid
        if self.channels_in.read() is None:
            self.channels_in.write(
                {"target_left": 0.0, "target_right": 0.0,
                 "target_distance": 0.0, "threat_left": 0.0,
                 "threat_right": 0.0, "threat_intensity": 0.0,
                 "attack_opportunity": 0.0, "avoidance_value": 0.0,
                 "approach_value": 0.0, "goal_relevance": 0.5,
                 "low_health_context": 0.0, "ts_ns": self.clock.now_ns(),
                 "target_center": 0.0})

    def on_stop(self) -> None:
        self.runtime.close()

    def stop(self, join_timeout: float = 8.0) -> None:
        """Stop cleanly even when a canonical chunk is still in flight."""
        self.stop_event.set()
        th = self._thread
        if th is None:
            return
        th.join(timeout=join_timeout)
        if th.is_alive():
            force = getattr(self.runtime, "force_terminate", None)
            if force is not None:
                force()
                th.join(timeout=2.0)

    # -- main step ----------------------------------------------------------
    def step(self) -> None:
        from ..perception.fly_channels import to_sensory_rates
        from ..schemas import FlyChannels

        snap = self.channels_in.read()
        if snap is None:
            return
        ch = FlyChannels.from_dict(snap.payload)
        rates = to_sensory_rates(ch)          # -> D8 channel names
        d8_rates = {"target_left": rates["target_left_hz"],
                    "target_right": rates["target_right_hz"],
                    "looming": rates["looming_hz"]}
        rec: BrainChunkRecord = self.runtime.advance_chunk(d8_rates,
                                                           chunk_ms=self.chunk_ms)
        intention = self.decoder.decode(rec)

        out = {
            "ts_ns": self.clock.now_ns(),
            "runtime": rec["runtime"],
            "transport": rec.get(
                "transport", getattr(self.runtime, "transport_label", "inprocess")),
            "chunk_id": rec["chunk_id"],
            "t_bio_s": rec["t_bio_s"],
            "chunk_ms": rec["chunk_ms"],
            "chunk_wall_s": rec["wall_s"],
            "sensory_rates_hz": rec["sensory_rates_hz"],
            "dn_rates_hz": rec["pop_rates_hz"],
            "intention": intention.to_dict(),
            "decoder_config": self.decoder.decoder_config,
        }
        self.current = out
        self.out.write(out, ts_ns=out["ts_ns"])
        self.events.publish({"kind": "brain_chunk", **out}, ts_ns=out["ts_ns"])
        # honest throughput accounting
        if rec["wall_s"] > 0:
            self.bio_to_wall = (rec["chunk_ms"] / 1000.0) / rec["wall_s"]
        self.stats["bio_s_per_wall_s"] = round(self.bio_to_wall, 4)
        self.stats["last_chunk_wall_ms"] = round(rec["wall_s"] * 1000, 1)
