"""DigitalFlyLab lab/ package tests — bus, clock, schemas, pipeline,
action layer, memory/world, replay, modes.

Run:  python -m pytest tests/test_lab.py -q
(no brian2 required — the canonical runtime is import-guarded; the mock
runtime exercises the full pipeline contract).
"""

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.bus import (Bus, BusMode, RateGovernor, StateChannel,  # noqa: E402
                     StreamChannel)
from lab.clock import Clock
from lab.modes import (AgentMode, ModeConfig, DEFAULT_MODE,  # noqa: E402
                       validate_mode_config)
from lab.schemas import (AbilityAvailability, EnemyState, FlyChannels,  # noqa: E402
                         Intention, INTENTIONS, PlayerState, TargetState,
                         UIState, WorldObservation)


# ---------------------------------------------------------------------------
# clock
# ---------------------------------------------------------------------------

class TestClock:
    def test_monotonic_and_export_adopt(self):
        parent = Clock()
        t1 = parent.now_ns()
        time.sleep(0.001)
        spec = parent.export()
        child = Clock.adopt(spec)
        t2 = parent.now_ns()
        t3 = child.now_ns()
        # child continues the parent timeline (same reference frame)
        assert abs(t2 - t3) < 50_000_000  # < 50 ms tolerance for adoption
        assert t2 >= t1

    def test_elapsed_ms(self):
        c = Clock()
        t0 = c.now_ns()
        time.sleep(0.005)
        assert 3.0 <= Clock.elapsed_ms(t0) <= 100.0


# ---------------------------------------------------------------------------
# bus
# ---------------------------------------------------------------------------

class TestStreamChannel:
    def test_bounded_drop_oldest_never_blocks(self):
        ch = StreamChannel("t", maxsize=4)
        for i in range(20):
            ch.publish({"i": i})
        assert ch.qsize() == 4
        assert ch.dropped == 16
        items = ch.drain()
        assert [e.payload["i"] for e in items] == [16, 17, 18, 19]

    def test_newest_wins(self):
        ch = StreamChannel("t", maxsize=8)
        for i in range(10):
            ch.publish({"i": i})
        newest = ch.newest()
        assert newest.payload["i"] == 9
        assert ch.empty()

    def test_seq_strictly_increasing(self):
        ch = StreamChannel("t", maxsize=3)
        seqs = [ch.publish({"x": 1}).seq for _ in range(7)]
        assert seqs == sorted(seqs) and len(set(seqs)) == 7

    def test_ts_monotonic(self):
        ch = StreamChannel("t", maxsize=3)
        ts = [ch.publish({}).ts_ns for _ in range(10)]
        assert ts == sorted(ts)


class TestStateChannel:
    def test_latest_wins_and_missed(self):
        ch = StateChannel("s")
        assert ch.read() is None
        for i in range(5):
            ch.write({"v": i})
        snap = ch.read()
        assert snap.payload["v"] == 4
        assert snap.missed == 4       # 5 writes, first read missed 4

    def test_read_returns_freshest_after_overwrite(self):
        ch = StateChannel("s")
        ch.write({"v": 1})
        ch.read()
        ch.write({"v": 2})
        ch.write({"v": 3})
        snap = ch.read()
        assert snap.payload["v"] == 3
        assert snap.missed == 1


class TestBus:
    def test_specs_roundtrip(self):
        b = Bus()
        b.stream("capture.frames", maxsize=4)
        b.state("world.observation")
        b2 = Bus.from_specs(b.specs())
        assert b2.stream("capture.frames").maxsize == 4
        assert b2.state("world.observation") is not None

    def test_metrics(self):
        b = Bus()
        s = b.stream("s", maxsize=2)
        st = b.state("q")
        s.publish({})
        st.write({})
        m = b.metrics()
        assert m["streams"]["s"]["published"] == 1
        assert m["states"]["q"]["writes"] == 1


class TestRateGovernor:
    def test_paces_below_target(self):
        gov = RateGovernor(target_hz=50.0)
        t0 = time.monotonic()
        n = 20
        for _ in range(n):
            gov.tick()
        dt = time.monotonic() - t0
        assert dt >= 0.7 * n / 50.0    # approximately on period


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_world_observation_roundtrip(self):
        obs = WorldObservation(
            ts_ns=1, frame_ref={"frame_id": 3},
            player=PlayerState(position=[0.5, 0.6], heading=0.1,
                               health=0.8, stamina=0.9, health_units="fraction"),
            target=TargetState(type="quest_marker", direction=0.5,
                               distance=0.4, confidence=0.9),
            enemies=[EnemyState(direction=-0.2, distance=0.3, attacking=True,
                                threat=0.8, confidence=0.85, type="melee")],
            ui=UIState(combat=True),
            abilities=[AbilityAvailability(ability_id="a1", available=True)],
            notes={"x": 1})
        d = obs.to_dict()
        obs2 = WorldObservation.from_dict(json.loads(json.dumps(d)))
        assert obs2.primary_enemy().attacking is True
        assert obs2.target.confidence == 0.9
        assert obs2.ui.combat is True

    def test_fly_channels_roundtrip_and_vector(self):
        ch = FlyChannels(target_left=0.5, threat_intensity=0.9)
        v = ch.as_vector()
        assert len(v) == 12
        ch2 = FlyChannels.from_dict(json.loads(json.dumps(ch.to_dict())))
        assert ch2.threat_intensity == 0.9

    def test_intention_validation(self):
        i = Intention(name="APPROACH", confidence=0.8)
        assert i.name in INTENTIONS
        with pytest.raises(ValueError):
            Intention(name="PRESS_W_MACRO")     # not a canonical intention


# ---------------------------------------------------------------------------
# fly channel encoding
# ---------------------------------------------------------------------------

class TestFlyChannels:
    def _obs(self, target_dir=0.7, target_dist=0.5, enemy_dir=-0.6,
             enemy_dist=0.3, attacking=True):
        return WorldObservation(
            ts_ns=1,
            player=PlayerState(health=0.2, health_units="fraction"),
            target=TargetState(direction=target_dir, distance=target_dist,
                               confidence=0.9),
            enemies=[EnemyState(direction=enemy_dir, distance=enemy_dist,
                                attacking=attacking, threat=0.7,
                                confidence=0.9)])

    def test_lateralization(self):
        from lab.perception.fly_channels import encode, lateral_weights
        # target to the right -> right channel dominates
        ch = encode(self._obs(target_dir=1.0))
        assert ch.target_right > ch.target_left
        # centered -> center channel dominates
        ch = encode(self._obs(target_dir=0.0))
        assert ch.target_center > ch.target_left
        assert ch.target_center > ch.target_right
        # unknown bearing -> zeros
        l, r, c = lateral_weights(None)
        assert (l, r, c) == (0.0, 0.0, 0.0)

    def test_threat_and_windup(self):
        from lab.perception.fly_channels import encode
        ch_attack = encode(self._obs(attacking=True, enemy_dir=-0.8))
        ch_calm = encode(self._obs(attacking=False, enemy_dir=-0.8))
        assert ch_attack.threat_intensity > ch_calm.threat_intensity
        assert ch_attack.threat_left > ch_attack.threat_right  # enemy left
        assert ch_calm.attack_opportunity == 1.0               # in range, calm

    def test_low_health_context(self):
        from lab.perception.fly_channels import encode
        ch = encode(self._obs())
        assert ch.low_health_context == 1.0     # health 0.2 < 0.25

    def test_to_sensory_rates_maps_to_verified_channels(self):
        from lab.perception.fly_channels import to_sensory_rates
        rates = to_sensory_rates(FlyChannels(target_left=1.0,
                                             threat_intensity=1.0))
        assert set(rates) == {"target_left_hz", "target_right_hz", "looming_hz"}
        assert rates["target_left_hz"] == 150.0   # frozen D8 target drive ceiling
        assert rates["looming_hz"] == 150.0       # Gate-2 looming drive scale


# ---------------------------------------------------------------------------
# intention decoder
# ---------------------------------------------------------------------------

class TestIntentionDecoder:
    def _chunk(self, counts, chunk_ms=50.0, chunk_id=0):
        return {"pop_counts": counts, "chunk_ms": chunk_ms,
                "chunk_id": chunk_id, "ts_ns": 1}

    def _decoder(self):
        from lab.brain.intent import IntentionDecoder
        sizes = {"P9_left": 20, "P9_right": 20, "BPN_bilateral": 20,
                 "RRN_bilateral": 20, "MDN_bilateral": 20, "GF_left": 20,
                 "FG_bilateral": 20, "BB_bilateral": 20}
        return IntentionDecoder(sizes)

    def test_escape_on_giant_fiber(self):
        dec = self._decoder()
        # 30 Hz * 20 neurons * 0.3 s = 180 spikes in window
        for i in range(6):
            it = dec.decode(self._chunk({"GF_left": 180 // 6}))
        assert it.name == "ESCAPE"

    def test_turn_lateralization(self):
        dec = self._decoder()
        # P9_left strongly dominant (D9 rule: left P9 -> ipsiversive left turn)
        for i in range(6):
            it = dec.decode(self._chunk({"P9_left": 15, "P9_right": 0}))
        assert it.name == "TURN_LEFT"

    def test_stop_failsafe(self):
        dec = self._decoder()
        for i in range(6):
            it = dec.decode(self._chunk({}))       # total silence
        assert it.name == "STOP"
        assert it.notes.get("rule") == "failsafe"

    def test_decoder_never_sees_world_state(self):
        # decode() only reads pop_counts + chunk_ms from the chunk record;
        # extra keys (e.g. observation) are ignored — enforced by contract
        dec = self._decoder()
        it = dec.decode(self._chunk({}) | {"observation": {"fake": 1}})
        assert it.name == "STOP"

    def test_retreat_on_mdn(self):
        dec = self._decoder()
        for i in range(6):
            it = dec.decode(self._chunk({"MDN_bilateral": 30}))
        assert it.name == "RETREAT"


# ---------------------------------------------------------------------------
# ability registry + resolver
# ---------------------------------------------------------------------------

class TestAbilitySystem:
    def _registry(self):
        from lab.action.ability_registry import AbilityRecord, AbilityRegistry
        r = AbilityRegistry()
        r.register(AbilityRecord("slash", name="Slash", input_binding="key:Q",
                                 intent_tags=["ATTACK_LIGHT", "APPROACH"],
                                 range="close", confidence=0.9))
        r.register(AbilityRecord("pistol", name="Pistol",
                                 intent_tags=["ATTACK_RANGED"],
                                 range="long", confidence=0.7,
                                 learned_utility=0.8))
        r.register(AbilityRecord("block", name="Block",
                                 intent_tags=["DEFEND"],
                                 defensive_property="block",
                                 range="close", confidence=0.8))
        return r

    def test_registry_validation_and_cache(self, tmp_path):
        from lab.action.ability_registry import AbilityRecord
        r = self._registry()
        with pytest.raises(KeyError):
            r.register(AbilityRecord("slash", intent_tags=["ATTACK_LIGHT"]))
        with pytest.raises(ValueError):
            AbilityRecord("bad", intent_tags=["NOT_AN_INTENTION"])
        p = tmp_path / "abilities.json"
        r.cache_path = p
        r.save()
        r2 = self._registry()
        r2.cache_path = p
        r2.load()
        assert r2.size() == 3
        assert r2.get("pistol").learned_utility == 0.8

    def test_hard_gate_never_overrides_category(self):
        from lab.action.ability_resolver import AbilityResolver
        from lab.schemas import Intention
        res = AbilityResolver(self._registry())
        # fly says DEFEND; only 'block' is compatible — even though pistol
        # has the highest learned utility, it MUST NOT be chosen
        ra = res.resolve(Intention(name="DEFEND"), distance=0.9)
        assert ra.ability_id == "block"
        # fly says ATTACK_RANGED far away; slash is NOT compatible
        ra = res.resolve(Intention(name="ATTACK_RANGED"), distance=0.1)
        assert ra.ability_id == "pistol"

    def test_blocked_when_no_compatible_available(self):
        from lab.action.ability_resolver import AbilityResolver
        from lab.schemas import Intention
        res = AbilityResolver(self._registry())
        ra = res.resolve(Intention(name="ESCAPE"),
                         cooldown_state={"block": False})
        # ESCAPE has no compatible ability registered -> blocked, no substitution
        assert ra.ability_id == ""
        assert ra.blocked_reason == "no_compatible_available"

    def test_cooldown_observed_blocks(self):
        from lab.action.ability_resolver import AbilityResolver
        from lab.schemas import Intention
        res = AbilityResolver(self._registry())
        ra = res.resolve(Intention(name="DEFEND"),
                         cooldown_state={"block": False})
        assert ra.blocked_reason == "no_compatible_available"

    def test_range_fit_scoring(self):
        from lab.action.ability_resolver import AbilityResolver
        from lab.schemas import Intention
        res = AbilityResolver(self._registry())
        ra = res.resolve(Intention(name="ATTACK_RANGED"), distance=0.05)
        assert ra.ability_id == "pistol"
        assert ra.candidates[0][0] == "pistol"

    def test_record_result_updates_utility(self):
        r = self._registry()
        r.record_result("pistol", success=True)
        r.record_result("pistol", success=True)
        assert r.get("pistol").learned_utility == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# motor executor (shadow mode = default, gated autonomy)
# ---------------------------------------------------------------------------

class TestMotorExecutor:
    def _lab(self):
        from lab.app import DigitalFlyLab
        return DigitalFlyLab(dashboard=False)

    def test_shadow_mode_emits_no_real_inputs(self):
        lab = self._lab()
        lab.start()
        try:
            lab.drive_synthetic(seconds=1.0, fps=20.0)
        finally:
            lab.stop()
        assert lab.executor.autonomy_enabled is False
        assert lab.executor.backend.name == "safe_noop"
        assert lab.executor.stats["inputs_emitted"] == 0

    def test_hold_refresh_no_retrigger(self):
        lab = self._lab()
        lab.start()
        try:
            lab.drive_synthetic(seconds=1.0, fps=20.0)
        finally:
            lab.stop()
        # repeated identical actions refresh holds instead of re-triggering
        assert lab.executor.stats["hold_refreshes"] > 0

    def test_autonomy_cannot_be_enabled_via_dev_cli(self):
        import lab.app as app
        with pytest.raises(SystemExit):
            app.main(["--autonomy"])


# ---------------------------------------------------------------------------
# memory / value / world model [ENGINEERED]
# ---------------------------------------------------------------------------

class TestWorldLayer:
    @staticmethod
    def _lab():
        from lab.app import DigitalFlyLab
        return DigitalFlyLab(dashboard=False)

    def test_memory_async_and_drop_honesty(self, tmp_path):
        from lab.world.memory import MemoryStore
        m = MemoryStore(tmp_path, max_queue=8)
        m.start()
        for i in range(50):
            m.put_semantic(f"k{i}", i)
        m.stop(persist=True)
        assert m.written + m.dropped >= 50
        assert (tmp_path / "semantic.json").exists()
        assert (tmp_path / "episodic.jsonl").exists()

    def test_value_table_laplace(self):
        from lab.world.value import ValueTable
        v = ValueTable()
        assert v.value(("enemy", "melee"), "slash") == 0.5   # default
        for _ in range(3):
            v.observe(("enemy", "melee"), "slash", True)
        v.observe(("enemy", "melee"), "slash", False)
        # Laplace: (3 succ + 1) / (4 uses + 2)
        assert v.value(("enemy", "melee"), "slash") == pytest.approx(4 / 6)
        best, val = v.best(("enemy", "melee"), ["slash", "pistol"])
        assert best == "slash"

    def test_world_model_entities_and_observation_wins(self):
        from lab.world.world_model import Entity, WorldModel
        wm = WorldModel()
        wm.upsert_entity(Entity("w1", "WEAPON", label="Cutlass",
                                tier="UNVERIFIED"))
        notes = wm.update_from_observation({
            "ts_ns": 5,
            "player": {"position": [0.4, 0.6], "heading": 0.0},
            "enemies": [{"type": "melee", "distance": 0.3,
                         "direction": 0.1, "attacking": False,
                         "threat": 0.2, "confidence": 0.9}],
        })
        assert any("player entity" in n for n in notes)
        assert wm.get_entity("enemy:live:0").tier == "VERIFIED"
        # verified knowledge not downgraded by unverified guide claims
        wm.upsert_entity(Entity("enemy:live:0", "ENEMY", tier="UNVERIFIED"))
        assert wm.get_entity("enemy:live:0").tier == "VERIFIED"

    def test_planner_outputs_goals_not_keys(self):
        from lab.schemas import EnemyState, PlayerState, TargetState, UIState, WorldObservation
        lab = self._lab()
        # seed a world observation so the planner has something to select on
        obs = WorldObservation(
            ts_ns=1, player=PlayerState(health=0.9, health_units="fraction"),
            target=TargetState(type="quest_marker", direction=0.3,
                               distance=0.4, confidence=0.9),
            enemies=[EnemyState(direction=0.1, distance=0.3, attacking=False,
                                threat=0.2, confidence=0.9)],
            ui=UIState())
        lab.bus.state("world.observation").write(obs.to_dict())
        lab.planner.step()
        snap = lab.bus.state("helper.goal").read()
        assert snap is not None
        g = snap.payload["goal"]
        assert g["kind"] in ("combat", "travel", "observe")
        assert g["kind"] == "combat"      # enemy present -> combat goal
        # the helper emits relevance/bias values only — no key bindings
        assert "goal_relevance" in snap.payload
        assert "bindings" not in g


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

class TestModes:
    def test_mode_b_requires_frozen_config(self):
        validate_mode_config(ModeConfig(AgentMode.BIOLOGICAL_PLASTICITY_FLY,
                                        plasticity_frozen_config="gate4_v2_final"))
        with pytest.raises(ValueError):
            validate_mode_config(ModeConfig(AgentMode.BIOLOGICAL_PLASTICITY_FLY))

    def test_mode_a_rejects_plasticity(self):
        with pytest.raises(ValueError):
            validate_mode_config(ModeConfig(AgentMode.PURE_FIXED_FLY,
                                            plasticity_frozen_config="x"))

    def test_default_is_hybrid(self):
        assert DEFAULT_MODE.mode is AgentMode.HYBRID_FLY

    def test_brain_worker_rejects_mode_b_live(self):
        from lab.brain.worker import BrainWorker
        from lab.modes import AgentMode
        with pytest.raises(RuntimeError, match="FROZEN"):
            BrainWorker(self._bus(), agent_mode=AgentMode.BIOLOGICAL_PLASTICITY_FLY)

    @staticmethod
    def _bus():
        from lab.bus import Bus
        return Bus()


# ---------------------------------------------------------------------------
# capture adapters
# ---------------------------------------------------------------------------

class TestCapture:
    def test_synthetic_capture_labels_and_publishes(self):
        from lab.bus import Bus
        from lab.capture.base import SyntheticCapture
        bus = Bus()
        cap = SyntheticCapture(bus, fps=30.0)
        cap.start()
        p = cap.tick(0.0)
        assert p["synthetic"] is True
        assert p["source"] == "synthetic"
        assert p["width"] == 640 and p["height"] == 360
        env = cap.frames.newest()
        assert env.payload["frame_id"] == 1
        cap.stop()

    def test_windows_capture_unavailable_off_windows(self):
        from lab.capture.base import (CaptureError, NullCapture,
                                      create_windows_capture, windows_capture_available)
        if windows_capture_available():
            pytest.skip("on Windows")
        with pytest.raises(CaptureError):
            create_windows_capture(Bus())
        # null capture never emits
        n = NullCapture(Bus())
        n.start(); n.stop()

    def test_windows_only_modules_not_imported_off_windows(self):
        import platform
        if platform.system() == "Windows":
            pytest.skip("on Windows")
        import lab.action
        assert "windows_input" not in sys.modules
        import lab.capture
        assert "windows_graphics_capture" not in sys.modules


# ---------------------------------------------------------------------------
# end-to-end pipeline + replay (mock runtime)
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_self_test_passes(self):
        from lab.app import self_test
        res = self_test()
        assert res["ok"] is True
        assert res["meta"]["mode"] == "PASSIVE"

    def test_replay_decision_chains_complete(self):
        import glob
        from lab.app import DigitalFlyLab
        from lab.replay import load_replay, replay_decision_chains
        lab = DigitalFlyLab(dashboard=False)
        lab.start()
        try:
            lab.drive_synthetic(seconds=1.5, fps=25.0)
        finally:
            lab.stop()
        events = load_replay(lab.session_dir / "replay.jsonl")
        topics = {e["topic"] for e in events}
        assert {"perception.fast.events", "brain.events",
                "action.inputs"} <= topics
        chains = replay_decision_chains(events)
        assert len(chains) >= 5
        for c in chains:
            assert c["intention"] in INTENTIONS
            assert "sensory_rates_hz" in c and "dn_rates_hz" in c

    def test_mock_runtime_labeled_everywhere(self):
        from lab.app import DigitalFlyLab
        from lab.replay import load_replay
        lab = DigitalFlyLab(dashboard=False)
        lab.start()
        try:
            lab.drive_synthetic(seconds=1.0, fps=20.0)
        finally:
            lab.stop()
        events = load_replay(lab.session_dir / "replay.jsonl")
        brain_events = [e for e in events if e["topic"] == "brain.events"]
        assert brain_events
        assert all(e["payload"]["runtime"] == "mock" for e in brain_events)

    def test_no_unbounded_frame_queue(self):
        from lab.app import DigitalFlyLab
        lab = DigitalFlyLab(dashboard=False)
        # every stream channel must be bounded (drop-oldest policy)
        for spec in lab.bus.specs():
            if spec["kind"] == "stream":
                assert spec["maxsize"] <= 256, spec

    def test_latency_budget_measured(self):
        from lab.app import self_test
        res = self_test()
        rep = res["report"]
        assert rep["capture"]["published"] > 0


# ---------------------------------------------------------------------------
# dashboard (snapshot reader, never owns the clock)
# ---------------------------------------------------------------------------

class TestDashboard:
    def test_text_renderer_three_columns(self, capsys):
        from lab.dashboard.dashboard import (DashboardWorker,
                                             TextDashboardRenderer)
        from lab.bus import Bus
        bus = Bus()
        dw = DashboardWorker(bus, renderer=TextDashboardRenderer())
        bus.state("brain.output").write({"runtime": "mock", "chunk_id": 1,
                                         "chunk_ms": 50,
                                         "intention": {"name": "STOP",
                                                       "confidence": 0.7},
                                         "dn_rates_hz": {"P9_left": 2.0}})
        bus.state("fly.channels").write(FlyChannels(threat_intensity=0.4).to_dict())
        bus.state("helper.goal").write({"goal": {"label": "observe world"},
                                        "goal_relevance": 0.3})
        bus.state("action.selected").write({"autonomy": False,
                                            "ability_id": "",
                                            "bindings": [], "backend": "safe_noop"})
        bus.state("world.observation").write({"enemies": [], "target": {},
                                              "player": {}, "ui": {}})
        out = dw.take_snapshot()
        rendered = dw.renderer.render(out, None)
        for col in ("FLY BRAIN", "HYBRID HELPER", "ACTION SYSTEM"):
            assert col in rendered
        assert "mock" in rendered

    def test_dashboard_never_blocks_control(self):
        # a failing renderer is contained: control continues (errors counted)
        from lab.dashboard.dashboard import DashboardWorker
        from lab.bus import Bus

        class ExplodingRenderer:
            name = "boom"
            def render(self, *a, **k):
                raise RuntimeError("chart too slow")
        bus = Bus()
        dw = DashboardWorker(bus, renderer=ExplodingRenderer())
        dw.step()          # a crashing renderer must not propagate — control continues
        dw.step()
        assert dw.stats["errors"] >= 2


# ---------------------------------------------------------------------------
# benchmark machinery (fast checks only)
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_hardware_inventory_nonempty(self):
        from lab.benchmark.windows_hardware import hardware_inventory
        inv = hardware_inventory()
        assert inv["platform"] in ("Linux", "Windows", "Darwin")
        assert inv["logical_cores"] and inv["logical_cores"] >= 1

    def test_summarize(self):
        from lab.benchmark.windows_hardware import summarize_ms
        s = summarize_ms([1.0, 2.0, 3.0, 10.0])
        assert s["n"] == 4 and s["p50_ms"] == 2.5 and s["max_ms"] == 10.0

    def test_choose_live_settings_conservative(self):
        from lab.benchmark.windows_hardware import choose_live_settings
        bench = {"stage_latency": {
            "brain_chunk_wall": {"p95_ms": 300.0},
            "full_sensory_to_action": {"p95_ms": 400.0}}}
        ls = choose_live_settings(bench, None)
        assert ls["brain_chunk_ms"] == 100          # 300ms wall -> 100ms chunks
        assert ls["brain_hz"] <= 10.0
        assert ls["s2a_target_met"] is False        # honest
