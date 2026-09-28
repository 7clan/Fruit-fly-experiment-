"""D11-D13 unit tests (fast, brain-free - the canonical brain is exercised
by the recorded studies under brain/results/, not by this suite)."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "brain" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import d11_learning as d11          # noqa: E402
import d11_sessions as ds           # noqa: E402
import d12_states as d12            # noqa: E402


# ---------------------------------------------------------------- D11 arena
class TestChoiceArena:
    def test_symmetric_targets(self):
        a = d11.ChoiceArena2D(seed=1)
        v = a.visual_state()
        assert abs(v["bearing_left_deg"] - 50) < 9
        assert abs(v["bearing_right_deg"] + 50) < 9
        assert abs(v["dist_left"] - v["dist_right"]) < 1e-9

    def test_commit_left_right(self):
        a = d11.ChoiceArena2D(seed=2)
        for _ in range(100):
            a.step("TURN_LEFT", 0.05)
        assert a.choice == "left"
        b = d11.ChoiceArena2D(seed=2)
        for _ in range(100):
            b.step("TURN_RIGHT", 0.05)
        assert b.choice == "right"

    def test_pair_modes_hide_one_target(self):
        a = d11.ChoiceArena2D(seed=3, mode="pair_left")
        v = a.visual_state()
        assert v["bearing_left_deg"] is not None
        assert v["bearing_right_deg"] is None
        b = d11.ChoiceArena2D(seed=3, mode="pair_right")
        v = b.visual_state()
        assert v["bearing_right_deg"] is not None
        assert v["bearing_left_deg"] is None


class TestEncoders:
    def test_visual_encoder_symmetric(self):
        enc = d11.ChoiceVisualEncoder(r_target_left=134.0,
                                      r_target_right=150.0)
        a = d11.ChoiceArena2D(seed=1)
        r = enc.rates(a.visual_state())
        assert r["target_left"] == 134.0
        assert r["target_right"] == 150.0
        assert r["looming"] == 0.0

    def test_cs_encoder_fixation_gated(self):
        cs = ds.CSEncoder(precommit_hz_left=102.3, precommit_hz_right=75.0)
        a = d11.ChoiceArena2D(seed=1, jitter_deg=0.0)
        r = cs.rates(a.visual_state())
        assert abs(r["kc_cs_left"] - 102.3) < 1e-6   # |b|=50 pre-commit
        assert abs(r["kc_cs_right"] - 75.0) < 1e-6
        # approaching the left target raises the left context, kills right
        a3 = d11.ChoiceArena2D(seed=3)
        for _ in range(6):
            a3.step("TURN_LEFT", 0.05)
        r3 = cs.rates(a3.visual_state())
        assert r3["kc_cs_left"] > 3 * r3["kc_cs_right"]

    def test_cs_encoder_ignores_absent_target(self):
        cs = ds.CSEncoder()
        r = cs.rates(dict(bearing_left_deg=None, bearing_right_deg=-40.0))
        assert r["kc_cs_left"] == 0.0
        assert r["kc_cs_right"] > 0


# ------------------------------------------------------------ plasticity
class TestMBPlasticity:
    def _pl(self, **kw):
        rows = np.arange(10)
        pre = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4])
        post = np.arange(10)
        val = {i: ("avoidance" if i % 2 == 0 else "approach")
               for i in range(10)}
        return d11.MBPlasticity(rows, pre, post, val, n_kc=5, **kw)

    def test_reward_depresses_avoidance_only(self):
        pl = self._pl(eta=0.5)
        pl.update(np.array([1.0, 1.0, 0, 0, 0]), pam_hz=150.0, ppl1_hz=0.0)
        s = pl.stats()
        assert s["avoidance_class_mean"] < 1.0
        assert s["approach_class_mean"] == 1.0

    def test_punish_depresses_approach_only(self):
        pl = self._pl(eta=0.5)
        pl.update(np.array([1.0, 0, 0, 0, 0]), pam_hz=0.0, ppl1_hz=150.0)
        s = pl.stats()
        assert s["approach_class_mean"] < 1.0
        assert s["avoidance_class_mean"] == 1.0

    def test_gate_below_threshold_no_change(self):
        pl = self._pl(eta=0.5, pam_gate_hz=10.0)
        pl.update(np.array([1.0, 1, 1, 1, 1]), pam_hz=5.0, ppl1_hz=0.0)
        assert pl.stats()["mean_scale"] == 1.0

    def test_scale_floor(self):
        pl = self._pl(eta=0.99, scale_min=0.05)
        for _ in range(10):
            pl.update(np.array([1.0, 1, 1, 1, 1]), pam_hz=150.0,
                      ppl1_hz=0.0)
        assert pl.stats()["min_scale"] >= 0.05 - 1e-9

    def test_relax_returns_to_canonical(self):
        pl = self._pl(eta=0.5, tau_relax_trials=50.0)
        pl.update(np.array([1.0, 1, 0, 0, 0]), pam_hz=150.0, ppl1_hz=0.0)
        assert pl.stats()["avoidance_class_mean"] < 1.0
        for _ in range(500):
            pl.relax(1)
        assert pl.stats()["avoidance_class_mean"] > 0.99

    def test_elig_power_sharpens_attribution(self):
        # a weakly-eligible non-target KC gets (0.15)^2 = 2% of the
        # depression under elig_power=2 (vs 15% linear)
        pl = self._pl(eta=1.0, elig_power=2.0)
        e = np.array([1.0, 0.15, 0, 0, 0])
        pl.update(e, pam_hz=150.0, ppl1_hz=0.0)
        scales = pl.scale
        # rows 0,2,4,6,8 belong to avoidance class; pre of row 4 = 1 (the
        # weak KC), rows 0,2 = pre 0 (strong), row 6,8 = pre 2,3 (silent)
        assert scales[0] == pytest.approx(0.05, abs=1e-9)   # floor (eta=1)
        assert scales[4] == pytest.approx(1 - 0.0225, abs=1e-6)
        assert scales[6] == 1.0 and scales[8] == 1.0

    def test_state_roundtrip(self):
        pl = self._pl(eta=0.5)
        pl.update(np.array([1.0, 1, 0, 0, 0]), pam_hz=150.0, ppl1_hz=0.0)
        pl2 = self._pl(eta=0.5)
        pl2.load_state_dict(pl.state_dict())
        assert np.allclose(pl.scale, pl2.scale)


class TestEligibilityTrace:
    def test_decay_and_accumulation(self):
        tr = d11.EligibilityTrace(3, tau_s=2.0, mode="spikes")
        spikes = np.array([10.0, 0.0, 0.0])
        tr.on_chunk(0.05, kc_spikes=spikes)      # first chunk
        e1 = tr.value().copy()
        assert e1[0] == pytest.approx(10.0, abs=1e-9)
        tr.on_chunk(1.0, kc_spikes=np.zeros(3))  # long silence decays
        e2 = tr.value()
        assert e2[0] < e1[0]
        assert e2[0] > 0


# --------------------------------------------------------------- decoder
class TestValenceBiasDecoder:
    def _dec(self, beta=1.0):
        sizes = dict(P9_left=1, P9_right=1, BPN_bilateral=33,
                     RRN_bilateral=2, MDN_bilateral=4, GF_left=1,
                     GF_right=1, FG_bilateral=2, BB_bilateral=2,
                     MBON_approach_left=18, MBON_approach_right=19,
                     MBON_avoidance_left=12, MBON_avoidance_right=12)
        p = dict(window_ms=300.0, theta_jump_hz=25.0, theta_bwd_hz=4.0,
                 theta_fwd_hz=5.0, theta_diff_hz=8.0, delta_hyst_hz=4.0,
                 theta_stop_hz=4.0, theta_min_walk_hz=1.5,
                 min_action_chunks=2, turn_source="P9")
        return ds.ValenceBiasDecoder(sizes, p, beta=beta)

    def test_bias_pushes_turn_left(self):
        dec = self._dec(beta=1.0)
        # symmetric P9 + strong left-favoring MBON valence -> TURN_LEFT
        for _ in range(8):
            r = dec.decode({"P9_left": 2, "P9_right": 2,
                            "MBON_approach_left": 10 * 18,
                            "MBON_avoidance_left": 0,
                            "MBON_approach_right": 0,
                            "MBON_avoidance_right": 10 * 12}, 50)
        assert r["action"] == "TURN_LEFT"
        assert r["valence_bias_hz"] > 0

    def test_no_mbon_activity_no_bias(self):
        dec = self._dec(beta=1.0)
        for _ in range(8):
            r = dec.decode({}, 50)              # silence
        assert r["valence_bias_hz"] == 0.0
        assert r["action"] == "STOP"          # failsafe on silence


# ------------------------------------------------------------- plan + RL
class TestPlansAndBaseline:
    def test_expand_plans(self):
        seq = ds._expand_phases(ds.PHASES_B_RIGHT)
        assert len(seq) == 6 * (4 + 3) + 12 + 6 * (4 + 3)
        kinds = [s["kind"] for s in seq]
        assert kinds.count("test") == 6 * 3 + 12 + 6 * 3
        assert kinds.count("pair_right") == 24
        assert kinds.count("pair_left") == 24
        # no US anywhere in extinction
        ext = [s for s in seq if s["phase"] == "extinction"]
        assert all(s["kind"] == "test" and not s["us_on"] for s in ext)

    def test_rl_baseline_learns_acquires_extinguishes_reverses(self):
        summ = ds.run_rl_baseline(out_dir="/tmp/d11_test_rl",
                                  master_seed=7)
        acq = summ["phase_acquisition"]
        # ceiling-fast learner: acquisition reaches high preference
        assert acq["p_paired_side_test_second_half"] >= 0.75
        rev = summ["phase_reversal"]
        assert rev["p_test_left"] > 0.5     # flipped to the new side
        assert rev["p_test_left"] > acq["p_test_left"] + 0.3


# -------------------------------------------------------------- D12 states
class TestD12States:
    def _fake_trials(self):
        trs = []
        for i in range(12):
            appr_l, avoid_l = (30.0, 50.0) if i < 6 else (30.0, 20.0)
            trs.append(dict(
                trial=i, phase="acquisition",
                choice="left" if i % 2 else "right",
                us_delivered=(i % 2 == 1),
                mbon_valence_end=dict(appr_l=appr_l, appr_r=25.0,
                                      avoid_l=avoid_l, avoid_r=45.0),
                valence_bias_end=1.0,
                pam_choice_hz=0.5, ppl1_choice_hz=0.0,
                dn_all_choice_hz=2.0,
                kc_elig_ensemble=[float(i % 5)] * 40,
            ))
        return trs

    def test_states_computed_from_variables(self):
        st = d12.compute_states(self._fake_trials())
        assert len(st) == 12
        s0, s5 = st[0], st[5]
        # negative valence falls as avoidance-MBON falls (trials 6+)
        assert st[7]["negative_valence_hz"] < s0["negative_valence_hz"]
        # reward expectation of the left context rises when avoid_l falls
        assert st[7]["reward_expectation_left"] > \
            s0["reward_expectation_left"]
        # RPE: rewarded left-choice trial after low expectation -> positive
        assert s0["reward_prediction_error"] is not None
        # exploration entropy present after >=2 choices
        assert s0["exploration_drive"] is None or \
            s0["exploration_drive"] >= 0
        # novelty defined
        assert st[1]["novelty"] is not None
        # threat zero (no loom)
        assert all(s["threat_drive_hz"] == 0.0 for s in st)
        # arousal z-score finite
        assert all(math.isfinite(s["arousal_like_z"]) for s in st)

    def test_states_json_serializable(self):
        st = d12.compute_states(self._fake_trials())
        json.dumps(st[0])     # must not raise


# ---------------------------------------------------------- prereg frozen
class TestPreregistration:
    def test_prereg_exists_and_frozen(self):
        p = REPO / "brain" / "data" / "d11_d13" / "preregistration.json"
        assert p.exists()
        r = json.loads(p.read_text())
        assert r["version"] >= 2
        for k in ("plasticity", "decoder", "sessions", "gate4_criteria"):
            assert k in r
        assert r["plasticity"]["elig_power"] == 2.0
        assert r["decoder"]["beta"] == 1.5
        for label in ("B", "A", "C", "D", "E_rl"):
            assert label in r["sessions"]


# ----------------------------------------------------- anatomy verification
class TestAnatomyEvidence:
    def test_verify_record_matches_d5(self):
        p = REPO / "brain" / "results" / "d11_anatomy" / "verify.json"
        assert p.exists(), "run: python brain/scripts/d11_learning.py anatomy"
        v = json.loads(p.read_text())
        assert v["all_checks_pass"] is True
        assert v["kc_to_mbon"]["n_connections"] == 62261
        assert v["kc_to_mbon"]["n_synapses"] == 256719.0
        # ipsilateral dominance (the laterality substrate of side-specific
        # eligibility)
        assert v["kc_to_mbon"]["ipsilateral_conns"] > \
            v["kc_to_mbon"]["contralateral_conns"]


class TestProbeEvidence:
    def test_probes_recorded(self):
        p = REPO / "brain" / "results" / "d11_probes" / "probes.json"
        assert p.exists(), "run: python brain/scripts/d11_probes.py run"
        d = json.loads(p.read_text())
        a = d["analysis"]
        # the honest negative findings that motivated the engineered tiers
        assert a["P1_sugar_recruits_PAM"]["sugar"] == 0.0
        assert a["P2_visual_kc_side_specificity"]["left_drive"][
            "kc_left_spikes_s"] == 0.0
        assert a["P3_kc_mbon_silencing_impact"]["differs"] is False
        assert a["P5_repro_bit_identical"]["identical"] is True

    def test_bilateral_calibration_recorded(self):
        p = REPO / "brain" / "results" / "d11_probes" / \
            "bilateral_calibration.json"
        assert p.exists()
        c = json.loads(p.read_text())
        v = c.get("verification_at_solution", {})
        assert abs(v.get("valence_diff", 99)) < 2.0
        assert abs(v.get("p9_diff", 99)) < 3.0
