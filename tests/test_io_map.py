"""D4-D6 neural I/O map tests (light: no Brian2 network is built).

Validates the machine-readable map, its v783 ground truth, and the Gate-2
evidence trail. Skipped when the map / evidence files are absent.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brain" / "scripts"))

IO_MAP = ROOT / "brain" / "data" / "io_map" / "neural_io_map.json"
IDS = ROOT / "brain" / "data" / "io_map" / "ids"
GATE2 = ROOT / "brain" / "results" / "gate2_io" / "verdict.json"
PERF = ROOT / "brain" / "results" / "perf_study" / "perf_study.json"
HAVE_MAP = IO_MAP.is_file()


@unittest.skipUnless(HAVE_MAP, "io map not built (io_map_build.py)")
class TestNeuralIoMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = json.loads(IO_MAP.read_text())

    def test_sections_present(self):
        for key in ("sensory_populations", "learning_reward_populations",
                    "descending_motor_populations", "pathway_checks",
                    "proposed_input_channels", "proposed_output_actions"):
            self.assertIn(key, self.map)
            self.assertTrue(self.map[key], key)

    def test_all_populations_have_ids_and_stats(self):
        n = 0
        for section in ("sensory_populations", "learning_reward_populations",
                        "descending_motor_populations"):
            for p in self.map[section]:
                n += 1
                self.assertTrue(p["present_in_v783"], p["pop_id"])
                self.assertGreater(p["v783"]["n"], 0, p["pop_id"])
                f = IDS / f"{p['pop_id']}.json"
                self.assertTrue(f.is_file(), p["pop_id"])
                ids = json.loads(f.read_text())
                self.assertEqual(len(ids), p["v783"]["n"], p["pop_id"])
        self.assertGreaterEqual(n, 30)

    def test_expected_key_populations(self):
        by_id = {}
        for section in ("sensory_populations", "learning_reward_populations",
                        "descending_motor_populations"):
            for p in self.map[section]:
                by_id[p["pop_id"]] = p
        # D4
        self.assertEqual(by_id["LMC_L2"]["v783"]["n"], 1728)
        self.assertEqual(by_id["T4"]["v783"]["n"], 6243)
        self.assertEqual(by_id["LPLC2"]["v783"]["n"], 210)
        self.assertEqual(by_id["LPTC_HS"]["v783"]["n"], 6)
        # D5
        self.assertEqual(by_id["KC"]["v783"]["n"], 5177)
        self.assertEqual(by_id["MBON"]["v783"]["n"], 96)
        self.assertEqual(by_id["PAM"]["v783"]["n"], 307)
        self.assertEqual(by_id["PPL1"]["v783"]["n"], 16)
        # D6
        self.assertEqual(by_id["DN_ALL"]["v783"]["n"], 1305)
        self.assertEqual(by_id["MDN"]["v783"]["n"], 4)
        self.assertEqual(by_id["DNp09_P9"]["v783"]["n"], 2)
        self.assertEqual(by_id["RRN"]["v783"]["n"], 2)
        self.assertEqual(by_id["FG_CB0890"]["v783"]["n"], 2)
        self.assertEqual(by_id["BB_DNg60"]["v783"]["n"], 2)

    def test_key_pathway_checks(self):
        pc = self.map["pathway_checks"]
        self.assertGreaterEqual(
            pc["loom(LPLC2+LC4) -> giant fiber (DNp01)"]["n_connections"], 200)
        self.assertGreaterEqual(
            pc["LC9 -> P9/DNp09 (visual -> directed walk)"]["n_connections"], 50)
        self.assertGreaterEqual(
            pc["Kenyon cells -> MBONs (learning synapse site)"]["n_connections"], 50000)
        self.assertEqual(
            pc["LC16 -> MDN (object -> backward)"]["n_connections"], 0)  # mediated, not direct

    def test_input_channels_cover_required_game_channels(self):
        got = {c["channel_id"] for c in self.map["proposed_input_channels"]}
        for want in ("vis_target_left", "vis_target_right", "looming_threat",
                     "optic_flow", "motion_directional", "brightness_contrast",
                     "object_approach", "reward_pulse", "punish_pulse"):
            self.assertIn(want, got)

    def test_output_action_vocabulary(self):
        acts = {a["action_id"]: a for a in self.map["proposed_output_actions"]}
        for want in ("FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP", "STARTLE_JUMP"):
            self.assertIn(want, acts)
            self.assertTrue(acts[want]["decoder_rule"])
            self.assertTrue(acts[want]["known_behavior"])

    def test_engineered_labels_present(self):
        """Reward/punish channels must be labelled as engineered (not biological)."""
        for c in self.map["proposed_input_channels"]:
            if c["channel_id"] in ("reward_pulse", "punish_pulse"):
                self.assertIn("ENGINEERED", json.dumps(c))

    def test_mdn_ids_match_author_labels(self):
        """The 4 MDN ids in the map must be the Bidaye-lab FlyWire-labeled ones."""
        mdn = set(json.loads((IDS / "MDN.json").read_text()))
        labeled = {720575940616026939, 720575940610236514,
                   720575940631082808, 720575940640331472}
        self.assertEqual(mdn, labeled)


@unittest.skipUnless(GATE2.is_file(), "gate2 evidence not present")
class TestGate2Evidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = json.loads(GATE2.read_text())

    def test_overall_pass(self):
        self.assertTrue(self.v["overall_pass"])
        for crit, c in self.v["criteria"].items():
            if "ok" in c:
                self.assertTrue(c["ok"], crit)

    def test_loom_reaches_giant_fiber(self):
        motor = self.v["criteria"]["G2.2_motor_readout"]["loom_condition_motor_spikes"]
        self.assertGreaterEqual(motor["DNp01_GF"], 100)
        self.assertGreaterEqual(motor["DN_ALL"], 1000)

    def test_causal_control_collapse(self):
        c = self.v["criteria"]["G2.5_causal_control"]
        self.assertEqual(c["ctrl_downstream_active"], 0)
        self.assertGreater(c["stim_downstream_active"], 100)

    def test_reproducibility_hashes(self):
        for d in self.v["criteria"]["G2.3_reproducibility"]["detail"].values():
            self.assertTrue(d["identical"])
            self.assertNotEqual(d["sha256_rep0"], "0" * 64)


@unittest.skipUnless(PERF.is_file(), "perf study not present")
class TestPerfStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = json.loads(PERF.read_text())

    def test_all_five_modes_measured(self):
        got = set(self.d["results"].keys())
        for want in ("full_reference", "subcircuit", "sparse_active",
                     "cython_cold", "cython_warm", "cpp_standalone"):
            self.assertIn(want, got)
        for m, r in self.d["results"].items():
            self.assertEqual(r.get("status"), "OK", m)

    def test_sparse_active_exact(self):
        f = self.d["fidelity"]["sparse_active"]
        self.assertEqual(f["jaccard_active"], 1.0)
        self.assertEqual(f["spike_ratio_vs_ref"], 1.0)

    def test_cpp_bit_identical_and_faster(self):
        f = self.d["fidelity"]["cpp_standalone"]
        self.assertEqual(f["jaccard_active"], 1.0)
        cpp_bin = self.d["results"]["cpp_standalone"]["binary_only_run_s"]
        ref_run = self.d["results"]["full_reference"]["wall_run_s"]
        self.assertLess(cpp_bin * 2, ref_run)  # >= 2x speedup


if __name__ == "__main__":
    unittest.main()
