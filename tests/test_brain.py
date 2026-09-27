"""Brain-harness tests (light: no Brian2 network is built).

Skipped automatically when the third-party model is not cloned
(brain/third_party is gitignored; `setup_model.py clone` provides it).
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brain" / "scripts"))

MODEL_DIR = ROOT / "brain" / "third_party" / "Drosophila_brain_model"
HAVE_MODEL = MODEL_DIR.is_dir()

import numpy as np  # noqa: E402


@unittest.skipUnless(HAVE_MODEL, "third-party model not cloned (setup_model.py clone)")
class TestBrainHarness(unittest.TestCase):
    def test_manifest_verify(self):
        """Pinned commit + all file hashes must verify (fails closed)."""
        import subprocess
        r = subprocess.run(
            [sys.executable, str(ROOT / "brain" / "setup" / "setup_model.py"),
             "verify"], capture_output=True, text=True)
        self.assertIn("VERDICT: PASS", r.stdout, r.stdout + r.stderr)

    def test_v783_counts(self):
        import brainlib as bl
        _, df = bl.load_maps("783")
        self.assertEqual(len(df), 138639)

    def test_resolve_ids_known_missing(self):
        """The tutorial sugar set: 21 ids, exactly 1 absent from v783."""
        import brainlib as bl
        exc, missing = bl.resolve_ids(bl.SUGAR_IDS, "783")
        self.assertEqual(len(exc), 20)
        self.assertEqual(missing, [720575940620900446])
        exc630, missing630 = bl.resolve_ids(bl.SUGAR_IDS, "630")
        self.assertEqual((len(exc630), missing630), (21, []))

    def test_mn9_present_both_versions(self):
        import brainlib as bl
        for v in ("630", "783"):
            flyid2i, _ = bl.load_maps(v)
            self.assertIn(bl.MN9_ID, flyid2i)

    def test_gate_verdicts_committed(self):
        """The Gate-1 evidence verdicts exist and PASS."""
        base = ROOT / "brain" / "results"
        for rel, key in (("repro_783/verdict.json", "gate_reproducible_activity"),
                         ("spikes_783/verdict.json", "gate_spike_propagation")):
            with self.subTest(rel=rel):
                d = json.loads((base / rel).read_text())
                self.assertTrue(d[key], f"{rel}: {key} not True")


class TestPureHelpers(unittest.TestCase):
    """Runs even without the third-party model."""

    def test_peak_and_trim_helpers_callable(self):
        import brainlib as bl
        self.assertIsInstance(bl.peak_rss_kb(), int)
        bl.trim_memory()  # must not raise

    def test_public_rec_drops_spikes(self):
        import brainlib as bl
        rec = {"seed": 1, "spikes": {3: [0.1]}, "n_spikes": 1}
        self.assertNotIn("spikes", bl.public_rec(rec))
        self.assertEqual(bl.public_rec(rec)["n_spikes"], 1)

    def test_save_json_int_default(self):
        import brainlib as bl
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            bl.save_json(p, {"a": np.int64(5), "b": "s"})
            self.assertEqual(json.loads(p.read_text())["a"], 5)


if __name__ == "__main__":
    unittest.main()
