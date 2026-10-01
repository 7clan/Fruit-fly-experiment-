import numpy as np
import pytest


def test_brian_count_only_spike_monitor_matches_full_monitor():
    brian2 = pytest.importorskip("brian2")
    from brian2 import Hz, PoissonGroup, SpikeMonitor, Network, ms

    brian2.seed(12345)
    g = PoissonGroup(32, rates=np.linspace(0, 120, 32) * Hz)
    full = SpikeMonitor(g, record=True, name="full_mon_test")
    counts_only = SpikeMonitor(g, record=False, name="count_mon_test")
    net = Network(g, full, counts_only)
    net.run(50 * ms)

    assert int(full.num_spikes) == int(counts_only.num_spikes)
    np.testing.assert_array_equal(
        np.asarray(full.count[:], dtype=np.int64),
        np.asarray(counts_only.count[:], dtype=np.int64),
    )
