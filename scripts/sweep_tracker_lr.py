"""Sweep MOG2 learning rates for pause survival."""
import numpy as np
from flyagent.config import load_all_config
from flyagent.fly_tracker.source import SyntheticSource
from flyagent.fly_tracker.tracker import FlyTracker

CFG = load_all_config()
for lr in [None, 0.02, 0.01, 0.005, 0.002, 0.001]:
    fly_cfg = dict(CFG["experiment"]["synthetic_fly"])
    fly_cfg["cue_bias"] = 0.0
    rng = np.random.default_rng(42)
    src = SyntheticSource(CFG["tracking"]["source"], fly_cfg, rng, warmup_frames=20)
    ax, ay, aw, ah = src.arena_rect
    tr_cfg = dict(CFG["tracking"]["tracker"])
    if lr is not None:
        tr_cfg["learning_rate"] = lr
    tracker = FlyTracker(tr_cfg, roi=(ax, ay, aw, ah))
    lost = 0
    n = 300
    for i in range(n):
        _, frame = src.read()
        tp = tracker.update(frame)
        if not tp.found and i >= 25:
            lost += 1
    print(f"lr={lr!s:<6} lost {lost}/{n - 25} frames "
          f"({100 * lost / (n - 25):.1f}%)")
    src.close()
