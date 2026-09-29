"""Perception tiers: fast vision (immediate), heavy vision/OCR (slow
semantic), fly-channel encoder. All ENGINEERED."""
from .fly_channels import encode, to_sensory_rates  # noqa: F401
from .fast_vision import FastVisionWorker, HeuristicFastVision  # noqa: F401
from .heavy_vision import HeavyVisionWorker, MockOCRBackend  # noqa: F401
from .channel_encoder import FlyChannelEncoder  # noqa: F401
