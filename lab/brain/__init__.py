"""Brain layer: canonical runtime wrapper, intention decoder, live worker.
TIER: BIOLOGICAL (canonical) + explicitly-labeled mock for pipeline tests."""
from .runtime import (CanonicalBrianRuntime, MockBrainRuntime,  # noqa: F401
                      create_runtime, BrainChunkRecord)
from .intent import IntentionDecoder, CONFIG_ID as INTENT_CONFIG_ID  # noqa: F401
from .worker import BrainWorker  # noqa: F401
