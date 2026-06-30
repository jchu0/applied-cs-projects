"""AREngine - Autoregressive Inference Engine for LLMs."""

from .core import (
    GenerationConfig,
    GenerationOutput,
    KVCache,
    PagedKVCache,
    AutoregressiveModel,
)
from .sampling import (
    LogitsProcessor,
    TemperatureProcessor,
    TopKProcessor,
    TopPProcessor,
    sample_token,
    greedy_search,
    BeamSearchScorer,
    TypicalSampling,
)
from .batching import (
    Request,
    Batch,
    ContinuousBatcher,
    PrefillDecodeScheduler,
)
from .speculative import (
    SpeculativeDecoder,
    MedusaDecoder,
    LookaheadDecoder,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "GenerationConfig",
    "GenerationOutput",
    "KVCache",
    "PagedKVCache",
    "AutoregressiveModel",
    # Sampling
    "LogitsProcessor",
    "TemperatureProcessor",
    "TopKProcessor",
    "TopPProcessor",
    "sample_token",
    "greedy_search",
    "BeamSearchScorer",
    "TypicalSampling",
    # Batching
    "Request",
    "Batch",
    "ContinuousBatcher",
    "PrefillDecodeScheduler",
    # Speculative
    "SpeculativeDecoder",
    "MedusaDecoder",
    "LookaheadDecoder",
]
