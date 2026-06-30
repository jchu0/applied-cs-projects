"""Sampling strategies for text generation."""

from .samplers import (
    LogitsProcessor,
    TemperatureProcessor,
    TopKProcessor,
    TopPProcessor,
    RepetitionPenaltyProcessor,
    LogitsProcessorList,
    sample_token,
    greedy_search,
    BeamSearchScorer,
    ContrastiveSearch,
    TypicalSampling,
    EtaSampling,
)

__all__ = [
    "LogitsProcessor",
    "TemperatureProcessor",
    "TopKProcessor",
    "TopPProcessor",
    "RepetitionPenaltyProcessor",
    "LogitsProcessorList",
    "sample_token",
    "greedy_search",
    "BeamSearchScorer",
    "ContrastiveSearch",
    "TypicalSampling",
    "EtaSampling",
]
