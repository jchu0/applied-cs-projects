"""Sampling strategies for text generation."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import heapq

logger = logging.getLogger(__name__)


class LogitsProcessor:
    """Base class for logits processors."""

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        return logits


class TemperatureProcessor(LogitsProcessor):
    """Apply temperature scaling."""

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return logits


class TopKProcessor(LogitsProcessor):
    """Top-k filtering."""

    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        if self.top_k <= 0:
            return logits

        # Keep only top_k values
        top_k = min(self.top_k, logits.shape[-1])
        indices = np.argpartition(logits, -top_k, axis=-1)[..., -top_k:]

        mask = np.ones_like(logits) * -np.inf
        for i in range(logits.shape[0]):
            mask[i, indices[i]] = 0

        return logits + mask


class TopPProcessor(LogitsProcessor):
    """Top-p (nucleus) filtering."""

    def __init__(self, top_p: float):
        self.top_p = top_p

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        if self.top_p >= 1.0:
            return logits

        # Sort and compute cumulative probabilities
        sorted_indices = np.argsort(logits, axis=-1)[..., ::-1]
        sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)

        # Softmax
        sorted_probs = np.exp(sorted_logits - sorted_logits.max(axis=-1, keepdims=True))
        sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)

        cumsum = np.cumsum(sorted_probs, axis=-1)

        # Find cutoff
        cutoff_mask = cumsum > self.top_p
        cutoff_mask[..., 1:] = cutoff_mask[..., :-1]
        cutoff_mask[..., 0] = False

        # Apply mask
        sorted_logits[cutoff_mask] = -np.inf

        # Unsort
        original_indices = np.argsort(sorted_indices, axis=-1)
        return np.take_along_axis(sorted_logits, original_indices, axis=-1)


class RepetitionPenaltyProcessor(LogitsProcessor):
    """Penalize repeated tokens."""

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        if self.penalty == 1.0:
            return logits

        for i in range(logits.shape[0]):
            for token in input_ids[i]:
                if logits[i, token] > 0:
                    logits[i, token] /= self.penalty
                else:
                    logits[i, token] *= self.penalty

        return logits


class MinLengthProcessor(LogitsProcessor):
    """Prevent EOS before minimum length."""

    def __init__(self, min_length: int, eos_token_id: int):
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        current_length = input_ids.shape[1]
        if current_length < self.min_length:
            logits[:, self.eos_token_id] = -np.inf
        return logits


class LogitsProcessorList:
    """Chain of logits processors."""

    def __init__(self, processors: List[LogitsProcessor]):
        self.processors = processors

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        for processor in self.processors:
            logits = processor(input_ids, logits)
        return logits


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0
) -> np.ndarray:
    """
    Sample next token from logits.

    Args:
        logits: (batch, vocab) logits
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Top-p filtering

    Returns:
        Sampled tokens (batch,)
    """
    # Temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices = np.argpartition(logits, -top_k, axis=-1)[..., -top_k:]
        mask = np.ones_like(logits) * -np.inf
        for i in range(logits.shape[0]):
            mask[i, indices[i]] = 0
        logits = logits + mask

    # Softmax
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # Top-p
    if top_p < 1.0:
        sorted_indices = np.argsort(probs, axis=-1)[..., ::-1]
        sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
        cumsum = np.cumsum(sorted_probs, axis=-1)

        mask = cumsum > top_p
        mask[..., 1:] = mask[..., :-1]
        mask[..., 0] = False
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)

        # Sample from sorted
        tokens = []
        for i in range(probs.shape[0]):
            token_idx = np.random.choice(len(sorted_probs[i]), p=sorted_probs[i])
            tokens.append(sorted_indices[i, token_idx])
        return np.array(tokens)

    # Sample
    return np.array([
        np.random.choice(probs.shape[1], p=probs[i])
        for i in range(probs.shape[0])
    ])


def greedy_search(logits: np.ndarray) -> np.ndarray:
    """Greedy decoding - select highest probability token."""
    return np.argmax(logits, axis=-1)


@dataclass
class BeamHypothesis:
    """Single beam hypothesis."""
    tokens: List[int]
    score: float
    is_done: bool = False


class BeamSearchScorer:
    """Beam search scorer for multiple beams."""

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        length_penalty: float = 1.0,
        early_stopping: bool = False
    ):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping

        # Best hypotheses per batch
        self.hypotheses = [[] for _ in range(batch_size)]

    def process(
        self,
        input_ids: np.ndarray,
        next_scores: np.ndarray,
        next_tokens: np.ndarray,
        next_indices: np.ndarray,
        eos_token_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process beam search step.

        Returns:
            Updated (input_ids, beam_scores)
        """
        batch_size = input_ids.shape[0] // self.num_beams
        next_beam_tokens = []
        next_beam_indices = []
        next_beam_scores = []

        for batch_idx in range(batch_size):
            beam_start = batch_idx * self.num_beams
            beam_end = beam_start + self.num_beams

            for beam_rank, (token, score, beam_idx) in enumerate(
                zip(
                    next_tokens[beam_start:beam_end].flatten(),
                    next_scores[beam_start:beam_end].flatten(),
                    next_indices[beam_start:beam_end].flatten()
                )
            ):
                if token == eos_token_id:
                    # Finished beam
                    hyp = BeamHypothesis(
                        tokens=input_ids[beam_start + beam_idx].tolist() + [token],
                        score=score / (len(input_ids[0]) ** self.length_penalty)
                    )
                    self.hypotheses[batch_idx].append(hyp)
                else:
                    next_beam_tokens.append(token)
                    next_beam_indices.append(beam_start + beam_idx)
                    next_beam_scores.append(score)

                if len(next_beam_tokens) == self.num_beams * batch_size:
                    break

        # Update input_ids
        new_input_ids = input_ids[next_beam_indices]
        new_input_ids = np.concatenate([
            new_input_ids,
            np.array(next_beam_tokens).reshape(-1, 1)
        ], axis=1)

        return new_input_ids, np.array(next_beam_scores)

    def is_done(self) -> bool:
        """Check if all batches are done."""
        if self.early_stopping:
            return all(len(h) >= self.num_beams for h in self.hypotheses)
        return False

    def finalize(self) -> List[List[int]]:
        """Get final best sequences."""
        results = []
        for batch_hyps in self.hypotheses:
            if batch_hyps:
                best = max(batch_hyps, key=lambda h: h.score)
                results.append(best.tokens)
            else:
                results.append([])
        return results


class ContrastiveSearch:
    """Contrastive search for diverse generation."""

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha

    def select_token(
        self,
        logits: np.ndarray,
        past_hidden: np.ndarray,
        current_hidden: np.ndarray,
        top_k: int = 4
    ) -> int:
        """
        Select token using contrastive objective.

        Balances probability and distinctiveness from context.
        """
        # Get top-k candidates
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()

        top_indices = np.argsort(probs)[-top_k:]
        top_probs = probs[top_indices]

        # Compute degeneration penalty
        penalties = []
        for idx in top_indices:
            # Cosine similarity to past tokens
            hidden = current_hidden  # Would be per-token hidden
            sim = np.dot(past_hidden, hidden) / (
                np.linalg.norm(past_hidden, axis=-1) * np.linalg.norm(hidden) + 1e-8
            )
            penalty = sim.max() if len(sim) > 0 else 0
            penalties.append(penalty)

        # Contrastive score
        scores = (1 - self.alpha) * top_probs - self.alpha * np.array(penalties)

        return top_indices[np.argmax(scores)]


class TypicalSampling:
    """Typical sampling based on information content."""

    def __init__(self, mass: float = 0.9):
        self.mass = mass

    def sample(self, logits: np.ndarray) -> np.ndarray:
        """Sample using typical decoding."""
        # Compute entropy
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        log_probs = np.log(probs + 1e-10)
        entropy = -np.sum(probs * log_probs, axis=-1, keepdims=True)

        # Information content
        info = -log_probs

        # Distance from expected information
        shifted = np.abs(info - entropy)

        # Sort by typicality
        sorted_indices = np.argsort(shifted, axis=-1)
        sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)

        # Cumulative mass
        cumsum = np.cumsum(sorted_probs, axis=-1)
        mask = cumsum > self.mass
        mask[..., 1:] = mask[..., :-1]
        mask[..., 0] = False
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)

        # Sample
        tokens = []
        for i in range(probs.shape[0]):
            idx = np.random.choice(len(sorted_probs[i]), p=sorted_probs[i])
            tokens.append(sorted_indices[i, idx])

        return np.array(tokens)


class EtaSampling:
    """Eta sampling - entropy-based adaptive sampling."""

    def __init__(self, eta: float = 0.0003, epsilon: float = 0.0003):
        self.eta = eta
        self.epsilon = epsilon

    def sample(self, logits: np.ndarray) -> np.ndarray:
        """Sample with eta threshold."""
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        # Entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1, keepdims=True)

        # Threshold
        threshold = min(self.eta, np.sqrt(self.eta) * np.exp(-entropy))

        # Filter
        mask = probs < threshold
        probs[mask] = 0
        probs = probs / probs.sum(axis=-1, keepdims=True)

        return np.array([
            np.random.choice(probs.shape[1], p=probs[i])
            for i in range(probs.shape[0])
        ])
