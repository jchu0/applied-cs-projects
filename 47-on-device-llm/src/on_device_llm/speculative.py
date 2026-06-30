"""Speculative decoding for on-device LLM inference.

Speculative decoding uses a smaller/faster draft model to propose multiple
tokens that are then verified by the target model in a single forward pass,
improving inference throughput by 2-3x.

Reference: "Fast Inference from Transformers via Speculative Decoding"
https://arxiv.org/abs/2211.17192
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from on_device_llm.inference import (
    TransformerInference,
    Sampler,
    GenerationConfig,
)
from on_device_llm.operators import softmax


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    # Number of tokens to speculate per iteration
    num_speculative_tokens: int = 4

    # Temperature for draft model (lower = more confident)
    draft_temperature: float = 0.8

    # Temperature for target model
    target_temperature: float = 1.0

    # Top-k for both models
    top_k: int = 40

    # Top-p for both models
    top_p: float = 0.9

    # Maximum retries if all tokens rejected
    max_retries: int = 3


@dataclass
class SpeculativeStats:
    """Statistics for speculative decoding performance."""

    # Total tokens generated
    total_tokens: int = 0

    # Number of draft tokens accepted
    accepted_tokens: int = 0

    # Number of draft tokens rejected
    rejected_tokens: int = 0

    # Number of speculative iterations
    num_iterations: int = 0

    # Average acceptance rate
    @property
    def acceptance_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.accepted_tokens / (self.accepted_tokens + self.rejected_tokens)

    # Average tokens per iteration (speedup indicator)
    @property
    def tokens_per_iteration(self) -> float:
        if self.num_iterations == 0:
            return 0.0
        return self.total_tokens / self.num_iterations


class SpeculativeDecoder:
    """Speculative decoding engine.

    Uses a draft model to propose multiple tokens, then verifies them
    with the target model in a single forward pass. Accepted tokens
    are generated much faster than sequential decoding.
    """

    def __init__(
        self,
        target_model: TransformerInference,
        draft_model: TransformerInference,
        config: Optional[SpeculativeConfig] = None,
    ):
        """
        Initialize speculative decoder.

        Args:
            target_model: The main model for verification
            draft_model: Smaller/faster model for token proposals
            config: Speculative decoding configuration
        """
        self.target = target_model
        self.draft = draft_model
        self.config = config or SpeculativeConfig()
        self.stats = SpeculativeStats()

        # Samplers for each model
        self._draft_sampler = Sampler(
            temperature=self.config.draft_temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )
        self._target_sampler = Sampler(
            temperature=self.config.target_temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 100,
        eos_token_id: int = 2,
    ) -> Tuple[List[int], SpeculativeStats]:
        """
        Generate tokens using speculative decoding.

        Args:
            prompt_tokens: Input prompt token IDs
            max_new_tokens: Maximum number of tokens to generate
            eos_token_id: End of sequence token ID

        Returns:
            Tuple of (generated_tokens, statistics)
        """
        # Reset state
        self.target.reset()
        self.draft.reset()
        self._draft_sampler.reset()
        self._target_sampler.reset()
        self.stats = SpeculativeStats()

        # Process prompt through both models
        for i, token in enumerate(prompt_tokens[:-1]):
            self.target.forward(token, i)
            self.draft.forward(token, i)

        # Start generation
        position = len(prompt_tokens) - 1
        current_token = prompt_tokens[-1]
        output_tokens: List[int] = []

        while len(output_tokens) < max_new_tokens:
            # Speculate: draft model proposes tokens
            draft_tokens, draft_probs = self._speculate(
                current_token, position, self.config.num_speculative_tokens
            )

            # Verify: target model checks proposals
            accepted_tokens, next_token = self._verify(
                current_token, position, draft_tokens, draft_probs
            )

            # Update statistics
            self.stats.num_iterations += 1
            self.stats.accepted_tokens += len(accepted_tokens)
            self.stats.rejected_tokens += len(draft_tokens) - len(accepted_tokens)

            # Add accepted tokens to output
            for token in accepted_tokens:
                output_tokens.append(token)
                position += 1
                self.stats.total_tokens += 1

                if token == eos_token_id or len(output_tokens) >= max_new_tokens:
                    return output_tokens, self.stats

            # Add the resampled/bonus token
            if next_token is not None:
                output_tokens.append(next_token)
                position += 1
                self.stats.total_tokens += 1
                current_token = next_token

                if next_token == eos_token_id:
                    return output_tokens, self.stats
            elif accepted_tokens:
                current_token = accepted_tokens[-1]
            else:
                # All tokens rejected - generate one token with target
                logits = self.target.forward(current_token, position)
                current_token = self._target_sampler.sample(logits)
                output_tokens.append(current_token)
                position += 1
                self.stats.total_tokens += 1

                if current_token == eos_token_id:
                    return output_tokens, self.stats

        return output_tokens, self.stats

    def _speculate(
        self,
        start_token: int,
        start_position: int,
        num_tokens: int,
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Generate speculative tokens using draft model.

        Args:
            start_token: Starting token
            start_position: Starting position
            num_tokens: Number of tokens to speculate

        Returns:
            Tuple of (draft_tokens, draft_probabilities)
        """
        draft_tokens: List[int] = []
        draft_probs: List[np.ndarray] = []

        token = start_token
        position = start_position

        for _ in range(num_tokens):
            logits = self.draft.forward(token, position)

            # Get probability distribution
            if self.config.draft_temperature != 1.0:
                logits = logits / self.config.draft_temperature
            probs = softmax(logits)

            # Sample token
            token = self._draft_sampler.sample(logits)
            draft_tokens.append(token)
            draft_probs.append(probs)
            position += 1

        return draft_tokens, draft_probs

    def _verify(
        self,
        start_token: int,
        start_position: int,
        draft_tokens: List[int],
        draft_probs: List[np.ndarray],
    ) -> Tuple[List[int], Optional[int]]:
        """
        Verify draft tokens with target model.

        Uses rejection sampling to accept/reject draft tokens based on
        the ratio of target to draft probabilities.

        Args:
            start_token: Starting token
            start_position: Starting position
            draft_tokens: Tokens proposed by draft model
            draft_probs: Probability distributions from draft model

        Returns:
            Tuple of (accepted_tokens, next_token_from_target)
        """
        accepted_tokens: List[int] = []
        target_probs_list: List[np.ndarray] = []

        # Get target model probabilities for each position
        token = start_token
        position = start_position

        for draft_token in draft_tokens:
            logits = self.target.forward(token, position)

            if self.config.target_temperature != 1.0:
                logits = logits / self.config.target_temperature
            probs = softmax(logits)
            target_probs_list.append(probs)

            token = draft_token
            position += 1

        # Also get target probability at the last position for bonus token
        logits = self.target.forward(token, position)
        if self.config.target_temperature != 1.0:
            logits = logits / self.config.target_temperature
        final_target_probs = softmax(logits)

        # Verify each draft token using rejection sampling
        for i, (draft_token, draft_prob, target_prob) in enumerate(
            zip(draft_tokens, draft_probs, target_probs_list)
        ):
            # Acceptance probability: min(1, p_target / p_draft)
            p_draft = draft_prob[draft_token]
            p_target = target_prob[draft_token]

            if p_draft > 0:
                acceptance_prob = min(1.0, p_target / p_draft)
            else:
                acceptance_prob = 1.0 if p_target > 0 else 0.0

            # Accept or reject based on random sample
            if np.random.random() < acceptance_prob:
                accepted_tokens.append(draft_token)
            else:
                # Rejected - resample from adjusted distribution
                # q(x) = max(0, p_target(x) - p_draft(x))
                adjusted = np.maximum(0, target_prob - draft_prob)
                if adjusted.sum() > 0:
                    adjusted = adjusted / adjusted.sum()
                    next_token = int(np.random.choice(len(adjusted), p=adjusted))
                else:
                    # Fall back to target distribution
                    next_token = int(np.random.choice(len(target_prob), p=target_prob))
                return accepted_tokens, next_token

        # All tokens accepted - sample bonus token from target
        bonus_token = int(np.random.choice(len(final_target_probs), p=final_target_probs))
        return accepted_tokens, bonus_token

    def reset_stats(self) -> None:
        """Reset generation statistics."""
        self.stats = SpeculativeStats()


class SelfSpeculativeDecoder:
    """Self-speculative decoding using early exit.

    Instead of a separate draft model, uses early layers of the target
    model as the draft. This requires models with early exit capability
    or layer-wise output heads.
    """

    def __init__(
        self,
        model: TransformerInference,
        draft_layers: int = 4,
        config: Optional[SpeculativeConfig] = None,
    ):
        """
        Initialize self-speculative decoder.

        Args:
            model: The main model
            draft_layers: Number of layers to use for drafting
            config: Speculative decoding configuration
        """
        self.model = model
        self.draft_layers = draft_layers
        self.config = config or SpeculativeConfig()
        self.stats = SpeculativeStats()

        # Note: Full implementation would require model modifications
        # to support early exit. This is a placeholder structure.

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 100,
        eos_token_id: int = 2,
    ) -> Tuple[List[int], SpeculativeStats]:
        """
        Generate tokens using self-speculative decoding.

        Note: This is a simplified implementation. Full implementation
        requires model architecture changes for early exit.
        """
        # For now, fall back to standard generation
        # In production, this would use early layer outputs
        self.model.reset()
        self.stats = SpeculativeStats()

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=self.config.target_temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            eos_token_id=eos_token_id,
        )

        output_tokens = self.model.generate(prompt_tokens, gen_config)
        self.stats.total_tokens = len(output_tokens)
        self.stats.num_iterations = len(output_tokens)

        return output_tokens, self.stats


class LookaheadDecoder:
    """Lookahead decoding for parallel token generation.

    Uses n-gram caches and parallel verification to generate multiple
    tokens per forward pass without requiring a draft model.

    Reference: "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"
    """

    def __init__(
        self,
        model: TransformerInference,
        window_size: int = 4,
        ngram_size: int = 2,
    ):
        """
        Initialize lookahead decoder.

        Args:
            model: The main model
            window_size: Number of positions to look ahead
            ngram_size: Size of n-grams for caching
        """
        self.model = model
        self.window_size = window_size
        self.ngram_size = ngram_size
        self.stats = SpeculativeStats()

        # N-gram cache: maps (context) -> (continuation, count)
        self._ngram_cache: dict = {}

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 100,
        eos_token_id: int = 2,
    ) -> Tuple[List[int], SpeculativeStats]:
        """
        Generate tokens using lookahead decoding.

        Note: Simplified implementation. Full lookahead requires
        parallel attention computation which isn't supported in
        the basic inference engine.
        """
        self.model.reset()
        self.stats = SpeculativeStats()

        # Process prompt
        for i, token in enumerate(prompt_tokens[:-1]):
            self.model.forward(token, i)

        position = len(prompt_tokens) - 1
        token = prompt_tokens[-1]
        output_tokens: List[int] = []

        # Build initial context
        context = list(prompt_tokens[-(self.ngram_size - 1):]) if len(prompt_tokens) >= self.ngram_size - 1 else list(prompt_tokens)

        while len(output_tokens) < max_new_tokens:
            self.stats.num_iterations += 1

            # Check n-gram cache for potential continuation
            cache_key = tuple(context[-(self.ngram_size - 1):])
            cached = self._ngram_cache.get(cache_key)

            if cached and np.random.random() < 0.5:  # Use cache probabilistically
                # Try cached continuation
                candidate_token = cached[0]
                logits = self.model.forward(token, position)
                probs = softmax(logits)

                # Verify candidate
                if probs[candidate_token] > 0.1:  # Threshold
                    token = candidate_token
                    self.stats.accepted_tokens += 1
                else:
                    # Sample new token
                    token = int(np.random.choice(len(probs), p=probs))
                    self.stats.rejected_tokens += 1
            else:
                # Standard generation
                logits = self.model.forward(token, position)
                probs = softmax(logits)
                token = int(np.random.choice(len(probs), p=probs))

            # Update cache
            if len(context) >= self.ngram_size - 1:
                key = tuple(context[-(self.ngram_size - 1):])
                if key in self._ngram_cache:
                    old_token, count = self._ngram_cache[key]
                    if token == old_token:
                        self._ngram_cache[key] = (token, count + 1)
                else:
                    self._ngram_cache[key] = (token, 1)

            output_tokens.append(token)
            context.append(token)
            position += 1
            self.stats.total_tokens += 1

            if token == eos_token_id:
                break

        return output_tokens, self.stats

    def clear_cache(self) -> None:
        """Clear the n-gram cache."""
        self._ngram_cache.clear()
