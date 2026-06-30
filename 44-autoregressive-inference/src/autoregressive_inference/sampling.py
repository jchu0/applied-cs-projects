"""Token sampling strategies for autoregressive inference.

This module implements various sampling strategies including temperature scaling,
top-k, top-p (nucleus) sampling, and repetition penalties.
"""

from typing import List, Optional, Dict
from collections import Counter

# Try to import torch, fall back to numpy for testing
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np


def _is_torch_tensor(x):
    """Check if x is a torch tensor."""
    return HAS_TORCH and isinstance(x, torch.Tensor)

# Import from local module
try:
    from .requests import SamplingParams
except ImportError:
    from requests import SamplingParams


class TokenSampler:
    """Token sampling with various strategies."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the token sampler.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.seed = seed
        # Use instance-level random generators for reproducibility
        # Always initialize both to support mixed numpy/torch inputs
        if seed is not None:
            self._rng = np.random.RandomState(seed)
            if HAS_TORCH:
                self._torch_generator = torch.Generator()
                self._torch_generator.manual_seed(seed)
            else:
                self._torch_generator = None
        else:
            self._rng = np.random.RandomState()
            self._torch_generator = None

    def sample(
        self,
        logits,
        params: SamplingParams,
        generated_ids: Optional[List[int]] = None
    ):
        """
        Sample next token from logits.

        Args:
            logits: [batch_size, vocab_size] tensor of logits.
            params: Sampling parameters.
            generated_ids: Previously generated IDs for penalties.

        Returns:
            [batch_size] sampled token IDs.
        """
        # Make a copy to avoid modifying original
        if _is_torch_tensor(logits):
            logits = logits.clone()
        else:
            logits = logits.copy()

        # Apply repetition penalty
        if params.repetition_penalty != 1.0 and generated_ids:
            logits = self._apply_repetition_penalty(
                logits, generated_ids, params.repetition_penalty
            )

        # Apply frequency/presence penalties
        if (params.frequency_penalty != 0 or params.presence_penalty != 0) and generated_ids:
            logits = self._apply_frequency_presence_penalty(
                logits, generated_ids,
                params.frequency_penalty, params.presence_penalty
            )

        # Apply temperature
        if params.temperature != 1.0 and params.temperature > 0:
            logits = logits / params.temperature

        # Apply top-k filtering
        if params.top_k > 0:
            logits = self._top_k_filtering(logits, params.top_k)

        # Apply top-p (nucleus) filtering
        if params.top_p < 1.0:
            logits = self._top_p_filtering(logits, params.top_p)

        # Sample from distribution
        if _is_torch_tensor(logits):
            probs = F.softmax(logits, dim=-1)

            if params.temperature == 0:
                # Greedy
                result = torch.argmax(probs, dim=-1)
            else:
                if self._torch_generator is not None:
                    result = torch.multinomial(probs, num_samples=1, generator=self._torch_generator).squeeze(-1)
                else:
                    result = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Return scalar for single-item batch
            if result.numel() == 1:
                return result.item()
            return result
        else:
            # Softmax for numpy
            logits_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            if params.temperature == 0:
                # Greedy
                result = np.argmax(probs, axis=-1)
            else:
                # Sample from each row using instance RNG
                if len(probs.shape) == 1:
                    result = self._rng.choice(len(probs), p=probs)
                else:
                    result = np.array([
                        self._rng.choice(len(p), p=p) for p in probs
                    ])

            # Return scalar for single-item result
            if isinstance(result, np.ndarray) and result.size == 1:
                return int(result.flat[0])
            return result

    def sample_greedy(self, logits):
        """Greedy sampling (argmax).

        Args:
            logits: [batch_size, vocab_size] tensor of logits.

        Returns:
            [batch_size] token IDs.
        """
        if _is_torch_tensor(logits):
            return torch.argmax(logits, dim=-1)
        else:
            return np.argmax(logits, axis=-1)

    def sample_with_temperature(self, logits, temperature: float):
        """Sample with temperature scaling.

        Args:
            logits: [batch_size, vocab_size] tensor of logits.
            temperature: Temperature value (0 = greedy, 1 = standard).

        Returns:
            [batch_size] token IDs, or scalar for batch_size=1.
        """
        if temperature == 0:
            return self.sample_greedy(logits)

        scaled_logits = logits / temperature

        if _is_torch_tensor(logits):
            probs = F.softmax(scaled_logits, dim=-1)
            if self._torch_generator is not None:
                result = torch.multinomial(probs, num_samples=1, generator=self._torch_generator).squeeze(-1)
            else:
                result = torch.multinomial(probs, num_samples=1).squeeze(-1)
            if result.numel() == 1:
                return result.item()
            return result
        else:
            logits_max = np.max(scaled_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(scaled_logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            if len(probs.shape) == 1:
                result = self._rng.choice(len(probs), p=probs)
            else:
                result = np.array([
                    self._rng.choice(len(p), p=p) for p in probs
                ])

            # Return scalar for single-item result
            if isinstance(result, np.ndarray) and result.size == 1:
                return int(result.flat[0])
            return result

    def _top_k_filtering(self, logits, top_k: int):
        """Keep only top-k logits.

        Args:
            logits: [batch_size, vocab_size] tensor of logits.
            top_k: Number of top tokens to keep.

        Returns:
            Filtered logits tensor.
        """
        if top_k <= 0:
            return logits

        if _is_torch_tensor(logits):
            # Get top-k values
            values, _ = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
            min_value = values[:, -1].unsqueeze(-1) if len(logits.shape) > 1 else values[-1]

            return torch.where(
                logits < min_value,
                torch.full_like(logits, float('-inf')),
                logits
            )
        else:
            if len(logits.shape) == 1:
                top_k = min(top_k, len(logits))
                indices = np.argpartition(logits, -top_k)[-top_k:]
                # Explicitly mask all non-top-k tokens
                result = np.full_like(logits, float('-inf'))
                result[indices] = logits[indices]
                return result
            else:
                result = np.full_like(logits, float('-inf'))
                for i in range(logits.shape[0]):
                    top_k_actual = min(top_k, logits.shape[-1])
                    indices = np.argpartition(logits[i], -top_k_actual)[-top_k_actual:]
                    # Explicitly mask all non-top-k tokens
                    result[i, indices] = logits[i, indices]
                return result

    def _top_p_filtering(self, logits, top_p: float):
        """Nucleus sampling: keep tokens with cumulative prob <= top_p.

        Args:
            logits: [batch_size, vocab_size] tensor of logits.
            top_p: Cumulative probability threshold.

        Returns:
            Filtered logits tensor.
        """
        if top_p >= 1.0:
            return logits

        if _is_torch_tensor(logits):
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Find cutoff
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            if len(sorted_indices_to_remove.shape) > 1:
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
            else:
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False

            # Scatter back
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

            logits = logits.clone()
            logits[indices_to_remove] = float('-inf')
            return logits
        else:
            if len(logits.shape) == 1:
                sorted_indices = np.argsort(logits)[::-1]
                sorted_logits = logits[sorted_indices]

                # Softmax
                max_logit = np.max(sorted_logits)
                exp_logits = np.exp(sorted_logits - max_logit)
                probs = exp_logits / np.sum(exp_logits)
                cumulative_probs = np.cumsum(probs)

                # Find cutoff
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False

                # Create mask
                indices_to_remove = np.zeros(len(logits), dtype=bool)
                indices_to_remove[sorted_indices] = sorted_indices_to_remove

                result = logits.copy()
                result[indices_to_remove] = float('-inf')
                return result
            else:
                result = logits.copy()
                for i in range(logits.shape[0]):
                    sorted_indices = np.argsort(logits[i])[::-1]
                    sorted_logits = logits[i, sorted_indices]

                    max_logit = np.max(sorted_logits)
                    exp_logits = np.exp(sorted_logits - max_logit)
                    probs = exp_logits / np.sum(exp_logits)
                    cumulative_probs = np.cumsum(probs)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False

                    indices_to_remove = np.zeros(logits.shape[-1], dtype=bool)
                    indices_to_remove[sorted_indices] = sorted_indices_to_remove
                    result[i, indices_to_remove] = float('-inf')

                return result

    def _apply_repetition_penalty(
        self,
        logits,
        generated_ids: List[int],
        penalty: float
    ):
        """Apply repetition penalty to discourage repeats.

        Args:
            logits: [batch_size, vocab_size] tensor of logits.
            generated_ids: Previously generated token IDs.
            penalty: Penalty factor (>1 discourages, <1 encourages).

        Returns:
            Modified logits tensor.
        """
        if _is_torch_tensor(logits):
            logits = logits.clone()
            for token_id in set(generated_ids):
                if len(logits.shape) > 1:
                    for i in range(logits.shape[0]):
                        if logits[i, token_id] < 0:
                            logits[i, token_id] *= penalty
                        else:
                            logits[i, token_id] /= penalty
                else:
                    if logits[token_id] < 0:
                        logits[token_id] *= penalty
                    else:
                        logits[token_id] /= penalty
        else:
            logits = logits.copy()
            for token_id in set(generated_ids):
                if len(logits.shape) > 1:
                    for i in range(logits.shape[0]):
                        if logits[i, token_id] < 0:
                            logits[i, token_id] *= penalty
                        else:
                            logits[i, token_id] /= penalty
                else:
                    if logits[token_id] < 0:
                        logits[token_id] *= penalty
                    else:
                        logits[token_id] /= penalty

        return logits

    def _apply_frequency_presence_penalty(
        self,
        logits,
        generated_ids: List[int],
        frequency_penalty: float,
        presence_penalty: float
    ):
        """Apply frequency and presence penalties.

        Args:
            logits: [batch_size, vocab_size] tensor of logits.
            generated_ids: Previously generated token IDs.
            frequency_penalty: Penalty proportional to frequency.
            presence_penalty: Flat penalty for appearing at all.

        Returns:
            Modified logits tensor.
        """
        token_counts = Counter(generated_ids)

        if _is_torch_tensor(logits):
            logits = logits.clone()
            for token_id, count in token_counts.items():
                if len(logits.shape) > 1:
                    logits[:, token_id] -= frequency_penalty * count
                    logits[:, token_id] -= presence_penalty
                else:
                    logits[token_id] -= frequency_penalty * count
                    logits[token_id] -= presence_penalty
        else:
            logits = logits.copy()
            for token_id, count in token_counts.items():
                if len(logits.shape) > 1:
                    logits[:, token_id] -= frequency_penalty * count
                    logits[:, token_id] -= presence_penalty
                else:
                    logits[token_id] -= frequency_penalty * count
                    logits[token_id] -= presence_penalty

        return logits

    def get_top_tokens(
        self,
        logits,
        k: int = 10
    ) -> List[Dict]:
        """Get top-k tokens with their probabilities.

        Args:
            logits: [vocab_size] tensor of logits.
            k: Number of top tokens to return.

        Returns:
            List of dicts with 'token_id', 'logit', and 'probability'.
        """
        if _is_torch_tensor(logits):
            probs = F.softmax(logits, dim=-1)
            if len(logits.shape) > 1:
                logits = logits[0]
                probs = probs[0]

            top_probs, top_indices = torch.topk(probs, min(k, len(probs)))

            return [
                {
                    'token_id': idx.item(),
                    'logit': logits[idx].item(),
                    'probability': prob.item()
                }
                for idx, prob in zip(top_indices, top_probs)
            ]
        else:
            if len(logits.shape) > 1:
                logits = logits[0]

            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)

            top_indices = np.argpartition(probs, -min(k, len(probs)))[-min(k, len(probs)):]
            top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]

            return [
                {
                    'token_id': int(idx),
                    'logit': float(logits[idx]),
                    'probability': float(probs[idx])
                }
                for idx in top_indices
            ]
