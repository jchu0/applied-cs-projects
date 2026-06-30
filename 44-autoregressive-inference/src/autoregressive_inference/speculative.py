"""Speculative decoding for autoregressive inference.

This module implements speculative decoding using a small draft model
to generate multiple tokens that are verified by the target model.
"""

from typing import List, Optional, Any, Protocol, Tuple
from dataclasses import dataclass, field

try:
    import torch
    HAS_TORCH = True
except ImportError:
    import numpy as np
    HAS_TORCH = False

try:
    from .kv_cache import PagedKVCacheManager
    from .sampling import TokenSampler
    from .requests import SamplingParams
except ImportError:
    from kv_cache import PagedKVCacheManager
    from sampling import TokenSampler
    from requests import SamplingParams


class DraftModel(Protocol):
    """Protocol for draft models used in speculative decoding."""

    def forward(self, input_ids: Any) -> Any:
        """Forward pass.

        Args:
            input_ids: Input token IDs.

        Returns:
            Logits tensor.
        """
        ...


class TargetModel(Protocol):
    """Protocol for target models used in speculative decoding."""

    def forward_with_kv(
        self,
        input_ids: Any,
        kv_cache: PagedKVCacheManager,
        request_id: str
    ) -> Any:
        """Forward pass with KV cache.

        Args:
            input_ids: Input token IDs.
            kv_cache: KV cache manager.
            request_id: Request identifier.

        Returns:
            Logits tensor.
        """
        ...


@dataclass
class SpeculativeStats:
    """Statistics for speculative decoding."""
    total_drafted: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    acceptance_by_position: List[int] = field(default_factory=lambda: [0] * 8)
    rejection_by_position: List[int] = field(default_factory=lambda: [0] * 8)

    @property
    def acceptance_rate(self) -> float:
        """Get overall acceptance rate."""
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    @property
    def avg_accepted_per_step(self) -> float:
        """Get average tokens accepted per speculative step."""
        if self.total_drafted == 0:
            return 0.0
        steps = self.total_drafted / max(len([x for x in self.acceptance_by_position if x > 0]), 1)
        return self.total_accepted / steps if steps > 0 else 0.0


class SpeculativeDecoder:
    """
    Speculative decoding with small draft model.

    Uses a smaller model to draft multiple tokens,
    then verifies with the large model in parallel.
    """

    def __init__(
        self,
        target_model: Optional[TargetModel] = None,
        draft_model: Optional[DraftModel] = None,
        num_speculative_tokens: int = 4,
        sampler: Optional[TokenSampler] = None
    ):
        """Initialize speculative decoder.

        Args:
            target_model: The large target model.
            draft_model: The small draft model.
            num_speculative_tokens: Number of tokens to speculate.
            sampler: Token sampler.
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.num_speculative = num_speculative_tokens
        self.sampler = sampler or TokenSampler()

        # Statistics
        self.stats = SpeculativeStats()

    def generate_step(
        self,
        input_ids: Any,
        kv_cache: Optional[PagedKVCacheManager],
        request_id: str,
        sampling_params: SamplingParams
    ) -> List[int]:
        """
        Generate tokens using speculative decoding.

        Args:
            input_ids: Current input token IDs.
            kv_cache: KV cache manager.
            request_id: Request identifier.
            sampling_params: Sampling parameters.

        Returns:
            List of accepted tokens.
        """
        # Step 1: Draft tokens with small model
        draft_tokens = self._draft(input_ids, sampling_params)

        # Step 2: Verify with target model (parallel forward pass)
        accepted = self._verify(
            input_ids, draft_tokens, kv_cache, request_id, sampling_params
        )

        # Update stats
        self.stats.total_drafted += len(draft_tokens)
        self.stats.total_accepted += len(accepted)

        return accepted

    def _draft(
        self,
        input_ids: Any,
        sampling_params: SamplingParams
    ) -> List[int]:
        """Generate draft tokens with small model.

        Args:
            input_ids: Current input token IDs.
            sampling_params: Sampling parameters.

        Returns:
            List of drafted token IDs.
        """
        if self.draft_model is None:
            # Mock draft for testing
            return [100 + i for i in range(self.num_speculative)]

        draft_tokens = []
        current_ids = input_ids

        for i in range(self.num_speculative):
            if HAS_TORCH:
                with torch.no_grad():
                    logits = self.draft_model.forward(current_ids)

                # Sample greedily for drafting (faster)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                draft_tokens.append(next_token.item())

                current_ids = torch.cat([
                    current_ids,
                    next_token.unsqueeze(0).unsqueeze(0)
                ], dim=-1)
            else:
                logits = self.draft_model.forward(current_ids)
                next_token = np.argmax(logits[:, -1, :], axis=-1)
                draft_tokens.append(int(next_token))

                current_ids = np.concatenate([
                    current_ids,
                    [[int(next_token)]]
                ], axis=-1)

        return draft_tokens

    def _verify(
        self,
        input_ids: Any,
        draft_tokens: List[int],
        kv_cache: Optional[PagedKVCacheManager],
        request_id: str,
        sampling_params: SamplingParams
    ) -> List[int]:
        """Verify draft tokens with target model.

        Args:
            input_ids: Original input token IDs.
            draft_tokens: Drafted token IDs.
            kv_cache: KV cache manager.
            request_id: Request identifier.
            sampling_params: Sampling parameters.

        Returns:
            List of accepted token IDs.
        """
        if self.target_model is None:
            # Mock verification for testing
            # Accept first few tokens, reject later ones
            accepted = []
            for i, token in enumerate(draft_tokens):
                if i < 2:  # Accept first 2
                    accepted.append(token)
                    if i < len(self.stats.acceptance_by_position):
                        self.stats.acceptance_by_position[i] += 1
                else:
                    # Reject and return different token
                    accepted.append(token + 1)
                    if i < len(self.stats.rejection_by_position):
                        self.stats.rejection_by_position[i] += 1
                    self.stats.total_rejected += 1
                    break
            return accepted

        # Concatenate input with draft tokens
        if HAS_TORCH:
            draft_ids = torch.tensor([draft_tokens], device=input_ids.device)
            full_ids = torch.cat([input_ids, draft_ids], dim=-1)

            # Forward pass (gets logits for all positions)
            with torch.no_grad():
                logits = self.target_model.forward_with_kv(
                    full_ids, kv_cache, request_id
                )
        else:
            draft_ids = np.array([draft_tokens])
            full_ids = np.concatenate([input_ids, draft_ids], axis=-1)
            logits = self.target_model.forward_with_kv(
                full_ids, kv_cache, request_id
            )

        # Verify each draft token
        accepted = []

        for i, draft_token in enumerate(draft_tokens):
            pos = input_ids.shape[1] + i - 1 if input_ids.shape[1] > 0 else i
            token_logits = logits[:, pos:pos+1, :]

            # Sample from target distribution
            if HAS_TORCH:
                target_token = self.sampler.sample(
                    token_logits.squeeze(1),
                    sampling_params,
                    None
                )
                target_token_id = target_token.item()
            else:
                target_token = self.sampler.sample(
                    token_logits.squeeze(1),
                    sampling_params,
                    None
                )
                target_token_id = int(target_token)

            if target_token_id == draft_token:
                accepted.append(draft_token)
                if i < len(self.stats.acceptance_by_position):
                    self.stats.acceptance_by_position[i] += 1
            else:
                # Rejection - use target token and stop
                accepted.append(target_token_id)
                if i < len(self.stats.rejection_by_position):
                    self.stats.rejection_by_position[i] += 1
                self.stats.total_rejected += 1
                break

        return accepted

    def get_acceptance_rate(self) -> float:
        """Get average acceptance rate."""
        return self.stats.acceptance_rate

    def get_stats(self) -> dict:
        """Get detailed statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            'total_drafted': self.stats.total_drafted,
            'total_accepted': self.stats.total_accepted,
            'total_rejected': self.stats.total_rejected,
            'acceptance_rate': self.stats.acceptance_rate,
            'avg_accepted_per_step': self.stats.avg_accepted_per_step,
            'acceptance_by_position': self.stats.acceptance_by_position[:self.num_speculative],
            'rejection_by_position': self.stats.rejection_by_position[:self.num_speculative],
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = SpeculativeStats()


class TreeSpeculativeDecoder:
    """
    Tree-based speculative decoding.

    Generates a tree of draft tokens and verifies multiple paths
    in parallel for higher acceptance rates.
    """

    def __init__(
        self,
        target_model: Optional[TargetModel] = None,
        draft_model: Optional[DraftModel] = None,
        tree_width: int = 2,
        tree_depth: int = 4,
        sampler: Optional[TokenSampler] = None
    ):
        """Initialize tree speculative decoder.

        Args:
            target_model: The large target model.
            draft_model: The small draft model.
            tree_width: Number of candidates at each position.
            tree_depth: Maximum tree depth.
            sampler: Token sampler.
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.sampler = sampler or TokenSampler()

        self.stats = SpeculativeStats()

    def generate_step(
        self,
        input_ids: Any,
        kv_cache: Optional[PagedKVCacheManager],
        request_id: str,
        sampling_params: SamplingParams
    ) -> List[int]:
        """Generate tokens using tree speculative decoding.

        Args:
            input_ids: Current input token IDs.
            kv_cache: KV cache manager.
            request_id: Request identifier.
            sampling_params: Sampling parameters.

        Returns:
            List of accepted tokens from best path.
        """
        # Generate tree of draft tokens
        draft_tree = self._draft_tree(input_ids, sampling_params)

        # Verify tree and find best path
        accepted = self._verify_tree(
            input_ids, draft_tree, kv_cache, request_id, sampling_params
        )

        return accepted

    def _draft_tree(
        self,
        input_ids: Any,
        sampling_params: SamplingParams
    ) -> List[List[int]]:
        """Generate tree of draft tokens.

        Args:
            input_ids: Current input token IDs.
            sampling_params: Sampling parameters.

        Returns:
            List of paths (each path is a list of token IDs).
        """
        if self.draft_model is None:
            # Mock tree for testing - generate all tree_width^tree_depth paths
            from itertools import product
            all_paths = []
            # Generate all combinations of choices at each depth level
            choices = list(range(self.tree_width))
            for combo in product(choices, repeat=self.tree_depth):
                # Each path has tree_depth tokens, with values based on position and choice
                path = [100 + depth + choice * 10 for depth, choice in enumerate(combo)]
                all_paths.append(path)
            return all_paths

        # Generate multiple candidates at each level
        paths = [[]]  # Start with empty path

        for depth in range(self.tree_depth):
            new_paths = []
            for path in paths:
                # Build input for this path
                if HAS_TORCH:
                    path_ids = torch.tensor([path], device=input_ids.device) if path else None
                    if path_ids is not None:
                        current_ids = torch.cat([input_ids, path_ids], dim=-1)
                    else:
                        current_ids = input_ids

                    with torch.no_grad():
                        logits = self.draft_model.forward(current_ids)

                    # Get top-k candidates
                    top_tokens = torch.topk(logits[:, -1, :], self.tree_width, dim=-1)

                    for i in range(self.tree_width):
                        new_path = path + [top_tokens.indices[0, i].item()]
                        new_paths.append(new_path)
                else:
                    path_ids = np.array([path]) if path else None
                    if path_ids is not None:
                        current_ids = np.concatenate([input_ids, path_ids], axis=-1)
                    else:
                        current_ids = input_ids

                    logits = self.draft_model.forward(current_ids)

                    # Get top-k candidates
                    top_indices = np.argpartition(logits[0, -1, :], -self.tree_width)[-self.tree_width:]

                    for i in range(self.tree_width):
                        new_path = path + [int(top_indices[i])]
                        new_paths.append(new_path)

            paths = new_paths

        return paths

    def _verify_tree(
        self,
        input_ids: Any,
        draft_tree: List[List[int]],
        kv_cache: Optional[PagedKVCacheManager],
        request_id: str,
        sampling_params: SamplingParams
    ) -> List[int]:
        """Verify tree and find best path.

        Args:
            input_ids: Original input token IDs.
            draft_tree: Tree of draft paths.
            kv_cache: KV cache manager.
            request_id: Request identifier.
            sampling_params: Sampling parameters.

        Returns:
            Best accepted path.
        """
        if self.target_model is None:
            # Mock verification - return first path
            if draft_tree:
                path = draft_tree[0]
                self.stats.total_drafted += len(path)
                self.stats.total_accepted += min(2, len(path))
                return path[:2]
            return []

        # Batch verify all paths
        best_path = []
        best_length = 0

        for path in draft_tree:
            if HAS_TORCH:
                draft_ids = torch.tensor([path], device=input_ids.device)
                full_ids = torch.cat([input_ids, draft_ids], dim=-1)

                with torch.no_grad():
                    logits = self.target_model.forward_with_kv(
                        full_ids, kv_cache, request_id
                    )
            else:
                draft_ids = np.array([path])
                full_ids = np.concatenate([input_ids, draft_ids], axis=-1)
                logits = self.target_model.forward_with_kv(
                    full_ids, kv_cache, request_id
                )

            # Check how many tokens match
            accepted = []
            for i, draft_token in enumerate(path):
                pos = input_ids.shape[1] + i - 1 if input_ids.shape[1] > 0 else i
                token_logits = logits[:, pos:pos+1, :]

                if HAS_TORCH:
                    target_token = self.sampler.sample(
                        token_logits.squeeze(1),
                        sampling_params,
                        None
                    )
                    target_token_id = target_token.item()
                else:
                    target_token = self.sampler.sample(
                        token_logits.squeeze(1),
                        sampling_params,
                        None
                    )
                    target_token_id = int(target_token)

                if target_token_id == draft_token:
                    accepted.append(draft_token)
                else:
                    accepted.append(target_token_id)
                    break

            if len(accepted) > best_length:
                best_path = accepted
                best_length = len(accepted)

        self.stats.total_drafted += len(draft_tree[0]) if draft_tree else 0
        self.stats.total_accepted += best_length

        return best_path

    def get_stats(self) -> dict:
        """Get statistics."""
        return {
            'total_drafted': self.stats.total_drafted,
            'total_accepted': self.stats.total_accepted,
            'acceptance_rate': self.stats.acceptance_rate,
            'tree_width': self.tree_width,
            'tree_depth': self.tree_depth,
        }
