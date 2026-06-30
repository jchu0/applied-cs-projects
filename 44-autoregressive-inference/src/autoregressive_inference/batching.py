"""Continuous batching for autoregressive inference.

This module implements continuous batching that allows dynamic addition and
removal of requests during inference, maximizing GPU utilization.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

# Try to import torch, fall back to numpy for testing
try:
    import torch
    HAS_TORCH = True
except ImportError:
    import numpy as np
    HAS_TORCH = False

# Import from local module
try:
    from .requests import InferenceRequest, RequestStatus
except ImportError:
    from requests import InferenceRequest, RequestStatus


class ContinuousBatcher:
    """
    Continuous batching engine for maximizing GPU utilization.

    Key features:
    - Dynamic batch formation
    - Split prefill and decode phases
    - In-flight batching (add/remove requests)
    """

    def __init__(
        self,
        max_batch_size: int = 64,
        max_prefill_tokens: int = 4096,
        max_decode_tokens: int = 2048
    ):
        """Initialize the continuous batcher.

        Args:
            max_batch_size: Maximum number of requests in a batch.
            max_prefill_tokens: Maximum tokens for prefill phase.
            max_decode_tokens: Maximum tokens for decode phase.
        """
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.max_decode_tokens = max_decode_tokens

        # Current batches
        self.prefill_batch: List[InferenceRequest] = []
        self.decode_batch: List[InferenceRequest] = []

    def form_batches(
        self,
        pending_requests: List[InferenceRequest],
        running_requests: List[InferenceRequest]
    ) -> Tuple[List[InferenceRequest], List[InferenceRequest]]:
        """
        Form prefill and decode batches.

        Args:
            pending_requests: Requests waiting to be processed.
            running_requests: Requests currently being processed.

        Returns:
            Tuple of (prefill_batch, decode_batch).
        """
        # Decode batch: all running requests that completed prefill
        decode_batch = [
            r for r in running_requests
            if r.status == RequestStatus.RUNNING_DECODE
        ]

        # Limit decode batch size
        if len(decode_batch) > self.max_batch_size:
            decode_batch = decode_batch[:self.max_batch_size]

        # Calculate remaining capacity
        decode_tokens = len(decode_batch)  # 1 token per request
        remaining_capacity = self.max_prefill_tokens - decode_tokens

        # Fill prefill batch
        prefill_batch = []
        prefill_tokens = 0

        for request in pending_requests:
            tokens = len(request.prompt_token_ids)
            if prefill_tokens + tokens <= remaining_capacity:
                if len(prefill_batch) + len(decode_batch) < self.max_batch_size:
                    prefill_batch.append(request)
                    prefill_tokens += tokens

        return prefill_batch, decode_batch

    def can_add_request(
        self,
        request: InferenceRequest,
        current_batch_size: int,
        current_tokens: int
    ) -> bool:
        """Check if a request can be added to the current batch.

        Args:
            request: The request to potentially add.
            current_batch_size: Current number of requests in batch.
            current_tokens: Current total tokens in batch.

        Returns:
            True if the request can be added.
        """
        prompt_tokens = len(request.prompt_token_ids)
        return (
            current_batch_size < self.max_batch_size and
            current_tokens + prompt_tokens <= self.max_prefill_tokens
        )

    def estimate_batch_time(
        self,
        prefill_batch: List[InferenceRequest],
        decode_batch: List[InferenceRequest],
        tokens_per_second: float = 1000.0
    ) -> float:
        """Estimate time to process a batch.

        Args:
            prefill_batch: Prefill requests.
            decode_batch: Decode requests.
            tokens_per_second: Estimated throughput.

        Returns:
            Estimated time in seconds.
        """
        prefill_tokens = sum(len(r.prompt_token_ids) for r in prefill_batch)
        decode_tokens = len(decode_batch)

        # Prefill is compute-bound, decode is memory-bound
        # Rough estimate: prefill takes 2x longer per token
        total_effective_tokens = prefill_tokens * 2 + decode_tokens

        return total_effective_tokens / tokens_per_second

    def merge_for_execution(
        self,
        prefill_batch: List[InferenceRequest],
        decode_batch: List[InferenceRequest]
    ) -> 'BatchedInputs':
        """Merge prefill and decode into single execution batch.

        Args:
            prefill_batch: Prefill requests.
            decode_batch: Decode requests.

        Returns:
            BatchedInputs object for model execution.
        """
        return BatchedInputs(
            prefill_requests=prefill_batch,
            decode_requests=decode_batch
        )

    def get_batch_stats(
        self,
        prefill_batch: List[InferenceRequest],
        decode_batch: List[InferenceRequest]
    ) -> dict:
        """Get statistics about a batch.

        Args:
            prefill_batch: Prefill requests.
            decode_batch: Decode requests.

        Returns:
            Dictionary of batch statistics.
        """
        prefill_tokens = sum(len(r.prompt_token_ids) for r in prefill_batch)
        decode_tokens = len(decode_batch)

        return {
            'prefill_requests': len(prefill_batch),
            'decode_requests': len(decode_batch),
            'total_requests': len(prefill_batch) + len(decode_batch),
            'prefill_tokens': prefill_tokens,
            'decode_tokens': decode_tokens,
            'total_tokens': prefill_tokens + decode_tokens,
            'batch_utilization': (
                (len(prefill_batch) + len(decode_batch)) / self.max_batch_size
            ),
        }


@dataclass
class BatchedInputs:
    """Batched inputs for model execution."""
    prefill_requests: List[InferenceRequest] = field(default_factory=list)
    decode_requests: List[InferenceRequest] = field(default_factory=list)

    def get_input_ids(self) -> Any:
        """Get padded input token IDs.

        Returns:
            Tensor of shape [batch_size, max_seq_len].
        """
        all_ids = []

        # Prefill: full prompt
        for req in self.prefill_requests:
            if HAS_TORCH:
                all_ids.append(torch.tensor(req.prompt_token_ids))
            else:
                all_ids.append(np.array(req.prompt_token_ids))

        # Decode: just last token
        for req in self.decode_requests:
            if req.output_token_ids:
                token = req.output_token_ids[-1]
            else:
                # First decode token
                token = req.prompt_token_ids[-1]

            if HAS_TORCH:
                all_ids.append(torch.tensor([token]))
            else:
                all_ids.append(np.array([token]))

        if not all_ids:
            if HAS_TORCH:
                return torch.zeros(0, 0, dtype=torch.long)
            else:
                return np.zeros((0, 0), dtype=np.int64)

        # Pad to same length
        max_len = max(len(ids) for ids in all_ids)

        if HAS_TORCH:
            padded = torch.zeros(len(all_ids), max_len, dtype=torch.long)
            for i, ids in enumerate(all_ids):
                padded[i, :len(ids)] = ids
        else:
            padded = np.zeros((len(all_ids), max_len), dtype=np.int64)
            for i, ids in enumerate(all_ids):
                padded[i, :len(ids)] = ids

        return padded

    def get_position_ids(self) -> Any:
        """Get position IDs for each token.

        Returns:
            Tensor of shape [batch_size, max_seq_len].
        """
        positions = []

        for req in self.prefill_requests:
            seq_len = len(req.prompt_token_ids)
            if HAS_TORCH:
                positions.append(torch.arange(seq_len))
            else:
                positions.append(np.arange(seq_len))

        for req in self.decode_requests:
            # Position is total length so far
            pos = len(req.prompt_token_ids) + len(req.output_token_ids)
            if HAS_TORCH:
                positions.append(torch.tensor([pos]))
            else:
                positions.append(np.array([pos]))

        if not positions:
            if HAS_TORCH:
                return torch.zeros(0, 0, dtype=torch.long)
            else:
                return np.zeros((0, 0), dtype=np.int64)

        # Pad
        max_len = max(len(p) for p in positions)

        if HAS_TORCH:
            padded = torch.zeros(len(positions), max_len, dtype=torch.long)
            for i, pos in enumerate(positions):
                padded[i, :len(pos)] = pos
        else:
            padded = np.zeros((len(positions), max_len), dtype=np.int64)
            for i, pos in enumerate(positions):
                padded[i, :len(pos)] = pos

        return padded

    def get_attention_mask(self) -> Any:
        """Get attention mask for padding.

        Returns:
            Tensor of shape [batch_size, max_seq_len] with 1s for real tokens.
        """
        lengths = []

        for req in self.prefill_requests:
            lengths.append(len(req.prompt_token_ids))

        for req in self.decode_requests:
            lengths.append(1)  # Decode is always 1 token

        if not lengths:
            if HAS_TORCH:
                return torch.zeros(0, 0, dtype=torch.long)
            else:
                return np.zeros((0, 0), dtype=np.int64)

        max_len = max(lengths)

        if HAS_TORCH:
            mask = torch.zeros(len(lengths), max_len, dtype=torch.long)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1
        else:
            mask = np.zeros((len(lengths), max_len), dtype=np.int64)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1

        return mask

    def get_sequence_lengths(self) -> List[int]:
        """Get sequence lengths for each request.

        Returns:
            List of sequence lengths.
        """
        lengths = []

        for req in self.prefill_requests:
            lengths.append(len(req.prompt_token_ids))

        for req in self.decode_requests:
            lengths.append(1)

        return lengths

    def get_request_ids(self) -> List[str]:
        """Get request IDs in batch order.

        Returns:
            List of request IDs.
        """
        return (
            [r.request_id for r in self.prefill_requests] +
            [r.request_id for r in self.decode_requests]
        )

    @property
    def batch_size(self) -> int:
        """Get total batch size."""
        return len(self.prefill_requests) + len(self.decode_requests)

    @property
    def num_prefill(self) -> int:
        """Get number of prefill requests."""
        return len(self.prefill_requests)

    @property
    def num_decode(self) -> int:
        """Get number of decode requests."""
        return len(self.decode_requests)

    @property
    def total_prefill_tokens(self) -> int:
        """Get total number of prefill tokens."""
        return sum(len(r.prompt_token_ids) for r in self.prefill_requests)

    @property
    def is_empty(self) -> bool:
        """Check if batch is empty."""
        return self.batch_size == 0


class SchedulingPolicy:
    """Base class for batch scheduling policies."""

    def select_requests(
        self,
        pending: List[InferenceRequest],
        running: List[InferenceRequest],
        max_batch_size: int,
        max_tokens: int
    ) -> Tuple[List[InferenceRequest], List[InferenceRequest]]:
        """Select requests for the next batch.

        Args:
            pending: Pending requests.
            running: Running requests.
            max_batch_size: Maximum batch size.
            max_tokens: Maximum tokens.

        Returns:
            Tuple of (selected_prefill, selected_decode).
        """
        raise NotImplementedError


class FIFOPolicy(SchedulingPolicy):
    """First-in-first-out scheduling policy."""

    def select_requests(
        self,
        pending: List[InferenceRequest],
        running: List[InferenceRequest],
        max_batch_size: int,
        max_tokens: int
    ) -> Tuple[List[InferenceRequest], List[InferenceRequest]]:
        """Select requests in FIFO order."""
        # All running decode requests
        decode = [r for r in running if r.status == RequestStatus.RUNNING_DECODE]
        decode = decode[:max_batch_size]

        remaining_slots = max_batch_size - len(decode)
        remaining_tokens = max_tokens - len(decode)

        # Select pending in order of arrival
        prefill = []
        tokens = 0
        for req in sorted(pending, key=lambda r: r.arrival_time):
            req_tokens = len(req.prompt_token_ids)
            if len(prefill) < remaining_slots and tokens + req_tokens <= remaining_tokens:
                prefill.append(req)
                tokens += req_tokens

        return prefill, decode


class PriorityPolicy(SchedulingPolicy):
    """Priority-based scheduling policy."""

    def select_requests(
        self,
        pending: List[InferenceRequest],
        running: List[InferenceRequest],
        max_batch_size: int,
        max_tokens: int
    ) -> Tuple[List[InferenceRequest], List[InferenceRequest]]:
        """Select requests by priority."""
        # All running decode requests (prioritize completion)
        decode = [r for r in running if r.status == RequestStatus.RUNNING_DECODE]
        decode = decode[:max_batch_size]

        remaining_slots = max_batch_size - len(decode)
        remaining_tokens = max_tokens - len(decode)

        # Select pending by priority (higher priority first)
        prefill = []
        tokens = 0
        for req in sorted(pending):  # Uses __lt__ for priority (higher priority is "smaller")
            req_tokens = len(req.prompt_token_ids)
            if len(prefill) < remaining_slots and tokens + req_tokens <= remaining_tokens:
                prefill.append(req)
                tokens += req_tokens

        return prefill, decode


class ShortestJobFirstPolicy(SchedulingPolicy):
    """Shortest job first scheduling policy."""

    def select_requests(
        self,
        pending: List[InferenceRequest],
        running: List[InferenceRequest],
        max_batch_size: int,
        max_tokens: int
    ) -> Tuple[List[InferenceRequest], List[InferenceRequest]]:
        """Select shortest requests first."""
        decode = [r for r in running if r.status == RequestStatus.RUNNING_DECODE]
        decode = decode[:max_batch_size]

        remaining_slots = max_batch_size - len(decode)
        remaining_tokens = max_tokens - len(decode)

        # Select by shortest prompt length
        prefill = []
        tokens = 0
        for req in sorted(pending, key=lambda r: len(r.prompt_token_ids)):
            req_tokens = len(req.prompt_token_ids)
            if len(prefill) < remaining_slots and tokens + req_tokens <= remaining_tokens:
                prefill.append(req)
                tokens += req_tokens

        return prefill, decode
