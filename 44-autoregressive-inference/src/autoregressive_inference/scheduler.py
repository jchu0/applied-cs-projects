"""Inference scheduler for autoregressive generation.

This module coordinates all inference components including request management,
KV cache, batching, and sampling.
"""

from typing import List, Optional, Any, Protocol, Dict
import time

try:
    from .requests import InferenceRequest, RequestManager, RequestStatus
    from .kv_cache import PagedKVCacheManager
    from .batching import ContinuousBatcher
    from .sampling import TokenSampler
    from .speculative import SpeculativeDecoder, SpeculativeStats
except ImportError:
    from requests import InferenceRequest, RequestManager, RequestStatus
    from kv_cache import PagedKVCacheManager
    from batching import ContinuousBatcher
    from sampling import TokenSampler
    from speculative import SpeculativeDecoder, SpeculativeStats


class TransformerModel(Protocol):
    """Protocol for transformer models used by the scheduler."""

    @property
    def device(self) -> str:
        """Get the device the model is on."""
        ...

    def prefill(
        self,
        input_ids: Any,
        request_id: str,
        kv_cache: PagedKVCacheManager
    ) -> Any:
        """Run prefill phase.

        Args:
            input_ids: Input token IDs.
            request_id: Request identifier.
            kv_cache: KV cache manager.

        Returns:
            Logits tensor.
        """
        ...

    def decode(
        self,
        input_ids: Any,
        request_id: str,
        kv_cache: PagedKVCacheManager
    ) -> Any:
        """Run decode phase.

        Args:
            input_ids: Input token IDs (single token).
            request_id: Request identifier.
            kv_cache: KV cache manager.

        Returns:
            Logits tensor.
        """
        ...


class InferenceScheduler:
    """
    Main scheduler coordinating all inference components.
    """

    def __init__(
        self,
        model: Optional[TransformerModel] = None,
        kv_cache_manager: Optional[PagedKVCacheManager] = None,
        request_manager: Optional[RequestManager] = None,
        batcher: Optional[ContinuousBatcher] = None,
        sampler: Optional[TokenSampler] = None,
        eos_token_id: int = 2,
        speculative_decoder: Optional[SpeculativeDecoder] = None,
        use_speculative: bool = False,
    ):
        """Initialize the inference scheduler.

        Args:
            model: The transformer model.
            kv_cache_manager: KV cache manager.
            request_manager: Request manager.
            batcher: Continuous batcher.
            sampler: Token sampler.
            eos_token_id: End of sequence token ID.
            speculative_decoder: Optional speculative decoder for faster inference.
            use_speculative: Whether to use speculative decoding when available.
        """
        self.model = model
        self.kv_cache = kv_cache_manager
        self.request_manager = request_manager or RequestManager()
        self.batcher = batcher or ContinuousBatcher()
        self.sampler = sampler or TokenSampler()
        self.eos_token_id = eos_token_id
        self.speculative_decoder = speculative_decoder
        self.use_speculative = use_speculative and speculative_decoder is not None

        self.running = False
        self._step_count = 0

        # Metrics
        self._total_prefill_tokens = 0
        self._total_decode_tokens = 0
        self._total_speculative_tokens = 0
        self._start_time: Optional[float] = None

    def add_request(self, request: InferenceRequest) -> bool:
        """Add a request to be processed.

        Args:
            request: The inference request.

        Returns:
            True if added successfully.
        """
        return self.request_manager.add_request(request)

    def step(self) -> List[InferenceRequest]:
        """Execute one scheduling step.

        Returns:
            List of completed requests.
        """
        self._step_count += 1
        completed = []

        # Get pending requests
        pending = self.request_manager.get_next_requests(
            self.batcher.max_batch_size
        )

        # Get running requests
        running = self.request_manager.get_running_requests()

        if not pending and not running:
            return completed

        # Allocate KV cache for new requests
        if self.kv_cache:
            for request in pending:
                num_blocks = (
                    len(request.prompt_token_ids) +
                    request.sampling_params.max_tokens
                ) // self.kv_cache.config.block_size + 1

                block_ids = self.kv_cache.allocate_blocks(
                    request.request_id, num_blocks
                )

                if not block_ids:
                    # Out of memory - preempt oldest request
                    self._preempt_for_memory(request)
                else:
                    request.kv_cache_block_ids = block_ids

        # Form batches
        prefill_batch, decode_batch = self.batcher.form_batches(
            pending, running
        )

        # Execute prefill
        if prefill_batch:
            self._execute_prefill(prefill_batch)

        # Execute decode
        if decode_batch:
            newly_completed = self._execute_decode(decode_batch)
            completed.extend(newly_completed)

        return completed

    def _execute_prefill(self, batch: List[InferenceRequest]) -> None:
        """Execute prefill phase for a batch.

        Args:
            batch: List of requests to prefill.
        """
        if not self.model:
            # Mock prefill for testing
            for request in batch:
                request.status = RequestStatus.RUNNING_DECODE
                request.prefill_tokens = len(request.prompt_token_ids)
                self._total_prefill_tokens += len(request.prompt_token_ids)
            return

        for request in batch:
            # Import here to avoid circular import issues
            try:
                import torch
                input_ids = torch.tensor(
                    [request.prompt_token_ids],
                    device=self.model.device
                )
            except ImportError:
                import numpy as np
                input_ids = np.array([request.prompt_token_ids])

            # Forward pass with KV cache population
            try:
                import torch
                with torch.no_grad():
                    self.model.prefill(
                        input_ids,
                        request.request_id,
                        self.kv_cache
                    )
            except ImportError:
                self.model.prefill(
                    input_ids,
                    request.request_id,
                    self.kv_cache
                )

            # Update status
            request.status = RequestStatus.RUNNING_DECODE
            request.prefill_tokens = len(request.prompt_token_ids)
            self._total_prefill_tokens += len(request.prompt_token_ids)

    def _execute_decode(self, batch: List[InferenceRequest]) -> List[InferenceRequest]:
        """Execute decode phase for a batch.

        Args:
            batch: List of requests to decode.

        Returns:
            List of completed requests.
        """
        # Use speculative decoding if enabled
        if self.use_speculative and self.speculative_decoder is not None:
            return self._execute_decode_speculative(batch)

        completed = []

        if not self.model:
            # Mock decode for testing - generate a token and complete
            for request in batch:
                # Check if already at max tokens before generating
                if len(request.output_token_ids) >= request.sampling_params.max_tokens:
                    if self.kv_cache:
                        self.kv_cache.free_blocks_for_request(request.request_id)
                    self.request_manager.complete_request(request.request_id)
                    completed.append(request)
                    continue

                # Add a mock token
                request.output_token_ids.append(100)
                request.decode_tokens += 1
                self._total_decode_tokens += 1

                # Check stopping conditions
                if self._should_stop(request, 100):
                    if self.kv_cache:
                        self.kv_cache.free_blocks_for_request(request.request_id)
                    self.request_manager.complete_request(request.request_id)
                    completed.append(request)
            return completed

        for request in batch:
            # Get last token
            if request.output_token_ids:
                input_id = request.output_token_ids[-1]
            else:
                input_id = request.prompt_token_ids[-1]

            try:
                import torch
                input_ids = torch.tensor([[input_id]], device=self.model.device)

                with torch.no_grad():
                    logits = self.model.decode(
                        input_ids,
                        request.request_id,
                        self.kv_cache
                    )

                # Sample next token
                next_token = self.sampler.sample(
                    logits[:, -1, :],
                    request.sampling_params,
                    request.prompt_token_ids + request.output_token_ids
                )

                token_id = next_token.item()
            except ImportError:
                import numpy as np
                input_ids = np.array([[input_id]])

                logits = self.model.decode(
                    input_ids,
                    request.request_id,
                    self.kv_cache
                )

                next_token = self.sampler.sample(
                    logits[:, -1, :],
                    request.sampling_params,
                    request.prompt_token_ids + request.output_token_ids
                )

                token_id = int(next_token)

            request.output_token_ids.append(token_id)
            request.decode_tokens += 1
            self._total_decode_tokens += 1

            # Check stopping conditions
            if self._should_stop(request, token_id):
                if self.kv_cache:
                    self.kv_cache.free_blocks_for_request(request.request_id)
                self.request_manager.complete_request(request.request_id)
                completed.append(request)

        return completed

    def _execute_decode_speculative(
        self, batch: List[InferenceRequest]
    ) -> List[InferenceRequest]:
        """Execute decode phase using speculative decoding.

        Uses a draft model to speculate multiple tokens, then verifies with
        the target model. This can significantly speed up inference when
        the acceptance rate is high.

        Args:
            batch: List of requests to decode.

        Returns:
            List of completed requests.
        """
        completed = []

        for request in batch:
            # Check if already at max tokens before generating
            if len(request.output_token_ids) >= request.sampling_params.max_tokens:
                if self.kv_cache:
                    self.kv_cache.free_blocks_for_request(request.request_id)
                self.request_manager.complete_request(request.request_id)
                completed.append(request)
                continue

            # Get current context
            context = request.prompt_token_ids + request.output_token_ids

            # Run speculative decode step
            accepted_tokens = self.speculative_decoder.decode_step(
                context=context,
                sampling_params=request.sampling_params,
                request_id=request.request_id,
            )

            # Add accepted tokens to output
            for token_id in accepted_tokens:
                request.output_token_ids.append(token_id)
                request.decode_tokens += 1
                self._total_decode_tokens += 1
                self._total_speculative_tokens += 1

                # Check stopping conditions for each token
                if self._should_stop(request, token_id):
                    if self.kv_cache:
                        self.kv_cache.free_blocks_for_request(request.request_id)
                    self.request_manager.complete_request(request.request_id)
                    completed.append(request)
                    break

        return completed

    def _should_stop(self, request: InferenceRequest, token_id: int) -> bool:
        """Check if generation should stop.

        Args:
            request: The inference request.
            token_id: The generated token ID.

        Returns:
            True if generation should stop.
        """
        # Max tokens
        if len(request.output_token_ids) >= request.sampling_params.max_tokens:
            return True

        # EOS token
        if token_id == self.eos_token_id:
            return True

        # Stop sequences would be checked here
        # This would require a tokenizer to decode and check

        return False

    def _preempt_for_memory(self, new_request: InferenceRequest) -> bool:
        """Preempt a request to free memory.

        Args:
            new_request: The new request that needs memory.

        Returns:
            True if a request was preempted.
        """
        # Simple strategy: preempt oldest request
        running = self.request_manager.get_running_requests()
        if running:
            oldest = min(running, key=lambda r: r.arrival_time)
            if self.kv_cache:
                self.kv_cache.free_blocks_for_request(oldest.request_id)
            self.request_manager.preempt_request(oldest.request_id)
            return True
        return False

    def run(self, max_steps: Optional[int] = None) -> None:
        """Main loop.

        Args:
            max_steps: Maximum number of steps to run (None for infinite).
        """
        self.running = True
        self._start_time = time.time()
        step = 0

        while self.running:
            completed = self.step()

            if max_steps is not None:
                step += 1
                if step >= max_steps:
                    break

            # Check if there's nothing to do
            stats = self.request_manager.get_stats()
            if stats['pending'] == 0 and stats['running'] == 0:
                break

    def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False

    def get_step_count(self) -> int:
        """Get the current step count."""
        return self._step_count

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary of statistics.
        """
        request_stats = self.request_manager.get_stats()
        elapsed = time.time() - self._start_time if self._start_time else 0

        stats = {
            **request_stats,
            'step_count': self._step_count,
            'total_prefill_tokens': self._total_prefill_tokens,
            'total_decode_tokens': self._total_decode_tokens,
            'elapsed_time': elapsed,
            'tokens_per_second': (
                (self._total_prefill_tokens + self._total_decode_tokens) / elapsed
                if elapsed > 0 else 0
            ),
            'speculative_enabled': self.use_speculative,
            'speculative_tokens': self._total_speculative_tokens,
        }

        # Add speculative stats if available
        if self.speculative_decoder is not None:
            spec_stats = self.speculative_decoder.stats
            stats['speculative_acceptance_rate'] = spec_stats.acceptance_rate
            stats['speculative_total_drafted'] = spec_stats.total_drafted
            stats['speculative_total_accepted'] = spec_stats.total_accepted

        return stats

    def wait_for_completion(
        self,
        request_id: str,
        timeout: Optional[float] = None
    ) -> Optional[InferenceRequest]:
        """Wait for a specific request to complete.

        Args:
            request_id: The request ID to wait for.
            timeout: Maximum time to wait in seconds.

        Returns:
            The completed request, or None if timeout.
        """
        start = time.time()

        while True:
            request = self.request_manager.get_request(request_id)
            if request and request.status == RequestStatus.COMPLETED:
                return request

            if timeout and (time.time() - start) > timeout:
                return None

            # Do a step
            self.step()

            # Small sleep to avoid busy waiting
            time.sleep(0.001)
