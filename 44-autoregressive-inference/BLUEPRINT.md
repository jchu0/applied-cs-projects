# Project 43: Autoregressive Inference Engine (Non-vLLM)

## Executive Summary

A high-performance LLM inference engine implementing continuous batching, KV cache management, and speculative decoding from first principles. This project focuses on understanding and implementing the core mechanisms that enable efficient autoregressive text generation at scale, independent of existing frameworks like vLLM.

> **Concepts covered:** [§04 vLLM (continuous batching, paged KV cache)](../../04-ai-engineering/04-llm-inference/vllm/vllm.md) — this project *implements* the mechanisms vLLM provides · [§04 TGI](../../04-ai-engineering/04-llm-inference/tgi/tgi.md) · [§04 LLM serving at scale](../../04-ai-engineering/04-llm-inference/serving-at-scale/llm-serving-at-scale.md). Pairs with [Project 22 (long-context attention kernels)](../22-long-context-attention/), [Project 41 (quantization)](../41-vector-quantized-llm/), and [Project 39 (GPU memory manager)](../39-gpu-memory-manager/) for the KV-cache substrate. Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## Architecture Overview

### System Design

```
+------------------------------------------------------------------+
|                  Autoregressive Inference Engine                  |
+------------------------------------------------------------------+
|                                                                    |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Request Manager   |     | Continuous        |     | Token     | |
|  | (Queue/Priority)  |---->| Batcher           |---->| Generator | |
|  +-------------------+     +-------------------+     +-----------+ |
|         |                          |                       |       |
|         v                          v                       v       |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Scheduler         |     | KV Cache          |     | Sampler   | |
|  | (Prefill/Decode)  |     | Manager           |     | (Top-k/p) | |
|  +-------------------+     +-------------------+     +-----------+ |
|                                    |                               |
|  +----------------------------------------------------------+     |
|  |                    Execution Engine                       |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  |  | Prefill|  | Decode |  | Attn   |  | MLP    |           |     |
|  |  | Kernel |  | Kernel |  | Kernel |  | Kernel |           |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  +----------------------------------------------------------+     |
+------------------------------------------------------------------+
```

### Core Components

#### 1. Request Management

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time
import heapq
import threading
from collections import deque

class RequestStatus(Enum):
    PENDING = "pending"
    RUNNING_PREFILL = "running_prefill"
    RUNNING_DECODE = "running_decode"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"

class RequestPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class SamplingParams:
    """Parameters for token sampling."""
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    max_tokens: int = 256
    stop_sequences: List[str] = field(default_factory=list)
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

@dataclass
class InferenceRequest:
    """A single inference request."""
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    priority: RequestPriority = RequestPriority.NORMAL

    # Runtime state
    status: RequestStatus = RequestStatus.PENDING
    output_token_ids: List[int] = field(default_factory=list)

    # Timing
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # KV cache info
    kv_cache_block_ids: List[int] = field(default_factory=list)

    # Metrics
    prefill_tokens: int = 0
    decode_tokens: int = 0

    def __lt__(self, other):
        # For priority queue ordering
        return (self.priority.value, -self.arrival_time) > \
               (other.priority.value, -other.arrival_time)


class RequestManager:
    """Manage incoming requests with priority scheduling."""

    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.pending_queue: List[InferenceRequest] = []  # Min-heap
        self.running_requests: Dict[str, InferenceRequest] = {}
        self.completed_requests: Dict[str, InferenceRequest] = {}
        self._lock = threading.Lock()

        # Metrics
        self.total_requests = 0
        self.rejected_requests = 0

    def add_request(self, request: InferenceRequest) -> bool:
        """Add a new request to the queue."""
        with self._lock:
            if len(self.pending_queue) >= self.max_queue_size:
                self.rejected_requests += 1
                return False

            heapq.heappush(self.pending_queue, request)
            self.total_requests += 1
            return True

    def get_next_requests(self, max_requests: int) -> List[InferenceRequest]:
        """Get next batch of requests to process."""
        with self._lock:
            requests = []
            while self.pending_queue and len(requests) < max_requests:
                request = heapq.heappop(self.pending_queue)
                request.status = RequestStatus.RUNNING_PREFILL
                request.start_time = time.time()
                self.running_requests[request.request_id] = request
                requests.append(request)
            return requests

    def complete_request(self, request_id: str) -> None:
        """Mark a request as completed."""
        with self._lock:
            if request_id in self.running_requests:
                request = self.running_requests.pop(request_id)
                request.status = RequestStatus.COMPLETED
                request.end_time = time.time()
                self.completed_requests[request_id] = request

    def preempt_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Preempt a running request back to queue."""
        with self._lock:
            if request_id in self.running_requests:
                request = self.running_requests.pop(request_id)
                request.status = RequestStatus.PREEMPTED
                heapq.heappush(self.pending_queue, request)
                return request
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                'pending': len(self.pending_queue),
                'running': len(self.running_requests),
                'completed': len(self.completed_requests),
                'total': self.total_requests,
                'rejected': self.rejected_requests
            }
```

#### 2. KV Cache Management

```python
import torch
import numpy as np
from typing import Tuple

@dataclass
class KVCacheConfig:
    """Configuration for KV cache."""
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    block_size: int = 16  # Tokens per block
    dtype: torch.dtype = torch.float16
    device: str = 'cuda'

class KVCacheBlock:
    """A single block of KV cache memory."""

    def __init__(self, block_id: int, config: KVCacheConfig):
        self.block_id = block_id
        self.config = config

        # Allocate memory: [num_layers, 2, num_heads, block_size, head_dim]
        # 2 is for K and V
        self.data = torch.zeros(
            config.num_layers,
            2,  # K, V
            config.num_heads,
            config.block_size,
            config.head_dim,
            dtype=config.dtype,
            device=config.device
        )

        self.num_tokens = 0
        self.ref_count = 0  # For copy-on-write

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> int:
        """Append KV to block at specific layer."""
        if self.num_tokens >= self.config.block_size:
            raise RuntimeError("Block is full")

        pos = self.num_tokens
        self.data[layer_idx, 0, :, pos, :] = k
        self.data[layer_idx, 1, :, pos, :] = v

        if layer_idx == self.config.num_layers - 1:
            self.num_tokens += 1

        return pos

    def get_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get K and V tensors for a layer."""
        return (
            self.data[layer_idx, 0, :, :self.num_tokens, :],
            self.data[layer_idx, 1, :, :self.num_tokens, :]
        )

    def is_full(self) -> bool:
        return self.num_tokens >= self.config.block_size


class PagedKVCacheManager:
    """
    Paged KV cache manager for efficient memory utilization.

    Implements virtual memory-style paging for KV cache,
    allowing non-contiguous allocation and copy-on-write.
    """

    def __init__(self, config: KVCacheConfig, num_blocks: int):
        self.config = config
        self.num_blocks = num_blocks

        # Pre-allocate all blocks
        self.blocks = [
            KVCacheBlock(i, config) for i in range(num_blocks)
        ]

        # Free block list
        self.free_blocks: List[int] = list(range(num_blocks))

        # Request to block mapping
        self.request_blocks: Dict[str, List[int]] = {}

        self._lock = threading.Lock()

    def allocate_blocks(self, request_id: str, num_blocks: int) -> List[int]:
        """Allocate blocks for a request."""
        with self._lock:
            if len(self.free_blocks) < num_blocks:
                return []  # Not enough memory

            allocated = []
            for _ in range(num_blocks):
                block_id = self.free_blocks.pop()
                allocated.append(block_id)
                self.blocks[block_id].ref_count = 1

            self.request_blocks[request_id] = allocated
            return allocated

    def free_blocks_for_request(self, request_id: str) -> None:
        """Free all blocks for a request."""
        with self._lock:
            if request_id not in self.request_blocks:
                return

            for block_id in self.request_blocks[request_id]:
                block = self.blocks[block_id]
                block.ref_count -= 1

                if block.ref_count == 0:
                    # Reset block
                    block.num_tokens = 0
                    block.data.zero_()
                    self.free_blocks.append(block_id)

            del self.request_blocks[request_id]

    def get_block(self, block_id: int) -> KVCacheBlock:
        """Get a block by ID."""
        return self.blocks[block_id]

    def append_to_cache(self,
                        request_id: str,
                        layer_idx: int,
                        k: torch.Tensor,
                        v: torch.Tensor) -> None:
        """Append KV to the cache for a request."""
        with self._lock:
            blocks = self.request_blocks.get(request_id, [])
            if not blocks:
                raise RuntimeError(f"No blocks for request {request_id}")

            # Find block with space
            for block_id in blocks:
                block = self.blocks[block_id]
                if not block.is_full():
                    block.append(layer_idx, k, v)
                    return

            # Need new block
            if not self.free_blocks:
                raise RuntimeError("Out of KV cache memory")

            new_block_id = self.free_blocks.pop()
            self.blocks[new_block_id].ref_count = 1
            blocks.append(new_block_id)
            self.blocks[new_block_id].append(layer_idx, k, v)

    def get_kv_for_request(self,
                           request_id: str,
                           layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get concatenated KV cache for a request at a layer."""
        blocks = self.request_blocks.get(request_id, [])

        if not blocks:
            return None, None

        k_list = []
        v_list = []

        for block_id in blocks:
            k, v = self.blocks[block_id].get_kv(layer_idx)
            if k.shape[1] > 0:  # Has tokens
                k_list.append(k)
                v_list.append(v)

        if not k_list:
            return None, None

        return torch.cat(k_list, dim=1), torch.cat(v_list, dim=1)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        used_blocks = self.num_blocks - len(self.free_blocks)
        block_size_bytes = (
            self.config.num_layers * 2 * self.config.num_heads *
            self.config.block_size * self.config.head_dim * 2  # float16
        )

        return {
            'total_blocks': self.num_blocks,
            'used_blocks': used_blocks,
            'free_blocks': len(self.free_blocks),
            'utilization': used_blocks / self.num_blocks,
            'memory_used_gb': used_blocks * block_size_bytes / 1e9
        }


class SlidingWindowCache:
    """KV cache with sliding window for long sequences."""

    def __init__(self, config: KVCacheConfig, window_size: int):
        self.config = config
        self.window_size = window_size

        # Allocate window buffer
        self.k_cache = torch.zeros(
            config.num_layers,
            config.num_heads,
            window_size,
            config.head_dim,
            dtype=config.dtype,
            device=config.device
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        self.position = 0  # Current write position (circular)
        self.length = 0    # Actual cached length

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append to sliding window cache."""
        pos = self.position % self.window_size
        self.k_cache[layer_idx, :, pos, :] = k
        self.v_cache[layer_idx, :, pos, :] = v

        if layer_idx == self.config.num_layers - 1:
            self.position += 1
            self.length = min(self.length + 1, self.window_size)

    def get_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV in correct order."""
        if self.length < self.window_size:
            return (
                self.k_cache[layer_idx, :, :self.length, :],
                self.v_cache[layer_idx, :, :self.length, :]
            )

        # Need to reorder for circular buffer
        start = self.position % self.window_size
        indices = torch.cat([
            torch.arange(start, self.window_size),
            torch.arange(0, start)
        ])

        return (
            self.k_cache[layer_idx, :, indices, :],
            self.v_cache[layer_idx, :, indices, :]
        )
```

#### 3. Continuous Batching

```python
class ContinuousBatcher:
    """
    Continuous batching engine for maximizing GPU utilization.

    Key features:
    - Dynamic batch formation
    - Split prefill and decode phases
    - In-flight batching (add/remove requests)
    """

    def __init__(self,
                 max_batch_size: int = 64,
                 max_prefill_tokens: int = 4096,
                 max_decode_tokens: int = 2048):
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.max_decode_tokens = max_decode_tokens

        # Current batches
        self.prefill_batch: List[InferenceRequest] = []
        self.decode_batch: List[InferenceRequest] = []

    def form_batches(self,
                     pending_requests: List[InferenceRequest],
                     running_requests: List[InferenceRequest]
                     ) -> Tuple[List[InferenceRequest], List[InferenceRequest]]:
        """
        Form prefill and decode batches.

        Returns:
            (prefill_batch, decode_batch)
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

    def merge_for_execution(self,
                            prefill_batch: List[InferenceRequest],
                            decode_batch: List[InferenceRequest]
                            ) -> 'BatchedInputs':
        """Merge prefill and decode into single execution batch."""
        return BatchedInputs(
            prefill_requests=prefill_batch,
            decode_requests=decode_batch
        )


@dataclass
class BatchedInputs:
    """Batched inputs for model execution."""
    prefill_requests: List[InferenceRequest]
    decode_requests: List[InferenceRequest]

    def get_input_ids(self) -> torch.Tensor:
        """Get padded input token IDs."""
        all_ids = []

        # Prefill: full prompt
        for req in self.prefill_requests:
            all_ids.append(torch.tensor(req.prompt_token_ids))

        # Decode: just last token
        for req in self.decode_requests:
            if req.output_token_ids:
                all_ids.append(torch.tensor([req.output_token_ids[-1]]))
            else:
                # First decode token
                all_ids.append(torch.tensor([req.prompt_token_ids[-1]]))

        # Pad to same length
        max_len = max(len(ids) for ids in all_ids)
        padded = torch.zeros(len(all_ids), max_len, dtype=torch.long)

        for i, ids in enumerate(all_ids):
            padded[i, :len(ids)] = ids

        return padded

    def get_position_ids(self) -> torch.Tensor:
        """Get position IDs for each token."""
        positions = []

        for req in self.prefill_requests:
            seq_len = len(req.prompt_token_ids)
            positions.append(torch.arange(seq_len))

        for req in self.decode_requests:
            # Position is total length so far
            pos = len(req.prompt_token_ids) + len(req.output_token_ids)
            positions.append(torch.tensor([pos]))

        # Pad
        max_len = max(len(p) for p in positions)
        padded = torch.zeros(len(positions), max_len, dtype=torch.long)

        for i, pos in enumerate(positions):
            padded[i, :len(pos)] = pos

        return padded

    @property
    def batch_size(self) -> int:
        return len(self.prefill_requests) + len(self.decode_requests)
```

#### 4. Token Sampling

```python
import torch.nn.functional as F

class TokenSampler:
    """Token sampling with various strategies."""

    def sample(self,
               logits: torch.Tensor,
               params: SamplingParams,
               generated_ids: Optional[List[int]] = None) -> torch.Tensor:
        """
        Sample next token from logits.

        Args:
            logits: [batch_size, vocab_size]
            params: Sampling parameters
            generated_ids: Previously generated IDs for penalties

        Returns:
            [batch_size] sampled token IDs
        """
        # Apply temperature
        if params.temperature != 1.0:
            logits = logits / params.temperature

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

        # Apply top-k filtering
        if params.top_k > 0:
            logits = self._top_k_filtering(logits, params.top_k)

        # Apply top-p (nucleus) filtering
        if params.top_p < 1.0:
            logits = self._top_p_filtering(logits, params.top_p)

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)

        if params.temperature == 0:
            # Greedy
            return torch.argmax(probs, dim=-1)
        else:
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _top_k_filtering(self,
                         logits: torch.Tensor,
                         top_k: int) -> torch.Tensor:
        """Keep only top-k logits."""
        if top_k == 0:
            return logits

        values, _ = torch.topk(logits, top_k, dim=-1)
        min_value = values[:, -1].unsqueeze(-1)

        return torch.where(
            logits < min_value,
            torch.full_like(logits, float('-inf')),
            logits
        )

    def _top_p_filtering(self,
                         logits: torch.Tensor,
                         top_p: float) -> torch.Tensor:
        """Nucleus sampling: keep tokens with cumulative prob <= top_p."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Find cutoff
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter back
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)

        logits[indices_to_remove] = float('-inf')
        return logits

    def _apply_repetition_penalty(self,
                                   logits: torch.Tensor,
                                   generated_ids: List[int],
                                   penalty: float) -> torch.Tensor:
        """Apply repetition penalty to discourage repeats."""
        for token_id in set(generated_ids):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= penalty
            else:
                logits[0, token_id] /= penalty
        return logits

    def _apply_frequency_presence_penalty(self,
                                           logits: torch.Tensor,
                                           generated_ids: List[int],
                                           frequency_penalty: float,
                                           presence_penalty: float) -> torch.Tensor:
        """Apply frequency and presence penalties."""
        from collections import Counter
        token_counts = Counter(generated_ids)

        for token_id, count in token_counts.items():
            logits[0, token_id] -= frequency_penalty * count
            logits[0, token_id] -= presence_penalty

        return logits
```

#### 5. Inference Scheduler

```python
class InferenceScheduler:
    """
    Main scheduler coordinating all inference components.
    """

    def __init__(self,
                 model: 'TransformerModel',
                 kv_cache_manager: PagedKVCacheManager,
                 request_manager: RequestManager,
                 batcher: ContinuousBatcher,
                 sampler: TokenSampler):
        self.model = model
        self.kv_cache = kv_cache_manager
        self.request_manager = request_manager
        self.batcher = batcher
        self.sampler = sampler

        self.running = False
        self._step_count = 0

    def step(self) -> List[InferenceRequest]:
        """Execute one scheduling step."""
        self._step_count += 1
        completed = []

        # Get pending requests
        pending = self.request_manager.get_next_requests(
            self.batcher.max_batch_size
        )

        # Get running requests
        running = list(self.request_manager.running_requests.values())

        if not pending and not running:
            return completed

        # Allocate KV cache for new requests
        for request in pending:
            num_blocks = (len(request.prompt_token_ids) +
                         request.sampling_params.max_tokens) // \
                         self.kv_cache.config.block_size + 1

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
        """Execute prefill phase for a batch."""
        for request in batch:
            # Get input
            input_ids = torch.tensor(
                [request.prompt_token_ids],
                device=self.model.device
            )

            # Forward pass with KV cache population
            with torch.no_grad():
                self.model.prefill(
                    input_ids,
                    request.request_id,
                    self.kv_cache
                )

            # Update status
            request.status = RequestStatus.RUNNING_DECODE
            request.prefill_tokens = len(request.prompt_token_ids)

    def _execute_decode(self, batch: List[InferenceRequest]) -> List[InferenceRequest]:
        """Execute decode phase for a batch."""
        completed = []

        for request in batch:
            # Get last token
            if request.output_token_ids:
                input_id = request.output_token_ids[-1]
            else:
                input_id = request.prompt_token_ids[-1]

            input_ids = torch.tensor([[input_id]], device=self.model.device)

            # Forward pass with KV cache
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
            request.output_token_ids.append(token_id)
            request.decode_tokens += 1

            # Check stopping conditions
            if self._should_stop(request, token_id):
                self.kv_cache.free_blocks_for_request(request.request_id)
                self.request_manager.complete_request(request.request_id)
                completed.append(request)

        return completed

    def _should_stop(self, request: InferenceRequest, token_id: int) -> bool:
        """Check if generation should stop."""
        # Max tokens
        if len(request.output_token_ids) >= request.sampling_params.max_tokens:
            return True

        # EOS token (assuming ID 2)
        if token_id == 2:
            return True

        # Stop sequences would be checked here

        return False

    def _preempt_for_memory(self, new_request: InferenceRequest) -> None:
        """Preempt a request to free memory."""
        # Simple strategy: preempt oldest request
        running = list(self.request_manager.running_requests.values())
        if running:
            oldest = min(running, key=lambda r: r.arrival_time)
            self.kv_cache.free_blocks_for_request(oldest.request_id)
            self.request_manager.preempt_request(oldest.request_id)

    def run(self) -> None:
        """Main loop."""
        self.running = True
        while self.running:
            completed = self.step()
            for request in completed:
                # Callback or notification
                pass

    def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False
```

### Enterprise Features

#### Multi-GPU Tensor Parallelism

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

class TensorParallelScheduler(InferenceScheduler):
    """Scheduler with tensor parallelism across GPUs."""

    def __init__(self,
                 model: 'TransformerModel',
                 kv_cache_manager: PagedKVCacheManager,
                 world_size: int,
                 rank: int):
        super().__init__(model, kv_cache_manager, None, None, None)

        self.world_size = world_size
        self.rank = rank

        # Initialize distributed
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        # Shard model across GPUs
        self._shard_model()

    def _shard_model(self) -> None:
        """Shard attention heads and FFN across GPUs."""
        # Attention: split heads
        for layer in self.model.layers:
            layer.attention.num_heads_per_gpu = \
                layer.attention.num_heads // self.world_size

            # Shard Q, K, V projections
            # ...

        # FFN: split intermediate dimension
        # ...

    def _all_reduce_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """All-reduce logits across GPUs."""
        dist.all_reduce(logits, op=dist.ReduceOp.SUM)
        return logits


class DistributedKVStore:
    """Distributed KV cache across multiple nodes."""

    def __init__(self, nodes: List[str], local_cache: PagedKVCacheManager):
        self.nodes = nodes
        self.local_cache = local_cache
        self.request_to_node: Dict[str, str] = {}

    def get_or_fetch(self,
                     request_id: str,
                     layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV from local cache or fetch from remote."""
        # Check local
        k, v = self.local_cache.get_kv_for_request(request_id, layer_idx)
        if k is not None:
            return k, v

        # Fetch from remote node
        node = self.request_to_node.get(request_id)
        if node:
            return self._fetch_remote(node, request_id, layer_idx)

        return None, None

    def _fetch_remote(self,
                      node: str,
                      request_id: str,
                      layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch KV cache from remote node."""
        # Implement RPC call
        pass


class AutoScaler:
    """Auto-scale inference replicas based on load."""

    def __init__(self,
                 min_replicas: int = 1,
                 max_replicas: int = 8,
                 target_qps: float = 100):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_qps = target_qps
        self.current_replicas = min_replicas

        # Metrics
        self.qps_history: List[float] = []

    def update_metrics(self, current_qps: float) -> None:
        """Update metrics and potentially scale."""
        self.qps_history.append(current_qps)

        # Keep last 60 samples (1 minute at 1 sample/sec)
        if len(self.qps_history) > 60:
            self.qps_history.pop(0)

        avg_qps = sum(self.qps_history) / len(self.qps_history)

        # Scale up
        if avg_qps > self.target_qps * 0.8:
            new_replicas = min(self.current_replicas + 1, self.max_replicas)
        # Scale down
        elif avg_qps < self.target_qps * 0.3:
            new_replicas = max(self.current_replicas - 1, self.min_replicas)
        else:
            new_replicas = self.current_replicas

        if new_replicas != self.current_replicas:
            self._scale(new_replicas)

    def _scale(self, target_replicas: int) -> None:
        """Scale to target number of replicas."""
        # Implement scaling logic
        self.current_replicas = target_replicas
```

#### Speculative Decoding

```python
class SpeculativeDecoder:
    """
    Speculative decoding with small draft model.

    Uses a smaller model to draft multiple tokens,
    then verifies with the large model in parallel.
    """

    def __init__(self,
                 target_model: 'TransformerModel',
                 draft_model: 'TransformerModel',
                 num_speculative_tokens: int = 4):
        self.target_model = target_model
        self.draft_model = draft_model
        self.num_speculative = num_speculative_tokens

        # Acceptance stats
        self.total_drafted = 0
        self.total_accepted = 0

    def generate_step(self,
                      input_ids: torch.Tensor,
                      kv_cache: PagedKVCacheManager,
                      request_id: str,
                      sampling_params: SamplingParams) -> List[int]:
        """
        Generate tokens using speculative decoding.

        Returns list of accepted tokens.
        """
        # Step 1: Draft tokens with small model
        draft_tokens = self._draft(input_ids, request_id, sampling_params)

        # Step 2: Verify with target model (parallel forward pass)
        accepted = self._verify(
            input_ids, draft_tokens, kv_cache, request_id, sampling_params
        )

        # Update stats
        self.total_drafted += len(draft_tokens)
        self.total_accepted += len(accepted)

        return accepted

    def _draft(self,
               input_ids: torch.Tensor,
               request_id: str,
               sampling_params: SamplingParams) -> List[int]:
        """Generate draft tokens with small model."""
        draft_tokens = []
        current_ids = input_ids

        for _ in range(self.num_speculative):
            with torch.no_grad():
                logits = self.draft_model.forward(current_ids)

            # Sample greedily for drafting
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            draft_tokens.append(next_token.item())

            current_ids = torch.cat([
                current_ids,
                next_token.unsqueeze(0)
            ], dim=-1)

        return draft_tokens

    def _verify(self,
                input_ids: torch.Tensor,
                draft_tokens: List[int],
                kv_cache: PagedKVCacheManager,
                request_id: str,
                sampling_params: SamplingParams) -> List[int]:
        """Verify draft tokens with target model."""
        # Concatenate input with draft tokens
        draft_ids = torch.tensor([draft_tokens], device=input_ids.device)
        full_ids = torch.cat([input_ids, draft_ids], dim=-1)

        # Forward pass (gets logits for all positions)
        with torch.no_grad():
            logits = self.target_model.forward_with_kv(
                full_ids, kv_cache, request_id
            )

        # Verify each draft token
        accepted = []
        sampler = TokenSampler()

        for i, draft_token in enumerate(draft_tokens):
            pos = input_ids.shape[1] + i - 1
            token_logits = logits[:, pos:pos+1, :]

            # Sample from target distribution
            target_token = sampler.sample(
                token_logits.squeeze(1),
                sampling_params,
                None
            )

            if target_token.item() == draft_token:
                accepted.append(draft_token)
            else:
                # Rejection - use target token and stop
                accepted.append(target_token.item())
                break

        return accepted

    def get_acceptance_rate(self) -> float:
        """Get average acceptance rate."""
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted
```

## API Reference

### Create Inference Engine

```python
# Initialize components
config = KVCacheConfig(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    max_seq_len=4096,
    block_size=16
)

kv_cache = PagedKVCacheManager(config, num_blocks=1000)
request_manager = RequestManager(max_queue_size=1000)
batcher = ContinuousBatcher(max_batch_size=64)
sampler = TokenSampler()

scheduler = InferenceScheduler(
    model=model,
    kv_cache_manager=kv_cache,
    request_manager=request_manager,
    batcher=batcher,
    sampler=sampler
)
```

### Submit Request

```python
request = InferenceRequest(
    request_id="req-001",
    prompt="Once upon a time",
    prompt_token_ids=[1, 432, 567, 89, 234],
    sampling_params=SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100
    ),
    priority=RequestPriority.HIGH
)

request_manager.add_request(request)
```

### Run Inference Loop

```python
while True:
    completed = scheduler.step()
    for request in completed:
        output = tokenizer.decode(request.output_token_ids)
        print(f"{request.request_id}: {output}")
```

## Implementation Phases

### Phase 1: Basic Generation (Weeks 1-2)
- Request data structures
- Simple token-by-token generation
- Basic KV cache (non-paged)
- Greedy sampling

### Phase 2: KV Cache Management (Weeks 3-4)
- Paged KV cache
- Block allocation/deallocation
- Memory tracking
- Sliding window cache

### Phase 3: Continuous Batching (Weeks 5-6)
- Prefill/decode separation
- Dynamic batch formation
- Request prioritization
- In-flight batching

### Phase 4: Advanced Sampling (Week 7)
- Top-k, top-p sampling
- Temperature scaling
- Repetition penalties
- Stop sequence detection

### Phase 5: Optimization (Weeks 8-10)
- Fused attention kernels
- Memory layout optimization
- CUDA graph compilation
- Quantized inference

### Phase 6: Enterprise Features (Weeks 11-14)
- Multi-GPU tensor parallelism
- Speculative decoding
- Distributed KV store
- Auto-scaling

## Testing Strategy

### Unit Tests

```python
class TestKVCache:
    def test_block_allocation(self):
        config = KVCacheConfig(
            num_layers=2, num_heads=4, head_dim=64,
            max_seq_len=256, block_size=16
        )
        manager = PagedKVCacheManager(config, num_blocks=10)

        blocks = manager.allocate_blocks("req-1", 3)
        assert len(blocks) == 3

        stats = manager.get_memory_usage()
        assert stats['used_blocks'] == 3

    def test_paged_append(self):
        # Test appending across block boundaries
        pass


class TestContinuousBatching:
    def test_batch_formation(self):
        batcher = ContinuousBatcher(max_batch_size=8)

        pending = [
            InferenceRequest("r1", "", [1]*100, SamplingParams()),
            InferenceRequest("r2", "", [1]*50, SamplingParams()),
        ]
        running = []

        prefill, decode = batcher.form_batches(pending, running)
        assert len(prefill) == 2
        assert len(decode) == 0


class TestSampling:
    def test_top_k(self):
        sampler = TokenSampler()
        logits = torch.randn(1, 1000)

        params = SamplingParams(top_k=10, temperature=1.0)
        token = sampler.sample(logits, params)

        assert 0 <= token.item() < 1000
```

### Performance Benchmarks

```python
class TestPerformance:
    def test_throughput(self):
        """Measure tokens per second."""
        # Target: >1000 tokens/sec on single GPU
        pass

    def test_latency(self):
        """Measure time to first token."""
        # Target: <100ms for short prompts
        pass

    def test_kv_cache_efficiency(self):
        """Measure memory utilization."""
        # Target: >90% utilization
        pass
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | >1000 tokens/sec | Single A100 |
| TTFT | <100ms | 100-token prompt |
| KV utilization | >90% | With paging |
| Batch utilization | >80% | Continuous batching |
| Speculative acceptance | >70% | Draft model quality |

## Dependencies

- PyTorch >= 2.0
- CUDA >= 11.0
- Transformers (for model loading)
- (Optional) Flash Attention
- (Optional) vLLM for comparison

## References

- Orca: A Distributed Serving System for Transformer-Based Models
- vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention
- Fast Inference from Transformers via Speculative Decoding
- Continuous Batching in LLM Inference
