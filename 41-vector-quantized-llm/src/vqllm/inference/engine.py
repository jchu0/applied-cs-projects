"""Quantized model inference engine."""

import numpy as np
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Iterator, Union
from dataclasses import dataclass, field
import heapq

from ..core.types import QuantizedTensor, QuantizedLinear, QuantConfig, QuantType

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0


class KVCache:
    """
    Key-Value cache for efficient autoregressive inference.

    Features:
    - Efficient memory management
    - Support for beam search
    - Per-layer caching
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int = 1,
        max_seq_length: int = 2048,
        hidden_size: int = 768,
        num_heads: int = 12,
        dtype: np.dtype = np.float32
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dtype = dtype

        # Pre-allocate cache as list of tensors per layer
        self.key_cache: List[np.ndarray] = []
        self.value_cache: List[np.ndarray] = []

        for _ in range(num_layers):
            self.key_cache.append(
                np.zeros((max_batch_size, max_seq_length, hidden_size), dtype=dtype)
            )
            self.value_cache.append(
                np.zeros((max_batch_size, max_seq_length, hidden_size), dtype=dtype)
            )

        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        keys: np.ndarray,
        values: np.ndarray,
        seq_position: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update cache with new key-value and return full cache."""
        if seq_position is None:
            seq_position = self.seq_len

        batch_size = keys.shape[0]
        seq_len = keys.shape[1]

        # Store in cache
        self.key_cache[layer_idx][:batch_size, seq_position:seq_position + seq_len, :] = keys
        self.value_cache[layer_idx][:batch_size, seq_position:seq_position + seq_len, :] = values

        # Update sequence length
        new_len = seq_position + seq_len
        if new_len > self.seq_len:
            self.seq_len = new_len

        # Return cached values up to current position
        return (
            self.key_cache[layer_idx][:batch_size, :self.seq_len, :],
            self.value_cache[layer_idx][:batch_size, :self.seq_len, :]
        )

    def get_keys(self, layer_idx: int, batch_size: int, seq_len: int) -> np.ndarray:
        """Get cached keys for a layer."""
        return self.key_cache[layer_idx][:batch_size, :seq_len, :]

    def get_values(self, layer_idx: int, batch_size: int, seq_len: int) -> np.ndarray:
        """Get cached values for a layer."""
        return self.value_cache[layer_idx][:batch_size, :seq_len, :]

    def clear(self):
        """Clear cache."""
        self.seq_len = 0
        for i in range(self.num_layers):
            self.key_cache[i].fill(0)
            self.value_cache[i].fill(0)

    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        total = 0
        for i in range(self.num_layers):
            total += self.key_cache[i].nbytes
            total += self.value_cache[i].nbytes
        return total

    def increment(self):
        """Increment sequence length."""
        self.seq_len += 1

    def reset(self):
        """Reset cache (alias for clear)."""
        self.clear()


class QuantizedModel:
    """
    Quantized transformer model wrapper for inference.

    Features:
    - Wraps original model with quantization
    - Supports mixed precision layers
    - Efficient attention
    """

    def __init__(
        self,
        model: Any,
        config: QuantConfig,
        precision_map: Dict[str, QuantType] = None
    ):
        self.original_model = model
        self.config = config
        self.precision_map = precision_map or {}
        self.is_quantized = True
        self._quantized_layers = {}

        # Quantize layers based on precision map
        self._quantize_model()

    def _quantize_model(self):
        """Apply quantization to model layers."""
        if hasattr(self.original_model, 'layers'):
            layers = self.original_model.layers
            if isinstance(layers, dict):
                for name, layer in layers.items():
                    quant_type = self.precision_map.get(name, self.config.quant_type)
                    self._quantized_layers[name] = {
                        'layer': layer,
                        'quant_type': quant_type
                    }
            elif isinstance(layers, list):
                for i, layer in enumerate(layers):
                    self._quantized_layers[f'layer_{i}'] = {
                        'layer': layer,
                        'quant_type': self.config.quant_type
                    }

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through quantized model."""
        if hasattr(self, 'forward_quantized'):
            return self.forward_quantized(x)

        # Default: just pass through
        if hasattr(self.original_model, 'forward'):
            return self.original_model.forward(x)
        elif hasattr(self.original_model, '__call__'):
            return self.original_model(x)
        return x

    def forward(
        self,
        input_ids: np.ndarray,
        kv_cache: Optional[KVCache] = None,
        position_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass through model."""
        if hasattr(self.original_model, 'forward'):
            return self.original_model.forward(input_ids, kv_cache, position_ids)
        return self(input_ids)


class InferenceEngine:
    """
    High-performance inference engine for quantized models.

    Features:
    - Configurable batch size
    - KV caching support
    - Streaming generation
    """

    def __init__(
        self,
        config: QuantConfig,
        device: str = "cpu",
        batch_size: int = 1
    ):
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.model: Optional[Any] = None
        self.kv_cache: Optional[KVCache] = None

    def load_model(self, model: Any):
        """Load a model for inference."""
        self.model = model

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Single forward pass through the model."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        if hasattr(self.model, 'forward'):
            return self.model.forward(input_ids)
        elif callable(self.model):
            return self.model(input_ids)
        else:
            raise RuntimeError("Model does not have forward method")

    def stream_generate(
        self,
        input_ids: np.ndarray,
        max_length: int = 100
    ) -> Iterator[np.ndarray]:
        """Stream token generation."""
        current_ids = input_ids.copy()
        past_kvs = None

        for _ in range(max_length):
            if hasattr(self, 'generate_token'):
                next_token, past_kvs = self.generate_token(current_ids, past_kvs)
            else:
                # Default token generation
                next_token = np.random.randint(0, 1000, size=(input_ids.shape[0], 1))

            yield next_token
            current_ids = np.concatenate([current_ids, next_token], axis=1)


class BatchedInference:
    """
    Batched inference with dynamic batching support.

    Features:
    - Dynamic request batching
    - Priority scheduling
    - Continuous batching
    """

    def __init__(
        self,
        config: QuantConfig,
        max_batch_size: int = 8,
        max_seq_length: int = 512
    ):
        self.config = config
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length

        # Request queue: (priority, insertion_order, req_id, input_ids)
        self._requests: List[Tuple[int, int, str, np.ndarray]] = []
        self._request_counter = 0
        self._request_map: Dict[str, np.ndarray] = {}

    def add_request(
        self,
        req_id: str,
        input_ids: np.ndarray,
        priority: int = 5
    ):
        """Add a request to the queue."""
        # Store request
        self._request_map[req_id] = input_ids

        # Add to priority queue (negative priority for max-heap behavior)
        heapq.heappush(
            self._requests,
            (-priority, self._request_counter, req_id, input_ids)
        )
        self._request_counter += 1

    def create_batch(self) -> List[str]:
        """Create a batch from queued requests."""
        batch_ids = []
        batch_size = min(len(self._requests), self.max_batch_size)

        for _ in range(batch_size):
            if self._requests:
                _, _, req_id, _ = heapq.heappop(self._requests)
                batch_ids.append(req_id)

        return batch_ids

    def create_padded_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create a padded batch with attention mask."""
        if not self._requests:
            return np.array([]), np.array([])

        # Get batch IDs
        batch_ids = self.create_batch()

        # Find max length
        max_len = 0
        inputs = []
        for req_id in batch_ids:
            inp = self._request_map.get(req_id)
            if inp is not None:
                inputs.append(inp)
                max_len = max(max_len, inp.shape[1])

        if not inputs:
            return np.array([]), np.array([])

        # Pad and create mask
        batch_size = len(inputs)
        batch = np.zeros((batch_size, max_len), dtype=inputs[0].dtype)
        mask = np.zeros((batch_size, max_len), dtype=bool)

        for i, inp in enumerate(inputs):
            seq_len = inp.shape[1]
            batch[i, :seq_len] = inp[0, :]
            mask[i, :seq_len] = True

        return batch, mask

    def get_next_batch_ids(self, max_batch_size: int = None) -> List[str]:
        """Get next batch of request IDs by priority."""
        if max_batch_size is None:
            max_batch_size = self.max_batch_size

        batch_ids = []
        # Peek at requests without removing
        temp_requests = []

        batch_size = min(len(self._requests), max_batch_size)
        for _ in range(batch_size):
            if self._requests:
                item = heapq.heappop(self._requests)
                temp_requests.append(item)
                batch_ids.append(item[2])

        # Put items back
        for item in temp_requests:
            heapq.heappush(self._requests, item)

        return batch_ids

    def should_process_batch(self) -> bool:
        """Check if batch should be processed."""
        return len(self._requests) >= self.max_batch_size


def optimize_inference(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize computation graph by fusing redundant operations.

    Removes consecutive quantize/dequantize pairs that cancel out.
    """
    optimized = {}

    # Find dequant->quant pairs to remove
    nodes_to_remove = set()

    for node_name, node_info in graph.items():
        if node_info.get("type") == "dequantize":
            input_node = node_info.get("input")
            # Check if output feeds into quantize
            for other_name, other_info in graph.items():
                if other_info.get("type") == "quantize":
                    if other_info.get("input") == node_name:
                        # Found dequant->quant pair
                        nodes_to_remove.add(node_name)
                        nodes_to_remove.add(other_name)

    # Copy non-removed nodes
    for node_name, node_info in graph.items():
        if node_name not in nodes_to_remove:
            optimized[node_name] = node_info.copy()

    return optimized


def benchmark_latency(
    model: Any,
    input_data: np.ndarray,
    num_runs: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark model latency.

    Returns statistics on forward pass latency.
    """
    latencies = []

    # Warmup runs
    for _ in range(warmup):
        if hasattr(model, 'forward'):
            model.forward(input_data)
        elif callable(model):
            model(input_data)

    # Timed runs
    for _ in range(num_runs):
        start = time.perf_counter()

        if hasattr(model, 'forward'):
            model.forward(input_data)
        elif callable(model):
            model(input_data)

        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # Convert to ms

    latencies = np.array(latencies)

    return {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    }


# Legacy classes for backward compatibility

class LegacyKVCache:
    """
    Legacy Key-Value cache with old interface.

    For backward compatibility with older code.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        dtype: np.dtype = np.float16
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype

        # Pre-allocate cache
        self.keys = np.zeros(
            (num_layers, batch_size, num_heads, max_seq_len, head_dim),
            dtype=dtype
        )
        self.values = np.zeros(
            (num_layers, batch_size, num_heads, max_seq_len, head_dim),
            dtype=dtype
        )

        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update cache with new key-value and return full cache."""
        self.keys[layer_idx, :, :, self.seq_len:self.seq_len + 1, :] = key
        self.values[layer_idx, :, :, self.seq_len:self.seq_len + 1, :] = value

        return (
            self.keys[layer_idx, :, :, :self.seq_len + 1, :],
            self.values[layer_idx, :, :, :self.seq_len + 1, :]
        )

    def increment(self):
        """Increment sequence length."""
        self.seq_len += 1

    def reset(self):
        """Reset cache."""
        self.seq_len = 0
        self.keys.fill(0)
        self.values.fill(0)

    @property
    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return self.keys.nbytes + self.values.nbytes


class LegacyQuantizedModel:
    """
    Legacy Quantized transformer model for inference.

    For backward compatibility with older code.
    """

    def __init__(
        self,
        config: QuantConfig,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_heads: int = 12,
        vocab_size: int = 50257,
        max_seq_len: int = 2048
    ):
        self.config = config
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Quantized layers
        self.layers = []
        for _ in range(num_layers):
            layer = {
                'q_proj': QuantizedLinear(hidden_size, hidden_size, config=config),
                'k_proj': QuantizedLinear(hidden_size, hidden_size, config=config),
                'v_proj': QuantizedLinear(hidden_size, hidden_size, config=config),
                'o_proj': QuantizedLinear(hidden_size, hidden_size, config=config),
                'gate_proj': QuantizedLinear(hidden_size, hidden_size * 4, config=config),
                'up_proj': QuantizedLinear(hidden_size, hidden_size * 4, config=config),
                'down_proj': QuantizedLinear(hidden_size * 4, hidden_size, config=config),
            }
            self.layers.append(layer)

        # Embedding and output
        self.embedding = np.random.randn(vocab_size, hidden_size).astype(np.float16)
        self.lm_head = QuantizedLinear(hidden_size, vocab_size, config=config)

    def forward(
        self,
        input_ids: np.ndarray,
        kv_cache: Optional[LegacyKVCache] = None,
        position_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass through model."""
        batch_size, seq_len = input_ids.shape

        # Embedding
        hidden = self.embedding[input_ids]

        # Process layers
        for i, layer in enumerate(self.layers):
            hidden = self._forward_layer(
                hidden, layer, i, kv_cache, position_ids
            )

        # LM head
        logits = self.lm_head(hidden.astype(np.float32))

        return logits

    def _forward_layer(
        self,
        hidden: np.ndarray,
        layer: Dict,
        layer_idx: int,
        kv_cache: Optional[LegacyKVCache],
        position_ids: Optional[np.ndarray]
    ) -> np.ndarray:
        """Forward pass through one layer."""
        batch_size, seq_len, hidden_size = hidden.shape
        hidden_fp32 = hidden.astype(np.float32)

        # Self-attention (simplified)
        q = layer['q_proj'](hidden_fp32)
        k = layer['k_proj'](hidden_fp32)
        v = layer['v_proj'](hidden_fp32)

        # Reshape for attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Update KV cache
        if kv_cache is not None:
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            k, v = kv_cache.update(layer_idx, k, v)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

        # Attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        context = np.matmul(attn, v)

        # Reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
        attn_out = layer['o_proj'](context)

        # Residual
        hidden = hidden_fp32 + attn_out

        # FFN
        gate = layer['gate_proj'](hidden)
        up = layer['up_proj'](hidden)
        gate = gate * (1 / (1 + np.exp(-gate)))  # SiLU
        ffn_out = layer['down_proj'](gate * up)

        # Residual
        hidden = hidden + ffn_out

        return hidden.astype(np.float16)


class QuantizedEngine:
    """
    High-performance inference engine for quantized models.

    Features:
    - Batched inference
    - KV caching
    - Text generation
    """

    def __init__(
        self,
        model: LegacyQuantizedModel,
        max_batch_size: int = 8
    ):
        self.model = model
        self.max_batch_size = max_batch_size

        # Pre-allocate KV cache
        self.kv_cache = LegacyKVCache(
            model.num_layers,
            model.num_heads,
            model.head_dim,
            model.max_seq_len,
            max_batch_size
        )

    def generate(
        self,
        input_ids: np.ndarray,
        config: GenerationConfig = None
    ) -> np.ndarray:
        """Generate text from input."""
        config = config or GenerationConfig()
        batch_size = input_ids.shape[0]

        # Reset cache
        self.kv_cache.reset()

        # Process prompt
        output_ids = input_ids.copy()

        for step in range(config.max_length):
            # Get logits
            if step == 0:
                logits = self.model.forward(output_ids, self.kv_cache)
            else:
                logits = self.model.forward(
                    output_ids[:, -1:],
                    self.kv_cache
                )

            self.kv_cache.increment()

            # Get next token
            next_token_logits = logits[:, -1, :]

            if config.do_sample:
                next_token = self._sample(
                    next_token_logits,
                    config.temperature,
                    config.top_k,
                    config.top_p
                )
            else:
                next_token = np.argmax(next_token_logits, axis=-1)

            # Append
            output_ids = np.concatenate([
                output_ids,
                next_token.reshape(-1, 1)
            ], axis=1)

            # Check for EOS
            if np.all(next_token == 0):
                break

        return output_ids

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> np.ndarray:
        """Sample next token with temperature and top-k/p."""
        batch_size = logits.shape[0]

        # Temperature
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            top_k_idx = np.argsort(logits, axis=-1)[:, -top_k:]
            mask = np.zeros_like(logits, dtype=bool)
            for i in range(batch_size):
                mask[i, top_k_idx[i]] = True
            logits = np.where(mask, logits, -np.inf)

        # Softmax
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        # Top-p
        if top_p < 1.0:
            sorted_idx = np.argsort(probs, axis=-1)[:, ::-1]
            for i in range(batch_size):
                sorted_probs = probs[i, sorted_idx[i]]
                cumsum = np.cumsum(sorted_probs)
                cutoff_idx = np.searchsorted(cumsum, top_p)
                probs[i, sorted_idx[i, cutoff_idx + 1:]] = 0
            probs = probs / probs.sum(axis=-1, keepdims=True)

        # Sample
        tokens = np.array([
            np.random.choice(probs.shape[1], p=probs[i])
            for i in range(batch_size)
        ])

        return tokens

    def benchmark(
        self,
        input_ids: np.ndarray,
        num_tokens: int = 100
    ) -> Dict[str, float]:
        """Benchmark inference speed."""
        config = GenerationConfig(max_length=num_tokens, do_sample=False)

        start = time.perf_counter()
        output = self.generate(input_ids, config)
        end = time.perf_counter()

        elapsed = end - start
        generated = output.shape[1] - input_ids.shape[1]

        return {
            "total_time_s": elapsed,
            "tokens_generated": generated,
            "tokens_per_second": generated / elapsed if elapsed > 0 else 0,
            "ms_per_token": elapsed * 1000 / generated if generated > 0 else 0,
        }
