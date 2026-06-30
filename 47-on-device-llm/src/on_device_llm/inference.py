"""Transformer inference engine for on-device LLM.

This module provides the main inference engine for running transformer
models on-device with efficient memory usage and optimized operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from on_device_llm.loader import GGUFLoader, MockGGUFLoader, ModelConfig
from on_device_llm.memory import KVCache
from on_device_llm.operators import matmul_f32, rmsnorm, softmax, rope_embed, silu


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    eos_token_id: int = 2
    pad_token_id: int = 0

    # Sampling options
    do_sample: bool = True
    num_beams: int = 1  # 1 = greedy/sampling, >1 = beam search

    def validate(self) -> None:
        """Validate configuration."""
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be >= 1")


class Sampler:
    """Token sampling strategies."""

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0
    ):
        """
        Initialize sampler.

        Args:
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
            top_p: Nucleus sampling probability threshold
            repetition_penalty: Penalty for repeated tokens
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self._generated_tokens: List[int] = []

    def sample(self, logits: np.ndarray) -> int:
        """
        Sample next token from logits.

        Args:
            logits: Logit scores [vocab_size]

        Returns:
            Sampled token ID
        """
        logits = logits.copy()

        # Apply repetition penalty
        if self.repetition_penalty != 1.0 and self._generated_tokens:
            for token_id in set(self._generated_tokens):
                if logits[token_id] > 0:
                    logits[token_id] /= self.repetition_penalty
                else:
                    logits[token_id] *= self.repetition_penalty

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Top-k filtering
        if self.top_k > 0:
            top_k = min(self.top_k, len(logits))
            indices = np.argsort(logits)[-top_k:]
            mask = np.full_like(logits, float('-inf'))
            mask[indices] = logits[indices]
            logits = mask

        # Softmax
        probs = softmax(logits)

        # Top-p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_indices])
            cutoff_idx = np.searchsorted(cumsum, self.top_p) + 1
            cutoff_idx = min(cutoff_idx, len(sorted_indices))

            mask = np.zeros_like(probs)
            mask[sorted_indices[:cutoff_idx]] = probs[sorted_indices[:cutoff_idx]]
            probs = mask / mask.sum()

        # Sample
        token_id = int(np.random.choice(len(probs), p=probs))
        self._generated_tokens.append(token_id)

        return token_id

    def greedy(self, logits: np.ndarray) -> int:
        """
        Greedy decoding (argmax).

        Args:
            logits: Logit scores [vocab_size]

        Returns:
            Token ID with highest logit
        """
        token_id = int(np.argmax(logits))
        self._generated_tokens.append(token_id)
        return token_id

    def reset(self) -> None:
        """Reset sampler state."""
        self._generated_tokens.clear()

    @classmethod
    def from_config(cls, config: GenerationConfig) -> 'Sampler':
        """Create sampler from generation config."""
        return cls(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty
        )


class TransformerInference:
    """On-device transformer inference engine."""

    def __init__(
        self,
        loader: Union[GGUFLoader, MockGGUFLoader],
        window_size: Optional[int] = None
    ):
        """
        Initialize inference engine.

        Args:
            loader: Model loader (GGUF or mock)
            window_size: Sliding window size for KV cache (None = no window)
        """
        self.loader = loader
        self.config = loader.config

        # Initialize KV cache
        self.kv_cache = KVCache(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_seq_len,
            window_size=window_size
        )

        # Pre-load embedding table
        self._embed_tokens: Optional[np.ndarray] = None
        self._load_embedding()

    def _load_embedding(self) -> None:
        """Load token embedding table."""
        self._embed_tokens = self.loader.get_tensor('token_embd.weight')

    def forward(self, token_id: int, position: int) -> np.ndarray:
        """
        Forward pass for a single token.

        Args:
            token_id: Input token ID
            position: Position in sequence

        Returns:
            Logits [vocab_size]
        """
        # Embed token
        hidden = self._embed_tokens[token_id].copy()

        # Process each layer
        for layer_idx in range(self.config.num_layers):
            hidden = self._forward_layer(hidden, layer_idx, position)

        # Final norm
        norm_weight = self.loader.get_tensor('output_norm.weight')
        hidden = rmsnorm(hidden, norm_weight, self.config.norm_eps)

        # Output projection
        lm_head = self.loader.get_tensor('output.weight')
        logits = matmul_f32(hidden.reshape(1, -1), lm_head.T).flatten()

        return logits

    def _forward_layer(
        self,
        hidden: np.ndarray,
        layer_idx: int,
        position: int
    ) -> np.ndarray:
        """
        Forward pass for one transformer layer.

        Args:
            hidden: Input hidden state [hidden_size]
            layer_idx: Layer index
            position: Position in sequence

        Returns:
            Output hidden state [hidden_size]
        """
        # Pre-attention norm
        norm_weight = self.loader.get_tensor(f'blk.{layer_idx}.attn_norm.weight')
        normed = rmsnorm(hidden, norm_weight, self.config.norm_eps)

        # Self-attention
        attn_out = self._attention(normed, layer_idx, position)

        # Residual
        hidden = hidden + attn_out

        # Pre-FFN norm
        ffn_norm = self.loader.get_tensor(f'blk.{layer_idx}.ffn_norm.weight')
        normed = rmsnorm(hidden, ffn_norm, self.config.norm_eps)

        # FFN
        ffn_out = self._ffn(normed, layer_idx)

        # Residual
        hidden = hidden + ffn_out

        return hidden

    def _attention(
        self,
        x: np.ndarray,
        layer_idx: int,
        position: int
    ) -> np.ndarray:
        """
        Multi-head attention with GQA support.

        Args:
            x: Input hidden state [hidden_size]
            layer_idx: Layer index
            position: Position in sequence

        Returns:
            Attention output [hidden_size]
        """
        head_dim = self.config.head_dim
        num_heads = self.config.num_heads
        num_kv_heads = self.config.num_kv_heads

        # Load weights
        wq = self.loader.get_tensor(f'blk.{layer_idx}.attn_q.weight')
        wk = self.loader.get_tensor(f'blk.{layer_idx}.attn_k.weight')
        wv = self.loader.get_tensor(f'blk.{layer_idx}.attn_v.weight')
        wo = self.loader.get_tensor(f'blk.{layer_idx}.attn_output.weight')

        # Project Q, K, V
        q = matmul_f32(x.reshape(1, -1), wq.T).flatten()
        k = matmul_f32(x.reshape(1, -1), wk.T).flatten()
        v = matmul_f32(x.reshape(1, -1), wv.T).flatten()

        # Reshape to heads
        q = q.reshape(num_heads, head_dim)
        k = k.reshape(num_kv_heads, head_dim)
        v = v.reshape(num_kv_heads, head_dim)

        # Apply RoPE
        for h in range(num_heads):
            q[h] = rope_embed(q[h], position, self.config.rope_theta)
        for h in range(num_kv_heads):
            k[h] = rope_embed(k[h], position, self.config.rope_theta)

        # Update KV cache
        self.kv_cache.append(layer_idx, k, v)

        # Get cached KV - use position+1 to include current token
        # (kv_cache.get() uses self.length which is only updated after last layer)
        cache_len = position + 1
        if self.kv_cache.window_size:
            cache_len = min(cache_len, self.kv_cache.window_size)
        k_cache = self.kv_cache.k_cache[layer_idx, :cache_len]
        v_cache = self.kv_cache.v_cache[layer_idx, :cache_len]

        # Compute attention
        output = np.zeros(num_heads * head_dim, dtype=np.float32)

        # GQA: repeat KV heads if needed
        kv_repeat = num_heads // num_kv_heads

        for h in range(num_heads):
            kv_h = h // kv_repeat  # KV head index

            # Attention scores
            scores = np.zeros(len(k_cache), dtype=np.float32)
            for i in range(len(k_cache)):
                scores[i] = np.dot(q[h], k_cache[i, kv_h]) / np.sqrt(head_dim)

            # Softmax
            probs = softmax(scores)

            # Weighted sum of values
            head_out = np.zeros(head_dim, dtype=np.float32)
            for i in range(len(v_cache)):
                head_out += probs[i] * v_cache[i, kv_h]

            output[h * head_dim:(h + 1) * head_dim] = head_out

        # Output projection
        return matmul_f32(output.reshape(1, -1), wo.T).flatten()

    def _ffn(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Feed-forward network (SwiGLU).

        Args:
            x: Input hidden state [hidden_size]
            layer_idx: Layer index

        Returns:
            FFN output [hidden_size]
        """
        # Load weights
        w_gate = self.loader.get_tensor(f'blk.{layer_idx}.ffn_gate.weight')
        w_up = self.loader.get_tensor(f'blk.{layer_idx}.ffn_up.weight')
        w_down = self.loader.get_tensor(f'blk.{layer_idx}.ffn_down.weight')

        # Gate and up projections
        gate = matmul_f32(x.reshape(1, -1), w_gate.T).flatten()
        up = matmul_f32(x.reshape(1, -1), w_up.T).flatten()

        # SiLU activation on gate
        gate = silu(gate)

        # Element-wise product
        hidden = gate * up

        # Down projection
        return matmul_f32(hidden.reshape(1, -1), w_down.T).flatten()

    def generate(
        self,
        prompt_tokens: List[int],
        config: Optional[GenerationConfig] = None
    ) -> List[int]:
        """
        Generate tokens from a prompt.

        Args:
            prompt_tokens: Input token IDs
            config: Generation configuration

        Returns:
            Generated token IDs (excluding prompt)
        """
        if config is None:
            config = GenerationConfig()

        config.validate()

        # Reset KV cache
        self.kv_cache.clear()

        # Create sampler
        sampler = Sampler.from_config(config)

        # Process prompt (fill KV cache)
        for i, token in enumerate(prompt_tokens[:-1]):
            self.forward(token, i)

        # Generate
        position = len(prompt_tokens) - 1
        token = prompt_tokens[-1]
        output_tokens: List[int] = []

        for _ in range(config.max_new_tokens):
            # Stop if we've reached max sequence length
            if position >= self.config.max_seq_len:
                break

            logits = self.forward(token, position)

            if config.do_sample:
                token = sampler.sample(logits)
            else:
                token = sampler.greedy(logits)

            output_tokens.append(token)
            position += 1

            if token == config.eos_token_id:
                break

        return output_tokens

    def reset(self) -> None:
        """Reset inference state."""
        self.kv_cache.clear()

    def memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self.kv_cache.memory_usage()


class BatchInference:
    """Batch inference for multiple sequences."""

    def __init__(
        self,
        loader: Union[GGUFLoader, MockGGUFLoader],
        max_batch_size: int = 8
    ):
        """
        Initialize batch inference.

        Args:
            loader: Model loader
            max_batch_size: Maximum batch size
        """
        self.loader = loader
        self.config = loader.config
        self.max_batch_size = max_batch_size

        # Create inference engines for each batch slot
        self._engines: List[TransformerInference] = [
            TransformerInference(loader)
            for _ in range(max_batch_size)
        ]

    def forward_batch(
        self,
        token_ids: List[int],
        positions: List[int],
        batch_indices: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Forward pass for a batch of tokens.

        Args:
            token_ids: Token IDs for each sequence
            positions: Positions for each sequence
            batch_indices: Which engine to use for each token

        Returns:
            Logits for each sequence
        """
        if batch_indices is None:
            batch_indices = list(range(len(token_ids)))

        results = []
        for tid, pos, idx in zip(token_ids, positions, batch_indices):
            if idx >= self.max_batch_size:
                raise ValueError(f"Batch index {idx} >= max_batch_size {self.max_batch_size}")
            logits = self._engines[idx].forward(tid, pos)
            results.append(logits)

        return results

    def reset(self, batch_index: Optional[int] = None) -> None:
        """
        Reset inference state.

        Args:
            batch_index: Specific engine to reset (None = all)
        """
        if batch_index is None:
            for engine in self._engines:
                engine.reset()
        else:
            self._engines[batch_index].reset()
