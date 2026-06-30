"""Inference and generation for on-device LLM."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator
import numpy as np
import time

from ..core.model import LLMWeights, ModelConfig
from ..runtime.runtime import DeviceRuntime, ExecutionContext, RuntimeConfig


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_tokens: list[int] = field(default_factory=lambda: [2])  # EOS


@dataclass
class TokenOutput:
    """Single token output."""
    token_id: int
    logprob: float
    is_stop: bool = False


@dataclass
class GenerationStats:
    """Statistics for generation run."""
    prompt_tokens: int
    generated_tokens: int
    prompt_time_ms: float
    generation_time_ms: float
    tokens_per_second: float
    memory_peak_mb: float


class Sampler:
    """Token sampling strategies for mobile."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.generated_tokens: list[int] = []

    def sample(self, logits: np.ndarray) -> TokenOutput:
        """Sample next token from logits."""
        # Apply repetition penalty
        if self.config.repetition_penalty != 1.0 and self.generated_tokens:
            for token_id in set(self.generated_tokens):
                if logits[token_id] > 0:
                    logits[token_id] /= self.config.repetition_penalty
                else:
                    logits[token_id] *= self.config.repetition_penalty

        # Temperature scaling
        if self.config.temperature > 0:
            logits = logits / self.config.temperature

        # Top-k filtering
        if self.config.top_k > 0:
            indices = np.argpartition(logits, -self.config.top_k)[-self.config.top_k:]
            mask = np.ones_like(logits, dtype=bool)
            mask[indices] = False
            logits[mask] = -np.inf

        # Convert to probabilities
        probs = self._softmax(logits)

        # Top-p (nucleus) filtering
        if self.config.top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)
            cutoff_idx = np.searchsorted(cumsum, self.config.top_p)
            cutoff_idx = min(cutoff_idx + 1, len(probs))

            mask = np.ones_like(probs, dtype=bool)
            mask[sorted_indices[:cutoff_idx]] = False
            probs[mask] = 0
            probs = probs / probs.sum()

        # Sample
        if self.config.temperature == 0:
            token_id = int(np.argmax(probs))
        else:
            token_id = int(np.random.choice(len(probs), p=probs))

        logprob = float(np.log(probs[token_id] + 1e-10))
        is_stop = token_id in self.config.stop_tokens

        self.generated_tokens.append(token_id)

        return TokenOutput(token_id=token_id, logprob=logprob, is_stop=is_stop)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    def reset(self) -> None:
        """Reset sampler state."""
        self.generated_tokens = []


class LLMEngine:
    """Main inference engine for on-device LLM."""

    def __init__(
        self,
        weights: LLMWeights,
        runtime_config: RuntimeConfig | None = None
    ):
        self.weights = weights
        self.config = weights.config

        if runtime_config is None:
            runtime_config = RuntimeConfig(
                memory_limit_mb=weights.memory_size() / (1024 * 1024) + 256
            )

        self.runtime = DeviceRuntime(runtime_config)
        self.context: ExecutionContext | None = None

    def _ensure_context(self, max_length: int) -> ExecutionContext:
        """Ensure execution context is created."""
        if self.context is None or self.context.max_length < max_length:
            self.context = self.runtime.create_context(self.config)
        return self.context

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        """Get embeddings for tokens."""
        # Simple embedding lookup
        embed_weight = self.weights.embed_tokens.dequantize()
        return embed_weight[token_ids]

    def forward(
        self,
        input_ids: np.ndarray,
        ctx: ExecutionContext
    ) -> np.ndarray:
        """Forward pass through model."""
        # Get embeddings
        hidden = self.embed(input_ids)

        # Add batch dimension if needed
        if hidden.ndim == 2:
            hidden = hidden[np.newaxis, ...]

        # Process through layers
        for i, layer in enumerate(self.weights.layers):
            hidden = self.runtime.execute_layer(ctx, layer, hidden, i)

        # Final norm
        hidden = self.runtime.operators["rmsnorm"](hidden, self.weights.norm)

        # LM head
        if self.weights.lm_head:
            logits = self.runtime.operators["matmul"](hidden, self.weights.lm_head)
        else:
            # Tied embeddings
            logits = self.runtime.operators["matmul"](hidden, self.weights.embed_tokens)

        return logits

    def generate(
        self,
        prompt_ids: list[int],
        gen_config: GenerationConfig | None = None
    ) -> Iterator[TokenOutput]:
        """Generate tokens from prompt."""
        if gen_config is None:
            gen_config = GenerationConfig()

        max_length = len(prompt_ids) + gen_config.max_tokens
        ctx = self._ensure_context(max_length)
        ctx.reset()

        sampler = Sampler(gen_config)

        # Process prompt
        prompt_array = np.array(prompt_ids)
        logits = self.forward(prompt_array, ctx)
        ctx.kv_cache.advance(len(prompt_ids))

        # Get logits for last position
        last_logits = logits[0, -1, :]

        # Generate tokens
        for _ in range(gen_config.max_tokens):
            # Sample next token
            output = sampler.sample(last_logits.copy())
            yield output

            if output.is_stop:
                break

            # Forward pass for new token
            next_token = np.array([[output.token_id]])
            logits = self.forward(next_token, ctx)
            ctx.kv_cache.advance(1)

            last_logits = logits[0, -1, :]

    def generate_with_stats(
        self,
        prompt_ids: list[int],
        gen_config: GenerationConfig | None = None
    ) -> tuple[list[int], GenerationStats]:
        """Generate tokens and return statistics."""
        if gen_config is None:
            gen_config = GenerationConfig()

        max_length = len(prompt_ids) + gen_config.max_tokens
        ctx = self._ensure_context(max_length)
        ctx.reset()

        sampler = Sampler(gen_config)
        generated = []

        # Time prompt processing
        prompt_start = time.perf_counter()
        prompt_array = np.array(prompt_ids)
        logits = self.forward(prompt_array, ctx)
        ctx.kv_cache.advance(len(prompt_ids))
        prompt_time = time.perf_counter() - prompt_start

        last_logits = logits[0, -1, :]

        # Time generation
        gen_start = time.perf_counter()

        for _ in range(gen_config.max_tokens):
            output = sampler.sample(last_logits.copy())
            generated.append(output.token_id)

            if output.is_stop:
                break

            next_token = np.array([[output.token_id]])
            logits = self.forward(next_token, ctx)
            ctx.kv_cache.advance(1)
            last_logits = logits[0, -1, :]

        gen_time = time.perf_counter() - gen_start

        # Calculate stats
        tokens_per_second = len(generated) / gen_time if gen_time > 0 else 0
        memory_stats = ctx.get_memory_stats()

        stats = GenerationStats(
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated),
            prompt_time_ms=prompt_time * 1000,
            generation_time_ms=gen_time * 1000,
            tokens_per_second=tokens_per_second,
            memory_peak_mb=memory_stats.peak_mb
        )

        return generated, stats


class StreamingEngine:
    """Streaming inference for real-time generation."""

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.buffer: list[int] = []

    def start_stream(
        self,
        prompt_ids: list[int],
        gen_config: GenerationConfig | None = None
    ) -> Iterator[tuple[int, float]]:
        """Start streaming generation."""
        for output in self.engine.generate(prompt_ids, gen_config):
            self.buffer.append(output.token_id)
            yield output.token_id, output.logprob

            if output.is_stop:
                break

    def get_generated(self) -> list[int]:
        """Get all generated tokens."""
        return self.buffer.copy()

    def reset(self) -> None:
        """Reset streaming state."""
        self.buffer = []


class BatchEngine:
    """Batch inference for multiple sequences."""

    def __init__(self, engine: LLMEngine):
        self.engine = engine

    def generate_batch(
        self,
        prompts: list[list[int]],
        gen_config: GenerationConfig | None = None
    ) -> list[list[int]]:
        """Generate for multiple prompts."""
        # Simple sequential batch processing
        # For mobile, batching is usually not needed
        results = []
        for prompt in prompts:
            generated, _ = self.engine.generate_with_stats(prompt, gen_config)
            results.append(generated)
        return results


class ContinuousEngine:
    """Continuous generation with context management."""

    def __init__(self, engine: LLMEngine, max_context: int = 2048):
        self.engine = engine
        self.max_context = max_context
        self.history: list[int] = []

    def add_to_history(self, tokens: list[int]) -> None:
        """Add tokens to history."""
        self.history.extend(tokens)

        # Trim if exceeding max context
        if len(self.history) > self.max_context:
            # Keep last max_context tokens
            self.history = self.history[-self.max_context:]

    def generate_turn(
        self,
        input_ids: list[int],
        gen_config: GenerationConfig | None = None
    ) -> list[int]:
        """Generate response for a turn."""
        # Combine history with input
        full_prompt = self.history + input_ids
        self.add_to_history(input_ids)

        # Generate
        generated, _ = self.engine.generate_with_stats(full_prompt, gen_config)

        # Add generated to history
        self.add_to_history(generated)

        return generated

    def reset_history(self) -> None:
        """Reset conversation history."""
        self.history = []

    @property
    def context_usage(self) -> float:
        """Get context usage ratio."""
        return len(self.history) / self.max_context


def create_engine(
    weights: LLMWeights,
    memory_limit_mb: float = 512,
    num_threads: int = 4
) -> LLMEngine:
    """Create inference engine with configuration."""
    config = RuntimeConfig(
        num_threads=num_threads,
        memory_limit_mb=memory_limit_mb,
        context_length=weights.config.max_position
    )
    return LLMEngine(weights, config)
