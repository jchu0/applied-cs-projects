"""AI workload benchmarks."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import time

from ..core.benchmark import (
    Benchmark, BenchmarkConfig, Metric, MetricType,
    register_benchmark, Timer
)


@register_benchmark("llm_inference")
class LLMInferenceBenchmark(Benchmark):
    """Benchmark LLM inference throughput and latency."""

    def name(self) -> str:
        return "llm_inference"

    def setup(self) -> None:
        """Initialize model and data."""
        self.vocab_size = 32000
        self.hidden_size = 2048
        self.num_layers = self.config.extra_config.get("num_layers", 12)

        # Simulate model weights
        self.weights = [
            np.random.randn(self.hidden_size, self.hidden_size).astype(np.float16)
            for _ in range(self.num_layers * 4)
        ]

        # Input tokens
        self.input_ids = np.random.randint(
            0, self.vocab_size,
            (self.config.batch_size, self.config.sequence_length)
        )

    def run_iteration(self) -> dict[str, Any]:
        """Run single forward pass."""
        # Simulate forward pass
        hidden = np.random.randn(
            self.config.batch_size,
            self.config.sequence_length,
            self.hidden_size
        ).astype(np.float16)

        for weight in self.weights:
            # Simulated matmul
            hidden = hidden @ weight.T

        tokens_generated = self.config.batch_size
        return {
            "tokens_generated": tokens_generated,
            "output_shape": hidden.shape
        }

    def teardown(self) -> None:
        """Clean up."""
        self.weights = []

    def _compute_metrics(
        self,
        iteration_results: list[dict[str, Any]]
    ) -> list[Metric]:
        metrics = super()._compute_metrics(iteration_results)

        # Tokens per second
        total_tokens = sum(r["tokens_generated"] for r in iteration_results)
        total_time_s = sum(self.timer.elapsed_times) / 1000
        tokens_per_sec = total_tokens / total_time_s if total_time_s > 0 else 0

        metrics.append(Metric(
            name="tokens_per_second",
            value=tokens_per_sec,
            unit=MetricType.TOKENS_PER_SEC,
            lower_is_better=False
        ))

        return metrics


@register_benchmark("llm_generation")
class LLMGenerationBenchmark(Benchmark):
    """Benchmark autoregressive text generation."""

    def name(self) -> str:
        return "llm_generation"

    def setup(self) -> None:
        """Initialize for generation."""
        self.vocab_size = 32000
        self.hidden_size = 1024
        self.max_new_tokens = self.config.extra_config.get("max_new_tokens", 128)

        # Simplified model
        self.embed = np.random.randn(self.vocab_size, self.hidden_size).astype(np.float16)
        self.lm_head = np.random.randn(self.hidden_size, self.vocab_size).astype(np.float16)

        # Initial prompt
        self.prompt = np.random.randint(
            0, self.vocab_size,
            (self.config.batch_size, 32)
        )

    def run_iteration(self) -> dict[str, Any]:
        """Generate tokens autoregressively."""
        generated = []
        current = self.prompt.copy()

        for _ in range(self.max_new_tokens):
            # Get last token embedding
            last_token = current[:, -1]
            hidden = self.embed[last_token]

            # Project to vocab
            logits = hidden @ self.lm_head

            # Sample (greedy)
            next_token = np.argmax(logits, axis=-1)
            generated.append(next_token)

            # Update current (simplified)
            current = np.concatenate([
                current,
                next_token[:, np.newaxis]
            ], axis=1)

        return {
            "tokens_generated": len(generated) * self.config.batch_size,
            "sequence_length": len(generated)
        }

    def teardown(self) -> None:
        pass

    def _compute_metrics(
        self,
        iteration_results: list[dict[str, Any]]
    ) -> list[Metric]:
        metrics = super()._compute_metrics(iteration_results)

        # Generation speed
        total_tokens = sum(r["tokens_generated"] for r in iteration_results)
        total_time_s = sum(self.timer.elapsed_times) / 1000
        tokens_per_sec = total_tokens / total_time_s if total_time_s > 0 else 0

        metrics.append(Metric(
            name="generation_speed",
            value=tokens_per_sec,
            unit=MetricType.TOKENS_PER_SEC,
            lower_is_better=False
        ))

        # Time to first token (TTFT)
        if self.timer.elapsed_times:
            ttft = self.timer.elapsed_times[0] / self.max_new_tokens
            metrics.append(Metric(
                name="time_to_first_token",
                value=ttft,
                unit=MetricType.TIME_MS
            ))

        return metrics


@register_benchmark("training_throughput")
class TrainingThroughputBenchmark(Benchmark):
    """Benchmark training throughput."""

    def name(self) -> str:
        return "training_throughput"

    def setup(self) -> None:
        """Set up training loop."""
        self.hidden_size = 1024
        self.num_layers = 6

        # Model parameters (simplified)
        self.params = [
            np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32)
            for _ in range(self.num_layers * 4)
        ]

        # Batch data
        self.data = np.random.randn(
            self.config.batch_size,
            self.config.sequence_length,
            self.hidden_size
        ).astype(np.float32)

        self.learning_rate = 1e-4

    def run_iteration(self) -> dict[str, Any]:
        """Run single training step (forward + backward + update)."""
        # Forward pass
        hidden = self.data.copy()
        activations = [hidden]

        for param in self.params:
            hidden = hidden @ param.T
            activations.append(hidden)

        # Compute loss (MSE)
        target = np.zeros_like(hidden)
        loss = np.mean((hidden - target) ** 2)

        # Backward pass (simplified)
        grad = 2 * (hidden - target) / hidden.size

        for i in range(len(self.params) - 1, -1, -1):
            # Gradient for param
            param_grad = activations[i].reshape(-1, self.hidden_size).T @ grad.reshape(-1, self.hidden_size)

            # Update param
            self.params[i] -= self.learning_rate * param_grad

            # Gradient for input
            grad = grad @ self.params[i]

        samples_processed = self.config.batch_size

        return {
            "loss": float(loss),
            "samples": samples_processed
        }

    def teardown(self) -> None:
        pass

    def _compute_metrics(
        self,
        iteration_results: list[dict[str, Any]]
    ) -> list[Metric]:
        metrics = super()._compute_metrics(iteration_results)

        # Samples per second
        total_samples = sum(r["samples"] for r in iteration_results)
        total_time_s = sum(self.timer.elapsed_times) / 1000
        samples_per_sec = total_samples / total_time_s if total_time_s > 0 else 0

        metrics.append(Metric(
            name="samples_per_second",
            value=samples_per_sec,
            unit=MetricType.SAMPLES_PER_SEC,
            lower_is_better=False
        ))

        # Final loss
        if iteration_results:
            final_loss = iteration_results[-1]["loss"]
            metrics.append(Metric(
                name="final_loss",
                value=final_loss,
                unit=MetricType.LOSS
            ))

        return metrics


@register_benchmark("memory_bandwidth")
class MemoryBandwidthBenchmark(Benchmark):
    """Benchmark memory bandwidth."""

    def name(self) -> str:
        return "memory_bandwidth"

    def setup(self) -> None:
        """Allocate test buffers."""
        size_mb = self.config.extra_config.get("size_mb", 1024)
        self.size_bytes = size_mb * 1024 * 1024

        # Allocate buffers
        num_elements = self.size_bytes // 4  # float32
        self.src = np.random.randn(num_elements).astype(np.float32)
        self.dst = np.zeros(num_elements, dtype=np.float32)

    def run_iteration(self) -> dict[str, Any]:
        """Run memory copy."""
        np.copyto(self.dst, self.src)

        return {
            "bytes_copied": self.size_bytes
        }

    def teardown(self) -> None:
        del self.src
        del self.dst

    def _compute_metrics(
        self,
        iteration_results: list[dict[str, Any]]
    ) -> list[Metric]:
        metrics = super()._compute_metrics(iteration_results)

        # Bandwidth in GB/s
        total_bytes = sum(r["bytes_copied"] for r in iteration_results)
        total_time_s = sum(self.timer.elapsed_times) / 1000
        bandwidth_gbps = (total_bytes / total_time_s) / 1e9 if total_time_s > 0 else 0

        metrics.append(Metric(
            name="bandwidth_gbps",
            value=bandwidth_gbps,
            unit=MetricType.THROUGHPUT,
            lower_is_better=False,
            metadata={"unit": "GB/s"}
        ))

        return metrics


@register_benchmark("matmul")
class MatMulBenchmark(Benchmark):
    """Benchmark matrix multiplication."""

    def name(self) -> str:
        return "matmul"

    def setup(self) -> None:
        """Set up matrices."""
        m = self.config.extra_config.get("m", 4096)
        n = self.config.extra_config.get("n", 4096)
        k = self.config.extra_config.get("k", 4096)

        self.m, self.n, self.k = m, n, k

        # Create matrices
        dtype = np.float16 if self.config.precision == "fp16" else np.float32
        self.a = np.random.randn(m, k).astype(dtype)
        self.b = np.random.randn(k, n).astype(dtype)
        self.c = np.zeros((m, n), dtype=dtype)

    def run_iteration(self) -> dict[str, Any]:
        """Run matmul."""
        self.c = np.matmul(self.a, self.b)

        # FLOPs = 2 * M * N * K
        flops = 2 * self.m * self.n * self.k

        return {
            "flops": flops
        }

    def teardown(self) -> None:
        pass

    def _compute_metrics(
        self,
        iteration_results: list[dict[str, Any]]
    ) -> list[Metric]:
        metrics = super()._compute_metrics(iteration_results)

        # TFLOPS
        total_flops = sum(r["flops"] for r in iteration_results)
        total_time_s = sum(self.timer.elapsed_times) / 1000
        tflops = (total_flops / total_time_s) / 1e12 if total_time_s > 0 else 0

        metrics.append(Metric(
            name="tflops",
            value=tflops,
            unit=MetricType.TFLOPS,
            lower_is_better=False
        ))

        return metrics


@register_benchmark("attention")
class AttentionBenchmark(Benchmark):
    """Benchmark attention computation."""

    def name(self) -> str:
        return "attention"

    def setup(self) -> None:
        """Set up attention inputs."""
        batch = self.config.batch_size
        seq_len = self.config.sequence_length
        heads = self.config.extra_config.get("num_heads", 32)
        head_dim = self.config.extra_config.get("head_dim", 128)

        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.head_dim = head_dim

        dtype = np.float16 if self.config.precision == "fp16" else np.float32

        self.q = np.random.randn(batch, heads, seq_len, head_dim).astype(dtype)
        self.k = np.random.randn(batch, heads, seq_len, head_dim).astype(dtype)
        self.v = np.random.randn(batch, heads, seq_len, head_dim).astype(dtype)

    def run_iteration(self) -> dict[str, Any]:
        """Run attention."""
        # QK^T
        scores = np.matmul(self.q, self.k.transpose(0, 1, 3, 2))
        scores = scores / np.sqrt(self.head_dim)

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Attention output
        output = np.matmul(attn_weights, self.v)

        # FLOPs: 2*B*H*S*S*D (QK) + B*H*S*S (softmax) + 2*B*H*S*S*D (AV)
        flops = 4 * self.batch * self.heads * self.seq_len * self.seq_len * self.head_dim

        return {
            "flops": flops,
            "output_shape": output.shape
        }

    def teardown(self) -> None:
        pass


@register_benchmark("embedding")
class EmbeddingBenchmark(Benchmark):
    """Benchmark embedding lookup."""

    def name(self) -> str:
        return "embedding"

    def setup(self) -> None:
        """Set up embedding table."""
        self.vocab_size = self.config.extra_config.get("vocab_size", 32000)
        self.embed_dim = self.config.extra_config.get("embed_dim", 4096)

        self.embedding_table = np.random.randn(
            self.vocab_size, self.embed_dim
        ).astype(np.float32)

        self.input_ids = np.random.randint(
            0, self.vocab_size,
            (self.config.batch_size, self.config.sequence_length)
        )

    def run_iteration(self) -> dict[str, Any]:
        """Run embedding lookup."""
        output = self.embedding_table[self.input_ids]

        tokens = self.config.batch_size * self.config.sequence_length

        return {
            "tokens": tokens,
            "output_shape": output.shape
        }

    def teardown(self) -> None:
        pass


@register_benchmark("softmax")
class SoftmaxBenchmark(Benchmark):
    """Benchmark softmax computation."""

    def name(self) -> str:
        return "softmax"

    def setup(self) -> None:
        """Set up input tensor."""
        self.input_tensor = np.random.randn(
            self.config.batch_size,
            self.config.sequence_length,
            self.config.extra_config.get("vocab_size", 32000)
        ).astype(np.float32)

    def run_iteration(self) -> dict[str, Any]:
        """Run softmax."""
        # Stable softmax
        x_max = np.max(self.input_tensor, axis=-1, keepdims=True)
        exp_x = np.exp(self.input_tensor - x_max)
        output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        return {
            "elements": self.input_tensor.size
        }

    def teardown(self) -> None:
        pass
