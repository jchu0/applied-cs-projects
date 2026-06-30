"""Core benchmark definitions and metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import time
import statistics
import json
import os


class BenchmarkType(Enum):
    """Types of benchmarks."""
    INFERENCE = "inference"
    TRAINING = "training"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ACCURACY = "accuracy"


class MetricType(Enum):
    """Types of metrics."""
    TIME_MS = "time_ms"
    TIME_S = "time_s"
    THROUGHPUT = "throughput"  # items/sec
    MEMORY_MB = "memory_mb"
    MEMORY_GB = "memory_gb"
    FLOPS = "flops"
    TFLOPS = "tflops"
    ACCURACY = "accuracy"
    LOSS = "loss"
    TOKENS_PER_SEC = "tokens/s"
    SAMPLES_PER_SEC = "samples/s"


@dataclass
class Metric:
    """Single metric measurement."""
    name: str
    value: float
    unit: MetricType
    lower_is_better: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.value,
            "lower_is_better": self.lower_is_better,
            "metadata": self.metadata
        }


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    timeout_seconds: float = 300.0
    batch_size: int = 1
    sequence_length: int = 512
    num_threads: int = 4
    device: str = "cpu"
    precision: str = "fp32"
    extra_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    benchmark_name: str
    config: BenchmarkConfig
    metrics: list[Metric]
    raw_times_ms: list[float]
    success: bool = True
    error_message: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def mean_time_ms(self) -> float:
        return statistics.mean(self.raw_times_ms) if self.raw_times_ms else 0.0

    @property
    def std_time_ms(self) -> float:
        return statistics.stdev(self.raw_times_ms) if len(self.raw_times_ms) > 1 else 0.0

    @property
    def min_time_ms(self) -> float:
        return min(self.raw_times_ms) if self.raw_times_ms else 0.0

    @property
    def max_time_ms(self) -> float:
        return max(self.raw_times_ms) if self.raw_times_ms else 0.0

    @property
    def p50_time_ms(self) -> float:
        return statistics.median(self.raw_times_ms) if self.raw_times_ms else 0.0

    @property
    def p99_time_ms(self) -> float:
        if not self.raw_times_ms:
            return 0.0
        sorted_times = sorted(self.raw_times_ms)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def to_dict(self) -> dict:
        return {
            "benchmark_name": self.benchmark_name,
            "config": {
                "name": self.config.name,
                "warmup": self.config.warmup_iterations,
                "iterations": self.config.benchmark_iterations,
                "batch_size": self.config.batch_size,
                "sequence_length": self.config.sequence_length,
                "device": self.config.device,
                "precision": self.config.precision
            },
            "metrics": [m.to_dict() for m in self.metrics],
            "statistics": {
                "mean_ms": self.mean_time_ms,
                "std_ms": self.std_time_ms,
                "min_ms": self.min_time_ms,
                "max_ms": self.max_time_ms,
                "p50_ms": self.p50_time_ms,
                "p99_ms": self.p99_time_ms
            },
            "success": self.success,
            "error": self.error_message,
            "timestamp": self.timestamp
        }


class Timer:
    """High-precision timer."""

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed_times: list[float] = []

    def start(self) -> None:
        """Start timer."""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timer and return elapsed time in ms."""
        self.end_time = time.perf_counter()
        elapsed = (self.end_time - self.start_time) * 1000
        self.elapsed_times.append(elapsed)
        return elapsed

    def reset(self) -> None:
        """Reset timer."""
        self.elapsed_times = []

    @property
    def last(self) -> float:
        """Get last recorded time."""
        return self.elapsed_times[-1] if self.elapsed_times else 0.0

    @property
    def mean(self) -> float:
        """Get mean time."""
        return statistics.mean(self.elapsed_times) if self.elapsed_times else 0.0


class MemoryTracker:
    """Track memory usage."""

    def __init__(self):
        self.peak_memory_mb = 0.0
        self.current_memory_mb = 0.0
        self._measurements: list[float] = []

    def snapshot(self) -> float:
        """Take memory snapshot (simulated)."""
        # In real implementation, would use torch.cuda.memory_allocated() etc.
        import random
        memory = random.uniform(100, 1000)  # Simulated
        self._measurements.append(memory)
        self.current_memory_mb = memory
        self.peak_memory_mb = max(self.peak_memory_mb, memory)
        return memory

    def reset(self) -> None:
        """Reset tracker."""
        self._measurements = []
        self.peak_memory_mb = 0.0
        self.current_memory_mb = 0.0


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.timer = Timer()
        self.memory_tracker = MemoryTracker()

    @abstractmethod
    def name(self) -> str:
        """Get benchmark name."""
        pass

    @abstractmethod
    def setup(self) -> None:
        """Set up benchmark (load models, data, etc.)."""
        pass

    @abstractmethod
    def run_iteration(self) -> dict[str, Any]:
        """Run single benchmark iteration."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Clean up after benchmark."""
        pass

    def warmup(self) -> None:
        """Run warmup iterations."""
        for _ in range(self.config.warmup_iterations):
            self.run_iteration()

    def benchmark(self) -> BenchmarkResult:
        """Run complete benchmark."""
        try:
            self.setup()
            self.warmup()

            self.timer.reset()
            iteration_results = []

            for _ in range(self.config.benchmark_iterations):
                self.timer.start()
                result = self.run_iteration()
                self.timer.stop()
                iteration_results.append(result)

            metrics = self._compute_metrics(iteration_results)

            return BenchmarkResult(
                benchmark_name=self.name(),
                config=self.config,
                metrics=metrics,
                raw_times_ms=self.timer.elapsed_times.copy(),
                success=True
            )

        except Exception as e:
            return BenchmarkResult(
                benchmark_name=self.name(),
                config=self.config,
                metrics=[],
                raw_times_ms=[],
                success=False,
                error_message=str(e)
            )

        finally:
            self.teardown()

    def _compute_metrics(
        self,
        iteration_results: list[dict[str, Any]]
    ) -> list[Metric]:
        """Compute metrics from iteration results."""
        metrics = []

        # Time metrics
        metrics.append(Metric(
            name="mean_latency",
            value=self.timer.mean,
            unit=MetricType.TIME_MS
        ))

        # Throughput
        if self.timer.mean > 0:
            throughput = 1000.0 / self.timer.mean  # items/sec
            metrics.append(Metric(
                name="throughput",
                value=throughput,
                unit=MetricType.THROUGHPUT,
                lower_is_better=False
            ))

        # Memory
        self.memory_tracker.snapshot()
        metrics.append(Metric(
            name="peak_memory",
            value=self.memory_tracker.peak_memory_mb,
            unit=MetricType.MEMORY_MB
        ))

        return metrics


@dataclass
class BenchmarkSuite:
    """Collection of benchmarks."""
    name: str
    benchmarks: list[Benchmark] = field(default_factory=list)
    description: str = ""

    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add benchmark to suite."""
        self.benchmarks.append(benchmark)

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmarks in suite."""
        results = []
        for bench in self.benchmarks:
            result = bench.benchmark()
            results.append(result)
        return results


class BenchmarkRegistry:
    """Registry of available benchmarks."""

    def __init__(self):
        self._benchmarks: dict[str, type[Benchmark]] = {}

    def register(
        self,
        name: str,
        benchmark_class: type[Benchmark]
    ) -> None:
        """Register a benchmark."""
        self._benchmarks[name] = benchmark_class

    def get(self, name: str) -> type[Benchmark] | None:
        """Get benchmark class by name."""
        return self._benchmarks.get(name)

    def list_benchmarks(self) -> list[str]:
        """List all registered benchmarks."""
        return list(self._benchmarks.keys())

    def create(
        self,
        name: str,
        config: BenchmarkConfig
    ) -> Benchmark | None:
        """Create benchmark instance."""
        cls = self.get(name)
        if cls:
            return cls(config)
        return None


# Global registry
registry = BenchmarkRegistry()


def register_benchmark(name: str) -> Callable:
    """Decorator to register benchmark."""
    def decorator(cls: type[Benchmark]) -> type[Benchmark]:
        registry.register(name, cls)
        return cls
    return decorator


def save_results(
    results: list[BenchmarkResult],
    filepath: str
) -> None:
    """Save benchmark results to JSON."""
    data = [r.to_dict() for r in results]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_results(filepath: str) -> list[dict]:
    """Load benchmark results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_results(
    baseline: list[BenchmarkResult],
    current: list[BenchmarkResult]
) -> dict[str, dict[str, float]]:
    """Compare two sets of benchmark results."""
    comparison = {}

    baseline_by_name = {r.benchmark_name: r for r in baseline}
    current_by_name = {r.benchmark_name: r for r in current}

    for name in baseline_by_name:
        if name in current_by_name:
            base = baseline_by_name[name]
            curr = current_by_name[name]

            if base.mean_time_ms > 0:
                speedup = base.mean_time_ms / curr.mean_time_ms
            else:
                speedup = 1.0

            comparison[name] = {
                "baseline_ms": base.mean_time_ms,
                "current_ms": curr.mean_time_ms,
                "speedup": speedup,
                "improved": speedup > 1.0
            }

    return comparison
