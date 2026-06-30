"""Pytest fixtures for AI Benchmark Suite tests."""

import sys
from pathlib import Path as PathLib

# Add src directory to Python path
src_path = PathLib(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import tempfile
import os
import json
import numpy as np
from pathlib import Path

from aibench.core.benchmark import (
    BenchmarkType, MetricType, Metric, BenchmarkConfig, BenchmarkResult,
    Timer, MemoryTracker, Benchmark, BenchmarkSuite, BenchmarkRegistry,
    registry, save_results, load_results
)
from aibench.benchmarks.workloads import (
    LLMInferenceBenchmark, LLMGenerationBenchmark, TrainingThroughputBenchmark,
    MemoryBandwidthBenchmark, MatMulBenchmark, AttentionBenchmark,
    EmbeddingBenchmark, SoftmaxBenchmark
)
from aibench.runner.runner import RunnerConfig, BenchmarkRunner, BenchmarkScheduler
from aibench.report.report import ReportConfig, ReportGenerator, ComparisonReport, MetricsAnalyzer


# --- Basic Configuration Fixtures ---

@pytest.fixture
def default_config():
    """Create a default benchmark configuration."""
    return BenchmarkConfig(
        name="test_benchmark",
        warmup_iterations=1,
        benchmark_iterations=3,
        timeout_seconds=60.0,
        batch_size=2,
        sequence_length=64,
        num_threads=2,
        device="cpu",
        precision="fp32"
    )


@pytest.fixture
def minimal_config():
    """Create a minimal benchmark configuration for fast tests."""
    return BenchmarkConfig(
        name="minimal_test",
        warmup_iterations=0,
        benchmark_iterations=1,
        timeout_seconds=10.0,
        batch_size=1,
        sequence_length=16,
        num_threads=1,
        device="cpu",
        precision="fp32"
    )


@pytest.fixture
def fp16_config():
    """Create an FP16 benchmark configuration."""
    return BenchmarkConfig(
        name="fp16_test",
        warmup_iterations=1,
        benchmark_iterations=2,
        batch_size=2,
        sequence_length=32,
        device="cpu",
        precision="fp16"
    )


# --- Metric Fixtures ---

@pytest.fixture
def sample_metric():
    """Create a sample metric."""
    return Metric(
        name="latency",
        value=10.5,
        unit=MetricType.TIME_MS,
        lower_is_better=True,
        metadata={"device": "cpu"}
    )


@pytest.fixture
def sample_metrics():
    """Create a list of sample metrics."""
    return [
        Metric(name="mean_latency", value=15.0, unit=MetricType.TIME_MS),
        Metric(name="throughput", value=100.0, unit=MetricType.THROUGHPUT, lower_is_better=False),
        Metric(name="peak_memory", value=256.0, unit=MetricType.MEMORY_MB),
    ]


# --- Timer and Memory Tracker Fixtures ---

@pytest.fixture
def timer():
    """Create a Timer instance."""
    return Timer()


@pytest.fixture
def memory_tracker():
    """Create a MemoryTracker instance."""
    return MemoryTracker()


# --- Benchmark Result Fixtures ---

@pytest.fixture
def successful_result(default_config, sample_metrics):
    """Create a successful benchmark result."""
    return BenchmarkResult(
        benchmark_name="test_benchmark",
        config=default_config,
        metrics=sample_metrics,
        raw_times_ms=[10.0, 15.0, 12.0, 14.0, 11.0],
        success=True,
        error_message=""
    )


@pytest.fixture
def failed_result(default_config):
    """Create a failed benchmark result."""
    return BenchmarkResult(
        benchmark_name="failed_benchmark",
        config=default_config,
        metrics=[],
        raw_times_ms=[],
        success=False,
        error_message="Benchmark failed due to timeout"
    )


@pytest.fixture
def multiple_results(default_config, sample_metrics):
    """Create multiple benchmark results for testing."""
    return [
        BenchmarkResult(
            benchmark_name="benchmark_1",
            config=default_config,
            metrics=sample_metrics,
            raw_times_ms=[10.0, 12.0, 11.0],
            success=True
        ),
        BenchmarkResult(
            benchmark_name="benchmark_2",
            config=default_config,
            metrics=sample_metrics,
            raw_times_ms=[20.0, 22.0, 21.0],
            success=True
        ),
        BenchmarkResult(
            benchmark_name="benchmark_3",
            config=default_config,
            metrics=[],
            raw_times_ms=[],
            success=False,
            error_message="Failed"
        ),
    ]


# --- Temporary Directory Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_output_dir(temp_dir):
    """Create a temporary output directory."""
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir)
    return output_dir


@pytest.fixture
def temp_results_file(temp_dir, multiple_results):
    """Create a temporary file with saved results."""
    filepath = os.path.join(temp_dir, "results.json")
    save_results(multiple_results, filepath)
    return filepath


# --- Runner Configuration Fixtures ---

@pytest.fixture
def runner_config(temp_output_dir):
    """Create a runner configuration."""
    return RunnerConfig(
        output_dir=temp_output_dir,
        parallel=False,
        max_workers=2,
        verbose=False,
        save_results=True
    )


@pytest.fixture
def parallel_runner_config(temp_output_dir):
    """Create a parallel runner configuration."""
    return RunnerConfig(
        output_dir=temp_output_dir,
        parallel=True,
        max_workers=2,
        verbose=False,
        save_results=True
    )


# --- Report Configuration Fixtures ---

@pytest.fixture
def report_config(temp_output_dir):
    """Create a report configuration."""
    return ReportConfig(
        output_dir=temp_output_dir,
        format="markdown",
        include_charts=True,
        include_raw_data=False,
        title="Test Report"
    )


@pytest.fixture
def html_report_config(temp_output_dir):
    """Create an HTML report configuration."""
    return ReportConfig(
        output_dir=temp_output_dir,
        format="html",
        include_charts=True,
        title="HTML Test Report"
    )


@pytest.fixture
def json_report_config(temp_output_dir):
    """Create a JSON report configuration."""
    return ReportConfig(
        output_dir=temp_output_dir,
        format="json",
        title="JSON Test Report"
    )


# --- Benchmark Instance Fixtures ---

@pytest.fixture
def llm_inference_benchmark(minimal_config):
    """Create an LLM inference benchmark instance."""
    config = BenchmarkConfig(
        name="llm_inference_test",
        warmup_iterations=0,
        benchmark_iterations=1,
        batch_size=1,
        sequence_length=16,
        extra_config={"num_layers": 2}
    )
    return LLMInferenceBenchmark(config)


@pytest.fixture
def llm_generation_benchmark(minimal_config):
    """Create an LLM generation benchmark instance."""
    config = BenchmarkConfig(
        name="llm_generation_test",
        warmup_iterations=0,
        benchmark_iterations=1,
        batch_size=1,
        extra_config={"max_new_tokens": 4}
    )
    return LLMGenerationBenchmark(config)


@pytest.fixture
def training_benchmark(minimal_config):
    """Create a training throughput benchmark instance."""
    config = BenchmarkConfig(
        name="training_test",
        warmup_iterations=0,
        benchmark_iterations=1,
        batch_size=2,
        sequence_length=16
    )
    return TrainingThroughputBenchmark(config)


@pytest.fixture
def memory_bandwidth_benchmark(minimal_config):
    """Create a memory bandwidth benchmark instance."""
    config = BenchmarkConfig(
        name="memory_test",
        warmup_iterations=0,
        benchmark_iterations=1,
        extra_config={"size_mb": 1}  # Small size for tests
    )
    return MemoryBandwidthBenchmark(config)


@pytest.fixture
def matmul_benchmark(minimal_config):
    """Create a matrix multiplication benchmark instance."""
    config = BenchmarkConfig(
        name="matmul_test",
        warmup_iterations=0,
        benchmark_iterations=1,
        precision="fp32",
        extra_config={"m": 64, "n": 64, "k": 64}  # Small matrices for tests
    )
    return MatMulBenchmark(config)


@pytest.fixture
def attention_benchmark(minimal_config):
    """Create an attention benchmark instance."""
    config = BenchmarkConfig(
        name="attention_test",
        warmup_iterations=0,
        benchmark_iterations=1,
        batch_size=1,
        sequence_length=16,
        precision="fp32",
        extra_config={"num_heads": 4, "head_dim": 32}
    )
    return AttentionBenchmark(config)


@pytest.fixture
def embedding_benchmark(minimal_config):
    """Create an embedding benchmark instance."""
    config = BenchmarkConfig(
        name="embedding_test",
        warmup_iterations=0,
        benchmark_iterations=1,
        batch_size=2,
        sequence_length=16,
        extra_config={"vocab_size": 1000, "embed_dim": 256}
    )
    return EmbeddingBenchmark(config)


@pytest.fixture
def softmax_benchmark(minimal_config):
    """Create a softmax benchmark instance."""
    config = BenchmarkConfig(
        name="softmax_test",
        warmup_iterations=0,
        benchmark_iterations=1,
        batch_size=2,
        sequence_length=16,
        extra_config={"vocab_size": 1000}
    )
    return SoftmaxBenchmark(config)


# --- Benchmark Suite Fixtures ---

@pytest.fixture
def benchmark_suite(matmul_benchmark, softmax_benchmark):
    """Create a benchmark suite with multiple benchmarks."""
    suite = BenchmarkSuite(
        name="test_suite",
        description="Test benchmark suite"
    )
    suite.add_benchmark(matmul_benchmark)
    suite.add_benchmark(softmax_benchmark)
    return suite


# --- Registry Fixtures ---

@pytest.fixture
def fresh_registry():
    """Create a fresh benchmark registry."""
    return BenchmarkRegistry()


# --- Baseline/Comparison Fixtures ---

@pytest.fixture
def baseline_results(default_config):
    """Create baseline results for comparison tests."""
    return [
        BenchmarkResult(
            benchmark_name="benchmark_a",
            config=default_config,
            metrics=[Metric(name="latency", value=100.0, unit=MetricType.TIME_MS)],
            raw_times_ms=[100.0],
            success=True
        ),
        BenchmarkResult(
            benchmark_name="benchmark_b",
            config=default_config,
            metrics=[Metric(name="latency", value=200.0, unit=MetricType.TIME_MS)],
            raw_times_ms=[200.0],
            success=True
        ),
    ]


@pytest.fixture
def current_results(default_config):
    """Create current results for comparison tests (faster than baseline)."""
    return [
        BenchmarkResult(
            benchmark_name="benchmark_a",
            config=default_config,
            metrics=[Metric(name="latency", value=50.0, unit=MetricType.TIME_MS)],
            raw_times_ms=[50.0],
            success=True
        ),
        BenchmarkResult(
            benchmark_name="benchmark_b",
            config=default_config,
            metrics=[Metric(name="latency", value=220.0, unit=MetricType.TIME_MS)],
            raw_times_ms=[220.0],
            success=True
        ),
    ]


# --- Helper Class for Custom Benchmark Testing ---

class SimpleBenchmark(Benchmark):
    """A simple benchmark implementation for testing."""

    def __init__(self, config: BenchmarkConfig, should_fail: bool = False):
        super().__init__(config)
        self.should_fail = should_fail
        self.setup_called = False
        self.teardown_called = False
        self.iterations_run = 0

    def name(self) -> str:
        return "simple_benchmark"

    def setup(self) -> None:
        self.setup_called = True
        if self.should_fail:
            raise RuntimeError("Setup failed intentionally")

    def run_iteration(self) -> dict:
        self.iterations_run += 1
        return {"value": 42}

    def teardown(self) -> None:
        self.teardown_called = True


@pytest.fixture
def simple_benchmark(minimal_config):
    """Create a simple benchmark for testing."""
    return SimpleBenchmark(minimal_config)


@pytest.fixture
def failing_benchmark(minimal_config):
    """Create a failing benchmark for testing."""
    return SimpleBenchmark(minimal_config, should_fail=True)
