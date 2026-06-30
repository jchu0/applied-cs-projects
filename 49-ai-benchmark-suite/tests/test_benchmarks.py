"""Tests for individual benchmark types."""

import pytest
import numpy as np

from aibench.core.benchmark import (
    BenchmarkType, MetricType, Metric, BenchmarkConfig, BenchmarkResult,
    Timer, MemoryTracker, Benchmark, BenchmarkSuite, BenchmarkRegistry,
    registry, register_benchmark, save_results, load_results, compare_results
)
from aibench.benchmarks.workloads import (
    LLMInferenceBenchmark, LLMGenerationBenchmark, TrainingThroughputBenchmark,
    MemoryBandwidthBenchmark, MatMulBenchmark, AttentionBenchmark,
    EmbeddingBenchmark, SoftmaxBenchmark
)


class TestBenchmarkEnums:
    """Tests for benchmark enumerations."""

    def test_benchmark_type_values(self):
        """Test BenchmarkType enum values."""
        assert BenchmarkType.INFERENCE.value == "inference"
        assert BenchmarkType.TRAINING.value == "training"
        assert BenchmarkType.MEMORY.value == "memory"
        assert BenchmarkType.THROUGHPUT.value == "throughput"
        assert BenchmarkType.LATENCY.value == "latency"
        assert BenchmarkType.ACCURACY.value == "accuracy"

    def test_metric_type_values(self):
        """Test MetricType enum values."""
        assert MetricType.TIME_MS.value == "time_ms"
        assert MetricType.TIME_S.value == "time_s"
        assert MetricType.THROUGHPUT.value == "throughput"
        assert MetricType.MEMORY_MB.value == "memory_mb"
        assert MetricType.MEMORY_GB.value == "memory_gb"
        assert MetricType.FLOPS.value == "flops"
        assert MetricType.TFLOPS.value == "tflops"
        assert MetricType.ACCURACY.value == "accuracy"
        assert MetricType.LOSS.value == "loss"
        assert MetricType.TOKENS_PER_SEC.value == "tokens/s"
        assert MetricType.SAMPLES_PER_SEC.value == "samples/s"


class TestMetric:
    """Tests for Metric dataclass."""

    def test_metric_creation(self, sample_metric):
        """Test basic metric creation."""
        assert sample_metric.name == "latency"
        assert sample_metric.value == 10.5
        assert sample_metric.unit == MetricType.TIME_MS
        assert sample_metric.lower_is_better is True
        assert sample_metric.metadata == {"device": "cpu"}

    def test_metric_to_dict(self, sample_metric):
        """Test metric serialization to dict."""
        d = sample_metric.to_dict()
        assert d["name"] == "latency"
        assert d["value"] == 10.5
        assert d["unit"] == "time_ms"
        assert d["lower_is_better"] is True
        assert d["metadata"] == {"device": "cpu"}

    def test_metric_default_values(self):
        """Test metric with default values."""
        metric = Metric(name="test", value=1.0, unit=MetricType.TIME_MS)
        assert metric.lower_is_better is True
        assert metric.metadata == {}

    def test_metric_higher_is_better(self):
        """Test metric where higher is better (e.g., throughput)."""
        metric = Metric(
            name="throughput",
            value=1000.0,
            unit=MetricType.THROUGHPUT,
            lower_is_better=False
        )
        assert metric.lower_is_better is False


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_config_creation(self, default_config):
        """Test benchmark config creation."""
        assert default_config.name == "test_benchmark"
        assert default_config.warmup_iterations == 1
        assert default_config.benchmark_iterations == 3
        assert default_config.batch_size == 2
        assert default_config.sequence_length == 64
        assert default_config.device == "cpu"
        assert default_config.precision == "fp32"

    def test_config_defaults(self):
        """Test benchmark config default values."""
        config = BenchmarkConfig(name="minimal")
        assert config.warmup_iterations == 3
        assert config.benchmark_iterations == 10
        assert config.timeout_seconds == 300.0
        assert config.batch_size == 1
        assert config.sequence_length == 512
        assert config.num_threads == 4
        assert config.device == "cpu"
        assert config.precision == "fp32"
        assert config.extra_config == {}

    def test_config_extra_config(self):
        """Test benchmark config with extra configuration."""
        config = BenchmarkConfig(
            name="test",
            extra_config={"model_name": "bert", "hidden_size": 768}
        )
        assert config.extra_config["model_name"] == "bert"
        assert config.extra_config["hidden_size"] == 768


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_result_creation(self, successful_result):
        """Test benchmark result creation."""
        assert successful_result.benchmark_name == "test_benchmark"
        assert successful_result.success is True
        assert len(successful_result.metrics) == 3
        assert len(successful_result.raw_times_ms) == 5

    def test_result_statistics(self, successful_result):
        """Test benchmark result statistical properties."""
        # raw_times_ms = [10.0, 15.0, 12.0, 14.0, 11.0]
        assert successful_result.mean_time_ms == pytest.approx(12.4, rel=0.01)
        # Use population stdev approximation with sample stdev
        assert successful_result.std_time_ms == pytest.approx(2.07, rel=0.05)
        assert successful_result.min_time_ms == 10.0
        assert successful_result.max_time_ms == 15.0
        assert successful_result.p50_time_ms == 12.0  # median

    def test_result_p99(self):
        """Test p99 calculation."""
        config = BenchmarkConfig(name="test")
        result = BenchmarkResult(
            benchmark_name="test",
            config=config,
            metrics=[],
            raw_times_ms=list(range(1, 101))  # 1 to 100
        )
        # p99 should be at index 99 (which is value 100 in 1-indexed list)
        # The implementation uses: int(len * 0.99) = int(100 * 0.99) = 99
        # sorted_times[99] = 100
        assert result.p99_time_ms == 100

    def test_result_empty_times(self):
        """Test result with empty times."""
        config = BenchmarkConfig(name="test")
        result = BenchmarkResult(
            benchmark_name="test",
            config=config,
            metrics=[],
            raw_times_ms=[]
        )
        assert result.mean_time_ms == 0.0
        assert result.std_time_ms == 0.0
        assert result.min_time_ms == 0.0
        assert result.max_time_ms == 0.0
        assert result.p50_time_ms == 0.0
        assert result.p99_time_ms == 0.0

    def test_result_single_time(self):
        """Test result with single time measurement."""
        config = BenchmarkConfig(name="test")
        result = BenchmarkResult(
            benchmark_name="test",
            config=config,
            metrics=[],
            raw_times_ms=[10.0]
        )
        assert result.mean_time_ms == 10.0
        assert result.std_time_ms == 0.0  # Can't compute stdev with 1 sample

    def test_result_to_dict(self, successful_result):
        """Test result serialization to dict."""
        d = successful_result.to_dict()
        assert d["benchmark_name"] == "test_benchmark"
        assert d["success"] is True
        assert "statistics" in d
        assert d["statistics"]["mean_ms"] == pytest.approx(12.4, rel=0.01)
        assert "metrics" in d
        assert len(d["metrics"]) == 3

    def test_failed_result(self, failed_result):
        """Test failed benchmark result."""
        assert failed_result.success is False
        assert failed_result.error_message == "Benchmark failed due to timeout"
        assert len(failed_result.metrics) == 0


class TestTimer:
    """Tests for Timer class."""

    def test_timer_basic(self, timer):
        """Test basic timer functionality."""
        timer.start()
        # Small sleep to ensure measurable time
        import time
        time.sleep(0.01)
        elapsed = timer.stop()

        assert elapsed > 0
        assert len(timer.elapsed_times) == 1
        assert timer.last == elapsed

    def test_timer_multiple_measurements(self, timer):
        """Test timer with multiple measurements."""
        import time

        for _ in range(3):
            timer.start()
            time.sleep(0.001)
            timer.stop()

        assert len(timer.elapsed_times) == 3
        assert timer.mean > 0
        assert all(t > 0 for t in timer.elapsed_times)

    def test_timer_reset(self, timer):
        """Test timer reset."""
        timer.start()
        timer.stop()
        assert len(timer.elapsed_times) == 1

        timer.reset()
        assert len(timer.elapsed_times) == 0
        assert timer.last == 0.0
        assert timer.mean == 0.0

    def test_timer_last_empty(self, timer):
        """Test timer.last when no measurements."""
        assert timer.last == 0.0

    def test_timer_mean_empty(self, timer):
        """Test timer.mean when no measurements."""
        assert timer.mean == 0.0


class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    def test_memory_tracker_snapshot(self, memory_tracker):
        """Test memory snapshot."""
        memory = memory_tracker.snapshot()
        assert memory > 0
        assert memory_tracker.current_memory_mb == memory
        assert memory_tracker.peak_memory_mb >= memory

    def test_memory_tracker_multiple_snapshots(self, memory_tracker):
        """Test multiple memory snapshots."""
        for _ in range(5):
            memory_tracker.snapshot()

        assert len(memory_tracker._measurements) == 5
        assert memory_tracker.peak_memory_mb > 0

    def test_memory_tracker_reset(self, memory_tracker):
        """Test memory tracker reset."""
        memory_tracker.snapshot()
        memory_tracker.snapshot()

        memory_tracker.reset()
        assert len(memory_tracker._measurements) == 0
        assert memory_tracker.peak_memory_mb == 0.0
        assert memory_tracker.current_memory_mb == 0.0


class TestBenchmarkBase:
    """Tests for the base Benchmark class."""

    def test_simple_benchmark_run(self, simple_benchmark):
        """Test running a simple benchmark."""
        result = simple_benchmark.benchmark()

        assert result.success is True
        assert simple_benchmark.setup_called is True
        assert simple_benchmark.teardown_called is True
        assert simple_benchmark.iterations_run >= 1

    def test_benchmark_warmup(self, default_config):
        """Test benchmark warmup iterations."""
        from tests.conftest import SimpleBenchmark

        config = BenchmarkConfig(
            name="warmup_test",
            warmup_iterations=3,
            benchmark_iterations=2
        )
        benchmark = SimpleBenchmark(config)
        result = benchmark.benchmark()

        # warmup + benchmark iterations
        assert benchmark.iterations_run == 5

    def test_failing_benchmark(self, failing_benchmark):
        """Test benchmark that fails during setup."""
        result = failing_benchmark.benchmark()

        assert result.success is False
        assert "Setup failed intentionally" in result.error_message
        assert failing_benchmark.teardown_called is True

    def test_benchmark_metrics(self, simple_benchmark):
        """Test that benchmark produces metrics."""
        result = simple_benchmark.benchmark()

        assert len(result.metrics) > 0
        metric_names = [m.name for m in result.metrics]
        assert "mean_latency" in metric_names
        assert "throughput" in metric_names
        assert "peak_memory" in metric_names


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite class."""

    def test_suite_creation(self):
        """Test benchmark suite creation."""
        suite = BenchmarkSuite(
            name="test_suite",
            description="Test suite"
        )
        assert suite.name == "test_suite"
        assert len(suite.benchmarks) == 0

    def test_suite_add_benchmark(self, benchmark_suite):
        """Test adding benchmarks to suite."""
        assert len(benchmark_suite.benchmarks) == 2

    def test_suite_run_all(self, benchmark_suite):
        """Test running all benchmarks in suite."""
        results = benchmark_suite.run_all()

        assert len(results) == 2
        assert all(isinstance(r, BenchmarkResult) for r in results)


class TestBenchmarkRegistry:
    """Tests for BenchmarkRegistry class."""

    def test_registry_register(self, fresh_registry):
        """Test registering a benchmark."""
        from tests.conftest import SimpleBenchmark

        fresh_registry.register("simple", SimpleBenchmark)
        assert fresh_registry.get("simple") == SimpleBenchmark

    def test_registry_get_nonexistent(self, fresh_registry):
        """Test getting a nonexistent benchmark."""
        assert fresh_registry.get("nonexistent") is None

    def test_registry_list_benchmarks(self, fresh_registry):
        """Test listing benchmarks."""
        from tests.conftest import SimpleBenchmark

        fresh_registry.register("bench1", SimpleBenchmark)
        fresh_registry.register("bench2", SimpleBenchmark)

        benchmarks = fresh_registry.list_benchmarks()
        assert "bench1" in benchmarks
        assert "bench2" in benchmarks

    def test_registry_create(self, fresh_registry, minimal_config):
        """Test creating benchmark from registry."""
        from tests.conftest import SimpleBenchmark

        fresh_registry.register("simple", SimpleBenchmark)
        benchmark = fresh_registry.create("simple", minimal_config)

        assert benchmark is not None
        assert isinstance(benchmark, SimpleBenchmark)

    def test_registry_create_nonexistent(self, fresh_registry, minimal_config):
        """Test creating nonexistent benchmark."""
        benchmark = fresh_registry.create("nonexistent", minimal_config)
        assert benchmark is None

    def test_global_registry(self):
        """Test that global registry has registered benchmarks."""
        benchmarks = registry.list_benchmarks()
        assert "llm_inference" in benchmarks
        assert "llm_generation" in benchmarks
        assert "training_throughput" in benchmarks
        assert "memory_bandwidth" in benchmarks
        assert "matmul" in benchmarks
        assert "attention" in benchmarks
        assert "embedding" in benchmarks
        assert "softmax" in benchmarks


class TestLLMInferenceBenchmark:
    """Tests for LLM inference benchmark."""

    def test_benchmark_name(self, llm_inference_benchmark):
        """Test benchmark name."""
        assert llm_inference_benchmark.name() == "llm_inference"

    def test_benchmark_setup(self, llm_inference_benchmark):
        """Test benchmark setup."""
        llm_inference_benchmark.setup()

        assert llm_inference_benchmark.vocab_size == 32000
        assert llm_inference_benchmark.hidden_size == 2048
        assert llm_inference_benchmark.num_layers == 2
        assert len(llm_inference_benchmark.weights) == 8  # num_layers * 4

    def test_benchmark_run(self, llm_inference_benchmark):
        """Test complete benchmark run."""
        result = llm_inference_benchmark.benchmark()

        assert result.success is True
        assert result.benchmark_name == "llm_inference"

        metric_names = [m.name for m in result.metrics]
        assert "tokens_per_second" in metric_names

    def test_benchmark_teardown(self, llm_inference_benchmark):
        """Test benchmark teardown."""
        llm_inference_benchmark.setup()
        llm_inference_benchmark.teardown()

        assert llm_inference_benchmark.weights == []


class TestLLMGenerationBenchmark:
    """Tests for LLM generation benchmark."""

    def test_benchmark_name(self, llm_generation_benchmark):
        """Test benchmark name."""
        assert llm_generation_benchmark.name() == "llm_generation"

    def test_benchmark_setup(self, llm_generation_benchmark):
        """Test benchmark setup."""
        llm_generation_benchmark.setup()

        assert llm_generation_benchmark.vocab_size == 32000
        assert llm_generation_benchmark.hidden_size == 1024
        assert llm_generation_benchmark.max_new_tokens == 4

    def test_benchmark_run(self, llm_generation_benchmark):
        """Test complete benchmark run."""
        result = llm_generation_benchmark.benchmark()

        assert result.success is True

        metric_names = [m.name for m in result.metrics]
        assert "generation_speed" in metric_names
        assert "time_to_first_token" in metric_names


class TestTrainingThroughputBenchmark:
    """Tests for training throughput benchmark."""

    def test_benchmark_name(self, training_benchmark):
        """Test benchmark name."""
        assert training_benchmark.name() == "training_throughput"

    def test_benchmark_setup(self, training_benchmark):
        """Test benchmark setup."""
        training_benchmark.setup()

        assert training_benchmark.hidden_size == 1024
        assert training_benchmark.num_layers == 6
        assert len(training_benchmark.params) == 24  # num_layers * 4

    def test_benchmark_run(self, training_benchmark):
        """Test complete benchmark run."""
        result = training_benchmark.benchmark()

        assert result.success is True

        metric_names = [m.name for m in result.metrics]
        assert "samples_per_second" in metric_names
        assert "final_loss" in metric_names

    def test_run_iteration_returns_loss(self, training_benchmark):
        """Test that run_iteration returns loss value."""
        training_benchmark.setup()
        result = training_benchmark.run_iteration()

        assert "loss" in result
        assert "samples" in result
        assert result["loss"] >= 0


class TestMemoryBandwidthBenchmark:
    """Tests for memory bandwidth benchmark."""

    def test_benchmark_name(self, memory_bandwidth_benchmark):
        """Test benchmark name."""
        assert memory_bandwidth_benchmark.name() == "memory_bandwidth"

    def test_benchmark_setup(self, memory_bandwidth_benchmark):
        """Test benchmark setup."""
        memory_bandwidth_benchmark.setup()

        assert memory_bandwidth_benchmark.size_bytes == 1 * 1024 * 1024  # 1 MB

    def test_benchmark_run(self, memory_bandwidth_benchmark):
        """Test complete benchmark run."""
        result = memory_bandwidth_benchmark.benchmark()

        assert result.success is True

        metric_names = [m.name for m in result.metrics]
        assert "bandwidth_gbps" in metric_names

    def test_run_iteration_copies_data(self, memory_bandwidth_benchmark):
        """Test that run_iteration performs memory copy."""
        memory_bandwidth_benchmark.setup()
        result = memory_bandwidth_benchmark.run_iteration()

        assert "bytes_copied" in result
        assert result["bytes_copied"] == memory_bandwidth_benchmark.size_bytes
        # Verify copy was performed
        np.testing.assert_array_equal(
            memory_bandwidth_benchmark.src,
            memory_bandwidth_benchmark.dst
        )


class TestMatMulBenchmark:
    """Tests for matrix multiplication benchmark."""

    def test_benchmark_name(self, matmul_benchmark):
        """Test benchmark name."""
        assert matmul_benchmark.name() == "matmul"

    def test_benchmark_setup(self, matmul_benchmark):
        """Test benchmark setup."""
        matmul_benchmark.setup()

        assert matmul_benchmark.m == 64
        assert matmul_benchmark.n == 64
        assert matmul_benchmark.k == 64
        assert matmul_benchmark.a.shape == (64, 64)
        assert matmul_benchmark.b.shape == (64, 64)

    def test_benchmark_run(self, matmul_benchmark):
        """Test complete benchmark run."""
        result = matmul_benchmark.benchmark()

        assert result.success is True

        metric_names = [m.name for m in result.metrics]
        assert "tflops" in metric_names

    def test_run_iteration_computes_flops(self, matmul_benchmark):
        """Test that run_iteration returns FLOPS count."""
        matmul_benchmark.setup()
        result = matmul_benchmark.run_iteration()

        expected_flops = 2 * 64 * 64 * 64
        assert result["flops"] == expected_flops

    def test_fp16_precision(self):
        """Test matmul with FP16 precision."""
        config = BenchmarkConfig(
            name="matmul_fp16",
            warmup_iterations=0,
            benchmark_iterations=1,
            precision="fp16",
            extra_config={"m": 32, "n": 32, "k": 32}
        )
        benchmark = MatMulBenchmark(config)
        benchmark.setup()

        assert benchmark.a.dtype == np.float16
        assert benchmark.b.dtype == np.float16


class TestAttentionBenchmark:
    """Tests for attention benchmark."""

    def test_benchmark_name(self, attention_benchmark):
        """Test benchmark name."""
        assert attention_benchmark.name() == "attention"

    def test_benchmark_setup(self, attention_benchmark):
        """Test benchmark setup."""
        attention_benchmark.setup()

        assert attention_benchmark.batch == 1
        assert attention_benchmark.seq_len == 16
        assert attention_benchmark.heads == 4
        assert attention_benchmark.head_dim == 32

        expected_shape = (1, 4, 16, 32)
        assert attention_benchmark.q.shape == expected_shape
        assert attention_benchmark.k.shape == expected_shape
        assert attention_benchmark.v.shape == expected_shape

    def test_benchmark_run(self, attention_benchmark):
        """Test complete benchmark run."""
        result = attention_benchmark.benchmark()

        assert result.success is True

    def test_run_iteration_computes_attention(self, attention_benchmark):
        """Test that run_iteration computes attention correctly."""
        attention_benchmark.setup()
        result = attention_benchmark.run_iteration()

        assert "flops" in result
        assert "output_shape" in result

        # Output should have same shape as V
        expected_shape = (1, 4, 16, 32)
        assert result["output_shape"] == expected_shape


class TestEmbeddingBenchmark:
    """Tests for embedding benchmark."""

    def test_benchmark_name(self, embedding_benchmark):
        """Test benchmark name."""
        assert embedding_benchmark.name() == "embedding"

    def test_benchmark_setup(self, embedding_benchmark):
        """Test benchmark setup."""
        embedding_benchmark.setup()

        assert embedding_benchmark.vocab_size == 1000
        assert embedding_benchmark.embed_dim == 256
        assert embedding_benchmark.embedding_table.shape == (1000, 256)

    def test_benchmark_run(self, embedding_benchmark):
        """Test complete benchmark run."""
        result = embedding_benchmark.benchmark()

        assert result.success is True

    def test_run_iteration_lookup(self, embedding_benchmark):
        """Test that run_iteration performs embedding lookup."""
        embedding_benchmark.setup()
        result = embedding_benchmark.run_iteration()

        assert "tokens" in result
        assert "output_shape" in result

        expected_tokens = 2 * 16  # batch_size * sequence_length
        assert result["tokens"] == expected_tokens


class TestSoftmaxBenchmark:
    """Tests for softmax benchmark."""

    def test_benchmark_name(self, softmax_benchmark):
        """Test benchmark name."""
        assert softmax_benchmark.name() == "softmax"

    def test_benchmark_setup(self, softmax_benchmark):
        """Test benchmark setup."""
        softmax_benchmark.setup()

        expected_shape = (2, 16, 1000)  # batch, seq_len, vocab_size
        assert softmax_benchmark.input_tensor.shape == expected_shape

    def test_benchmark_run(self, softmax_benchmark):
        """Test complete benchmark run."""
        result = softmax_benchmark.benchmark()

        assert result.success is True

    def test_run_iteration_returns_elements(self, softmax_benchmark):
        """Test that run_iteration returns element count."""
        softmax_benchmark.setup()
        result = softmax_benchmark.run_iteration()

        expected_elements = 2 * 16 * 1000
        assert result["elements"] == expected_elements


class TestSaveLoadResults:
    """Tests for save/load results functions."""

    def test_save_results(self, temp_dir, multiple_results):
        """Test saving results to file."""
        filepath = f"{temp_dir}/results.json"
        save_results(multiple_results, filepath)

        import os
        assert os.path.exists(filepath)

    def test_load_results(self, temp_results_file):
        """Test loading results from file."""
        data = load_results(temp_results_file)

        assert len(data) == 3
        assert data[0]["benchmark_name"] == "benchmark_1"
        assert data[1]["benchmark_name"] == "benchmark_2"

    def test_save_load_roundtrip(self, temp_dir, successful_result):
        """Test save and load roundtrip."""
        filepath = f"{temp_dir}/roundtrip.json"
        save_results([successful_result], filepath)

        data = load_results(filepath)
        assert len(data) == 1
        assert data[0]["benchmark_name"] == successful_result.benchmark_name
        assert data[0]["success"] == successful_result.success


class TestCompareResults:
    """Tests for compare_results function."""

    def test_compare_results_speedup(self, baseline_results, current_results):
        """Test comparison showing speedup."""
        comparison = compare_results(baseline_results, current_results)

        # benchmark_a: 100ms -> 50ms = 2x speedup
        assert "benchmark_a" in comparison
        assert comparison["benchmark_a"]["speedup"] == pytest.approx(2.0, rel=0.01)
        assert comparison["benchmark_a"]["improved"] is True

        # benchmark_b: 200ms -> 220ms = regression
        assert "benchmark_b" in comparison
        assert comparison["benchmark_b"]["speedup"] < 1.0
        assert comparison["benchmark_b"]["improved"] is False

    def test_compare_results_missing_benchmark(self):
        """Test comparison when benchmarks don't match."""
        config = BenchmarkConfig(name="test")

        baseline = [
            BenchmarkResult(
                benchmark_name="only_in_baseline",
                config=config,
                metrics=[],
                raw_times_ms=[100.0]
            )
        ]

        current = [
            BenchmarkResult(
                benchmark_name="only_in_current",
                config=config,
                metrics=[],
                raw_times_ms=[50.0]
            )
        ]

        comparison = compare_results(baseline, current)
        # Should be empty since no matching benchmarks
        assert len(comparison) == 0

    def test_compare_results_zero_baseline(self):
        """Test comparison when baseline has zero time."""
        config = BenchmarkConfig(name="test")

        baseline = [
            BenchmarkResult(
                benchmark_name="bench",
                config=config,
                metrics=[],
                raw_times_ms=[0.0]  # Zero time
            )
        ]

        current = [
            BenchmarkResult(
                benchmark_name="bench",
                config=config,
                metrics=[],
                raw_times_ms=[50.0]
            )
        ]

        comparison = compare_results(baseline, current)
        assert comparison["bench"]["speedup"] == 1.0  # Default to 1.0
