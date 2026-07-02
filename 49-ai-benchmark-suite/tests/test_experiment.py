"""Tests for experiment management module."""

import pytest
import tempfile
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aibench.experiment import (
    WorkloadCategory,
    WorkloadConfig,
    Workload,
    LLMInferenceWorkload,
    RAGWorkload,
    ANNSearchWorkload,
    GPUKernelWorkload,
    ExperimentConfig,
    ExperimentResult,
    ExperimentManager,
    RegressionDetector,
    MultiNodeHarness,
)
from aibench.experiment.experiment import (
    WorkloadResult,
    WorkloadSummary,
    WorkloadRunner,
    Regression,
    NodeConfig,
)


class TestWorkloadCategory:
    """Test WorkloadCategory enum."""

    def test_all_categories(self):
        """Test all category values exist."""
        assert WorkloadCategory.LLM_INFERENCE
        assert WorkloadCategory.RAG
        assert WorkloadCategory.ANN_SEARCH
        assert WorkloadCategory.GPU_KERNEL
        assert WorkloadCategory.TRAINING
        assert WorkloadCategory.DATA_PIPELINE
        assert WorkloadCategory.EMBEDDING


class TestWorkloadConfig:
    """Test WorkloadConfig dataclass."""

    def test_creation(self):
        """Test config creation with defaults."""
        config = WorkloadConfig(
            name="test",
            category=WorkloadCategory.LLM_INFERENCE
        )

        assert config.name == "test"
        assert config.category == WorkloadCategory.LLM_INFERENCE
        assert config.num_requests == 1000
        assert config.concurrency == 1
        assert config.warmup_requests == 100

    def test_with_params(self):
        """Test config with custom parameters."""
        config = WorkloadConfig(
            name="llm_test",
            category=WorkloadCategory.LLM_INFERENCE,
            params={'max_tokens': 256, 'model': 'gpt2'},
            num_requests=500,
            concurrency=4
        )

        assert config.params['max_tokens'] == 256
        assert config.num_requests == 500
        assert config.concurrency == 4


class TestLLMInferenceWorkload:
    """Test LLM inference workload."""

    def test_setup(self):
        """Test workload setup."""
        config = WorkloadConfig(
            name="llm_test",
            category=WorkloadCategory.LLM_INFERENCE,
            params={'max_tokens': 64},
            dataset_size=100
        )

        workload = LLMInferenceWorkload(config)
        workload.setup()

        assert len(workload.prompts) == 100
        assert workload.max_tokens == 64

    def test_run_single(self):
        """Test single inference run."""
        config = WorkloadConfig(
            name="llm_test",
            category=WorkloadCategory.LLM_INFERENCE,
            dataset_size=10
        )

        workload = LLMInferenceWorkload(config)
        workload.setup()

        result = workload.run_single(0)

        assert 'latency_ms' in result
        assert 'input_tokens' in result
        assert 'output_tokens' in result
        assert 'tokens_per_second' in result

    def test_teardown(self):
        """Test workload cleanup."""
        config = WorkloadConfig(
            name="llm_test",
            category=WorkloadCategory.LLM_INFERENCE
        )

        workload = LLMInferenceWorkload(config)
        workload.setup()
        workload.teardown()

        assert workload.prompts == []


class TestRAGWorkload:
    """Test RAG workload."""

    def test_setup(self):
        """Test RAG workload setup."""
        config = WorkloadConfig(
            name="rag_test",
            category=WorkloadCategory.RAG,
            params={'top_k': 10},
            dataset_size=50
        )

        workload = RAGWorkload(config)
        workload.setup()

        assert len(workload.queries) == 50
        assert workload.top_k == 10

    def test_run_single(self):
        """Test single RAG run."""
        config = WorkloadConfig(
            name="rag_test",
            category=WorkloadCategory.RAG,
            dataset_size=10
        )

        workload = RAGWorkload(config)
        workload.setup()

        result = workload.run_single(0)

        assert 'total_latency_ms' in result
        assert 'retrieval_latency_ms' in result
        assert 'generation_latency_ms' in result
        assert 'num_docs_retrieved' in result


class TestANNSearchWorkload:
    """Test ANN search workload."""

    def test_setup(self):
        """Test ANN workload setup."""
        config = WorkloadConfig(
            name="ann_test",
            category=WorkloadCategory.ANN_SEARCH,
            params={'dim': 64, 'k': 5},
            dataset_size=100
        )

        workload = ANNSearchWorkload(config)
        workload.setup()

        assert workload.queries.shape == (100, 64)
        assert workload.k == 5

    def test_run_single(self):
        """Test single ANN search."""
        config = WorkloadConfig(
            name="ann_test",
            category=WorkloadCategory.ANN_SEARCH,
            params={'dim': 32},
            dataset_size=10
        )

        workload = ANNSearchWorkload(config)
        workload.setup()

        result = workload.run_single(0)

        assert 'latency_ms' in result
        assert 'recall' in result


class TestGPUKernelWorkload:
    """Test GPU kernel workload."""

    def test_setup(self):
        """Test GPU kernel setup."""
        config = WorkloadConfig(
            name="kernel_test",
            category=WorkloadCategory.GPU_KERNEL,
            params={'kernel': 'matmul', 'm': 1024, 'n': 1024, 'k': 1024}
        )

        workload = GPUKernelWorkload(config)
        workload.setup()

        assert workload.kernel_type == 'matmul'
        assert 'A' in workload.inputs
        assert 'B' in workload.inputs

    def test_run_single(self):
        """Test single kernel run."""
        config = WorkloadConfig(
            name="kernel_test",
            category=WorkloadCategory.GPU_KERNEL,
            params={'m': 512, 'n': 512, 'k': 512}
        )

        workload = GPUKernelWorkload(config)
        workload.setup()

        result = workload.run_single(0)

        assert 'latency_ms' in result
        assert 'tflops' in result


class TestWorkloadRunner:
    """Test WorkloadRunner."""

    def test_sequential_run(self):
        """Test sequential workload execution."""
        config = WorkloadConfig(
            name="test",
            category=WorkloadCategory.LLM_INFERENCE,
            num_requests=10,
            warmup_requests=2,
            dataset_size=5
        )

        workload = LLMInferenceWorkload(config)
        runner = WorkloadRunner()

        summary = runner.run(workload)

        assert summary.workload_name == "test"
        assert summary.total_requests == 10
        assert summary.successful_requests <= 10
        assert summary.requests_per_second > 0

    def test_concurrent_run(self):
        """Test concurrent workload execution."""
        config = WorkloadConfig(
            name="concurrent_test",
            category=WorkloadCategory.LLM_INFERENCE,
            num_requests=20,
            concurrency=4,
            warmup_requests=2,
            dataset_size=5
        )

        workload = LLMInferenceWorkload(config)
        runner = WorkloadRunner()

        summary = runner.run(workload)

        assert summary.total_requests == 20

    def test_progress_callback(self):
        """Test progress callback."""
        progress_calls = []

        def callback(completed, total):
            progress_calls.append((completed, total))

        config = WorkloadConfig(
            name="progress_test",
            category=WorkloadCategory.LLM_INFERENCE,
            num_requests=200,
            warmup_requests=0,
            dataset_size=5
        )

        workload = LLMInferenceWorkload(config)
        runner = WorkloadRunner()
        runner.run(workload, progress_callback=callback)

        # Should have some progress calls
        assert len(progress_calls) > 0


class _WarmupFailingWorkload(Workload):
    """Workload whose run_single fails only during warmup requests."""

    def __init__(self, config, warmup_count):
        super().__init__(config)
        self._warmup_count = warmup_count

    def setup(self):
        pass

    def run_single(self, request_id):
        # First `warmup_count` calls correspond to warmup and must fail.
        if request_id < self._warmup_count:
            raise RuntimeError("warmup boom")
        return {"latency_ms": 1.0}

    def teardown(self):
        pass


class TestWarmupFailureLogging:
    """Warmup failures must be logged, not silently swallowed."""

    def test_warmup_failure_is_logged(self, caplog):
        config = WorkloadConfig(
            name="warmup_fail",
            category=WorkloadCategory.LLM_INFERENCE,
            num_requests=3,
            warmup_requests=2,
            dataset_size=5,
        )
        workload = _WarmupFailingWorkload(config, warmup_count=2)
        runner = WorkloadRunner()

        with caplog.at_level("WARNING", logger="aibench.experiment.experiment"):
            summary = runner.run(workload)

        # Warmup failures surfaced via logging.
        warmup_logs = [r for r in caplog.records if "Warmup request" in r.message]
        assert len(warmup_logs) == 2
        # The run itself still completes (warmup failures are non-fatal).
        assert summary.total_requests == 3


class TestExperimentConfig:
    """Test ExperimentConfig."""

    def test_creation(self):
        """Test config creation."""
        workloads = [
            WorkloadConfig(name="w1", category=WorkloadCategory.LLM_INFERENCE),
            WorkloadConfig(name="w2", category=WorkloadCategory.RAG),
        ]

        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            workloads=workloads,
            random_seed=123
        )

        assert config.name == "test_experiment"
        assert len(config.workloads) == 2
        assert config.random_seed == 123


class TestExperimentManager:
    """Test ExperimentManager."""

    def test_creation(self):
        """Test manager creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(results_dir=tmpdir)
            assert manager.results_dir.exists()

    def test_run_experiment(self):
        """Test running an experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(results_dir=tmpdir)

            config = ExperimentConfig(
                name="test_exp",
                workloads=[
                    WorkloadConfig(
                        name="small_test",
                        category=WorkloadCategory.LLM_INFERENCE,
                        num_requests=5,
                        warmup_requests=1,
                        dataset_size=3
                    )
                ]
            )

            result = manager.run_experiment(config)

            assert result.config.name == "test_exp"
            assert len(result.summaries) == 1
            assert result.start_time < result.end_time

    def test_save_and_load(self):
        """Test saving and loading results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(results_dir=tmpdir)

            config = ExperimentConfig(
                name="save_test",
                workloads=[
                    WorkloadConfig(
                        name="w1",
                        category=WorkloadCategory.LLM_INFERENCE,
                        num_requests=3,
                        warmup_requests=0,
                        dataset_size=2
                    )
                ]
            )

            result = manager.run_experiment(config)

            # Find saved file
            saved_files = list(manager.results_dir.glob("*.json"))
            assert len(saved_files) == 1

            # Load it back
            loaded = manager.load_result(str(saved_files[0]))

            assert loaded.config.name == "save_test"
            assert len(loaded.summaries) == 1

    def test_compare_results(self):
        """Test comparing experiment results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(results_dir=tmpdir)

            # Run two experiments
            config1 = ExperimentConfig(
                name="baseline",
                workloads=[
                    WorkloadConfig(
                        name="test_workload",
                        category=WorkloadCategory.LLM_INFERENCE,
                        num_requests=5,
                        warmup_requests=0,
                        dataset_size=2
                    )
                ]
            )

            config2 = ExperimentConfig(
                name="current",
                workloads=[
                    WorkloadConfig(
                        name="test_workload",
                        category=WorkloadCategory.LLM_INFERENCE,
                        num_requests=5,
                        warmup_requests=0,
                        dataset_size=2
                    )
                ]
            )

            result1 = manager.run_experiment(config1)
            result2 = manager.run_experiment(config2)

            comparison = manager.compare_results(result1, result2)

            assert comparison['baseline'] == "baseline"
            assert comparison['compare'] == "current"
            assert 'test_workload' in comparison['workloads']


class TestRegressionDetector:
    """Test RegressionDetector."""

    def test_no_regressions(self):
        """Test when no regressions exist."""
        # Create mock results with same performance
        baseline = ExperimentResult(
            config=ExperimentConfig(name="baseline"),
            summaries=[
                WorkloadSummary(
                    workload_name="test",
                    total_requests=100,
                    successful_requests=100,
                    failed_requests=0,
                    duration_seconds=10,
                    latency_p50=10,
                    latency_p90=15,
                    latency_p99=20,
                    latency_mean=12,
                    latency_std=2,
                    requests_per_second=10
                )
            ],
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        current = ExperimentResult(
            config=ExperimentConfig(name="current"),
            summaries=[
                WorkloadSummary(
                    workload_name="test",
                    total_requests=100,
                    successful_requests=100,
                    failed_requests=0,
                    duration_seconds=10,
                    latency_p50=10,
                    latency_p90=15,
                    latency_p99=20,
                    latency_mean=12,
                    latency_std=2,
                    requests_per_second=10
                )
            ],
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        detector = RegressionDetector()
        regressions = detector.detect(baseline, current)

        assert len(regressions) == 0

    def test_throughput_regression(self):
        """Test throughput regression detection."""
        baseline = ExperimentResult(
            config=ExperimentConfig(name="baseline"),
            summaries=[
                WorkloadSummary(
                    workload_name="test",
                    total_requests=100,
                    successful_requests=100,
                    failed_requests=0,
                    duration_seconds=10,
                    latency_p50=10,
                    latency_p90=15,
                    latency_p99=20,
                    latency_mean=12,
                    latency_std=2,
                    requests_per_second=100  # Baseline: 100 req/s
                )
            ],
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        current = ExperimentResult(
            config=ExperimentConfig(name="current"),
            summaries=[
                WorkloadSummary(
                    workload_name="test",
                    total_requests=100,
                    successful_requests=100,
                    failed_requests=0,
                    duration_seconds=10,
                    latency_p50=10,
                    latency_p90=15,
                    latency_p99=20,
                    latency_mean=12,
                    latency_std=2,
                    requests_per_second=80  # Current: 80 req/s (20% regression)
                )
            ],
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        detector = RegressionDetector(throughput_threshold=0.05)
        regressions = detector.detect(baseline, current)

        assert len(regressions) >= 1
        throughput_regs = [r for r in regressions if r.metric == 'throughput']
        assert len(throughput_regs) == 1
        assert throughput_regs[0].change_pct < 0

    def test_latency_regression(self):
        """Test latency regression detection."""
        baseline = ExperimentResult(
            config=ExperimentConfig(name="baseline"),
            summaries=[
                WorkloadSummary(
                    workload_name="test",
                    total_requests=100,
                    successful_requests=100,
                    failed_requests=0,
                    duration_seconds=10,
                    latency_p50=10,
                    latency_p90=15,
                    latency_p99=20,  # Baseline p99
                    latency_mean=12,
                    latency_std=2,
                    requests_per_second=100
                )
            ],
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        current = ExperimentResult(
            config=ExperimentConfig(name="current"),
            summaries=[
                WorkloadSummary(
                    workload_name="test",
                    total_requests=100,
                    successful_requests=100,
                    failed_requests=0,
                    duration_seconds=10,
                    latency_p50=10,
                    latency_p90=18,
                    latency_p99=30,  # Current p99 (50% regression)
                    latency_mean=14,
                    latency_std=3,
                    requests_per_second=100
                )
            ],
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        detector = RegressionDetector(latency_threshold=0.10)
        regressions = detector.detect(baseline, current)

        latency_regs = [r for r in regressions if r.metric == 'latency_p99']
        assert len(latency_regs) == 1
        assert latency_regs[0].change_pct > 0

    def test_severity_classification(self):
        """Test regression severity classification."""
        detector = RegressionDetector(
            minor_threshold=0.05,
            moderate_threshold=0.15,
            severe_threshold=0.30
        )

        assert detector._classify_severity(0.03) == 'minor'
        assert detector._classify_severity(0.10) == 'minor'
        assert detector._classify_severity(0.20) == 'moderate'
        assert detector._classify_severity(0.50) == 'severe'

    def test_generate_report(self):
        """Test report generation."""
        detector = RegressionDetector()

        regressions = [
            Regression(
                workload="test1",
                metric="throughput",
                baseline_value=100,
                current_value=70,
                change_pct=-30,
                severity="severe"
            ),
            Regression(
                workload="test2",
                metric="latency_p99",
                baseline_value=20,
                current_value=24,
                change_pct=20,
                severity="moderate"
            ),
        ]

        report = detector.generate_report(regressions)

        assert "Performance Regression Report" in report
        assert "SEVERE" in report
        assert "test1" in report
        assert "MODERATE" in report
        assert "test2" in report

    def test_no_regressions_report(self):
        """Test report when no regressions."""
        detector = RegressionDetector()
        report = detector.generate_report([])

        assert "No regressions detected" in report


class TestMultiNodeHarness:
    """Test MultiNodeHarness."""

    def test_creation(self):
        """Test harness creation."""
        nodes = [
            NodeConfig(host="node1.example.com"),
            NodeConfig(host="node2.example.com"),
        ]

        harness = MultiNodeHarness(nodes)

        assert len(harness.nodes) == 2

    def test_compare_nodes_empty(self):
        """Test node comparison with no results."""
        harness = MultiNodeHarness([])
        comparison = harness.compare_nodes()

        assert comparison == {}

    def test_aggregate_empty_raises(self):
        """Test aggregation with no results raises."""
        harness = MultiNodeHarness([])

        with pytest.raises(ValueError):
            harness.aggregate_results()

    def test_node_config_defaults(self):
        """Test NodeConfig default field values."""
        nc = NodeConfig(host="myhost")
        assert nc.host == "myhost"
        assert nc.port == 22
        assert nc.username == ""
        assert nc.ssh_key_path == ""
        assert nc.working_dir == "/tmp/benchmark"

    def test_run_distributed_sequential(self):
        """Test sequential distributed execution."""
        nodes = [
            NodeConfig(host="node-a"),
            NodeConfig(host="node-b"),
        ]
        harness = MultiNodeHarness(nodes)

        config = ExperimentConfig(
            name="dist_test",
            workloads=[
                WorkloadConfig(
                    name="llm",
                    category=WorkloadCategory.LLM_INFERENCE,
                    num_requests=3,
                    warmup_requests=1,
                    dataset_size=2,
                )
            ],
        )

        with tempfile.TemporaryDirectory():
            results = harness.run_distributed(config, parallel=False)

        assert len(results) == 2
        assert "node-a" in results
        assert "node-b" in results

    def test_run_distributed_parallel(self):
        """Test parallel distributed execution."""
        nodes = [
            NodeConfig(host="p-node-1"),
            NodeConfig(host="p-node-2"),
        ]
        harness = MultiNodeHarness(nodes)

        config = ExperimentConfig(
            name="par_test",
            workloads=[
                WorkloadConfig(
                    name="rag",
                    category=WorkloadCategory.RAG,
                    num_requests=3,
                    warmup_requests=1,
                    dataset_size=2,
                )
            ],
        )

        results = harness.run_distributed(config, parallel=True)

        assert len(results) == 2
        assert "p-node-1" in results
        assert "p-node-2" in results

    def test_run_distributed_returns_experiment_results(self):
        """Test that distributed results contain valid ExperimentResult objects."""
        nodes = [NodeConfig(host="r-node")]
        harness = MultiNodeHarness(nodes)

        config = ExperimentConfig(
            name="result_test",
            workloads=[
                WorkloadConfig(
                    name="ann",
                    category=WorkloadCategory.ANN_SEARCH,
                    num_requests=3,
                    warmup_requests=1,
                    dataset_size=2,
                )
            ],
        )

        results = harness.run_distributed(config, parallel=False)
        result = results["r-node"]

        assert isinstance(result, ExperimentResult)
        assert len(result.summaries) == 1
        assert result.summaries[0].total_requests > 0
        assert result.start_time < result.end_time

    def test_aggregate_results_with_data(self):
        """Test aggregation with actual distributed results."""
        nodes = [
            NodeConfig(host="agg-a"),
            NodeConfig(host="agg-b"),
        ]
        harness = MultiNodeHarness(nodes)

        config = ExperimentConfig(
            name="agg_test",
            workloads=[
                WorkloadConfig(
                    name="llm",
                    category=WorkloadCategory.LLM_INFERENCE,
                    num_requests=5,
                    warmup_requests=1,
                    dataset_size=2,
                )
            ],
        )

        harness.run_distributed(config, parallel=False)
        summary = harness.aggregate_results()

        assert summary.workload_name.endswith("_aggregate")
        assert summary.total_requests > 0
        assert summary.custom_metrics['num_nodes'] == 2
        assert summary.requests_per_second > 0

    def test_aggregate_results_explicit_param(self):
        """Test aggregation with explicitly passed results dict."""
        nodes = [NodeConfig(host="exp-node")]
        harness = MultiNodeHarness(nodes)

        config = ExperimentConfig(
            name="exp_agg",
            workloads=[
                WorkloadConfig(
                    name="gpu",
                    category=WorkloadCategory.GPU_KERNEL,
                    num_requests=3,
                    warmup_requests=1,
                    dataset_size=2,
                )
            ],
        )

        results = harness.run_distributed(config, parallel=False)
        summary = harness.aggregate_results(results=results)

        assert summary.total_requests > 0

    def test_compare_nodes_with_data(self):
        """Test node comparison with actual results."""
        nodes = [
            NodeConfig(host="cmp-1"),
            NodeConfig(host="cmp-2"),
        ]
        harness = MultiNodeHarness(nodes)

        config = ExperimentConfig(
            name="cmp_test",
            workloads=[
                WorkloadConfig(
                    name="llm",
                    category=WorkloadCategory.LLM_INFERENCE,
                    num_requests=3,
                    warmup_requests=1,
                    dataset_size=2,
                )
            ],
        )

        harness.run_distributed(config, parallel=True)
        comparison = harness.compare_nodes()

        assert 'nodes' in comparison
        assert 'rankings' in comparison
        assert 'by_throughput' in comparison['rankings']
        assert len(comparison['rankings']['by_throughput']) == 2
        assert "cmp-1" in comparison['nodes']
        assert "cmp-2" in comparison['nodes']

    def test_run_distributed_multiple_workloads(self):
        """Test distributed execution with multiple workloads."""
        nodes = [NodeConfig(host="multi-wl")]
        harness = MultiNodeHarness(nodes)

        config = ExperimentConfig(
            name="multi_wl_test",
            workloads=[
                WorkloadConfig(
                    name="llm",
                    category=WorkloadCategory.LLM_INFERENCE,
                    num_requests=3,
                    warmup_requests=1,
                    dataset_size=2,
                ),
                WorkloadConfig(
                    name="rag",
                    category=WorkloadCategory.RAG,
                    num_requests=3,
                    warmup_requests=1,
                    dataset_size=2,
                ),
            ],
        )

        results = harness.run_distributed(config, parallel=False)
        result = results["multi-wl"]
        assert len(result.summaries) == 2

    def test_run_distributed_stores_results(self):
        """Test that run_distributed stores results in _results."""
        nodes = [NodeConfig(host="store-node")]
        harness = MultiNodeHarness(nodes)

        config = ExperimentConfig(
            name="store_test",
            workloads=[
                WorkloadConfig(
                    name="ann",
                    category=WorkloadCategory.ANN_SEARCH,
                    num_requests=3,
                    warmup_requests=1,
                    dataset_size=2,
                )
            ],
        )

        assert len(harness._results) == 0
        harness.run_distributed(config, parallel=False)
        assert len(harness._results) == 1
        assert "store-node" in harness._results


class TestRegressionDetectorEdgeCases:
    """Edge case tests for RegressionDetector."""

    def _make_summary(self, name, throughput=100.0, p50=10.0, p99=50.0):
        return WorkloadSummary(
            workload_name=name,
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            duration_seconds=1.0,
            latency_p50=p50,
            latency_p90=30.0,
            latency_p99=p99,
            latency_mean=20.0,
            latency_std=5.0,
            requests_per_second=throughput,
        )

    def _make_result(self, summaries):
        return ExperimentResult(
            config=ExperimentConfig(name="test"),
            summaries=summaries,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    def test_no_matching_workloads(self):
        """Baseline and current have different workload names."""
        detector = RegressionDetector()
        baseline = self._make_result([self._make_summary("workload_a")])
        current = self._make_result([self._make_summary("workload_b")])

        regressions = detector.detect(baseline, current)
        assert len(regressions) == 0

    def test_improvement_not_flagged(self):
        """Performance improvements should not be flagged as regressions."""
        detector = RegressionDetector()
        baseline = self._make_result([self._make_summary("wl", throughput=100, p99=50)])
        current = self._make_result([self._make_summary("wl", throughput=200, p99=25)])

        regressions = detector.detect(baseline, current)
        assert len(regressions) == 0

    def test_p50_latency_regression(self):
        """Detect p50 latency regression."""
        detector = RegressionDetector()
        baseline = self._make_result([self._make_summary("wl", p50=10.0)])
        current = self._make_result([self._make_summary("wl", p50=20.0)])

        regressions = detector.detect(baseline, current)
        p50_regs = [r for r in regressions if r.metric == 'latency_p50']
        assert len(p50_regs) == 1
        assert p50_regs[0].change_pct > 0

    def test_throughput_regression(self):
        """Detect throughput regression."""
        detector = RegressionDetector()
        baseline = self._make_result([self._make_summary("wl", throughput=100)])
        current = self._make_result([self._make_summary("wl", throughput=50)])

        regressions = detector.detect(baseline, current)
        tp_regs = [r for r in regressions if r.metric == 'throughput']
        assert len(tp_regs) == 1
        assert tp_regs[0].change_pct < 0

    def test_severity_classification(self):
        """Test that regression severity is classified correctly."""
        detector = RegressionDetector()
        # 50% throughput drop = severe (>30%)
        baseline = self._make_result([self._make_summary("wl", throughput=100)])
        current = self._make_result([self._make_summary("wl", throughput=40)])

        regressions = detector.detect(baseline, current)
        tp_regs = [r for r in regressions if r.metric == 'throughput']
        assert len(tp_regs) == 1
        assert tp_regs[0].severity == 'severe'

    def test_report_with_minor_regressions(self):
        """Test that minor regressions appear in the report."""
        detector = RegressionDetector()
        # 8% throughput drop = minor
        baseline = self._make_result([self._make_summary("wl", throughput=100)])
        current = self._make_result([self._make_summary("wl", throughput=90)])

        regressions = detector.detect(baseline, current)
        report = detector.generate_report(regressions)
        assert "MINOR" in report

    def test_zero_baseline_no_crash(self):
        """Zero baseline values should not cause division by zero."""
        detector = RegressionDetector()
        baseline = self._make_result([self._make_summary("wl", throughput=0, p50=0, p99=0)])
        current = self._make_result([self._make_summary("wl", throughput=100, p50=10, p99=50)])

        regressions = detector.detect(baseline, current)
        # Should not crash, and no regressions (can't compute % change from 0)
        assert isinstance(regressions, list)


class TestIntegration:
    """Integration tests for experiment module."""

    def test_full_experiment_workflow(self):
        """Test complete experiment workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(results_dir=tmpdir)

            # Create experiment with multiple workloads
            config = ExperimentConfig(
                name="integration_test",
                description="Full integration test",
                workloads=[
                    WorkloadConfig(
                        name="llm",
                        category=WorkloadCategory.LLM_INFERENCE,
                        num_requests=5,
                        warmup_requests=1,
                        dataset_size=3
                    ),
                    WorkloadConfig(
                        name="rag",
                        category=WorkloadCategory.RAG,
                        num_requests=5,
                        warmup_requests=1,
                        dataset_size=3
                    ),
                ]
            )

            # Run experiment
            result = manager.run_experiment(config)

            # Verify results
            assert len(result.summaries) == 2
            assert result.config.system_info  # System info collected

            # Check summaries have valid data
            for summary in result.summaries:
                assert summary.total_requests > 0
                assert summary.latency_mean > 0

    def test_regression_detection_workflow(self):
        """Test regression detection in workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(results_dir=tmpdir)
            detector = RegressionDetector()

            # Run baseline
            baseline_config = ExperimentConfig(
                name="baseline",
                workloads=[
                    WorkloadConfig(
                        name="test",
                        category=WorkloadCategory.LLM_INFERENCE,
                        num_requests=10,
                        warmup_requests=1,
                        dataset_size=5
                    )
                ]
            )
            baseline = manager.run_experiment(baseline_config)

            # Run current
            current_config = ExperimentConfig(
                name="current",
                workloads=[
                    WorkloadConfig(
                        name="test",
                        category=WorkloadCategory.LLM_INFERENCE,
                        num_requests=10,
                        warmup_requests=1,
                        dataset_size=5
                    )
                ]
            )
            current = manager.run_experiment(current_config)

            # Detect regressions
            regressions = detector.detect(baseline, current)

            # Generate report (should work even with no regressions)
            report = detector.generate_report(regressions)
            assert isinstance(report, str)
