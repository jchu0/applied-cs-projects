"""Tests for benchmark runner functionality."""

import pytest
import os
import json
import tempfile

from aibench.core.benchmark import (
    BenchmarkConfig, BenchmarkResult, BenchmarkSuite, registry
)
from aibench.runner.runner import (
    RunnerConfig, BenchmarkRunner, CLIRunner, BenchmarkScheduler, run_benchmarks,
    main,
)
from aibench.benchmarks.workloads import MatMulBenchmark, SoftmaxBenchmark


class TestRunnerConfig:
    """Tests for RunnerConfig dataclass."""

    def test_default_config(self):
        """Test default runner configuration."""
        config = RunnerConfig()

        assert config.output_dir == "./benchmark_results"
        assert config.parallel is False
        assert config.max_workers == 4
        assert config.verbose is True
        assert config.save_results is True
        assert config.compare_baseline is None
        assert config.filter_benchmarks is None

    def test_custom_config(self, temp_output_dir):
        """Test custom runner configuration."""
        config = RunnerConfig(
            output_dir=temp_output_dir,
            parallel=True,
            max_workers=8,
            verbose=False,
            filter_benchmarks=["matmul", "softmax"]
        )

        assert config.output_dir == temp_output_dir
        assert config.parallel is True
        assert config.max_workers == 8
        assert config.verbose is False
        assert config.filter_benchmarks == ["matmul", "softmax"]


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_runner_creation(self, runner_config):
        """Test runner creation."""
        runner = BenchmarkRunner(runner_config)

        assert runner.config == runner_config
        assert len(runner.results) == 0
        assert os.path.exists(runner_config.output_dir)

    def test_runner_creates_output_dir(self, temp_dir):
        """Test runner creates output directory if it doesn't exist."""
        output_dir = os.path.join(temp_dir, "new_dir", "nested")
        config = RunnerConfig(output_dir=output_dir)

        runner = BenchmarkRunner(config)

        assert os.path.exists(output_dir)

    def test_run_single_benchmark(self, runner_config, matmul_benchmark):
        """Test running a single benchmark."""
        runner = BenchmarkRunner(runner_config)
        result = runner.run_benchmark(matmul_benchmark)

        assert result.success is True
        assert result.benchmark_name == "matmul"

    def test_run_benchmark_failure(self, runner_config, failing_benchmark):
        """Test running a failing benchmark."""
        runner = BenchmarkRunner(runner_config)
        result = runner.run_benchmark(failing_benchmark)

        assert result.success is False
        assert len(result.error_message) > 0

    def test_run_suite(self, runner_config, benchmark_suite):
        """Test running a benchmark suite."""
        runner = BenchmarkRunner(runner_config)
        results = runner.run_suite(benchmark_suite)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert len(runner.results) == 2

    def test_run_suite_parallel(self, parallel_runner_config, benchmark_suite):
        """Test running a benchmark suite in parallel."""
        runner = BenchmarkRunner(parallel_runner_config)
        results = runner.run_suite(benchmark_suite)

        assert len(results) == 2
        # All should complete (order may vary)
        assert len([r for r in results if r.success]) == 2

    def test_run_from_registry(self, runner_config, minimal_config):
        """Test running benchmarks from registry."""
        runner = BenchmarkRunner(runner_config)
        results = runner.run_from_registry(
            ["matmul", "softmax"],
            minimal_config
        )

        assert len(results) == 2
        benchmark_names = [r.benchmark_name for r in results]
        assert "matmul" in benchmark_names
        assert "softmax" in benchmark_names

    def test_run_from_registry_filtered(self, runner_config, minimal_config):
        """Test running filtered benchmarks from registry."""
        # Set filter in config
        runner_config.filter_benchmarks = ["matmul"]

        runner = BenchmarkRunner(runner_config)
        results = runner.run_from_registry(
            ["matmul", "softmax"],
            minimal_config
        )

        assert len(results) == 1
        assert results[0].benchmark_name == "matmul"

    def test_run_from_registry_nonexistent(self, runner_config, minimal_config):
        """Test running nonexistent benchmark from registry."""
        runner = BenchmarkRunner(runner_config)
        results = runner.run_from_registry(
            ["nonexistent_benchmark"],
            minimal_config
        )

        assert len(results) == 0

    def test_save_results(self, runner_config, minimal_config):
        """Test saving results to file."""
        runner = BenchmarkRunner(runner_config)
        # run_from_registry adds results to runner.results
        runner.run_from_registry(["matmul"], minimal_config)

        filepath = runner.save("test_results.json")

        assert os.path.exists(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["benchmark_name"] == "matmul"

    def test_save_results_auto_filename(self, runner_config, matmul_benchmark):
        """Test saving results with auto-generated filename."""
        runner = BenchmarkRunner(runner_config)
        runner.run_benchmark(matmul_benchmark)

        filepath = runner.save()

        assert os.path.exists(filepath)
        assert filepath.endswith(".json")
        assert "results_" in os.path.basename(filepath)

    def test_compare_with_baseline(self, runner_config, temp_dir, minimal_config):
        """Test comparing results with baseline."""
        # Create baseline file
        baseline_data = [
            {
                "benchmark_name": "matmul",
                "config": {"name": "test"},
                "success": True,
                "statistics": {"mean_ms": 100.0}
            }
        ]
        baseline_path = os.path.join(temp_dir, "baseline.json")
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f)

        # Run current benchmarks
        runner = BenchmarkRunner(runner_config)
        runner.run_from_registry(["matmul"], minimal_config)

        # Compare
        comparison = runner.compare_with_baseline(baseline_path)

        assert "matmul" in comparison
        assert "speedup" in comparison["matmul"]
        assert "baseline_ms" in comparison["matmul"]
        assert "current_ms" in comparison["matmul"]


class TestBenchmarkRunnerHooks:
    """Tests for BenchmarkRunner hooks."""

    def test_add_hook(self, runner_config):
        """Test adding a hook."""
        runner = BenchmarkRunner(runner_config)

        hook_called = []

        def my_hook(**kwargs):
            hook_called.append(True)

        runner.add_hook("before_benchmark", my_hook)
        assert len(runner._hooks["before_benchmark"]) == 1

    def test_before_benchmark_hook(self, runner_config, matmul_benchmark):
        """Test before_benchmark hook is called."""
        runner = BenchmarkRunner(runner_config)

        hook_data = {}

        def before_hook(benchmark):
            hook_data["called"] = True
            hook_data["benchmark_name"] = benchmark.name()

        runner.add_hook("before_benchmark", before_hook)
        runner.run_benchmark(matmul_benchmark)

        assert hook_data["called"] is True
        assert hook_data["benchmark_name"] == "matmul"

    def test_after_benchmark_hook(self, runner_config, matmul_benchmark):
        """Test after_benchmark hook is called."""
        runner = BenchmarkRunner(runner_config)

        hook_data = {}

        def after_hook(benchmark, result):
            hook_data["called"] = True
            hook_data["success"] = result.success

        runner.add_hook("after_benchmark", after_hook)
        runner.run_benchmark(matmul_benchmark)

        assert hook_data["called"] is True
        assert hook_data["success"] is True

    def test_on_error_hook(self, runner_config, failing_benchmark):
        """Test on_error hook is called on failure."""
        runner = BenchmarkRunner(runner_config)

        hook_data = {}

        def error_hook(benchmark, error):
            hook_data["called"] = True
            hook_data["error"] = error

        runner.add_hook("on_error", error_hook)
        runner.run_benchmark(failing_benchmark)

        assert hook_data["called"] is True
        assert "Setup failed" in hook_data["error"]

    def test_hook_exception_handling(self, runner_config, matmul_benchmark):
        """Test that hook exceptions don't break the runner."""
        runner = BenchmarkRunner(runner_config)

        def bad_hook(**kwargs):
            raise RuntimeError("Hook error")

        runner.add_hook("before_benchmark", bad_hook)

        # Should still complete despite hook error
        result = runner.run_benchmark(matmul_benchmark)
        assert result.success is True


class TestBenchmarkRunnerSummary:
    """Tests for BenchmarkRunner summary output."""

    def test_print_summary_empty(self, runner_config, capsys):
        """Test print_summary with no results."""
        runner = BenchmarkRunner(runner_config)
        runner.config.verbose = True  # Enable verbose for test
        runner.print_summary()

        # Should not crash with empty results
        captured = capsys.readouterr()
        assert captured.out == ""  # No output for empty results

    def test_print_summary_with_results(self, runner_config, minimal_config, capsys):
        """Test print_summary with results."""
        runner_config.verbose = True
        runner = BenchmarkRunner(runner_config)

        # run_from_registry adds results to runner.results
        runner.run_from_registry(["matmul", "softmax"], minimal_config)
        runner.print_summary()

        captured = capsys.readouterr()
        assert "BENCHMARK RESULTS SUMMARY" in captured.out
        assert "matmul" in captured.out
        assert "softmax" in captured.out


class TestCLIRunner:
    """Tests for CLIRunner class."""

    def test_cli_runner_creation(self):
        """Test CLI runner creation."""
        cli = CLIRunner()

        assert cli.runner_config is not None

    def test_cli_list_benchmarks(self, capsys):
        """Test --list flag."""
        cli = CLIRunner()
        result = cli.run(["--list"])

        assert result == 0
        captured = capsys.readouterr()
        assert "Available benchmarks:" in captured.out

    def test_cli_run_specific_benchmark(self, temp_output_dir):
        """Test running specific benchmark via CLI."""
        cli = CLIRunner()
        cli.runner_config.output_dir = temp_output_dir
        cli.runner_config.verbose = False
        cli.runner_config.save_results = False

        result = cli.run([
            "--benchmark", "matmul",
            "--iterations", "1",
            "--output", temp_output_dir
        ])

        # Should succeed
        assert result == 0

    def test_cli_batch_size(self, temp_output_dir):
        """Test --batch-size flag."""
        cli = CLIRunner()
        cli.runner_config.output_dir = temp_output_dir
        cli.runner_config.verbose = False
        cli.runner_config.save_results = False

        result = cli.run([
            "--benchmark", "softmax",
            "--batch-size", "4",
            "--iterations", "1"
        ])

        assert result == 0

    def test_cli_sequence_length(self, temp_output_dir):
        """Test --seq-len flag."""
        cli = CLIRunner()
        cli.runner_config.output_dir = temp_output_dir
        cli.runner_config.verbose = False
        cli.runner_config.save_results = False

        result = cli.run([
            "--benchmark", "embedding",
            "--seq-len", "32",
            "--iterations", "1"
        ])

        assert result == 0

    def test_cli_quiet_mode(self, temp_output_dir, capsys):
        """Test --quiet flag."""
        cli = CLIRunner()
        cli.runner_config.output_dir = temp_output_dir
        cli.runner_config.save_results = False

        result = cli.run([
            "--benchmark", "matmul",
            "--iterations", "1",
            "--quiet"
        ])

        assert result == 0
        assert cli.runner_config.verbose is False

    def test_cli_help_flag(self, capsys):
        """--help prints usage and exits 0 without running benchmarks."""
        cli = CLIRunner()
        result = cli.run(["--help"])

        assert result == 0
        captured = capsys.readouterr()
        assert "usage: aibench" in captured.out
        assert "--benchmark" in captured.out

    def test_cli_help_short_flag(self, capsys):
        """-h behaves like --help."""
        cli = CLIRunner()
        result = cli.run(["-h"])

        assert result == 0
        assert "usage: aibench" in capsys.readouterr().out

    def test_cli_help_takes_precedence(self, capsys):
        """--help wins even when other flags are present (no benchmarks run)."""
        cli = CLIRunner()
        result = cli.run(["--benchmark", "matmul", "--help"])

        assert result == 0
        assert "usage: aibench" in capsys.readouterr().out


class TestMainEntryPoint:
    """Tests for the module-level main() console entry point."""

    def test_main_help(self, monkeypatch, capsys):
        """main() is callable as the console script and handles --help."""
        monkeypatch.setattr("sys.argv", ["aibench", "--help"])
        result = main()

        assert result == 0
        assert "usage: aibench" in capsys.readouterr().out

    def test_main_list(self, monkeypatch, capsys):
        """main() forwards args to the CLI runner (--list)."""
        monkeypatch.setattr("sys.argv", ["aibench", "--list"])
        result = main()

        assert result == 0
        assert "Available benchmarks:" in capsys.readouterr().out


class TestBenchmarkScheduler:
    """Tests for BenchmarkScheduler class."""

    def test_scheduler_creation(self):
        """Test scheduler creation."""
        scheduler = BenchmarkScheduler()

        assert len(scheduler.pending) == 0
        assert len(scheduler.completed) == 0

    def test_schedule_benchmark(self, minimal_config):
        """Test scheduling a benchmark."""
        scheduler = BenchmarkScheduler()

        scheduler.schedule("matmul", minimal_config)
        scheduler.schedule("softmax", minimal_config)

        assert len(scheduler.pending) == 2

    def test_run_scheduled(self, runner_config, minimal_config):
        """Test running scheduled benchmarks."""
        scheduler = BenchmarkScheduler()
        scheduler.schedule("matmul", minimal_config)
        scheduler.schedule("softmax", minimal_config)

        runner = BenchmarkRunner(runner_config)
        results = scheduler.run_scheduled(runner)

        assert len(results) == 2
        assert len(scheduler.pending) == 0  # Cleared after run
        assert len(scheduler.completed) == 2

    def test_run_scheduled_nonexistent(self, runner_config, minimal_config):
        """Test running nonexistent scheduled benchmark."""
        scheduler = BenchmarkScheduler()
        scheduler.schedule("nonexistent", minimal_config)

        runner = BenchmarkRunner(runner_config)
        results = scheduler.run_scheduled(runner)

        # Should not crash, just skip nonexistent
        assert len(results) == 0


class TestRunBenchmarksFunction:
    """Tests for the run_benchmarks convenience function."""

    def test_run_benchmarks_specific(self, temp_output_dir):
        """Test run_benchmarks with specific benchmarks."""
        config = BenchmarkConfig(
            name="test",
            warmup_iterations=0,
            benchmark_iterations=1,
            extra_config={"m": 32, "n": 32, "k": 32}
        )

        results = run_benchmarks(
            benchmarks=["matmul"],
            config=config,
            output_dir=temp_output_dir
        )

        assert len(results) == 1
        assert results[0].benchmark_name == "matmul"

    def test_run_benchmarks_default_config(self, temp_output_dir):
        """Test run_benchmarks with default config."""
        # Clear any existing registry entries that might take long
        results = run_benchmarks(
            benchmarks=["softmax"],
            output_dir=temp_output_dir
        )

        assert len(results) == 1


class TestRunnerEdgeCases:
    """Tests for edge cases in runner functionality."""

    def test_empty_benchmark_list(self, runner_config, minimal_config):
        """Test running with empty benchmark list."""
        runner = BenchmarkRunner(runner_config)
        results = runner.run_from_registry([], minimal_config)

        assert len(results) == 0

    def test_empty_suite(self, runner_config):
        """Test running empty suite."""
        suite = BenchmarkSuite(name="empty_suite")
        runner = BenchmarkRunner(runner_config)
        results = runner.run_suite(suite)

        assert len(results) == 0

    def test_results_accumulation(self, runner_config, minimal_config):
        """Test that results accumulate across multiple runs."""
        runner = BenchmarkRunner(runner_config)

        # run_from_registry adds results to runner.results
        runner.run_from_registry(["matmul"], minimal_config)
        assert len(runner.results) == 1

        runner.run_from_registry(["softmax"], minimal_config)
        assert len(runner.results) == 2

    def test_mixed_success_failure(self, runner_config, benchmark_suite, failing_benchmark):
        """Test mixed success and failure results."""
        runner = BenchmarkRunner(runner_config)

        # Add failing benchmark to suite
        benchmark_suite.add_benchmark(failing_benchmark)

        # run_suite accumulates results
        runner.run_suite(benchmark_suite)

        # Suite has: matmul + softmax + failing = 3 benchmarks
        assert len(runner.results) == 3

        success_count = sum(1 for r in runner.results if r.success)
        failure_count = sum(1 for r in runner.results if not r.success)

        assert success_count == 2
        assert failure_count == 1


class TestRunnerConcurrency:
    """Tests for concurrent/parallel runner functionality."""

    def test_parallel_execution_results(self, parallel_runner_config):
        """Test that parallel execution produces correct results."""
        # Create suite with multiple benchmarks
        suite = BenchmarkSuite(name="parallel_test")

        config = BenchmarkConfig(
            name="fast_test",
            warmup_iterations=0,
            benchmark_iterations=1,
            extra_config={"m": 32, "n": 32, "k": 32}
        )

        for _ in range(3):
            suite.add_benchmark(MatMulBenchmark(config))

        runner = BenchmarkRunner(parallel_runner_config)
        results = runner.run_suite(suite)

        assert len(results) == 3
        # All should succeed
        assert all(r.success for r in results)

    def test_parallel_with_different_benchmarks(self, parallel_runner_config, minimal_config):
        """Test parallel execution with different benchmark types."""
        runner = BenchmarkRunner(parallel_runner_config)

        suite = BenchmarkSuite(name="mixed_suite")
        suite.add_benchmark(MatMulBenchmark(minimal_config))
        suite.add_benchmark(SoftmaxBenchmark(minimal_config))

        results = runner.run_suite(suite)

        benchmark_names = [r.benchmark_name for r in results]
        assert "matmul" in benchmark_names
        assert "softmax" in benchmark_names
