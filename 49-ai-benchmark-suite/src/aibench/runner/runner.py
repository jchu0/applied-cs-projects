"""Benchmark runner and execution engine."""

from dataclasses import dataclass, field
from typing import Any, Callable
import time
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.benchmark import (
    Benchmark, BenchmarkConfig, BenchmarkResult, BenchmarkSuite,
    registry, save_results, load_results, compare_results
)


@dataclass
class RunnerConfig:
    """Configuration for benchmark runner."""
    output_dir: str = "./benchmark_results"
    parallel: bool = False
    max_workers: int = 4
    verbose: bool = True
    save_results: bool = True
    compare_baseline: str | None = None
    filter_benchmarks: list[str] | None = None


class BenchmarkRunner:
    """Execute benchmarks and collect results."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self.results: list[BenchmarkResult] = []
        self._hooks: dict[str, list[Callable]] = {
            "before_benchmark": [],
            "after_benchmark": [],
            "on_error": []
        }

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def add_hook(
        self,
        event: str,
        callback: Callable
    ) -> None:
        """Add callback hook."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _call_hooks(self, event: str, **kwargs) -> None:
        """Call all hooks for event."""
        for callback in self._hooks.get(event, []):
            try:
                callback(**kwargs)
            except Exception as e:
                if self.config.verbose:
                    print(f"Hook error: {e}")

    def run_benchmark(
        self,
        benchmark: Benchmark
    ) -> BenchmarkResult:
        """Run single benchmark."""
        self._call_hooks("before_benchmark", benchmark=benchmark)

        if self.config.verbose:
            print(f"Running benchmark: {benchmark.name()}")

        start = time.time()
        result = benchmark.benchmark()
        elapsed = time.time() - start

        if self.config.verbose:
            status = "PASS" if result.success else "FAIL"
            print(f"  [{status}] {elapsed:.2f}s - mean: {result.mean_time_ms:.2f}ms")

        self._call_hooks("after_benchmark", benchmark=benchmark, result=result)

        if not result.success:
            self._call_hooks("on_error", benchmark=benchmark, error=result.error_message)

        return result

    def run_suite(
        self,
        suite: BenchmarkSuite
    ) -> list[BenchmarkResult]:
        """Run all benchmarks in suite."""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Running benchmark suite: {suite.name}")
            print(f"{'='*60}\n")

        results = []

        if self.config.parallel:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self.run_benchmark, bench): bench
                    for bench in suite.benchmarks
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
        else:
            for bench in suite.benchmarks:
                result = self.run_benchmark(bench)
                results.append(result)

        self.results.extend(results)
        return results

    def run_from_registry(
        self,
        benchmark_names: list[str],
        config: BenchmarkConfig
    ) -> list[BenchmarkResult]:
        """Run benchmarks from registry."""
        results = []

        for name in benchmark_names:
            # Filter if specified
            if self.config.filter_benchmarks:
                if name not in self.config.filter_benchmarks:
                    continue

            benchmark = registry.create(name, config)
            if benchmark:
                result = self.run_benchmark(benchmark)
                results.append(result)
            elif self.config.verbose:
                print(f"Benchmark not found: {name}")

        self.results.extend(results)
        return results

    def save(self, filename: str | None = None) -> str:
        """Save results to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"results_{timestamp}.json"

        filepath = os.path.join(self.config.output_dir, filename)
        save_results(self.results, filepath)

        if self.config.verbose:
            print(f"\nResults saved to: {filepath}")

        return filepath

    def compare_with_baseline(
        self,
        baseline_path: str
    ) -> dict[str, dict[str, float]]:
        """Compare current results with baseline."""
        baseline_data = load_results(baseline_path)

        # Convert loaded data to BenchmarkResult objects (simplified)
        baseline_results = []
        for data in baseline_data:
            result = BenchmarkResult(
                benchmark_name=data["benchmark_name"],
                config=BenchmarkConfig(name=data["config"]["name"]),
                metrics=[],
                raw_times_ms=[data["statistics"]["mean_ms"]],
                success=data["success"]
            )
            baseline_results.append(result)

        comparison = compare_results(baseline_results, self.results)

        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Comparison with baseline")
            print(f"{'='*60}")

            for name, stats in comparison.items():
                speedup = stats["speedup"]
                indicator = "↑" if stats["improved"] else "↓"
                print(f"{name}: {speedup:.2f}x {indicator}")

        return comparison

    def print_summary(self) -> None:
        """Print results summary."""
        if not self.results:
            return

        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*60}")

        total_passed = sum(1 for r in self.results if r.success)
        total_failed = len(self.results) - total_passed

        print(f"Total: {len(self.results)} | Passed: {total_passed} | Failed: {total_failed}\n")

        for result in self.results:
            status = "✓" if result.success else "✗"
            print(f"{status} {result.benchmark_name}")
            print(f"    Mean: {result.mean_time_ms:.2f}ms")
            print(f"    Std:  {result.std_time_ms:.2f}ms")
            print(f"    P99:  {result.p99_time_ms:.2f}ms")

            # Print key metrics
            for metric in result.metrics:
                if metric.name not in ["mean_latency", "peak_memory"]:
                    print(f"    {metric.name}: {metric.value:.2f} {metric.unit.value}")
            print()


class CLIRunner:
    """Command-line interface for benchmarks."""

    def __init__(self):
        self.runner_config = RunnerConfig()

    def run(self, args: list[str]) -> int:
        """Run benchmarks from command line."""
        # Parse arguments (simplified)
        benchmark_names = []
        bench_config = BenchmarkConfig(name="cli")

        i = 0
        while i < len(args):
            arg = args[i]

            if arg == "--benchmark" and i + 1 < len(args):
                benchmark_names.append(args[i + 1])
                i += 2
            elif arg == "--batch-size" and i + 1 < len(args):
                bench_config.batch_size = int(args[i + 1])
                i += 2
            elif arg == "--seq-len" and i + 1 < len(args):
                bench_config.sequence_length = int(args[i + 1])
                i += 2
            elif arg == "--iterations" and i + 1 < len(args):
                bench_config.benchmark_iterations = int(args[i + 1])
                i += 2
            elif arg == "--output" and i + 1 < len(args):
                self.runner_config.output_dir = args[i + 1]
                i += 2
            elif arg == "--compare" and i + 1 < len(args):
                self.runner_config.compare_baseline = args[i + 1]
                i += 2
            elif arg == "--list":
                self._list_benchmarks()
                return 0
            elif arg == "--quiet":
                self.runner_config.verbose = False
                i += 1
            else:
                i += 1

        if not benchmark_names:
            benchmark_names = registry.list_benchmarks()

        runner = BenchmarkRunner(self.runner_config)
        results = runner.run_from_registry(benchmark_names, bench_config)

        if self.runner_config.save_results:
            runner.save()

        if self.runner_config.compare_baseline:
            runner.compare_with_baseline(self.runner_config.compare_baseline)

        runner.print_summary()

        # Return non-zero if any benchmark failed
        return 0 if all(r.success for r in results) else 1

    def _list_benchmarks(self) -> None:
        """List available benchmarks."""
        print("Available benchmarks:")
        for name in registry.list_benchmarks():
            print(f"  - {name}")


class BenchmarkScheduler:
    """Schedule and manage benchmark runs."""

    def __init__(self):
        self.pending: list[tuple[str, BenchmarkConfig]] = []
        self.completed: list[BenchmarkResult] = []

    def schedule(
        self,
        benchmark_name: str,
        config: BenchmarkConfig
    ) -> None:
        """Schedule a benchmark run."""
        self.pending.append((benchmark_name, config))

    def run_scheduled(
        self,
        runner: BenchmarkRunner
    ) -> list[BenchmarkResult]:
        """Run all scheduled benchmarks."""
        for name, config in self.pending:
            benchmark = registry.create(name, config)
            if benchmark:
                result = runner.run_benchmark(benchmark)
                self.completed.append(result)

        self.pending.clear()
        return self.completed


def run_benchmarks(
    benchmarks: list[str] | None = None,
    config: BenchmarkConfig | None = None,
    output_dir: str = "./results"
) -> list[BenchmarkResult]:
    """Convenience function to run benchmarks."""
    if config is None:
        config = BenchmarkConfig(name="default")

    if benchmarks is None:
        benchmarks = registry.list_benchmarks()

    runner_config = RunnerConfig(output_dir=output_dir)
    runner = BenchmarkRunner(runner_config)

    results = runner.run_from_registry(benchmarks, config)
    runner.save()
    runner.print_summary()

    return results
