# Project 48: End-to-End AI Systems Benchmark Suite

## Executive Summary

A comprehensive benchmarking framework for objective evaluation of AI/ML infrastructure stacks. This system provides standardized workloads, precise measurement of throughput and latency metrics, reproducibility controls, and rich reporting capabilities to enable fair comparison of systems and detection of performance regressions.

> **Concepts covered:** [§05 Metrics](../../05-cross-cutting-concerns/observability/metrics/metrics.md) · [§05 CI/CD (regression gating)](../../05-cross-cutting-concerns/ci-cd/) · [§07 Benchmarks](../../07-infrastructure/benchmarks/). Companion to [Project 09 (data observability)](../09-data-observability/) for the data side; benchmarks the inference stack of [Projects 22, 41, 44, 47](../) and the GPU stack of [19, 39, 46, 48](../). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## Architecture Overview

### System Design

```
+------------------------------------------------------------------+
|                    AI Benchmark Suite                             |
+------------------------------------------------------------------+
|                                                                    |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Workload          |     | Benchmark         |     | Experiment| |
|  | Definitions       |---->| Runner            |---->| Manager   | |
|  +-------------------+     +-------------------+     +-----------+ |
|         |                          |                       |       |
|         v                          v                       v       |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Load Generator    |     | Metrics           |     | Results   | |
|  | (Traffic Patterns)|     | Collector         |     | Storage   | |
|  +-------------------+     +-------------------+     +-----------+ |
|                                    |                               |
|  +----------------------------------------------------------+     |
|  |                    Reporting Engine                       |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  |  | HTML   |  | Charts |  | JSON   |  | Compare|           |     |
|  |  | Report |  | (Plot) |  | Export |  | Diffs  |           |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  +----------------------------------------------------------+     |
+------------------------------------------------------------------+
```

### Core Components

#### 1. Workload Definitions

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from abc import ABC, abstractmethod
import time

class WorkloadCategory(Enum):
    LLM_INFERENCE = "llm_inference"
    RAG = "rag"
    ANN_SEARCH = "ann_search"
    GPU_KERNEL = "gpu_kernel"
    TRAINING = "training"
    DATA_PIPELINE = "data_pipeline"
    EMBEDDING = "embedding"

@dataclass
class WorkloadConfig:
    """Configuration for a benchmark workload."""
    name: str
    category: WorkloadCategory
    description: str = ""

    # Workload parameters
    params: Dict[str, Any] = field(default_factory=dict)

    # Load profile
    num_requests: int = 1000
    concurrency: int = 1
    duration_seconds: Optional[int] = None  # Run for time instead of count

    # Warmup
    warmup_requests: int = 100

    # Dataset
    dataset_path: Optional[str] = None
    dataset_size: int = 10000

class Workload(ABC):
    """Abstract base class for workloads."""

    def __init__(self, config: WorkloadConfig):
        self.config = config

    @abstractmethod
    def setup(self) -> None:
        """Initialize workload (load data, models, etc.)."""
        pass

    @abstractmethod
    def run_single(self, request_id: int) -> Dict[str, Any]:
        """Run single request and return metrics."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup resources."""
        pass

    def validate(self, result: Dict[str, Any]) -> bool:
        """Validate result correctness."""
        return True


class LLMInferenceWorkload(Workload):
    """LLM inference benchmark."""

    def __init__(self, config: WorkloadConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.prompts = []

    def setup(self):
        # Load model
        model_name = self.config.params.get('model', 'llama-7b')
        self.max_tokens = self.config.params.get('max_tokens', 128)

        # Load prompts
        if self.config.dataset_path:
            self._load_prompts(self.config.dataset_path)
        else:
            self._generate_synthetic_prompts()

    def _load_prompts(self, path: str):
        with open(path, 'r') as f:
            self.prompts = [line.strip() for line in f]

    def _generate_synthetic_prompts(self):
        # Generate prompts of varying lengths
        templates = [
            "Explain the concept of {}",
            "Write a short story about {}",
            "What are the main differences between {} and {}?",
        ]
        # ... generate prompts
        self.prompts = ["Hello, how are you?"] * self.config.dataset_size

    def run_single(self, request_id: int) -> Dict[str, Any]:
        prompt = self.prompts[request_id % len(self.prompts)]

        start = time.perf_counter()

        # Simulate inference (real implementation would call model)
        # tokens = self.model.generate(prompt, max_tokens=self.max_tokens)

        end = time.perf_counter()

        # Simulated metrics
        input_tokens = len(prompt.split())
        output_tokens = self.max_tokens

        return {
            'latency_ms': (end - start) * 1000,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'tokens_per_second': output_tokens / (end - start) if end > start else 0,
            'time_to_first_token_ms': 50,  # Simulated
        }

    def teardown(self):
        self.model = None
        self.prompts = []


class RAGWorkload(Workload):
    """RAG system benchmark."""

    def __init__(self, config: WorkloadConfig):
        super().__init__(config)
        self.queries = []
        self.index = None
        self.retriever = None

    def setup(self):
        # Setup vector index and retriever
        top_k = self.config.params.get('top_k', 5)
        index_type = self.config.params.get('index_type', 'hnsw')

        # Load queries
        self._load_queries()

    def _load_queries(self):
        # Load or generate queries
        self.queries = ["What is machine learning?"] * self.config.dataset_size

    def run_single(self, request_id: int) -> Dict[str, Any]:
        query = self.queries[request_id % len(self.queries)]

        # Retrieval phase
        retrieval_start = time.perf_counter()
        # docs = self.retriever.retrieve(query)
        retrieval_end = time.perf_counter()

        # Generation phase
        generation_start = time.perf_counter()
        # response = self.llm.generate(query, context=docs)
        generation_end = time.perf_counter()

        return {
            'total_latency_ms': (generation_end - retrieval_start) * 1000,
            'retrieval_latency_ms': (retrieval_end - retrieval_start) * 1000,
            'generation_latency_ms': (generation_end - generation_start) * 1000,
            'num_docs_retrieved': 5,
        }

    def teardown(self):
        pass


class ANNSearchWorkload(Workload):
    """ANN vector search benchmark."""

    def __init__(self, config: WorkloadConfig):
        super().__init__(config)
        self.index = None
        self.queries = None

    def setup(self):
        num_vectors = self.config.params.get('num_vectors', 1000000)
        dim = self.config.params.get('dim', 128)
        index_type = self.config.params.get('index_type', 'hnsw')

        # Build or load index
        # Generate query vectors
        import numpy as np
        self.queries = np.random.randn(
            self.config.dataset_size, dim
        ).astype(np.float32)

    def run_single(self, request_id: int) -> Dict[str, Any]:
        query = self.queries[request_id % len(self.queries)]

        start = time.perf_counter()
        # results = self.index.search(query, k=10)
        end = time.perf_counter()

        return {
            'latency_ms': (end - start) * 1000,
            'recall': 0.95,  # Would compute vs ground truth
        }

    def teardown(self):
        self.index = None
        self.queries = None


class GPUKernelWorkload(Workload):
    """GPU kernel benchmark (matmul, attention, etc.)."""

    def __init__(self, config: WorkloadConfig):
        super().__init__(config)
        self.inputs = None

    def setup(self):
        import numpy as np

        kernel_type = self.config.params.get('kernel', 'matmul')
        m = self.config.params.get('m', 4096)
        n = self.config.params.get('n', 4096)
        k = self.config.params.get('k', 4096)

        # Pre-allocate inputs
        self.inputs = {
            'A': np.random.randn(m, k).astype(np.float16),
            'B': np.random.randn(k, n).astype(np.float16),
        }

    def run_single(self, request_id: int) -> Dict[str, Any]:
        start = time.perf_counter()

        # Execute kernel
        # C = matmul(A, B)

        end = time.perf_counter()

        # Calculate FLOPS
        m, k = self.inputs['A'].shape
        _, n = self.inputs['B'].shape
        flops = 2 * m * n * k

        return {
            'latency_ms': (end - start) * 1000,
            'tflops': flops / (end - start) / 1e12,
        }

    def teardown(self):
        self.inputs = None
```

#### 2. Benchmark Runner

```python
import statistics
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import List

@dataclass
class BenchmarkResult:
    """Result of a single benchmark request."""
    request_id: int
    start_time: float
    end_time: float
    metrics: Dict[str, Any]
    success: bool
    error: Optional[str] = None

@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark run."""
    workload_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float

    # Latency percentiles (ms)
    latency_p50: float
    latency_p90: float
    latency_p99: float
    latency_mean: float
    latency_std: float

    # Throughput
    requests_per_second: float

    # Additional metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

class BenchmarkRunner:
    """Execute benchmarks with load generation."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def run(self,
            workload: Workload,
            progress_callback: Optional[Callable] = None) -> BenchmarkSummary:
        """
        Run benchmark with configured load.

        Args:
            workload: Workload to benchmark
            progress_callback: Called with progress updates

        Returns:
            Summary of benchmark results
        """
        config = workload.config
        self.results = []

        # Setup
        workload.setup()

        # Warmup
        for i in range(config.warmup_requests):
            workload.run_single(i)

        # Run benchmark
        start_time = time.time()

        if config.concurrency == 1:
            # Sequential execution
            self._run_sequential(workload, config.num_requests, progress_callback)
        else:
            # Concurrent execution
            self._run_concurrent(workload, config.num_requests,
                               config.concurrency, progress_callback)

        end_time = time.time()

        # Teardown
        workload.teardown()

        # Compute summary
        return self._compute_summary(
            workload.config.name,
            end_time - start_time
        )

    def _run_sequential(self,
                        workload: Workload,
                        num_requests: int,
                        progress_callback: Optional[Callable]):
        """Run requests sequentially."""
        for i in range(num_requests):
            start = time.time()

            try:
                metrics = workload.run_single(i)
                success = True
                error = None
            except Exception as e:
                metrics = {}
                success = False
                error = str(e)

            end = time.time()

            result = BenchmarkResult(
                request_id=i,
                start_time=start,
                end_time=end,
                metrics=metrics,
                success=success,
                error=error
            )
            self.results.append(result)

            if progress_callback and i % 100 == 0:
                progress_callback(i, num_requests)

    def _run_concurrent(self,
                        workload: Workload,
                        num_requests: int,
                        concurrency: int,
                        progress_callback: Optional[Callable]):
        """Run requests concurrently."""
        completed = [0]
        lock = threading.Lock()

        def run_request(request_id: int) -> BenchmarkResult:
            start = time.time()

            try:
                metrics = workload.run_single(request_id)
                success = True
                error = None
            except Exception as e:
                metrics = {}
                success = False
                error = str(e)

            end = time.time()

            with lock:
                completed[0] += 1
                if progress_callback and completed[0] % 100 == 0:
                    progress_callback(completed[0], num_requests)

            return BenchmarkResult(
                request_id=request_id,
                start_time=start,
                end_time=end,
                metrics=metrics,
                success=success,
                error=error
            )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(run_request, i)
                for i in range(num_requests)
            ]

            for future in futures:
                self.results.append(future.result())

    def _compute_summary(self,
                         workload_name: str,
                         duration: float) -> BenchmarkSummary:
        """Compute summary statistics."""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        # Extract latencies
        latencies = []
        for r in successful:
            if 'latency_ms' in r.metrics:
                latencies.append(r.metrics['latency_ms'])
            else:
                latencies.append((r.end_time - r.start_time) * 1000)

        if not latencies:
            latencies = [0]

        # Sort for percentiles
        latencies_sorted = sorted(latencies)

        def percentile(data, p):
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        # Aggregate custom metrics
        custom_metrics = {}
        metric_names = set()
        for r in successful:
            metric_names.update(r.metrics.keys())

        for name in metric_names:
            if name == 'latency_ms':
                continue
            values = [r.metrics[name] for r in successful if name in r.metrics]
            if values and all(isinstance(v, (int, float)) for v in values):
                custom_metrics[f'{name}_mean'] = statistics.mean(values)

        return BenchmarkSummary(
            workload_name=workload_name,
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            duration_seconds=duration,
            latency_p50=percentile(latencies_sorted, 50),
            latency_p90=percentile(latencies_sorted, 90),
            latency_p99=percentile(latencies_sorted, 99),
            latency_mean=statistics.mean(latencies),
            latency_std=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            requests_per_second=len(successful) / duration if duration > 0 else 0,
            custom_metrics=custom_metrics
        )
```

#### 3. Experiment Manager

```python
import json
import hashlib
from datetime import datetime
from pathlib import Path

@dataclass
class ExperimentConfig:
    """Configuration for a benchmark experiment."""
    name: str
    description: str = ""

    # Workloads to run
    workloads: List[WorkloadConfig] = field(default_factory=list)

    # Environment info
    system_info: Dict[str, str] = field(default_factory=dict)

    # Reproducibility
    random_seed: int = 42
    git_commit: str = ""

    # Variants
    variants: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ExperimentResult:
    """Result of a complete experiment."""
    config: ExperimentConfig
    summaries: List[BenchmarkSummary]
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExperimentManager:
    """Manage and track benchmark experiments."""

    def __init__(self, results_dir: str = './results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run complete experiment with all workloads."""
        start_time = datetime.now()
        summaries = []

        # Collect system info
        config.system_info = self._collect_system_info()

        runner = BenchmarkRunner()

        for workload_config in config.workloads:
            # Create workload instance
            workload = self._create_workload(workload_config)

            # Run benchmark
            summary = runner.run(workload, progress_callback=self._progress)
            summaries.append(summary)

            print(f"Completed {workload_config.name}: "
                  f"{summary.requests_per_second:.1f} req/s, "
                  f"p99={summary.latency_p99:.1f}ms")

        end_time = datetime.now()

        result = ExperimentResult(
            config=config,
            summaries=summaries,
            start_time=start_time,
            end_time=end_time
        )

        # Save result
        self._save_result(result)

        return result

    def _create_workload(self, config: WorkloadConfig) -> Workload:
        """Create workload instance from config."""
        workload_classes = {
            WorkloadCategory.LLM_INFERENCE: LLMInferenceWorkload,
            WorkloadCategory.RAG: RAGWorkload,
            WorkloadCategory.ANN_SEARCH: ANNSearchWorkload,
            WorkloadCategory.GPU_KERNEL: GPUKernelWorkload,
        }

        cls = workload_classes.get(config.category, Workload)
        return cls(config)

    def _collect_system_info(self) -> Dict[str, str]:
        """Collect system information for reproducibility."""
        import platform

        info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
        }

        # Try to get GPU info
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total',
                 '--format=csv,noheader'],
                capture_output=True, text=True
            )
            info['gpu'] = result.stdout.strip()
        except:
            pass

        return info

    def _progress(self, completed: int, total: int):
        """Progress callback."""
        pct = 100 * completed / total
        print(f"\rProgress: {completed}/{total} ({pct:.1f}%)", end='')

    def _save_result(self, result: ExperimentResult):
        """Save experiment result to disk."""
        # Generate unique ID
        result_id = hashlib.md5(
            f"{result.config.name}{result.start_time}".encode()
        ).hexdigest()[:8]

        # Save as JSON
        path = self.results_dir / f"{result.config.name}_{result_id}.json"

        data = {
            'config': {
                'name': result.config.name,
                'description': result.config.description,
                'system_info': result.config.system_info,
            },
            'summaries': [
                {
                    'workload_name': s.workload_name,
                    'total_requests': s.total_requests,
                    'successful_requests': s.successful_requests,
                    'failed_requests': s.failed_requests,
                    'duration_seconds': s.duration_seconds,
                    'latency_p50': s.latency_p50,
                    'latency_p90': s.latency_p90,
                    'latency_p99': s.latency_p99,
                    'latency_mean': s.latency_mean,
                    'requests_per_second': s.requests_per_second,
                    'custom_metrics': s.custom_metrics,
                }
                for s in result.summaries
            ],
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved results to {path}")

    def load_result(self, path: str) -> ExperimentResult:
        """Load experiment result from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct result
        summaries = []
        for s in data['summaries']:
            summaries.append(BenchmarkSummary(
                workload_name=s['workload_name'],
                total_requests=s['total_requests'],
                successful_requests=s['successful_requests'],
                failed_requests=s['failed_requests'],
                duration_seconds=s['duration_seconds'],
                latency_p50=s['latency_p50'],
                latency_p90=s['latency_p90'],
                latency_p99=s['latency_p99'],
                latency_mean=s['latency_mean'],
                latency_std=0,
                requests_per_second=s['requests_per_second'],
                custom_metrics=s.get('custom_metrics', {})
            ))

        config = ExperimentConfig(
            name=data['config']['name'],
            description=data['config'].get('description', ''),
            system_info=data['config'].get('system_info', {})
        )

        return ExperimentResult(
            config=config,
            summaries=summaries,
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time'])
        )

    def compare_results(self,
                        result1: ExperimentResult,
                        result2: ExperimentResult) -> Dict[str, Any]:
        """Compare two experiment results."""
        comparison = {
            'baseline': result1.config.name,
            'compare': result2.config.name,
            'workloads': {}
        }

        # Match workloads by name
        summaries1 = {s.workload_name: s for s in result1.summaries}
        summaries2 = {s.workload_name: s for s in result2.summaries}

        for name in set(summaries1.keys()) & set(summaries2.keys()):
            s1, s2 = summaries1[name], summaries2[name]

            comparison['workloads'][name] = {
                'throughput_change': (s2.requests_per_second - s1.requests_per_second) / s1.requests_per_second * 100,
                'latency_p99_change': (s2.latency_p99 - s1.latency_p99) / s1.latency_p99 * 100,
                'latency_p50_change': (s2.latency_p50 - s1.latency_p50) / s1.latency_p50 * 100,
            }

        return comparison
```

#### 4. Reporting Engine

```python
class ReportGenerator:
    """Generate benchmark reports."""

    def generate_html(self, result: ExperimentResult, output_path: str):
        """Generate HTML report with charts."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{result.config.name} - Benchmark Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; margin: 10px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>{result.config.name}</h1>
            <p>{result.config.description}</p>
            <p><strong>Duration:</strong> {(result.end_time - result.start_time).total_seconds():.1f}s</p>

            <h2>System Info</h2>
            <ul>
                {''.join(f'<li><strong>{k}:</strong> {v}</li>' for k, v in result.config.system_info.items())}
            </ul>

            <h2>Results</h2>
            {''.join(self._generate_workload_html(s) for s in result.summaries)}

            {self._generate_charts(result)}
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html)

    def _generate_workload_html(self, summary: BenchmarkSummary) -> str:
        """Generate HTML for single workload summary."""
        return f"""
        <div class="summary">
            <h3>{summary.workload_name}</h3>
            <div class="metric">
                <strong>Throughput:</strong> {summary.requests_per_second:.1f} req/s
            </div>
            <div class="metric">
                <strong>P50:</strong> {summary.latency_p50:.2f}ms
            </div>
            <div class="metric">
                <strong>P90:</strong> {summary.latency_p90:.2f}ms
            </div>
            <div class="metric">
                <strong>P99:</strong> {summary.latency_p99:.2f}ms
            </div>
            <div class="metric">
                <strong>Success Rate:</strong> {100*summary.successful_requests/summary.total_requests:.1f}%
            </div>
        </div>
        """

    def _generate_charts(self, result: ExperimentResult) -> str:
        """Generate Plotly charts."""
        # Latency comparison chart
        workloads = [s.workload_name for s in result.summaries]
        p50s = [s.latency_p50 for s in result.summaries]
        p90s = [s.latency_p90 for s in result.summaries]
        p99s = [s.latency_p99 for s in result.summaries]

        return f"""
        <div class="chart" id="latency-chart"></div>
        <script>
            var data = [
                {{x: {workloads}, y: {p50s}, name: 'P50', type: 'bar'}},
                {{x: {workloads}, y: {p90s}, name: 'P90', type: 'bar'}},
                {{x: {workloads}, y: {p99s}, name: 'P99', type: 'bar'}}
            ];
            Plotly.newPlot('latency-chart', data, {{
                title: 'Latency by Workload',
                yaxis: {{title: 'Latency (ms)'}}
            }});
        </script>
        """

    def generate_json(self, result: ExperimentResult, output_path: str):
        """Export results as JSON."""
        data = {
            'name': result.config.name,
            'timestamp': result.start_time.isoformat(),
            'summaries': [
                {
                    'workload': s.workload_name,
                    'throughput': s.requests_per_second,
                    'latency_p50': s.latency_p50,
                    'latency_p90': s.latency_p90,
                    'latency_p99': s.latency_p99,
                }
                for s in result.summaries
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
```

### Enterprise Features

#### Regression Detection

```python
class RegressionDetector:
    """Detect performance regressions between runs."""

    def __init__(self,
                 throughput_threshold: float = 0.05,
                 latency_threshold: float = 0.10):
        self.throughput_threshold = throughput_threshold
        self.latency_threshold = latency_threshold

    def detect(self,
               baseline: ExperimentResult,
               current: ExperimentResult) -> List[Dict]:
        """
        Detect regressions between baseline and current.

        Returns list of detected regressions.
        """
        regressions = []

        # Match workloads
        baseline_summaries = {s.workload_name: s for s in baseline.summaries}
        current_summaries = {s.workload_name: s for s in current.summaries}

        for name in baseline_summaries.keys() & current_summaries.keys():
            base = baseline_summaries[name]
            curr = current_summaries[name]

            # Check throughput regression
            throughput_change = (curr.requests_per_second - base.requests_per_second) / base.requests_per_second

            if throughput_change < -self.throughput_threshold:
                regressions.append({
                    'workload': name,
                    'metric': 'throughput',
                    'baseline': base.requests_per_second,
                    'current': curr.requests_per_second,
                    'change_pct': throughput_change * 100
                })

            # Check latency regression
            latency_change = (curr.latency_p99 - base.latency_p99) / base.latency_p99

            if latency_change > self.latency_threshold:
                regressions.append({
                    'workload': name,
                    'metric': 'latency_p99',
                    'baseline': base.latency_p99,
                    'current': curr.latency_p99,
                    'change_pct': latency_change * 100
                })

        return regressions


class MultiNodeHarness:
    """Run benchmarks across multiple nodes."""

    def __init__(self, nodes: List[str]):
        self.nodes = nodes

    def run_distributed(self,
                        config: ExperimentConfig) -> List[ExperimentResult]:
        """Run experiment on multiple nodes."""
        results = []

        for node in self.nodes:
            # SSH to node and run benchmark
            # Collect results
            pass

        return results

    def aggregate_results(self,
                          results: List[ExperimentResult]) -> ExperimentResult:
        """Aggregate results from multiple nodes."""
        # Combine summaries
        pass
```

## API Reference

### Define and Run Benchmark

```python
# Create workload config
llm_config = WorkloadConfig(
    name="llm_7b_inference",
    category=WorkloadCategory.LLM_INFERENCE,
    params={
        'model': 'llama-7b',
        'max_tokens': 128
    },
    num_requests=1000,
    concurrency=4
)

# Create experiment
experiment = ExperimentConfig(
    name="LLM Inference Benchmark",
    description="Benchmark LLM inference throughput",
    workloads=[llm_config]
)

# Run
manager = ExperimentManager()
result = manager.run_experiment(experiment)

# Generate report
reporter = ReportGenerator()
reporter.generate_html(result, 'report.html')
```

### Compare Results

```python
# Load previous result
baseline = manager.load_result('results/baseline.json')

# Check for regressions
detector = RegressionDetector()
regressions = detector.detect(baseline, result)

if regressions:
    print("Regressions detected!")
    for reg in regressions:
        print(f"  {reg['workload']}: {reg['metric']} "
              f"changed {reg['change_pct']:.1f}%")
```

## Implementation Phases

### Phase 1: Core Framework (Weeks 1-2)
- Workload base class
- Benchmark runner
- Basic metrics collection
- Result storage

### Phase 2: Workload Library (Weeks 3-5)
- LLM inference workload
- RAG workload
- ANN search workload
- GPU kernel workload

### Phase 3: Experiment Management (Weeks 6-7)
- Experiment configuration
- System info collection
- Result comparison
- Variants support

### Phase 4: Reporting (Weeks 8-9)
- HTML reports
- Charts with Plotly
- JSON export
- Comparison views

### Phase 5: Enterprise (Weeks 10-14)
- Regression detection
- Multi-node harness
- Canary benchmarks
- Containerized benchmarks

## Performance Targets

| Metric | Target |
|--------|--------|
| Measurement overhead | <1% |
| Result accuracy | CV <5% |
| Report generation | <5s |

## Dependencies

- NumPy
- Plotly (for charts)
- (Optional) CUDA for GPU benchmarks

## References

- MLPerf
- BenchmarkAI
- Perfkit Benchmarker
