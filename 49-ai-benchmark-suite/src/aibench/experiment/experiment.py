"""Experiment management for AI benchmarks.

This module provides experiment orchestration, workload definitions,
regression detection, and multi-node benchmark execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import json
import platform
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor


# =============================================================================
# Workload Categories and Configuration
# =============================================================================

class WorkloadCategory(Enum):
    """Categories of benchmark workloads."""
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


@dataclass
class WorkloadResult:
    """Result of a single workload request."""
    request_id: int
    start_time: float
    end_time: float
    metrics: Dict[str, Any]
    success: bool
    error: Optional[str] = None


@dataclass
class WorkloadSummary:
    """Summary statistics for a workload run."""
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


# =============================================================================
# Workload Base Class and Implementations
# =============================================================================

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
    """LLM inference benchmark workload."""

    def __init__(self, config: WorkloadConfig):
        super().__init__(config)
        self.prompts: List[str] = []
        self.max_tokens: int = 128

    def setup(self) -> None:
        """Initialize LLM inference workload."""
        self.max_tokens = self.config.params.get('max_tokens', 128)

        # Load prompts
        if self.config.dataset_path:
            self._load_prompts(self.config.dataset_path)
        else:
            self._generate_synthetic_prompts()

    def _load_prompts(self, path: str) -> None:
        """Load prompts from file."""
        try:
            with open(path, 'r') as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            self._generate_synthetic_prompts()

    def _generate_synthetic_prompts(self) -> None:
        """Generate synthetic prompts for testing."""
        templates = [
            "Explain the concept of machine learning in simple terms.",
            "Write a short story about a robot learning to cook.",
            "What are the main differences between Python and JavaScript?",
            "Describe the benefits of renewable energy sources.",
            "How does a neural network learn from data?",
        ]
        self.prompts = templates * (self.config.dataset_size // len(templates) + 1)
        self.prompts = self.prompts[:self.config.dataset_size]

    def run_single(self, request_id: int) -> Dict[str, Any]:
        """Run single inference request."""
        prompt = self.prompts[request_id % len(self.prompts)]

        start = time.perf_counter()

        # Simulate inference (real implementation would call model)
        # In production: tokens = model.generate(prompt, max_tokens=self.max_tokens)
        time.sleep(0.001)  # Simulate processing time

        end = time.perf_counter()

        # Calculate metrics
        input_tokens = len(prompt.split())
        output_tokens = self.max_tokens

        return {
            'latency_ms': (end - start) * 1000,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'tokens_per_second': output_tokens / (end - start) if end > start else 0,
            'time_to_first_token_ms': (end - start) * 1000 * 0.1,  # Simulated TTFT
        }

    def teardown(self) -> None:
        """Cleanup resources."""
        self.prompts = []


class RAGWorkload(Workload):
    """RAG (Retrieval-Augmented Generation) benchmark workload."""

    def __init__(self, config: WorkloadConfig):
        super().__init__(config)
        self.queries: List[str] = []
        self.top_k: int = 5

    def setup(self) -> None:
        """Initialize RAG workload."""
        self.top_k = self.config.params.get('top_k', 5)
        self._load_queries()

    def _load_queries(self) -> None:
        """Load or generate queries."""
        if self.config.dataset_path:
            try:
                with open(self.config.dataset_path, 'r') as f:
                    self.queries = [line.strip() for line in f if line.strip()]
                return
            except FileNotFoundError:
                pass

        # Generate synthetic queries
        synthetic = [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Explain quantum computing basics.",
            "What are the benefits of exercise?",
            "Describe the water cycle.",
        ]
        self.queries = synthetic * (self.config.dataset_size // len(synthetic) + 1)
        self.queries = self.queries[:self.config.dataset_size]

    def run_single(self, request_id: int) -> Dict[str, Any]:
        """Run single RAG request."""
        query = self.queries[request_id % len(self.queries)]

        # Retrieval phase
        retrieval_start = time.perf_counter()
        time.sleep(0.0005)  # Simulate retrieval
        retrieval_end = time.perf_counter()

        # Generation phase
        generation_start = time.perf_counter()
        time.sleep(0.001)  # Simulate generation
        generation_end = time.perf_counter()

        return {
            'total_latency_ms': (generation_end - retrieval_start) * 1000,
            'retrieval_latency_ms': (retrieval_end - retrieval_start) * 1000,
            'generation_latency_ms': (generation_end - generation_start) * 1000,
            'num_docs_retrieved': self.top_k,
            'latency_ms': (generation_end - retrieval_start) * 1000,
        }

    def teardown(self) -> None:
        """Cleanup resources."""
        self.queries = []


class ANNSearchWorkload(Workload):
    """ANN (Approximate Nearest Neighbor) vector search workload."""

    def __init__(self, config: WorkloadConfig):
        super().__init__(config)
        self.queries: Any = None  # numpy array
        self.dim: int = 128
        self.k: int = 10

    def setup(self) -> None:
        """Initialize ANN search workload."""
        import numpy as np

        self.dim = self.config.params.get('dim', 128)
        self.k = self.config.params.get('k', 10)
        num_queries = self.config.dataset_size

        # Generate random query vectors
        self.queries = np.random.randn(num_queries, self.dim).astype(np.float32)

    def run_single(self, request_id: int) -> Dict[str, Any]:
        """Run single ANN search request."""
        query = self.queries[request_id % len(self.queries)]

        start = time.perf_counter()
        time.sleep(0.0001)  # Simulate search
        end = time.perf_counter()

        return {
            'latency_ms': (end - start) * 1000,
            'recall': 0.95,  # Would compute against ground truth in real implementation
            'k': self.k,
        }

    def teardown(self) -> None:
        """Cleanup resources."""
        self.queries = None


class GPUKernelWorkload(Workload):
    """GPU kernel benchmark workload (matmul, attention, etc.)."""

    def __init__(self, config: WorkloadConfig):
        super().__init__(config)
        self.inputs: Dict[str, Any] = {}
        self.kernel_type: str = "matmul"

    def setup(self) -> None:
        """Initialize GPU kernel workload."""
        import numpy as np

        self.kernel_type = self.config.params.get('kernel', 'matmul')
        m = self.config.params.get('m', 4096)
        n = self.config.params.get('n', 4096)
        k = self.config.params.get('k', 4096)

        # Pre-allocate inputs
        self.inputs = {
            'A': np.random.randn(m, k).astype(np.float32),
            'B': np.random.randn(k, n).astype(np.float32),
        }
        self.m, self.k_dim, self.n = m, k, n

    def run_single(self, request_id: int) -> Dict[str, Any]:
        """Run single kernel execution."""
        start = time.perf_counter()

        # Simulate kernel execution (real implementation would run CUDA kernel)
        if self.kernel_type == 'matmul':
            # C = np.dot(self.inputs['A'], self.inputs['B'])
            pass
        time.sleep(0.0001)

        end = time.perf_counter()

        # Calculate FLOPS
        flops = 2 * self.m * self.n * self.k_dim

        return {
            'latency_ms': (end - start) * 1000,
            'tflops': flops / (end - start) / 1e12 if end > start else 0,
            'kernel_type': self.kernel_type,
        }

    def teardown(self) -> None:
        """Cleanup resources."""
        self.inputs = {}


# =============================================================================
# Experiment Configuration and Results
# =============================================================================

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

    # Variants (different configurations to compare)
    variants: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Result of a complete experiment."""
    config: ExperimentConfig
    summaries: List[WorkloadSummary]
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Workload Runner
# =============================================================================

class WorkloadRunner:
    """Execute workloads and collect results."""

    def __init__(self):
        self.results: List[WorkloadResult] = []

    def run(
        self,
        workload: Workload,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> WorkloadSummary:
        """Run workload with configured load."""
        config = workload.config
        self.results = []

        # Setup
        workload.setup()

        # Warmup
        for i in range(config.warmup_requests):
            try:
                workload.run_single(i)
            except Exception:
                pass

        # Run benchmark
        start_time = time.time()

        if config.concurrency == 1:
            self._run_sequential(workload, config.num_requests, progress_callback)
        else:
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

    def _run_sequential(
        self,
        workload: Workload,
        num_requests: int,
        progress_callback: Optional[Callable[[int, int], None]]
    ) -> None:
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

            result = WorkloadResult(
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

    def _run_concurrent(
        self,
        workload: Workload,
        num_requests: int,
        concurrency: int,
        progress_callback: Optional[Callable[[int, int], None]]
    ) -> None:
        """Run requests concurrently."""
        completed = [0]
        lock = threading.Lock()

        def run_request(request_id: int) -> WorkloadResult:
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

            return WorkloadResult(
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

    def _compute_summary(
        self,
        workload_name: str,
        duration: float
    ) -> WorkloadSummary:
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

        def percentile(data: List[float], p: int) -> float:
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        # Aggregate custom metrics
        custom_metrics: Dict[str, float] = {}
        metric_names = set()
        for r in successful:
            metric_names.update(r.metrics.keys())

        for name in metric_names:
            if name == 'latency_ms':
                continue
            values = [r.metrics[name] for r in successful if name in r.metrics]
            if values and all(isinstance(v, (int, float)) for v in values):
                custom_metrics[f'{name}_mean'] = statistics.mean(values)

        return WorkloadSummary(
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


# =============================================================================
# Experiment Manager
# =============================================================================

class ExperimentManager:
    """Manage and track benchmark experiments."""

    # Workload class registry
    WORKLOAD_CLASSES = {
        WorkloadCategory.LLM_INFERENCE: LLMInferenceWorkload,
        WorkloadCategory.RAG: RAGWorkload,
        WorkloadCategory.ANN_SEARCH: ANNSearchWorkload,
        WorkloadCategory.GPU_KERNEL: GPUKernelWorkload,
    }

    def __init__(self, results_dir: str = './results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(
        self,
        config: ExperimentConfig,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> ExperimentResult:
        """Run complete experiment with all workloads."""
        start_time = datetime.now()
        summaries: List[WorkloadSummary] = []

        # Collect system info
        config.system_info = self._collect_system_info()

        runner = WorkloadRunner()

        for workload_config in config.workloads:
            # Create workload instance
            workload = self._create_workload(workload_config)

            # Progress wrapper
            def workload_progress(completed: int, total: int) -> None:
                if progress_callback:
                    progress_callback(workload_config.name, completed, total)

            # Run benchmark
            summary = runner.run(workload, progress_callback=workload_progress)
            summaries.append(summary)

            print(f"Completed {workload_config.name}: "
                  f"{summary.requests_per_second:.1f} req/s, "
                  f"p99={summary.latency_p99:.2f}ms")

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
        cls = self.WORKLOAD_CLASSES.get(config.category)
        if cls is None:
            raise ValueError(f"Unknown workload category: {config.category}")
        return cls(config)

    def _collect_system_info(self) -> Dict[str, str]:
        """Collect system information for reproducibility."""
        info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine(),
        }

        # Try to get GPU info
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info['gpu'] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return info

    def _save_result(self, result: ExperimentResult) -> str:
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
                'random_seed': result.config.random_seed,
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
                    'latency_std': s.latency_std,
                    'requests_per_second': s.requests_per_second,
                    'custom_metrics': s.custom_metrics,
                }
                for s in result.summaries
            ],
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'metadata': result.metadata,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved results to {path}")
        return str(path)

    def load_result(self, path: str) -> ExperimentResult:
        """Load experiment result from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct summaries
        summaries = []
        for s in data['summaries']:
            summaries.append(WorkloadSummary(
                workload_name=s['workload_name'],
                total_requests=s['total_requests'],
                successful_requests=s['successful_requests'],
                failed_requests=s['failed_requests'],
                duration_seconds=s['duration_seconds'],
                latency_p50=s['latency_p50'],
                latency_p90=s['latency_p90'],
                latency_p99=s['latency_p99'],
                latency_mean=s['latency_mean'],
                latency_std=s.get('latency_std', 0),
                requests_per_second=s['requests_per_second'],
                custom_metrics=s.get('custom_metrics', {})
            ))

        config = ExperimentConfig(
            name=data['config']['name'],
            description=data['config'].get('description', ''),
            system_info=data['config'].get('system_info', {}),
            random_seed=data['config'].get('random_seed', 42)
        )

        return ExperimentResult(
            config=config,
            summaries=summaries,
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
            metadata=data.get('metadata', {})
        )

    def compare_results(
        self,
        result1: ExperimentResult,
        result2: ExperimentResult
    ) -> Dict[str, Any]:
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

            # Avoid division by zero
            throughput_change = 0 if s1.requests_per_second == 0 else \
                (s2.requests_per_second - s1.requests_per_second) / s1.requests_per_second * 100

            latency_p99_change = 0 if s1.latency_p99 == 0 else \
                (s2.latency_p99 - s1.latency_p99) / s1.latency_p99 * 100

            latency_p50_change = 0 if s1.latency_p50 == 0 else \
                (s2.latency_p50 - s1.latency_p50) / s1.latency_p50 * 100

            comparison['workloads'][name] = {
                'throughput_change': throughput_change,
                'latency_p99_change': latency_p99_change,
                'latency_p50_change': latency_p50_change,
                'baseline_throughput': s1.requests_per_second,
                'current_throughput': s2.requests_per_second,
                'baseline_p99': s1.latency_p99,
                'current_p99': s2.latency_p99,
            }

        return comparison


# =============================================================================
# Regression Detection
# =============================================================================

@dataclass
class Regression:
    """Represents a detected performance regression."""
    workload: str
    metric: str
    baseline_value: float
    current_value: float
    change_pct: float
    severity: str  # 'minor', 'moderate', 'severe'


class RegressionDetector:
    """Detect performance regressions between benchmark runs."""

    def __init__(
        self,
        throughput_threshold: float = 0.05,  # 5% regression threshold
        latency_threshold: float = 0.10,     # 10% regression threshold
        minor_threshold: float = 0.05,
        moderate_threshold: float = 0.15,
        severe_threshold: float = 0.30
    ):
        self.throughput_threshold = throughput_threshold
        self.latency_threshold = latency_threshold
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold

    def detect(
        self,
        baseline: ExperimentResult,
        current: ExperimentResult
    ) -> List[Regression]:
        """
        Detect regressions between baseline and current.

        Returns list of detected regressions.
        """
        regressions: List[Regression] = []

        # Match workloads
        baseline_summaries = {s.workload_name: s for s in baseline.summaries}
        current_summaries = {s.workload_name: s for s in current.summaries}

        for name in baseline_summaries.keys() & current_summaries.keys():
            base = baseline_summaries[name]
            curr = current_summaries[name]

            # Check throughput regression
            if base.requests_per_second > 0:
                throughput_change = (curr.requests_per_second - base.requests_per_second) / base.requests_per_second

                if throughput_change < -self.throughput_threshold:
                    regressions.append(Regression(
                        workload=name,
                        metric='throughput',
                        baseline_value=base.requests_per_second,
                        current_value=curr.requests_per_second,
                        change_pct=throughput_change * 100,
                        severity=self._classify_severity(abs(throughput_change))
                    ))

            # Check latency regression (p99)
            if base.latency_p99 > 0:
                latency_change = (curr.latency_p99 - base.latency_p99) / base.latency_p99

                if latency_change > self.latency_threshold:
                    regressions.append(Regression(
                        workload=name,
                        metric='latency_p99',
                        baseline_value=base.latency_p99,
                        current_value=curr.latency_p99,
                        change_pct=latency_change * 100,
                        severity=self._classify_severity(abs(latency_change))
                    ))

            # Check p50 latency regression
            if base.latency_p50 > 0:
                p50_change = (curr.latency_p50 - base.latency_p50) / base.latency_p50

                if p50_change > self.latency_threshold:
                    regressions.append(Regression(
                        workload=name,
                        metric='latency_p50',
                        baseline_value=base.latency_p50,
                        current_value=curr.latency_p50,
                        change_pct=p50_change * 100,
                        severity=self._classify_severity(abs(p50_change))
                    ))

        return regressions

    def _classify_severity(self, change: float) -> str:
        """Classify regression severity."""
        if change >= self.severe_threshold:
            return 'severe'
        elif change >= self.moderate_threshold:
            return 'moderate'
        else:
            return 'minor'

    def generate_report(self, regressions: List[Regression]) -> str:
        """Generate human-readable regression report."""
        if not regressions:
            return "No regressions detected."

        lines = ["Performance Regression Report", "=" * 40, ""]

        # Group by severity
        severe = [r for r in regressions if r.severity == 'severe']
        moderate = [r for r in regressions if r.severity == 'moderate']
        minor = [r for r in regressions if r.severity == 'minor']

        if severe:
            lines.append("SEVERE REGRESSIONS:")
            for r in severe:
                lines.append(f"  - {r.workload}: {r.metric} "
                             f"changed {r.change_pct:+.1f}%")
            lines.append("")

        if moderate:
            lines.append("MODERATE REGRESSIONS:")
            for r in moderate:
                lines.append(f"  - {r.workload}: {r.metric} "
                             f"changed {r.change_pct:+.1f}%")
            lines.append("")

        if minor:
            lines.append("MINOR REGRESSIONS:")
            for r in minor:
                lines.append(f"  - {r.workload}: {r.metric} "
                             f"changed {r.change_pct:+.1f}%")

        return "\n".join(lines)


# =============================================================================
# Multi-Node Harness
# =============================================================================

@dataclass
class NodeConfig:
    """Configuration for a benchmark node."""
    host: str
    port: int = 22
    username: str = ""
    ssh_key_path: str = ""
    working_dir: str = "/tmp/benchmark"


class MultiNodeHarness:
    """Run benchmarks across multiple nodes."""

    def __init__(self, nodes: List[NodeConfig]):
        self.nodes = nodes
        self._results: Dict[str, ExperimentResult] = {}

    def run_distributed(
        self,
        config: ExperimentConfig,
        parallel: bool = True
    ) -> Dict[str, ExperimentResult]:
        """
        Run experiment on multiple nodes.

        Args:
            config: Experiment configuration
            parallel: Whether to run on all nodes in parallel

        Returns:
            Dict mapping node host to ExperimentResult
        """
        results: Dict[str, ExperimentResult] = {}

        if parallel:
            with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
                futures = {
                    executor.submit(self._run_on_node, node, config): node
                    for node in self.nodes
                }

                for future in futures:
                    node = futures[future]
                    try:
                        result = future.result()
                        results[node.host] = result
                    except Exception as e:
                        print(f"Error on node {node.host}: {e}")
        else:
            for node in self.nodes:
                try:
                    result = self._run_on_node(node, config)
                    results[node.host] = result
                except Exception as e:
                    print(f"Error on node {node.host}: {e}")

        self._results = results
        return results

    def _run_on_node(
        self,
        node: NodeConfig,
        config: ExperimentConfig
    ) -> ExperimentResult:
        """Run benchmark on a single node."""
        # In a real implementation, this would:
        # 1. SSH to node
        # 2. Copy benchmark code/config
        # 3. Execute benchmark
        # 4. Collect results

        # For now, simulate by running locally with node identifier
        manager = ExperimentManager(results_dir=f"./results/{node.host}")

        # Modify config to identify node
        node_config = ExperimentConfig(
            name=f"{config.name}__{node.host}",
            description=f"{config.description} (Node: {node.host})",
            workloads=config.workloads,
            random_seed=config.random_seed,
        )

        return manager.run_experiment(node_config)

    def aggregate_results(
        self,
        results: Optional[Dict[str, ExperimentResult]] = None
    ) -> WorkloadSummary:
        """
        Aggregate results from multiple nodes.

        Returns combined summary statistics.
        """
        if results is None:
            results = self._results

        if not results:
            raise ValueError("No results to aggregate")

        # Collect all summaries
        all_summaries: Dict[str, List[WorkloadSummary]] = {}

        for node_host, result in results.items():
            for summary in result.summaries:
                base_name = summary.workload_name.split("__")[0]
                if base_name not in all_summaries:
                    all_summaries[base_name] = []
                all_summaries[base_name].append(summary)

        # Aggregate each workload
        aggregated: List[WorkloadSummary] = []

        for workload_name, summaries in all_summaries.items():
            total_requests = sum(s.total_requests for s in summaries)
            successful_requests = sum(s.successful_requests for s in summaries)
            total_duration = sum(s.duration_seconds for s in summaries)

            # Average latencies
            latency_p50 = statistics.mean(s.latency_p50 for s in summaries)
            latency_p90 = statistics.mean(s.latency_p90 for s in summaries)
            latency_p99 = statistics.mean(s.latency_p99 for s in summaries)
            latency_mean = statistics.mean(s.latency_mean for s in summaries)

            # Sum throughput
            total_throughput = sum(s.requests_per_second for s in summaries)

            aggregated.append(WorkloadSummary(
                workload_name=f"{workload_name}_aggregate",
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=total_requests - successful_requests,
                duration_seconds=total_duration / len(summaries),
                latency_p50=latency_p50,
                latency_p90=latency_p90,
                latency_p99=latency_p99,
                latency_mean=latency_mean,
                latency_std=0,  # Would need to compute properly
                requests_per_second=total_throughput,
                custom_metrics={
                    'num_nodes': len(summaries),
                    'total_requests_all_nodes': total_requests,
                }
            ))

        # Return first aggregated summary (or combine all)
        if aggregated:
            return aggregated[0]

        raise ValueError("No summaries to aggregate")

    def compare_nodes(self) -> Dict[str, Any]:
        """Compare performance across nodes."""
        if not self._results:
            return {}

        comparison = {
            'nodes': {},
            'rankings': {}
        }

        # Extract metrics per node
        for node_host, result in self._results.items():
            node_metrics = {}
            for summary in result.summaries:
                node_metrics[summary.workload_name] = {
                    'throughput': summary.requests_per_second,
                    'latency_p99': summary.latency_p99,
                }
            comparison['nodes'][node_host] = node_metrics

        # Rank nodes by throughput
        throughputs = {
            host: sum(
                s.requests_per_second
                for s in result.summaries
            )
            for host, result in self._results.items()
        }

        ranked = sorted(throughputs.items(), key=lambda x: x[1], reverse=True)
        comparison['rankings']['by_throughput'] = [host for host, _ in ranked]

        return comparison
