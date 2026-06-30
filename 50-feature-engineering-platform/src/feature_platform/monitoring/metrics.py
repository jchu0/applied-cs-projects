"""Metrics collection for feature platform."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import time
import threading


@dataclass
class FeatureMetrics:
    """Metrics for a single feature."""

    feature_view: str
    feature_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    count: int = 0
    null_count: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_view": self.feature_view,
            "feature_name": self.feature_name,
            "timestamp": self.timestamp.isoformat(),
            "count": self.count,
            "null_count": self.null_count,
            "null_ratio": self.null_count / self.count if self.count > 0 else 0,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_value,
            "max": self.max_value,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass
class OperationMetrics:
    """Metrics for an operation."""

    operation: str
    duration_ms: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collector for feature platform metrics.

    Tracks:
    - Feature statistics
    - Operation latencies
    - Error rates
    - Request counts
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._feature_metrics: Dict[str, List[FeatureMetrics]] = {}
        self._operation_metrics: List[OperationMetrics] = []
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()

    def record_feature_metrics(self, metrics: FeatureMetrics) -> None:
        """Record metrics for a feature."""
        if not self.enabled:
            return

        with self._lock:
            key = f"{metrics.feature_view}:{metrics.feature_name}"
            if key not in self._feature_metrics:
                self._feature_metrics[key] = []
            self._feature_metrics[key].append(metrics)

            # Keep only last 1000 records
            if len(self._feature_metrics[key]) > 1000:
                self._feature_metrics[key] = self._feature_metrics[key][-1000:]

    def record_operation(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        **metadata,
    ) -> None:
        """Record an operation metric."""
        if not self.enabled:
            return

        metric = OperationMetrics(
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata,
        )

        with self._lock:
            self._operation_metrics.append(metric)

            # Keep only last 10000 records
            if len(self._operation_metrics) > 10000:
                self._operation_metrics = self._operation_metrics[-10000:]

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        if not self.enabled:
            return

        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        if not self.enabled:
            return

        with self._lock:
            self._gauges[name] = value

    def get_counter(self, name: str) -> int:
        """Get a counter value."""
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> Optional[float]:
        """Get a gauge value."""
        with self._lock:
            return self._gauges.get(name)

    def get_feature_metrics(
        self,
        feature_view: str,
        feature_name: str,
        limit: int = 100,
    ) -> List[FeatureMetrics]:
        """Get metrics for a feature."""
        with self._lock:
            key = f"{feature_view}:{feature_name}"
            metrics = self._feature_metrics.get(key, [])
            return metrics[-limit:]

    def get_operation_stats(
        self,
        operation: Optional[str] = None,
        minutes: int = 60,
    ) -> Dict[str, Any]:
        """Get operation statistics."""
        cutoff = datetime.utcnow().timestamp() - (minutes * 60)

        with self._lock:
            filtered = [
                m for m in self._operation_metrics
                if m.timestamp.timestamp() > cutoff
                and (operation is None or m.operation == operation)
            ]

        if not filtered:
            return {
                "count": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "p50_duration_ms": 0.0,
                "p95_duration_ms": 0.0,
                "p99_duration_ms": 0.0,
            }

        durations = [m.duration_ms for m in filtered]
        successes = [m.success for m in filtered]

        import numpy as np
        durations_arr = np.array(durations)

        return {
            "count": len(filtered),
            "success_rate": sum(successes) / len(successes),
            "avg_duration_ms": float(np.mean(durations_arr)),
            "p50_duration_ms": float(np.percentile(durations_arr, 50)),
            "p95_duration_ms": float(np.percentile(durations_arr, 95)),
            "p99_duration_ms": float(np.percentile(durations_arr, 99)),
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "feature_metrics_count": sum(
                    len(m) for m in self._feature_metrics.values()
                ),
                "operation_metrics_count": len(self._operation_metrics),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._feature_metrics.clear()
            self._operation_metrics.clear()
            self._counters.clear()
            self._gauges.clear()


class MetricsTimer:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricsCollector,
        operation: str,
        **metadata,
    ):
        self.collector = collector
        self.operation = operation
        self.metadata = metadata
        self.start_time: Optional[float] = None
        self.success = True

    def __enter__(self) -> "MetricsTimer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.success = False

        duration_ms = (time.time() - self.start_time) * 1000
        self.collector.record_operation(
            self.operation,
            duration_ms,
            self.success,
            **self.metadata,
        )

    def mark_failure(self) -> None:
        """Mark the operation as failed."""
        self.success = False


# Global metrics collector
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def set_metrics_collector(collector: MetricsCollector) -> None:
    """Set the global metrics collector."""
    global _global_collector
    _global_collector = collector
