"""Quality metrics collection and monitoring."""

from collections import defaultdict
from typing import Any
import time
import numpy as np


class QualityMetricsCollector:
    """Collects and aggregates quality metrics."""

    def __init__(self):
        self.metrics: dict[str, list] = defaultdict(list)

    def record(
        self,
        component: str,
        metric_name: str,
        value: float,
        timestamp: float = None
    ):
        """Record a metric value.

        Args:
            component: Component name
            metric_name: Metric name
            value: Metric value
            timestamp: Optional timestamp
        """
        key = f"{component}.{metric_name}"
        self.metrics[key].append({
            "value": value,
            "timestamp": timestamp or time.time()
        })

    def get_aggregates(self, component: str = None) -> dict[str, dict]:
        """Get aggregated metrics.

        Args:
            component: Optional component filter

        Returns:
            Aggregated metrics
        """
        results = {}

        for key, values in self.metrics.items():
            if component and not key.startswith(component):
                continue

            vals = [v["value"] for v in values]
            if not vals:
                continue

            results[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "count": len(vals)
            }

        return results

    def get_recent(
        self,
        component: str,
        metric_name: str,
        n: int = 100
    ) -> list[float]:
        """Get recent metric values.

        Args:
            component: Component name
            metric_name: Metric name
            n: Number of values

        Returns:
            Recent values
        """
        key = f"{component}.{metric_name}"
        values = self.metrics.get(key, [])
        return [v["value"] for v in values[-n:]]

    def clear(self, component: str = None):
        """Clear metrics.

        Args:
            component: Optional component to clear
        """
        if component:
            keys_to_remove = [k for k in self.metrics if k.startswith(component)]
            for k in keys_to_remove:
                del self.metrics[k]
        else:
            self.metrics.clear()


class PipelineMetrics:
    """Specialized metrics for pipeline execution."""

    def __init__(self):
        self.collector = QualityMetricsCollector()
        self._request_count = 0
        self._error_count = 0

    def record_latency(self, stage: str, latency_ms: float):
        """Record stage latency."""
        self.collector.record(stage, "latency_ms", latency_ms)

    def record_quality(self, stage: str, score: float):
        """Record quality score."""
        self.collector.record(stage, "quality", score)

    def record_throughput(self, requests_per_second: float):
        """Record throughput."""
        self.collector.record("pipeline", "throughput", requests_per_second)

    def record_request(self, success: bool = True):
        """Record a request."""
        self._request_count += 1
        if not success:
            self._error_count += 1

    def get_error_rate(self) -> float:
        """Get error rate."""
        if self._request_count == 0:
            return 0.0
        return self._error_count / self._request_count

    def get_summary(self) -> dict:
        """Get metrics summary."""
        aggregates = self.collector.get_aggregates()

        return {
            "request_count": self._request_count,
            "error_rate": self.get_error_rate(),
            "stage_metrics": aggregates
        }


# Global metrics instance
_pipeline_metrics = None


def get_metrics() -> PipelineMetrics:
    """Get global metrics instance."""
    global _pipeline_metrics
    if _pipeline_metrics is None:
        _pipeline_metrics = PipelineMetrics()
    return _pipeline_metrics
