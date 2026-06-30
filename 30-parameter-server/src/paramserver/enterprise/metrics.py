"""Performance metrics for parameter server.

Tracks throughput, latency, and other performance metrics
for monitoring and optimization.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot.

    Attributes:
        timestamp: When metrics were captured.
        throughput: Updates per second.
        latency_ms: Average latency in milliseconds.
        push_count: Number of push operations.
        pull_count: Number of pull operations.
        bytes_sent: Total bytes sent.
        bytes_received: Total bytes received.
    """
    timestamp: float = field(default_factory=time.time)
    throughput: float = 0.0
    latency_ms: float = 0.0
    push_count: int = 0
    pull_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0


class MetricsCollector:
    """Collects and aggregates performance metrics.

    Tracks various performance metrics for the parameter server
    and provides aggregated statistics.

    Attributes:
        window_size: Number of samples to keep for averaging.
    """

    def __init__(self, window_size: int = 1000):
        """Initialize metrics collector.

        Args:
            window_size: Number of samples to keep.
        """
        self.window_size = window_size

        # Counters
        self._push_count = 0
        self._pull_count = 0
        self._bytes_sent = 0
        self._bytes_received = 0

        # Latency samples
        self._push_latencies: List[float] = []
        self._pull_latencies: List[float] = []

        # Timing
        self._start_time = time.time()
        self._last_report_time = time.time()

    def record_push(
        self,
        latency_ms: float,
        bytes_count: int = 0,
    ) -> None:
        """Record a push operation.

        Args:
            latency_ms: Operation latency in milliseconds.
            bytes_count: Bytes transmitted.
        """
        self._push_count += 1
        self._bytes_sent += bytes_count

        self._push_latencies.append(latency_ms)
        if len(self._push_latencies) > self.window_size:
            self._push_latencies.pop(0)

    def record_pull(
        self,
        latency_ms: float,
        bytes_count: int = 0,
    ) -> None:
        """Record a pull operation.

        Args:
            latency_ms: Operation latency in milliseconds.
            bytes_count: Bytes received.
        """
        self._pull_count += 1
        self._bytes_received += bytes_count

        self._pull_latencies.append(latency_ms)
        if len(self._pull_latencies) > self.window_size:
            self._pull_latencies.pop(0)

    def get_throughput(self) -> float:
        """Get operations per second.

        Returns:
            Throughput in ops/sec.
        """
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0

        total_ops = self._push_count + self._pull_count
        return total_ops / elapsed

    def get_push_latency(self) -> float:
        """Get average push latency.

        Returns:
            Average latency in milliseconds.
        """
        if not self._push_latencies:
            return 0.0
        return sum(self._push_latencies) / len(self._push_latencies)

    def get_pull_latency(self) -> float:
        """Get average pull latency.

        Returns:
            Average latency in milliseconds.
        """
        if not self._pull_latencies:
            return 0.0
        return sum(self._pull_latencies) / len(self._pull_latencies)

    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics snapshot.

        Returns:
            PerformanceMetrics instance.
        """
        all_latencies = self._push_latencies + self._pull_latencies
        avg_latency = 0.0
        if all_latencies:
            avg_latency = sum(all_latencies) / len(all_latencies)

        return PerformanceMetrics(
            timestamp=time.time(),
            throughput=self.get_throughput(),
            latency_ms=avg_latency,
            push_count=self._push_count,
            pull_count=self._pull_count,
            bytes_sent=self._bytes_sent,
            bytes_received=self._bytes_received,
        )

    def get_stats(self) -> Dict[str, float]:
        """Get statistics dictionary.

        Returns:
            Dictionary of statistics.
        """
        elapsed = time.time() - self._start_time

        return {
            "elapsed_seconds": elapsed,
            "total_pushes": self._push_count,
            "total_pulls": self._pull_count,
            "throughput_ops_sec": self.get_throughput(),
            "avg_push_latency_ms": self.get_push_latency(),
            "avg_pull_latency_ms": self.get_pull_latency(),
            "bytes_sent": self._bytes_sent,
            "bytes_received": self._bytes_received,
            "bandwidth_sent_mb_sec": (
                self._bytes_sent / (1024 * 1024) / elapsed
                if elapsed > 0 else 0
            ),
            "bandwidth_recv_mb_sec": (
                self._bytes_received / (1024 * 1024) / elapsed
                if elapsed > 0 else 0
            ),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._push_count = 0
        self._pull_count = 0
        self._bytes_sent = 0
        self._bytes_received = 0
        self._push_latencies.clear()
        self._pull_latencies.clear()
        self._start_time = time.time()
        self._last_report_time = time.time()
