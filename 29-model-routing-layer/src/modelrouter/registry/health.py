"""Health checking for workers."""

import time
import asyncio
import logging
from typing import Any

from ..schemas import WorkerInfo, GPUInfo

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitors GPU utilization and health."""

    def __init__(self, poll_interval: int = 5):
        """Initialize monitor.

        Args:
            poll_interval: Polling interval in seconds
        """
        self.poll_interval = poll_interval

    async def collect_metrics(self, worker_id: str) -> GPUInfo:
        """Collect GPU metrics.

        Args:
            worker_id: Worker ID

        Returns:
            GPU metrics
        """
        # Mock implementation (would use pynvml in production)
        return GPUInfo(
            device_name="NVIDIA A100",
            memory_total_mb=40960,
            memory_used_mb=8192,
            utilization_percent=45.0,
            temperature_celsius=65
        )

    async def start_monitoring(self, worker_id: str, callback):
        """Start continuous monitoring.

        Args:
            worker_id: Worker ID
            callback: Callback for metrics
        """
        while True:
            try:
                metrics = await self.collect_metrics(worker_id)
                await callback(worker_id, metrics)
            except Exception as e:
                logger.error(f"GPU monitoring failed: {e}")

            await asyncio.sleep(self.poll_interval)


class HealthChecker:
    """Performs health checks on workers."""

    def __init__(
        self,
        registry,
        check_interval: int = 10,
        unhealthy_threshold: int = 3
    ):
        """Initialize health checker.

        Args:
            registry: Worker registry
            check_interval: Check interval in seconds
            unhealthy_threshold: Failures before unhealthy
        """
        self.registry = registry
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.failure_counts: dict[str, int] = {}
        self._running = False

    async def start(self):
        """Start health checking loop."""
        self._running = True

        while self._running:
            workers = await self.registry.get_workers(status=None)

            for worker in workers:
                is_healthy = await self._check_worker(worker)
                await self._update_status(worker.worker_id, is_healthy)

            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop health checking."""
        self._running = False

    async def _check_worker(self, worker: WorkerInfo) -> bool:
        """Check if worker is healthy.

        Args:
            worker: Worker info

        Returns:
            True if healthy
        """
        try:
            # Check heartbeat freshness
            if time.time() - worker.last_heartbeat > 30:
                logger.warning(f"Worker {worker.worker_id} heartbeat stale")
                return False

            # Mock health check (would ping worker in production)
            return await self._ping_worker(worker)

        except Exception as e:
            logger.error(f"Health check failed for {worker.worker_id}: {e}")
            return False

    async def _ping_worker(self, worker: WorkerInfo) -> bool:
        """Ping worker endpoint.

        Args:
            worker: Worker info

        Returns:
            True if responsive
        """
        # Mock implementation
        return True

    async def _update_status(self, worker_id: str, is_healthy: bool):
        """Update worker health status.

        Args:
            worker_id: Worker ID
            is_healthy: Health status
        """
        if is_healthy:
            self.failure_counts[worker_id] = 0
            await self.registry.update_status(worker_id, "healthy")
        else:
            self.failure_counts[worker_id] = self.failure_counts.get(worker_id, 0) + 1

            if self.failure_counts[worker_id] >= self.unhealthy_threshold:
                logger.warning(f"Marking worker {worker_id} as unhealthy")
                await self.registry.update_status(worker_id, "unhealthy")

    async def check_once(self) -> dict[str, bool]:
        """Run one health check cycle.

        Returns:
            Worker ID to health status mapping
        """
        workers = await self.registry.get_workers(status=None)
        results = {}

        for worker in workers:
            is_healthy = await self._check_worker(worker)
            await self._update_status(worker.worker_id, is_healthy)
            results[worker.worker_id] = is_healthy

        return results
