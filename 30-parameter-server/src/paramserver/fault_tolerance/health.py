"""Health monitoring for parameter servers and workers."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status of a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthRecord:
    """Health record for a monitored component.

    Attributes:
        component_id: Identifier for the component.
        component_type: Type of component (server, worker).
        status: Current health status.
        last_heartbeat: Time of last heartbeat.
        consecutive_failures: Number of consecutive health check failures.
        metadata: Additional component metadata.
    """
    component_id: str
    component_type: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_heartbeat: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """Monitors health of parameter servers and workers.

    Tracks heartbeats, detects failures, and triggers recovery actions.

    Attributes:
        heartbeat_interval: Expected interval between heartbeats.
        failure_threshold: Number of missed heartbeats before failure.
    """

    def __init__(
        self,
        heartbeat_interval: float = 5.0,
        failure_threshold: int = 3,
        degraded_threshold: int = 1,
    ):
        """Initialize health monitor.

        Args:
            heartbeat_interval: Expected time between heartbeats in seconds.
            failure_threshold: Missed heartbeats before marking unhealthy.
            degraded_threshold: Missed heartbeats before marking degraded.
        """
        self.heartbeat_interval = heartbeat_interval
        self.failure_threshold = failure_threshold
        self.degraded_threshold = degraded_threshold

        # Track health records
        self._records: Dict[str, HealthRecord] = {}

        # Failure callbacks
        self._failure_callbacks: List[Callable[[str, HealthRecord], None]] = []
        self._recovery_callbacks: List[Callable[[str, HealthRecord], None]] = []

        # Monitoring state
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    def register_component(
        self,
        component_id: str,
        component_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a component for health monitoring.

        Args:
            component_id: Unique identifier for the component.
            component_type: Type of component (e.g., "server", "worker").
            metadata: Optional additional metadata.
        """
        record = HealthRecord(
            component_id=component_id,
            component_type=component_type,
            status=HealthStatus.HEALTHY,
            last_heartbeat=time.time(),
            metadata=metadata or {},
        )
        self._records[component_id] = record

    def unregister_component(self, component_id: str) -> Optional[HealthRecord]:
        """Unregister a component.

        Args:
            component_id: Component to unregister.

        Returns:
            Removed HealthRecord or None.
        """
        return self._records.pop(component_id, None)

    async def record_heartbeat(
        self,
        component_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a heartbeat from a component.

        Args:
            component_id: Component sending heartbeat.
            metadata: Optional updated metadata.
        """
        async with self._lock:
            if component_id not in self._records:
                # Auto-register unknown components
                self.register_component(component_id, "unknown", metadata)
                return

            record = self._records[component_id]
            was_unhealthy = record.status == HealthStatus.UNHEALTHY

            record.last_heartbeat = time.time()
            record.consecutive_failures = 0
            record.status = HealthStatus.HEALTHY

            if metadata:
                record.metadata.update(metadata)

            # Trigger recovery callbacks if recovering from failure
            if was_unhealthy:
                logger.info("Component %s recovered (now HEALTHY)", component_id)
                for callback in self._recovery_callbacks:
                    try:
                        callback(component_id, record)
                    except Exception:
                        pass

    async def check_health(self, component_id: str) -> HealthStatus:
        """Check health status of a component.

        Args:
            component_id: Component to check.

        Returns:
            Current health status.
        """
        if component_id not in self._records:
            return HealthStatus.UNKNOWN

        record = self._records[component_id]
        return self._evaluate_health(record)

    def _evaluate_health(self, record: HealthRecord) -> HealthStatus:
        """Evaluate health status based on heartbeat timing.

        Args:
            record: Health record to evaluate.

        Returns:
            Evaluated health status.
        """
        elapsed = time.time() - record.last_heartbeat
        missed = int(elapsed / self.heartbeat_interval)

        if missed >= self.failure_threshold:
            return HealthStatus.UNHEALTHY
        elif missed >= self.degraded_threshold:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    async def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started (interval=%.1fs)", self.heartbeat_interval)

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            await self._check_all_components()
            await asyncio.sleep(self.heartbeat_interval)

    async def _check_all_components(self) -> None:
        """Check health of all registered components."""
        async with self._lock:
            for component_id, record in self._records.items():
                old_status = record.status
                new_status = self._evaluate_health(record)

                if new_status != old_status:
                    record.status = new_status
                    log = (
                        logger.warning
                        if new_status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED)
                        else logger.info
                    )
                    log(
                        "Component %s health %s -> %s",
                        component_id, old_status.value, new_status.value,
                    )

                    if new_status == HealthStatus.UNHEALTHY:
                        record.consecutive_failures += 1
                        # Trigger failure callbacks
                        for callback in self._failure_callbacks:
                            try:
                                callback(component_id, record)
                            except Exception:
                                pass

    def add_failure_callback(
        self,
        callback: Callable[[str, HealthRecord], None],
    ) -> None:
        """Add a callback for component failures.

        Args:
            callback: Function called with (component_id, record) on failure.
        """
        self._failure_callbacks.append(callback)

    def add_recovery_callback(
        self,
        callback: Callable[[str, HealthRecord], None],
    ) -> None:
        """Add a callback for component recovery.

        Args:
            callback: Function called with (component_id, record) on recovery.
        """
        self._recovery_callbacks.append(callback)

    def get_status(self, component_id: str) -> Optional[HealthStatus]:
        """Get current status of a component.

        Args:
            component_id: Component to query.

        Returns:
            Health status or None if not found.
        """
        if component_id not in self._records:
            return None
        return self._records[component_id].status

    def get_record(self, component_id: str) -> Optional[HealthRecord]:
        """Get full health record for a component.

        Args:
            component_id: Component to query.

        Returns:
            HealthRecord or None.
        """
        return self._records.get(component_id)

    def get_all_records(self) -> Dict[str, HealthRecord]:
        """Get all health records.

        Returns:
            Dictionary of component_id to HealthRecord.
        """
        return self._records.copy()

    def get_unhealthy_components(self) -> List[str]:
        """Get list of unhealthy component IDs.

        Returns:
            List of unhealthy component IDs.
        """
        return [
            cid for cid, record in self._records.items()
            if record.status == HealthStatus.UNHEALTHY
        ]

    def get_healthy_components(self) -> List[str]:
        """Get list of healthy component IDs.

        Returns:
            List of healthy component IDs.
        """
        return [
            cid for cid, record in self._records.items()
            if record.status == HealthStatus.HEALTHY
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics.

        Returns:
            Dictionary of statistics.
        """
        status_counts = {s.value: 0 for s in HealthStatus}
        for record in self._records.values():
            status_counts[record.status.value] += 1

        return {
            "total_components": len(self._records),
            "monitoring_active": self._monitoring,
            "heartbeat_interval": self.heartbeat_interval,
            "failure_threshold": self.failure_threshold,
            "status_counts": status_counts,
        }

    def is_component_healthy(self, component_id: str) -> bool:
        """Check if a component is healthy.

        Args:
            component_id: Component to check.

        Returns:
            True if healthy, False otherwise.
        """
        if component_id not in self._records:
            return False
        return self._records[component_id].status == HealthStatus.HEALTHY

    async def wait_for_healthy(
        self,
        component_id: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """Wait for a component to become healthy.

        Args:
            component_id: Component to wait for.
            timeout: Maximum wait time in seconds.

        Returns:
            True if component became healthy, False if timed out.
        """
        start = time.time()
        while True:
            if self.is_component_healthy(component_id):
                return True

            if timeout and (time.time() - start) > timeout:
                return False

            await asyncio.sleep(self.heartbeat_interval / 2)
