"""Node management for distributed training infrastructure."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4
import structlog

from ml_orchestrator.core.models import ResourceRequest, WorkerInfo, WorkerStatus
from ml_orchestrator.core.exceptions import WorkerNotFoundError, WorkerUnhealthyError


logger = structlog.get_logger(__name__)


class NodeHealth(str, Enum):
    """Health status of a node."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class NodeMetrics:
    """Metrics collected from a node."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    gpu_utilization: list[float] = field(default_factory=list)
    gpu_memory_used: list[float] = field(default_factory=list)
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0
    load_average_15m: float = 0.0


@dataclass
class NodeEvent:
    """Event that occurred on a node."""

    id: str = field(default_factory=lambda: str(uuid4()))
    node_id: str = ""
    event_type: str = ""
    message: str = ""
    severity: str = "info"  # info, warning, error
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class NodeManager:
    """
    Manages worker nodes in the cluster.

    Responsibilities:
    - Node registration and deregistration
    - Health monitoring via heartbeats
    - Metrics collection
    - Node lifecycle management
    """

    def __init__(
        self,
        heartbeat_timeout_seconds: int = 60,
        health_check_interval_seconds: float = 10.0,
        drain_timeout_seconds: int = 300,
    ):
        self._nodes: dict[str, WorkerInfo] = {}
        self._metrics: dict[str, list[NodeMetrics]] = {}  # node_id -> metrics history
        self._events: list[NodeEvent] = []
        self._heartbeat_timeout = heartbeat_timeout_seconds
        self._health_check_interval = health_check_interval_seconds
        self._drain_timeout = drain_timeout_seconds
        self._lock = asyncio.Lock()
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._callbacks: dict[str, list[Callable]] = {}

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for node events."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    async def _emit_event(self, event: str, node: WorkerInfo, data: dict[str, Any]) -> None:
        """Emit event to callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(node, data)
                else:
                    callback(node, data)
            except Exception as e:
                logger.error("callback_error", event=event, error=str(e))

    async def start(self) -> None:
        """Start the node manager background tasks."""
        if self._running:
            return
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("node_manager_started")

    async def stop(self) -> None:
        """Stop the node manager."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("node_manager_stopped")

    async def _health_check_loop(self) -> None:
        """Background loop for health checking."""
        while self._running:
            try:
                await self._check_all_nodes()
            except Exception as e:
                logger.error("health_check_error", error=str(e))
            await asyncio.sleep(self._health_check_interval)

    async def _check_all_nodes(self) -> None:
        """Check health of all nodes."""
        async with self._lock:
            now = datetime.utcnow()
            for node_id, node in list(self._nodes.items()):
                old_status = node.status
                age = (now - node.last_heartbeat).total_seconds()

                if age > self._heartbeat_timeout:
                    if node.status != WorkerStatus.OFFLINE:
                        node.status = WorkerStatus.UNHEALTHY
                        if age > self._heartbeat_timeout * 2:
                            node.status = WorkerStatus.OFFLINE
                            await self._record_event(
                                node_id, "node_offline", "Node is offline"
                            )

                if old_status != node.status:
                    await self._emit_event(
                        "status_changed",
                        node,
                        {"old_status": old_status.value, "new_status": node.status.value},
                    )

    async def _record_event(
        self,
        node_id: str,
        event_type: str,
        message: str,
        severity: str = "info",
        metadata: Optional[dict[str, Any]] = None,
    ) -> NodeEvent:
        """Record a node event."""
        event = NodeEvent(
            node_id=node_id,
            event_type=event_type,
            message=message,
            severity=severity,
            metadata=metadata or {},
        )
        self._events.append(event)
        # Keep last 10000 events
        if len(self._events) > 10000:
            self._events = self._events[-10000:]
        return event

    async def register_node(
        self,
        hostname: str,
        ip_address: str,
        resources: ResourceRequest,
        port: int = 8000,
        labels: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> WorkerInfo:
        """
        Register a new node.

        Args:
            hostname: Node hostname
            ip_address: Node IP address
            resources: Available resources
            port: Node port
            labels: Node labels for scheduling
            metadata: Additional metadata

        Returns:
            WorkerInfo for the registered node
        """
        async with self._lock:
            node = WorkerInfo(
                hostname=hostname,
                ip_address=ip_address,
                port=port,
                resources=resources,
                status=WorkerStatus.READY,
                labels=labels or {},
                metadata=metadata or {},
            )
            self._nodes[node.id] = node
            self._metrics[node.id] = []

            await self._record_event(
                node.id,
                "node_registered",
                f"Node {hostname} registered",
            )

            logger.info(
                "node_registered",
                node_id=node.id,
                hostname=hostname,
                resources=resources.model_dump(),
            )

            await self._emit_event("node_registered", node, {})
            return node

    async def unregister_node(self, node_id: str) -> Optional[WorkerInfo]:
        """Unregister a node."""
        async with self._lock:
            node = self._nodes.pop(node_id, None)
            if node:
                self._metrics.pop(node_id, None)
                await self._record_event(
                    node_id,
                    "node_unregistered",
                    f"Node {node.hostname} unregistered",
                )
                logger.info("node_unregistered", node_id=node_id)
                await self._emit_event("node_unregistered", node, {})
            return node

    async def get_node(self, node_id: str) -> WorkerInfo:
        """
        Get node by ID.

        Raises:
            WorkerNotFoundError: If node not found
        """
        async with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                raise WorkerNotFoundError(node_id)
            return node

    async def list_nodes(
        self,
        status: Optional[WorkerStatus] = None,
        labels: Optional[dict[str, str]] = None,
        healthy_only: bool = False,
    ) -> list[WorkerInfo]:
        """List nodes with optional filtering."""
        async with self._lock:
            nodes = list(self._nodes.values())

            if status:
                nodes = [n for n in nodes if n.status == status]

            if healthy_only:
                nodes = [n for n in nodes if n.is_healthy()]

            if labels:
                filtered = []
                for node in nodes:
                    if all(
                        node.labels.get(k) == v for k, v in labels.items()
                    ):
                        filtered.append(node)
                nodes = filtered

            return nodes

    async def heartbeat(self, node_id: str, metrics: Optional[NodeMetrics] = None) -> bool:
        """
        Update node heartbeat and optionally metrics.

        Returns:
            True if heartbeat accepted, False if node not found
        """
        async with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False

            node.last_heartbeat = datetime.utcnow()

            # Update status if recovering
            if node.status in (WorkerStatus.UNHEALTHY, WorkerStatus.OFFLINE):
                node.status = WorkerStatus.READY
                await self._record_event(
                    node_id,
                    "node_recovered",
                    f"Node {node.hostname} recovered",
                )

            # Store metrics
            if metrics:
                if node_id not in self._metrics:
                    self._metrics[node_id] = []
                self._metrics[node_id].append(metrics)
                # Keep last 1000 metrics per node
                if len(self._metrics[node_id]) > 1000:
                    self._metrics[node_id] = self._metrics[node_id][-1000:]

            return True

    async def update_node_status(self, node_id: str, status: WorkerStatus) -> bool:
        """Update node status."""
        async with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False

            old_status = node.status
            node.status = status

            if old_status != status:
                await self._record_event(
                    node_id,
                    "status_changed",
                    f"Status changed from {old_status.value} to {status.value}",
                )
                await self._emit_event(
                    "status_changed",
                    node,
                    {"old_status": old_status.value, "new_status": status.value},
                )

            return True

    async def drain_node(self, node_id: str, wait: bool = True) -> bool:
        """
        Drain a node (prepare for maintenance).

        Sets status to DRAINING and optionally waits for jobs to complete.

        Args:
            node_id: Node to drain
            wait: Whether to wait for jobs to finish

        Returns:
            True if drain successful
        """
        async with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False

            node.status = WorkerStatus.DRAINING
            await self._record_event(
                node_id,
                "node_draining",
                f"Node {node.hostname} draining",
            )

        if wait and node.current_jobs:
            # Wait for jobs to complete
            start = datetime.utcnow()
            while (datetime.utcnow() - start).total_seconds() < self._drain_timeout:
                async with self._lock:
                    node = self._nodes.get(node_id)
                    if not node or not node.current_jobs:
                        break
                await asyncio.sleep(1.0)

        async with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.status = WorkerStatus.OFFLINE
                await self._record_event(
                    node_id,
                    "node_drained",
                    f"Node {node.hostname} drained",
                )

        return True

    async def uncordon_node(self, node_id: str) -> bool:
        """
        Uncordon a node (return to service after drain/maintenance).
        """
        async with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False

            if node.status in (WorkerStatus.DRAINING, WorkerStatus.OFFLINE):
                node.status = WorkerStatus.READY
                await self._record_event(
                    node_id,
                    "node_uncordoned",
                    f"Node {node.hostname} returned to service",
                )
                return True

            return False

    async def get_node_metrics(
        self,
        node_id: str,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[NodeMetrics]:
        """Get metrics for a node."""
        async with self._lock:
            metrics = self._metrics.get(node_id, [])

            if since:
                metrics = [m for m in metrics if m.timestamp > since]

            return metrics[-limit:]

    async def get_node_events(
        self,
        node_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[NodeEvent]:
        """Get node events with filtering."""
        events = self._events

        if node_id:
            events = [e for e in events if e.node_id == node_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if since:
            events = [e for e in events if e.timestamp > since]

        return events[-limit:]

    def _cluster_health(self) -> dict[str, Any]:
        """Compute cluster health. Caller must hold self._lock (not reentrant)."""
        total = len(self._nodes)
        if total == 0:
            return {
                "status": NodeHealth.UNKNOWN.value,
                "total_nodes": 0,
                "healthy_nodes": 0,
                "degraded_nodes": 0,
                "unhealthy_nodes": 0,
            }

        healthy = sum(1 for n in self._nodes.values() if n.is_healthy())
        degraded = sum(
            1 for n in self._nodes.values()
            if n.status == WorkerStatus.DRAINING
        )
        unhealthy = total - healthy - degraded

        # Determine overall status
        if healthy == total:
            status = NodeHealth.HEALTHY
        elif healthy > total * 0.5:
            status = NodeHealth.DEGRADED
        else:
            status = NodeHealth.UNHEALTHY

        return {
            "status": status.value,
            "total_nodes": total,
            "healthy_nodes": healthy,
            "degraded_nodes": degraded,
            "unhealthy_nodes": unhealthy,
            "health_percent": (healthy / total) * 100,
        }

    async def get_cluster_health(self) -> dict[str, Any]:
        """Get overall cluster health."""
        async with self._lock:
            return self._cluster_health()

    async def get_stats(self) -> dict[str, Any]:
        """Get node manager statistics."""
        async with self._lock:
            # Use the lock-free helper: get_cluster_health() would re-acquire
            # self._lock, and asyncio.Lock is not reentrant (would deadlock).
            health = self._cluster_health()

            # Aggregate resources
            total_cpus = sum(n.resources.cpus for n in self._nodes.values())
            total_memory = sum(n.resources.memory_gb for n in self._nodes.values())
            total_gpus = sum(n.resources.gpus for n in self._nodes.values())

            allocated_cpus = sum(
                n.allocated_resources.cpus for n in self._nodes.values()
            )
            allocated_memory = sum(
                n.allocated_resources.memory_gb for n in self._nodes.values()
            )
            allocated_gpus = sum(
                n.allocated_resources.gpus for n in self._nodes.values()
            )

            return {
                "cluster_health": health,
                "resources": {
                    "cpus": {"total": total_cpus, "allocated": allocated_cpus},
                    "memory_gb": {"total": total_memory, "allocated": allocated_memory},
                    "gpus": {"total": total_gpus, "allocated": allocated_gpus},
                },
                "running": self._running,
                "total_events": len(self._events),
            }
