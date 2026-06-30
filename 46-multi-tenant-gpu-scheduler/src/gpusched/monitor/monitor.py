"""GPU cluster monitoring and quota management."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List, Dict
from enum import Enum
import time
from collections import defaultdict, deque

from ..core.resources import (
    Cluster, Node, Pod, Job, GPU, Queue, Tenant,
    JobState, PriorityClass
)
from ..allocator.allocator import GPUAllocation, HybridAllocator


class MetricType(Enum):
    """Types of metrics that can be collected."""
    GPU_UTILIZATION = "gpu_utilization"
    GPU_MEMORY = "gpu_memory"
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_USAGE = "memory_usage"
    NETWORK_IO = "network_io"
    POWER_USAGE = "power_usage"
    TEMPERATURE = "temperature"


class MetricPoint:
    """A single metric data point."""

    def __init__(
        self,
        metric_type: MetricType,
        value: float,
        timestamp: Optional[float] = None,
        labels: Optional[dict] = None
    ):
        self.metric_type = metric_type
        self.value = value
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.labels = labels or {}


class GPUMetrics:
    """GPU utilization metrics."""

    def __init__(
        self,
        gpu_id: str,
        utilization: float = 0.0,
        memory_used_gb: float = 0.0,
        memory_total_gb: float = 0.0,
        temperature: float = 0.0,
        power_usage: float = 0.0,
        sm_clock_mhz: int = 0,
        memory_clock_mhz: int = 0,
        pcie_throughput_mb: float = 0.0,
        error_count: int = 0,
        timestamp: Optional[float] = None
    ):
        self.gpu_id = gpu_id
        self.utilization = utilization
        self.memory_used_gb = memory_used_gb
        self.memory_total_gb = memory_total_gb
        self.temperature = temperature
        self.power_usage = power_usage
        self.sm_clock_mhz = sm_clock_mhz
        self.memory_clock_mhz = memory_clock_mhz
        self.pcie_throughput_mb = pcie_throughput_mb
        self.error_count = error_count
        self.timestamp = timestamp or time.time()

    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization percentage."""
        if self.memory_total_gb == 0:
            return 0.0
        return (self.memory_used_gb / self.memory_total_gb) * 100.0


class NodeMetrics:
    """Node-level metrics."""

    def __init__(
        self,
        node_id: str,
        gpu_metrics: Optional[List[GPUMetrics]] = None,
        cpu_utilization: float = 0.0,
        memory_used_gb: float = 0.0,
        memory_total_gb: float = 0.0,
        network_rx_mb: float = 0.0,
        network_tx_mb: float = 0.0,
        network_rx_bytes: int = 0,
        network_tx_bytes: int = 0,
        disk_io_mb: float = 0.0,
        timestamp: Optional[float] = None
    ):
        self.node_id = node_id
        self.gpu_metrics = gpu_metrics or []
        self.cpu_utilization = cpu_utilization
        self.memory_used_gb = memory_used_gb
        self.memory_total_gb = memory_total_gb
        self.network_rx_mb = network_rx_mb
        self.network_tx_mb = network_tx_mb
        self.network_rx_bytes = network_rx_bytes
        self.network_tx_bytes = network_tx_bytes
        self.disk_io_mb = disk_io_mb
        self.timestamp = timestamp or time.time()

    @property
    def avg_gpu_utilization(self) -> float:
        """Average GPU utilization across all GPUs."""
        if not self.gpu_metrics:
            return 0.0
        return sum(g.utilization for g in self.gpu_metrics) / len(self.gpu_metrics)

    @property
    def avg_gpu_temperature(self) -> float:
        """Average GPU temperature across all GPUs."""
        if not self.gpu_metrics:
            return 0.0
        return sum(g.temperature for g in self.gpu_metrics) / len(self.gpu_metrics)

    @property
    def avg_gpu_power(self) -> float:
        """Average GPU power usage across all GPUs."""
        if not self.gpu_metrics:
            return 0.0
        return sum(g.power_usage for g in self.gpu_metrics) / len(self.gpu_metrics)


class ClusterMetrics:
    """Cluster-level metrics."""

    def __init__(
        self,
        cluster_id: str,
        node_metrics: Optional[List[NodeMetrics]] = None,
        total_gpus: int = 0,
        available_gpus: int = 0,
        running_pods: int = 0,
        pending_pods: int = 0,
        total_jobs: int = 0,
        running_jobs: int = 0,
        pending_jobs: int = 0,
        failed_jobs: int = 0,
        timestamp: Optional[float] = None,
        avg_gpu_utilization: Optional[float] = None,
        gpu_availability: Optional[float] = None
    ):
        self.cluster_id = cluster_id
        self.node_metrics = node_metrics or []
        self.total_gpus = total_gpus
        self.available_gpus = available_gpus
        self.running_pods = running_pods
        self.pending_pods = pending_pods
        self.total_jobs = total_jobs
        self.running_jobs = running_jobs
        self.pending_jobs = pending_jobs
        self.failed_jobs = failed_jobs
        self.timestamp = timestamp or time.time()
        self._avg_gpu_utilization = avg_gpu_utilization
        self._gpu_availability = gpu_availability

    @property
    def gpu_availability(self) -> float:
        """GPU availability percentage."""
        if self._gpu_availability is not None:
            return self._gpu_availability
        if self.total_gpus == 0:
            return 0.0
        return (self.available_gpus / self.total_gpus) * 100.0

    @property
    def avg_gpu_utilization(self) -> float:
        """Average GPU utilization across all nodes."""
        if self._avg_gpu_utilization is not None:
            return self._avg_gpu_utilization
        all_gpus = []
        for node in self.node_metrics:
            all_gpus.extend(node.gpu_metrics)
        if not all_gpus:
            return 0.0
        return sum(g.utilization for g in all_gpus) / len(all_gpus)

    @avg_gpu_utilization.setter
    def avg_gpu_utilization(self, value: float):
        """Set average GPU utilization."""
        self._avg_gpu_utilization = value

    @property
    def avg_utilization(self) -> float:
        """Alias for avg_gpu_utilization."""
        return self.avg_gpu_utilization


@dataclass
class JobMetrics:
    """Job-level metrics."""
    job_id: str
    timestamp: float
    gpu_hours: float
    avg_gpu_utilization: float
    peak_gpu_memory_gb: float
    total_wait_time: float


@dataclass
class TenantMetrics:
    """Tenant-level metrics."""
    tenant_id: str
    timestamp: float
    gpu_quota: int
    gpu_used: int
    jobs_running: int
    jobs_pending: int
    gpu_hours_used: float


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Alert:
    """An alert for monitoring conditions."""

    def __init__(
        self,
        alert_id: str = "",
        level: AlertLevel = AlertLevel.INFO,
        message: str = "",
        timestamp: Optional[float] = None,
        resource_id: str = "",
        source: str = "",
        details: Optional[Dict[str, Any]] = None,
        metric_type: Optional[MetricType] = None,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
        resolved: bool = False,
        resolved_at: Optional[float] = None
    ):
        self.alert_id = alert_id
        self.level = level
        self.message = message
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.resource_id = resource_id
        self.source = source
        self.details = details or {}
        self.metric_type = metric_type
        self.value = value
        self.threshold = threshold
        self.resolved = resolved
        self.resolved_at = resolved_at


class AlertManager:
    """Manages alerts for the monitoring system."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: Dict[str, Alert] = {}
        self.alert_rules: List[dict] = []
        self._alert_counter = 0

    def create_alert(
        self,
        level: AlertLevel,
        message: str,
        resource_id: str = "",
        source: str = "",
        details: Optional[Dict[str, Any]] = None,
        metric_type: Optional[MetricType] = None,
        value: Optional[float] = None,
        threshold: Optional[float] = None
    ) -> str:
        """Create and store a new alert. Returns alert_id."""
        self._alert_counter += 1
        alert_id = f"alert-{self._alert_counter}"
        alert = Alert(
            alert_id=alert_id,
            level=level,
            message=message,
            timestamp=time.time(),
            resource_id=resource_id,
            source=source,
            details=details,
            metric_type=metric_type,
            value=value,
            threshold=threshold
        )
        self.alerts.append(alert)
        self.active_alerts[alert_id] = alert
        return alert_id

    def add_rule(
        self,
        name: str,
        metric_type: MetricType,
        threshold: float,
        level: AlertLevel,
        comparison: str = ">"
    ):
        """Add an alert rule."""
        self.alert_rules.append({
            "name": name,
            "metric_type": metric_type,
            "threshold": threshold,
            "level": level,
            "comparison": comparison
        })

    def check_metric(
        self,
        metric: MetricPoint,
        resource_id: str
    ) -> Optional[Alert]:
        """Check a metric against alert rules."""
        for rule in self.alert_rules:
            if rule["metric_type"] != metric.metric_type:
                continue

            triggered = False
            if rule["comparison"] == ">":
                triggered = metric.value > rule["threshold"]
            elif rule["comparison"] == "<":
                triggered = metric.value < rule["threshold"]
            elif rule["comparison"] == ">=":
                triggered = metric.value >= rule["threshold"]
            elif rule["comparison"] == "<=":
                triggered = metric.value <= rule["threshold"]

            if triggered:
                return self.create_alert(
                    level=rule["level"],
                    message=f"{rule['name']}: {metric.value} {rule['comparison']} {rule['threshold']}",
                    resource_id=resource_id,
                    metric_type=metric.metric_type,
                    value=metric.value,
                    threshold=rule["threshold"]
                )

        return None

    def check_alert_conditions(self, cluster_metrics) -> List[Alert]:
        """Check alert conditions against ClusterMetrics object."""
        alerts = []

        # Check high utilization
        util = getattr(cluster_metrics, 'avg_gpu_utilization', 0.0)
        if util > 90:
            alert = Alert(
                alert_id=f"alert-{self._alert_counter + 1}",
                level=AlertLevel.WARNING,
                message=f"High GPU utilization: {util:.1f}%"
            )
            self._alert_counter += 1
            self.alerts.append(alert)
            self.active_alerts[alert.alert_id] = alert
            alerts.append(alert)

        # Check low availability
        avail = getattr(cluster_metrics, 'gpu_availability', 100.0)
        if avail < 10:
            alert = Alert(
                alert_id=f"alert-{self._alert_counter + 1}",
                level=AlertLevel.WARNING,
                message=f"Low GPU availability: {avail:.1f}%"
            )
            self._alert_counter += 1
            self.alerts.append(alert)
            self.active_alerts[alert.alert_id] = alert
            alerts.append(alert)

        return alerts

    def check_conditions(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check alert conditions against metrics dict."""
        triggered = []
        for rule in self.alert_rules:
            metric_name = rule["metric_type"].value
            if metric_name in metrics:
                value = metrics[metric_name]
                threshold = rule["threshold"]

                check = False
                if rule["comparison"] == ">":
                    check = value > threshold
                elif rule["comparison"] == "<":
                    check = value < threshold
                elif rule["comparison"] == ">=":
                    check = value >= threshold
                elif rule["comparison"] == "<=":
                    check = value <= threshold

                if check:
                    alert = self.create_alert(
                        level=rule["level"],
                        message=f"{rule['name']}: {value} {rule['comparison']} {threshold}",
                        metric_type=rule["metric_type"],
                        value=value,
                        threshold=threshold
                    )
                    triggered.append(alert)

        return triggered

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved = True
            alert.resolved_at = time.time()
            self.resolved_alerts[alert_id] = alert
            return True

        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = time.time()
                self.resolved_alerts[alert_id] = alert
                return True
        return False

    def get_active_alerts(self, min_level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get all active (unresolved) alerts, optionally filtered by minimum level."""
        level_order = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.ERROR: 2,
            AlertLevel.CRITICAL: 3
        }
        active = [a for a in self.alerts if not a.resolved]
        if min_level is not None:
            min_order = level_order.get(min_level, 0)
            active = [a for a in active if level_order.get(a.level, 0) >= min_order]
        return active

    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by severity level."""
        return [a for a in self.alerts if a.level == level]


class MetricsAggregator:
    """Aggregates metrics over time windows."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def add_metric(self, key: str, metric: MetricPoint):
        """Add a metric to the aggregator."""
        self.metrics[key].append(metric)
        self.time_series[key].append((metric.timestamp, metric.value))
        self._cleanup(key)

    def add_value(self, key: str, value: float, timestamp: Optional[float] = None):
        """Add a raw value to the aggregator."""
        ts = timestamp or time.time()
        self.time_series[key].append((ts, value))

    def _cleanup(self, key: str):
        """Remove metrics older than window."""
        cutoff = time.time() - self.window_seconds
        self.metrics[key] = [
            m for m in self.metrics[key]
            if m.timestamp > cutoff
        ]

    def get_average(self, key: str) -> Optional[float]:
        """Get average value for a metric key."""
        metrics = self.metrics.get(key, [])
        if not metrics:
            # Try time_series
            series = self.time_series.get(key, [])
            if series:
                cutoff = time.time() - self.window_seconds
                values = [v for t, v in series if t > cutoff]
                if values:
                    return sum(values) / len(values)
            return None
        return sum(m.value for m in metrics) / len(metrics)

    def get_max(self, key: str) -> Optional[float]:
        """Get max value for a metric key."""
        metrics = self.metrics.get(key, [])
        if not metrics:
            series = self.time_series.get(key, [])
            if series:
                cutoff = time.time() - self.window_seconds
                values = [v for t, v in series if t > cutoff]
                if values:
                    return max(values)
            return None
        return max(m.value for m in metrics)

    def get_min(self, key: str) -> Optional[float]:
        """Get min value for a metric key."""
        metrics = self.metrics.get(key, [])
        if not metrics:
            series = self.time_series.get(key, [])
            if series:
                cutoff = time.time() - self.window_seconds
                values = [v for t, v in series if t > cutoff]
                if values:
                    return min(values)
            return None
        return min(m.value for m in metrics)

    def get_count(self, key: str) -> int:
        """Get count of metrics in window."""
        return len(self.metrics.get(key, []))

    def get_percentile(self, key: str, percentile: float) -> Optional[float]:
        """Get percentile value for a metric key."""
        series = self.time_series.get(key, [])
        if not series:
            return None

        cutoff = time.time() - self.window_seconds
        values = sorted([v for t, v in series if t > cutoff])
        if not values:
            return None

        idx = int(len(values) * percentile / 100)
        idx = min(idx, len(values) - 1)
        return values[idx]

    def aggregate(self, key: str) -> Dict[str, Optional[float]]:
        """Get all aggregated stats for a key."""
        return {
            "avg": self.get_average(key),
            "min": self.get_min(key),
            "max": self.get_max(key),
            "p50": self.get_percentile(key, 50),
            "p95": self.get_percentile(key, 95),
            "p99": self.get_percentile(key, 99),
            "count": self.get_count(key)
        }

    def calculate_percentiles(
        self,
        values_or_key,
        percentiles: List[float] = None
    ) -> Dict[str, Optional[float]]:
        """Calculate multiple percentiles for a key or list of values."""
        # Handle both old interface (key, percentiles) and new interface (values, percentiles=)
        if isinstance(values_or_key, list) and all(isinstance(v, (int, float)) for v in values_or_key):
            # New interface: values list as first argument
            values = sorted(values_or_key)
            if not values:
                return {}
            if percentiles is None:
                percentiles = [25, 50, 75, 90, 99]
            result = {}
            for p in percentiles:
                # Use linear interpolation for accurate percentile calculation
                n = len(values)
                k = (n - 1) * p / 100.0
                f = int(k)
                c = f + 1 if f < n - 1 else f
                if f == c:
                    result[f"p{int(p)}"] = float(values[f])
                else:
                    result[f"p{int(p)}"] = values[f] + (values[c] - values[f]) * (k - f)
            return result
        else:
            # Old interface: key as string
            key = str(values_or_key)
            if percentiles is None:
                percentiles = [25, 50, 75, 90, 99]
            return {f"p{int(p)}": self.get_percentile(key, p) for p in percentiles}

    def aggregate_over_time(
        self,
        metrics: List[Any],
        window_minutes: int = 5
    ) -> Dict[str, Any]:
        """Aggregate metrics over a time window."""
        if not metrics:
            return {"avg_gpu_utilization": {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}}

        # Filter by time window relative to latest timestamp (not current time)
        window_seconds = window_minutes * 60
        latest_ts = max(getattr(m, 'timestamp', 0) for m in metrics)
        recent = [m for m in metrics
                  if hasattr(m, 'timestamp') and (latest_ts - m.timestamp) <= window_seconds]

        if not recent:
            return {"avg_gpu_utilization": {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}}

        # Extract utilization values
        values = [getattr(m, 'avg_gpu_utilization', 0.0) for m in recent]

        return {
            "avg_gpu_utilization": {
                "mean": sum(values) / len(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "count": len(values)
            }
        }


class HealthChecker:
    """Checks health of cluster components."""

    def __init__(self, cluster: Optional[Cluster] = None):
        self.cluster = cluster
        self.health_status: Dict[str, dict] = {}
        self.last_check: Optional[float] = None

    def check_gpu_health(self, gpu: GPU) -> Dict[str, Any]:
        """Check health of a GPU."""
        issues = []

        if hasattr(gpu, 'temperature') and gpu.temperature > 85:
            issues.append(f"GPU {gpu.gpu_id} high temperature: {gpu.temperature}C")

        if gpu.available_memory_gb < 0.1 * gpu.total_memory_gb:
            issues.append(f"GPU {gpu.gpu_id} low memory")

        if hasattr(gpu, 'error_count') and gpu.error_count > 0:
            issues.append(f"GPU {gpu.gpu_id} has errors: {gpu.error_count}")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "gpu_id": gpu.gpu_id,
            "timestamp": time.time()
        }

    def check_node_health(self, node: Node) -> Dict[str, Any]:
        """Check health of a node."""
        issues = []
        gpu_health = []

        # Check node Ready condition
        if not node.conditions.get("Ready", True):
            issues.append(f"Node {node.node_id} is not ready")

        for gpu in node.gpus:
            health = self.check_gpu_health(gpu)
            gpu_health.append(health)
            issues.extend(health["issues"])

        status = {
            "healthy": len(issues) == 0,
            "issues": issues,
            "timestamp": time.time(),
            "node_id": node.node_id,
            "gpu_count": len(node.gpus),
            "gpu_available": sum(1 for g in node.gpus if not g.allocated_jobs),
            "gpu_health": gpu_health
        }

        self.health_status[node.node_id] = status
        return status

    def check_cluster_health(self, cluster: Optional[Cluster] = None) -> Dict[str, Any]:
        """Check overall cluster health."""
        target_cluster = cluster if cluster is not None else self.cluster
        if target_cluster is None:
            return {"healthy": False, "issues": ["No cluster configured"]}

        node_health = {}
        total_issues = []
        healthy_nodes = 0
        total_gpus = 0
        healthy_gpus = 0

        for node_id, node in target_cluster.nodes.items():
            health = self.check_node_health(node)
            node_health[node_id] = health
            total_issues.extend(health["issues"])

            if health["healthy"]:
                healthy_nodes += 1

            for gpu_health in health.get("gpu_health", []):
                total_gpus += 1
                if gpu_health.get("healthy", False):
                    healthy_gpus += 1

        self.last_check = time.time()

        total_nodes = len(target_cluster.nodes)
        if healthy_nodes == total_nodes and total_nodes > 0:
            overall_health = "healthy"
        elif healthy_nodes == 0:
            overall_health = "critical"
        else:
            overall_health = "degraded"

        return {
            "healthy": len(total_issues) == 0,
            "total_issues": len(total_issues),
            "issues": total_issues,
            "nodes": node_health,
            "timestamp": self.last_check,
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
            "total_gpus": total_gpus,
            "healthy_gpus": healthy_gpus,
            "overall_health": overall_health
        }

    def get_status(self, node_id: Optional[str] = None) -> Dict:
        """Get health status."""
        if node_id:
            return self.health_status.get(node_id, {})
        return self.health_status


class MetricsCollector:
    """Collects metrics from the cluster."""

    def __init__(
        self,
        cluster: Cluster,
        collection_interval: float = 60.0,
        max_history: int = 1000
    ):
        self.cluster = cluster
        self.collection_interval = collection_interval
        self.max_history = max_history
        self.metrics_history: List[ClusterMetrics] = []
        self.aggregator = MetricsAggregator()
        self.last_collection: Optional[float] = None

    def collect_gpu_metrics(self, gpu: GPU) -> GPUMetrics:
        """Collect metrics from a single GPU."""
        return GPUMetrics(
            gpu_id=gpu.gpu_id,
            utilization=getattr(gpu, 'utilization', 0.0),
            memory_used_gb=gpu.total_memory_gb - gpu.available_memory_gb,
            memory_total_gb=gpu.total_memory_gb,
            temperature=getattr(gpu, 'temperature', 0.0),
            power_usage=getattr(gpu, 'power_usage', 0.0)
        )

    def collect_node_metrics(self, node: Node) -> NodeMetrics:
        """Collect metrics from a single node."""
        gpu_metrics = [self.collect_gpu_metrics(gpu) for gpu in node.gpus]

        return NodeMetrics(
            node_id=node.node_id,
            gpu_metrics=gpu_metrics,
            cpu_utilization=getattr(node, 'cpu_utilization', 0.0),
            memory_used_gb=getattr(node, 'memory_used_gb', 0.0),
            memory_total_gb=getattr(node, 'memory_total_gb', 0.0)
        )

    def collect_cluster_metrics(self) -> ClusterMetrics:
        """Collect metrics from the entire cluster."""
        node_metrics = [
            self.collect_node_metrics(node)
            for node in self.cluster.nodes.values()
        ]

        total_gpus = sum(len(node.gpus) for node in self.cluster.nodes.values())
        available_gpus = sum(
            sum(1 for gpu in node.gpus if not gpu.allocated_jobs)
            for node in self.cluster.nodes.values()
        )

        # Count jobs
        total_jobs = len(self.cluster.jobs)
        running_jobs = sum(1 for j in self.cluster.jobs.values() if j.state == JobState.RUNNING)
        pending_jobs = sum(1 for j in self.cluster.jobs.values() if j.state == JobState.PENDING)

        return ClusterMetrics(
            cluster_id=self.cluster.cluster_id,
            node_metrics=node_metrics,
            total_gpus=total_gpus,
            available_gpus=available_gpus,
            total_jobs=total_jobs,
            running_jobs=running_jobs,
            pending_jobs=pending_jobs
        )

    def collect_metrics(self) -> ClusterMetrics:
        """Collect all metrics and store in history."""
        metrics = self.collect_cluster_metrics()
        self.metrics_history.append(metrics)
        self.last_collection = time.time()

        # Enforce max_history limit
        while len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)

        # Update aggregator
        self.aggregator.add_value("gpu_utilization", metrics.avg_gpu_utilization)
        self.aggregator.add_value("gpu_availability", metrics.gpu_availability)

        return metrics

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]

        return {
            "cluster_id": latest.cluster_id,
            "total_gpus": latest.total_gpus,
            "available_gpus": latest.available_gpus,
            "avg_gpu_utilization": latest.avg_gpu_utilization,
            "gpu_utilization": latest.avg_gpu_utilization,
            "total_jobs": latest.total_jobs,
            "running_jobs": latest.running_jobs,
            "pending_jobs": latest.pending_jobs,
            "collection_count": len(self.metrics_history),
            "last_collection": self.last_collection,
            "timestamp": latest.timestamp
        }


class QuotaManager:
    """Manages tenant GPU quotas."""

    def __init__(self, cluster: Cluster, allocator: HybridAllocator):
        self.cluster = cluster
        self.allocator = allocator
        self.tenant_usage: Dict[str, TenantMetrics] = {}

    def get_tenant_usage(self, tenant_id: str) -> TenantMetrics:
        """Get current GPU usage for a tenant."""
        tenant = self.cluster.tenants.get(tenant_id)
        if not tenant:
            return TenantMetrics(
                tenant_id=tenant_id,
                timestamp=time.time(),
                gpu_quota=0,
                gpu_used=0,
                jobs_running=0,
                jobs_pending=0,
                gpu_hours_used=0.0
            )

        # Count running jobs for tenant
        running = sum(
            1 for j in self.cluster.jobs.values()
            if j.tenant_id == tenant_id and j.state == JobState.RUNNING
        )
        pending = sum(
            1 for j in self.cluster.jobs.values()
            if j.tenant_id == tenant_id and j.state == JobState.PENDING
        )

        # Count allocated GPUs
        allocated = sum(
            alloc.gpu_fraction
            for alloc in self.allocator.get_all_allocations()
            if hasattr(alloc, 'tenant_id') and alloc.tenant_id == tenant_id
        )

        return TenantMetrics(
            tenant_id=tenant_id,
            timestamp=time.time(),
            gpu_quota=tenant.gpu_quota,
            gpu_used=int(allocated),
            jobs_running=running,
            jobs_pending=pending,
            gpu_hours_used=0.0  # Would need tracking
        )

    def check_quota(self, tenant_id: str, requested_gpus: int) -> bool:
        """Check if tenant has quota for requested GPUs."""
        usage = self.get_tenant_usage(tenant_id)
        return usage.gpu_used + requested_gpus <= usage.gpu_quota

    def get_all_tenant_usage(self) -> Dict[str, TenantMetrics]:
        """Get usage for all tenants."""
        return {
            tenant_id: self.get_tenant_usage(tenant_id)
            for tenant_id in self.cluster.tenants
        }


class PreemptionManager:
    """Manages job preemption."""

    def __init__(self, cluster: Cluster, allocator: HybridAllocator):
        self.cluster = cluster
        self.allocator = allocator
        self.preemption_history: List[Dict] = []

    def find_preemption_candidates(
        self,
        required_gpus: int,
        priority: PriorityClass
    ) -> List[Job]:
        """Find jobs that can be preempted."""
        candidates = []

        for job in self.cluster.jobs.values():
            if job.state != JobState.RUNNING:
                continue

            # Can only preempt lower priority jobs
            if job.priority.value < priority.value:
                candidates.append(job)

        # Sort by priority (lowest first) then by start time (oldest first)
        candidates.sort(key=lambda j: (j.priority.value, j.start_time or 0))

        return candidates

    def preempt_job(self, job: Job) -> bool:
        """Preempt a running job."""
        if job.state != JobState.RUNNING:
            return False

        # Release allocations
        for pod in job.pods:
            self.allocator.release(pod.pod_id)

        job.state = JobState.PREEMPTED

        self.preemption_history.append({
            "job_id": job.job_id,
            "timestamp": time.time(),
            "priority": job.priority.value
        })

        return True

    def get_preemption_stats(self) -> Dict[str, Any]:
        """Get preemption statistics."""
        return {
            "total_preemptions": len(self.preemption_history),
            "recent": self.preemption_history[-10:]
        }


class ClusterMonitor:
    """Main cluster monitoring class."""

    def __init__(
        self,
        cluster: Cluster,
        allocator: Optional[HybridAllocator] = None,
        collection_interval: float = 60.0
    ):
        self.cluster = cluster
        self.allocator = allocator
        self.collection_interval = collection_interval

        self.collector = MetricsCollector(cluster, collection_interval)
        self.health_checker = HealthChecker(cluster)
        self.alert_manager = AlertManager()

        if allocator:
            self.quota = QuotaManager(cluster, allocator)
            self.preemption = PreemptionManager(cluster, allocator)
        else:
            self.quota = None
            self.preemption = None

        self.alerts: List[Dict] = []
        self._running = False

    def collect_all_metrics(self) -> ClusterMetrics:
        """Collect all cluster metrics."""
        return self.collector.collect_metrics()

    def check_alerts(self) -> List[Dict]:
        """Check for alert conditions."""
        new_alerts = []

        # Check GPU utilization
        metrics = self.collector.collect_cluster_metrics()

        if metrics.avg_gpu_utilization > 90:
            new_alerts.append({
                "level": "warning",
                "message": f"High GPU utilization: {metrics.avg_gpu_utilization:.1f}%",
                "timestamp": time.time()
            })

        if metrics.available_gpus == 0:
            new_alerts.append({
                "level": "critical",
                "message": "No GPUs available",
                "timestamp": time.time()
            })

        self.alerts.extend(new_alerts)
        return new_alerts

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        cluster_metrics = self.collector.collect_cluster_metrics()

        tenant_usage = {}
        if self.quota:
            tenant_usage = {
                tid: {
                    "gpu_used": m.gpu_used,
                    "gpu_quota": m.gpu_quota,
                    "jobs_running": m.jobs_running
                }
                for tid, m in self.quota.get_all_tenant_usage().items()
            }

        queue_status = {}
        for qid, queue in self.cluster.queues.items():
            queue_status[qid] = {
                "pending": len([j for j in queue.jobs if j.state == JobState.PENDING]),
                "running": len([j for j in queue.jobs if j.state == JobState.RUNNING])
            }

        return {
            "cluster": {
                "total_gpus": cluster_metrics.total_gpus,
                "available_gpus": cluster_metrics.available_gpus,
                "utilization": cluster_metrics.avg_utilization,
                "total_jobs": cluster_metrics.total_jobs,
                "running_jobs": cluster_metrics.running_jobs,
                "pending_jobs": cluster_metrics.pending_jobs
            },
            "tenants": tenant_usage,
            "queues": queue_status,
            "preemptions": self.preemption.get_preemption_stats() if self.preemption else {},
            "alerts": self.alerts[-20:]
        }

    def start(self):
        """Start monitoring."""
        self._running = True

    def start_monitoring(self, interval: float = 60.0):
        """Start monitoring with specified interval."""
        self.collection_interval = interval
        self.monitoring_interval = interval
        self._running = True
        self._monitoring_loop()

    def stop(self):
        """Stop monitoring."""
        self._running = False

    @property
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._running

    def _monitoring_loop(self):
        """Internal monitoring loop."""
        while self._running:
            self.run_monitoring_cycle()
            time.sleep(self.collection_interval)

    def get_status(self) -> Dict[str, Any]:
        """Get monitor status."""
        # Get cluster health
        cluster_health = self.health_checker.check_cluster_health(self.cluster)

        # Get current metrics summary
        current_metrics = self.collector.get_metrics_summary()

        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            "running": self._running,
            "is_monitoring": self._running,
            "cluster_id": self.cluster.cluster_id,
            "collection_interval": self.collection_interval,
            "metrics_count": len(self.collector.metrics_history),
            "alert_count": len(self.alerts),
            "cluster_health": cluster_health.get("overall_health", "unknown"),
            "current_metrics": current_metrics,
            "active_alerts": [{"level": a.level.value, "message": a.message} for a in active_alerts]
        }

    def get_gpu_stats(self) -> List[Dict[str, Any]]:
        """Get GPU statistics as a list of per-GPU stats."""
        stats = []
        for node in self.cluster.nodes.values():
            for gpu in node.gpus:
                stats.append({
                    "gpu_id": gpu.gpu_id,
                    "node_id": node.node_id,
                    "gpu_type": gpu.gpu_type.value if hasattr(gpu.gpu_type, 'value') else str(gpu.gpu_type),
                    "total_memory_gb": gpu.total_memory_gb,
                    "available_memory_gb": gpu.available_memory_gb,
                    "utilization": getattr(gpu, 'utilization', 0.0),
                    "temperature": getattr(gpu, 'temperature', 0.0),
                    "power_usage": getattr(gpu, 'power_usage', 0.0),
                    "allocated": len(gpu.allocated_jobs) > 0
                })
        return stats

    def export_metrics(self, format: str = "json") -> Any:
        """Export metrics in specified format."""
        cluster_metrics = self.collector.collect_cluster_metrics()

        if format == "json":
            return {
                "timestamp": time.time(),
                "cluster_id": cluster_metrics.cluster_id,
                "metrics": {
                    "gpu_utilization": cluster_metrics.avg_gpu_utilization,
                    "total_gpus": cluster_metrics.total_gpus,
                    "available_gpus": cluster_metrics.available_gpus
                },
                "gpu_utilization": cluster_metrics.avg_gpu_utilization,
                "total_gpus": cluster_metrics.total_gpus,
                "available_gpus": cluster_metrics.available_gpus,
                "nodes": [
                    {
                        "node_id": nm.node_id,
                        "gpu_count": len(nm.gpu_metrics),
                        "avg_utilization": nm.avg_gpu_utilization
                    }
                    for nm in cluster_metrics.node_metrics
                ]
            }
        elif format == "prometheus":
            lines = [
                f"# HELP gpu_utilization GPU utilization percentage",
                f"# TYPE gpu_utilization gauge",
                f'gpu_utilization{{cluster="{cluster_metrics.cluster_id}"}} {cluster_metrics.avg_gpu_utilization}',
                f"# HELP gpu_total Total GPUs",
                f"# TYPE gpu_total gauge",
                f'gpu_total{{cluster="{cluster_metrics.cluster_id}"}} {cluster_metrics.total_gpus}',
            ]
            return "\n".join(lines)

        return None

    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run one monitoring cycle."""
        self.collect_all_metrics()
        new_alerts = self.check_alerts()
        dashboard = self.get_dashboard_data()
        dashboard["new_alerts"] = new_alerts
        return dashboard


# Alias for backward compatibility
GPUMonitor = ClusterMonitor
