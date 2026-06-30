"""Enterprise features: schema registry, multi-region, monitoring."""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# --- Schema Registry ---

@dataclass
class SchemaVersion:
    """Schema version info."""

    schema_id: int
    version: int
    schema: str
    schema_type: str = "AVRO"


class SchemaRegistry:
    """Client for interacting with Schema Registry."""

    def __init__(self, url: str, auth: Optional[tuple] = None):
        """
        Initialize Schema Registry client.

        Args:
            url: Schema Registry URL
            auth: Optional (username, password) tuple
        """
        self.url = url.rstrip("/")
        self.auth = auth
        self._cache: Dict[str, SchemaVersion] = {}

    def register_schema(
        self,
        subject: str,
        schema: str,
        schema_type: str = "AVRO",
    ) -> int:
        """
        Register a new schema version.

        Args:
            subject: Subject name (typically topic-value or topic-key)
            schema: Schema string
            schema_type: Schema type (AVRO, JSON, PROTOBUF)

        Returns:
            Schema ID
        """
        # In production, would make HTTP POST to registry
        # Simulated response
        schema_id = hash(schema) % 100000

        self._cache[subject] = SchemaVersion(
            schema_id=schema_id,
            version=len(self._cache) + 1,
            schema=schema,
            schema_type=schema_type,
        )

        logger.info(f"Registered schema for {subject} with ID {schema_id}")
        return schema_id

    def get_schema(self, schema_id: int) -> Optional[str]:
        """Get schema by ID."""
        for version in self._cache.values():
            if version.schema_id == schema_id:
                return version.schema
        return None

    def get_latest_schema(self, subject: str) -> Optional[SchemaVersion]:
        """Get latest schema for subject."""
        return self._cache.get(subject)

    def check_compatibility(
        self,
        subject: str,
        schema: str,
        compatibility: str = "BACKWARD",
    ) -> bool:
        """
        Check if schema is compatible.

        Args:
            subject: Subject name
            schema: New schema to check
            compatibility: Compatibility level

        Returns:
            True if compatible
        """
        # In production, would make HTTP POST to registry
        # Simplified compatibility check
        current = self._cache.get(subject)
        if not current:
            return True

        # Real implementation would use Avro schema resolution
        return True

    def set_compatibility(
        self,
        subject: str,
        level: str,
    ) -> None:
        """Set compatibility level for subject."""
        logger.info(f"Set compatibility for {subject} to {level}")

    def get_subjects(self) -> List[str]:
        """List all subjects."""
        return list(self._cache.keys())


# --- Multi-Region Replication ---

@dataclass
class ClusterConfig:
    """Configuration for a Kafka cluster."""

    name: str
    bootstrap_servers: str
    region: str
    config: Dict[str, str] = field(default_factory=dict)


@dataclass
class ReplicationFlow:
    """Replication flow between clusters."""

    source_cluster: str
    target_cluster: str
    topics: str = ".*"  # Topic pattern
    enabled: bool = True
    config: Dict[str, str] = field(default_factory=dict)


class MirrorMaker:
    """Configure and manage MirrorMaker 2 replication."""

    def __init__(self):
        self._clusters: Dict[str, ClusterConfig] = {}
        self._flows: List[ReplicationFlow] = []

    def add_cluster(self, config: ClusterConfig) -> None:
        """Add a cluster configuration."""
        self._clusters[config.name] = config
        logger.info(f"Added cluster {config.name} in {config.region}")

    def add_replication_flow(self, flow: ReplicationFlow) -> None:
        """Add a replication flow."""
        if flow.source_cluster not in self._clusters:
            raise ValueError(f"Source cluster {flow.source_cluster} not found")
        if flow.target_cluster not in self._clusters:
            raise ValueError(f"Target cluster {flow.target_cluster} not found")

        self._flows.append(flow)
        logger.info(
            f"Added replication {flow.source_cluster} -> {flow.target_cluster}"
        )

    def generate_config(self) -> str:
        """Generate MirrorMaker 2 configuration file."""
        lines = []

        # Clusters
        cluster_names = ",".join(self._clusters.keys())
        lines.append(f"clusters = {cluster_names}")
        lines.append("")

        for name, cluster in self._clusters.items():
            lines.append(f"{name}.bootstrap.servers = {cluster.bootstrap_servers}")
            for key, value in cluster.config.items():
                lines.append(f"{name}.{key} = {value}")
            lines.append("")

        # Replication flows
        for flow in self._flows:
            prefix = f"{flow.source_cluster}->{flow.target_cluster}"
            lines.append(f"{prefix}.enabled = {str(flow.enabled).lower()}")
            lines.append(f"{prefix}.topics = {flow.topics}")

            for key, value in flow.config.items():
                lines.append(f"{prefix}.{key} = {value}")

            lines.append("")

        return "\n".join(lines)

    def get_lag_metrics(self) -> Dict[str, int]:
        """Get replication lag metrics."""
        # In production, would query Kafka metrics
        return {
            f"{flow.source_cluster}->{flow.target_cluster}": 0
            for flow in self._flows
        }


# --- Log Compaction ---

@dataclass
class CompactedTopicConfig:
    """Configuration for a compacted topic."""

    name: str
    partitions: int = 3
    replication_factor: int = 3
    min_cleanable_dirty_ratio: float = 0.1
    delete_retention_ms: int = 86400000  # 1 day
    segment_ms: int = 604800000  # 7 days
    min_compaction_lag_ms: int = 0


class CompactedTopicManager:
    """Manage compacted topics for materialized views."""

    def __init__(self, admin_client: Any):
        self._admin = admin_client
        self._topics: Dict[str, CompactedTopicConfig] = {}

    def create_compacted_topic(self, config: CompactedTopicConfig) -> None:
        """Create a compacted topic."""
        topic_config = {
            "cleanup.policy": "compact",
            "min.cleanable.dirty.ratio": str(config.min_cleanable_dirty_ratio),
            "delete.retention.ms": str(config.delete_retention_ms),
            "segment.ms": str(config.segment_ms),
            "min.compaction.lag.ms": str(config.min_compaction_lag_ms),
        }

        # In production, would create topic via admin client
        self._topics[config.name] = config
        logger.info(f"Created compacted topic {config.name}")

    def tombstone(self, topic: str, key: str) -> None:
        """Send tombstone to delete key from compacted topic."""
        # In production, would produce null value to topic
        logger.info(f"Tombstone sent for key {key} in {topic}")


# --- Monitoring ---

@dataclass
class ProducerMetrics:
    """Producer metrics."""

    messages_sent: int = 0
    bytes_sent: int = 0
    errors: int = 0
    avg_latency_ms: float = 0
    p99_latency_ms: float = 0


@dataclass
class ConsumerMetrics:
    """Consumer metrics."""

    messages_consumed: int = 0
    bytes_consumed: int = 0
    lag: int = 0
    partitions_assigned: int = 0


@dataclass
class JobMetrics:
    """Flink job metrics."""

    records_in_per_second: float = 0
    records_out_per_second: float = 0
    bytes_in_per_second: float = 0
    bytes_out_per_second: float = 0
    checkpoint_duration_ms: int = 0
    checkpoint_size_bytes: int = 0
    backpressure_ratio: float = 0


class MetricsCollector:
    """Collect and expose streaming metrics."""

    def __init__(self):
        self._producer_metrics: Dict[str, ProducerMetrics] = {}
        self._consumer_metrics: Dict[str, ConsumerMetrics] = {}
        self._job_metrics: Dict[str, JobMetrics] = {}

    def record_producer_metrics(
        self,
        producer_id: str,
        metrics: ProducerMetrics,
    ) -> None:
        """Record producer metrics."""
        self._producer_metrics[producer_id] = metrics

    def record_consumer_metrics(
        self,
        consumer_id: str,
        metrics: ConsumerMetrics,
    ) -> None:
        """Record consumer metrics."""
        self._consumer_metrics[consumer_id] = metrics

    def record_job_metrics(
        self,
        job_id: str,
        metrics: JobMetrics,
    ) -> None:
        """Record job metrics."""
        self._job_metrics[job_id] = metrics

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "producers": {
                k: v.__dict__ for k, v in self._producer_metrics.items()
            },
            "consumers": {
                k: v.__dict__ for k, v in self._consumer_metrics.items()
            },
            "jobs": {
                k: v.__dict__ for k, v in self._job_metrics.items()
            },
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for producer_id, metrics in self._producer_metrics.items():
            lines.append(
                f'streaming_producer_messages_total{{id="{producer_id}"}} {metrics.messages_sent}'
            )
            lines.append(
                f'streaming_producer_errors_total{{id="{producer_id}"}} {metrics.errors}'
            )

        for consumer_id, metrics in self._consumer_metrics.items():
            lines.append(
                f'streaming_consumer_lag{{id="{consumer_id}"}} {metrics.lag}'
            )
            lines.append(
                f'streaming_consumer_messages_total{{id="{consumer_id}"}} {metrics.messages_consumed}'
            )

        for job_id, metrics in self._job_metrics.items():
            lines.append(
                f'streaming_job_records_in_per_second{{id="{job_id}"}} {metrics.records_in_per_second}'
            )
            lines.append(
                f'streaming_job_backpressure{{id="{job_id}"}} {metrics.backpressure_ratio}'
            )

        return "\n".join(lines)


# --- Alerting ---

@dataclass
class Alert:
    """Alert definition."""

    name: str
    condition: str
    threshold: float
    severity: str = "warning"  # info, warning, error, critical
    message: str = ""


class AlertManager:
    """Manage alerts for streaming platform."""

    def __init__(self):
        self._alerts: List[Alert] = []
        self._triggered: List[tuple] = []  # (alert, timestamp, value)

    def register_alert(self, alert: Alert) -> None:
        """Register an alert."""
        self._alerts.append(alert)
        logger.info(f"Registered alert: {alert.name}")

    def check_alerts(self, metrics: Dict[str, Any]) -> List[tuple]:
        """
        Check all alerts against current metrics.

        Args:
            metrics: Current metrics

        Returns:
            List of triggered alerts
        """
        triggered = []

        for alert in self._alerts:
            try:
                # Extract metric value (simplified)
                value = self._extract_metric(metrics, alert.condition)

                if value is not None and value > alert.threshold:
                    triggered.append((alert, time.time(), value))
                    self._triggered.append((alert, time.time(), value))
                    logger.warning(
                        f"Alert triggered: {alert.name} "
                        f"(value={value}, threshold={alert.threshold})"
                    )
            except Exception as e:
                logger.error(f"Error checking alert {alert.name}: {e}")

        return triggered

    def _extract_metric(
        self,
        metrics: Dict[str, Any],
        condition: str,
    ) -> Optional[float]:
        """Extract metric value from metrics dict."""
        # Simple dot notation extraction
        parts = condition.split(".")
        value = metrics

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

        return float(value) if value is not None else None

    def get_triggered_alerts(
        self,
        since: Optional[float] = None,
    ) -> List[tuple]:
        """Get triggered alerts since timestamp."""
        if since is None:
            return self._triggered

        return [(a, t, v) for a, t, v in self._triggered if t >= since]


# Common alerts
CONSUMER_LAG_ALERT = Alert(
    name="high_consumer_lag",
    condition="consumers.lag",
    threshold=10000,
    severity="warning",
    message="Consumer lag exceeds 10,000 messages",
)

BACKPRESSURE_ALERT = Alert(
    name="high_backpressure",
    condition="jobs.backpressure_ratio",
    threshold=0.5,
    severity="error",
    message="Job backpressure exceeds 50%",
)

ERROR_RATE_ALERT = Alert(
    name="high_error_rate",
    condition="producers.errors",
    threshold=100,
    severity="critical",
    message="Producer errors exceed 100",
)


# --- Autoscaling ---

@dataclass
class ScalingPolicy:
    """Autoscaling policy configuration."""

    name: str
    metric: str  # Metric to monitor
    target_value: float  # Target metric value
    min_replicas: int = 1
    max_replicas: int = 10
    scale_up_cooldown_seconds: int = 60
    scale_down_cooldown_seconds: int = 300
    scale_up_step: int = 2  # How many replicas to add
    scale_down_step: int = 1  # How many replicas to remove


@dataclass
class ScalingDecision:
    """Result of a scaling evaluation."""

    should_scale: bool
    direction: str  # "up", "down", or "none"
    current_replicas: int
    target_replicas: int
    reason: str


class FlinkAutoscaler:
    """
    Autoscaler for Flink streaming jobs.

    Monitors metrics and adjusts parallelism based on scaling policies.
    Supports:
    - Backpressure-based scaling
    - Lag-based scaling
    - CPU/memory-based scaling
    - Custom metric-based scaling
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        k8s_client: Any = None,  # kubernetes.client or mock
    ):
        """
        Initialize the autoscaler.

        Args:
            metrics_collector: Metrics source
            k8s_client: Kubernetes API client (optional, for K8s deployments)
        """
        self._metrics_collector = metrics_collector or MetricsCollector()
        self._k8s_client = k8s_client
        self._policies: Dict[str, ScalingPolicy] = {}
        self._current_replicas: Dict[str, int] = {}
        self._last_scale_time: Dict[str, float] = {}
        self._scaling_history: List[tuple] = []  # (job_id, time, old, new, reason)

    def register_policy(self, job_id: str, policy: ScalingPolicy) -> None:
        """
        Register a scaling policy for a job.

        Args:
            job_id: Flink job ID
            policy: Scaling policy
        """
        self._policies[job_id] = policy
        self._current_replicas[job_id] = policy.min_replicas
        logger.info(f"Registered autoscaling policy for job {job_id}: {policy.name}")

    def unregister_policy(self, job_id: str) -> None:
        """Remove scaling policy for a job."""
        self._policies.pop(job_id, None)
        self._current_replicas.pop(job_id, None)
        self._last_scale_time.pop(job_id, None)

    def evaluate_scaling(self, job_id: str) -> ScalingDecision:
        """
        Evaluate if a job should be scaled.

        Args:
            job_id: Flink job ID

        Returns:
            ScalingDecision with recommendation
        """
        policy = self._policies.get(job_id)
        if not policy:
            return ScalingDecision(
                should_scale=False,
                direction="none",
                current_replicas=0,
                target_replicas=0,
                reason="No policy registered",
            )

        current = self._current_replicas.get(job_id, policy.min_replicas)
        metrics = self._get_job_metrics(job_id)

        if not metrics:
            return ScalingDecision(
                should_scale=False,
                direction="none",
                current_replicas=current,
                target_replicas=current,
                reason="No metrics available",
            )

        # Get the metric value based on policy
        metric_value = self._extract_metric_value(metrics, policy.metric)
        if metric_value is None:
            return ScalingDecision(
                should_scale=False,
                direction="none",
                current_replicas=current,
                target_replicas=current,
                reason=f"Metric {policy.metric} not found",
            )

        # Check cooldown
        last_scale = self._last_scale_time.get(job_id, 0)
        time_since_scale = time.time() - last_scale

        # Determine scaling direction
        if metric_value > policy.target_value * 1.2:  # 20% above target
            if time_since_scale < policy.scale_up_cooldown_seconds:
                return ScalingDecision(
                    should_scale=False,
                    direction="none",
                    current_replicas=current,
                    target_replicas=current,
                    reason=f"In scale-up cooldown ({int(policy.scale_up_cooldown_seconds - time_since_scale)}s remaining)",
                )

            target = min(current + policy.scale_up_step, policy.max_replicas)
            if target > current:
                return ScalingDecision(
                    should_scale=True,
                    direction="up",
                    current_replicas=current,
                    target_replicas=target,
                    reason=f"Metric {policy.metric}={metric_value:.2f} > target {policy.target_value}",
                )

        elif metric_value < policy.target_value * 0.5:  # 50% below target
            if time_since_scale < policy.scale_down_cooldown_seconds:
                return ScalingDecision(
                    should_scale=False,
                    direction="none",
                    current_replicas=current,
                    target_replicas=current,
                    reason=f"In scale-down cooldown ({int(policy.scale_down_cooldown_seconds - time_since_scale)}s remaining)",
                )

            target = max(current - policy.scale_down_step, policy.min_replicas)
            if target < current:
                return ScalingDecision(
                    should_scale=True,
                    direction="down",
                    current_replicas=current,
                    target_replicas=target,
                    reason=f"Metric {policy.metric}={metric_value:.2f} < target {policy.target_value}",
                )

        return ScalingDecision(
            should_scale=False,
            direction="none",
            current_replicas=current,
            target_replicas=current,
            reason=f"Metric {policy.metric}={metric_value:.2f} within acceptable range",
        )

    def apply_scaling(self, job_id: str, decision: ScalingDecision) -> bool:
        """
        Apply a scaling decision.

        Args:
            job_id: Flink job ID
            decision: Scaling decision to apply

        Returns:
            True if scaling was successful
        """
        if not decision.should_scale:
            return False

        old_replicas = decision.current_replicas
        new_replicas = decision.target_replicas

        # In production, would trigger savepoint and restart job
        # with new parallelism, or use Kubernetes HPA/operator
        success = self._scale_job(job_id, new_replicas)

        if success:
            self._current_replicas[job_id] = new_replicas
            self._last_scale_time[job_id] = time.time()
            self._scaling_history.append(
                (job_id, time.time(), old_replicas, new_replicas, decision.reason)
            )

            logger.info(
                f"Scaled job {job_id}: {old_replicas} -> {new_replicas} "
                f"(reason: {decision.reason})"
            )

        return success

    def run_autoscaling_cycle(self) -> Dict[str, ScalingDecision]:
        """
        Run one autoscaling evaluation cycle for all registered jobs.

        Returns:
            Dictionary of job_id -> ScalingDecision
        """
        decisions = {}

        for job_id in self._policies:
            decision = self.evaluate_scaling(job_id)
            decisions[job_id] = decision

            if decision.should_scale:
                self.apply_scaling(job_id, decision)

        return decisions

    def get_scaling_history(
        self,
        job_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[tuple]:
        """
        Get scaling history.

        Args:
            job_id: Optional filter by job ID
            limit: Maximum entries to return

        Returns:
            List of (job_id, timestamp, old_replicas, new_replicas, reason)
        """
        history = self._scaling_history
        if job_id:
            history = [h for h in history if h[0] == job_id]
        return history[-limit:]

    def get_current_replicas(self, job_id: str) -> int:
        """Get current replica count for a job."""
        return self._current_replicas.get(job_id, 0)

    def _get_job_metrics(self, job_id: str) -> Optional[JobMetrics]:
        """Get metrics for a specific job."""
        all_metrics = self._metrics_collector.get_all_metrics()
        jobs = all_metrics.get("jobs", {})
        job_data = jobs.get(job_id)

        if job_data:
            return JobMetrics(**job_data)
        return None

    def _extract_metric_value(
        self,
        metrics: JobMetrics,
        metric_name: str,
    ) -> Optional[float]:
        """Extract a metric value from JobMetrics."""
        metric_map = {
            "backpressure": metrics.backpressure_ratio,
            "backpressure_ratio": metrics.backpressure_ratio,
            "records_in": metrics.records_in_per_second,
            "records_in_per_second": metrics.records_in_per_second,
            "records_out": metrics.records_out_per_second,
            "records_out_per_second": metrics.records_out_per_second,
            "bytes_in": metrics.bytes_in_per_second,
            "bytes_out": metrics.bytes_out_per_second,
            "checkpoint_duration": metrics.checkpoint_duration_ms,
            "checkpoint_duration_ms": metrics.checkpoint_duration_ms,
        }
        return metric_map.get(metric_name)

    def _scale_job(self, job_id: str, replicas: int) -> bool:
        """
        Scale a Flink job to the specified parallelism.

        In production, this would:
        1. Trigger a savepoint
        2. Cancel the job
        3. Restart with new parallelism

        For Kubernetes deployments, could use:
        - Flink Kubernetes Operator
        - Kubernetes HPA
        - Custom deployment controller
        """
        if self._k8s_client:
            return self._scale_with_k8s(job_id, replicas)

        # Simulated scaling for non-K8s environments
        logger.info(f"Simulated scaling for job {job_id} to {replicas} replicas")
        return True

    def _scale_with_k8s(self, job_id: str, replicas: int) -> bool:
        """Scale using Kubernetes API."""
        try:
            # Would patch the FlinkDeployment or Deployment resource
            # Example: kubectl patch flinkdeployment {job_id} -p '{"spec":{"job":{"parallelism": N}}}'
            logger.info(f"K8s scaling for job {job_id} to {replicas}")
            return True
        except Exception as e:
            logger.error(f"K8s scaling failed for job {job_id}: {e}")
            return False


# Common scaling policies
BACKPRESSURE_SCALING_POLICY = ScalingPolicy(
    name="backpressure_autoscaling",
    metric="backpressure_ratio",
    target_value=0.3,  # Target 30% backpressure
    min_replicas=1,
    max_replicas=20,
    scale_up_cooldown_seconds=60,
    scale_down_cooldown_seconds=300,
)

THROUGHPUT_SCALING_POLICY = ScalingPolicy(
    name="throughput_autoscaling",
    metric="records_in_per_second",
    target_value=10000,  # Target 10k records/sec per replica
    min_replicas=2,
    max_replicas=50,
    scale_up_cooldown_seconds=120,
    scale_down_cooldown_seconds=600,
)


@dataclass
class SavepointConfig:
    """Configuration for savepoint-based scaling."""

    savepoint_path: str
    timeout_seconds: int = 300
    drain: bool = True  # Wait for in-flight data


class SavepointManager:
    """Manage savepoints for scaling operations."""

    def __init__(self, flink_rest_url: str = "http://localhost:8081"):
        """
        Initialize savepoint manager.

        Args:
            flink_rest_url: Flink REST API URL
        """
        self._flink_url = flink_rest_url.rstrip("/")
        self._savepoints: Dict[str, str] = {}  # job_id -> savepoint_path

    def trigger_savepoint(
        self,
        job_id: str,
        target_directory: str,
        cancel_job: bool = False,
    ) -> Optional[str]:
        """
        Trigger a savepoint for a Flink job.

        Args:
            job_id: Flink job ID
            target_directory: Directory to store savepoint
            cancel_job: Whether to cancel job after savepoint

        Returns:
            Savepoint path if successful
        """
        # In production, would POST to Flink REST API:
        # POST /jobs/{job_id}/savepoints
        savepoint_path = f"{target_directory}/savepoint-{job_id}-{int(time.time())}"
        self._savepoints[job_id] = savepoint_path

        logger.info(f"Triggered savepoint for job {job_id}: {savepoint_path}")
        return savepoint_path

    def get_latest_savepoint(self, job_id: str) -> Optional[str]:
        """Get the latest savepoint path for a job."""
        return self._savepoints.get(job_id)

    def restore_from_savepoint(
        self,
        job_id: str,
        savepoint_path: str,
        new_parallelism: int,
    ) -> bool:
        """
        Restore a job from savepoint with new parallelism.

        Args:
            job_id: Flink job ID
            savepoint_path: Path to savepoint
            new_parallelism: New parallelism level

        Returns:
            True if successful
        """
        # In production, would POST to Flink REST API:
        # POST /jars/{jar_id}/run with savepointPath and parallelism
        logger.info(
            f"Restoring job {job_id} from {savepoint_path} "
            f"with parallelism {new_parallelism}"
        )
        return True


class AutoscalingController:
    """
    High-level controller for automated scaling.

    Combines metrics collection, scaling decisions, and savepoint management
    for seamless autoscaling of Flink jobs.
    """

    def __init__(
        self,
        autoscaler: FlinkAutoscaler,
        savepoint_manager: SavepointManager,
        poll_interval_seconds: int = 30,
    ):
        """
        Initialize the autoscaling controller.

        Args:
            autoscaler: Autoscaler instance
            savepoint_manager: Savepoint manager instance
            poll_interval_seconds: How often to check for scaling
        """
        self._autoscaler = autoscaler
        self._savepoint_manager = savepoint_manager
        self._poll_interval = poll_interval_seconds
        self._running = False

    async def start(self) -> None:
        """Start the autoscaling control loop."""
        self._running = True
        logger.info("Starting autoscaling controller")

        while self._running:
            try:
                decisions = self._autoscaler.run_autoscaling_cycle()

                for job_id, decision in decisions.items():
                    if decision.should_scale:
                        logger.info(
                            f"Autoscaling event: {job_id} "
                            f"{decision.current_replicas} -> {decision.target_replicas}"
                        )

            except Exception as e:
                logger.error(f"Autoscaling cycle error: {e}")

            await self._async_sleep(self._poll_interval)

    async def stop(self) -> None:
        """Stop the autoscaling control loop."""
        self._running = False
        logger.info("Stopping autoscaling controller")

    async def _async_sleep(self, seconds: float) -> None:
        """Async sleep helper."""
        import asyncio
        await asyncio.sleep(seconds)
