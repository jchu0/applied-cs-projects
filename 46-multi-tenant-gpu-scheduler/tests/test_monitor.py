"""Unit tests for GPU monitoring module."""

import pytest
import time
from unittest.mock import MagicMock, patch
from collections import deque

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpusched.core.resources import (
    GPUType, JobState, create_gpu, create_node, create_training_job,
    Cluster, Pod, Container, GPUResources
)
from gpusched.monitor.monitor import (
    MetricType, MetricPoint, GPUMetrics, NodeMetrics,
    ClusterMetrics, MetricsCollector, MetricsAggregator,
    AlertLevel, Alert, AlertManager, HealthChecker, GPUMonitor
)


class TestMetricPoint:
    """Tests for MetricPoint class."""

    def test_initialization(self):
        """Test metric point initialization."""
        with patch('time.time', return_value=1000.0):
            point = MetricPoint(
                metric_type=MetricType.GPU_UTILIZATION,
                value=75.5,
                labels={"gpu_id": "gpu-001", "node": "host-001"}
            )

        assert point.metric_type == MetricType.GPU_UTILIZATION
        assert point.value == 75.5
        assert point.timestamp == 1000.0
        assert point.labels["gpu_id"] == "gpu-001"


class TestGPUMetrics:
    """Tests for GPUMetrics class."""

    def test_initialization(self):
        """Test GPU metrics initialization."""
        metrics = GPUMetrics(
            gpu_id="gpu-001",
            utilization=85.0,
            memory_used_gb=60.0,
            memory_total_gb=80.0,
            temperature=72.0,
            power_usage=250.0,
            pcie_throughput_mb=1500.0,
            error_count=0
        )

        assert metrics.gpu_id == "gpu-001"
        assert metrics.utilization == 85.0
        assert metrics.memory_utilization == 75.0  # 60/80

    def test_memory_utilization(self):
        """Test memory utilization calculation."""
        metrics = GPUMetrics(
            gpu_id="gpu-001",
            memory_used_gb=40.0,
            memory_total_gb=80.0
        )

        assert metrics.memory_utilization == 50.0

        # Zero total memory
        metrics.memory_total_gb = 0
        assert metrics.memory_utilization == 0.0


class TestNodeMetrics:
    """Tests for NodeMetrics class."""

    def test_initialization(self):
        """Test node metrics initialization."""
        gpu_metrics = [
            GPUMetrics("gpu-001", utilization=80.0),
            GPUMetrics("gpu-002", utilization=60.0)
        ]

        node_metrics = NodeMetrics(
            node_id="node-001",
            gpu_metrics=gpu_metrics,
            cpu_utilization=45.0,
            memory_used_gb=128.0,
            memory_total_gb=256.0,
            network_rx_mb=100.0,
            network_tx_mb=150.0,
            disk_io_mb=50.0
        )

        assert node_metrics.node_id == "node-001"
        assert len(node_metrics.gpu_metrics) == 2
        assert node_metrics.avg_gpu_utilization == 70.0

    def test_avg_gpu_metrics(self):
        """Test average GPU metrics calculation."""
        gpu_metrics = [
            GPUMetrics("gpu-001", utilization=80.0, temperature=70.0, power_usage=250.0),
            GPUMetrics("gpu-002", utilization=60.0, temperature=65.0, power_usage=200.0),
            GPUMetrics("gpu-003", utilization=90.0, temperature=75.0, power_usage=280.0)
        ]

        node_metrics = NodeMetrics(
            node_id="node-001",
            gpu_metrics=gpu_metrics
        )

        assert node_metrics.avg_gpu_utilization == pytest.approx(76.67, rel=1e-2)
        assert node_metrics.avg_gpu_temperature == 70.0
        assert node_metrics.avg_gpu_power == pytest.approx(243.33, rel=1e-2)


class TestClusterMetrics:
    """Tests for ClusterMetrics class."""

    def test_initialization(self):
        """Test cluster metrics initialization."""
        node1_metrics = NodeMetrics(
            node_id="node-001",
            gpu_metrics=[
                GPUMetrics("gpu-001", utilization=80.0),
                GPUMetrics("gpu-002", utilization=60.0)
            ]
        )
        node2_metrics = NodeMetrics(
            node_id="node-002",
            gpu_metrics=[
                GPUMetrics("gpu-003", utilization=90.0),
                GPUMetrics("gpu-004", utilization=70.0)
            ]
        )

        cluster_metrics = ClusterMetrics(
            cluster_id="cluster-001",
            node_metrics=[node1_metrics, node2_metrics],
            total_gpus=4,
            available_gpus=1,
            running_pods=10,
            pending_pods=5,
            total_jobs=15,
            failed_jobs=2
        )

        assert cluster_metrics.cluster_id == "cluster-001"
        assert cluster_metrics.total_gpus == 4
        assert cluster_metrics.gpu_availability == 25.0  # 1/4
        assert cluster_metrics.avg_gpu_utilization == 75.0  # (80+60+90+70)/4

    def test_gpu_metrics_aggregation(self):
        """Test GPU metrics aggregation across cluster."""
        node_metrics = [
            NodeMetrics(
                node_id="node-001",
                gpu_metrics=[
                    GPUMetrics("gpu-001", utilization=100.0),
                    GPUMetrics("gpu-002", utilization=0.0)
                ]
            ),
            NodeMetrics(
                node_id="node-002",
                gpu_metrics=[
                    GPUMetrics("gpu-003", utilization=50.0)
                ]
            )
        ]

        cluster_metrics = ClusterMetrics(
            cluster_id="cluster-001",
            node_metrics=node_metrics,
            total_gpus=3
        )

        assert cluster_metrics.avg_gpu_utilization == 50.0  # (100+0+50)/3


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_initialization(self):
        """Test collector initialization."""
        cluster = Cluster("cluster-001")
        collector = MetricsCollector(cluster)

        assert collector.cluster == cluster
        assert collector.collection_interval == 60
        assert len(collector.metrics_history) == 0

    def test_collect_gpu_metrics(self):
        """Test collecting GPU metrics."""
        cluster = Cluster("cluster-001")
        node = create_node("host-001", num_gpus=2)

        # Set some GPU states
        node.gpus[0].utilization = 75.0
        node.gpus[0].temperature = 70.0
        node.gpus[0].power_usage = 250.0

        cluster.add_node(node)
        collector = MetricsCollector(cluster)

        gpu_metrics = collector.collect_gpu_metrics(node.gpus[0])

        assert gpu_metrics.gpu_id == node.gpus[0].gpu_id
        assert gpu_metrics.utilization == 75.0
        assert gpu_metrics.temperature == 70.0
        assert gpu_metrics.power_usage == 250.0

    def test_collect_node_metrics(self):
        """Test collecting node metrics."""
        cluster = Cluster("cluster-001")
        node = create_node("host-001", num_gpus=2)
        cluster.add_node(node)

        collector = MetricsCollector(cluster)
        node_metrics = collector.collect_node_metrics(node)

        assert node_metrics.node_id == node.node_id
        assert len(node_metrics.gpu_metrics) == 2

    def test_collect_cluster_metrics(self):
        """Test collecting cluster metrics."""
        cluster = Cluster("cluster-001")

        # Add nodes
        node1 = create_node("host-001", num_gpus=2)
        node2 = create_node("host-002", num_gpus=2)
        cluster.add_node(node1)
        cluster.add_node(node2)

        # Add some jobs and pods
        job = create_training_job("job-001")
        cluster.submit_job(job)

        collector = MetricsCollector(cluster)
        cluster_metrics = collector.collect_cluster_metrics()

        assert cluster_metrics.cluster_id == "cluster-001"
        assert cluster_metrics.total_gpus == 4
        assert len(cluster_metrics.node_metrics) == 2

    def test_collect_metrics(self):
        """Test full metrics collection."""
        cluster = Cluster("cluster-001")
        node = create_node("host-001", num_gpus=1)
        cluster.add_node(node)

        collector = MetricsCollector(cluster, max_history=10)

        # Collect multiple times
        for i in range(15):
            with patch('time.time', return_value=1000.0 + i):
                collector.collect_metrics()

        # Should only keep max_history entries
        assert len(collector.metrics_history) == 10

    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        cluster = Cluster("cluster-001")
        node = create_node("host-001", num_gpus=2)
        cluster.add_node(node)

        collector = MetricsCollector(cluster)
        collector.collect_metrics()

        summary = collector.get_metrics_summary()

        assert "cluster_id" in summary
        assert "total_gpus" in summary
        assert "avg_gpu_utilization" in summary
        assert "timestamp" in summary


class TestMetricsAggregator:
    """Tests for MetricsAggregator class."""

    def test_aggregate_over_time(self):
        """Test time-based aggregation."""
        aggregator = MetricsAggregator()

        # Create time series data
        metrics = []
        for i in range(10):
            cluster_metrics = ClusterMetrics(
                cluster_id="cluster-001",
                node_metrics=[],
                timestamp=1000.0 + i * 60  # 1 minute intervals
            )
            # Varying utilization
            cluster_metrics.avg_gpu_utilization = 50.0 + i * 5
            metrics.append(cluster_metrics)

        agg = aggregator.aggregate_over_time(metrics, window_minutes=5)

        assert agg["avg_gpu_utilization"]["mean"] > 0
        assert agg["avg_gpu_utilization"]["max"] >= agg["avg_gpu_utilization"]["min"]

    def test_calculate_percentiles(self):
        """Test percentile calculation."""
        aggregator = MetricsAggregator()

        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        percentiles = aggregator.calculate_percentiles(
            values,
            percentiles=[25, 50, 75, 90, 99]
        )

        assert percentiles["p50"] == 55.0  # Median
        assert percentiles["p25"] < percentiles["p75"]
        assert percentiles["p90"] > percentiles["p50"]


class TestAlert:
    """Tests for Alert class."""

    def test_initialization(self):
        """Test alert initialization."""
        with patch('time.time', return_value=1000.0):
            alert = Alert(
                alert_id="alert-001",
                level=AlertLevel.WARNING,
                message="High GPU utilization",
                source="gpu-001",
                details={"utilization": 95.0}
            )

        assert alert.alert_id == "alert-001"
        assert alert.level == AlertLevel.WARNING
        assert alert.timestamp == 1000.0
        assert not alert.resolved


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_create_alert(self):
        """Test creating alerts."""
        manager = AlertManager()

        alert_id = manager.create_alert(
            level=AlertLevel.ERROR,
            message="GPU failure",
            source="gpu-001",
            details={"error": "Memory error"}
        )

        assert alert_id in manager.active_alerts
        assert manager.active_alerts[alert_id].level == AlertLevel.ERROR

    def test_resolve_alert(self):
        """Test resolving alerts."""
        manager = AlertManager()

        alert_id = manager.create_alert(
            level=AlertLevel.WARNING,
            message="High temperature",
            source="gpu-001"
        )

        manager.resolve_alert(alert_id)

        assert alert_id not in manager.active_alerts
        assert alert_id in manager.resolved_alerts
        assert manager.resolved_alerts[alert_id].resolved

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        manager = AlertManager()

        # Create alerts of different levels
        manager.create_alert(AlertLevel.INFO, "Info message", "source1")
        manager.create_alert(AlertLevel.WARNING, "Warning message", "source2")
        manager.create_alert(AlertLevel.ERROR, "Error message", "source3")
        manager.create_alert(AlertLevel.CRITICAL, "Critical message", "source4")

        all_alerts = manager.get_active_alerts()
        assert len(all_alerts) == 4

        critical_alerts = manager.get_active_alerts(min_level=AlertLevel.CRITICAL)
        assert len(critical_alerts) == 1

        error_plus = manager.get_active_alerts(min_level=AlertLevel.ERROR)
        assert len(error_plus) == 2  # ERROR and CRITICAL

    def test_check_alert_conditions(self):
        """Test checking alert conditions."""
        manager = AlertManager()
        cluster_metrics = ClusterMetrics(
            cluster_id="cluster-001",
            node_metrics=[],
            avg_gpu_utilization=95.0,
            gpu_availability=5.0
        )

        alerts = manager.check_alert_conditions(cluster_metrics)

        # Should trigger high utilization alert
        assert len(alerts) > 0
        assert any("utilization" in alert.message.lower() for alert in alerts)

        # Should trigger low availability alert
        assert any("availability" in alert.message.lower() for alert in alerts)


class TestHealthChecker:
    """Tests for HealthChecker class."""

    def test_check_gpu_health(self):
        """Test GPU health checking."""
        checker = HealthChecker()

        # Healthy GPU
        gpu = create_gpu("node-001")
        gpu.temperature = 70.0
        gpu.utilization = 80.0

        health = checker.check_gpu_health(gpu)
        assert health["healthy"]
        assert len(health["issues"]) == 0

        # Overheating GPU
        gpu.temperature = 95.0
        health = checker.check_gpu_health(gpu)
        assert not health["healthy"]
        assert any("temperature" in issue.lower() for issue in health["issues"])

    def test_check_node_health(self):
        """Test node health checking."""
        checker = HealthChecker()

        # Healthy node
        node = create_node("host-001", num_gpus=2)
        health = checker.check_node_health(node)
        assert health["healthy"]

        # Unhealthy node
        node.conditions["Ready"] = False
        health = checker.check_node_health(node)
        assert not health["healthy"]
        assert any("not ready" in issue.lower() for issue in health["issues"])

    def test_check_cluster_health(self):
        """Test cluster health checking."""
        cluster = Cluster("cluster-001")
        node1 = create_node("host-001", num_gpus=2)
        node2 = create_node("host-002", num_gpus=2)
        cluster.add_node(node1)
        cluster.add_node(node2)

        checker = HealthChecker()
        health = checker.check_cluster_health(cluster)

        assert health["total_nodes"] == 2
        assert health["healthy_nodes"] == 2
        assert health["total_gpus"] == 4
        assert health["healthy_gpus"] == 4
        assert health["overall_health"] == "healthy"

        # Make one node unhealthy
        node1.conditions["Ready"] = False
        health = checker.check_cluster_health(cluster)
        assert health["healthy_nodes"] == 1
        assert health["overall_health"] == "degraded"


class TestGPUMonitor:
    """Tests for main GPUMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        cluster = Cluster("cluster-001")
        monitor = GPUMonitor(cluster)

        assert monitor.cluster == cluster
        assert monitor.collector is not None
        assert monitor.alert_manager is not None
        assert monitor.health_checker is not None

    def test_start_monitoring(self):
        """Test starting monitoring."""
        cluster = Cluster("cluster-001")
        node = create_node("host-001", num_gpus=1)
        cluster.add_node(node)

        monitor = GPUMonitor(cluster)

        # Mock the monitoring loop
        with patch.object(monitor, '_monitoring_loop') as mock_loop:
            monitor.start_monitoring(interval=10)
            assert monitor.monitoring_interval == 10
            assert monitor.is_monitoring
            mock_loop.assert_called_once()

    def test_get_status(self):
        """Test getting monitor status."""
        cluster = Cluster("cluster-001")
        node = create_node("host-001", num_gpus=2)
        cluster.add_node(node)

        monitor = GPUMonitor(cluster)
        monitor.collector.collect_metrics()

        status = monitor.get_status()

        assert "is_monitoring" in status
        assert "cluster_health" in status
        assert "current_metrics" in status
        assert "active_alerts" in status

    def test_get_gpu_stats(self):
        """Test getting GPU statistics."""
        cluster = Cluster("cluster-001")
        node = create_node("host-001", num_gpus=2)
        cluster.add_node(node)

        monitor = GPUMonitor(cluster)
        stats = monitor.get_gpu_stats()

        assert len(stats) == 2
        assert all("gpu_id" in s for s in stats)
        assert all("node_id" in s for s in stats)

    def test_export_metrics(self):
        """Test exporting metrics."""
        cluster = Cluster("cluster-001")
        node = create_node("host-001", num_gpus=1)
        cluster.add_node(node)

        monitor = GPUMonitor(cluster)
        monitor.collector.collect_metrics()

        # Test Prometheus format
        prometheus_metrics = monitor.export_metrics(format="prometheus")
        assert "gpu_utilization" in prometheus_metrics
        assert "TYPE" in prometheus_metrics

        # Test JSON format
        json_metrics = monitor.export_metrics(format="json")
        assert isinstance(json_metrics, dict)
        assert "metrics" in json_metrics
        assert "timestamp" in json_metrics


class TestPublicExports:
    """Tests that documented public API is exported from the package roots."""

    def test_monitor_subpackage_exports(self):
        """Types documented in docs/API.md are importable from gpusched.monitor."""
        import gpusched.monitor as m

        for name in [
            "MetricType", "MetricPoint", "GPUMetrics", "NodeMetrics",
            "JobMetrics", "TenantMetrics", "ClusterMetrics", "MetricsCollector",
            "MetricsAggregator", "HealthChecker", "AlertLevel", "Alert",
            "AlertManager", "QuotaManager", "PreemptionManager",
            "ClusterMonitor", "GPUMonitor",
        ]:
            assert name in m.__all__, f"{name} missing from gpusched.monitor.__all__"
            assert hasattr(m, name), f"{name} not importable from gpusched.monitor"

    def test_alertmanager_exported_from_package_root(self):
        """AlertManager is named in the README features and must be public."""
        import gpusched

        for name in ["AlertManager", "AlertLevel", "Alert", "HealthChecker",
                     "ClusterMonitor", "MetricsCollector"]:
            assert name in gpusched.__all__, f"{name} missing from gpusched.__all__"

        from gpusched import AlertManager as RootAlertManager
        from gpusched.monitor.monitor import AlertManager as ModAlertManager
        assert RootAlertManager is ModAlertManager

    def test_readme_alert_example_runs(self):
        """The AlertManager usage shown in docs/API.md works end to end."""
        from gpusched import AlertManager, AlertLevel

        am = AlertManager()
        alert_id = am.create_alert(
            level=AlertLevel.WARNING,
            message="GPU utilization above 90%",
            source="gpu-001",
            details={"utilization": 92.5},
        )
        assert alert_id in am.active_alerts
        assert am.get_active_alerts(min_level=AlertLevel.WARNING)
        assert am.resolve_alert(alert_id) is True
        assert alert_id not in am.active_alerts