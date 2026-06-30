"""Pytest fixtures for Data Observability tests."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
from datetime import datetime, timedelta
from observability.models import (
    AnomalyType, AlertStatus, ColumnStats, ColumnMetadata,
    TableMetadata, Anomaly, Alert, DataHealthScore, LineageInfo,
    ImpactAnalysis, Pipeline, SchemaChange, MetricHistory
)
from observability.config import DetectorConfig, CollectorConfig, AlertConfig
from observability.collectors import InMemoryCollector
from observability.detector import AnomalyDetector
from observability.alerting import AlertingEngine, LogChannel
from observability.lineage import LineageGraph
from observability.health import HealthScorer
from observability.forecaster import MetricForecaster
from observability.pii import PIIDetector


@pytest.fixture
def sample_column_stats():
    """Sample column statistics."""
    return ColumnStats(
        null_count=10,
        null_ratio=0.01,
        distinct_count=1000,
        min_value=0,
        max_value=10000,
        mean=5000.0,
        stddev=1500.0,
        histogram=[100, 200, 300, 250, 150]
    )


@pytest.fixture
def sample_column_metadata(sample_column_stats):
    """Sample column metadata."""
    return ColumnMetadata(
        name="user_id",
        data_type="INT64",
        nullable=False,
        description="Primary key for users",
        stats=sample_column_stats
    )


@pytest.fixture
def sample_table_metadata(sample_column_metadata):
    """Sample table metadata."""
    return TableMetadata(
        table_id="db.schema.users",
        database="production",
        schema="public",
        table_name="users",
        columns=[sample_column_metadata],
        row_count=100000,
        size_bytes=1024 * 1024 * 100,  # 100MB
        last_modified=datetime.now(),
        partitions=["date=2024-01-01"],
        owner="data_team",
        tags={"pii": "true", "tier": "critical"}
    )


@pytest.fixture
def sample_anomaly():
    """Sample anomaly."""
    return Anomaly(
        anomaly_id="anom_001",
        table_id="db.schema.users",
        column_name="email",
        anomaly_type=AnomalyType.VOLUME,
        severity="warning",
        detected_at=datetime.now(),
        metric_value=50000,
        expected_range=(90000, 110000),
        description="Row count dropped significantly",
        context={"previous_value": 100000}
    )


@pytest.fixture
def sample_alert(sample_anomaly):
    """Sample alert."""
    return Alert(
        alert_id="alert_001",
        anomaly=sample_anomaly,
        created_at=datetime.now(),
        status="active"
    )


@pytest.fixture
def detector_config():
    """Sample detector configuration."""
    return DetectorConfig(
        volume_threshold=2.5,
        freshness_threshold_hours=24,
        null_rate_threshold=0.1,
        schema_change_enabled=True,
        distribution_check_enabled=True
    )


@pytest.fixture
def collector_config():
    """Sample collector configuration."""
    return CollectorConfig(
        refresh_interval_seconds=300,
        batch_size=100
    )


@pytest.fixture
def alert_config():
    """Sample alert configuration."""
    return AlertConfig(
        critical_channels=["pagerduty"],
        warning_channels=["slack"],
        info_channels=["log"],
        escalation_timeout_minutes=30
    )


@pytest.fixture
def in_memory_collector(sample_table_metadata):
    """In-memory collector with sample data."""
    collector = InMemoryCollector()
    collector.add_table(sample_table_metadata)
    return collector


@pytest.fixture
def anomaly_detector(detector_config, in_memory_collector):
    """Configured anomaly detector."""
    return AnomalyDetector(config=detector_config, collector=in_memory_collector)


@pytest.fixture
def alert_engine(alert_config):
    """Configured alert engine."""
    engine = AlertingEngine(config=alert_config)
    engine.add_channel("log", LogChannel())
    return engine


@pytest.fixture
def lineage_graph():
    """Empty lineage graph."""
    return LineageGraph()


@pytest.fixture
def health_calculator():
    """Health calculator instance."""
    return HealthScorer()


@pytest.fixture
def metric_forecaster():
    """Metric forecaster instance."""
    return MetricForecaster()


@pytest.fixture
def pii_detector():
    """PII detector instance."""
    return PIIDetector()


@pytest.fixture
def sample_metric_history():
    """Sample metric history for forecasting."""
    now = datetime.now()
    return [
        MetricHistory(
            table_id="db.schema.users",
            metric_name="row_count",
            value=100000 + i * 100,
            timestamp=now - timedelta(days=30-i)
        )
        for i in range(30)
    ]


@pytest.fixture
def sample_tables_for_lineage():
    """Sample tables for lineage testing."""
    now = datetime.now()
    return [
        TableMetadata(
            table_id="raw.events",
            database="raw",
            schema="public",
            table_name="events",
            columns=[],
            row_count=1000000,
            size_bytes=1024 * 1024 * 500,
            last_modified=now
        ),
        TableMetadata(
            table_id="staging.events_clean",
            database="staging",
            schema="public",
            table_name="events_clean",
            columns=[],
            row_count=950000,
            size_bytes=1024 * 1024 * 450,
            last_modified=now
        ),
        TableMetadata(
            table_id="analytics.user_events",
            database="analytics",
            schema="public",
            table_name="user_events",
            columns=[],
            row_count=500000,
            size_bytes=1024 * 1024 * 200,
            last_modified=now
        ),
    ]
