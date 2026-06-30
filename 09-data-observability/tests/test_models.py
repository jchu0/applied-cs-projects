"""Tests for data models."""

import pytest
from datetime import datetime
from observability.models import (
    AnomalyType, AlertStatus, ColumnStats, ColumnMetadata,
    TableMetadata, Anomaly, Alert, DataHealthScore, LineageInfo,
    ImpactAnalysis, Pipeline, SchemaChange, MetricHistory
)


class TestAnomalyType:
    """Tests for AnomalyType enum."""

    def test_all_anomaly_types(self):
        """Test all anomaly types are defined."""
        assert AnomalyType.VOLUME.value == "volume"
        assert AnomalyType.FRESHNESS.value == "freshness"
        assert AnomalyType.SCHEMA.value == "schema"
        assert AnomalyType.DISTRIBUTION.value == "distribution"
        assert AnomalyType.NULL_RATE.value == "null_rate"
        assert AnomalyType.UNIQUENESS.value == "uniqueness"
        assert AnomalyType.CUSTOM.value == "custom"

    def test_anomaly_type_count(self):
        """Test correct number of anomaly types."""
        assert len(AnomalyType) == 7


class TestAlertStatus:
    """Tests for AlertStatus enum."""

    def test_all_alert_statuses(self):
        """Test all alert statuses are defined."""
        assert AlertStatus.ACTIVE.value == "active"
        assert AlertStatus.ACKNOWLEDGED.value == "acknowledged"
        assert AlertStatus.RESOLVED.value == "resolved"


class TestColumnStats:
    """Tests for ColumnStats dataclass."""

    def test_column_stats_creation(self, sample_column_stats):
        """Test column stats creation."""
        stats = sample_column_stats
        assert stats.null_count == 10
        assert stats.null_ratio == 0.01
        assert stats.distinct_count == 1000
        assert stats.min_value == 0
        assert stats.max_value == 10000
        assert stats.mean == 5000.0
        assert stats.stddev == 1500.0
        assert len(stats.histogram) == 5

    def test_column_stats_optional_fields(self):
        """Test column stats with optional fields."""
        stats = ColumnStats(
            null_count=0,
            null_ratio=0.0,
            distinct_count=100
        )
        assert stats.min_value is None
        assert stats.max_value is None
        assert stats.mean is None
        assert stats.stddev is None
        assert stats.histogram is None


class TestColumnMetadata:
    """Tests for ColumnMetadata dataclass."""

    def test_column_metadata_creation(self, sample_column_metadata):
        """Test column metadata creation."""
        col = sample_column_metadata
        assert col.name == "user_id"
        assert col.data_type == "INT64"
        assert col.nullable is False
        assert col.description == "Primary key for users"
        assert col.stats is not None

    def test_column_metadata_minimal(self):
        """Test column metadata with minimal fields."""
        col = ColumnMetadata(
            name="test",
            data_type="STRING",
            nullable=True
        )
        assert col.name == "test"
        assert col.description is None
        assert col.stats is None


class TestTableMetadata:
    """Tests for TableMetadata dataclass."""

    def test_table_metadata_creation(self, sample_table_metadata):
        """Test table metadata creation."""
        table = sample_table_metadata
        assert table.table_id == "db.schema.users"
        assert table.database == "production"
        assert table.schema == "public"
        assert table.table_name == "users"
        assert len(table.columns) == 1
        assert table.row_count == 100000
        assert table.size_bytes == 1024 * 1024 * 100
        assert table.owner == "data_team"
        assert "pii" in table.tags

    def test_table_metadata_default_fields(self):
        """Test table metadata default fields."""
        table = TableMetadata(
            table_id="test",
            database="db",
            schema="public",
            table_name="test",
            columns=[],
            row_count=0,
            size_bytes=0,
            last_modified=datetime.now()
        )
        assert table.partitions == []
        assert table.owner == ""
        assert table.tags == {}


class TestAnomaly:
    """Tests for Anomaly dataclass."""

    def test_anomaly_creation(self, sample_anomaly):
        """Test anomaly creation."""
        anomaly = sample_anomaly
        assert anomaly.anomaly_id == "anom_001"
        assert anomaly.table_id == "db.schema.users"
        assert anomaly.column_name == "email"
        assert anomaly.anomaly_type == AnomalyType.VOLUME
        assert anomaly.severity == "warning"
        assert anomaly.metric_value == 50000
        assert anomaly.expected_range == (90000, 110000)
        assert "dropped" in anomaly.description.lower()

    def test_anomaly_without_column(self):
        """Test anomaly without column name."""
        anomaly = Anomaly(
            anomaly_id="test",
            table_id="test.table",
            column_name=None,
            anomaly_type=AnomalyType.FRESHNESS,
            severity="critical",
            detected_at=datetime.now(),
            metric_value=48.0,
            expected_range=(0, 24),
            description="Data is stale"
        )
        assert anomaly.column_name is None


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self, sample_alert):
        """Test alert creation."""
        alert = sample_alert
        assert alert.alert_id == "alert_001"
        assert alert.anomaly is not None
        assert alert.status == "active"
        assert alert.acknowledged_by is None
        assert alert.acknowledged_at is None
        assert alert.resolved_at is None

    def test_alert_acknowledged(self, sample_anomaly):
        """Test acknowledged alert."""
        alert = Alert(
            alert_id="alert_002",
            anomaly=sample_anomaly,
            created_at=datetime.now(),
            status="acknowledged",
            acknowledged_by="user@example.com",
            acknowledged_at=datetime.now()
        )
        assert alert.status == "acknowledged"
        assert alert.acknowledged_by == "user@example.com"
        assert alert.acknowledged_at is not None


class TestDataHealthScore:
    """Tests for DataHealthScore dataclass."""

    def test_health_score_creation(self):
        """Test health score creation."""
        score = DataHealthScore(
            table_id="test.table",
            overall_score=85.5,
            freshness_score=90.0,
            volume_score=80.0,
            schema_score=95.0,
            quality_score=77.0,
            calculated_at=datetime.now(),
            factors=[{"name": "null_rate", "impact": -5}]
        )
        assert score.overall_score == 85.5
        assert score.freshness_score == 90.0
        assert len(score.factors) == 1

    def test_health_score_range(self):
        """Test health score values are valid."""
        score = DataHealthScore(
            table_id="test",
            overall_score=100.0,
            freshness_score=100.0,
            volume_score=100.0,
            schema_score=100.0,
            quality_score=100.0,
            calculated_at=datetime.now()
        )
        assert 0 <= score.overall_score <= 100
        assert 0 <= score.freshness_score <= 100
        assert 0 <= score.volume_score <= 100
        assert 0 <= score.schema_score <= 100
        assert 0 <= score.quality_score <= 100


class TestLineageInfo:
    """Tests for LineageInfo dataclass."""

    def test_lineage_info_creation(self):
        """Test lineage info creation."""
        lineage = LineageInfo(
            table_id="analytics.users",
            upstream=["raw.users", "raw.profiles"],
            downstream=["dashboard.users"],
            transformations=["join", "filter"]
        )
        assert len(lineage.upstream) == 2
        assert len(lineage.downstream) == 1
        assert len(lineage.transformations) == 2

    def test_lineage_info_defaults(self):
        """Test lineage info defaults."""
        lineage = LineageInfo(table_id="test")
        assert lineage.upstream == []
        assert lineage.downstream == []
        assert lineage.transformations == []


class TestImpactAnalysis:
    """Tests for ImpactAnalysis dataclass."""

    def test_impact_analysis_creation(self, sample_table_metadata):
        """Test impact analysis creation."""
        impact = ImpactAnalysis(
            source_table="raw.events",
            affected_tables=[sample_table_metadata],
            affected_pipelines=["etl_users", "etl_events"],
            affected_dashboards=["user_dashboard"],
            total_downstream=5
        )
        assert impact.source_table == "raw.events"
        assert len(impact.affected_tables) == 1
        assert len(impact.affected_pipelines) == 2
        assert len(impact.affected_dashboards) == 1
        assert impact.total_downstream == 5


class TestPipeline:
    """Tests for Pipeline dataclass."""

    def test_pipeline_creation(self):
        """Test pipeline creation."""
        pipeline = Pipeline(
            pipeline_id="pipe_001",
            name="ETL Users",
            orchestrator="airflow",
            schedule="0 * * * *",
            source_tables=["raw.users"],
            target_tables=["staging.users"],
            last_run=datetime.now(),
            last_status="success"
        )
        assert pipeline.pipeline_id == "pipe_001"
        assert pipeline.orchestrator == "airflow"
        assert pipeline.last_status == "success"

    def test_pipeline_defaults(self):
        """Test pipeline defaults."""
        pipeline = Pipeline(
            pipeline_id="test",
            name="Test",
            orchestrator="dagster",
            schedule="@daily",
            source_tables=[],
            target_tables=[]
        )
        assert pipeline.last_run is None
        assert pipeline.last_status is None


class TestSchemaChange:
    """Tests for SchemaChange dataclass."""

    def test_column_added(self):
        """Test column added change."""
        change = SchemaChange(
            change_type="column_added",
            column_name="new_column",
            new_value="STRING"
        )
        assert change.change_type == "column_added"
        assert change.old_value is None
        assert change.new_value == "STRING"

    def test_column_removed(self):
        """Test column removed change."""
        change = SchemaChange(
            change_type="column_removed",
            column_name="old_column",
            old_value="INT64"
        )
        assert change.change_type == "column_removed"
        assert change.old_value == "INT64"
        assert change.new_value is None

    def test_type_changed(self):
        """Test type changed."""
        change = SchemaChange(
            change_type="type_changed",
            column_name="user_id",
            old_value="INT32",
            new_value="INT64"
        )
        assert change.change_type == "type_changed"
        assert change.old_value == "INT32"
        assert change.new_value == "INT64"


class TestMetricHistory:
    """Tests for MetricHistory dataclass."""

    def test_metric_history_creation(self):
        """Test metric history creation."""
        metric = MetricHistory(
            table_id="test.table",
            metric_name="row_count",
            value=100000,
            timestamp=datetime.now(),
            metadata={"source": "collector"}
        )
        assert metric.table_id == "test.table"
        assert metric.metric_name == "row_count"
        assert metric.value == 100000
        assert "source" in metric.metadata

    def test_metric_history_defaults(self):
        """Test metric history defaults."""
        metric = MetricHistory(
            table_id="test",
            metric_name="test",
            value=0,
            timestamp=datetime.now()
        )
        assert metric.metadata == {}
