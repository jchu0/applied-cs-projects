"""Tests for configuration module."""

import pytest
from observability.config import (
    DetectorConfig, CollectorConfig, WarehouseConfig,
    StorageConfig, AlertConfig, ObservabilityConfig
)


class TestDetectorConfig:
    """Tests for DetectorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DetectorConfig()
        assert config.min_history_points == 10
        assert config.volume_threshold == 3.0
        assert config.freshness_threshold_hours == 24.0
        assert config.null_rate_threshold == 3.0
        assert config.distribution_contamination == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DetectorConfig(
            min_history_points=20,
            volume_threshold=2.5,
            freshness_threshold_hours=12.0,
            null_rate_threshold=2.0,
            distribution_contamination=0.05
        )
        assert config.min_history_points == 20
        assert config.volume_threshold == 2.5
        assert config.freshness_threshold_hours == 12.0


class TestCollectorConfig:
    """Tests for CollectorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CollectorConfig()
        assert config.sample_size == 1000
        assert config.profile_timeout_seconds == 300
        assert config.batch_size == 100
        assert config.parallel_workers == 4

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CollectorConfig(
            sample_size=5000,
            profile_timeout_seconds=600,
            batch_size=50,
            parallel_workers=8
        )
        assert config.sample_size == 5000
        assert config.parallel_workers == 8


class TestWarehouseConfig:
    """Tests for WarehouseConfig."""

    def test_creation(self):
        """Test warehouse config creation."""
        config = WarehouseConfig(
            warehouse_type="snowflake",
            host="account.snowflakecomputing.com",
            port=443,
            database="analytics",
            user="etl_user",
            password="secret",
            options={"warehouse": "compute_wh"}
        )
        assert config.warehouse_type == "snowflake"
        assert config.host == "account.snowflakecomputing.com"
        assert "warehouse" in config.options

    def test_to_connection_params(self):
        """Test conversion to connection parameters."""
        config = WarehouseConfig(
            warehouse_type="postgres",
            host="localhost",
            port=5432,
            database="test",
            user="test_user",
            password="test_pass",
            options={"sslmode": "require"}
        )
        params = config.to_connection_params()
        assert params["host"] == "localhost"
        assert params["port"] == 5432
        assert params["database"] == "test"
        assert params["user"] == "test_user"
        assert params["password"] == "test_pass"
        assert params["sslmode"] == "require"


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_values(self):
        """Test default storage configuration."""
        config = StorageConfig()
        assert config.backend == "memory"
        assert config.connection_string is None
        assert config.retention_days == 90

    def test_postgres_storage(self):
        """Test postgres storage configuration."""
        config = StorageConfig(
            backend="postgres",
            connection_string="postgresql://user:pass@localhost/obs",
            retention_days=180
        )
        assert config.backend == "postgres"
        assert "postgresql" in config.connection_string
        assert config.retention_days == 180


class TestAlertConfig:
    """Tests for AlertConfig."""

    def test_default_values(self):
        """Test default alert configuration."""
        config = AlertConfig()
        assert config.default_severity == "warning"
        assert config.escalation_minutes == 30
        assert config.dedup_window_minutes == 60
        assert "slack" in config.channels

    def test_custom_channels(self):
        """Test custom alert channels."""
        config = AlertConfig(
            channels=["pagerduty", "slack", "email"]
        )
        assert len(config.channels) == 3
        assert "pagerduty" in config.channels


class TestObservabilityConfig:
    """Tests for ObservabilityConfig."""

    def test_default_values(self):
        """Test default observability configuration."""
        config = ObservabilityConfig.default()
        assert config.detector is not None
        assert config.collector is not None
        assert config.storage is not None
        assert config.alerts is not None
        assert config.warehouses == {}

    def test_add_warehouse(self):
        """Test adding warehouse configuration."""
        config = ObservabilityConfig()
        wh_config = WarehouseConfig(
            warehouse_type="bigquery",
            host="bigquery.googleapis.com",
            port=443,
            database="project.dataset",
            user="sa@project.iam.gserviceaccount.com",
            password="key_file_path"
        )
        config.add_warehouse("production_bq", wh_config)
        assert "production_bq" in config.warehouses
        assert config.warehouses["production_bq"].warehouse_type == "bigquery"

    def test_nested_configs(self):
        """Test accessing nested configurations."""
        config = ObservabilityConfig(
            detector=DetectorConfig(volume_threshold=2.0),
            collector=CollectorConfig(batch_size=200),
            storage=StorageConfig(retention_days=365),
            alerts=AlertConfig(escalation_minutes=15)
        )
        assert config.detector.volume_threshold == 2.0
        assert config.collector.batch_size == 200
        assert config.storage.retention_days == 365
        assert config.alerts.escalation_minutes == 15
