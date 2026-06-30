"""Configuration classes for Feature Engineering Platform."""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional
from enum import Enum


class StoreType(Enum):
    """Types of feature stores."""

    MEMORY = "memory"
    REDIS = "redis"
    DYNAMODB = "dynamodb"
    PARQUET = "parquet"
    DUCKDB = "duckdb"
    POSTGRES = "postgres"


@dataclass
class OnlineStoreConfig:
    """Configuration for online feature store."""

    store_type: StoreType = StoreType.MEMORY
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    connection_pool_size: int = 10
    timeout_ms: int = 1000
    ttl: timedelta = field(default_factory=lambda: timedelta(days=1))
    prefix: str = "feature:"
    options: Dict[str, Any] = field(default_factory=dict)

    def get_connection_string(self) -> str:
        """Get connection string for the store."""
        if self.store_type == StoreType.REDIS:
            protocol = "rediss" if self.ssl else "redis"
            auth = f":{self.password}@" if self.password else ""
            return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"
        return ""


@dataclass
class OfflineStoreConfig:
    """Configuration for offline feature store."""

    store_type: StoreType = StoreType.PARQUET
    path: str = "./feature_store"
    database: str = "features"
    schema: str = "public"
    connection_string: Optional[str] = None
    partition_by: List[str] = field(default_factory=lambda: ["date"])
    file_format: str = "parquet"
    compression: str = "snappy"
    row_group_size: int = 100000
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistryConfig:
    """Configuration for feature registry."""

    registry_type: str = "file"  # file, sql, s3
    path: str = "./feature_registry"
    connection_string: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_prefix: str = "registry"
    cache_ttl: timedelta = field(default_factory=lambda: timedelta(minutes=5))


@dataclass
class FeatureStoreConfig:
    """Main configuration for the feature store."""

    project_name: str = "default"
    online_store: OnlineStoreConfig = field(default_factory=OnlineStoreConfig)
    offline_store: OfflineStoreConfig = field(default_factory=OfflineStoreConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    entity_key_serialization: str = "json"  # json, msgpack
    feature_view_ttl: timedelta = field(default_factory=lambda: timedelta(days=1))
    enable_caching: bool = True
    cache_size: int = 10000
    log_level: str = "INFO"


@dataclass
class PipelineConfig:
    """Configuration for feature pipeline execution."""

    max_workers: int = 4
    batch_size: int = 10000
    checkpoint_interval: int = 1000
    retry_count: int = 3
    retry_delay_seconds: int = 5
    timeout_seconds: int = 3600
    enable_checkpointing: bool = True
    checkpoint_path: str = "./checkpoints"
    enable_metrics: bool = True
    metrics_port: int = 9090


@dataclass
class ValidationConfig:
    """Configuration for feature validation."""

    enable_schema_validation: bool = True
    enable_statistical_validation: bool = True
    enable_drift_detection: bool = True
    null_threshold: float = 0.1  # Max allowed null ratio
    outlier_std_threshold: float = 3.0  # Standard deviations for outlier detection
    drift_threshold: float = 0.1  # Max allowed drift score
    min_samples_for_validation: int = 100
    validation_window_days: int = 7
    reference_window_days: int = 30
    alert_on_validation_failure: bool = True


@dataclass
class TransformConfig:
    """Configuration for a transformation."""

    name: str
    transformer_type: str
    input_columns: List[str]
    output_columns: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    fit_on_train: bool = True
    save_state: bool = True


@dataclass
class SerializationConfig:
    """Configuration for serialization."""

    format: str = "parquet"  # parquet, json, msgpack
    compression: str = "snappy"  # snappy, gzip, lz4, none
    include_metadata: bool = True
    version: str = "1.0"


@dataclass
class MonitoringConfig:
    """Configuration for feature monitoring."""

    enable_metrics: bool = True
    metrics_backend: str = "prometheus"  # prometheus, statsd, datadog
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log"])
    slack_webhook: Optional[str] = None
    pagerduty_key: Optional[str] = None
    alert_cooldown_minutes: int = 15
    health_check_interval_seconds: int = 60
