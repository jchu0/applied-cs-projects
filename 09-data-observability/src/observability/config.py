"""Configuration for Data Observability Platform."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DetectorConfig:
    """Configuration for anomaly detection."""

    min_history_points: int = 10
    volume_threshold: float = 3.0
    freshness_threshold_hours: float = 24.0
    null_rate_threshold: float = 3.0
    distribution_contamination: float = 0.1


@dataclass
class CollectorConfig:
    """Configuration for metadata collection."""

    sample_size: int = 1000
    profile_timeout_seconds: int = 300
    batch_size: int = 100
    parallel_workers: int = 4


@dataclass
class WarehouseConfig:
    """Configuration for data warehouse connection."""

    warehouse_type: str  # snowflake, bigquery, postgres, etc.
    host: str
    port: int
    database: str
    user: str
    password: str
    options: Dict[str, Any] = field(default_factory=dict)

    def to_connection_params(self) -> Dict[str, Any]:
        """Convert to connection parameters."""
        params = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }
        params.update(self.options)
        return params


@dataclass
class StorageConfig:
    """Configuration for metadata storage."""

    backend: str = "memory"  # memory, postgres, redis
    connection_string: Optional[str] = None
    retention_days: int = 90


@dataclass
class AlertConfig:
    """Configuration for alerting."""

    default_severity: str = "warning"
    escalation_minutes: int = 30
    dedup_window_minutes: int = 60
    channels: List[str] = field(default_factory=lambda: ["slack"])


@dataclass
class ObservabilityConfig:
    """Main configuration for observability platform."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    warehouses: Dict[str, WarehouseConfig] = field(default_factory=dict)

    def add_warehouse(self, name: str, config: WarehouseConfig) -> None:
        """Add a warehouse configuration."""
        self.warehouses[name] = config

    @classmethod
    def default(cls) -> "ObservabilityConfig":
        """Create default configuration."""
        return cls()
