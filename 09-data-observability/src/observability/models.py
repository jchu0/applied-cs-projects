"""Core data models for Data Observability Platform."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AnomalyType(Enum):
    """Types of data anomalies."""

    VOLUME = "volume"
    FRESHNESS = "freshness"
    SCHEMA = "schema"
    DISTRIBUTION = "distribution"
    NULL_RATE = "null_rate"
    UNIQUENESS = "uniqueness"
    CUSTOM = "custom"


class AlertStatus(Enum):
    """Status of an alert."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class ColumnStats:
    """Statistical profile for a column."""

    null_count: int
    null_ratio: float
    distinct_count: int
    min_value: Any = None
    max_value: Any = None
    mean: Optional[float] = None
    stddev: Optional[float] = None
    histogram: Optional[List[int]] = None


@dataclass
class ColumnMetadata:
    """Metadata for a column."""

    name: str
    data_type: str
    nullable: bool
    description: Optional[str] = None
    stats: Optional[ColumnStats] = None


@dataclass
class TableMetadata:
    """Metadata for a data table."""

    table_id: str
    database: str
    schema: str
    table_name: str
    columns: List[ColumnMetadata]
    row_count: int
    size_bytes: int
    last_modified: datetime
    partitions: List[str] = field(default_factory=list)
    owner: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Anomaly:
    """Detected anomaly."""

    anomaly_id: str
    table_id: str
    column_name: Optional[str]
    anomaly_type: AnomalyType
    severity: str  # critical, warning, info
    detected_at: datetime
    metric_value: float
    expected_range: Tuple[float, float]
    description: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert entity."""

    alert_id: str
    anomaly: Anomaly
    created_at: datetime
    status: str = "active"
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class DataHealthScore:
    """Health score for a data asset."""

    table_id: str
    overall_score: float  # 0-100
    freshness_score: float
    volume_score: float
    schema_score: float
    quality_score: float
    calculated_at: datetime
    factors: List[dict] = field(default_factory=list)


@dataclass
class LineageInfo:
    """Lineage information for a table."""

    table_id: str
    upstream: List[str] = field(default_factory=list)
    downstream: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)


@dataclass
class ImpactAnalysis:
    """Impact analysis result."""

    source_table: str
    affected_tables: List[TableMetadata]
    affected_pipelines: List[str]
    affected_dashboards: List[str]
    total_downstream: int


@dataclass
class Pipeline:
    """Data pipeline entity."""

    pipeline_id: str
    name: str
    orchestrator: str  # airflow, dagster, etc.
    schedule: str
    source_tables: List[str]
    target_tables: List[str]
    last_run: Optional[datetime] = None
    last_status: Optional[str] = None


@dataclass
class SchemaChange:
    """Schema change information."""

    change_type: str  # column_added, column_removed, type_changed
    column_name: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None


@dataclass
class MetricHistory:
    """Historical metric data point."""

    table_id: str
    metric_name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
