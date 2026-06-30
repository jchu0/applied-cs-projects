"""Configuration for Semantic Layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WarehouseConfig:
    """Configuration for data warehouse connection."""

    warehouse_type: str  # snowflake, bigquery, postgres, redshift
    host: str = ""
    port: int = 0
    database: str = ""
    schema: str = ""
    user: str = ""
    password: str = ""
    account: str = ""  # For Snowflake
    project: str = ""  # For BigQuery
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for a dbt model."""

    name: str
    materialized: str = "view"
    schema: str = ""
    unique_key: Optional[str] = None
    incremental_strategy: Optional[str] = None
    partition_by: Optional[Dict[str, Any]] = None
    cluster_by: Optional[List[str]] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class SourceConfig:
    """Configuration for a data source."""

    name: str
    database: str
    schema: str
    loader: str = ""
    loaded_at_field: str = ""
    freshness_warn_after_hours: int = 12
    freshness_error_after_hours: int = 24
    tables: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SemanticLayerConfig:
    """Main configuration for semantic layer."""

    project_name: str = "analytics"
    version: str = "1.0.0"
    warehouse: Optional[WarehouseConfig] = None
    model_defaults: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="default")
    )
    staging_schema: str = "staging"
    marts_schema: str = "analytics"
    metric_definitions_path: str = "semantic_layer/metrics"
    dimension_definitions_path: str = "semantic_layer/dimensions"


@dataclass
class MetricFilter:
    """Filter definition for a metric."""

    field: str
    operator: str
    value: str


@dataclass
class MetricMeta:
    """Metadata for a metric."""

    owner: str = ""
    tier: int = 1
    is_percentage: bool = False
    deprecated: bool = False
    tags: List[str] = field(default_factory=list)
