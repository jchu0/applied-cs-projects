"""Configuration types for the data lakehouse."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from pyspark.sql.types import StructType
else:
    StructType = Any  # Allow runtime usage without pyspark


class Layer(Enum):
    """Medallion architecture layers."""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


@dataclass
class LakehouseConfig:
    """Main configuration for the lakehouse."""

    # Storage paths
    lakehouse_path: str
    checkpoint_path: str

    # Spark configuration
    spark_master: str = "local[*]"
    app_name: str = "data-lakehouse"

    # Delta Lake settings
    enable_change_data_feed: bool = True
    auto_compact: bool = True
    optimize_write: bool = True

    # Retention settings
    log_retention_days: int = 30
    deleted_file_retention_hours: int = 168  # 7 days

    # Metastore
    metastore_uri: Optional[str] = None

    def get_layer_path(self, layer: Layer) -> str:
        """Get the path for a specific layer."""
        return f"{self.lakehouse_path}/{layer.value}"


@dataclass
class DeltaTableConfig:
    """Configuration for a Delta Lake table."""

    name: str
    path: str
    layer: Layer
    schema: Optional["StructType"] = None
    partition_columns: List[str] = field(default_factory=list)
    z_order_columns: List[str] = field(default_factory=list)
    description: str = ""

    # Delta-specific configurations
    enable_change_data_feed: bool = False
    auto_compact: bool = True
    optimize_write: bool = True

    # Retention settings
    log_retention_days: int = 30
    deleted_file_retention_hours: int = 168

    # Quality constraints
    constraints: Dict[str, str] = field(default_factory=dict)

    def to_table_properties(self) -> Dict[str, str]:
        """Convert to Delta Lake table properties."""
        return {
            "delta.enableChangeDataFeed": str(self.enable_change_data_feed).lower(),
            "delta.autoOptimize.autoCompact": str(self.auto_compact).lower(),
            "delta.autoOptimize.optimizeWrite": str(self.optimize_write).lower(),
            "delta.logRetentionDuration": f"interval {self.log_retention_days} days",
            "delta.deletedFileRetentionDuration": f"interval {self.deleted_file_retention_hours} hours",
        }


@dataclass
class IncrementalConfig:
    """Configuration for incremental processing."""

    watermark_column: str
    merge_keys: List[str]
    scd_type: int = 1  # 1 for Type 1 (overwrite), 2 for Type 2 (history)

    # For SCD Type 2
    effective_date_col: str = "effective_date"
    end_date_col: str = "end_date"
    current_flag_col: str = "is_current"


@dataclass
class QualityRule:
    """Data quality rule definition."""

    name: str
    column: str
    expectation_type: str  # e.g., "not_null", "unique", "in_set"
    expectation_kwargs: Dict = field(default_factory=dict)
    severity: str = "error"  # error, warning


@dataclass
class ColumnStats:
    """Column-level statistics for query optimization."""

    column_name: str
    data_type: str
    num_nulls: int
    min_value: any
    max_value: any
    num_distinct: Optional[int] = None
    avg_length: Optional[float] = None
    mean: Optional[float] = None
    stddev: Optional[float] = None


@dataclass
class TableLineage:
    """Track table dependencies."""

    table_name: str
    upstream_tables: List[str]
    downstream_tables: List[str]
    transformation_query: str
    refresh_schedule: str
