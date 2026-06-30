# Data Lakehouse API Documentation

## Table of Contents
1. [LakehouseProcessor API](#lakehouseprocessor-api)
2. [DeltaLog API](#deltalog-api)
3. [Optimizer API](#optimizer-api)
4. [Quality Engine API](#quality-engine-api)
5. [Streaming API](#streaming-api)
6. [Configuration API](#configuration-api)

---

## LakehouseProcessor API

### Class: `LakehouseProcessor`

Main processor for lakehouse operations implementing medallion architecture.

#### Constructor

```python
LakehouseProcessor(spark: SparkSession, config: Optional[LakehouseConfig] = None)
```

**Parameters:**
- `spark`: Active SparkSession instance
- `config`: Optional lakehouse configuration

**Example:**
```python
from pyspark.sql import SparkSession
from lakehouse.processor import LakehouseProcessor
from lakehouse.config import LakehouseConfig

spark = SparkSession.builder.appName("lakehouse").getOrCreate()
config = LakehouseConfig(
    bronze_path="/data/bronze",
    silver_path="/data/silver",
    gold_path="/data/gold"
)
processor = LakehouseProcessor(spark, config)
```

#### Methods

##### `bronze_ingestion()`

Ingest raw data into bronze layer with metadata enrichment.

```python
bronze_ingestion(
    source_path: str,
    bronze_path: str,
    source_name: str,
    schema: Optional[StructType] = None,
    file_format: str = "json",
    options: Dict[str, str] = {}
) -> None
```

**Parameters:**
- `source_path`: Path to source data
- `bronze_path`: Path to bronze table
- `source_name`: Name of data source for tracking
- `schema`: Optional schema (inferred if not provided)
- `file_format`: Format of source files (json, csv, parquet, avro, xml)
- `options`: Additional read options

**Example:**
```python
processor.bronze_ingestion(
    source_path="/raw/users/*.json",
    bronze_path="/bronze/users",
    source_name="user_api",
    file_format="json",
    options={"multiLine": "true"}
)
```

##### `silver_transformation()`

Transform bronze data to silver layer with cleaning and validation.

```python
silver_transformation(
    bronze_path: str,
    silver_path: str,
    transformation_func: Callable[[DataFrame], DataFrame],
    partition_columns: Optional[List[str]] = None
) -> None
```

**Parameters:**
- `bronze_path`: Path to bronze table
- `silver_path`: Path to silver table
- `transformation_func`: Function to transform data
- `partition_columns`: Columns to partition by

**Example:**
```python
def clean_users(df):
    return (
        df.filter(col("email").isNotNull())
          .filter(col("age").between(0, 120))
          .withColumn("email_domain", split(col("email"), "@")[1])
    )

processor.silver_transformation(
    bronze_path="/bronze/users",
    silver_path="/silver/users",
    transformation_func=clean_users,
    partition_columns=["registration_year"]
)
```

##### `gold_aggregation()`

Create business-level aggregations in gold layer.

```python
gold_aggregation(
    silver_path: str,
    gold_path: str,
    aggregation_func: Callable[[DataFrame], DataFrame]
) -> None
```

**Example:**
```python
def daily_revenue(df):
    return (
        df.groupBy("date", "product_category")
          .agg(
              sum("amount").alias("total_revenue"),
              count("*").alias("transaction_count")
          )
    )

processor.gold_aggregation(
    silver_path="/silver/transactions",
    gold_path="/gold/daily_revenue",
    aggregation_func=daily_revenue
)
```

##### `apply_scd_type2()`

Apply Slowly Changing Dimension Type 2 logic.

```python
apply_scd_type2(
    silver_path: str,
    updates_df: DataFrame,
    key_columns: List[str],
    track_columns: List[str]
) -> None
```

**Parameters:**
- `silver_path`: Path to dimension table
- `updates_df`: DataFrame with updates
- `key_columns`: Primary key columns
- `track_columns`: Columns to track changes

**Example:**
```python
updates = spark.read.json("/updates/customers.json")

processor.apply_scd_type2(
    silver_path="/silver/dim_customer",
    updates_df=updates,
    key_columns=["customer_id"],
    track_columns=["address", "phone", "email"]
)
```

##### `merge_delta_tables()`

Perform MERGE operation on Delta tables.

```python
merge_delta_tables(
    target_path: str,
    source_df: DataFrame,
    merge_condition: str,
    update_condition: Optional[str] = None,
    update_set: Optional[Dict[str, str]] = None,
    insert_condition: Optional[str] = None
) -> None
```

**Example:**
```python
source = spark.read.parquet("/staging/products")

processor.merge_delta_tables(
    target_path="/silver/products",
    source_df=source,
    merge_condition="target.product_id = source.product_id",
    update_condition="target.price != source.price",
    update_set={"price": "source.price", "updated_at": "current_timestamp()"},
    insert_condition="source.product_id IS NOT NULL"
)
```

---

## DeltaLog API

### Class: `DeltaLog`

Manages Delta Lake transaction logs and metadata.

#### Constructor

```python
DeltaLog(table_path: str)
```

#### Methods

##### `commit()`

Commit actions to transaction log.

```python
commit(actions: List[Action]) -> int
```

**Returns:** Version number of the commit

**Example:**
```python
from lakehouse.delta_log import DeltaLog, AddFile, RemoveFile

log = DeltaLog("/data/my_table")

# Add a file
add_action = AddFile(
    path="part-001.parquet",
    partition_values={"date": "2024-01-01"},
    size=1048576,
    modification_time=1704067200,
    data_change=True
)

version = log.commit([add_action])
print(f"Committed version: {version}")
```

##### `read_version()`

Read actions from a specific version.

```python
read_version(version: int) -> List[Dict[str, Any]]
```

**Example:**
```python
actions = log.read_version(5)
for action in actions:
    if "add" in action:
        print(f"Added file: {action['add']['path']}")
```

##### `time_travel()`

Get table snapshot at a specific version.

```python
time_travel(version: int) -> List[Action]
```

**Example:**
```python
snapshot_v10 = log.time_travel(10)
print(f"Table had {len(snapshot_v10)} files at version 10")
```

##### `create_checkpoint()`

Create a checkpoint for faster metadata operations.

```python
create_checkpoint() -> None
```

**Example:**
```python
# Create checkpoint after many commits
if log.current_version % 10 == 0:
    log.create_checkpoint()
```

##### `get_active_files()`

Get list of currently active files.

```python
get_active_files() -> List[AddFile]
```

**Example:**
```python
active_files = log.get_active_files()
total_size = sum(f.size for f in active_files)
print(f"Table size: {total_size / (1024**3):.2f} GB")
```

---

## Optimizer API

### Class: `StorageOptimizer`

Optimizes storage layout and file organization.

#### Methods

##### `optimize_table()`

Comprehensive table optimization including compaction and Z-ordering.

```python
optimize_table(
    table_path: str,
    compact: bool = True,
    z_order_columns: Optional[List[str]] = None,
    collect_stats: bool = True
) -> Dict[str, Any]
```

**Returns:** Optimization results and metrics

**Example:**
```python
from lakehouse.optimizer import StorageOptimizer

optimizer = StorageOptimizer(spark)

results = optimizer.optimize_table(
    table_path="/silver/sales",
    compact=True,
    z_order_columns=["customer_id", "product_id"],
    collect_stats=True
)

print(f"Files compacted: {results['files_compacted']}")
print(f"Size reduction: {results['size_reduction_pct']}%")
```

##### `vacuum_table()`

Remove old files no longer referenced by Delta log.

```python
vacuum_table(
    table_path: str,
    retention_hours: int = 168,
    dry_run: bool = True
) -> Dict[str, Any]
```

**Example:**
```python
# Dry run first
dry_run_results = optimizer.vacuum_table(
    table_path="/bronze/events",
    retention_hours=168,
    dry_run=True
)
print(f"Would remove {dry_run_results['files_to_remove']} files")

# Actual vacuum
vacuum_results = optimizer.vacuum_table(
    table_path="/bronze/events",
    retention_hours=168,
    dry_run=False
)
```

### Class: `QueryOptimizer`

Optimizes query performance.

##### `analyze_query_plan()`

Analyze and suggest optimizations for a query.

```python
analyze_query_plan(query_df: DataFrame) -> Dict[str, Any]
```

**Example:**
```python
from lakehouse.optimizer import QueryOptimizer

query_optimizer = QueryOptimizer(spark)

# Complex query
query = spark.sql("""
    SELECT customer_id, SUM(amount) as total
    FROM sales
    WHERE date >= '2024-01-01'
    GROUP BY customer_id
""")

analysis = query_optimizer.analyze_query_plan(query)
print(f"Estimated cost: {analysis['estimated_cost']}")
print(f"Suggestions: {analysis['optimization_suggestions']}")
```

---

## Quality Engine API

### Class: `QualityEngine`

Data quality validation and monitoring.

#### Methods

##### Expectation Methods

Chain multiple expectations for comprehensive validation.

```python
expect_column_to_exist(column: str) -> QualityEngine
expect_column_values_to_not_be_null(column: str) -> QualityEngine
expect_column_values_to_be_unique(column: str) -> QualityEngine
expect_column_values_to_be_between(column: str, min_value: Any, max_value: Any) -> QualityEngine
expect_column_values_to_match_regex(column: str, pattern: str) -> QualityEngine
expect_column_values_to_be_in_set(column: str, value_set: Set[Any]) -> QualityEngine
```

##### `validate()`

Run all expectations against a DataFrame.

```python
validate(df: DataFrame) -> ValidationResult
```

**Example:**
```python
from lakehouse.quality import QualityEngine

quality = QualityEngine()

result = (
    quality
    .expect_column_to_exist("email")
    .expect_column_values_to_not_be_null("customer_id")
    .expect_column_values_to_match_regex("email", r"^[^@]+@[^@]+\.[^@]+$")
    .expect_column_values_to_be_between("age", 0, 120)
    .validate(customer_df)
)

if not result.success:
    print(f"Validation failures: {result.failures}")
    print(f"Failed rows: {result.failed_row_count}")
```

##### `profile_data()`

Generate statistical profile of data.

```python
profile_data(df: DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]
```

**Example:**
```python
profile = quality.profile_data(
    df=sales_df,
    columns=["amount", "quantity", "discount"]
)

for col, stats in profile.items():
    print(f"{col}:")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std Dev: {stats['stddev']}")
    print(f"  Nulls: {stats['null_count']}")
```

##### `detect_anomalies()`

Detect anomalies in data.

```python
detect_anomalies(
    df: DataFrame,
    column: str,
    method: str = "zscore",
    threshold: float = 3.0
) -> List[Dict[str, Any]]
```

**Example:**
```python
anomalies = quality.detect_anomalies(
    df=transactions_df,
    column="amount",
    method="zscore",
    threshold=3.0
)

print(f"Found {len(anomalies)} anomalies")
for anomaly in anomalies[:5]:
    print(f"  Value: {anomaly['value']}, Z-score: {anomaly['zscore']}")
```

---

## Streaming API

### Class: `StreamProcessor`

Handle streaming data pipelines.

#### Methods

##### `start_bronze_stream()`

Start streaming ingestion to bronze layer.

```python
start_bronze_stream(
    source_path: str,
    bronze_path: str,
    checkpoint_path: str,
    source_format: str = "json",
    trigger: str = "10 seconds"
) -> StreamingQuery
```

**Example:**
```python
from lakehouse.streaming import StreamProcessor

stream_processor = StreamProcessor(spark, config)

bronze_stream = stream_processor.start_bronze_stream(
    source_path="/streaming/events",
    bronze_path="/bronze/events",
    checkpoint_path="/checkpoints/bronze_events",
    source_format="json",
    trigger="5 seconds"
)

# Monitor stream
while bronze_stream.isActive:
    print(f"Processed: {bronze_stream.lastProgress}")
    time.sleep(10)
```

##### `start_silver_stream()`

Start streaming transformation to silver layer.

```python
start_silver_stream(
    bronze_path: str,
    silver_path: str,
    checkpoint_path: str,
    transformation_func: Callable[[DataFrame, int], DataFrame],
    trigger: str = "10 seconds"
) -> StreamingQuery
```

**Example:**
```python
def transform_events(batch_df, batch_id):
    return (
        batch_df
        .filter(col("event_type").isin(["click", "purchase"]))
        .withColumn("processed_at", current_timestamp())
    )

silver_stream = stream_processor.start_silver_stream(
    bronze_path="/bronze/events",
    silver_path="/silver/events",
    checkpoint_path="/checkpoints/silver_events",
    transformation_func=transform_events,
    trigger="10 seconds"
)
```

##### `create_windowed_aggregation()`

Create windowed streaming aggregation.

```python
create_windowed_aggregation(
    source_path: str,
    output_path: str,
    checkpoint_path: str,
    window_duration: str,
    watermark_duration: str,
    group_columns: List[str],
    aggregations: Dict[str, str]
) -> StreamingQuery
```

**Example:**
```python
windowed_stream = stream_processor.create_windowed_aggregation(
    source_path="/bronze/clickstream",
    output_path="/silver/clickstream_agg",
    checkpoint_path="/checkpoints/windowed_agg",
    window_duration="5 minutes",
    watermark_duration="10 minutes",
    group_columns=["user_id", "window"],
    aggregations={
        "event_id": "count",
        "duration": "avg"
    }
)
```

---

## Configuration API

### Class: `LakehouseConfig`

Configuration for lakehouse operations.

```python
@dataclass
class LakehouseConfig:
    bronze_path: str
    silver_path: str
    gold_path: str
    checkpoint_path: str = "/checkpoints"
    enable_cdc: bool = False
    vacuum_retention_hours: int = 168
    optimize_interval_hours: int = 24
    default_file_format: str = "delta"
    compression_codec: str = "snappy"
    target_file_size_mb: int = 128
    enable_auto_optimize: bool = True
    enable_auto_compact: bool = True
    schema_enforcement_mode: str = "strict"  # strict, merge, overwrite
```

**Example:**
```python
from lakehouse.config import LakehouseConfig

config = LakehouseConfig(
    bronze_path="s3://my-bucket/bronze",
    silver_path="s3://my-bucket/silver",
    gold_path="s3://my-bucket/gold",
    checkpoint_path="s3://my-bucket/checkpoints",
    enable_cdc=True,
    vacuum_retention_hours=168,
    compression_codec="zstd",
    target_file_size_mb=256,
    schema_enforcement_mode="merge"
)
```

### Class: `DeltaTableConfig`

Configuration for individual Delta tables.

```python
@dataclass
class DeltaTableConfig:
    name: str
    path: str
    layer: Layer
    partition_columns: List[str] = field(default_factory=list)
    z_order_columns: List[str] = field(default_factory=list)
    enable_change_data_feed: bool = False
    auto_compact: bool = True
    optimize_write: bool = True
    target_file_size_mb: int = 128
    log_retention_days: int = 30
    deleted_file_retention_days: int = 7
```

**Example:**
```python
from lakehouse.config import DeltaTableConfig, Layer

table_config = DeltaTableConfig(
    name="fact_sales",
    path="/silver/fact_sales",
    layer=Layer.SILVER,
    partition_columns=["year", "month"],
    z_order_columns=["customer_id", "product_id"],
    enable_change_data_feed=True,
    target_file_size_mb=256
)

# Convert to Delta table properties
properties = table_config.to_table_properties()
```

---

## Error Handling

All API methods follow consistent error handling patterns:

```python
from lakehouse.exceptions import (
    LakehouseException,
    SchemaException,
    ValidationException,
    OptimizationException
)

try:
    processor.bronze_ingestion(
        source_path="/invalid/path",
        bronze_path="/bronze/table"
    )
except LakehouseException as e:
    print(f"Lakehouse error: {e.message}")
    print(f"Error code: {e.error_code}")
    print(f"Details: {e.details}")
```

## Logging

Enable detailed logging for debugging:

```python
import logging

# Configure lakehouse logging
logging.getLogger("lakehouse").setLevel(logging.DEBUG)

# Log to file
handler = logging.FileHandler("lakehouse.log")
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logging.getLogger("lakehouse").addHandler(handler)
```

## Best Practices

1. **Always use schemas** when possible for better performance and reliability
2. **Partition appropriately** - not too fine, not too coarse
3. **Run OPTIMIZE regularly** on frequently queried tables
4. **Use Z-ordering** on columns frequently used in filters
5. **Set up quality checks** at each layer transition
6. **Monitor streaming jobs** and set up alerting for failures
7. **Use checkpoints** for streaming resilience
8. **Implement proper error handling** and retry logic
9. **Profile data regularly** to understand characteristics
10. **Document transformations** for maintainability