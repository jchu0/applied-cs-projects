# Data Lakehouse (Delta Lake / Spark / Flink) - Technical Blueprint

## Executive Summary

This project implements a production-grade data lakehouse architecture combining the flexibility of data lakes with the ACID guarantees and performance of data warehouses. Built on Delta Lake with Spark/Flink compute engines, it demonstrates mastery of modern data platform design, distributed computing, and analytical data modeling.

> **Concepts covered:** [§02 Spark / data processing](../../../02-data-engineering/02-data-processing/spark/) · [§02 Dimensional modeling](../../../02-data-engineering/03-data-warehousing/dimensional-modeling/dimensional-modeling.md) · [§02 dbt transformations](../../../02-data-engineering/03-data-warehousing/dbt/dbt-transformations.md) · [§02 Kafka / Flink streaming](../../../02-data-engineering/04-streaming/). Pairs with [Project 10 (semantic layer on top)](../../10-warehouse-semantic-layer/) and [Project 17 (DuckDB-lite query engine)](../../17-columnar-query-engine/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../../CONCEPT_TO_PROJECT_MAP.md).

**Primary Goals:**
- Build medallion architecture (Bronze → Silver → Gold) with ACID transactions
- Implement unified batch and streaming data processing
- Enable time travel, schema evolution, and data versioning
- Optimize query performance through Z-ordering and partition pruning

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Orchestration Layer (Airflow/Dagster)           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │   Ingestion  │    │  Processing  │    │   Serving    │           │
│  │              │    │              │    │              │           │
│  │  Kafka/CDC   │───▶│  Spark/Flink │───▶│  Query APIs  │           │
│  │  Batch Load  │    │  dbt Models  │    │  Dashboards  │           │
│  └──────────────┘    └──────────────┘    └──────────────┘           │
│          │                   │                   │                   │
│          ▼                   ▼                   ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     Delta Lake Storage                       │    │
│  │  ┌─────────┐     ┌─────────┐     ┌─────────┐                │    │
│  │  │ Bronze  │────▶│ Silver  │────▶│  Gold   │                │    │
│  │  │  (Raw)  │     │(Cleaned)│     │ (Curated)│                │    │
│  │  └─────────┘     └─────────┘     └─────────┘                │    │
│  │                                                              │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  S3/ADLS/GCS Object Storage                          │    │    │
│  │  │  (Parquet files + Delta Transaction Log)             │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                Metadata & Governance (Unity Catalog / HMS)          │
└─────────────────────────────────────────────────────────────────────┘
```

### Medallion Architecture Detail

```
┌─────────────────────────────────────────────────────────────────┐
│                        Bronze Layer (Raw)                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  - Raw data landed as-is from sources                    │    │
│  │  - Full history with append-only pattern                 │    │
│  │  - Metadata: source, ingestion_ts, batch_id              │    │
│  │  - No transformations, schema-on-read                    │    │
│  │  - Partitioned by date/source                            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼ (Cleanse, Dedupe, Validate)
┌─────────────────────────────────────────────────────────────────┐
│                       Silver Layer (Cleaned)                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  - Cleaned, deduplicated, validated data                 │    │
│  │  - Enforced schemas with evolution support               │    │
│  │  - Business keys, SCD Type 2 for dimensions              │    │
│  │  - Conformed data types and formats                      │    │
│  │  - Optimized for joins (Z-ordered on join keys)          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼ (Aggregate, Enrich, Model)
┌─────────────────────────────────────────────────────────────────┐
│                        Gold Layer (Curated)                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  - Business-ready, aggregated datasets                   │    │
│  │  - Star/snowflake schemas for analytics                  │    │
│  │  - Pre-computed metrics and KPIs                         │    │
│  │  - Optimized for query patterns (partitioned, Z-ordered) │    │
│  │  - Documented with business glossary                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Internals

### Delta Lake Transaction Protocol

```python
# Delta Lake Transaction Log Structure
class DeltaLog:
    """
    Delta Lake uses a transaction log (_delta_log/) to provide ACID guarantees.
    Each transaction creates a new JSON file (00000000000000000000.json).
    """

    def __init__(self, table_path: str):
        self.table_path = table_path
        self.log_path = f"{table_path}/_delta_log"
        self.snapshot = None

    def get_snapshot(self, version: Optional[int] = None) -> 'Snapshot':
        """
        Read the current state of the table.
        Uses checkpoints (parquet) for efficiency with log replay.
        """
        # Find latest checkpoint
        checkpoint_version = self._find_latest_checkpoint()

        if checkpoint_version:
            # Read checkpoint
            state = self._read_checkpoint(checkpoint_version)
            start_version = checkpoint_version + 1
        else:
            state = TableState()
            start_version = 0

        # Replay logs since checkpoint
        target = version or self._get_latest_version()
        for v in range(start_version, target + 1):
            actions = self._read_log_file(v)
            state = state.apply_actions(actions)

        return Snapshot(version=target, state=state)

    def commit(self, actions: List['Action']) -> int:
        """
        Atomically commit a set of actions.
        Uses optimistic concurrency control.
        """
        # Retry loop for concurrent commits
        for attempt in range(MAX_RETRIES):
            try:
                version = self._get_latest_version() + 1
                log_file = f"{self.log_path}/{version:020d}.json"

                # Atomic write (put-if-absent semantics)
                self._atomic_write(log_file, actions)
                return version

            except ConcurrentWriteException:
                # Conflict detected, retry with new version
                continue

        raise CommitFailedException("Max retries exceeded")

class Action:
    """Base class for Delta Lake actions"""
    pass

class AddFile(Action):
    path: str
    partition_values: Dict[str, str]
    size: int
    modification_time: int
    data_change: bool
    stats: Optional[str]  # JSON column statistics
    tags: Optional[Dict[str, str]]

class RemoveFile(Action):
    path: str
    deletion_timestamp: int
    data_change: bool
    extended_file_metadata: bool
    partition_values: Dict[str, str]

class Metadata(Action):
    id: str
    name: str
    description: str
    format: Format
    schema_string: str
    partition_columns: List[str]
    configuration: Dict[str, str]

class Protocol(Action):
    min_reader_version: int
    min_writer_version: int
```

### Spark Processing Engine

```python
from pyspark.sql import SparkSession
from delta import DeltaTable
from pyspark.sql.functions import *
from pyspark.sql.types import *

class LakehouseProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def bronze_ingestion(
        self,
        source_path: str,
        bronze_path: str,
        source_name: str,
        schema: StructType = None
    ):
        """
        Ingest raw data into bronze layer with metadata.
        """
        # Read raw data (schema inference or provided)
        if schema:
            df = self.spark.read.schema(schema).json(source_path)
        else:
            df = self.spark.read.json(source_path)

        # Add bronze metadata columns
        df_bronze = df \
            .withColumn("_source", lit(source_name)) \
            .withColumn("_ingestion_ts", current_timestamp()) \
            .withColumn("_batch_id", lit(str(uuid.uuid4()))) \
            .withColumn("_input_file", input_file_name())

        # Write to bronze (append-only)
        df_bronze.write \
            .format("delta") \
            .mode("append") \
            .partitionBy("_ingestion_date") \
            .save(bronze_path)

    def bronze_to_silver(
        self,
        bronze_path: str,
        silver_path: str,
        dedup_keys: List[str],
        watermark_col: str,
        transformations: List[Callable]
    ):
        """
        Transform bronze to silver with cleaning, deduplication, and validation.
        """
        # Read bronze data (incremental based on watermark)
        bronze_df = self.spark.read.format("delta").load(bronze_path)

        # Get watermark for incremental processing
        if DeltaTable.isDeltaTable(self.spark, silver_path):
            silver_table = DeltaTable.forPath(self.spark, silver_path)
            max_watermark = silver_table.toDF() \
                .agg(max(watermark_col)) \
                .collect()[0][0]
            bronze_df = bronze_df.filter(col(watermark_col) > max_watermark)

        # Apply transformations
        df = bronze_df
        for transform in transformations:
            df = transform(df)

        # Deduplicate (keep latest by watermark)
        window = Window.partitionBy(dedup_keys).orderBy(col(watermark_col).desc())
        df_deduped = df \
            .withColumn("_row_num", row_number().over(window)) \
            .filter(col("_row_num") == 1) \
            .drop("_row_num")

        # Validate data quality
        df_validated = self._apply_quality_checks(df_deduped)

        # Merge into silver (upsert)
        if DeltaTable.isDeltaTable(self.spark, silver_path):
            silver_table = DeltaTable.forPath(self.spark, silver_path)

            merge_condition = " AND ".join([
                f"target.{key} = source.{key}" for key in dedup_keys
            ])

            silver_table.alias("target").merge(
                df_validated.alias("source"),
                merge_condition
            ).whenMatchedUpdateAll() \
             .whenNotMatchedInsertAll() \
             .execute()
        else:
            # Initial load
            df_validated.write \
                .format("delta") \
                .mode("overwrite") \
                .save(silver_path)

    def silver_to_gold(
        self,
        silver_tables: Dict[str, str],
        gold_path: str,
        aggregation_query: str,
        z_order_cols: List[str] = None
    ):
        """
        Build gold layer aggregations from silver tables.
        """
        # Register silver tables as temp views
        for name, path in silver_tables.items():
            self.spark.read.format("delta").load(path).createOrReplaceTempView(name)

        # Execute aggregation query
        gold_df = self.spark.sql(aggregation_query)

        # Write to gold
        gold_df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(gold_path)

        # Optimize with Z-ordering
        if z_order_cols:
            gold_table = DeltaTable.forPath(self.spark, gold_path)
            gold_table.optimize().zOrderBy(z_order_cols).executeCompaction()

    def _apply_quality_checks(self, df: DataFrame) -> DataFrame:
        """Apply data quality constraints"""
        # This would integrate with Great Expectations or similar
        return df
```

### Flink Streaming Engine

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.table.expressions import col, lit
from pyflink.table.window import Tumble

class StreamingLakehouse:
    def __init__(self):
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.env.enable_checkpointing(60000)  # 1 minute checkpoints

        settings = EnvironmentSettings.new_instance() \
            .in_streaming_mode() \
            .build()
        self.t_env = StreamTableEnvironment.create(self.env, settings)

        # Configure Delta Lake connector
        self._configure_delta()

    def _configure_delta(self):
        """Configure Flink for Delta Lake"""
        self.t_env.execute_sql("""
            CREATE CATALOG delta_catalog WITH (
                'type' = 'delta-catalog',
                'catalog-type' = 'in-memory'
            )
        """)

    def streaming_bronze_ingestion(
        self,
        kafka_config: dict,
        bronze_path: str
    ):
        """
        Stream data from Kafka to Bronze layer.
        """
        # Create Kafka source
        self.t_env.execute_sql(f"""
            CREATE TABLE kafka_source (
                event_id STRING,
                event_type STRING,
                payload STRING,
                event_time TIMESTAMP(3),
                WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
            ) WITH (
                'connector' = 'kafka',
                'topic' = '{kafka_config['topic']}',
                'properties.bootstrap.servers' = '{kafka_config['bootstrap_servers']}',
                'properties.group.id' = '{kafka_config['group_id']}',
                'scan.startup.mode' = 'latest-offset',
                'format' = 'json'
            )
        """)

        # Create Delta sink
        self.t_env.execute_sql(f"""
            CREATE TABLE bronze_sink (
                event_id STRING,
                event_type STRING,
                payload STRING,
                event_time TIMESTAMP(3),
                _ingestion_ts TIMESTAMP(3),
                _partition_date STRING
            ) PARTITIONED BY (_partition_date)
            WITH (
                'connector' = 'delta',
                'table-path' = '{bronze_path}'
            )
        """)

        # Insert with metadata
        self.t_env.execute_sql("""
            INSERT INTO bronze_sink
            SELECT
                event_id,
                event_type,
                payload,
                event_time,
                CURRENT_TIMESTAMP as _ingestion_ts,
                DATE_FORMAT(event_time, 'yyyy-MM-dd') as _partition_date
            FROM kafka_source
        """)

    def streaming_aggregations(
        self,
        silver_path: str,
        gold_path: str,
        window_size: str = "1 HOUR"
    ):
        """
        Continuous aggregations from Silver to Gold.
        """
        # Read from Silver as streaming table
        self.t_env.execute_sql(f"""
            CREATE TABLE silver_source (
                user_id STRING,
                event_type STRING,
                amount DECIMAL(10, 2),
                event_time TIMESTAMP(3),
                WATERMARK FOR event_time AS event_time - INTERVAL '10' SECOND
            ) WITH (
                'connector' = 'delta',
                'table-path' = '{silver_path}',
                'mode' = 'streaming'
            )
        """)

        # Windowed aggregation
        result = self.t_env.sql_query(f"""
            SELECT
                user_id,
                TUMBLE_START(event_time, INTERVAL '{window_size}') as window_start,
                TUMBLE_END(event_time, INTERVAL '{window_size}') as window_end,
                COUNT(*) as event_count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM silver_source
            GROUP BY
                user_id,
                TUMBLE(event_time, INTERVAL '{window_size}')
        """)

        # Write to Gold
        self.t_env.execute_sql(f"""
            CREATE TABLE gold_sink (
                user_id STRING,
                window_start TIMESTAMP(3),
                window_end TIMESTAMP(3),
                event_count BIGINT,
                total_amount DECIMAL(10, 2),
                avg_amount DECIMAL(10, 2)
            ) WITH (
                'connector' = 'delta',
                'table-path' = '{gold_path}'
            )
        """)

        result.execute_insert("gold_sink")
```

---

## Data Structures

### Table Configuration

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class Layer(Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"

@dataclass
class DeltaTableConfig:
    """Configuration for a Delta Lake table"""
    name: str
    path: str
    layer: Layer
    schema: StructType
    partition_columns: List[str]
    z_order_columns: List[str]
    description: str

    # Delta-specific configurations
    enable_change_data_feed: bool = False
    auto_compact: bool = True
    optimize_write: bool = True

    # Retention settings
    log_retention_days: int = 30
    deleted_file_retention_hours: int = 168  # 7 days

    # Quality constraints
    constraints: Dict[str, str] = None  # CHECK constraints

    def to_table_properties(self) -> Dict[str, str]:
        return {
            "delta.enableChangeDataFeed": str(self.enable_change_data_feed).lower(),
            "delta.autoOptimize.autoCompact": str(self.auto_compact).lower(),
            "delta.autoOptimize.optimizeWrite": str(self.optimize_write).lower(),
            "delta.logRetentionDuration": f"interval {self.log_retention_days} days",
            "delta.deletedFileRetentionDuration": f"interval {self.deleted_file_retention_hours} hours",
        }

@dataclass
class IncrementalConfig:
    """Configuration for incremental processing"""
    watermark_column: str
    merge_keys: List[str]
    scd_type: int = 1  # 1 for Type 1 (overwrite), 2 for Type 2 (history)

    # For SCD Type 2
    effective_date_col: str = "effective_date"
    end_date_col: str = "end_date"
    current_flag_col: str = "is_current"

@dataclass
class QualityRule:
    """Data quality rule definition"""
    name: str
    column: str
    expectation_type: str  # e.g., "not_null", "unique", "in_set"
    expectation_kwargs: Dict
    severity: str = "error"  # error, warning

@dataclass
class TableLineage:
    """Track table dependencies"""
    table_name: str
    upstream_tables: List[str]
    downstream_tables: List[str]
    transformation_query: str
    refresh_schedule: str
```

### Column Statistics

```python
@dataclass
class ColumnStats:
    """Column-level statistics for query optimization"""
    column_name: str
    data_type: str
    num_nulls: int
    min_value: any
    max_value: any
    num_distinct: int = None
    avg_length: float = None

    # For numeric columns
    mean: float = None
    stddev: float = None

    # For string columns
    top_k_values: List[tuple] = None  # (value, count)

@dataclass
class FileStats:
    """File-level statistics stored in Delta Log"""
    path: str
    partition_values: Dict[str, str]
    size_bytes: int
    num_records: int
    column_stats: Dict[str, ColumnStats]
```

---

## API Design

### Python SDK

```python
class LakehouseClient:
    """High-level client for Lakehouse operations"""

    def __init__(self, config: LakehouseConfig):
        self.spark = self._init_spark(config)
        self.catalog = Catalog(config.metastore_uri)
        self.quality_engine = QualityEngine(config.great_expectations_root)

    # Table Management
    def create_table(self, config: DeltaTableConfig) -> None:
        """Create a new Delta table with configuration"""
        pass

    def get_table(self, table_name: str) -> DeltaTable:
        """Get a Delta table by name"""
        pass

    def drop_table(self, table_name: str, purge: bool = False) -> None:
        """Drop a table, optionally purging files"""
        pass

    # Time Travel
    def read_version(self, table_name: str, version: int) -> DataFrame:
        """Read a specific version of a table"""
        return self.spark.read.format("delta") \
            .option("versionAsOf", version) \
            .table(table_name)

    def read_timestamp(self, table_name: str, timestamp: str) -> DataFrame:
        """Read table as of a specific timestamp"""
        return self.spark.read.format("delta") \
            .option("timestampAsOf", timestamp) \
            .table(table_name)

    def restore_version(self, table_name: str, version: int) -> None:
        """Restore a table to a previous version"""
        table = DeltaTable.forName(self.spark, table_name)
        table.restoreToVersion(version)

    # Optimization
    def optimize(
        self,
        table_name: str,
        partition_filter: str = None,
        z_order_by: List[str] = None
    ) -> Dict:
        """Optimize table files with optional Z-ordering"""
        table = DeltaTable.forName(self.spark, table_name)

        optimize_builder = table.optimize()

        if partition_filter:
            optimize_builder = optimize_builder.where(partition_filter)

        if z_order_by:
            metrics = optimize_builder.zOrderBy(z_order_by).executeCompaction()
        else:
            metrics = optimize_builder.executeCompaction()

        return metrics

    def vacuum(self, table_name: str, retention_hours: int = 168) -> None:
        """Remove old files not in current table state"""
        table = DeltaTable.forName(self.spark, table_name)
        table.vacuum(retention_hours)

    # Change Data Feed
    def read_changes(
        self,
        table_name: str,
        start_version: int = None,
        end_version: int = None,
        start_timestamp: str = None,
        end_timestamp: str = None
    ) -> DataFrame:
        """Read change data feed for a table"""
        reader = self.spark.read.format("delta") \
            .option("readChangeFeed", "true")

        if start_version:
            reader = reader.option("startingVersion", start_version)
        if end_version:
            reader = reader.option("endingVersion", end_version)
        if start_timestamp:
            reader = reader.option("startingTimestamp", start_timestamp)
        if end_timestamp:
            reader = reader.option("endingTimestamp", end_timestamp)

        return reader.table(table_name)

    # Data Quality
    def validate(
        self,
        df: DataFrame,
        expectation_suite: str
    ) -> ValidationResult:
        """Validate DataFrame against expectation suite"""
        return self.quality_engine.validate(df, expectation_suite)

    # Lineage
    def get_lineage(self, table_name: str) -> TableLineage:
        """Get upstream and downstream dependencies"""
        pass

    # History
    def get_history(
        self,
        table_name: str,
        limit: int = None
    ) -> DataFrame:
        """Get table history"""
        table = DeltaTable.forName(self.spark, table_name)
        return table.history(limit)
```

### REST API

```yaml
openapi: 3.0.0
info:
  title: Lakehouse API
  version: 1.0.0

paths:
  /tables:
    get:
      summary: List all tables
      parameters:
        - name: layer
          in: query
          schema:
            type: string
            enum: [bronze, silver, gold]
    post:
      summary: Create a new table
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TableConfig'

  /tables/{table_name}:
    get:
      summary: Get table metadata
    delete:
      summary: Drop table

  /tables/{table_name}/query:
    post:
      summary: Query table data
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                sql:
                  type: string
                version:
                  type: integer
                timestamp:
                  type: string

  /tables/{table_name}/history:
    get:
      summary: Get table history
      parameters:
        - name: limit
          in: query
          schema:
            type: integer

  /tables/{table_name}/optimize:
    post:
      summary: Optimize table
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                partition_filter:
                  type: string
                z_order_columns:
                  type: array
                  items:
                    type: string

  /tables/{table_name}/vacuum:
    post:
      summary: Vacuum old files
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                retention_hours:
                  type: integer
                  default: 168

  /tables/{table_name}/changes:
    get:
      summary: Read change data feed
      parameters:
        - name: start_version
          in: query
          schema:
            type: integer
        - name: end_version
          in: query
          schema:
            type: integer
```

---

## Enterprise Features

### 1. Workflow Orchestration (Airflow Integration)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-platform',
    'depends_on_past': True,
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'lakehouse_daily_etl',
    default_args=default_args,
    schedule_interval='0 6 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['lakehouse', 'production'],
) as dag:

    # Bronze ingestion tasks
    ingest_events = SparkSubmitOperator(
        task_id='ingest_events_to_bronze',
        application='/opt/spark/jobs/bronze_ingestion.py',
        conf={
            'spark.sql.extensions': 'io.delta.sql.DeltaSparkSessionExtension',
            'spark.sql.catalog.spark_catalog': 'org.apache.spark.sql.delta.catalog.DeltaCatalog',
        },
        application_args=[
            '--source', 's3://raw-data/events/{{ ds }}/',
            '--target', 's3://lakehouse/bronze/events/',
        ],
    )

    # Silver transformation tasks
    transform_events = SparkSubmitOperator(
        task_id='transform_events_to_silver',
        application='/opt/spark/jobs/silver_transformation.py',
        application_args=[
            '--bronze-path', 's3://lakehouse/bronze/events/',
            '--silver-path', 's3://lakehouse/silver/events/',
            '--date', '{{ ds }}',
        ],
    )

    # Data quality check
    quality_check = PythonOperator(
        task_id='quality_check_silver',
        python_callable=run_quality_checks,
        op_kwargs={
            'table': 'silver.events',
            'suite': 'events_quality_suite',
        },
    )

    # Gold aggregation
    build_gold = SparkSubmitOperator(
        task_id='build_gold_aggregations',
        application='/opt/spark/jobs/gold_aggregation.py',
        application_args=[
            '--silver-tables', 'events,users,products',
            '--gold-path', 's3://lakehouse/gold/user_metrics/',
        ],
    )

    # Optimization
    optimize_tables = PythonOperator(
        task_id='optimize_delta_tables',
        python_callable=optimize_lakehouse_tables,
        op_kwargs={
            'tables': ['silver.events', 'gold.user_metrics'],
        },
    )

    # Dependencies
    ingest_events >> transform_events >> quality_check >> build_gold >> optimize_tables
```

### 2. Data Quality (Great Expectations)

```python
import great_expectations as gx
from great_expectations.core.batch import BatchRequest
from great_expectations.checkpoint import SimpleCheckpoint

class LakehouseQualityEngine:
    def __init__(self, ge_root: str):
        self.context = gx.get_context(context_root_dir=ge_root)

    def create_expectation_suite(
        self,
        suite_name: str,
        expectations: List[dict]
    ):
        """Create an expectation suite programmatically"""
        suite = self.context.create_expectation_suite(suite_name)

        for exp in expectations:
            suite.add_expectation(
                expectation_configuration=gx.core.ExpectationConfiguration(
                    expectation_type=exp['type'],
                    kwargs=exp['kwargs'],
                    meta=exp.get('meta', {})
                )
            )

        self.context.save_expectation_suite(suite)

    def validate_delta_table(
        self,
        table_path: str,
        suite_name: str
    ) -> ValidationResult:
        """Validate a Delta table against an expectation suite"""

        # Create batch request for Delta table
        batch_request = BatchRequest(
            datasource_name="lakehouse",
            data_connector_name="delta_connector",
            data_asset_name=table_path,
        )

        # Run checkpoint
        checkpoint = SimpleCheckpoint(
            name=f"validate_{suite_name}",
            data_context=self.context,
            validations=[{
                "batch_request": batch_request,
                "expectation_suite_name": suite_name,
            }]
        )

        result = checkpoint.run()

        return result

# Example expectation suite for Silver events
events_expectations = [
    {
        "type": "expect_column_to_exist",
        "kwargs": {"column": "event_id"}
    },
    {
        "type": "expect_column_values_to_not_be_null",
        "kwargs": {"column": "event_id"}
    },
    {
        "type": "expect_column_values_to_be_unique",
        "kwargs": {"column": "event_id"}
    },
    {
        "type": "expect_column_values_to_be_in_set",
        "kwargs": {
            "column": "event_type",
            "value_set": ["click", "view", "purchase", "signup"]
        }
    },
    {
        "type": "expect_column_values_to_be_between",
        "kwargs": {
            "column": "amount",
            "min_value": 0,
            "max_value": 100000
        }
    },
    {
        "type": "expect_table_row_count_to_be_between",
        "kwargs": {
            "min_value": 1000,
            "max_value": 10000000
        }
    },
]
```

### 3. Cost Optimization

```python
class LakehouseCostOptimizer:
    """Optimize storage and compute costs"""

    def __init__(self, client: LakehouseClient):
        self.client = client

    def analyze_storage_costs(self, table_name: str) -> StorageReport:
        """Analyze storage efficiency of a table"""
        table = self.client.get_table(table_name)

        # Get file statistics
        files = table.toDF().select("_metadata.file_path", "_metadata.file_size")

        total_size = files.agg(sum("file_size")).collect()[0][0]
        file_count = files.count()
        avg_file_size = total_size / file_count if file_count > 0 else 0

        # Get version history size
        history = table.history()
        versions = history.count()

        # Identify small files (< 128MB)
        small_files = files.filter(col("file_size") < 128 * 1024 * 1024).count()

        return StorageReport(
            total_size_bytes=total_size,
            file_count=file_count,
            avg_file_size_bytes=avg_file_size,
            small_file_count=small_files,
            version_count=versions,
            recommendations=self._generate_recommendations(
                avg_file_size, small_files, versions
            )
        )

    def optimize_storage(
        self,
        table_name: str,
        target_file_size_mb: int = 1024
    ):
        """Optimize storage by compacting files"""
        table = self.client.get_table(table_name)

        # Enable optimized writes for future
        self.client.spark.sql(f"""
            ALTER TABLE {table_name}
            SET TBLPROPERTIES (
                'delta.autoOptimize.optimizeWrite' = 'true',
                'delta.autoOptimize.autoCompact' = 'true',
                'delta.targetFileSize' = '{target_file_size_mb * 1024 * 1024}'
            )
        """)

        # Compact existing files
        table.optimize().executeCompaction()

    def setup_lifecycle_policies(
        self,
        table_name: str,
        log_retention_days: int = 30,
        vacuum_retention_hours: int = 168
    ):
        """Configure lifecycle policies for automatic cleanup"""
        self.client.spark.sql(f"""
            ALTER TABLE {table_name}
            SET TBLPROPERTIES (
                'delta.logRetentionDuration' = 'interval {log_retention_days} days',
                'delta.deletedFileRetentionDuration' = 'interval {vacuum_retention_hours} hours'
            )
        """)

    def schedule_vacuum(self, table_name: str, retention_hours: int = 168):
        """Vacuum table to remove old files"""
        table = self.client.get_table(table_name)
        table.vacuum(retention_hours)
```

---

## Performance Considerations

### Z-Order Optimization

```python
def apply_z_order(
    table: DeltaTable,
    columns: List[str],
    partition_filter: str = None
):
    """
    Apply Z-ordering for multi-dimensional clustering.
    Improves query performance for filters on multiple columns.
    """
    optimizer = table.optimize()

    if partition_filter:
        optimizer = optimizer.where(partition_filter)

    # Z-order interleaves bits from multiple columns
    # to preserve locality in multi-dimensional space
    metrics = optimizer.zOrderBy(columns).executeCompaction()

    return metrics

# Best practices:
# - Z-order on columns frequently used together in filters
# - Limit to 3-4 columns (diminishing returns beyond)
# - High-cardinality columns benefit most
# - Re-run after significant data changes
```

### Partition Pruning

```python
def design_partition_strategy(
    table_name: str,
    query_patterns: List[str],
    data_characteristics: dict
) -> List[str]:
    """
    Design optimal partition strategy based on query patterns.
    """
    recommendations = []

    # Analyze query patterns for common filters
    filter_columns = extract_filter_columns(query_patterns)

    # Check cardinality
    for col in filter_columns:
        cardinality = data_characteristics.get(f"{col}_cardinality", 0)

        # Ideal: 10K-100K files after partitioning
        if cardinality < 1000:
            recommendations.append(col)
        elif cardinality < 10000:
            # Consider composite partitioning
            recommendations.append(f"date_trunc('month', {col})")

    return recommendations

# Partition pruning in action
def query_with_pruning(spark, table_path: str, date: str, region: str):
    """
    Query that leverages partition pruning.
    Only reads files matching partition predicates.
    """
    return spark.read.format("delta") \
        .load(table_path) \
        .filter(f"date = '{date}' AND region = '{region}'")  # Pruned!
```

### Compaction Strategy

```python
class CompactionManager:
    """Manage file compaction for optimal performance"""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def auto_compact_strategy(
        self,
        table_path: str,
        target_size_mb: int = 1024
    ):
        """
        Implement auto-compaction strategy.
        Triggers when:
        - Too many small files in a partition
        - Average file size drops below threshold
        """
        table = DeltaTable.forPath(self.spark, table_path)

        # Analyze current state
        files_df = self.spark.sql(f"""
            DESCRIBE DETAIL delta.`{table_path}`
        """)

        num_files = files_df.select("numFiles").collect()[0][0]
        size_bytes = files_df.select("sizeInBytes").collect()[0][0]

        avg_size_mb = (size_bytes / num_files) / (1024 * 1024)

        # Compact if average file size is less than 50% of target
        if avg_size_mb < target_size_mb * 0.5:
            table.optimize().executeCompaction()
            return True

        return False
```

---

## Stretch Goals

### 1. Auto-Compaction Service

```python
class AutoCompactionService:
    """Background service for automatic table compaction"""

    def __init__(self, config: CompactionConfig):
        self.config = config
        self.scheduler = BackgroundScheduler()

    def start(self):
        """Start the auto-compaction service"""
        self.scheduler.add_job(
            self._compaction_job,
            'interval',
            hours=self.config.check_interval_hours,
            id='auto_compaction'
        )
        self.scheduler.start()

    def _compaction_job(self):
        """Check and compact tables that need it"""
        for table in self.config.managed_tables:
            if self._needs_compaction(table):
                self._compact_table(table)

    def _needs_compaction(self, table: str) -> bool:
        """Check if table needs compaction based on heuristics"""
        metrics = self._get_table_metrics(table)

        # Trigger if:
        # 1. More than N small files
        # 2. Average file size below threshold
        # 3. More than N versions since last compaction
        return (
            metrics.small_file_count > self.config.max_small_files or
            metrics.avg_file_size_mb < self.config.min_file_size_mb or
            metrics.versions_since_compact > self.config.max_versions
        )
```

### 2. Change Data Feed (CDF) Tracking

```python
class ChangeDataFeedProcessor:
    """Process and propagate change data feeds"""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def enable_cdf(self, table_name: str):
        """Enable change data feed on a table"""
        self.spark.sql(f"""
            ALTER TABLE {table_name}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

    def read_changes(
        self,
        table_name: str,
        start_version: int,
        end_version: int = None
    ) -> DataFrame:
        """Read changes between versions"""
        reader = self.spark.read.format("delta") \
            .option("readChangeFeed", "true") \
            .option("startingVersion", start_version)

        if end_version:
            reader = reader.option("endingVersion", end_version)

        return reader.table(table_name)

    def propagate_changes(
        self,
        source_table: str,
        target_table: str,
        transform_fn: Callable[[DataFrame], DataFrame] = None
    ):
        """Propagate changes from source to target table"""
        # Get last processed version
        last_version = self._get_last_processed_version(source_table, target_table)

        # Read changes
        changes = self.read_changes(source_table, last_version + 1)

        if changes.count() == 0:
            return

        # Apply transformation
        if transform_fn:
            changes = transform_fn(changes)

        # Process by change type
        inserts = changes.filter("_change_type = 'insert'").drop("_change_type", "_commit_version", "_commit_timestamp")
        updates = changes.filter("_change_type = 'update_postimage'").drop("_change_type", "_commit_version", "_commit_timestamp")
        deletes = changes.filter("_change_type = 'delete'")

        # Apply to target
        target = DeltaTable.forName(self.spark, target_table)
        # ... merge logic
```

### 3. Streaming Joins

```python
class StreamingJoinProcessor:
    """Handle streaming joins with state management"""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def stream_stream_join(
        self,
        left_stream_path: str,
        right_stream_path: str,
        join_key: str,
        watermark_delay: str,
        output_path: str
    ):
        """Join two streams with watermarking"""
        # Read left stream with watermark
        left = self.spark.readStream \
            .format("delta") \
            .load(left_stream_path) \
            .withWatermark("event_time", watermark_delay)

        # Read right stream with watermark
        right = self.spark.readStream \
            .format("delta") \
            .load(right_stream_path) \
            .withWatermark("event_time", watermark_delay)

        # Join with time constraint
        joined = left.join(
            right,
            on=join_key,
            how="inner"
        ).filter(
            abs(left.event_time - right.event_time) < expr("interval 1 hour")
        )

        # Write to output
        query = joined.writeStream \
            .format("delta") \
            .outputMode("append") \
            .option("checkpointLocation", f"{output_path}/_checkpoint") \
            .start(output_path)

        return query
```

---

## Testing Strategy

### Unit Tests

```python
import pytest
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

@pytest.fixture(scope="session")
def spark():
    builder = SparkSession.builder \
        .appName("lakehouse-tests") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .master("local[*]")

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    yield spark
    spark.stop()

class TestBronzeIngestion:
    def test_adds_metadata_columns(self, spark, tmp_path):
        # Create test data
        df = spark.createDataFrame([
            {"id": 1, "value": "test"}
        ])

        # Ingest
        processor = LakehouseProcessor(spark)
        bronze_path = str(tmp_path / "bronze")
        processor.bronze_ingestion(df, bronze_path, "test_source")

        # Verify
        result = spark.read.format("delta").load(bronze_path)
        assert "_source" in result.columns
        assert "_ingestion_ts" in result.columns
        assert result.filter("_source = 'test_source'").count() == 1

class TestSilverTransformation:
    def test_deduplication(self, spark, tmp_path):
        # Create bronze with duplicates
        bronze_df = spark.createDataFrame([
            {"id": 1, "value": "old", "ts": "2024-01-01"},
            {"id": 1, "value": "new", "ts": "2024-01-02"},
        ])

        # Transform
        processor = LakehouseProcessor(spark)
        silver_path = str(tmp_path / "silver")
        processor.bronze_to_silver(
            bronze_df,
            silver_path,
            dedup_keys=["id"],
            watermark_col="ts"
        )

        # Verify
        result = spark.read.format("delta").load(silver_path)
        assert result.count() == 1
        assert result.collect()[0]["value"] == "new"
```

### Integration Tests

```python
class TestEndToEndPipeline:
    def test_bronze_silver_gold_flow(self, spark, tmp_path):
        """Test complete medallion architecture flow"""
        # Setup paths
        bronze = str(tmp_path / "bronze")
        silver = str(tmp_path / "silver")
        gold = str(tmp_path / "gold")

        processor = LakehouseProcessor(spark)

        # Bronze ingestion
        raw_data = spark.createDataFrame([
            {"user_id": 1, "event": "click", "amount": 10.0, "ts": "2024-01-01"},
            {"user_id": 1, "event": "purchase", "amount": 50.0, "ts": "2024-01-02"},
            {"user_id": 2, "event": "click", "amount": 5.0, "ts": "2024-01-01"},
        ])
        processor.bronze_ingestion(raw_data, bronze, "events")

        # Silver transformation
        processor.bronze_to_silver(
            bronze, silver,
            dedup_keys=["user_id", "ts"],
            watermark_col="ts",
            transformations=[clean_data, validate_schema]
        )

        # Gold aggregation
        processor.silver_to_gold(
            {"events": silver},
            gold,
            """
            SELECT
                user_id,
                COUNT(*) as event_count,
                SUM(amount) as total_amount
            FROM events
            GROUP BY user_id
            """
        )

        # Verify gold
        gold_df = spark.read.format("delta").load(gold)
        assert gold_df.count() == 2

        user_1 = gold_df.filter("user_id = 1").collect()[0]
        assert user_1["event_count"] == 2
        assert user_1["total_amount"] == 60.0
```

### Performance Tests

```python
class TestPerformance:
    def test_query_with_z_order(self, spark, large_dataset):
        """Verify Z-ordering improves query performance"""
        # Create table without Z-order
        unoptimized_path = "/tmp/unoptimized"
        large_dataset.write.format("delta").save(unoptimized_path)

        # Create table with Z-order
        optimized_path = "/tmp/optimized"
        large_dataset.write.format("delta").save(optimized_path)
        DeltaTable.forPath(spark, optimized_path) \
            .optimize() \
            .zOrderBy(["user_id", "event_date"]) \
            .executeCompaction()

        # Benchmark query
        query = "SELECT * FROM table WHERE user_id = 12345 AND event_date = '2024-01-15'"

        unoptimized_time = benchmark_query(spark, unoptimized_path, query)
        optimized_time = benchmark_query(spark, optimized_path, query)

        # Z-ordered should be at least 2x faster
        assert optimized_time < unoptimized_time * 0.5
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- Set up Spark cluster with Delta Lake
- Implement basic medallion architecture
- Bronze ingestion for batch sources
- Simple Silver transformations

### Phase 2: Core Features (Weeks 3-4)
- Schema enforcement and evolution
- Incremental processing with watermarks
- Time travel implementation
- Basic data quality checks

### Phase 3: Optimization (Weeks 5-6)
- Z-ordering implementation
- Partition pruning strategies
- Auto-compaction
- Vacuum scheduling

### Phase 4: Enterprise Features (Weeks 7-8)
- Airflow/Dagster orchestration
- Great Expectations integration
- Monitoring and alerting
- Cost optimization tools

### Phase 5: Advanced Features (Weeks 9-10)
- Streaming ingestion with Flink
- Change data feed
- Streaming joins
- Multi-cluster deployment

---

## References

- [Delta Lake Documentation](https://docs.delta.io/)
- [Delta Lake Protocol](https://github.com/delta-io/delta/blob/master/PROTOCOL.md)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Apache Flink Documentation](https://flink.apache.org/docs/)
- [Great Expectations](https://docs.greatexpectations.io/)
- [Databricks Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)
