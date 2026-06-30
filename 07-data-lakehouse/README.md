# Data Lakehouse

> **Concepts covered:** §02 data-engineering — `01-data-pipelines`, `02-data-processing`, `06-infrastructure`

A production-grade data lakehouse implementation combining the best of data lakes and data warehouses. Built on Delta Lake principles with medallion architecture, ACID transactions, and comprehensive data quality validation.

## Overview

The Data Lakehouse provides a unified platform for both analytical and operational workloads, eliminating the need for separate systems. It offers the flexibility and cost-efficiency of data lakes with the performance and reliability of data warehouses.

### Key Benefits
- **Unified Platform**: Single system for all data workloads
- **Cost Effective**: Open formats on object storage
- **High Performance**: Optimized for both batch and streaming
- **ACID Compliance**: Full transaction support
- **Schema Evolution**: Adapt to changing requirements
- **Time Travel**: Query historical data versions

## Features

### Core Capabilities
- **Medallion Architecture**: Progressive data refinement (Bronze → Silver → Gold)
- **ACID Transactions**: Full ACID compliance with Delta Lake
- **Time Travel**: Query and restore historical versions
- **Schema Evolution**: Flexible schema changes without downtime
- **Data Quality**: Built-in validation and anomaly detection
- **Performance Optimization**: Z-ordering, compaction, and caching
- **Streaming Support**: Real-time data processing
- **Multi-Format Support**: JSON, CSV, Parquet, Avro, XML

### Enterprise Features
- **Multi-Tenancy**: Isolated environments for different teams
- **Access Control**: Fine-grained permissions
- **Audit Logging**: Complete operation history
- **Data Lineage**: Track data flow and transformations
- **Monitoring**: Comprehensive metrics and alerts
- **High Availability**: Fault-tolerant architecture

## Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.8+
- Java 8 or 11
- 8GB+ RAM
- 100GB+ storage

# Install dependencies
pip install pyspark>=3.3.0 delta-spark>=2.3.0 pandas pyarrow
```

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/data-lakehouse.git
cd data-lakehouse

# Install package
pip install -e .

# Run tests to verify installation
pytest tests/unit -v
```

### Basic Usage

```python
from pyspark.sql import SparkSession
from lakehouse.processor import LakehouseProcessor
from lakehouse.config import LakehouseConfig

# Initialize Spark with Delta Lake
spark = SparkSession.builder \
    .appName("lakehouse-quickstart") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Configure lakehouse
config = LakehouseConfig(
    bronze_path="./data/bronze",
    silver_path="./data/silver",
    gold_path="./data/gold"
)

# Create processor
processor = LakehouseProcessor(spark, config)

# Ingest raw data to bronze
processor.bronze_ingestion(
    source_path="./raw_data/*.json",
    bronze_path=config.bronze_path + "/events",
    source_name="event_stream"
)

print("Data successfully ingested to bronze layer!")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                            │
│  (Files, APIs, Streams, Databases, IoT)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Bronze Layer (Raw)                         │
│  • Raw data ingestion    • Schema inference                 │
│  • Metadata enrichment   • Immutable storage                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Silver Layer (Refined)                      │
│  • Data cleansing        • Deduplication                    │
│  • Type standardization  • SCD Type 2                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Gold Layer (Business)                      │
│  • Business aggregations • KPIs & Metrics                   │
│  • Feature engineering   • ML-ready datasets                │
└─────────────────────────────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              │   Delta Lake    │
              │ Transaction Log │
              └─────────────────┘
```

## Examples

### 1. End-to-End Pipeline

```python
from lakehouse.processor import LakehouseProcessor
from lakehouse.quality import QualityEngine

# Initialize components
processor = LakehouseProcessor(spark)
quality = QualityEngine()

# Step 1: Bronze ingestion with schema inference
processor.bronze_ingestion(
    source_path="/data/raw/sales/*.json",
    bronze_path="/lakehouse/bronze/sales",
    source_name="sales_api",
    file_format="json"
)

# Step 2: Silver transformation with quality checks
def clean_sales(df):
    return (
        df.filter(col("amount") > 0)
          .filter(col("customer_id").isNotNull())
          .withColumn("date", to_date(col("timestamp")))
    )

processor.silver_transformation(
    bronze_path="/lakehouse/bronze/sales",
    silver_path="/lakehouse/silver/sales",
    transformation_func=clean_sales
)

# Step 3: Validate data quality
silver_df = spark.read.format("delta").load("/lakehouse/silver/sales")
validation_result = (
    quality
    .expect_column_values_to_not_be_null("customer_id")
    .expect_column_values_to_be_between("amount", 0, 1000000)
    .expect_column_values_to_match_regex("email", r"^[^@]+@[^@]+\.[^@]+$")
    .validate(silver_df)
)

if validation_result.success:
    print("✓ Data quality checks passed")

# Step 4: Gold aggregation
def calculate_metrics(df):
    return (
        df.groupBy("date", "product_category")
          .agg(
              sum("amount").alias("total_revenue"),
              count("*").alias("transaction_count"),
              avg("amount").alias("avg_transaction_value")
          )
    )

processor.gold_aggregation(
    silver_path="/lakehouse/silver/sales",
    gold_path="/lakehouse/gold/daily_metrics",
    aggregation_func=calculate_metrics
)
```

### 2. Streaming Processing

```python
from lakehouse.streaming import StreamProcessor

stream_processor = StreamProcessor(spark, config)

# Start bronze streaming ingestion
bronze_stream = stream_processor.start_bronze_stream(
    source_path="/streaming/events",
    bronze_path="/lakehouse/bronze/events",
    checkpoint_path="/checkpoints/bronze",
    source_format="json",
    trigger="10 seconds"
)

# Transform stream to silver
def transform_events(batch_df, batch_id):
    return (
        batch_df
        .filter(col("event_type").isin(["click", "purchase"]))
        .withColumn("processed_at", current_timestamp())
    )

silver_stream = stream_processor.start_silver_stream(
    bronze_path="/lakehouse/bronze/events",
    silver_path="/lakehouse/silver/events",
    checkpoint_path="/checkpoints/silver",
    transformation_func=transform_events
)

# Monitor streams
print(f"Bronze stream active: {bronze_stream.isActive}")
print(f"Silver stream active: {silver_stream.isActive}")
```

### 3. Time Travel and History

```python
# Read specific version
df_v5 = spark.read.format("delta").option("versionAsOf", 5).load("/lakehouse/silver/sales")

# Read by timestamp
df_snapshot = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-15 10:00:00") \
    .load("/lakehouse/silver/sales")

# View table history
history = spark.sql("DESCRIBE HISTORY delta.`/lakehouse/silver/sales`")
history.show(10, truncate=False)

# Restore to previous version
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "/lakehouse/silver/sales")
delta_table.restoreToVersion(3)
```

### 4. Optimization and Maintenance

```python
from lakehouse.optimizer import StorageOptimizer

optimizer = StorageOptimizer(spark)

# Optimize with Z-ordering for query performance
optimizer.optimize_table(
    table_path="/lakehouse/silver/sales",
    z_order_columns=["customer_id", "product_id"],
    compact=True
)

# Vacuum old files
vacuum_result = optimizer.vacuum_table(
    table_path="/lakehouse/silver/sales",
    retention_hours=168,  # 7 days
    dry_run=False
)

print(f"Removed {vacuum_result['files_removed']} old files")

# Analyze table for statistics
spark.sql("ANALYZE TABLE delta.`/lakehouse/silver/sales` COMPUTE STATISTICS")
```

### 5. Data Quality Validation

```python
from lakehouse.quality import QualityEngine

quality = QualityEngine()

# Define comprehensive quality checks
result = (
    quality
    # Schema validation
    .expect_column_to_exist("customer_id")
    .expect_column_to_exist("amount")

    # Nullability checks
    .expect_column_values_to_not_be_null("customer_id")
    .expect_column_values_to_not_be_null("transaction_date")

    # Value constraints
    .expect_column_values_to_be_between("amount", 0, 1000000)
    .expect_column_values_to_be_between("quantity", 1, 100)

    # Pattern matching
    .expect_column_values_to_match_regex("email", r"^[^@]+@[^@]+\.[^@]+$")
    .expect_column_values_to_match_regex("phone", r"^\+?[1-9]\d{1,14}$")

    # Uniqueness
    .expect_column_values_to_be_unique("transaction_id")

    # Set membership
    .expect_column_values_to_be_in_set("status", ["pending", "completed", "cancelled"])

    # Execute validation
    .validate(df)
)

# Handle results
if not result.success:
    print(f"Quality check failures: {len(result.failures)}")
    for failure in result.failures:
        print(f"  - {failure.expectation}: {failure.message}")
else:
    print("✓ All quality checks passed")

# Profile data for insights
profile = quality.profile_data(df, columns=["amount", "quantity"])
print(f"Data profile: {profile}")
```

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# With coverage
pytest --cov=lakehouse --cov-report=html --cov-report=term

# Specific test file
pytest tests/unit/test_processor.py -v
```

### Test Structure

```
tests/
├── unit/                      # Unit tests
│   ├── test_processor.py      # Processor tests
│   ├── test_delta_log.py      # Delta log tests
│   ├── test_optimizer.py      # Optimizer tests
│   └── test_quality.py        # Quality engine tests
├── integration/               # Integration tests
│   ├── test_pipeline.py       # End-to-end pipeline tests
│   ├── test_streaming.py      # Streaming tests
│   └── test_optimization.py   # Optimization tests
└── fixtures/                  # Test fixtures
    └── sample_data.py        # Sample data generators
```

### Writing Tests

```python
import pytest
from lakehouse.processor import LakehouseProcessor

def test_bronze_ingestion(spark_session, tmp_path):
    """Test bronze layer ingestion."""
    processor = LakehouseProcessor(spark_session)

    # Create test data
    test_data = spark_session.createDataFrame([
        (1, "Alice", 100),
        (2, "Bob", 200)
    ], ["id", "name", "amount"])

    source_path = str(tmp_path / "source")
    bronze_path = str(tmp_path / "bronze")

    test_data.write.json(source_path)

    # Test ingestion
    processor.bronze_ingestion(
        source_path=source_path,
        bronze_path=bronze_path,
        source_name="test"
    )

    # Verify
    result = spark_session.read.format("delta").load(bronze_path)
    assert result.count() == 2
    assert "ingestion_timestamp" in result.columns
```

## Configuration

### Environment Variables

```bash
# Spark configuration
export SPARK_HOME=/opt/spark
export SPARK_MASTER=spark://master:7077

# Lakehouse paths
export BRONZE_PATH=s3://bucket/bronze
export SILVER_PATH=s3://bucket/silver
export GOLD_PATH=s3://bucket/gold

# Performance tuning
export SPARK_EXECUTOR_MEMORY=4g
export SPARK_EXECUTOR_CORES=4
```

### Configuration File

```yaml
# config/lakehouse.yaml
lakehouse:
  paths:
    bronze: s3://your-bucket/bronze
    silver: s3://your-bucket/silver
    gold: s3://your-bucket/gold
    checkpoints: s3://your-bucket/checkpoints

  optimization:
    auto_compact: true
    target_file_size_mb: 128
    z_order_columns:
      - customer_id
      - product_id

  quality:
    enable_validation: true
    fail_on_error: false
    sample_rate: 0.1

  streaming:
    trigger_interval: 10s
    watermark_duration: 10m
    max_files_per_trigger: 100
```

## Performance Optimization

### Best Practices

1. **File Management**
   - Target file size: 128-256 MB
   - Regular compaction to reduce small files
   - Partition by date/category for large tables

2. **Query Optimization**
   - Use Z-ordering on filter columns
   - Enable adaptive query execution
   - Cache frequently accessed tables

3. **Resource Tuning**
   ```python
   spark.conf.set("spark.sql.adaptive.enabled", "true")
   spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
   spark.conf.set("spark.sql.shuffle.partitions", "200")
   ```

## Deployment

### Docker

```bash
# Build image
docker build -t data-lakehouse .

# Run container
docker run -p 8080:8080 -v ./data:/data data-lakehouse
```

### Kubernetes

```bash
# Deploy with Helm
helm install lakehouse ./helm/lakehouse \
  --set image.tag=latest \
  --set spark.executor.instances=3

# Submit job
kubectl apply -f k8s/spark-job.yaml
```

### Cloud Platforms

- **AWS EMR**: See [docs/deployment/aws.md](docs/deployment/aws.md)
- **Azure Databricks**: See [docs/deployment/azure.md](docs/deployment/azure.md)
- **Google Dataproc**: See [docs/deployment/gcp.md](docs/deployment/gcp.md)

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and components
- [API Reference](docs/API.md) - Complete API documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute

## Monitoring

### Metrics

The lakehouse exposes Prometheus metrics on port 8000:

```python
# Available metrics
lakehouse_jobs_total{layer="bronze",status="success"}
lakehouse_job_duration_seconds{layer="silver"}
lakehouse_active_streams
lakehouse_data_quality_score{table="sales"}
```

### Dashboards

Import the provided Grafana dashboard for visualization:
- [grafana/lakehouse-dashboard.json](grafana/lakehouse-dashboard.json)

## Troubleshooting

### Common Issues

1. **OutOfMemoryError**
   ```bash
   # Increase executor memory
   spark.conf.set("spark.executor.memory", "8g")
   ```

2. **Slow Queries**
   ```python
   # Analyze query plan
   df.explain(True)

   # Add Z-ordering
   optimizer.optimize_table(path, z_order_columns=["col1", "col2"])
   ```

3. **Schema Mismatch**
   ```python
   # Enable schema merge
   spark.read.option("mergeSchema", "true").format("delta").load(path)
   ```

See [docs/troubleshooting.md](docs/troubleshooting.md) for more solutions.

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repo
git clone https://github.com/your-org/data-lakehouse.git

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: https://docs.lakehouse.io
- **Issues**: https://github.com/your-org/lakehouse/issues
- **Discussions**: https://github.com/your-org/lakehouse/discussions
- **Slack**: [Join our Slack](https://lakehouse-slack.com)

## Acknowledgments

Built with:
- [Apache Spark](https://spark.apache.org/)
- [Delta Lake](https://delta.io/)
- [PySpark](https://spark.apache.org/docs/latest/api/python/)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{data_lakehouse,
  title = {Data Lakehouse: Production-Grade Implementation},
  author = {Your Organization},
  year = {2024},
  url = {https://github.com/your-org/data-lakehouse}
}
```