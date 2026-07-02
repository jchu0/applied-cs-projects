"""Comprehensive unit tests for lakehouse processor and medallion architecture."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("pyspark")

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from lakehouse.config import LakehouseConfig
from lakehouse.processor import LakehouseProcessor, MedallionLayer


@pytest.fixture(scope="module")
def spark(spark_available):
    """Create a SparkSession for testing.

    Depends on the session-scoped ``spark_available`` guard so tests SKIP with a
    clear reason when the Java gateway is broken, instead of erroring at setup.
    """
    try:
        return (
            SparkSession.builder.appName("test_lakehouse")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.sql.adaptive.enabled", "false")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .getOrCreate()
        )
    except BaseException as exc:  # noqa: BLE001
        pytest.skip(f"Spark JVM unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture
def temp_lakehouse_dir():
    """Create temporary directory for lakehouse storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def lakehouse_config(temp_lakehouse_dir):
    """Create lakehouse configuration for testing."""
    return LakehouseConfig(
        bronze_path=str(temp_lakehouse_dir / "bronze"),
        silver_path=str(temp_lakehouse_dir / "silver"),
        gold_path=str(temp_lakehouse_dir / "gold"),
        checkpoint_path=str(temp_lakehouse_dir / "checkpoints"),
        enable_cdc=True,
        vacuum_retention_hours=168,
        optimize_interval_hours=24,
    )


@pytest.fixture
def processor(spark, lakehouse_config):
    """Create LakehouseProcessor instance."""
    return LakehouseProcessor(spark, lakehouse_config)


class TestBronzeLayer:
    """Test bronze layer ingestion and processing."""

    def test_bronze_ingestion_json(self, processor, spark, temp_lakehouse_dir):
        """Test ingesting JSON data into bronze layer."""
        # Create test JSON data
        test_data = [
            {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
            {"id": 2, "name": "Bob", "age": 25, "city": "San Francisco"},
            {"id": 3, "name": "Charlie", "age": 35, "city": "Chicago"},
        ]

        source_path = temp_lakehouse_dir / "source" / "users.json"
        source_path.parent.mkdir(parents=True)

        # Write test data
        pd.DataFrame(test_data).to_json(source_path, orient="records", lines=True)

        bronze_path = str(temp_lakehouse_dir / "bronze" / "users")

        # Ingest to bronze
        processor.bronze_ingestion(
            source_path=str(source_path),
            bronze_path=bronze_path,
            source_name="user_system",
            file_format="json",
        )

        # Verify bronze table
        bronze_df = spark.read.format("delta").load(bronze_path)

        assert bronze_df.count() == 3
        assert "ingestion_timestamp" in bronze_df.columns
        assert "source_file" in bronze_df.columns
        assert "source_system" in bronze_df.columns

        # Check data integrity
        data = bronze_df.select("id", "name").collect()
        assert len(data) == 3
        assert any(row.name == "Alice" for row in data)

    def test_bronze_ingestion_csv_with_schema(self, processor, spark, temp_lakehouse_dir):
        """Test ingesting CSV data with explicit schema."""
        # Define schema
        schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("product_name", StringType(), True),
            StructField("price", IntegerType(), True),
            StructField("category", StringType(), True),
        ])

        # Create test CSV data
        test_data = [
            {"product_id": 1, "product_name": "Laptop", "price": 999, "category": "Electronics"},
            {"product_id": 2, "product_name": "Mouse", "price": 25, "category": "Electronics"},
            {"product_id": 3, "product_name": "Book", "price": 15, "category": "Media"},
        ]

        source_path = temp_lakehouse_dir / "source" / "products.csv"
        source_path.parent.mkdir(parents=True)
        pd.DataFrame(test_data).to_csv(source_path, index=False)

        bronze_path = str(temp_lakehouse_dir / "bronze" / "products")

        # Ingest with schema
        processor.bronze_ingestion(
            source_path=str(source_path),
            bronze_path=bronze_path,
            source_name="product_catalog",
            schema=schema,
            file_format="csv",
        )

        # Verify schema enforcement
        bronze_df = spark.read.format("delta").load(bronze_path)
        assert bronze_df.schema["price"].dataType == IntegerType()
        assert bronze_df.filter(col("price") > 20).count() == 2

    def test_bronze_incremental_load(self, processor, spark, temp_lakehouse_dir):
        """Test incremental loading into bronze layer."""
        source_dir = temp_lakehouse_dir / "source" / "events"
        source_dir.mkdir(parents=True)
        bronze_path = str(temp_lakehouse_dir / "bronze" / "events")

        # Initial load
        batch1 = [{"event_id": 1, "event_type": "click", "timestamp": "2024-01-01"}]
        pd.DataFrame(batch1).to_json(source_dir / "batch1.json", orient="records", lines=True)

        processor.bronze_ingestion(
            source_path=str(source_dir / "batch1.json"),
            bronze_path=bronze_path,
            source_name="event_stream",
            file_format="json",
        )

        # Incremental load
        batch2 = [{"event_id": 2, "event_type": "view", "timestamp": "2024-01-02"}]
        pd.DataFrame(batch2).to_json(source_dir / "batch2.json", orient="records", lines=True)

        processor.bronze_ingestion(
            source_path=str(source_dir / "batch2.json"),
            bronze_path=bronze_path,
            source_name="event_stream",
            file_format="json",
        )

        # Verify both batches are present
        bronze_df = spark.read.format("delta").load(bronze_path)
        assert bronze_df.count() == 2
        assert bronze_df.filter(col("event_id") == 1).count() == 1
        assert bronze_df.filter(col("event_id") == 2).count() == 1


class TestSilverLayer:
    """Test silver layer transformation and cleaning."""

    def test_silver_transformation(self, processor, spark, temp_lakehouse_dir):
        """Test transforming bronze data to silver layer."""
        # Create bronze data
        bronze_data = spark.createDataFrame([
            (1, "Alice", 30, "alice@example.com", "2024-01-01"),
            (2, "Bob", 25, "invalid-email", "2024-01-02"),
            (3, "Charlie", 35, "charlie@example.com", "2024-01-03"),
            (4, None, 28, "david@example.com", "2024-01-04"),
        ], ["id", "name", "age", "email", "date"])

        bronze_path = str(temp_lakehouse_dir / "bronze" / "users")
        bronze_data.write.format("delta").mode("overwrite").save(bronze_path)

        silver_path = str(temp_lakehouse_dir / "silver" / "users")

        # Transform to silver with cleaning rules
        def clean_users(df):
            return (
                df.filter(col("name").isNotNull())
                .filter(col("email").contains("@"))
                .withColumn("email_domain", col("email").substr(col("email").indexOf("@") + 2, 100))
            )

        processor.silver_transformation(
            bronze_path=bronze_path,
            silver_path=silver_path,
            transformation_func=clean_users,
        )

        # Verify silver data
        silver_df = spark.read.format("delta").load(silver_path)

        assert silver_df.count() == 2  # Only valid records
        assert "email_domain" in silver_df.columns
        assert silver_df.filter(col("name") == "Bob").count() == 0  # Invalid email filtered

    def test_silver_deduplication(self, processor, spark, temp_lakehouse_dir):
        """Test deduplication in silver layer."""
        # Create bronze data with duplicates
        bronze_data = spark.createDataFrame([
            (1, "Alice", "2024-01-01 10:00:00"),
            (1, "Alice", "2024-01-01 11:00:00"),  # Duplicate with later timestamp
            (2, "Bob", "2024-01-01 10:00:00"),
            (3, "Charlie", "2024-01-01 10:00:00"),
            (3, "Charlie", "2024-01-01 09:00:00"),  # Duplicate with earlier timestamp
        ], ["id", "name", "timestamp"])

        bronze_path = str(temp_lakehouse_dir / "bronze" / "events")
        bronze_data.write.format("delta").mode("overwrite").save(bronze_path)

        silver_path = str(temp_lakehouse_dir / "silver" / "events")

        # Deduplicate keeping latest record
        processor.silver_deduplication(
            bronze_path=bronze_path,
            silver_path=silver_path,
            dedup_columns=["id"],
            order_column="timestamp",
        )

        # Verify deduplication
        silver_df = spark.read.format("delta").load(silver_path)

        assert silver_df.count() == 3  # Three unique IDs
        alice_record = silver_df.filter(col("id") == 1).collect()[0]
        assert "11:00:00" in alice_record.timestamp  # Latest timestamp kept

    def test_silver_scd_type2(self, processor, spark, temp_lakehouse_dir):
        """Test Slowly Changing Dimension Type 2 in silver layer."""
        # Initial silver data
        initial_data = spark.createDataFrame([
            (1, "Alice", "New York", "2024-01-01", "2999-12-31", True),
            (2, "Bob", "San Francisco", "2024-01-01", "2999-12-31", True),
        ], ["id", "name", "city", "valid_from", "valid_to", "is_current"])

        silver_path = str(temp_lakehouse_dir / "silver" / "customers")
        initial_data.write.format("delta").mode("overwrite").save(silver_path)

        # New data with changes
        updates = spark.createDataFrame([
            (1, "Alice", "Boston"),  # Alice moved
            (2, "Bob", "San Francisco"),  # Bob unchanged
            (3, "Charlie", "Chicago"),  # New customer
        ], ["id", "name", "city"])

        # Apply SCD Type 2
        processor.apply_scd_type2(
            silver_path=silver_path,
            updates_df=updates,
            key_columns=["id"],
            track_columns=["city"],
        )

        # Verify SCD logic
        silver_df = spark.read.format("delta").load(silver_path)

        # Alice should have 2 records
        alice_records = silver_df.filter(col("id") == 1).orderBy("valid_from").collect()
        assert len(alice_records) == 2
        assert alice_records[0].city == "New York"
        assert alice_records[0].is_current is False
        assert alice_records[1].city == "Boston"
        assert alice_records[1].is_current is True

        # Bob should have 1 record (unchanged)
        assert silver_df.filter((col("id") == 2) & col("is_current")).count() == 1

        # Charlie should have 1 record (new)
        assert silver_df.filter((col("id") == 3) & col("is_current")).count() == 1


class TestGoldLayer:
    """Test gold layer aggregation and business logic."""

    def test_gold_aggregation(self, processor, spark, temp_lakehouse_dir):
        """Test creating aggregated gold table."""
        # Create silver data
        silver_data = spark.createDataFrame([
            ("2024-01-01", "Electronics", 100, 2),
            ("2024-01-01", "Electronics", 200, 1),
            ("2024-01-01", "Books", 25, 5),
            ("2024-01-02", "Electronics", 150, 3),
            ("2024-01-02", "Books", 30, 4),
        ], ["date", "category", "revenue", "quantity"])

        silver_path = str(temp_lakehouse_dir / "silver" / "sales")
        silver_data.write.format("delta").mode("overwrite").save(silver_path)

        gold_path = str(temp_lakehouse_dir / "gold" / "daily_sales")

        # Create gold aggregation
        def daily_sales_agg(df):
            return (
                df.groupBy("date", "category")
                .agg(
                    {"revenue": "sum", "quantity": "sum"}
                )
                .withColumnRenamed("sum(revenue)", "total_revenue")
                .withColumnRenamed("sum(quantity)", "total_quantity")
                .withColumn("avg_price", col("total_revenue") / col("total_quantity"))
            )

        processor.gold_aggregation(
            silver_path=silver_path,
            gold_path=gold_path,
            aggregation_func=daily_sales_agg,
        )

        # Verify gold table
        gold_df = spark.read.format("delta").load(gold_path)

        assert gold_df.count() == 4
        assert "avg_price" in gold_df.columns

        electronics_jan1 = gold_df.filter(
            (col("date") == "2024-01-01") & (col("category") == "Electronics")
        ).collect()[0]
        assert electronics_jan1.total_revenue == 300
        assert electronics_jan1.total_quantity == 3

    def test_gold_business_metrics(self, processor, spark, temp_lakehouse_dir):
        """Test calculating business metrics in gold layer."""
        # Create silver customer data
        customers = spark.createDataFrame([
            (1, "Alice", "2023-01-01", 5, 1000),
            (2, "Bob", "2023-06-01", 3, 500),
            (3, "Charlie", "2023-09-01", 8, 2000),
            (4, "David", "2024-01-01", 1, 100),
        ], ["customer_id", "name", "first_purchase", "purchase_count", "total_spent"])

        silver_path = str(temp_lakehouse_dir / "silver" / "customers")
        customers.write.format("delta").mode("overwrite").save(silver_path)

        gold_path = str(temp_lakehouse_dir / "gold" / "customer_segments")

        # Calculate customer segments
        def segment_customers(df):
            return df.withColumn(
                "segment",
                when(col("total_spent") >= 1000, "Premium")
                .when(col("total_spent") >= 500, "Standard")
                .otherwise("Basic")
            ).withColumn(
                "lifetime_value", col("total_spent") / col("purchase_count")
            )

        processor.create_business_view(
            silver_path=silver_path,
            gold_path=gold_path,
            business_logic=segment_customers,
        )

        # Verify business logic
        gold_df = spark.read.format("delta").load(gold_path)

        assert gold_df.filter(col("segment") == "Premium").count() == 2
        assert gold_df.filter(col("segment") == "Basic").count() == 1
        assert "lifetime_value" in gold_df.columns


class TestDeltaFeatures:
    """Test Delta Lake specific features."""

    def test_time_travel(self, processor, spark, temp_lakehouse_dir):
        """Test time travel queries on Delta tables."""
        table_path = str(temp_lakehouse_dir / "delta_table")

        # Create initial version
        v0_data = spark.createDataFrame([(1, "v0"), (2, "v0")], ["id", "version"])
        v0_data.write.format("delta").mode("overwrite").save(table_path)

        # Update to create version 1
        v1_data = spark.createDataFrame([(3, "v1"), (4, "v1")], ["id", "version"])
        v1_data.write.format("delta").mode("append").save(table_path)

        # Query different versions
        current = spark.read.format("delta").load(table_path)
        assert current.count() == 4

        version_0 = spark.read.format("delta").option("versionAsOf", 0).load(table_path)
        assert version_0.count() == 2
        assert version_0.filter(col("version") == "v0").count() == 2

    def test_merge_operation(self, processor, spark, temp_lakehouse_dir):
        """Test MERGE operation on Delta tables."""
        target_path = str(temp_lakehouse_dir / "target")

        # Create target table
        target_data = spark.createDataFrame([
            (1, "Alice", 30),
            (2, "Bob", 25),
            (3, "Charlie", 35),
        ], ["id", "name", "age"])
        target_data.write.format("delta").mode("overwrite").save(target_path)

        # Create source data with updates and new records
        source_data = spark.createDataFrame([
            (2, "Bob", 26),  # Update Bob's age
            (4, "David", 40),  # New record
        ], ["id", "name", "age"])

        # Perform merge
        processor.merge_delta_tables(
            target_path=target_path,
            source_df=source_data,
            merge_condition="target.id = source.id",
            update_condition="target.age != source.age",
        )

        # Verify merge results
        result = spark.read.format("delta").load(target_path)
        assert result.count() == 4

        bob = result.filter(col("id") == 2).collect()[0]
        assert bob.age == 26  # Updated

        david = result.filter(col("id") == 4).collect()
        assert len(david) == 1  # Inserted

    def test_optimize_and_vacuum(self, processor, spark, temp_lakehouse_dir):
        """Test OPTIMIZE and VACUUM operations."""
        table_path = str(temp_lakehouse_dir / "optimized_table")

        # Create many small files
        for i in range(10):
            small_data = spark.createDataFrame([(i, f"data_{i}")], ["id", "value"])
            small_data.write.format("delta").mode("append").save(table_path)

        # Check initial file count
        initial_files = spark.read.format("delta").load(table_path).inputFiles()
        assert len(initial_files) >= 10

        # Optimize table
        processor.optimize_table(table_path)

        # Check optimized file count
        optimized_files = spark.read.format("delta").load(table_path).inputFiles()
        assert len(optimized_files) < len(initial_files)

        # Vacuum old files
        processor.vacuum_table(table_path, retention_hours=0)