"""Integration tests for data lakehouse system."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("pyspark")

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, max, min, sum

from lakehouse.config import LakehouseConfig
from lakehouse.delta_log import DeltaLog
from lakehouse.optimizer import StorageOptimizer
from lakehouse.processor import LakehouseProcessor
from lakehouse.quality import QualityEngine
from lakehouse.streaming import StreamProcessor


@pytest.fixture(scope="module")
def spark():
    """Create SparkSession for integration testing."""
    return (
        SparkSession.builder.appName("integration_test")
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .getOrCreate()
    )


@pytest.fixture
def lakehouse_env(spark):
    """Set up complete lakehouse environment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Create directory structure
        (base_path / "raw").mkdir()
        (base_path / "bronze").mkdir()
        (base_path / "silver").mkdir()
        (base_path / "gold").mkdir()
        (base_path / "checkpoints").mkdir()

        config = LakehouseConfig(
            bronze_path=str(base_path / "bronze"),
            silver_path=str(base_path / "silver"),
            gold_path=str(base_path / "gold"),
            checkpoint_path=str(base_path / "checkpoints"),
            enable_cdc=True,
            vacuum_retention_hours=1,
            optimize_interval_hours=1,
        )

        processor = LakehouseProcessor(spark, config)
        quality_engine = QualityEngine()
        optimizer = StorageOptimizer(spark)

        yield {
            "base_path": base_path,
            "config": config,
            "processor": processor,
            "quality_engine": quality_engine,
            "optimizer": optimizer,
            "spark": spark,
        }


class TestEndToEndPipeline:
    """Test complete data pipeline from raw to gold."""

    def test_complete_medallion_pipeline(self, lakehouse_env):
        """Test full medallion architecture pipeline."""
        env = lakehouse_env

        # Step 1: Generate raw data
        raw_data = self._generate_sales_data()
        raw_path = env["base_path"] / "raw" / "sales.json"
        pd.DataFrame(raw_data).to_json(raw_path, orient="records", lines=True)

        # Step 2: Bronze ingestion
        bronze_path = str(env["base_path"] / "bronze" / "sales")
        env["processor"].bronze_ingestion(
            source_path=str(raw_path),
            bronze_path=bronze_path,
            source_name="sales_system",
            file_format="json",
        )

        # Verify bronze
        bronze_df = env["spark"].read.format("delta").load(bronze_path)
        assert bronze_df.count() == len(raw_data)
        assert "ingestion_timestamp" in bronze_df.columns

        # Step 3: Silver transformation with quality checks
        silver_path = str(env["base_path"] / "silver" / "sales")

        def clean_sales(df):
            return (
                df.filter(col("amount") > 0)
                .filter(col("customer_id").isNotNull())
                .withColumn("date", col("timestamp").cast("date"))
            )

        env["processor"].silver_transformation(
            bronze_path=bronze_path,
            silver_path=silver_path,
            transformation_func=clean_sales,
        )

        # Quality validation
        silver_df = env["spark"].read.format("delta").load(silver_path)
        quality_results = (
            env["quality_engine"]
            .expect_column_to_exist("customer_id")
            .expect_column_values_to_not_be_null("customer_id")
            .expect_column_values_to_be_between("amount", 0, 10000)
            .validate(silver_df)
        )
        assert quality_results.success

        # Step 4: Gold aggregation
        gold_path = str(env["base_path"] / "gold" / "daily_sales")

        def aggregate_daily(df):
            return (
                df.groupBy("date", "product_category")
                .agg(
                    count("*").alias("transaction_count"),
                    sum("amount").alias("total_revenue"),
                    sum("quantity").alias("total_units"),
                )
                .withColumn("avg_transaction_value", col("total_revenue") / col("transaction_count"))
            )

        env["processor"].gold_aggregation(
            silver_path=silver_path,
            gold_path=gold_path,
            aggregation_func=aggregate_daily,
        )

        # Verify gold
        gold_df = env["spark"].read.format("delta").load(gold_path)
        assert gold_df.count() > 0
        assert "avg_transaction_value" in gold_df.columns

    def test_incremental_processing(self, lakehouse_env):
        """Test incremental data processing."""
        env = lakehouse_env

        # Initial batch
        batch1 = self._generate_sales_data(num_records=100, date_str="2024-01-01")
        bronze_path = str(env["base_path"] / "bronze" / "incremental")

        # Process batch 1
        raw_path1 = env["base_path"] / "raw" / "batch1.json"
        pd.DataFrame(batch1).to_json(raw_path1, orient="records", lines=True)

        env["processor"].bronze_ingestion(
            source_path=str(raw_path1),
            bronze_path=bronze_path,
            source_name="incremental_source",
            file_format="json",
        )

        initial_count = env["spark"].read.format("delta").load(bronze_path).count()
        assert initial_count == 100

        # Process batch 2
        batch2 = self._generate_sales_data(num_records=50, date_str="2024-01-02")
        raw_path2 = env["base_path"] / "raw" / "batch2.json"
        pd.DataFrame(batch2).to_json(raw_path2, orient="records", lines=True)

        env["processor"].bronze_ingestion(
            source_path=str(raw_path2),
            bronze_path=bronze_path,
            source_name="incremental_source",
            file_format="json",
        )

        final_count = env["spark"].read.format("delta").load(bronze_path).count()
        assert final_count == 150

    def test_scd_type2_integration(self, lakehouse_env):
        """Test SCD Type 2 implementation end-to-end."""
        env = lakehouse_env

        # Initial customer data
        initial_customers = [
            {"customer_id": 1, "name": "Alice", "city": "New York", "status": "Active"},
            {"customer_id": 2, "name": "Bob", "city": "Boston", "status": "Active"},
        ]

        silver_path = str(env["base_path"] / "silver" / "customers")

        # Load initial data
        initial_df = env["spark"].createDataFrame(initial_customers)
        initial_df.write.format("delta").mode("overwrite").save(silver_path)

        # Customer updates
        updates = [
            {"customer_id": 1, "name": "Alice", "city": "Chicago", "status": "Active"},  # Changed city
            {"customer_id": 2, "name": "Bob", "city": "Boston", "status": "Inactive"},  # Changed status
            {"customer_id": 3, "name": "Charlie", "city": "Dallas", "status": "Active"},  # New customer
        ]

        updates_df = env["spark"].createDataFrame(updates)

        # Apply SCD Type 2
        env["processor"].apply_scd_type2(
            silver_path=silver_path,
            updates_df=updates_df,
            key_columns=["customer_id"],
            track_columns=["city", "status"],
        )

        # Verify SCD results
        result_df = env["spark"].read.format("delta").load(silver_path)

        # Alice should have 2 records
        alice_records = result_df.filter(col("customer_id") == 1).count()
        assert alice_records == 2

        # Current Alice record should show Chicago
        current_alice = result_df.filter(
            (col("customer_id") == 1) & col("is_current")
        ).collect()[0]
        assert current_alice.city == "Chicago"

    def _generate_sales_data(self, num_records=1000, date_str="2024-01-01"):
        """Generate sample sales data."""
        import random

        categories = ["Electronics", "Clothing", "Books", "Food", "Sports"]
        data = []

        for i in range(num_records):
            data.append({
                "transaction_id": f"txn_{date_str}_{i:05d}",
                "timestamp": f"{date_str} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00",
                "customer_id": random.randint(1, 100),
                "product_id": random.randint(1, 500),
                "product_category": random.choice(categories),
                "amount": round(random.uniform(10, 1000), 2),
                "quantity": random.randint(1, 10),
                "payment_method": random.choice(["credit", "debit", "cash"]),
            })

        return data


class TestStreamingIntegration:
    """Test streaming capabilities integration."""

    def test_streaming_pipeline(self, lakehouse_env):
        """Test streaming data through medallion architecture."""
        env = lakehouse_env

        source_path = str(env["base_path"] / "streaming_source")
        bronze_path = str(env["base_path"] / "bronze" / "streaming")
        silver_path = str(env["base_path"] / "silver" / "streaming")
        checkpoint_bronze = str(env["base_path"] / "checkpoints" / "bronze")
        checkpoint_silver = str(env["base_path"] / "checkpoints" / "silver")

        # Create streaming processor
        stream_processor = StreamProcessor(env["spark"], env["config"])

        # Start bronze streaming
        bronze_stream = stream_processor.start_bronze_stream(
            source_path=source_path,
            bronze_path=bronze_path,
            checkpoint_path=checkpoint_bronze,
            source_format="json",
        )

        # Start silver streaming with transformation
        def transform_stream(batch_df, batch_id):
            return batch_df.filter(col("event_type").isin(["click", "purchase"]))

        silver_stream = stream_processor.start_silver_stream(
            bronze_path=bronze_path,
            silver_path=silver_path,
            checkpoint_path=checkpoint_silver,
            transformation_func=transform_stream,
        )

        # Generate streaming data
        for batch_num in range(3):
            batch_data = [
                {
                    "event_id": f"evt_{batch_num}_{i}",
                    "event_type": ["click", "view", "purchase"][i % 3],
                    "user_id": f"user_{i}",
                    "timestamp": datetime.now().isoformat(),
                }
                for i in range(10)
            ]

            batch_df = env["spark"].createDataFrame(batch_data)
            batch_df.write.format("json").mode("append").save(source_path)

            # Process batch
            bronze_stream.processAllAvailable()
            silver_stream.processAllAvailable()

        # Verify results
        bronze_df = env["spark"].read.format("delta").load(bronze_path)
        silver_df = env["spark"].read.format("delta").load(silver_path)

        assert bronze_df.count() == 30  # All events
        assert silver_df.count() == 20  # Only clicks and purchases

        # Stop streams
        bronze_stream.stop()
        silver_stream.stop()

    def test_watermarking_and_windowing(self, lakehouse_env):
        """Test watermarking and windowing in streaming."""
        env = lakehouse_env

        # Create streaming data with timestamps
        stream_data = []
        base_time = datetime.now()

        for i in range(100):
            stream_data.append({
                "event_id": f"evt_{i}",
                "value": i * 10,
                "event_time": (base_time + timedelta(seconds=i * 30)).isoformat(),
            })

        source_path = str(env["base_path"] / "windowed_source")
        output_path = str(env["base_path"] / "windowed_output")
        checkpoint_path = str(env["base_path"] / "checkpoints" / "windowed")

        # Write source data
        env["spark"].createDataFrame(stream_data).write.format("json").save(source_path)

        # Create windowed aggregation stream
        stream_processor = StreamProcessor(env["spark"], env["config"])

        windowed_stream = stream_processor.create_windowed_aggregation(
            source_path=source_path,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            window_duration="5 minutes",
            watermark_duration="10 minutes",
            group_columns=["window"],
            aggregations={"value": "sum"},
        )

        windowed_stream.processAllAvailable()

        # Verify windowed results
        result_df = env["spark"].read.format("delta").load(output_path)
        assert result_df.count() > 0
        assert "window" in result_df.columns
        assert "sum_value" in result_df.columns

        windowed_stream.stop()


class TestOptimizationIntegration:
    """Test optimization features integration."""

    def test_auto_optimization(self, lakehouse_env):
        """Test automatic optimization triggers."""
        env = lakehouse_env

        table_path = str(env["base_path"] / "optimized_table")

        # Create many small files
        for i in range(50):
            small_df = env["spark"].createDataFrame([
                (i, f"data_{i}", i * 100)
            ], ["id", "name", "value"])
            small_df.write.format("delta").mode("append").save(table_path)

        # Check initial state
        initial_files = env["spark"].read.format("delta").load(table_path).inputFiles()
        assert len(initial_files) >= 50

        # Run optimization
        optimization_result = env["optimizer"].optimize_table(
            table_path=table_path,
            compact=True,
            z_order_columns=["id"],
            collect_stats=True,
        )

        # Verify optimization
        optimized_files = env["spark"].read.format("delta").load(table_path).inputFiles()
        assert len(optimized_files) < len(initial_files)
        assert optimization_result["files_compacted"] > 0

    def test_vacuum_integration(self, lakehouse_env):
        """Test vacuum operation with Delta log."""
        env = lakehouse_env

        table_path = str(env["base_path"] / "vacuum_table")

        # Create and delete data
        for i in range(10):
            df = env["spark"].createDataFrame([(i, f"data_{i}")], ["id", "value"])
            df.write.format("delta").mode("append").save(table_path)

        # Delete some data
        from delta.tables import DeltaTable

        delta_table = DeltaTable.forPath(env["spark"], table_path)
        delta_table.delete(col("id") < 5)

        # Run vacuum
        vacuum_result = env["optimizer"].vacuum_table(
            table_path=table_path,
            retention_hours=0,
            dry_run=False,
        )

        assert vacuum_result["files_removed"] > 0

    def test_performance_monitoring(self, lakehouse_env):
        """Test performance monitoring and metrics collection."""
        env = lakehouse_env

        # Create test table
        table_path = str(env["base_path"] / "monitored_table")
        data = env["spark"].range(10000).selectExpr("id", "id * 10 as value")
        data.write.format("delta").mode("overwrite").save(table_path)

        # Collect performance metrics
        metrics = env["optimizer"].collect_performance_metrics(table_path)

        assert "table_size_bytes" in metrics
        assert "num_files" in metrics
        assert "avg_file_size_bytes" in metrics
        assert "num_partitions" in metrics

        # Run query and collect query metrics
        query = env["spark"].read.format("delta").load(table_path).filter(col("value") > 5000)
        query.count()

        query_metrics = env["optimizer"].get_query_metrics(query)
        assert "files_scanned" in query_metrics
        assert "rows_produced" in query_metrics


class TestDataQualityIntegration:
    """Test data quality features integration."""

    def test_quality_gates(self, lakehouse_env):
        """Test quality gates in pipeline."""
        env = lakehouse_env

        # Create test data with quality issues
        test_data = [
            {"id": 1, "email": "valid@example.com", "age": 25},
            {"id": 2, "email": "invalid-email", "age": 150},  # Invalid email and age
            {"id": 3, "email": "another@example.com", "age": 30},
            {"id": None, "email": "null@example.com", "age": 35},  # Null ID
        ]

        df = env["spark"].createDataFrame(test_data)

        # Define quality rules
        quality_results = (
            env["quality_engine"]
            .expect_column_values_to_not_be_null("id")
            .expect_column_values_to_match_regex("email", r"^[^@]+@[^@]+\.[^@]+$")
            .expect_column_values_to_be_between("age", 0, 120)
            .validate(df)
        )

        assert not quality_results.success
        assert len(quality_results.failures) == 3

        # Clean data based on quality results
        clean_df = df.filter(
            col("id").isNotNull() &
            col("email").rlike(r"^[^@]+@[^@]+\.[^@]+$") &
            col("age").between(0, 120)
        )

        assert clean_df.count() == 2

    def test_anomaly_detection(self, lakehouse_env):
        """Test anomaly detection in data."""
        env = lakehouse_env

        # Create time series data with anomalies
        normal_data = [
            {"timestamp": f"2024-01-{i:02d}", "value": 100 + i * 2}
            for i in range(1, 20)
        ]

        # Insert anomalies
        normal_data.append({"timestamp": "2024-01-21", "value": 500})  # Spike
        normal_data.append({"timestamp": "2024-01-22", "value": -50})  # Negative

        df = env["spark"].createDataFrame(normal_data)

        # Detect anomalies
        anomalies = env["quality_engine"].detect_anomalies(
            df=df,
            column="value",
            method="zscore",
            threshold=2,
        )

        assert len(anomalies) >= 2
        assert any(a["value"] == 500 for a in anomalies)
        assert any(a["value"] == -50 for a in anomalies)


class TestEnterpriseFeatures:
    """Test enterprise features integration."""

    def test_multi_tenant_isolation(self, lakehouse_env):
        """Test multi-tenant data isolation."""
        env = lakehouse_env

        # Create tenant-specific tables
        tenants = ["tenant_a", "tenant_b", "tenant_c"]

        for tenant in tenants:
            tenant_path = str(env["base_path"] / "tenants" / tenant / "data")

            # Create tenant-specific data
            data = env["spark"].createDataFrame([
                (i, f"{tenant}_record_{i}") for i in range(10)
            ], ["id", "value"])

            data.write.format("delta").mode("overwrite").save(tenant_path)

            # Verify isolation
            tenant_df = env["spark"].read.format("delta").load(tenant_path)
            assert tenant_df.count() == 10
            assert all(tenant in row.value for row in tenant_df.collect())

    def test_audit_logging(self, lakehouse_env):
        """Test audit logging for compliance."""
        env = lakehouse_env

        audit_log = []

        def log_operation(operation, table, user="test_user"):
            audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "table": table,
                "user": user,
            })

        # Perform operations with audit logging
        table_path = str(env["base_path"] / "audited_table")

        # Create
        log_operation("CREATE", "audited_table")
        env["spark"].range(10).write.format("delta").save(table_path)

        # Update
        log_operation("UPDATE", "audited_table")
        from delta.tables import DeltaTable
        delta_table = DeltaTable.forPath(env["spark"], table_path)
        delta_table.update(col("id") < 5, {"id": col("id") * 10})

        # Delete
        log_operation("DELETE", "audited_table")
        delta_table.delete(col("id") > 50)

        # Verify audit log
        assert len(audit_log) == 3
        assert audit_log[0]["operation"] == "CREATE"
        assert audit_log[1]["operation"] == "UPDATE"
        assert audit_log[2]["operation"] == "DELETE"