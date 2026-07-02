"""Unit tests for lakehouse optimizer."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession

from lakehouse.optimizer import (
    CompactionStrategy,
    OptimizationPlan,
    QueryOptimizer,
    StorageOptimizer,
    ZOrderOptimizer,
)


@pytest.fixture
def spark(spark_available):
    """Create SparkSession for testing.

    Depends on the session-scoped ``spark_available`` guard so tests SKIP with a
    clear reason when the Java gateway is broken, instead of erroring at setup.
    """
    try:
        return SparkSession.builder.appName("test_optimizer").master("local[*]").getOrCreate()
    except BaseException as exc:  # noqa: BLE001
        pytest.skip(f"Spark JVM unavailable: {type(exc).__name__}: {exc}")


class TestStorageOptimizer:
    """Test storage optimization features."""

    def test_identify_small_files(self, spark):
        """Test identifying small files for compaction."""
        optimizer = StorageOptimizer(spark)

        # Mock file listing with various sizes
        files = [
            {"path": "file1.parquet", "size": 1024 * 1024},  # 1MB
            {"path": "file2.parquet", "size": 10 * 1024 * 1024},  # 10MB
            {"path": "file3.parquet", "size": 500 * 1024},  # 500KB
            {"path": "file4.parquet", "size": 128 * 1024 * 1024},  # 128MB
            {"path": "file5.parquet", "size": 2 * 1024 * 1024},  # 2MB
        ]

        small_files = optimizer.identify_small_files(files, threshold_mb=5)

        assert len(small_files) == 3  # Files under 5MB
        assert all(f["size"] < 5 * 1024 * 1024 for f in small_files)

    def test_calculate_compaction_bins(self, spark):
        """Test bin packing for file compaction."""
        optimizer = StorageOptimizer(spark)

        # Files to compact
        files = [
            {"path": "file1.parquet", "size": 10 * 1024 * 1024, "partition": "date=2024-01-01"},
            {"path": "file2.parquet", "size": 20 * 1024 * 1024, "partition": "date=2024-01-01"},
            {"path": "file3.parquet", "size": 30 * 1024 * 1024, "partition": "date=2024-01-01"},
            {"path": "file4.parquet", "size": 15 * 1024 * 1024, "partition": "date=2024-01-02"},
            {"path": "file5.parquet", "size": 25 * 1024 * 1024, "partition": "date=2024-01-02"},
        ]

        bins = optimizer.calculate_compaction_bins(files, target_size_mb=50)

        # Should group files efficiently
        assert len(bins) > 0
        for bin_files in bins:
            total_size = sum(f["size"] for f in bin_files)
            assert total_size <= 60 * 1024 * 1024  # Allow some overhead

    def test_optimize_partition_layout(self, spark, tmp_path):
        """Test optimizing partition layout."""
        optimizer = StorageOptimizer(spark)

        # Create test data with skewed partitions
        data = spark.createDataFrame([
            ("2024-01-01", "US", i) for i in range(100)
        ] + [
            ("2024-01-01", "UK", i) for i in range(10)
        ] + [
            ("2024-01-02", "US", i) for i in range(50)
        ], ["date", "country", "value"])

        table_path = str(tmp_path / "partitioned_table")
        data.write.mode("overwrite").partitionBy("date", "country").parquet(table_path)

        # Analyze partition skew
        skew_report = optimizer.analyze_partition_skew(table_path)

        assert "partitions" in skew_report
        assert "max_size" in skew_report
        assert "min_size" in skew_report
        assert "skew_ratio" in skew_report

    def test_vacuum_strategy(self, spark):
        """Test vacuum strategy for deleted files."""
        optimizer = StorageOptimizer(spark)

        # Mock deleted files with timestamps
        deleted_files = [
            {"path": "old1.parquet", "deletion_timestamp": datetime.now() - timedelta(days=8)},
            {"path": "old2.parquet", "deletion_timestamp": datetime.now() - timedelta(days=2)},
            {"path": "recent.parquet", "deletion_timestamp": datetime.now() - timedelta(hours=1)},
        ]

        # Files to vacuum (older than 7 days)
        files_to_vacuum = optimizer.get_vacuum_candidates(deleted_files, retention_days=7)

        assert len(files_to_vacuum) == 1
        assert files_to_vacuum[0]["path"] == "old1.parquet"

    def test_compression_recommendations(self, spark):
        """Test recommending compression codecs."""
        optimizer = StorageOptimizer(spark)

        # Test data characteristics
        data_profile = {
            "avg_row_size": 1024,
            "cardinality": "high",
            "data_type": "mixed",
            "query_pattern": "analytical",
        }

        codec = optimizer.recommend_compression(data_profile)

        assert codec in ["snappy", "zstd", "lz4", "gzip"]
        # For analytical workloads with mixed data, expect zstd or snappy
        assert codec in ["zstd", "snappy"]


class TestQueryOptimizer:
    """Test query optimization features."""

    def test_predicate_pushdown(self, spark):
        """Test predicate pushdown optimization."""
        optimizer = QueryOptimizer(spark)

        # Mock query plan
        query_plan = {
            "type": "Filter",
            "condition": "date >= '2024-01-01' AND country = 'US'",
            "child": {
                "type": "Scan",
                "table": "sales",
                "columns": ["date", "country", "amount"]
            }
        }

        optimized = optimizer.optimize_predicate_pushdown(query_plan)

        # Predicate should be pushed to scan
        assert optimized["type"] == "Scan"
        assert "pushed_filters" in optimized
        assert "date >= '2024-01-01'" in optimized["pushed_filters"]

    def test_column_pruning(self, spark):
        """Test column pruning optimization."""
        optimizer = QueryOptimizer(spark)

        # Query using only subset of columns
        query = "SELECT date, amount FROM sales WHERE country = 'US'"
        required_columns = optimizer.identify_required_columns(query)

        assert set(required_columns) == {"date", "amount", "country"}
        assert "unnecessary_column" not in required_columns

    def test_join_reordering(self, spark):
        """Test join reordering for optimal performance."""
        optimizer = QueryOptimizer(spark)

        # Mock join statistics
        joins = [
            {"tables": ["orders", "customers"], "estimated_rows": 1000000},
            {"tables": ["orders", "products"], "estimated_rows": 5000000},
            {"tables": ["customers", "regions"], "estimated_rows": 10000},
        ]

        optimal_order = optimizer.optimize_join_order(joins)

        # Should start with smallest join
        assert optimal_order[0]["tables"] == ["customers", "regions"]

    def test_broadcast_join_detection(self, spark):
        """Test detecting opportunities for broadcast joins."""
        optimizer = QueryOptimizer(spark)

        # Table sizes
        tables = {
            "large_table": 10 * 1024 * 1024 * 1024,  # 10GB
            "small_table": 10 * 1024 * 1024,  # 10MB
            "medium_table": 500 * 1024 * 1024,  # 500MB
        }

        # Check broadcast recommendations
        broadcast_candidates = optimizer.identify_broadcast_candidates(
            tables, threshold_mb=100
        )

        assert "small_table" in broadcast_candidates
        assert "large_table" not in broadcast_candidates
        assert "medium_table" not in broadcast_candidates

    def test_cache_recommendations(self, spark):
        """Test recommending tables for caching."""
        optimizer = QueryOptimizer(spark)

        # Query patterns
        query_log = [
            {"table": "dim_date", "frequency": 100},
            {"table": "fact_sales", "frequency": 50},
            {"table": "dim_product", "frequency": 80},
            {"table": "staging_temp", "frequency": 1},
        ]

        cache_recommendations = optimizer.recommend_caching(query_log, min_frequency=10)

        assert "dim_date" in cache_recommendations
        assert "dim_product" in cache_recommendations
        assert "staging_temp" not in cache_recommendations


class TestZOrderOptimizer:
    """Test Z-order optimization for multi-dimensional clustering."""

    def test_z_order_column_selection(self, spark):
        """Test selecting columns for Z-ordering."""
        optimizer = ZOrderOptimizer(spark)

        # Column statistics
        column_stats = {
            "customer_id": {"cardinality": 1000000, "query_frequency": 80},
            "product_id": {"cardinality": 10000, "query_frequency": 90},
            "date": {"cardinality": 365, "query_frequency": 100},
            "amount": {"cardinality": 100000, "query_frequency": 20},
        }

        selected_columns = optimizer.select_zorder_columns(column_stats, max_columns=2)

        assert len(selected_columns) == 2
        assert "date" in selected_columns  # Highest query frequency
        assert "product_id" in selected_columns  # High frequency, moderate cardinality

    def test_z_order_effectiveness(self, spark):
        """Test measuring Z-order effectiveness."""
        optimizer = ZOrderOptimizer(spark)

        # Mock file statistics before and after Z-ordering
        before_stats = {
            "files_scanned_avg": 100,
            "data_scanned_mb_avg": 1000,
            "query_time_ms_avg": 5000,
        }

        after_stats = {
            "files_scanned_avg": 20,
            "data_scanned_mb_avg": 200,
            "query_time_ms_avg": 1000,
        }

        effectiveness = optimizer.calculate_effectiveness(before_stats, after_stats)

        assert effectiveness["file_reduction_pct"] == 80
        assert effectiveness["data_reduction_pct"] == 80
        assert effectiveness["time_reduction_pct"] == 80

    def test_z_curve_encoding(self, spark):
        """Test Z-curve encoding for values."""
        optimizer = ZOrderOptimizer(spark)

        # Test 2D points
        points = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 2),
        ]

        z_values = [optimizer.encode_z_value(x, y) for x, y in points]

        # Z-values should preserve locality
        assert z_values[0] < z_values[-1]  # (0,0) before (2,2)
        assert len(set(z_values)) == len(z_values)  # All unique


class TestCompactionStrategy:
    """Test file compaction strategies."""

    def test_bin_packing_compaction(self):
        """Test bin packing algorithm for compaction."""
        strategy = CompactionStrategy(target_file_size_mb=128)

        files = [
            {"name": "file1", "size_mb": 30},
            {"name": "file2", "size_mb": 50},
            {"name": "file3", "size_mb": 40},
            {"name": "file4", "size_mb": 60},
            {"name": "file5", "size_mb": 20},
            {"name": "file6", "size_mb": 70},
        ]

        bins = strategy.bin_packing(files)

        # Check all files are assigned
        all_files = set()
        for bin_files in bins:
            all_files.update(f["name"] for f in bin_files)
        assert len(all_files) == len(files)

        # Check bin sizes don't exceed target too much
        for bin_files in bins:
            total_size = sum(f["size_mb"] for f in bin_files)
            assert total_size <= strategy.target_file_size_mb * 1.2  # Allow 20% overflow

    def test_adaptive_compaction(self):
        """Test adaptive compaction based on workload."""
        strategy = CompactionStrategy(target_file_size_mb=128)

        # Simulate different workloads
        workloads = [
            {"type": "streaming", "avg_file_size_mb": 10, "write_frequency": "high"},
            {"type": "batch", "avg_file_size_mb": 100, "write_frequency": "low"},
            {"type": "mixed", "avg_file_size_mb": 50, "write_frequency": "medium"},
        ]

        for workload in workloads:
            target_size = strategy.adapt_target_size(workload)

            if workload["type"] == "streaming":
                assert target_size < 128  # Smaller files for streaming
            elif workload["type"] == "batch":
                assert target_size >= 128  # Larger files for batch

    def test_incremental_compaction(self):
        """Test incremental compaction strategy."""
        strategy = CompactionStrategy(target_file_size_mb=128)

        # Files with timestamps
        files = [
            {"name": f"file{i}", "size_mb": 10, "created": datetime.now() - timedelta(hours=i)}
            for i in range(20)
        ]

        # Compact only old files
        files_to_compact = strategy.select_incremental(
            files, max_files=10, min_age_hours=5
        )

        assert len(files_to_compact) <= 10
        assert all(f["created"] < datetime.now() - timedelta(hours=5) for f in files_to_compact)


class TestOptimizationPlan:
    """Test optimization plan generation and execution."""

    def test_create_optimization_plan(self, spark):
        """Test creating comprehensive optimization plan."""
        planner = OptimizationPlan(spark)

        table_stats = {
            "table_name": "sales",
            "size_gb": 100,
            "num_files": 10000,
            "num_partitions": 365,
            "avg_file_size_mb": 10,
            "deleted_files": 500,
            "last_optimized": datetime.now() - timedelta(days=7),
        }

        plan = planner.create_plan(table_stats)

        assert "compaction" in plan
        assert "vacuum" in plan
        assert "z_order" in plan
        assert "estimated_duration_minutes" in plan

    def test_prioritize_optimizations(self, spark):
        """Test prioritizing optimization tasks."""
        planner = OptimizationPlan(spark)

        tasks = [
            {"type": "compaction", "priority": "medium", "impact": "high"},
            {"type": "vacuum", "priority": "low", "impact": "low"},
            {"type": "z_order", "priority": "high", "impact": "high"},
            {"type": "stats", "priority": "high", "impact": "medium"},
        ]

        prioritized = planner.prioritize_tasks(tasks)

        # Z-order should be first (high priority, high impact)
        assert prioritized[0]["type"] == "z_order"
        # Vacuum should be last (low priority, low impact)
        assert prioritized[-1]["type"] == "vacuum"

    def test_estimate_optimization_cost(self, spark):
        """Test estimating cost of optimizations."""
        planner = OptimizationPlan(spark)

        optimization = {
            "type": "compaction",
            "num_files": 1000,
            "total_size_gb": 10,
            "target_files": 50,
        }

        cost = planner.estimate_cost(optimization)

        assert "compute_hours" in cost
        assert "io_gb" in cost
        assert "estimated_cost_usd" in cost
        assert cost["compute_hours"] > 0
        assert cost["io_gb"] >= 10  # At least the data size

    def test_optimization_schedule(self, spark):
        """Test scheduling optimizations."""
        planner = OptimizationPlan(spark)

        tables = [
            {"name": "fact_sales", "priority": "high", "size_gb": 1000},
            {"name": "dim_customer", "priority": "medium", "size_gb": 10},
            {"name": "staging_temp", "priority": "low", "size_gb": 1},
        ]

        schedule = planner.create_schedule(tables, maintenance_window_hours=8)

        assert len(schedule) > 0
        assert schedule[0]["name"] == "fact_sales"  # High priority first

        # Check schedule fits in maintenance window
        total_duration = sum(task.get("estimated_duration_hours", 0) for task in schedule)
        assert total_duration <= 8