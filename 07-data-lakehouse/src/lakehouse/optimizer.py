"""Optimization features for the lakehouse."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Lazy pyspark imports
PYSPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, sum as spark_sum
    PYSPARK_AVAILABLE = True
except ImportError:
    SparkSession = Any
    col = spark_sum = None


def _require_pyspark():
    """Raise an error if pyspark is not available."""
    if not PYSPARK_AVAILABLE:
        raise ImportError(
            "PySpark is required for table optimization. "
            "Install it with: pip install pyspark>=3.4.0 delta-spark>=2.4.0"
        )


@dataclass
class StorageReport:
    """Storage analysis report for a table."""

    total_size_bytes: int
    file_count: int
    avg_file_size_bytes: float
    small_file_count: int
    version_count: int
    recommendations: List[str]


@dataclass
class OptimizationMetrics:
    """Metrics from an optimization operation."""

    num_files_added: int
    num_files_removed: int
    files_added_size: int
    files_removed_size: int
    z_order_stats: Optional[Dict] = None


class TableOptimizer:
    """Optimize Delta Lake tables for better performance."""

    def __init__(self, spark: "SparkSession"):
        _require_pyspark()
        self.spark = spark

    def optimize_table(
        self,
        table_path: str,
        partition_filter: Optional[str] = None,
        z_order_columns: Optional[List[str]] = None,
    ) -> OptimizationMetrics:
        """
        Optimize table files with optional Z-ordering.

        Args:
            table_path: Path to Delta table
            partition_filter: Optional partition filter (e.g., "date = '2024-01-01'")
            z_order_columns: Columns to Z-order by for multi-dimensional clustering

        Returns:
            Optimization metrics
        """
        from delta import DeltaTable

        table = DeltaTable.forPath(self.spark, table_path)
        optimize_builder = table.optimize()

        if partition_filter:
            optimize_builder = optimize_builder.where(partition_filter)

        if z_order_columns:
            metrics = optimize_builder.zOrderBy(z_order_columns).executeCompaction()
        else:
            metrics = optimize_builder.executeCompaction()

        # Extract metrics from result
        metrics_row = metrics.select("metrics.*").collect()[0]

        return OptimizationMetrics(
            num_files_added=metrics_row.numFilesAdded,
            num_files_removed=metrics_row.numFilesRemoved,
            files_added_size=metrics_row.numBytesAdded if hasattr(metrics_row, 'numBytesAdded') else 0,
            files_removed_size=metrics_row.numBytesRemoved if hasattr(metrics_row, 'numBytesRemoved') else 0,
        )

    def vacuum_table(
        self,
        table_path: str,
        retention_hours: int = 168,
        dry_run: bool = False,
    ) -> List[str]:
        """
        Remove old files not in current table state.

        Args:
            table_path: Path to Delta table
            retention_hours: Hours to retain deleted files (default 7 days)
            dry_run: If True, only list files to be deleted

        Returns:
            List of files deleted (or to be deleted if dry_run)
        """
        from delta import DeltaTable

        table = DeltaTable.forPath(self.spark, table_path)

        if dry_run:
            files = table.vacuum(retention_hours, dryRun=True)
            return [row.path for row in files.collect()]
        else:
            table.vacuum(retention_hours)
            return []

    def analyze_storage(self, table_path: str) -> StorageReport:
        """
        Analyze storage efficiency of a table.

        Args:
            table_path: Path to Delta table

        Returns:
            StorageReport with analysis and recommendations
        """
        from delta import DeltaTable

        table = DeltaTable.forPath(self.spark, table_path)

        # Get table details
        details = self.spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`").collect()[0]

        num_files = details.numFiles
        size_bytes = details.sizeInBytes if details.sizeInBytes else 0

        avg_file_size = size_bytes / num_files if num_files > 0 else 0

        # Count small files (< 128MB)
        small_file_threshold = 128 * 1024 * 1024

        # Get file list from Delta log
        df = self.spark.read.format("delta").load(table_path)
        file_stats = df.select("_metadata.file_path", "_metadata.file_size").distinct()

        small_files = file_stats.filter(col("file_size") < small_file_threshold).count()

        # Get version count
        history = table.history()
        version_count = history.count()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_file_size, small_files, version_count
        )

        return StorageReport(
            total_size_bytes=size_bytes,
            file_count=num_files,
            avg_file_size_bytes=avg_file_size,
            small_file_count=small_files,
            version_count=version_count,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self, avg_file_size: float, small_files: int, versions: int
    ) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []

        target_size = 1024 * 1024 * 1024  # 1GB target

        if avg_file_size < target_size * 0.5:
            recommendations.append(
                f"Run OPTIMIZE to compact files. Current avg: {avg_file_size / 1024 / 1024:.1f}MB"
            )

        if small_files > 10:
            recommendations.append(
                f"Found {small_files} small files (<128MB). Consider auto-compaction."
            )

        if versions > 100:
            recommendations.append(
                f"Found {versions} versions. Consider running VACUUM to clean old files."
            )

        if not recommendations:
            recommendations.append("Table is well optimized.")

        return recommendations

    def configure_auto_optimize(
        self,
        table_path: str,
        auto_compact: bool = True,
        optimize_write: bool = True,
        target_file_size_mb: int = 1024,
    ) -> None:
        """
        Configure automatic optimization settings for a table.

        Args:
            table_path: Path to Delta table
            auto_compact: Enable automatic file compaction
            optimize_write: Enable optimized writes
            target_file_size_mb: Target file size in MB
        """
        self.spark.sql(f"""
            ALTER TABLE delta.`{table_path}`
            SET TBLPROPERTIES (
                'delta.autoOptimize.autoCompact' = '{str(auto_compact).lower()}',
                'delta.autoOptimize.optimizeWrite' = '{str(optimize_write).lower()}',
                'delta.targetFileSize' = '{target_file_size_mb * 1024 * 1024}'
            )
        """)

    def configure_retention(
        self,
        table_path: str,
        log_retention_days: int = 30,
        deleted_file_retention_hours: int = 168,
    ) -> None:
        """
        Configure retention settings for a table.

        Args:
            table_path: Path to Delta table
            log_retention_days: How long to keep transaction logs
            deleted_file_retention_hours: How long to keep deleted files
        """
        self.spark.sql(f"""
            ALTER TABLE delta.`{table_path}`
            SET TBLPROPERTIES (
                'delta.logRetentionDuration' = 'interval {log_retention_days} days',
                'delta.deletedFileRetentionDuration' = 'interval {deleted_file_retention_hours} hours'
            )
        """)


class PartitionOptimizer:
    """Optimize partition strategies for Delta tables."""

    def __init__(self, spark: "SparkSession"):
        _require_pyspark()
        self.spark = spark

    def analyze_partition_strategy(
        self,
        table_path: str,
        candidate_columns: List[str],
    ) -> Dict[str, Dict]:
        """
        Analyze candidate columns for partitioning.

        Args:
            table_path: Path to Delta table
            candidate_columns: Columns to analyze

        Returns:
            Dict with cardinality and recommendations for each column
        """
        df = self.spark.read.format("delta").load(table_path)

        results = {}
        for column in candidate_columns:
            cardinality = df.select(column).distinct().count()

            if cardinality < 100:
                recommendation = "Good partition candidate"
            elif cardinality < 1000:
                recommendation = "Consider coarser granularity (e.g., year-month)"
            elif cardinality < 10000:
                recommendation = "High cardinality - consider Z-ordering instead"
            else:
                recommendation = "Too high cardinality for partitioning"

            results[column] = {
                "cardinality": cardinality,
                "recommendation": recommendation,
            }

        return results

    def get_partition_stats(self, table_path: str) -> Dict:
        """
        Get statistics about current partitions.

        Args:
            table_path: Path to Delta table

        Returns:
            Dict with partition statistics
        """
        df = self.spark.read.format("delta").load(table_path)

        # Get partition columns from table
        details = self.spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`").collect()[0]
        partition_columns = details.partitionColumns

        if not partition_columns:
            return {"partitioned": False}

        # Count files per partition
        partition_counts = (
            df.select("_metadata.file_path", *partition_columns)
            .distinct()
            .groupBy(partition_columns)
            .count()
        )

        stats = partition_counts.agg(
            spark_sum("count").alias("total_files"),
        ).collect()[0]

        num_partitions = partition_counts.count()

        return {
            "partitioned": True,
            "partition_columns": partition_columns,
            "num_partitions": num_partitions,
            "total_files": stats.total_files,
            "avg_files_per_partition": stats.total_files / num_partitions if num_partitions > 0 else 0,
        }


class CompactionScheduler:
    """Schedule and manage automatic compaction."""

    def __init__(self, spark: "SparkSession"):
        _require_pyspark()
        self.spark = spark
        self.optimizer = TableOptimizer(spark)

    def should_compact(
        self,
        table_path: str,
        max_small_files: int = 10,
        min_file_size_mb: float = 128,
        max_versions_since_compact: int = 50,
    ) -> bool:
        """
        Check if table needs compaction based on heuristics.

        Args:
            table_path: Path to Delta table
            max_small_files: Threshold for small file count
            min_file_size_mb: Minimum acceptable average file size
            max_versions_since_compact: Max versions since last compaction

        Returns:
            True if compaction is recommended
        """
        report = self.optimizer.analyze_storage(table_path)

        avg_size_mb = report.avg_file_size_bytes / (1024 * 1024)

        return (
            report.small_file_count > max_small_files
            or avg_size_mb < min_file_size_mb
            or report.version_count > max_versions_since_compact
        )

    def compact_if_needed(
        self,
        table_path: str,
        z_order_columns: Optional[List[str]] = None,
        **thresholds,
    ) -> Optional[OptimizationMetrics]:
        """
        Compact table if it meets compaction criteria.

        Args:
            table_path: Path to Delta table
            z_order_columns: Optional columns for Z-ordering
            **thresholds: Threshold overrides for should_compact

        Returns:
            OptimizationMetrics if compaction was performed, None otherwise
        """
        if self.should_compact(table_path, **thresholds):
            return self.optimizer.optimize_table(
                table_path, z_order_columns=z_order_columns
            )
        return None

    def schedule_maintenance(
        self,
        table_path: str,
        vacuum_retention_hours: int = 168,
        z_order_columns: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run full maintenance on a table: analyze, compact, vacuum.

        Args:
            table_path: Path to Delta table
            vacuum_retention_hours: Retention for vacuum
            z_order_columns: Optional columns for Z-ordering

        Returns:
            Dict with maintenance results
        """
        results = {}

        # Analyze
        report = self.optimizer.analyze_storage(table_path)
        results["analysis"] = {
            "file_count": report.file_count,
            "avg_file_size_mb": report.avg_file_size_bytes / (1024 * 1024),
            "small_files": report.small_file_count,
            "recommendations": report.recommendations,
        }

        # Compact if needed
        if report.small_file_count > 10 or report.avg_file_size_bytes < 128 * 1024 * 1024:
            metrics = self.optimizer.optimize_table(
                table_path, z_order_columns=z_order_columns
            )
            results["optimization"] = {
                "files_added": metrics.num_files_added,
                "files_removed": metrics.num_files_removed,
            }

        # Vacuum
        self.optimizer.vacuum_table(table_path, vacuum_retention_hours)
        results["vacuum"] = {"retention_hours": vacuum_retention_hours}

        return results


# =============================================================================
# Additional Classes for Test Compatibility
# =============================================================================


class StorageOptimizer(TableOptimizer):
    """Storage optimizer extending TableOptimizer with additional features.

    Provides storage analysis and optimization for Delta Lake tables.
    """

    def identify_small_files(
        self,
        files: List[Dict],
        threshold_mb: float = 128,
    ) -> List[Dict]:
        """Identify files smaller than threshold.

        Args:
            files: List of file dicts with 'path' and 'size' keys
            threshold_mb: Size threshold in MB

        Returns:
            List of files smaller than threshold
        """
        threshold_bytes = threshold_mb * 1024 * 1024
        return [f for f in files if f.get("size", 0) < threshold_bytes]

    def calculate_compaction_bins(
        self,
        files: List[Dict],
        target_size_mb: float = 1024,
    ) -> List[List[Dict]]:
        """Group small files into bins for compaction.

        Args:
            files: List of file dicts with 'path' and 'size' keys
            target_size_mb: Target bin size in MB

        Returns:
            List of file groups (bins) for compaction
        """
        target_bytes = target_size_mb * 1024 * 1024
        bins = []
        current_bin = []
        current_size = 0

        for f in sorted(files, key=lambda x: x.get("size", 0)):
            if current_size + f.get("size", 0) > target_bytes and current_bin:
                bins.append(current_bin)
                current_bin = []
                current_size = 0
            current_bin.append(f)
            current_size += f.get("size", 0)

        if current_bin:
            bins.append(current_bin)

        return bins

    def estimate_compaction_benefit(
        self,
        table_path: str,
    ) -> Dict:
        """Estimate benefit of running compaction.

        Args:
            table_path: Path to Delta table

        Returns:
            Dict with estimated benefits
        """
        report = self.analyze_storage(table_path)

        # Estimate file reduction
        current_files = report.file_count
        optimal_files = max(1, report.total_size_bytes // (1024 * 1024 * 1024))  # 1GB target

        return {
            "current_files": current_files,
            "estimated_files_after": optimal_files,
            "file_reduction_percent": (
                (current_files - optimal_files) / current_files * 100
                if current_files > 0 else 0
            ),
            "small_files_to_compact": report.small_file_count,
        }


class QueryOptimizer:
    """Optimizer for Delta Lake queries.

    Analyzes and optimizes query patterns for better performance.
    """

    def __init__(self, spark: "SparkSession"):
        _require_pyspark()
        self.spark = spark
        self._query_stats = []

    def analyze_query_plan(self, query: str) -> Dict:
        """Analyze query execution plan.

        Args:
            query: SQL query string

        Returns:
            Dict with plan analysis
        """
        df = self.spark.sql(f"EXPLAIN EXTENDED {query}")
        plan = df.collect()[0][0] if df.count() > 0 else ""

        return {
            "plan": plan,
            "uses_partition_pruning": "PartitionFilters" in plan,
            "uses_data_skipping": "DataFilters" in plan,
            "estimated_rows": self._extract_row_estimate(plan),
        }

    def _extract_row_estimate(self, plan: str) -> int:
        """Extract estimated row count from plan."""
        # Simplified extraction
        import re
        match = re.search(r"Statistics\(sizeInBytes=\d+, rowCount=(\d+)", plan)
        return int(match.group(1)) if match else 0

    def suggest_indexes(
        self,
        table_path: str,
        sample_queries: List[str],
    ) -> List[str]:
        """Suggest columns for Z-ordering based on query patterns.

        Args:
            table_path: Path to Delta table
            sample_queries: List of common queries

        Returns:
            List of recommended Z-order columns
        """
        column_usage = {}

        for query in sample_queries:
            # Simple parsing - look for WHERE clause columns
            import re
            where_matches = re.findall(r'WHERE.*?(\w+)\s*[=<>]', query, re.IGNORECASE)
            for col in where_matches:
                column_usage[col] = column_usage.get(col, 0) + 1

        # Sort by usage frequency
        sorted_cols = sorted(column_usage.items(), key=lambda x: x[1], reverse=True)
        return [col for col, _ in sorted_cols[:3]]  # Top 3 columns

    def profile_query(self, query: str) -> Dict:
        """Profile query execution.

        Args:
            query: SQL query string

        Returns:
            Dict with profiling results
        """
        import time

        start = time.time()
        df = self.spark.sql(query)
        count = df.count()
        elapsed = time.time() - start

        result = {
            "query": query,
            "row_count": count,
            "execution_time_seconds": elapsed,
            "rows_per_second": count / elapsed if elapsed > 0 else 0,
        }

        self._query_stats.append(result)
        return result

    def get_query_stats(self) -> List[Dict]:
        """Get accumulated query statistics."""
        return self._query_stats


class ZOrderOptimizer:
    """Optimizer for Z-ordering column selection.

    Analyzes query patterns and column cardinality to recommend
    optimal Z-order columns.
    """

    def __init__(self, spark: "SparkSession"):
        _require_pyspark()
        self.spark = spark

    def analyze_column_for_zorder(
        self,
        table_path: str,
        column: str,
    ) -> Dict:
        """Analyze a column's suitability for Z-ordering.

        Args:
            table_path: Path to Delta table
            column: Column name to analyze

        Returns:
            Dict with analysis results
        """
        df = self.spark.read.format("delta").load(table_path)

        cardinality = df.select(column).distinct().count()
        total_rows = df.count()

        selectivity = cardinality / total_rows if total_rows > 0 else 0

        if selectivity < 0.01:
            recommendation = "Excellent for Z-ordering"
        elif selectivity < 0.1:
            recommendation = "Good for Z-ordering"
        elif selectivity < 0.5:
            recommendation = "Moderate benefit from Z-ordering"
        else:
            recommendation = "Low benefit - consider partitioning instead"

        return {
            "column": column,
            "cardinality": cardinality,
            "selectivity": selectivity,
            "recommendation": recommendation,
        }

    def recommend_zorder_columns(
        self,
        table_path: str,
        candidate_columns: List[str],
        max_columns: int = 4,
    ) -> List[str]:
        """Recommend columns for Z-ordering.

        Args:
            table_path: Path to Delta table
            candidate_columns: Columns to consider
            max_columns: Maximum columns to recommend

        Returns:
            List of recommended columns in priority order
        """
        analyses = []
        for col in candidate_columns:
            analysis = self.analyze_column_for_zorder(table_path, col)
            analyses.append((col, analysis["selectivity"]))

        # Sort by selectivity (lower is better for Z-ordering)
        sorted_cols = sorted(analyses, key=lambda x: x[1])
        return [col for col, _ in sorted_cols[:max_columns]]

    def estimate_zorder_benefit(
        self,
        table_path: str,
        columns: List[str],
        sample_queries: Optional[List[str]] = None,
    ) -> Dict:
        """Estimate benefit of Z-ordering for given columns.

        Args:
            table_path: Path to Delta table
            columns: Columns to Z-order by
            sample_queries: Optional queries to analyze

        Returns:
            Dict with estimated benefits
        """
        df = self.spark.read.format("delta").load(table_path)

        # Get file count and size
        details = self.spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`").collect()[0]

        return {
            "columns": columns,
            "estimated_file_skip_rate": 0.7,  # Typical improvement
            "current_file_count": details.numFiles,
            "recommendation": (
                "Z-ordering on these columns should improve query performance "
                "for queries filtering on these columns"
            ),
        }


@dataclass
class CompactionStrategy:
    """Strategy configuration for table compaction.

    Attributes:
        target_file_size_mb: Target file size after compaction
        max_concurrent_files: Maximum files to compact at once
        enable_zorder: Whether to apply Z-ordering during compaction
        zorder_columns: Columns for Z-ordering
        partition_filter: Optional partition filter
    """

    target_file_size_mb: int = 1024
    max_concurrent_files: int = 100
    enable_zorder: bool = False
    zorder_columns: Optional[List[str]] = None
    partition_filter: Optional[str] = None

    def should_compact(self, file_count: int, avg_size_mb: float) -> bool:
        """Check if compaction should be performed.

        Args:
            file_count: Current number of files
            avg_size_mb: Current average file size in MB

        Returns:
            True if compaction is recommended
        """
        # Compact if too many small files or average size is too small
        return file_count > 10 and avg_size_mb < self.target_file_size_mb * 0.5

    def bin_packing(self, files: List[Dict]) -> List[List[Dict]]:
        """Group files into bins using bin packing algorithm.

        Args:
            files: List of file dicts with 'name' and 'size_mb' keys

        Returns:
            List of bins, each containing files that fit within target size
        """
        # Sort files by size (descending) for better bin packing
        sorted_files = sorted(files, key=lambda x: x.get("size_mb", 0), reverse=True)
        bins: List[List[Dict]] = []
        bin_sizes: List[float] = []

        for f in sorted_files:
            size = f.get("size_mb", 0)
            placed = False

            # Try to place in existing bin (first fit decreasing)
            for i, bin_size in enumerate(bin_sizes):
                if bin_size + size <= self.target_file_size_mb * 1.2:  # Allow 20% overflow
                    bins[i].append(f)
                    bin_sizes[i] += size
                    placed = True
                    break

            # Create new bin if needed
            if not placed:
                bins.append([f])
                bin_sizes.append(size)

        return bins

    def adapt_target_size(self, workload: Dict) -> int:
        """Adapt target file size based on workload characteristics.

        Args:
            workload: Dict with 'type', 'avg_file_size_mb', 'write_frequency'

        Returns:
            Adapted target file size in MB
        """
        workload_type = workload.get("type", "mixed")
        write_frequency = workload.get("write_frequency", "medium")

        base_size = self.target_file_size_mb

        if workload_type == "streaming":
            # Smaller files for streaming to reduce latency
            if write_frequency == "high":
                return base_size // 4  # 32MB for high-frequency streaming
            return base_size // 2  # 64MB for normal streaming
        elif workload_type == "batch":
            # Larger files for batch to improve throughput
            if write_frequency == "low":
                return base_size * 2  # 256MB for low-frequency batch
            return base_size  # 128MB for normal batch
        else:  # mixed
            return base_size  # Default size

    def select_incremental(
        self,
        files: List[Dict],
        max_files: int = 100,
        min_age_hours: int = 1,
    ) -> List[Dict]:
        """Select files for incremental compaction based on age.

        Args:
            files: List of file dicts with 'name', 'size_mb', 'created' keys
            max_files: Maximum number of files to select
            min_age_hours: Minimum age in hours for files to be eligible

        Returns:
            List of files eligible for compaction
        """
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(hours=min_age_hours)

        # Filter files older than min_age_hours
        eligible_files = [
            f for f in files
            if f.get("created", datetime.now()) < cutoff
        ]

        # Sort by age (oldest first) and limit
        sorted_files = sorted(
            eligible_files,
            key=lambda x: x.get("created", datetime.now())
        )

        return sorted_files[:max_files]


class OptimizationPlan:
    """Optimization planner for Delta tables.

    Analyzes tables and creates optimization plans with scheduling
    and cost estimation.
    """

    def __init__(self, spark: "SparkSession"):
        """Initialize the optimization planner.

        Args:
            spark: SparkSession to use
        """
        _require_pyspark()
        self.spark = spark

    def create_plan(self, table_stats: Dict) -> Dict:
        """Create a comprehensive optimization plan for a table.

        Args:
            table_stats: Dict with table statistics including:
                - table_name, size_gb, num_files, num_partitions
                - avg_file_size_mb, deleted_files, last_optimized

        Returns:
            Dict with optimization plan including compaction, vacuum, z_order
        """
        plan = {
            "compaction": False,
            "vacuum": False,
            "z_order": False,
            "estimated_duration_minutes": 0,
            "operations": [],
        }

        num_files = table_stats.get("num_files", 0)
        avg_file_size_mb = table_stats.get("avg_file_size_mb", 0)
        deleted_files = table_stats.get("deleted_files", 0)
        size_gb = table_stats.get("size_gb", 0)

        # Check if compaction is needed
        if num_files > 100 or avg_file_size_mb < 128:
            plan["compaction"] = True
            plan["operations"].append("compaction")
            plan["estimated_duration_minutes"] += num_files * 0.1

        # Check if vacuum is needed
        if deleted_files > 0:
            plan["vacuum"] = True
            plan["operations"].append("vacuum")
            plan["estimated_duration_minutes"] += size_gb * 0.5

        # Check if Z-ordering would help
        if num_files > 50:
            plan["z_order"] = True
            plan["operations"].append("z_order")
            plan["estimated_duration_minutes"] += size_gb * 2

        return plan

    def prioritize_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """Prioritize optimization tasks by priority and impact.

        Args:
            tasks: List of task dicts with 'type', 'priority', 'impact'

        Returns:
            Sorted list of tasks (highest priority/impact first)
        """
        priority_order = {"high": 3, "medium": 2, "low": 1}
        impact_order = {"high": 3, "medium": 2, "low": 1}

        def task_score(task: Dict) -> tuple:
            priority = priority_order.get(task.get("priority", "medium"), 2)
            impact = impact_order.get(task.get("impact", "medium"), 2)
            return (priority, impact)

        return sorted(tasks, key=task_score, reverse=True)

    def estimate_cost(self, optimization: Dict) -> Dict:
        """Estimate cost of an optimization operation.

        Args:
            optimization: Dict with 'type', 'num_files', 'total_size_gb'

        Returns:
            Dict with compute_hours, io_gb, estimated_cost_usd
        """
        num_files = optimization.get("num_files", 0)
        total_size_gb = optimization.get("total_size_gb", 0)
        opt_type = optimization.get("type", "compaction")

        # Estimate compute hours based on operation type
        if opt_type == "compaction":
            compute_hours = (num_files * 0.001) + (total_size_gb * 0.05)
        elif opt_type == "vacuum":
            compute_hours = total_size_gb * 0.01
        elif opt_type == "z_order":
            compute_hours = total_size_gb * 0.1
        else:
            compute_hours = total_size_gb * 0.02

        # I/O is at least the data size, often 2x for read+write
        io_gb = total_size_gb * 2

        # Rough cost estimate: $0.10/compute-hour + $0.01/GB I/O
        estimated_cost = compute_hours * 0.10 + io_gb * 0.01

        return {
            "compute_hours": compute_hours,
            "io_gb": io_gb,
            "estimated_cost_usd": estimated_cost,
        }

    def create_schedule(
        self,
        tables: List[Dict],
        maintenance_window_hours: int = 8,
    ) -> List[Dict]:
        """Create an optimization schedule for multiple tables.

        Args:
            tables: List of table dicts with 'name', 'priority', 'size_gb'
            maintenance_window_hours: Available maintenance window in hours

        Returns:
            List of scheduled tasks that fit in the maintenance window
        """
        priority_order = {"high": 3, "medium": 2, "low": 1}

        # Sort tables by priority
        sorted_tables = sorted(
            tables,
            key=lambda t: priority_order.get(t.get("priority", "medium"), 2),
            reverse=True,
        )

        schedule = []
        total_hours = 0

        for table in sorted_tables:
            size_gb = table.get("size_gb", 1)
            # Estimate duration based on size
            estimated_hours = max(0.5, size_gb * 0.01)

            if total_hours + estimated_hours <= maintenance_window_hours:
                schedule.append({
                    "name": table["name"],
                    "priority": table.get("priority", "medium"),
                    "size_gb": size_gb,
                    "estimated_duration_hours": estimated_hours,
                })
                total_hours += estimated_hours

        return schedule
