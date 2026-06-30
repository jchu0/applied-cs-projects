"""Enterprise features for the lakehouse."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

# Lazy pyspark imports
PYSPARK_AVAILABLE = False
try:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql.functions import col, expr, sum as spark_sum
    PYSPARK_AVAILABLE = True
except ImportError:
    DataFrame = Any
    SparkSession = Any
    col = expr = spark_sum = None


def _require_pyspark():
    """Raise an error if pyspark is not available."""
    if not PYSPARK_AVAILABLE:
        raise ImportError(
            "PySpark is required for enterprise features. "
            "Install it with: pip install pyspark>=3.4.0 delta-spark>=2.4.0"
        )


# --- Workflow Orchestration ---

@dataclass
class TaskConfig:
    """Configuration for a lakehouse task."""

    task_id: str
    task_type: str  # ingestion, transformation, aggregation, optimization
    source_path: Optional[str] = None
    target_path: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    retry_count: int = 2
    retry_delay_minutes: int = 5
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAGConfig:
    """Configuration for a lakehouse DAG."""

    dag_id: str
    schedule: str
    tasks: List[TaskConfig]
    default_args: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class WorkflowOrchestrator:
    """Orchestrate lakehouse workflows."""

    def __init__(self, spark: "SparkSession"):
        _require_pyspark()
        self.spark = spark
        self.registered_tasks: Dict[str, Callable] = {}

    def register_task(self, task_type: str, handler: Callable) -> None:
        """Register a task handler."""
        self.registered_tasks[task_type] = handler

    def create_dag_definition(self, config: DAGConfig) -> Dict:
        """
        Create an Airflow-compatible DAG definition.

        Args:
            config: DAG configuration

        Returns:
            Dict representation of DAG for serialization
        """
        return {
            "dag_id": config.dag_id,
            "schedule_interval": config.schedule,
            "default_args": {
                "owner": config.default_args.get("owner", "data-platform"),
                "depends_on_past": config.default_args.get("depends_on_past", True),
                "email_on_failure": config.default_args.get("email_on_failure", True),
                "retries": config.default_args.get("retries", 2),
                "retry_delay_minutes": config.default_args.get("retry_delay", 5),
            },
            "tags": config.tags,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "source_path": task.source_path,
                    "target_path": task.target_path,
                    "depends_on": task.depends_on,
                    "params": task.params,
                }
                for task in config.tasks
            ],
        }

    def execute_task(self, task: TaskConfig) -> Dict:
        """
        Execute a single task.

        Args:
            task: Task configuration

        Returns:
            Execution result
        """
        handler = self.registered_tasks.get(task.task_type)
        if not handler:
            raise ValueError(f"Unknown task type: {task.task_type}")

        start_time = datetime.utcnow()
        try:
            result = handler(
                spark=self.spark,
                source_path=task.source_path,
                target_path=task.target_path,
                **task.params,
            )
            status = "success"
            error = None
        except Exception as e:
            result = None
            status = "failed"
            error = str(e)

        end_time = datetime.utcnow()

        return {
            "task_id": task.task_id,
            "status": status,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "result": result,
            "error": error,
        }

    def execute_dag(self, config: DAGConfig) -> Dict:
        """
        Execute all tasks in a DAG respecting dependencies.

        Args:
            config: DAG configuration

        Returns:
            Execution results for all tasks
        """
        results = {}
        completed = set()

        # Simple topological execution
        remaining = list(config.tasks)

        while remaining:
            executed_this_round = False

            for task in remaining[:]:
                # Check if all dependencies are satisfied
                if all(dep in completed for dep in task.depends_on):
                    result = self.execute_task(task)
                    results[task.task_id] = result

                    if result["status"] == "success":
                        completed.add(task.task_id)
                        remaining.remove(task)
                        executed_this_round = True
                    else:
                        # Task failed - don't execute dependents
                        remaining.remove(task)
                        executed_this_round = True

            if not executed_this_round and remaining:
                # Circular dependency or blocked
                raise RuntimeError(f"Cannot execute remaining tasks: {[t.task_id for t in remaining]}")

        return results


# --- Monitoring and Alerting ---

@dataclass
class TableMetrics:
    """Metrics for a lakehouse table."""

    table_path: str
    timestamp: datetime
    row_count: int
    file_count: int
    size_bytes: int
    version: int
    last_modified: datetime
    partition_count: int = 0
    avg_file_size_bytes: float = 0


@dataclass
class PipelineMetrics:
    """Metrics for a pipeline run."""

    pipeline_id: str
    run_id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    rows_processed: int = 0
    bytes_processed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0


@dataclass
class Alert:
    """Alert definition."""

    alert_id: str
    severity: str  # info, warning, error, critical
    condition: str
    threshold: Any
    message: str


class LakehouseMonitor:
    """Monitor lakehouse tables and pipelines."""

    def __init__(self, spark: "SparkSession"):
        _require_pyspark()
        self.spark = spark
        self.alerts: List[Alert] = []

    def get_table_metrics(self, table_path: str) -> TableMetrics:
        """
        Get current metrics for a table.

        Args:
            table_path: Path to Delta table

        Returns:
            TableMetrics
        """
        from delta import DeltaTable

        table = DeltaTable.forPath(self.spark, table_path)

        # Get details
        details = self.spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`").collect()[0]

        # Get row count
        row_count = table.toDF().count()

        # Get current version
        history = table.history(1).collect()[0]
        version = history.version
        last_modified = history.timestamp

        # Get partition count
        partition_columns = details.partitionColumns
        if partition_columns:
            partition_count = (
                table.toDF()
                .select(partition_columns)
                .distinct()
                .count()
            )
        else:
            partition_count = 0

        num_files = details.numFiles
        size_bytes = details.sizeInBytes if details.sizeInBytes else 0
        avg_file_size = size_bytes / num_files if num_files > 0 else 0

        return TableMetrics(
            table_path=table_path,
            timestamp=datetime.utcnow(),
            row_count=row_count,
            file_count=num_files,
            size_bytes=size_bytes,
            version=version,
            last_modified=last_modified,
            partition_count=partition_count,
            avg_file_size_bytes=avg_file_size,
        )

    def register_alert(self, alert: Alert) -> None:
        """Register an alert."""
        self.alerts.append(alert)

    def check_alerts(self, metrics: TableMetrics) -> List[Dict]:
        """
        Check all alerts against current metrics.

        Args:
            metrics: Current table metrics

        Returns:
            List of triggered alerts
        """
        triggered = []

        for alert in self.alerts:
            try:
                # Evaluate condition (simplified - production would use safe eval)
                metric_value = getattr(metrics, alert.condition, None)
                if metric_value is not None:
                    if isinstance(alert.threshold, (int, float)):
                        if metric_value > alert.threshold:
                            triggered.append({
                                "alert_id": alert.alert_id,
                                "severity": alert.severity,
                                "message": alert.message,
                                "current_value": metric_value,
                                "threshold": alert.threshold,
                            })
            except Exception:
                pass

        return triggered

    def get_history_summary(
        self,
        table_path: str,
        limit: int = 100,
    ) -> Dict:
        """
        Get summary of recent table history.

        Args:
            table_path: Path to Delta table
            limit: Number of history entries

        Returns:
            Summary statistics
        """
        from delta import DeltaTable

        table = DeltaTable.forPath(self.spark, table_path)
        history = table.history(limit)

        history_df = history.toPandas()

        return {
            "total_operations": len(history_df),
            "operation_counts": history_df["operation"].value_counts().to_dict(),
            "first_version": history_df["version"].min(),
            "latest_version": history_df["version"].max(),
            "time_range": {
                "start": str(history_df["timestamp"].min()),
                "end": str(history_df["timestamp"].max()),
            },
        }


# --- Cost Optimization ---

@dataclass
class CostReport:
    """Cost analysis report."""

    table_path: str
    storage_cost_estimate: float
    compute_cost_estimate: float
    optimization_potential: float
    recommendations: List[str]


class CostOptimizer:
    """Optimize storage and compute costs."""

    def __init__(self, spark: "SparkSession", storage_cost_per_gb: float = 0.023):
        _require_pyspark()
        self.spark = spark
        self.storage_cost_per_gb = storage_cost_per_gb

    def analyze_storage_costs(self, table_path: str) -> CostReport:
        """
        Analyze storage costs for a table.

        Args:
            table_path: Path to Delta table

        Returns:
            CostReport with analysis and recommendations
        """
        from delta import DeltaTable

        table = DeltaTable.forPath(self.spark, table_path)

        # Get current storage
        details = self.spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`").collect()[0]
        size_bytes = details.sizeInBytes if details.sizeInBytes else 0
        num_files = details.numFiles

        # Calculate costs
        size_gb = size_bytes / (1024 ** 3)
        monthly_storage_cost = size_gb * self.storage_cost_per_gb

        # Analyze optimization potential
        avg_file_size = size_bytes / num_files if num_files > 0 else 0
        target_file_size = 1024 * 1024 * 1024  # 1GB

        # Small files increase API costs
        if avg_file_size < target_file_size:
            optimization_potential = 0.1  # 10% savings potential
        else:
            optimization_potential = 0

        # Get version history for retention analysis
        history = table.history()
        version_count = history.count()

        # Generate recommendations
        recommendations = []

        if avg_file_size < 128 * 1024 * 1024:
            recommendations.append(
                f"Compact small files to reduce API costs. Current avg: {avg_file_size / 1024 / 1024:.1f}MB"
            )

        if version_count > 100:
            recommendations.append(
                f"Run VACUUM to remove {version_count} old versions and reduce storage."
            )

        if size_gb > 100:
            recommendations.append(
                "Consider partitioning strategy to improve query pruning and reduce scan costs."
            )

        return CostReport(
            table_path=table_path,
            storage_cost_estimate=monthly_storage_cost,
            compute_cost_estimate=0,  # Would need query history
            optimization_potential=monthly_storage_cost * optimization_potential,
            recommendations=recommendations,
        )

    def estimate_query_cost(
        self,
        table_path: str,
        query: str,
        cost_per_tb_scanned: float = 5.0,
    ) -> Dict:
        """
        Estimate cost of a query.

        Args:
            table_path: Path to Delta table
            query: SQL query
            cost_per_tb_scanned: Cost per TB scanned

        Returns:
            Cost estimate
        """
        # Get query plan
        df = self.spark.read.format("delta").load(table_path)
        df.createOrReplaceTempView("table")

        plan = self.spark.sql(f"EXPLAIN COST {query}")
        plan_text = plan.collect()[0][0]

        # Estimate bytes scanned from plan (simplified)
        # In production, would parse the plan properly
        details = self.spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`").collect()[0]
        size_bytes = details.sizeInBytes if details.sizeInBytes else 0

        # Assume full scan for estimate (partition pruning would reduce this)
        tb_scanned = size_bytes / (1024 ** 4)
        estimated_cost = tb_scanned * cost_per_tb_scanned

        return {
            "estimated_bytes_scanned": size_bytes,
            "estimated_tb_scanned": tb_scanned,
            "estimated_cost": estimated_cost,
            "plan": plan_text,
        }

    def apply_cost_optimizations(
        self,
        table_path: str,
        vacuum_retention_hours: int = 168,
        enable_auto_optimize: bool = True,
    ) -> Dict:
        """
        Apply cost optimization settings to a table.

        Args:
            table_path: Path to Delta table
            vacuum_retention_hours: Retention for vacuum
            enable_auto_optimize: Enable auto-optimization

        Returns:
            Applied optimizations
        """
        from delta import DeltaTable

        optimizations = []

        # Enable auto-optimize
        if enable_auto_optimize:
            self.spark.sql(f"""
                ALTER TABLE delta.`{table_path}`
                SET TBLPROPERTIES (
                    'delta.autoOptimize.autoCompact' = 'true',
                    'delta.autoOptimize.optimizeWrite' = 'true'
                )
            """)
            optimizations.append("Enabled auto-compaction and optimized writes")

        # Configure retention
        self.spark.sql(f"""
            ALTER TABLE delta.`{table_path}`
            SET TBLPROPERTIES (
                'delta.logRetentionDuration' = 'interval 30 days',
                'delta.deletedFileRetentionDuration' = 'interval {vacuum_retention_hours} hours'
            )
        """)
        optimizations.append(f"Set retention to 30 days logs, {vacuum_retention_hours}h deleted files")

        # Run vacuum
        table = DeltaTable.forPath(self.spark, table_path)
        table.vacuum(vacuum_retention_hours)
        optimizations.append(f"Vacuumed old files (>{vacuum_retention_hours}h)")

        # Run optimize
        table.optimize().executeCompaction()
        optimizations.append("Compacted small files")

        return {
            "table_path": table_path,
            "optimizations_applied": optimizations,
        }
