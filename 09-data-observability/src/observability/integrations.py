"""Integrations with data orchestrators and tools."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from observability.models import Anomaly, AnomalyType

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for an integration."""

    name: str
    enabled: bool = True
    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}


class AirflowIntegration:
    """Integration with Apache Airflow."""

    def __init__(self, observability_client: "ObservabilityClient"):
        self.client = observability_client

    def create_lineage_callback(
        self, source_tables: List[str], target_table: str
    ) -> Callable:
        """Create callback for lineage tracking."""

        def callback(context: Dict[str, Any]) -> None:
            dag_id = context.get("dag", {})
            if hasattr(dag_id, "dag_id"):
                dag_id = dag_id.dag_id
            else:
                dag_id = str(dag_id)

            task_id = context.get("task", {})
            if hasattr(task_id, "task_id"):
                task_id = task_id.task_id
            else:
                task_id = str(task_id)

            run_id = context.get("run_id", "unknown")

            for source in source_tables:
                self.client.add_lineage(
                    source_table=source,
                    target_table=target_table,
                    transformation=f"airflow:{dag_id}.{task_id}",
                    context={
                        "run_id": run_id,
                        "execution_date": str(context.get("execution_date")),
                    },
                )

        return callback

    def create_quality_check(
        self, table_id: str, fail_on_critical: bool = True
    ) -> Callable:
        """Create quality check function for Airflow task."""

        def check_quality(**context) -> bool:
            # Run anomaly detection
            anomalies = self.client.check_table(table_id)

            if anomalies:
                for anomaly in anomalies:
                    logger.warning(
                        f"Anomaly detected: {anomaly.anomaly_type.value} - "
                        f"{anomaly.description}"
                    )
                    if fail_on_critical and anomaly.severity == "critical":
                        raise Exception(
                            f"Critical anomaly: {anomaly.description}"
                        )

            return True

        return check_quality

    def create_freshness_sensor_check(
        self, table_id: str, min_health_score: float = 80.0
    ) -> Callable:
        """Create check function for freshness sensor."""

        def check_freshness() -> bool:
            health = self.client.get_health(table_id)
            return health.freshness_score >= min_health_score

        return check_freshness


class DbtIntegration:
    """Integration with dbt."""

    def __init__(self, observability_client: "ObservabilityClient"):
        self.client = observability_client

    def process_manifest(self, manifest: Dict[str, Any]) -> None:
        """Process dbt manifest to extract lineage."""
        nodes = manifest.get("nodes", {})

        for node_id, node in nodes.items():
            if node.get("resource_type") == "model":
                target_table = self._get_table_id(node)

                # Extract upstream dependencies
                depends_on = node.get("depends_on", {}).get("nodes", [])
                for dep_id in depends_on:
                    dep_node = nodes.get(dep_id)
                    if dep_node:
                        source_table = self._get_table_id(dep_node)
                        self.client.add_lineage(
                            source_table=source_table,
                            target_table=target_table,
                            transformation=f"dbt:{node.get('name', node_id)}",
                        )

    def _get_table_id(self, node: Dict[str, Any]) -> str:
        """Get table ID from dbt node."""
        database = node.get("database", "")
        schema = node.get("schema", "")
        alias = node.get("alias") or node.get("name", "")
        return f"{database}.{schema}.{alias}"

    def process_run_results(self, results: Dict[str, Any]) -> None:
        """Process dbt run results for monitoring."""
        for result in results.get("results", []):
            status = result.get("status")
            node = result.get("node", {})

            if status == "error":
                logger.error(f"dbt model failed: {node.get('name')}")
                # Could trigger alert here


class SparkIntegration:
    """Integration with Apache Spark."""

    def __init__(self, observability_client: "ObservabilityClient"):
        self.client = observability_client

    def extract_lineage_from_plan(
        self, logical_plan: str, target_table: str
    ) -> List[str]:
        """Extract source tables from Spark logical plan."""
        # Simplified extraction - real implementation would parse plan
        source_tables = []

        # Look for table references in plan
        import re

        table_pattern = r"(\w+)\.(\w+)\.(\w+)"
        matches = re.findall(table_pattern, logical_plan)

        for match in matches:
            table_id = ".".join(match)
            if table_id != target_table:
                source_tables.append(table_id)

        return list(set(source_tables))

    def create_listener(self) -> "SparkLineageListener":
        """Create Spark listener for lineage tracking."""
        return SparkLineageListener(self.client)


class SparkLineageListener:
    """Spark listener for lineage tracking."""

    def __init__(self, client: "ObservabilityClient"):
        self.client = client

    def on_query_execution(
        self, query_id: str, plan: str, output_table: str
    ) -> None:
        """Handle query execution event."""
        # Extract lineage from query plan
        # In production, would parse Spark's LogicalPlan
        logger.info(f"Query executed: {query_id}")


class ObservabilityClient:
    """Client for Data Observability Platform."""

    def __init__(self, base_url: str = "", api_key: str = ""):
        self.base_url = base_url
        self.api_key = api_key
        self._lineage: List[Dict[str, Any]] = []
        self._checked_tables: Dict[str, List[Anomaly]] = {}

    def add_lineage(
        self,
        source_table: str,
        target_table: str,
        transformation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add lineage relationship."""
        self._lineage.append({
            "source": source_table,
            "target": target_table,
            "transformation": transformation,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        })
        logger.info(f"Added lineage: {source_table} -> {target_table}")

    def check_table(self, table_id: str) -> List[Anomaly]:
        """Check table for anomalies."""
        # In production, would call API
        return self._checked_tables.get(table_id, [])

    def get_health(self, table_id: str) -> "HealthScore":
        """Get health score for table."""
        # In production, would call API
        from observability.models import DataHealthScore

        return DataHealthScore(
            table_id=table_id,
            overall_score=100.0,
            freshness_score=100.0,
            volume_score=100.0,
            schema_score=100.0,
            quality_score=100.0,
            calculated_at=datetime.now(),
        )

    def get_table(self, table_id: str) -> Optional[Dict[str, Any]]:
        """Get table details."""
        # In production, would call API
        return None

    def get_lineage(
        self, table_id: str, direction: str = "both", depth: int = 3
    ) -> Dict[str, List]:
        """Get table lineage."""
        upstream = []
        downstream = []

        for entry in self._lineage:
            if entry["target"] == table_id:
                upstream.append(entry["source"])
            if entry["source"] == table_id:
                downstream.append(entry["target"])

        result = {}
        if direction in ("both", "upstream"):
            result["upstream"] = upstream
        if direction in ("both", "downstream"):
            result["downstream"] = downstream

        return result

    def get_anomalies(
        self,
        table_id: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Anomaly]:
        """Get anomalies."""
        # In production, would call API
        return []

    def acknowledge_anomaly(
        self, anomaly_id: str, notes: Optional[str] = None
    ) -> bool:
        """Acknowledge an anomaly."""
        # In production, would call API
        logger.info(f"Acknowledged anomaly: {anomaly_id}")
        return True

    def get_impact(self, table_id: str) -> Dict[str, Any]:
        """Get impact analysis."""
        lineage = self.get_lineage(table_id, direction="downstream", depth=10)
        return {
            "source_table": table_id,
            "affected_tables": lineage.get("downstream", []),
            "total_downstream": len(lineage.get("downstream", [])),
        }

    def add_table_tags(self, table_id: str, tags: Dict[str, str]) -> None:
        """Add tags to a table."""
        logger.info(f"Added tags to {table_id}: {tags}")

    def add_column_description(
        self, table_id: str, column_name: str, description: str
    ) -> None:
        """Add description to a column."""
        logger.info(
            f"Added description to {table_id}.{column_name}: {description}"
        )


class AutoRemediation:
    """Automatic remediation for common issues."""

    def __init__(self):
        self._handlers: Dict[AnomalyType, Callable] = {}

    def register_handler(
        self, anomaly_type: AnomalyType, handler: Callable
    ) -> None:
        """Register a remediation handler."""
        self._handlers[anomaly_type] = handler
        logger.info(f"Registered handler for {anomaly_type.value}")

    async def attempt_remediation(
        self, anomaly: Anomaly
    ) -> "RemediationResult":
        """Attempt automatic remediation."""
        handler = self._handlers.get(anomaly.anomaly_type)

        if not handler:
            return RemediationResult(
                success=False,
                message="No handler registered for this anomaly type",
            )

        try:
            success = await handler(anomaly)

            if success:
                return RemediationResult(
                    success=True,
                    message="Automatic remediation successful",
                )
            else:
                return RemediationResult(
                    success=False,
                    message="Remediation attempted but unsuccessful",
                )
        except Exception as e:
            return RemediationResult(
                success=False,
                message=f"Remediation failed: {str(e)}",
            )


@dataclass
class RemediationResult:
    """Result of remediation attempt."""

    success: bool
    message: str
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# Example remediation handlers
async def remediate_freshness(anomaly: Anomaly) -> bool:
    """Trigger pipeline re-run for stale data."""
    logger.info(f"Would trigger pipeline for {anomaly.table_id}")
    # In production, would trigger Airflow DAG
    return True


async def remediate_volume_drop(anomaly: Anomaly) -> bool:
    """Check upstream and retry."""
    logger.info(f"Would check upstream for {anomaly.table_id}")
    # In production, would check upstream health and retry
    return True
