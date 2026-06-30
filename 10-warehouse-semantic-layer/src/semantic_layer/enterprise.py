"""Enterprise features for semantic layer governance and operations."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from semantic_layer.models import MetricDefinition
from semantic_layer.query_engine import MetricCatalog

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access levels for metrics."""

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class FreshnessStatus(Enum):
    """Freshness status for metrics."""

    FRESH = "fresh"
    STALE = "stale"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ColumnLineage:
    """Lineage information for a column."""

    source_model: str
    source_column: str
    target_model: str
    target_column: str
    transformation: str = ""
    confidence: float = 1.0


@dataclass
class LineageGraph:
    """Graph of column-level lineage."""

    nodes: Dict[str, Set[str]] = field(default_factory=dict)  # model -> columns
    edges: List[ColumnLineage] = field(default_factory=list)

    def add_column(self, model: str, column: str) -> None:
        """Add a column node to the graph."""
        if model not in self.nodes:
            self.nodes[model] = set()
        self.nodes[model].add(column)

    def add_edge(self, lineage: ColumnLineage) -> None:
        """Add a lineage edge to the graph."""
        self.add_column(lineage.source_model, lineage.source_column)
        self.add_column(lineage.target_model, lineage.target_column)
        self.edges.append(lineage)

    def get_upstream(self, model: str, column: str) -> List[ColumnLineage]:
        """Get upstream lineage for a column."""
        return [
            edge for edge in self.edges
            if edge.target_model == model and edge.target_column == column
        ]

    def get_downstream(self, model: str, column: str) -> List[ColumnLineage]:
        """Get downstream lineage for a column."""
        return [
            edge for edge in self.edges
            if edge.source_model == model and edge.source_column == column
        ]


class LineageTracker:
    """Track column-level lineage across models."""

    def __init__(self):
        self._graph = LineageGraph()
        self._metric_lineage: Dict[str, List[str]] = {}  # metric -> source columns

    def track_transformation(
        self,
        source_model: str,
        source_columns: List[str],
        target_model: str,
        target_column: str,
        transformation: str = "",
    ) -> None:
        """Track a column transformation."""
        for source_col in source_columns:
            lineage = ColumnLineage(
                source_model=source_model,
                source_column=source_col,
                target_model=target_model,
                target_column=target_column,
                transformation=transformation,
            )
            self._graph.add_edge(lineage)

    def track_metric_lineage(
        self,
        metric_name: str,
        source_columns: List[str],
    ) -> None:
        """Track which source columns a metric depends on."""
        self._metric_lineage[metric_name] = source_columns

    def get_metric_dependencies(self, metric_name: str) -> List[str]:
        """Get source columns that a metric depends on."""
        return self._metric_lineage.get(metric_name, [])

    def get_impact_analysis(
        self,
        model: str,
        column: str,
    ) -> Dict[str, List[str]]:
        """Analyze impact of changing a column."""
        impacted = {"models": [], "columns": [], "metrics": []}

        # Find downstream models and columns
        to_process = [(model, column)]
        processed = set()

        while to_process:
            curr_model, curr_col = to_process.pop(0)
            if (curr_model, curr_col) in processed:
                continue
            processed.add((curr_model, curr_col))

            downstream = self._graph.get_downstream(curr_model, curr_col)
            for edge in downstream:
                if edge.target_model not in impacted["models"]:
                    impacted["models"].append(edge.target_model)
                col_key = f"{edge.target_model}.{edge.target_column}"
                if col_key not in impacted["columns"]:
                    impacted["columns"].append(col_key)
                to_process.append((edge.target_model, edge.target_column))

        # Find impacted metrics
        for metric_name, source_cols in self._metric_lineage.items():
            for col in source_cols:
                if col in impacted["columns"] or col == f"{model}.{column}":
                    if metric_name not in impacted["metrics"]:
                        impacted["metrics"].append(metric_name)

        return impacted

    def get_lineage_graph(self) -> LineageGraph:
        """Get the full lineage graph."""
        return self._graph


@dataclass
class FreshnessSLA:
    """SLA definition for metric freshness."""

    metric_name: str
    max_delay_minutes: int
    warning_threshold_minutes: int
    owner: str
    escalation_contacts: List[str] = field(default_factory=list)


@dataclass
class FreshnessCheck:
    """Result of a freshness check."""

    metric_name: str
    last_updated: Optional[datetime]
    status: FreshnessStatus
    delay_minutes: int
    sla_violated: bool
    message: str


class FreshnessMonitor:
    """Monitor metric freshness against SLAs."""

    def __init__(self, catalog: MetricCatalog):
        self.catalog = catalog
        self._slas: Dict[str, FreshnessSLA] = {}
        self._last_updated: Dict[str, datetime] = {}

    def set_sla(self, sla: FreshnessSLA) -> None:
        """Set freshness SLA for a metric."""
        self._slas[sla.metric_name] = sla

    def record_update(self, metric_name: str, timestamp: datetime) -> None:
        """Record when a metric was last updated."""
        self._last_updated[metric_name] = timestamp

    def check_freshness(self, metric_name: str) -> FreshnessCheck:
        """Check freshness of a metric."""
        sla = self._slas.get(metric_name)
        last_updated = self._last_updated.get(metric_name)

        if not last_updated:
            return FreshnessCheck(
                metric_name=metric_name,
                last_updated=None,
                status=FreshnessStatus.UNKNOWN,
                delay_minutes=0,
                sla_violated=False,
                message="No update recorded",
            )

        delay = datetime.now() - last_updated
        delay_minutes = int(delay.total_seconds() / 60)

        if not sla:
            status = FreshnessStatus.FRESH if delay_minutes < 60 else FreshnessStatus.STALE
            return FreshnessCheck(
                metric_name=metric_name,
                last_updated=last_updated,
                status=status,
                delay_minutes=delay_minutes,
                sla_violated=False,
                message=f"Last updated {delay_minutes} minutes ago (no SLA defined)",
            )

        sla_violated = delay_minutes > sla.max_delay_minutes
        if sla_violated:
            status = FreshnessStatus.CRITICAL
            message = f"SLA VIOLATED: {delay_minutes}min > {sla.max_delay_minutes}min"
        elif delay_minutes > sla.warning_threshold_minutes:
            status = FreshnessStatus.STALE
            message = f"WARNING: {delay_minutes}min > {sla.warning_threshold_minutes}min threshold"
        else:
            status = FreshnessStatus.FRESH
            message = f"Fresh: {delay_minutes}min delay"

        return FreshnessCheck(
            metric_name=metric_name,
            last_updated=last_updated,
            status=status,
            delay_minutes=delay_minutes,
            sla_violated=sla_violated,
            message=message,
        )

    def check_all(self) -> List[FreshnessCheck]:
        """Check freshness of all metrics with SLAs."""
        results = []
        for metric_name in self._slas:
            results.append(self.check_freshness(metric_name))
        return results


@dataclass
class AccessPolicy:
    """Access policy for a metric or dimension."""

    name: str
    level: AccessLevel
    allowed_roles: List[str] = field(default_factory=list)
    denied_roles: List[str] = field(default_factory=list)
    row_level_filter: Optional[str] = None
    column_mask: Optional[str] = None


class AccessController:
    """Control access to metrics and dimensions."""

    def __init__(self):
        self._policies: Dict[str, AccessPolicy] = {}
        self._user_roles: Dict[str, List[str]] = {}

    def set_policy(self, resource: str, policy: AccessPolicy) -> None:
        """Set access policy for a resource."""
        self._policies[resource] = policy

    def set_user_roles(self, user: str, roles: List[str]) -> None:
        """Set roles for a user."""
        self._user_roles[user] = roles

    def check_access(self, user: str, resource: str) -> bool:
        """Check if user has access to a resource."""
        policy = self._policies.get(resource)
        if not policy:
            return True  # No policy = public access

        user_roles = self._user_roles.get(user, [])

        # Check denied roles first
        for role in user_roles:
            if role in policy.denied_roles:
                return False

        # Check allowed roles
        if policy.allowed_roles:
            for role in user_roles:
                if role in policy.allowed_roles:
                    return True
            return False

        # Default based on access level
        if policy.level == AccessLevel.PUBLIC:
            return True
        elif policy.level == AccessLevel.INTERNAL:
            return bool(user_roles)  # Any role can access
        else:
            return False

    def get_row_filter(self, user: str, resource: str) -> Optional[str]:
        """Get row-level filter for user on resource."""
        policy = self._policies.get(resource)
        if not policy or not policy.row_level_filter:
            return None

        user_roles = self._user_roles.get(user, [])
        # Example: Replace {user_role} placeholder
        filter_sql = policy.row_level_filter
        if user_roles:
            filter_sql = filter_sql.replace("{user_role}", user_roles[0])
        return filter_sql

    def get_column_mask(self, user: str, resource: str) -> Optional[str]:
        """Get column mask for user on resource."""
        policy = self._policies.get(resource)
        if not policy:
            return None
        return policy.column_mask


@dataclass
class CICDConfig:
    """Configuration for CI/CD integration."""

    project_name: str
    repository_url: str
    branch: str = "main"
    dbt_project_dir: str = "."
    run_tests: bool = True
    run_freshness: bool = True
    fail_on_test_failure: bool = True
    slack_webhook: Optional[str] = None


class CICDIntegration:
    """Integration with CI/CD pipelines for semantic layer."""

    def __init__(self, config: CICDConfig):
        self.config = config
        self._test_results: List[Dict[str, Any]] = []

    def generate_github_actions(self) -> str:
        """Generate GitHub Actions workflow for semantic layer CI/CD."""
        return f"""name: Semantic Layer CI/CD

on:
  push:
    branches: [{self.config.branch}]
  pull_request:
    branches: [{self.config.branch}]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install dbt-core dbt-snowflake
          pip install -e {self.config.dbt_project_dir}

      - name: Run dbt deps
        run: dbt deps
        working-directory: {self.config.dbt_project_dir}

      - name: Run dbt compile
        run: dbt compile
        working-directory: {self.config.dbt_project_dir}

      - name: Run dbt test
        run: dbt test
        working-directory: {self.config.dbt_project_dir}
        if: {str(self.config.run_tests).lower()}

      - name: Check source freshness
        run: dbt source freshness
        working-directory: {self.config.dbt_project_dir}
        if: {str(self.config.run_freshness).lower()}

      - name: Validate metrics
        run: python -m semantic_layer.cli validate
        working-directory: {self.config.dbt_project_dir}
"""

    def generate_pre_commit_hooks(self) -> str:
        """Generate pre-commit hooks configuration."""
        return """repos:
  - repo: local
    hooks:
      - id: validate-metrics
        name: Validate Metric Definitions
        entry: python -m semantic_layer.cli validate
        language: python
        files: \\.yml$
        pass_filenames: false

      - id: check-metric-docs
        name: Check Metric Documentation
        entry: python -m semantic_layer.cli check-docs
        language: python
        files: \\.yml$
        pass_filenames: false

      - id: lint-sql
        name: Lint SQL
        entry: sqlfluff lint
        language: python
        types: [sql]
"""

    def validate_changes(
        self,
        changed_files: List[str],
        catalog: MetricCatalog,
    ) -> Dict[str, Any]:
        """Validate changes before deployment."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metrics_changed": [],
            "models_changed": [],
        }

        for file_path in changed_files:
            if file_path.endswith(".yml") or file_path.endswith(".yaml"):
                # Check for metric definition changes
                if "metrics" in file_path:
                    results["metrics_changed"].append(file_path)
            elif file_path.endswith(".sql"):
                results["models_changed"].append(file_path)

        # Validate metric definitions
        for metric in catalog.list_metrics():
            if not metric.description:
                results["warnings"].append(
                    f"Metric '{metric.name}' is missing description"
                )
            if not metric.meta.get("owner"):
                results["warnings"].append(
                    f"Metric '{metric.name}' has no owner defined"
                )

        return results


class DocumentationGenerator:
    """Generate documentation for semantic layer."""

    def __init__(self, catalog: MetricCatalog):
        self.catalog = catalog

    def generate_metric_docs(self) -> str:
        """Generate markdown documentation for all metrics."""
        docs = ["# Metric Catalog\n"]

        # Group by category
        metrics_by_category: Dict[str, List[MetricDefinition]] = {}
        for metric in self.catalog.list_metrics():
            category = metric.meta.get("category", "Uncategorized")
            if category not in metrics_by_category:
                metrics_by_category[category] = []
            metrics_by_category[category].append(metric)

        for category, metrics in sorted(metrics_by_category.items()):
            docs.append(f"\n## {category}\n")

            for metric in sorted(metrics, key=lambda m: m.name):
                docs.append(f"\n### {metric.label}\n")
                docs.append(f"**Name:** `{metric.name}`\n")
                docs.append(f"\n{metric.description}\n")
                docs.append(f"\n**Calculation:** {metric.calculation_method.value}\n")
                docs.append(f"\n**Expression:** `{metric.expression}`\n")

                if metric.dimensions:
                    docs.append("\n**Dimensions:**\n")
                    for dim in metric.dimensions:
                        docs.append(f"- {dim}\n")

                docs.append("\n**Time Grains:**\n")
                for grain in metric.time_grains:
                    docs.append(f"- {grain}\n")

                if metric.meta:
                    docs.append("\n**Metadata:**\n")
                    for key, value in metric.meta.items():
                        docs.append(f"- {key}: {value}\n")

        return "".join(docs)

    def generate_data_dictionary(self) -> str:
        """Generate data dictionary for all models."""
        docs = ["# Data Dictionary\n"]
        docs.append("\nThis document describes all models and their columns.\n")

        # Would iterate through models here
        docs.append("\n## Fact Tables\n")
        docs.append("\n## Dimension Tables\n")

        return "".join(docs)

    def generate_lineage_diagram(self, lineage: LineageTracker) -> str:
        """Generate Mermaid diagram for lineage."""
        diagram = ["```mermaid", "graph LR"]

        graph = lineage.get_lineage_graph()
        for edge in graph.edges:
            source = f"{edge.source_model}_{edge.source_column}"
            target = f"{edge.target_model}_{edge.target_column}"
            label = edge.transformation or ""
            if label:
                diagram.append(f"    {source} -->|{label}| {target}")
            else:
                diagram.append(f"    {source} --> {target}")

        diagram.append("```")
        return "\n".join(diagram)


class SemanticLayerGovernance:
    """Unified governance for the semantic layer."""

    def __init__(self, catalog: MetricCatalog):
        self.catalog = catalog
        self.lineage = LineageTracker()
        self.freshness = FreshnessMonitor(catalog)
        self.access = AccessController()
        self.docs = DocumentationGenerator(catalog)

    def get_governance_report(self) -> Dict[str, Any]:
        """Generate a governance report."""
        metrics = self.catalog.list_metrics()

        report = {
            "total_metrics": len(metrics),
            "metrics_with_owners": sum(
                1 for m in metrics if m.meta.get("owner")
            ),
            "metrics_with_docs": sum(
                1 for m in metrics if m.description
            ),
            "freshness_checks": len(self.freshness.check_all()),
            "access_policies": len(self.access._policies),
            "coverage": {
                "owner_coverage": 0.0,
                "doc_coverage": 0.0,
            },
        }

        if metrics:
            report["coverage"]["owner_coverage"] = (
                report["metrics_with_owners"] / len(metrics) * 100
            )
            report["coverage"]["doc_coverage"] = (
                report["metrics_with_docs"] / len(metrics) * 100
            )

        return report
