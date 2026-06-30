"""API for Semantic Layer metric queries."""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from semantic_layer.models import MetricDefinition, MetricQuery, QueryResult
from semantic_layer.query_engine import MetricCatalog, SemanticQueryEngine

logger = logging.getLogger(__name__)


@dataclass
class MetricQueryRequest:
    """Request to query metrics."""

    metrics: List[str]
    start_date: str
    end_date: str
    time_grain: str = "day"
    dimensions: List[str] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class MetricInfo:
    """Information about a metric for API response."""

    name: str
    label: str
    description: str
    dimensions: List[str]
    time_grains: List[str]
    calculation_method: str
    owner: str = ""
    tier: int = 1


@dataclass
class MetricQueryResponse:
    """Response from metric query."""

    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    sql: Optional[str] = None
    row_count: int = 0


class SemanticLayerAPI:
    """API for interacting with the semantic layer."""

    def __init__(
        self,
        catalog: MetricCatalog,
        engine: SemanticQueryEngine,
        warehouse_executor: Any = None,
    ):
        self.catalog = catalog
        self.engine = engine
        self.executor = warehouse_executor

    def list_metrics(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        tier: Optional[int] = None,
    ) -> List[MetricInfo]:
        """List available metrics with optional filtering."""
        metrics = self.catalog.list_metrics(category=category, search=search)

        # Filter by tier if specified
        if tier is not None:
            metrics = [
                m for m in metrics
                if m.meta.get("tier", 1) == tier
            ]

        return [
            MetricInfo(
                name=m.name,
                label=m.label,
                description=m.description,
                dimensions=m.dimensions,
                time_grains=[g.value if hasattr(g, 'value') else str(g) for g in m.time_grains],
                calculation_method=m.calculation_method.value,
                owner=m.meta.get("owner", ""),
                tier=m.meta.get("tier", 1),
            )
            for m in metrics
        ]

    def get_metric(self, metric_name: str) -> Optional[MetricInfo]:
        """Get details of a specific metric."""
        metric = self.catalog.get_metric(metric_name)
        if not metric:
            return None

        return MetricInfo(
            name=metric.name,
            label=metric.label,
            description=metric.description,
            dimensions=metric.dimensions,
            time_grains=[g.value if hasattr(g, 'value') else str(g) for g in metric.time_grains],
            calculation_method=metric.calculation_method.value,
            owner=metric.meta.get("owner", ""),
            tier=metric.meta.get("tier", 1),
        )

    def query_metric(
        self,
        metric_name: str,
        start_date: str,
        end_date: str,
        time_grain: str = "day",
        dimensions: Optional[List[str]] = None,
        filters: Optional[List[Dict[str, Any]]] = None,
        show_sql: bool = False,
    ) -> MetricQueryResponse:
        """Query a single metric."""
        return self.query_metrics(
            metrics=[metric_name],
            start_date=start_date,
            end_date=end_date,
            time_grain=time_grain,
            dimensions=dimensions,
            filters=filters,
            show_sql=show_sql,
        )

    def query_metrics(
        self,
        metrics: List[str],
        start_date: str,
        end_date: str,
        time_grain: str = "day",
        dimensions: Optional[List[str]] = None,
        filters: Optional[List[Dict[str, Any]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        show_sql: bool = False,
    ) -> MetricQueryResponse:
        """Query multiple metrics together."""
        query = MetricQuery(
            metrics=metrics,
            dimensions=dimensions or [],
            filters=filters or [],
            time_grain=time_grain,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

        # Validate query
        errors = self.engine.validate_query(query)
        if errors:
            raise ValueError(f"Query validation failed: {', '.join(errors)}")

        # Generate SQL
        sql = self.engine.generate_sql(query)

        # Execute query
        data = []
        if self.executor:
            # Would execute against actual warehouse
            pass

        return MetricQueryResponse(
            data=data,
            metadata={
                "metrics": metrics,
                "dimensions": dimensions or [],
                "time_grain": time_grain,
                "start_date": start_date,
                "end_date": end_date,
            },
            sql=sql if show_sql else None,
            row_count=len(data),
        )

    def get_dimensions_for_metric(self, metric_name: str) -> List[str]:
        """Get available dimensions for a metric."""
        metric = self.catalog.get_metric(metric_name)
        if not metric:
            return []
        return metric.dimensions

    def validate_query(self, request: MetricQueryRequest) -> List[str]:
        """Validate a query request."""
        query = MetricQuery(
            metrics=request.metrics,
            dimensions=request.dimensions,
            filters=request.filters,
            time_grain=request.time_grain,
            start_date=request.start_date,
            end_date=request.end_date,
            limit=request.limit,
            offset=request.offset,
        )
        return self.engine.validate_query(query)


class MetricValidator:
    """Validate metric definitions."""

    def __init__(self, catalog: MetricCatalog):
        self.catalog = catalog

    def validate_metric(self, metric: MetricDefinition) -> List[str]:
        """Validate a metric definition."""
        errors = []

        # Check required fields
        if not metric.name:
            errors.append("Metric name is required")

        if not metric.model:
            errors.append("Metric model is required")

        if not metric.expression:
            errors.append("Metric expression is required")

        if not metric.timestamp:
            errors.append("Metric timestamp is required")

        if not metric.time_grains:
            errors.append("At least one time grain is required")

        # Check for derived metric references
        if metric.calculation_method.value == "derived":
            # Extract referenced metrics
            import re
            refs = re.findall(r"metric\('(\w+)'\)", metric.expression)
            for ref in refs:
                if not self.catalog.get_metric(ref):
                    errors.append(f"Referenced metric not found: {ref}")

        return errors

    def validate_all_metrics(self) -> Dict[str, List[str]]:
        """Validate all metrics in the catalog."""
        results = {}

        for metric in self.catalog.list_metrics():
            errors = self.validate_metric(metric)
            if errors:
                results[metric.name] = errors

        return results


class MetricRegistry:
    """Registry for managing metric definitions."""

    def __init__(self):
        self._catalog = MetricCatalog()
        self._validator = MetricValidator(self._catalog)

    def register_metric(self, metric: MetricDefinition) -> None:
        """Register a metric in the catalog."""
        # Validate first
        errors = self._validator.validate_metric(metric)
        if errors:
            raise ValueError(f"Invalid metric: {', '.join(errors)}")

        self._catalog.add_metric(metric)
        logger.info(f"Registered metric: {metric.name}")

    def get_catalog(self) -> MetricCatalog:
        """Get the metric catalog."""
        return self._catalog

    def export_yaml(self) -> str:
        """Export all metrics to YAML format."""
        metrics = self._catalog.list_metrics()

        yaml_parts = ["version: 2", "", "metrics:"]

        for metric in metrics:
            yaml_parts.append(f"  - name: {metric.name}")
            yaml_parts.append(f'    label: "{metric.label}"')
            yaml_parts.append(f'    description: "{metric.description}"')
            yaml_parts.append(f"    model: ref('{metric.model}')")
            yaml_parts.append(f"    calculation_method: {metric.calculation_method.value}")
            yaml_parts.append(f"    expression: {metric.expression}")
            yaml_parts.append(f"    timestamp: {metric.timestamp}")

            grains = ", ".join(str(g) for g in metric.time_grains)
            yaml_parts.append(f"    time_grains: [{grains}]")

            if metric.dimensions:
                yaml_parts.append("    dimensions:")
                for dim in metric.dimensions:
                    yaml_parts.append(f"      - {dim}")

            if metric.filters:
                yaml_parts.append("    filters:")
                for f in metric.filters:
                    yaml_parts.append(f"      - field: {f['field']}")
                    yaml_parts.append(f"        operator: '{f['operator']}'")
                    yaml_parts.append(f"        value: {f['value']}")

            if metric.meta:
                yaml_parts.append("    meta:")
                for key, value in metric.meta.items():
                    yaml_parts.append(f"      {key}: {value}")

            yaml_parts.append("")

        return "\n".join(yaml_parts)

    def import_yaml(self, yaml_content: str) -> None:
        """Import metrics from YAML content."""
        import yaml

        data = yaml.safe_load(yaml_content)
        metrics = data.get("metrics", [])

        for m in metrics:
            from semantic_layer.models import CalculationMethod

            metric = MetricDefinition(
                name=m["name"],
                label=m.get("label", m["name"]),
                description=m.get("description", ""),
                model=m["model"].replace("ref('", "").replace("')", ""),
                calculation_method=CalculationMethod(m["calculation_method"]),
                expression=m["expression"],
                timestamp=m["timestamp"],
                time_grains=m.get("time_grains", ["day"]),
                dimensions=m.get("dimensions", []),
                filters=m.get("filters", []),
                meta=m.get("meta", {}),
            )
            self.register_metric(metric)
