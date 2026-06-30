"""Tests for the Semantic Layer API."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from semantic_layer.models import (
    CalculationMethod,
    TimeGrain,
    MetricDefinition,
    MetricQuery,
    QueryResult,
)
from semantic_layer.query_engine import MetricCatalog, SemanticQueryEngine
from semantic_layer.api import (
    MetricQueryRequest,
    MetricInfo,
    MetricQueryResponse,
    SemanticLayerAPI,
    MetricValidator,
    MetricRegistry,
)


class TestMetricQueryRequest:
    """Test MetricQueryRequest dataclass."""

    def test_create_basic_request(self):
        """Test creating a basic metric query request."""
        request = MetricQueryRequest(
            metrics=["revenue"],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        assert request.metrics == ["revenue"]
        assert request.start_date == "2024-01-01"
        assert request.end_date == "2024-12-31"
        assert request.time_grain == "day"  # Default
        assert request.dimensions == []
        assert request.filters == []

    def test_create_complex_request(self):
        """Test creating a complex query request."""
        request = MetricQueryRequest(
            metrics=["revenue", "users"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            time_grain="week",
            dimensions=["country", "segment"],
            filters=[{"field": "status", "operator": "=", "value": "active"}],
            limit=100,
            offset=0
        )

        assert len(request.metrics) == 2
        assert request.time_grain == "week"
        assert len(request.dimensions) == 2
        assert len(request.filters) == 1
        assert request.limit == 100


class TestSemanticLayerAPI:
    """Test SemanticLayerAPI functionality."""

    @pytest.fixture
    def catalog(self):
        """Create a catalog with test metrics."""
        catalog = MetricCatalog()

        metrics = [
            MetricDefinition(
                name="revenue",
                label="Total Revenue",
                description="Sum of all revenue",
                model="orders",
                calculation_method=CalculationMethod.SUM,
                expression="amount",
                timestamp="created_at",
                time_grains=[TimeGrain.DAY, TimeGrain.MONTH],
                dimensions=["country", "segment"],
                meta={"owner": "finance", "tier": 1}
            ),
            MetricDefinition(
                name="users",
                label="Active Users",
                description="Count of unique users",
                model="users",
                calculation_method=CalculationMethod.COUNT_DISTINCT,
                expression="user_id",
                timestamp="activity_date",
                time_grains=[TimeGrain.DAY],
                dimensions=["country"],
                meta={"owner": "product", "tier": 1}
            ),
            MetricDefinition(
                name="orders",
                label="Order Count",
                description="Number of orders",
                model="orders",
                calculation_method=CalculationMethod.COUNT,
                expression="order_id",
                timestamp="created_at",
                time_grains=[TimeGrain.DAY],
                dimensions=["status"],
                meta={"owner": "ops", "tier": 2}
            )
        ]

        for metric in metrics:
            catalog.add_metric(metric)

        return catalog

    @pytest.fixture
    def engine(self, catalog):
        """Create a query engine."""
        return SemanticQueryEngine(catalog, warehouse_type="snowflake")

    @pytest.fixture
    def api(self, catalog, engine):
        """Create API instance."""
        return SemanticLayerAPI(catalog, engine)

    def test_list_all_metrics(self, api):
        """Test listing all metrics."""
        metrics = api.list_metrics()

        assert len(metrics) == 3
        assert any(m.name == "revenue" for m in metrics)
        assert any(m.name == "users" for m in metrics)
        assert any(m.name == "orders" for m in metrics)

        # Check MetricInfo structure
        revenue_metric = next(m for m in metrics if m.name == "revenue")
        assert revenue_metric.label == "Total Revenue"
        assert revenue_metric.owner == "finance"
        assert revenue_metric.tier == 1
        assert "country" in revenue_metric.dimensions

    def test_list_metrics_with_search(self, api):
        """Test searching metrics."""
        metrics = api.list_metrics(search="revenue")

        assert len(metrics) == 1
        assert metrics[0].name == "revenue"

    def test_list_metrics_by_tier(self, api):
        """Test filtering metrics by tier."""
        tier1_metrics = api.list_metrics(tier=1)
        assert len(tier1_metrics) == 2

        tier2_metrics = api.list_metrics(tier=2)
        assert len(tier2_metrics) == 1
        assert tier2_metrics[0].name == "orders"

    def test_get_metric_details(self, api):
        """Test getting details of a specific metric."""
        metric = api.get_metric("revenue")

        assert metric is not None
        assert metric.name == "revenue"
        assert metric.label == "Total Revenue"
        assert metric.calculation_method == "sum"
        assert "day" in metric.time_grains
        assert "month" in metric.time_grains

    def test_get_nonexistent_metric(self, api):
        """Test getting a metric that doesn't exist."""
        metric = api.get_metric("nonexistent")
        assert metric is None

    def test_query_single_metric(self, api):
        """Test querying a single metric."""
        response = api.query_metric(
            metric_name="revenue",
            start_date="2024-01-01",
            end_date="2024-01-31",
            time_grain="day",
            dimensions=["country"]
        )

        assert isinstance(response, MetricQueryResponse)
        assert response.metadata["metrics"] == ["revenue"]
        assert response.metadata["dimensions"] == ["country"]
        assert response.metadata["time_grain"] == "day"

    def test_query_multiple_metrics(self, api):
        """Test querying multiple metrics."""
        response = api.query_metrics(
            metrics=["revenue", "users"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            time_grain="week",
            dimensions=["country"],
            show_sql=True
        )

        assert response.metadata["metrics"] == ["revenue", "users"]
        assert response.sql is not None  # SQL shown because show_sql=True

    def test_query_metrics_with_filters(self, api):
        """Test querying with filters."""
        filters = [
            {"field": "country", "operator": "IN", "value": ["US", "CA"]},
            {"field": "segment", "operator": "=", "value": "enterprise"}
        ]

        response = api.query_metrics(
            metrics=["revenue"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            time_grain="month",
            dimensions=["country"],
            filters=filters,
            limit=100,
            offset=0
        )

        assert response.metadata["metrics"] == ["revenue"]
        assert response.metadata["dimensions"] == ["country"]

    def test_query_invalid_metric(self, api):
        """Test querying an invalid metric."""
        with pytest.raises(ValueError, match="Query validation failed"):
            api.query_metric(
                metric_name="invalid_metric",
                start_date="2024-01-01",
                end_date="2024-01-31"
            )

    def test_get_dimensions_for_metric(self, api):
        """Test getting available dimensions for a metric."""
        dimensions = api.get_dimensions_for_metric("revenue")

        assert "country" in dimensions
        assert "segment" in dimensions

        # Test nonexistent metric
        dimensions = api.get_dimensions_for_metric("nonexistent")
        assert dimensions == []

    def test_validate_query_request(self, api):
        """Test validating a query request."""
        request = MetricQueryRequest(
            metrics=["revenue"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            time_grain="month"
        )

        errors = api.validate_query(request)
        assert len(errors) == 0

        # Test invalid request
        invalid_request = MetricQueryRequest(
            metrics=["invalid_metric"],
            start_date="2024-12-31",
            end_date="2024-01-01",  # Invalid date range
            time_grain="invalid"
        )

        errors = api.validate_query(invalid_request)
        assert len(errors) > 0


class TestMetricValidator:
    """Test MetricValidator functionality."""

    @pytest.fixture
    def catalog(self):
        """Create a catalog."""
        catalog = MetricCatalog()

        # Add a base metric for derived metric testing
        base_metric = MetricDefinition(
            name="total_orders",
            label="Total Orders",
            description="Count of orders",
            model="orders",
            calculation_method=CalculationMethod.COUNT,
            expression="order_id",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY]
        )
        catalog.add_metric(base_metric)

        return catalog

    @pytest.fixture
    def validator(self, catalog):
        """Create a validator."""
        return MetricValidator(catalog)

    def test_validate_valid_metric(self, validator):
        """Test validating a valid metric."""
        metric = MetricDefinition(
            name="revenue",
            label="Revenue",
            description="Total revenue",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY]
        )

        errors = validator.validate_metric(metric)
        assert len(errors) == 0

    def test_validate_metric_missing_name(self, validator):
        """Test validating metric with missing name."""
        metric = MetricDefinition(
            name="",
            label="Revenue",
            description="Total revenue",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY]
        )

        errors = validator.validate_metric(metric)
        assert "Metric name is required" in errors

    def test_validate_metric_missing_fields(self, validator):
        """Test validating metric with missing required fields."""
        metric = MetricDefinition(
            name="test",
            label="Test",
            description="Test metric",
            model="",  # Missing
            calculation_method=CalculationMethod.SUM,
            expression="",  # Missing
            timestamp="",  # Missing
            time_grains=[]  # Empty
        )

        errors = validator.validate_metric(metric)
        assert "Metric model is required" in errors
        assert "Metric expression is required" in errors
        assert "Metric timestamp is required" in errors
        assert "At least one time grain is required" in errors

    def test_validate_derived_metric_valid(self, validator):
        """Test validating a valid derived metric."""
        metric = MetricDefinition(
            name="avg_order_value",
            label="Average Order Value",
            description="Average order value",
            model="orders",
            calculation_method=CalculationMethod.DERIVED,
            expression="SUM(amount) / {{ metric('total_orders') }}",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY]
        )

        errors = validator.validate_metric(metric)
        assert len(errors) == 0  # total_orders exists in catalog

    def test_validate_derived_metric_invalid_reference(self, validator):
        """Test validating derived metric with invalid reference."""
        metric = MetricDefinition(
            name="test",
            label="Test",
            description="Test",
            model="orders",
            calculation_method=CalculationMethod.DERIVED,
            expression="{{ metric('nonexistent') }} * 2",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY]
        )

        errors = validator.validate_metric(metric)
        assert any("Referenced metric not found: nonexistent" in e for e in errors)

    def test_validate_all_metrics(self, validator, catalog):
        """Test validating all metrics in catalog."""
        # Add an invalid metric
        invalid_metric = MetricDefinition(
            name="invalid",
            label="Invalid",
            description="Invalid metric",
            model="",  # Missing model
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="date",
            time_grains=[TimeGrain.DAY]
        )
        catalog._metrics["invalid"] = invalid_metric  # Bypass validation

        results = validator.validate_all_metrics()

        assert "invalid" in results
        assert "Metric model is required" in results["invalid"]
        assert "total_orders" not in results  # Valid metric not in errors


class TestMetricRegistry:
    """Test MetricRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create a metric registry."""
        return MetricRegistry()

    def test_register_valid_metric(self, registry):
        """Test registering a valid metric."""
        metric = MetricDefinition(
            name="revenue",
            label="Revenue",
            description="Total revenue",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY]
        )

        registry.register_metric(metric)

        catalog = registry.get_catalog()
        retrieved = catalog.get_metric("revenue")
        assert retrieved is not None
        assert retrieved.name == "revenue"

    def test_register_invalid_metric(self, registry):
        """Test registering an invalid metric."""
        metric = MetricDefinition(
            name="",  # Invalid - no name
            label="Test",
            description="Test",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="date",
            time_grains=[TimeGrain.DAY]
        )

        with pytest.raises(ValueError, match="Invalid metric"):
            registry.register_metric(metric)

    def test_export_yaml(self, registry):
        """Test exporting metrics to YAML."""
        # Register metrics
        metric1 = MetricDefinition(
            name="revenue",
            label="Total Revenue",
            description="Sum of revenue",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY, TimeGrain.MONTH],
            dimensions=["country"],
            filters=[{"field": "status", "operator": "=", "value": "completed"}],
            meta={"owner": "finance", "tier": 1}
        )
        registry.register_metric(metric1)

        yaml_content = registry.export_yaml()

        assert "version: 2" in yaml_content
        assert "metrics:" in yaml_content
        assert "name: revenue" in yaml_content
        assert 'label: "Total Revenue"' in yaml_content
        assert "model: ref('orders')" in yaml_content
        assert "calculation_method: sum" in yaml_content
        assert "dimensions:" in yaml_content
        assert "- country" in yaml_content
        assert "filters:" in yaml_content
        assert "owner: finance" in yaml_content

    @pytest.mark.skipif(
        not __import__('importlib.util', fromlist=['find_spec']).find_spec('yaml'),
        reason="PyYAML not installed"
    )
    def test_import_yaml(self, registry):
        """Test importing metrics from YAML."""
        yaml_content = """
version: 2

metrics:
  - name: imported_revenue
    label: "Imported Revenue"
    description: "Revenue from imports"
    model: ref('orders')
    calculation_method: sum
    expression: amount
    timestamp: created_at
    time_grains: [day, month]
    dimensions:
      - country
      - segment
    filters:
      - field: type
        operator: '='
        value: import
    meta:
      owner: finance
      tier: 1
"""

        registry.import_yaml(yaml_content)

        catalog = registry.get_catalog()
        metric = catalog.get_metric("imported_revenue")

        assert metric is not None
        assert metric.label == "Imported Revenue"
        assert metric.model == "orders"
        assert metric.calculation_method == CalculationMethod.SUM
        assert "country" in metric.dimensions
        assert len(metric.filters) == 1
        assert metric.meta["owner"] == "finance"