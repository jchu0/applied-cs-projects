"""Pytest configuration and fixtures for Semantic Layer tests."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
from typing import List
from unittest.mock import Mock, AsyncMock

from semantic_layer.models import (
    CalculationMethod,
    TimeGrain,
    MetricDefinition,
    Dimension,
)
from semantic_layer.query_engine import MetricCatalog, SemanticQueryEngine
from semantic_layer.api import MetricRegistry, SemanticLayerAPI


@pytest.fixture
def sample_metrics() -> List[MetricDefinition]:
    """Create sample metrics for testing."""
    return [
        MetricDefinition(
            name="revenue",
            label="Total Revenue",
            description="Sum of all revenue",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="order_amount",
            timestamp="order_date",
            time_grains=[TimeGrain.DAY, TimeGrain.WEEK, TimeGrain.MONTH],
            dimensions=["customer_type", "product_category", "region"],
            meta={"owner": "finance", "tier": 1}
        ),
        MetricDefinition(
            name="active_users",
            label="Active Users",
            description="Count of unique active users",
            model="user_activity",
            calculation_method=CalculationMethod.COUNT_DISTINCT,
            expression="user_id",
            timestamp="activity_date",
            time_grains=[TimeGrain.DAY, TimeGrain.WEEK],
            dimensions=["platform", "country"],
            filters=[{"field": "is_active", "operator": "=", "value": "true"}],
            meta={"owner": "product", "tier": 1}
        ),
        MetricDefinition(
            name="order_count",
            label="Order Count",
            description="Total number of orders",
            model="orders",
            calculation_method=CalculationMethod.COUNT,
            expression="order_id",
            timestamp="order_date",
            time_grains=[TimeGrain.DAY, TimeGrain.MONTH],
            dimensions=["customer_type", "region"],
            meta={"owner": "ops", "tier": 2}
        ),
        MetricDefinition(
            name="avg_order_value",
            label="Average Order Value",
            description="Average value per order",
            model="orders",
            calculation_method=CalculationMethod.DERIVED,
            expression="{{ metric('revenue') }} / {{ metric('order_count') }}",
            timestamp="order_date",
            time_grains=[TimeGrain.MONTH],
            dimensions=["customer_type"],
            meta={"owner": "analytics", "tier": 2}
        )
    ]


@pytest.fixture
def sample_dimensions() -> List[Dimension]:
    """Create sample dimensions for testing."""
    return [
        Dimension(
            name="customer_type",
            label="Customer Type",
            description="Type of customer (B2B, B2C, etc.)",
            data_type="string",
            model="customers",
            column="customer_type"
        ),
        Dimension(
            name="region",
            label="Region",
            description="Geographic region",
            data_type="string",
            model="locations",
            column="region_code",
            hierarchy=["region", "country", "state", "city"]
        ),
        Dimension(
            name="order_date",
            label="Order Date",
            description="Date when order was placed",
            data_type="date",
            model="orders",
            column="created_at",
            is_time=True
        ),
        Dimension(
            name="product_category",
            label="Product Category",
            description="Category of the product",
            data_type="string",
            model="products",
            column="category",
            hierarchy=["category", "subcategory", "product"]
        )
    ]


@pytest.fixture
def metric_catalog(sample_metrics) -> MetricCatalog:
    """Create a catalog with sample metrics."""
    catalog = MetricCatalog()
    for metric in sample_metrics:
        catalog.add_metric(metric)
    return catalog


@pytest.fixture
def metric_registry(sample_metrics) -> MetricRegistry:
    """Create a registry with sample metrics."""
    registry = MetricRegistry()
    for metric in sample_metrics:
        # Skip validation for test setup
        registry._catalog.add_metric(metric)
    return registry


@pytest.fixture
def query_engine(metric_catalog) -> SemanticQueryEngine:
    """Create a query engine with Snowflake as default."""
    return SemanticQueryEngine(metric_catalog, warehouse_type="snowflake")


@pytest.fixture
def semantic_api(metric_catalog, query_engine) -> SemanticLayerAPI:
    """Create a semantic layer API instance."""
    return SemanticLayerAPI(metric_catalog, query_engine)


@pytest.fixture
def mock_warehouse_connection():
    """Create a mock warehouse connection."""
    connection = Mock()
    connection.execute = AsyncMock(return_value=[])
    connection.fetch = AsyncMock(return_value=[])
    connection.close = AsyncMock()
    return connection


@pytest.fixture
def sample_query_results():
    """Sample query results for testing."""
    return [
        {
            "period": "2024-01-01",
            "region": "NA",
            "customer_type": "B2B",
            "revenue": 150000,
            "order_count": 75
        },
        {
            "period": "2024-01-01",
            "region": "EU",
            "customer_type": "B2B",
            "revenue": 120000,
            "order_count": 60
        },
        {
            "period": "2024-01-01",
            "region": "NA",
            "customer_type": "B2C",
            "revenue": 80000,
            "order_count": 200
        },
        {
            "period": "2024-01-01",
            "region": "EU",
            "customer_type": "B2C",
            "revenue": 65000,
            "order_count": 180
        }
    ]


@pytest.fixture
def yaml_metric_definitions():
    """Sample YAML content for metric definitions."""
    return """
version: 2

metrics:
  - name: gross_revenue
    label: "Gross Revenue"
    description: "Total gross revenue before discounts"
    model: ref('fct_orders')
    calculation_method: sum
    expression: gross_amount
    timestamp: order_date
    time_grains: [day, week, month, quarter, year]
    dimensions:
      - customer_segment
      - product_line
      - sales_channel
    filters:
      - field: order_status
        operator: 'in'
        value: ['completed', 'shipped']
    meta:
      owner: finance-team
      tier: 1
      refresh_frequency: daily

  - name: net_revenue
    label: "Net Revenue"
    description: "Revenue after discounts and returns"
    model: ref('fct_orders')
    calculation_method: sum
    expression: gross_amount - discount_amount - return_amount
    timestamp: order_date
    time_grains: [day, month, quarter, year]
    dimensions:
      - customer_segment
      - region
    meta:
      owner: finance-team
      tier: 1

  - name: customer_lifetime_value
    label: "Customer Lifetime Value"
    description: "Average lifetime value of customers"
    model: ref('fct_customer_metrics')
    calculation_method: average
    expression: total_lifetime_spend
    timestamp: first_order_date
    time_grains: [month, quarter, year]
    dimensions:
      - customer_segment
      - acquisition_channel
    meta:
      owner: analytics-team
      tier: 2
"""


@pytest.fixture
def mock_dbt_manifest():
    """Mock dbt manifest for testing."""
    return {
        "nodes": {
            "model.project.fct_orders": {
                "name": "fct_orders",
                "description": "Order fact table",
                "columns": {
                    "order_id": {"name": "order_id", "description": "Order ID"},
                    "order_amount": {"name": "order_amount", "description": "Order amount"},
                    "order_date": {"name": "order_date", "description": "Order date"},
                    "customer_id": {"name": "customer_id", "description": "Customer ID"}
                },
                "config": {"materialized": "table"},
                "tags": ["fact", "orders"]
            },
            "model.project.dim_customers": {
                "name": "dim_customers",
                "description": "Customer dimension",
                "columns": {
                    "customer_id": {"name": "customer_id", "description": "Customer ID"},
                    "customer_type": {"name": "customer_type", "description": "Customer type"},
                    "customer_segment": {"name": "customer_segment", "description": "Segment"}
                },
                "config": {"materialized": "table"},
                "tags": ["dimension", "customers"]
            }
        },
        "metrics": {
            "metric.project.revenue": {
                "name": "revenue",
                "label": "Revenue",
                "calculation_method": "sum",
                "expression": "order_amount",
                "timestamp": "order_date",
                "time_grains": ["day", "month"],
                "dimensions": ["customer_type"],
                "model": "ref('fct_orders')"
            }
        },
        "sources": {
            "source.project.raw.orders": {
                "name": "orders",
                "database": "raw",
                "schema": "public",
                "description": "Raw orders data"
            }
        }
    }


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration for tests."""
    import logging
    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.WARNING)