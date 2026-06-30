"""Tests for the semantic query engine."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from semantic_layer.models import (
    CalculationMethod,
    TimeGrain,
    MetricDefinition,
    MetricQuery,
    QueryResult,
)
from semantic_layer.query_engine import (
    MetricCatalog,
    SemanticQueryEngine,
    QueryExecutor,
    create_revenue_metric,
    create_user_metric,
)


class TestMetricCatalog:
    """Test MetricCatalog functionality."""

    @pytest.fixture
    def catalog(self):
        """Create a metric catalog."""
        return MetricCatalog()

    @pytest.fixture
    def sample_metric(self):
        """Create a sample metric."""
        return MetricDefinition(
            name="total_revenue",
            label="Total Revenue",
            description="Sum of all revenue",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY, TimeGrain.MONTH],
            dimensions=["country", "segment"],
            meta={"category": "financial", "owner": "finance"}
        )

    def test_add_metric(self, catalog, sample_metric):
        """Test adding a metric to the catalog."""
        catalog.add_metric(sample_metric)

        retrieved = catalog.get_metric("total_revenue")
        assert retrieved is not None
        assert retrieved.name == "total_revenue"
        assert retrieved.label == "Total Revenue"

    def test_get_nonexistent_metric(self, catalog):
        """Test getting a metric that doesn't exist."""
        result = catalog.get_metric("nonexistent")
        assert result is None

    def test_list_all_metrics(self, catalog, sample_metric):
        """Test listing all metrics."""
        # Add multiple metrics
        catalog.add_metric(sample_metric)

        user_metric = MetricDefinition(
            name="active_users",
            label="Active Users",
            description="Count of active users",
            model="users",
            calculation_method=CalculationMethod.COUNT_DISTINCT,
            expression="user_id",
            timestamp="activity_date",
            time_grains=[TimeGrain.DAY]
        )
        catalog.add_metric(user_metric)

        metrics = catalog.list_metrics()
        assert len(metrics) == 2
        assert any(m.name == "total_revenue" for m in metrics)
        assert any(m.name == "active_users" for m in metrics)

    def test_list_metrics_by_category(self, catalog):
        """Test filtering metrics by category."""
        financial_metric = MetricDefinition(
            name="revenue",
            label="Revenue",
            description="Revenue",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="date",
            time_grains=[TimeGrain.DAY],
            meta={"category": "financial"}
        )

        product_metric = MetricDefinition(
            name="users",
            label="Users",
            description="Users",
            model="users",
            calculation_method=CalculationMethod.COUNT,
            expression="user_id",
            timestamp="date",
            time_grains=[TimeGrain.DAY],
            meta={"category": "product"}
        )

        catalog.add_metric(financial_metric)
        catalog.add_metric(product_metric)

        financial_metrics = catalog.list_metrics(category="financial")
        assert len(financial_metrics) == 1
        assert financial_metrics[0].name == "revenue"

    def test_search_metrics(self, catalog):
        """Test searching metrics by name and description."""
        catalog.add_metric(MetricDefinition(
            name="total_revenue",
            label="Total Revenue",
            description="Sum of all revenue from orders",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="date",
            time_grains=[TimeGrain.DAY]
        ))

        catalog.add_metric(MetricDefinition(
            name="user_count",
            label="User Count",
            description="Total number of users",
            model="users",
            calculation_method=CalculationMethod.COUNT,
            expression="user_id",
            timestamp="date",
            time_grains=[TimeGrain.DAY]
        ))

        # Search by name
        results = catalog.list_metrics(search="revenue")
        assert len(results) == 1
        assert results[0].name == "total_revenue"

        # Search by description
        results = catalog.list_metrics(search="users")
        assert len(results) == 1
        assert results[0].name == "user_count"

    def test_add_and_get_dimension(self, catalog):
        """Test adding and retrieving dimensions."""
        dim_config = {
            "name": "country",
            "type": "categorical",
            "table": "customers"
        }

        catalog.add_dimension("country", dim_config)
        retrieved = catalog.get_dimension("country")

        assert retrieved is not None
        assert retrieved["name"] == "country"
        assert retrieved["type"] == "categorical"


class TestSemanticQueryEngine:
    """Test SemanticQueryEngine functionality."""

    @pytest.fixture
    def catalog(self):
        """Create a catalog with metrics."""
        catalog = MetricCatalog()

        # Add test metrics
        revenue_metric = MetricDefinition(
            name="total_revenue",
            label="Total Revenue",
            description="Sum of all revenue",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY, TimeGrain.MONTH],
            dimensions=["country", "segment"],
            filters=[{"field": "status", "operator": "=", "value": "'completed'"}]
        )
        catalog.add_metric(revenue_metric)

        user_metric = MetricDefinition(
            name="active_users",
            label="Active Users",
            description="Count of active users",
            model="users",
            calculation_method=CalculationMethod.COUNT_DISTINCT,
            expression="user_id",
            timestamp="activity_date",
            time_grains=[TimeGrain.DAY],
            dimensions=["country"]
        )
        catalog.add_metric(user_metric)

        # Add derived metric
        aov_metric = MetricDefinition(
            name="average_order_value",
            label="Average Order Value",
            description="Average value per order",
            model="orders",
            calculation_method=CalculationMethod.DERIVED,
            expression="{{ metric('total_revenue') }} / COUNT(DISTINCT order_id)",
            timestamp="created_at",
            time_grains=[TimeGrain.MONTH]
        )
        catalog.add_metric(aov_metric)

        return catalog

    @pytest.fixture
    def engine(self, catalog):
        """Create a query engine."""
        return SemanticQueryEngine(catalog, warehouse_type="snowflake")

    def test_generate_basic_sql(self, engine):
        """Test generating SQL for a basic query."""
        query = MetricQuery(
            metrics=["total_revenue"],
            dimensions=["country"],
            filters=[],
            time_grain="month",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        sql = engine.generate_sql(query)

        assert "SELECT" in sql
        assert "DATE_TRUNC('month'," in sql
        assert "SUM(amount)" in sql
        assert "country" in sql
        assert "FROM orders" in sql
        assert "WHERE" in sql
        assert "created_at >= '2024-01-01'" in sql
        assert "created_at < '2024-12-31'" in sql
        assert "GROUP BY" in sql
        assert "ORDER BY period" in sql

    def test_generate_sql_with_filters(self, engine):
        """Test SQL generation with additional filters."""
        query = MetricQuery(
            metrics=["total_revenue"],
            dimensions=["segment"],
            filters=[
                {"field": "country", "operator": "IN", "value": "('US', 'CA')"}
            ],
            time_grain="week",
            start_date="2024-01-01",
            end_date="2024-03-31",
            limit=100,
            offset=10
        )

        sql = engine.generate_sql(query)

        assert "country IN ('US', 'CA')" in sql
        assert "status = 'completed'" in sql  # Metric-level filter
        assert "LIMIT 100" in sql
        assert "OFFSET 10" in sql

    def test_generate_sql_multiple_metrics(self, engine):
        """Test SQL with multiple metrics."""
        query = MetricQuery(
            metrics=["total_revenue", "active_users"],
            dimensions=["country"],
            filters=[],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )

        # This should raise or handle multiple tables
        # For now, it uses the first metric's model
        sql = engine.generate_sql(query)

        assert "SUM(amount) as total_revenue" in sql
        assert "COUNT(DISTINCT user_id) as active_users" in sql

    def test_derived_metric_expansion(self, engine):
        """Test expansion of derived metrics."""
        query = MetricQuery(
            metrics=["average_order_value"],
            dimensions=[],
            filters=[],
            time_grain="month",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        sql = engine.generate_sql(query)

        # Should expand the metric reference
        assert "SUM(amount)" in sql  # Expanded from {{ metric('total_revenue') }}
        assert "COUNT(DISTINCT order_id)" in sql

    def test_warehouse_specific_sql(self):
        """Test SQL generation for different warehouse types."""
        catalog = MetricCatalog()
        metric = MetricDefinition(
            name="revenue",
            label="Revenue",
            description="Revenue",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY]
        )
        catalog.add_metric(metric)

        query = MetricQuery(
            metrics=["revenue"],
            dimensions=[],
            filters=[],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )

        # Test BigQuery
        engine = SemanticQueryEngine(catalog, warehouse_type="bigquery")
        sql = engine.generate_sql(query)
        assert "DATE_TRUNC(created_at, DAY)" in sql

        # Test Postgres
        engine = SemanticQueryEngine(catalog, warehouse_type="postgres")
        sql = engine.generate_sql(query)
        assert "DATE_TRUNC('day', created_at)" in sql

    def test_validate_query_success(self, engine):
        """Test successful query validation."""
        query = MetricQuery(
            metrics=["total_revenue"],
            dimensions=["country"],
            filters=[],
            time_grain="month",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        errors = engine.validate_query(query)
        assert len(errors) == 0

    def test_validate_query_missing_metric(self, engine):
        """Test validation with missing metric."""
        query = MetricQuery(
            metrics=["nonexistent_metric"],
            dimensions=[],
            filters=[],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        errors = engine.validate_query(query)
        assert len(errors) > 0
        assert "Metric not found: nonexistent_metric" in errors

    def test_validate_query_invalid_time_grain(self, engine):
        """Test validation with invalid time grain."""
        query = MetricQuery(
            metrics=["total_revenue"],
            dimensions=[],
            filters=[],
            time_grain="invalid_grain",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        errors = engine.validate_query(query)
        assert len(errors) > 0
        assert "Invalid time grain: invalid_grain" in errors

    def test_validate_query_invalid_dates(self, engine):
        """Test validation with invalid date range."""
        query = MetricQuery(
            metrics=["total_revenue"],
            dimensions=[],
            filters=[],
            time_grain="day",
            start_date="2024-12-31",
            end_date="2024-01-01"
        )

        errors = engine.validate_query(query)
        assert len(errors) > 0
        assert "start_date must be before end_date" in errors


class TestQueryExecutor:
    """Test QueryExecutor functionality."""

    @pytest.fixture
    def executor(self):
        """Create a query executor with mock connection."""
        connection = Mock()
        return QueryExecutor(connection)

    @pytest.fixture
    def engine(self):
        """Create a mock engine."""
        catalog = MetricCatalog()
        metric = MetricDefinition(
            name="revenue",
            label="Revenue",
            description="Revenue",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="date",
            time_grains=[TimeGrain.DAY]
        )
        catalog.add_metric(metric)
        return SemanticQueryEngine(catalog)

    @pytest.mark.asyncio
    async def test_execute_sql(self, executor):
        """Test executing SQL query."""
        sql = "SELECT * FROM orders"

        # Mock the connection to return data
        with patch.object(executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [
                {"period": "2024-01-01", "revenue": 10000}
            ]

            result = await executor.execute(sql)

            assert len(result) == 1
            assert result[0]["revenue"] == 10000
            mock_execute.assert_called_once_with(sql)

    @pytest.mark.asyncio
    async def test_execute_metric_query(self, executor, engine):
        """Test executing a metric query."""
        query = MetricQuery(
            metrics=["revenue"],
            dimensions=[],
            filters=[],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )

        with patch.object(executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [
                {"period": "2024-01-01", "revenue": 10000},
                {"period": "2024-01-02", "revenue": 12000}
            ]

            result = await executor.execute_metric_query(engine, query)

            assert isinstance(result, QueryResult)
            assert len(result.data) == 2
            assert result.row_count == 2
            assert result.metadata["metrics"] == ["revenue"]
            assert result.sql is not None

    @pytest.mark.asyncio
    async def test_execute_metric_query_validation_error(self, executor, engine):
        """Test executing an invalid metric query."""
        query = MetricQuery(
            metrics=["invalid_metric"],
            dimensions=[],
            filters=[],
            time_grain="day",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )

        with pytest.raises(ValueError, match="Query validation failed"):
            await executor.execute_metric_query(engine, query)


class TestMetricFactories:
    """Test factory functions for creating common metrics."""

    def test_create_revenue_metric(self):
        """Test creating a revenue metric."""
        metric = create_revenue_metric(
            model="sales",
            amount_column="total_amount",
            timestamp_column="sale_date"
        )

        assert metric.name == "total_revenue"
        assert metric.model == "sales"
        assert metric.expression == "total_amount"
        assert metric.timestamp == "sale_date"
        assert metric.calculation_method == CalculationMethod.SUM
        assert "customer_segment" in metric.dimensions
        assert metric.meta["owner"] == "finance-team"
        assert metric.meta["tier"] == 1

    def test_create_user_metric(self):
        """Test creating a user metric."""
        metric = create_user_metric(
            model="events",
            user_column="user_uuid",
            timestamp_column="event_time"
        )

        assert metric.name == "active_users"
        assert metric.model == "events"
        assert metric.expression == "user_uuid"
        assert metric.timestamp == "event_time"
        assert metric.calculation_method == CalculationMethod.COUNT_DISTINCT
        assert "country_code" in metric.dimensions
        assert metric.meta["owner"] == "product-team"