"""Tests for data models."""

import pytest
from dataclasses import asdict
from typing import List

from semantic_layer.models import (
    CalculationMethod,
    TimeGrain,
    Dimension,
    MetricDefinition,
    MetricQuery,
    QueryResult,
    Column,
    Model,
    Source,
    Test,
    Macro,
    DbtProject,
)


class TestCalculationMethod:
    """Test CalculationMethod enum."""

    def test_calculation_methods_exist(self):
        """Test all calculation methods are defined."""
        assert CalculationMethod.SUM.value == "sum"
        assert CalculationMethod.COUNT.value == "count"
        assert CalculationMethod.COUNT_DISTINCT.value == "count_distinct"
        assert CalculationMethod.AVERAGE.value == "average"
        assert CalculationMethod.MIN.value == "min"
        assert CalculationMethod.MAX.value == "max"
        assert CalculationMethod.DERIVED.value == "derived"

    def test_calculation_method_from_string(self):
        """Test creating calculation method from string."""
        assert CalculationMethod("sum") == CalculationMethod.SUM
        assert CalculationMethod("count") == CalculationMethod.COUNT

    def test_invalid_calculation_method(self):
        """Test invalid calculation method raises error."""
        with pytest.raises(ValueError):
            CalculationMethod("invalid")


class TestTimeGrain:
    """Test TimeGrain enum."""

    def test_time_grains_exist(self):
        """Test all time grains are defined."""
        assert TimeGrain.DAY.value == "day"
        assert TimeGrain.WEEK.value == "week"
        assert TimeGrain.MONTH.value == "month"
        assert TimeGrain.QUARTER.value == "quarter"
        assert TimeGrain.YEAR.value == "year"

    def test_time_grain_from_string(self):
        """Test creating time grain from string."""
        assert TimeGrain("day") == TimeGrain.DAY
        assert TimeGrain("month") == TimeGrain.MONTH


class TestDimension:
    """Test Dimension dataclass."""

    def test_create_dimension(self):
        """Test creating a dimension."""
        dim = Dimension(
            name="customer_segment",
            label="Customer Segment",
            description="Customer segmentation category",
            data_type="string",
            model="customers",
            column="segment"
        )

        assert dim.name == "customer_segment"
        assert dim.label == "Customer Segment"
        assert dim.description == "Customer segmentation category"
        assert dim.data_type == "string"
        assert dim.model == "customers"
        assert dim.column == "segment"
        assert dim.is_time is False
        assert dim.hierarchy is None

    def test_time_dimension(self):
        """Test creating a time dimension."""
        dim = Dimension(
            name="order_date",
            label="Order Date",
            description="Date when order was placed",
            data_type="date",
            model="orders",
            column="created_at",
            is_time=True
        )

        assert dim.is_time is True

    def test_dimension_with_hierarchy(self):
        """Test dimension with hierarchy."""
        dim = Dimension(
            name="location",
            label="Location",
            description="Geographic location",
            data_type="string",
            model="stores",
            column="country",
            hierarchy=["country", "state", "city"]
        )

        assert dim.hierarchy == ["country", "state", "city"]


class TestMetricDefinition:
    """Test MetricDefinition dataclass."""

    def test_create_simple_metric(self):
        """Test creating a simple metric."""
        metric = MetricDefinition(
            name="total_revenue",
            label="Total Revenue",
            description="Sum of all revenue",
            model="orders",
            calculation_method=CalculationMethod.SUM,
            expression="amount",
            timestamp="created_at",
            time_grains=[TimeGrain.DAY, TimeGrain.MONTH]
        )

        assert metric.name == "total_revenue"
        assert metric.label == "Total Revenue"
        assert metric.calculation_method == CalculationMethod.SUM
        assert metric.expression == "amount"
        assert metric.timestamp == "created_at"
        assert len(metric.time_grains) == 2
        assert TimeGrain.DAY in metric.time_grains
        assert metric.dimensions == []
        assert metric.filters == []
        assert metric.meta == {}

    def test_metric_with_dimensions_and_filters(self):
        """Test metric with dimensions and filters."""
        metric = MetricDefinition(
            name="active_users",
            label="Active Users",
            description="Count of active users",
            model="users",
            calculation_method=CalculationMethod.COUNT_DISTINCT,
            expression="user_id",
            timestamp="last_activity_at",
            time_grains=[TimeGrain.DAY],
            dimensions=["country", "plan_type"],
            filters=[{"field": "is_active", "operator": "=", "value": "true"}]
        )

        assert "country" in metric.dimensions
        assert "plan_type" in metric.dimensions
        assert len(metric.filters) == 1
        assert metric.filters[0]["field"] == "is_active"

    def test_derived_metric(self):
        """Test derived metric definition."""
        metric = MetricDefinition(
            name="average_order_value",
            label="Average Order Value",
            description="Average value per order",
            model="orders",
            calculation_method=CalculationMethod.DERIVED,
            expression="{{ metric('total_revenue') }} / {{ metric('total_orders') }}",
            timestamp="created_at",
            time_grains=[TimeGrain.MONTH],
            meta={"owner": "analytics-team", "tier": 1}
        )

        assert metric.calculation_method == CalculationMethod.DERIVED
        assert "{{ metric('total_revenue') }}" in metric.expression
        assert metric.meta["owner"] == "analytics-team"
        assert metric.meta["tier"] == 1


class TestMetricQuery:
    """Test MetricQuery dataclass."""

    def test_create_basic_query(self):
        """Test creating a basic metric query."""
        query = MetricQuery(
            metrics=["total_revenue"],
            dimensions=["country"],
            filters=[],
            time_grain="month",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        assert query.metrics == ["total_revenue"]
        assert query.dimensions == ["country"]
        assert query.time_grain == "month"
        assert query.start_date == "2024-01-01"
        assert query.end_date == "2024-12-31"
        assert query.limit is None
        assert query.offset is None

    def test_query_with_filters(self):
        """Test query with filters."""
        query = MetricQuery(
            metrics=["active_users", "total_revenue"],
            dimensions=["country", "plan_type"],
            filters=[
                {"field": "country", "operator": "in", "value": ["US", "CA"]},
                {"field": "plan_type", "operator": "=", "value": "premium"}
            ],
            time_grain="week",
            start_date="2024-01-01",
            end_date="2024-03-31",
            limit=100,
            offset=0
        )

        assert len(query.metrics) == 2
        assert len(query.dimensions) == 2
        assert len(query.filters) == 2
        assert query.limit == 100
        assert query.offset == 0


class TestQueryResult:
    """Test QueryResult dataclass."""

    def test_create_query_result(self):
        """Test creating a query result."""
        data = [
            {"period": "2024-01-01", "country": "US", "revenue": 10000},
            {"period": "2024-01-01", "country": "CA", "revenue": 5000}
        ]

        result = QueryResult(
            data=data,
            metadata={"query_time": "100ms"},
            sql="SELECT * FROM orders",
            row_count=2
        )

        assert len(result.data) == 2
        assert result.data[0]["revenue"] == 10000
        assert result.metadata["query_time"] == "100ms"
        assert "SELECT" in result.sql
        assert result.row_count == 2


class TestColumn:
    """Test Column dataclass."""

    def test_create_column(self):
        """Test creating a column."""
        col = Column(
            name="customer_id",
            description="Unique customer identifier",
            data_type="integer",
            tests=["not_null", "unique"]
        )

        assert col.name == "customer_id"
        assert col.data_type == "integer"
        assert "not_null" in col.tests
        assert "unique" in col.tests

    def test_column_with_metadata(self):
        """Test column with metadata."""
        col = Column(
            name="revenue",
            description="Total revenue amount",
            data_type="decimal",
            meta={"format": "currency", "precision": 2}
        )

        assert col.meta["format"] == "currency"
        assert col.meta["precision"] == 2


class TestModel:
    """Test Model dataclass."""

    def test_create_model(self):
        """Test creating a dbt model."""
        columns = [
            Column(name="id", description="Order ID", data_type="integer"),
            Column(name="amount", description="Order amount", data_type="decimal")
        ]

        model = Model(
            name="orders",
            description="Order transactions",
            columns=columns,
            config={"materialized": "table"}
        )

        assert model.name == "orders"
        assert len(model.columns) == 2
        assert model.columns[0].name == "id"
        assert model.config["materialized"] == "table"

    def test_model_with_tests(self):
        """Test model with tests."""
        model = Model(
            name="customers",
            description="Customer data",
            columns=[],
            tests=[
                {"test": "unique", "column_name": "customer_id"},
                {"test": "relationships", "to": "ref('orders')", "field": "customer_id"}
            ]
        )

        assert len(model.tests) == 2
        assert model.tests[0]["test"] == "unique"


class TestSource:
    """Test Source dataclass."""

    def test_create_source(self):
        """Test creating a source."""
        tables = [
            {"name": "raw_orders", "columns": []},
            {"name": "raw_customers", "columns": []}
        ]

        source = Source(
            name="raw_data",
            database="analytics",
            schema="staging",
            description="Raw data from production",
            loader="fivetran",
            tables=tables
        )

        assert source.name == "raw_data"
        assert source.database == "analytics"
        assert source.schema == "staging"
        assert source.loader == "fivetran"
        assert len(source.tables) == 2

    def test_source_with_freshness(self):
        """Test source with freshness check."""
        source = Source(
            name="events",
            database="analytics",
            schema="raw",
            description="Event stream data",
            loader="kafka",
            tables=[],
            freshness={
                "warn_after": {"count": 6, "period": "hour"},
                "error_after": {"count": 12, "period": "hour"}
            }
        )

        assert source.freshness is not None
        assert source.freshness["warn_after"]["count"] == 6


class TestDbtProject:
    """Test DbtProject dataclass."""

    def test_create_project(self):
        """Test creating a dbt project."""
        project = DbtProject(
            name="semantic_layer",
            version="1.0.0"
        )

        assert project.name == "semantic_layer"
        assert project.version == "1.0.0"
        assert project.models == []
        assert project.sources == []
        assert project.metrics == []

    def test_project_with_components(self):
        """Test project with various components."""
        model = Model(
            name="orders",
            description="Orders model",
            columns=[]
        )

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

        project = DbtProject(
            name="analytics",
            version="2.0.0",
            models=[model],
            metrics=[metric]
        )

        assert len(project.models) == 1
        assert project.models[0].name == "orders"
        assert len(project.metrics) == 1
        assert project.metrics[0].name == "revenue"