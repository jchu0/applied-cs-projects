"""Tests for dimension model definitions and builders."""

import pytest
from datetime import date
from typing import List

from semantic_layer.dimensions import (
    DimensionColumn,
    DimensionModel,
    DimCustomers,
    DimProducts,
    DimDates,
    DimensionBuilder,
)


class TestDimensionColumn:
    """Test DimensionColumn dataclass."""

    def test_create_basic_column(self):
        """Test creating a basic dimension column."""
        col = DimensionColumn(
            name="customer_id",
            data_type="string",
            description="Unique customer identifier",
        )

        assert col.name == "customer_id"
        assert col.data_type == "string"
        assert col.description == "Unique customer identifier"
        assert col.is_key is False
        assert col.is_natural_key is False
        assert col.is_scd_valid is False

    def test_create_key_column(self):
        """Test creating a primary key column."""
        col = DimensionColumn(
            name="product_id",
            data_type="integer",
            description="Primary key",
            is_key=True,
        )

        assert col.is_key is True
        assert col.is_natural_key is False

    def test_create_natural_key_column(self):
        """Test creating a natural key column."""
        col = DimensionColumn(
            name="email",
            data_type="string",
            description="Customer email - natural key",
            is_natural_key=True,
        )

        assert col.is_natural_key is True
        assert col.is_key is False

    def test_create_scd_valid_column(self):
        """Test creating an SCD valid column."""
        col = DimensionColumn(
            name="is_current",
            data_type="boolean",
            description="SCD2 valid flag",
            is_scd_valid=True,
        )

        assert col.is_scd_valid is True


class TestDimensionModel:
    """Test DimensionModel dataclass."""

    def test_create_basic_dimension(self):
        """Test creating a basic dimension model."""
        columns = [
            DimensionColumn(
                name="id",
                data_type="integer",
                description="Primary key",
                is_key=True,
            ),
            DimensionColumn(
                name="name",
                data_type="string",
                description="Name field",
            ),
        ]

        dim = DimensionModel(
            name="dim_test",
            description="Test dimension",
            source_model="stg_test",
            columns=columns,
            grain="one row per record",
        )

        assert dim.name == "dim_test"
        assert dim.description == "Test dimension"
        assert dim.source_model == "stg_test"
        assert len(dim.columns) == 2
        assert dim.grain == "one row per record"
        assert dim.scd_type == 1
        assert dim.unique_key == ""
        assert dim.config == {}

    def test_dimension_with_scd_type2(self):
        """Test dimension with SCD Type 2."""
        dim = DimensionModel(
            name="dim_customers_scd2",
            description="Customer dimension with history",
            source_model="stg_customers",
            columns=[],
            grain="one row per customer per version",
            scd_type=2,
            unique_key="customer_sk",
        )

        assert dim.scd_type == 2
        assert dim.unique_key == "customer_sk"

    def test_dimension_with_config(self):
        """Test dimension with custom config."""
        dim = DimensionModel(
            name="dim_products",
            description="Product dimension",
            source_model="stg_products",
            columns=[],
            grain="one row per product",
            config={
                "materialized": "table",
                "tags": ["daily", "core"],
                "schema": "marts",
            },
        )

        assert dim.config["materialized"] == "table"
        assert "daily" in dim.config["tags"]
        assert dim.config["schema"] == "marts"


class TestDimCustomers:
    """Test DimCustomers dimension model."""

    def test_get_definition(self):
        """Test getting customer dimension definition."""
        definition = DimCustomers.get_definition()

        assert definition.name == "dim_customers"
        assert "Customer dimension" in definition.description
        assert definition.source_model == "stg_stripe__customers"
        assert definition.grain == "one row per customer"
        assert definition.unique_key == "customer_id"

    def test_columns_exist(self):
        """Test customer dimension has required columns."""
        definition = DimCustomers.get_definition()
        column_names = [col.name for col in definition.columns]

        expected_columns = [
            "customer_id",
            "email",
            "name",
            "country_code",
            "created_at",
            "total_orders",
            "lifetime_value",
            "first_order_date",
            "last_order_date",
            "average_order_value",
            "customer_segment",
            "is_churned",
        ]

        for expected in expected_columns:
            assert expected in column_names, f"Missing column: {expected}"

    def test_primary_key_defined(self):
        """Test customer_id is defined as primary key."""
        definition = DimCustomers.get_definition()
        key_columns = [col for col in definition.columns if col.is_key]

        assert len(key_columns) == 1
        assert key_columns[0].name == "customer_id"

    def test_natural_key_defined(self):
        """Test email is defined as natural key."""
        definition = DimCustomers.get_definition()
        natural_key_columns = [col for col in definition.columns if col.is_natural_key]

        assert len(natural_key_columns) == 1
        assert natural_key_columns[0].name == "email"

    def test_generate_sql(self):
        """Test SQL generation for customer dimension."""
        sql = DimCustomers.generate_sql()

        assert "materialized='table'" in sql
        assert "unique_key='customer_id'" in sql
        assert "stg_stripe__customers" in sql
        assert "fct_orders" in sql
        assert "customer_segment" in sql
        assert "is_churned" in sql
        assert "lifetime_value" in sql

    def test_config_settings(self):
        """Test dimension config settings."""
        definition = DimCustomers.get_definition()

        assert definition.config["materialized"] == "table"
        assert "daily" in definition.config["tags"]


class TestDimProducts:
    """Test DimProducts dimension model."""

    def test_get_definition(self):
        """Test getting product dimension definition."""
        definition = DimProducts.get_definition()

        assert definition.name == "dim_products"
        assert definition.source_model == "stg_shopify__products"
        assert definition.grain == "one row per product"
        assert definition.unique_key == "product_id"

    def test_columns_exist(self):
        """Test product dimension has required columns."""
        definition = DimProducts.get_definition()
        column_names = [col.name for col in definition.columns]

        expected_columns = [
            "product_id",
            "product_name",
            "category",
            "subcategory",
            "brand",
            "price",
            "cost",
            "is_active",
        ]

        for expected in expected_columns:
            assert expected in column_names, f"Missing column: {expected}"

    def test_primary_key_defined(self):
        """Test product_id is defined as primary key."""
        definition = DimProducts.get_definition()
        key_columns = [col for col in definition.columns if col.is_key]

        assert len(key_columns) == 1
        assert key_columns[0].name == "product_id"


class TestDimDates:
    """Test DimDates dimension model."""

    def test_get_definition(self):
        """Test getting date dimension definition."""
        definition = DimDates.get_definition()

        assert definition.name == "dim_dates"
        assert definition.source_model == "date_spine"
        assert definition.grain == "one row per date"
        assert definition.unique_key == "date_key"

    def test_columns_exist(self):
        """Test date dimension has required columns."""
        definition = DimDates.get_definition()
        column_names = [col.name for col in definition.columns]

        expected_columns = [
            "date_key",
            "date_actual",
            "day_of_week",
            "day_name",
            "is_weekend",
            "week_of_year",
            "month_number",
            "month_name",
            "quarter",
            "year",
            "is_month_end",
            "is_quarter_end",
            "is_year_end",
        ]

        for expected in expected_columns:
            assert expected in column_names, f"Missing column: {expected}"

    def test_primary_key_defined(self):
        """Test date_key is defined as primary key."""
        definition = DimDates.get_definition()
        key_columns = [col for col in definition.columns if col.is_key]

        assert len(key_columns) == 1
        assert key_columns[0].name == "date_key"

    def test_config_settings(self):
        """Test date dimension config settings."""
        definition = DimDates.get_definition()

        assert definition.config["materialized"] == "table"
        assert "static" in definition.config["tags"]

    def test_generate_date_spine_basic(self):
        """Test generating date spine data."""
        start = date(2024, 1, 1)
        end = date(2024, 1, 7)

        dates = DimDates.generate_date_spine(start, end)

        assert len(dates) == 7
        assert dates[0]["date_key"] == 20240101
        assert dates[-1]["date_key"] == 20240107

    def test_generate_date_spine_day_names(self):
        """Test day names in date spine."""
        # January 1, 2024 was a Monday
        start = date(2024, 1, 1)
        end = date(2024, 1, 7)

        dates = DimDates.generate_date_spine(start, end)

        assert dates[0]["day_name"] == "Monday"
        assert dates[5]["day_name"] == "Saturday"
        assert dates[6]["day_name"] == "Sunday"

    def test_generate_date_spine_weekend_flag(self):
        """Test weekend flag in date spine."""
        start = date(2024, 1, 1)
        end = date(2024, 1, 7)

        dates = DimDates.generate_date_spine(start, end)

        # Monday through Friday should not be weekend
        for i in range(5):
            assert dates[i]["is_weekend"] is False

        # Saturday and Sunday should be weekend
        assert dates[5]["is_weekend"] is True
        assert dates[6]["is_weekend"] is True

    def test_generate_date_spine_week_of_year(self):
        """Test week of year calculation."""
        start = date(2024, 1, 1)
        end = date(2024, 1, 14)

        dates = DimDates.generate_date_spine(start, end)

        # Week numbers vary based on ISO week calculation
        assert dates[0]["week_of_year"] == 1
        assert dates[7]["week_of_year"] == 2

    def test_generate_date_spine_month_info(self):
        """Test month information in date spine."""
        start = date(2024, 1, 15)
        end = date(2024, 2, 15)

        dates = DimDates.generate_date_spine(start, end)

        # Check January dates
        assert dates[0]["month_number"] == 1
        assert dates[0]["month_name"] == "January"

        # Check February dates
        assert dates[-1]["month_number"] == 2
        assert dates[-1]["month_name"] == "February"

    def test_generate_date_spine_quarter(self):
        """Test quarter calculation."""
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)

        dates = DimDates.generate_date_spine(start, end)

        # Q1
        jan_date = next(d for d in dates if d["month_number"] == 1)
        assert jan_date["quarter"] == 1

        # Q2
        apr_date = next(d for d in dates if d["month_number"] == 4)
        assert apr_date["quarter"] == 2

        # Q3
        jul_date = next(d for d in dates if d["month_number"] == 7)
        assert jul_date["quarter"] == 3

        # Q4
        oct_date = next(d for d in dates if d["month_number"] == 10)
        assert oct_date["quarter"] == 4

    def test_generate_date_spine_month_end(self):
        """Test month end flag."""
        start = date(2024, 1, 1)
        end = date(2024, 2, 29)  # 2024 is a leap year

        dates = DimDates.generate_date_spine(start, end)

        # January 31 should be month end
        jan_31 = next(d for d in dates if d["date_key"] == 20240131)
        assert jan_31["is_month_end"] is True

        # January 30 should not be month end
        jan_30 = next(d for d in dates if d["date_key"] == 20240130)
        assert jan_30["is_month_end"] is False

        # February 29 (leap year) should be month end
        feb_29 = next(d for d in dates if d["date_key"] == 20240229)
        assert feb_29["is_month_end"] is True

    def test_generate_date_spine_quarter_end(self):
        """Test quarter end flag."""
        start = date(2024, 1, 1)
        end = date(2024, 6, 30)

        dates = DimDates.generate_date_spine(start, end)

        # March 31 should be quarter end
        mar_31 = next(d for d in dates if d["date_key"] == 20240331)
        assert mar_31["is_quarter_end"] is True

        # June 30 should be quarter end
        jun_30 = next(d for d in dates if d["date_key"] == 20240630)
        assert jun_30["is_quarter_end"] is True

        # March 30 should not be quarter end
        mar_30 = next(d for d in dates if d["date_key"] == 20240330)
        assert mar_30["is_quarter_end"] is False

    def test_generate_date_spine_year_end(self):
        """Test year end flag."""
        start = date(2024, 12, 25)
        end = date(2024, 12, 31)

        dates = DimDates.generate_date_spine(start, end)

        # December 31 should be year end
        dec_31 = next(d for d in dates if d["date_key"] == 20241231)
        assert dec_31["is_year_end"] is True

        # December 30 should not be year end
        dec_30 = next(d for d in dates if d["date_key"] == 20241230)
        assert dec_30["is_year_end"] is False

    def test_generate_date_spine_single_day(self):
        """Test generating date spine for a single day."""
        single_date = date(2024, 7, 4)

        dates = DimDates.generate_date_spine(single_date, single_date)

        assert len(dates) == 1
        assert dates[0]["date_key"] == 20240704
        assert dates[0]["month_name"] == "July"
        assert dates[0]["day_name"] == "Thursday"


class TestDimensionBuilder:
    """Test DimensionBuilder class."""

    def test_add_dimension(self):
        """Test adding a dimension to the builder."""
        builder = DimensionBuilder()

        dim = DimensionModel(
            name="dim_test",
            description="Test dimension",
            source_model="stg_test",
            columns=[],
            grain="one row per record",
        )

        result = builder.add_dimension(dim)

        # Should return self for chaining
        assert result is builder
        assert builder.get_dimension("dim_test") is not None

    def test_add_multiple_dimensions(self):
        """Test adding multiple dimensions."""
        builder = DimensionBuilder()

        dim1 = DimensionModel(
            name="dim_one",
            description="First dimension",
            source_model="stg_one",
            columns=[],
            grain="one row per record",
        )

        dim2 = DimensionModel(
            name="dim_two",
            description="Second dimension",
            source_model="stg_two",
            columns=[],
            grain="one row per record",
        )

        builder.add_dimension(dim1).add_dimension(dim2)

        assert len(builder.list_dimensions()) == 2
        assert "dim_one" in builder.list_dimensions()
        assert "dim_two" in builder.list_dimensions()

    def test_get_dimension(self):
        """Test retrieving a dimension."""
        builder = DimensionBuilder()

        dim = DimensionModel(
            name="dim_test",
            description="Test dimension",
            source_model="stg_test",
            columns=[],
            grain="one row per record",
        )

        builder.add_dimension(dim)

        retrieved = builder.get_dimension("dim_test")
        assert retrieved is not None
        assert retrieved.name == "dim_test"
        assert retrieved.description == "Test dimension"

    def test_get_nonexistent_dimension(self):
        """Test getting a dimension that doesn't exist."""
        builder = DimensionBuilder()

        result = builder.get_dimension("nonexistent")
        assert result is None

    def test_list_dimensions(self):
        """Test listing all dimensions."""
        builder = DimensionBuilder()

        dims = [
            DimensionModel(
                name=f"dim_{i}",
                description=f"Dimension {i}",
                source_model=f"stg_{i}",
                columns=[],
                grain="one row per record",
            )
            for i in range(5)
        ]

        for dim in dims:
            builder.add_dimension(dim)

        dim_list = builder.list_dimensions()

        assert len(dim_list) == 5
        for i in range(5):
            assert f"dim_{i}" in dim_list

    def test_generate_yaml_basic(self):
        """Test generating YAML schema for a dimension."""
        builder = DimensionBuilder()

        dim = DimensionModel(
            name="dim_test",
            description="Test dimension",
            source_model="stg_test",
            columns=[
                DimensionColumn(
                    name="test_id",
                    data_type="integer",
                    description="Primary key",
                    is_key=True,
                ),
                DimensionColumn(
                    name="name",
                    data_type="string",
                    description="Name field",
                ),
            ],
            grain="one row per record",
        )

        builder.add_dimension(dim)

        yaml_output = builder.generate_yaml("dim_test")

        assert "version: 2" in yaml_output
        assert "name: dim_test" in yaml_output
        assert '"Test dimension"' in yaml_output
        assert "name: test_id" in yaml_output
        assert "unique" in yaml_output
        assert "not_null" in yaml_output

    def test_generate_yaml_key_tests(self):
        """Test YAML includes tests for key columns."""
        builder = DimensionBuilder()

        dim = DimensionModel(
            name="dim_test",
            description="Test dimension",
            source_model="stg_test",
            columns=[
                DimensionColumn(
                    name="id",
                    data_type="integer",
                    description="Primary key",
                    is_key=True,
                ),
            ],
            grain="one row per record",
        )

        builder.add_dimension(dim)

        yaml_output = builder.generate_yaml("dim_test")

        # Key columns should have unique and not_null tests
        assert "unique" in yaml_output
        assert "not_null" in yaml_output

    def test_generate_yaml_nonexistent(self):
        """Test generating YAML for nonexistent dimension."""
        builder = DimensionBuilder()

        result = builder.generate_yaml("nonexistent")
        assert result == ""

    def test_add_predefined_dimensions(self):
        """Test adding predefined dimension models."""
        builder = DimensionBuilder()

        builder.add_dimension(DimCustomers.get_definition())
        builder.add_dimension(DimProducts.get_definition())
        builder.add_dimension(DimDates.get_definition())

        dim_list = builder.list_dimensions()

        assert "dim_customers" in dim_list
        assert "dim_products" in dim_list
        assert "dim_dates" in dim_list
