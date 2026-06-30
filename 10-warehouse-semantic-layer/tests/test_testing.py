"""Tests for the dbt testing framework module."""

import pytest
from typing import List, Dict, Any

from semantic_layer.testing import (
    TestType,
    TestDefinition,
    TestResult,
    TestBuilder,
    DataQualityChecker,
    create_dimension_tests,
    create_fact_tests,
)


class TestTestType:
    """Test TestType enum."""

    def test_test_types_exist(self):
        """Test all test types are defined."""
        assert TestType.UNIQUE.value == "unique"
        assert TestType.NOT_NULL.value == "not_null"
        assert TestType.ACCEPTED_VALUES.value == "accepted_values"
        assert TestType.RELATIONSHIPS.value == "relationships"
        assert TestType.EXPRESSION_IS_TRUE.value == "expression_is_true"
        assert TestType.CUSTOM.value == "custom"

    def test_test_type_from_string(self):
        """Test creating test type from string."""
        assert TestType("unique") == TestType.UNIQUE
        assert TestType("not_null") == TestType.NOT_NULL
        assert TestType("relationships") == TestType.RELATIONSHIPS


class TestTestDefinition:
    """Test TestDefinition dataclass."""

    def test_create_basic_test(self):
        """Test creating a basic test definition."""
        test = TestDefinition(
            name="unique_orders_order_id",
            test_type=TestType.UNIQUE,
            model="orders",
            column="order_id",
        )

        assert test.name == "unique_orders_order_id"
        assert test.test_type == TestType.UNIQUE
        assert test.model == "orders"
        assert test.column == "order_id"
        assert test.config == {}
        assert test.severity == "error"

    def test_test_with_config(self):
        """Test definition with configuration."""
        test = TestDefinition(
            name="accepted_values_status",
            test_type=TestType.ACCEPTED_VALUES,
            model="orders",
            column="status",
            config={"values": ["pending", "completed", "cancelled"]},
        )

        assert test.config["values"] == ["pending", "completed", "cancelled"]

    def test_test_with_warning_severity(self):
        """Test definition with warning severity."""
        test = TestDefinition(
            name="not_null_optional_field",
            test_type=TestType.NOT_NULL,
            model="customers",
            column="phone",
            severity="warn",
        )

        assert test.severity == "warn"

    def test_model_level_test(self):
        """Test definition at model level (no column)."""
        test = TestDefinition(
            name="custom_validation",
            test_type=TestType.CUSTOM,
            model="orders",
            column=None,
            config={"sql": "SELECT * FROM orders WHERE amount < 0"},
        )

        assert test.column is None
        assert "sql" in test.config


class TestTestResult:
    """Test TestResult dataclass."""

    def test_create_pass_result(self):
        """Test creating a passing test result."""
        result = TestResult(
            test_name="unique_orders_order_id",
            status="pass",
            failures=0,
            message="Test passed",
        )

        assert result.test_name == "unique_orders_order_id"
        assert result.status == "pass"
        assert result.failures == 0

    def test_create_fail_result(self):
        """Test creating a failing test result."""
        result = TestResult(
            test_name="not_null_customers_email",
            status="fail",
            failures=15,
            message="15 records with null email",
            sql="SELECT * FROM customers WHERE email IS NULL",
        )

        assert result.status == "fail"
        assert result.failures == 15
        assert result.sql is not None

    def test_create_warn_result(self):
        """Test creating a warning test result."""
        result = TestResult(
            test_name="freshness_check",
            status="warn",
            failures=0,
            message="Data is stale but within warning threshold",
        )

        assert result.status == "warn"

    def test_create_error_result(self):
        """Test creating an error test result."""
        result = TestResult(
            test_name="custom_test",
            status="error",
            failures=0,
            message="Test encountered an error: SQL syntax error",
        )

        assert result.status == "error"


class TestTestBuilder:
    """Test TestBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a test builder."""
        return TestBuilder()

    def test_add_unique_test(self, builder):
        """Test adding a unique test."""
        result = builder.add_unique_test("orders", "order_id")

        # Should return self for chaining
        assert result is builder

        tests = builder.get_tests()
        assert len(tests) == 1
        assert tests[0].test_type == TestType.UNIQUE
        assert tests[0].model == "orders"
        assert tests[0].column == "order_id"

    def test_add_not_null_test(self, builder):
        """Test adding a not_null test."""
        builder.add_not_null_test("customers", "customer_id")

        tests = builder.get_tests()
        assert len(tests) == 1
        assert tests[0].test_type == TestType.NOT_NULL

    def test_add_accepted_values_test(self, builder):
        """Test adding an accepted_values test."""
        builder.add_accepted_values_test(
            "orders", "status", ["pending", "completed", "cancelled"]
        )

        tests = builder.get_tests()
        assert len(tests) == 1
        assert tests[0].test_type == TestType.ACCEPTED_VALUES
        assert tests[0].config["values"] == ["pending", "completed", "cancelled"]

    def test_add_relationships_test(self, builder):
        """Test adding a relationships test."""
        builder.add_relationships_test(
            "orders", "customer_id", "customers", "id"
        )

        tests = builder.get_tests()
        assert len(tests) == 1
        assert tests[0].test_type == TestType.RELATIONSHIPS
        assert tests[0].config["to"] == "customers"
        assert tests[0].config["field"] == "id"

    def test_add_expression_test(self, builder):
        """Test adding an expression_is_true test."""
        builder.add_expression_test(
            "orders", "amount", ">= 0"
        )

        tests = builder.get_tests()
        assert len(tests) == 1
        assert tests[0].test_type == TestType.EXPRESSION_IS_TRUE
        assert tests[0].config["expression"] == ">= 0"

    def test_add_custom_test(self, builder):
        """Test adding a custom test."""
        sql = "SELECT * FROM orders WHERE amount < 0"
        builder.add_custom_test(
            "no_negative_amounts", "orders", sql
        )

        tests = builder.get_tests()
        assert len(tests) == 1
        assert tests[0].test_type == TestType.CUSTOM
        assert tests[0].config["sql"] == sql

    def test_chain_multiple_tests(self, builder):
        """Test chaining multiple tests."""
        builder \
            .add_unique_test("orders", "order_id") \
            .add_not_null_test("orders", "order_id") \
            .add_not_null_test("orders", "customer_id") \
            .add_relationships_test("orders", "customer_id", "customers", "id")

        tests = builder.get_tests()
        assert len(tests) == 4

    def test_get_tests_for_model(self, builder):
        """Test getting tests for a specific model."""
        builder.add_unique_test("orders", "order_id")
        builder.add_not_null_test("orders", "customer_id")
        builder.add_unique_test("customers", "customer_id")
        builder.add_not_null_test("customers", "email")

        order_tests = builder.get_tests_for_model("orders")
        customer_tests = builder.get_tests_for_model("customers")

        assert len(order_tests) == 2
        assert len(customer_tests) == 2
        assert all(t.model == "orders" for t in order_tests)
        assert all(t.model == "customers" for t in customer_tests)

    def test_test_with_severity(self, builder):
        """Test adding tests with custom severity."""
        builder.add_not_null_test("customers", "phone", severity="warn")

        tests = builder.get_tests()
        assert tests[0].severity == "warn"

    def test_generate_yaml_unique_and_not_null(self, builder):
        """Test YAML generation for unique and not_null tests."""
        builder.add_unique_test("orders", "order_id")
        builder.add_not_null_test("orders", "order_id")

        yaml_output = builder.generate_yaml("orders")

        assert "name: order_id" in yaml_output
        assert "- unique" in yaml_output
        assert "- not_null" in yaml_output

    def test_generate_yaml_accepted_values(self, builder):
        """Test YAML generation for accepted_values test."""
        builder.add_accepted_values_test(
            "orders", "status", ["pending", "completed"]
        )

        yaml_output = builder.generate_yaml("orders")

        assert "name: status" in yaml_output
        assert "- accepted_values:" in yaml_output
        assert "'pending'" in yaml_output
        assert "'completed'" in yaml_output

    def test_generate_yaml_relationships(self, builder):
        """Test YAML generation for relationships test."""
        builder.add_relationships_test(
            "orders", "customer_id", "customers", "id"
        )

        yaml_output = builder.generate_yaml("orders")

        assert "- relationships:" in yaml_output
        assert "ref('customers')" in yaml_output
        assert "field: id" in yaml_output

    def test_generate_yaml_expression(self, builder):
        """Test YAML generation for expression_is_true test."""
        builder.add_expression_test("orders", "amount", ">= 0")

        yaml_output = builder.generate_yaml("orders")

        assert "- dbt_utils.expression_is_true:" in yaml_output
        assert ">= 0" in yaml_output

    def test_generate_yaml_empty(self, builder):
        """Test YAML generation with no tests."""
        yaml_output = builder.generate_yaml("nonexistent")

        assert yaml_output == ""

    def test_generate_yaml_multiple_columns(self, builder):
        """Test YAML generation for multiple columns."""
        builder.add_unique_test("orders", "order_id")
        builder.add_not_null_test("orders", "order_id")
        builder.add_not_null_test("orders", "customer_id")
        builder.add_expression_test("orders", "amount", "> 0")

        yaml_output = builder.generate_yaml("orders")

        assert "name: order_id" in yaml_output
        assert "name: customer_id" in yaml_output
        assert "name: amount" in yaml_output


class TestDataQualityChecker:
    """Test DataQualityChecker class."""

    @pytest.fixture
    def checker(self):
        """Create a data quality checker."""
        return DataQualityChecker()

    def test_add_reconciliation_check(self, checker):
        """Test adding a reconciliation check."""
        result = checker.add_reconciliation_check(
            name="orders_revenue_check",
            source_query="SELECT SUM(amount) as total FROM raw.orders",
            target_query="SELECT SUM(order_total) as total FROM fct_orders",
            tolerance=0.01,
        )

        # Should return self for chaining
        assert result is checker
        assert len(checker._checks) == 1
        assert checker._checks[0]["type"] == "reconciliation"
        assert checker._checks[0]["tolerance"] == 0.01

    def test_add_row_count_check(self, checker):
        """Test adding a row count check."""
        checker.add_row_count_check(
            name="orders_row_count",
            model="fct_orders",
            min_rows=1000,
            max_rows=100000,
        )

        assert len(checker._checks) == 1
        assert checker._checks[0]["type"] == "row_count"
        assert checker._checks[0]["min_rows"] == 1000
        assert checker._checks[0]["max_rows"] == 100000

    def test_add_row_count_check_no_max(self, checker):
        """Test row count check with no maximum."""
        checker.add_row_count_check(
            name="orders_has_data",
            model="fct_orders",
            min_rows=1,
        )

        assert checker._checks[0]["max_rows"] is None

    def test_add_freshness_check(self, checker):
        """Test adding a freshness check."""
        checker.add_freshness_check(
            name="orders_freshness",
            model="fct_orders",
            timestamp_column="updated_at",
            max_hours=24,
        )

        assert len(checker._checks) == 1
        assert checker._checks[0]["type"] == "freshness"
        assert checker._checks[0]["timestamp_column"] == "updated_at"
        assert checker._checks[0]["max_hours"] == 24

    def test_chain_multiple_checks(self, checker):
        """Test chaining multiple checks."""
        checker \
            .add_reconciliation_check(
                "revenue_check",
                "SELECT SUM(amount) as total FROM raw.orders",
                "SELECT SUM(order_total) as total FROM fct_orders",
            ) \
            .add_row_count_check("row_count", "fct_orders", min_rows=100) \
            .add_freshness_check("freshness", "fct_orders", "updated_at", 12)

        assert len(checker._checks) == 3

    def test_generate_reconciliation_sql(self, checker):
        """Test generating SQL for reconciliation check."""
        check = {
            "name": "revenue_reconciliation",
            "source_query": "SELECT SUM(amount) as total FROM raw_orders",
            "target_query": "SELECT SUM(order_total) as total FROM fct_orders",
            "tolerance": 0.01,
        }

        sql = checker.generate_reconciliation_sql(check)

        assert "-- revenue_reconciliation" in sql
        assert "source_value" in sql
        assert "target_value" in sql
        assert "SELECT SUM(amount) as total FROM raw_orders" in sql
        assert "SELECT SUM(order_total) as total FROM fct_orders" in sql
        assert "diff_ratio" in sql
        assert "0.01" in sql


class TestFactoryFunctions:
    """Test factory functions for creating test suites."""

    def test_create_dimension_tests(self):
        """Test creating standard dimension tests."""
        builder = create_dimension_tests("dim_customers", "customer_id")

        tests = builder.get_tests()

        # Should have unique and not_null for primary key
        assert len(tests) == 2
        assert any(t.test_type == TestType.UNIQUE for t in tests)
        assert any(t.test_type == TestType.NOT_NULL for t in tests)
        assert all(t.model == "dim_customers" for t in tests)
        assert all(t.column == "customer_id" for t in tests)

    def test_create_fact_tests_basic(self):
        """Test creating standard fact tests with primary key only."""
        builder = create_fact_tests(
            model_name="fct_orders",
            primary_key="order_id",
            foreign_keys=[],
            measures=[],
        )

        tests = builder.get_tests()

        # Should have unique and not_null for primary key
        assert len(tests) == 2
        assert any(t.test_type == TestType.UNIQUE for t in tests)
        assert any(t.test_type == TestType.NOT_NULL for t in tests)

    def test_create_fact_tests_with_foreign_keys(self):
        """Test creating fact tests with foreign keys."""
        foreign_keys = [
            {
                "column": "customer_id",
                "references_model": "dim_customers",
                "references_column": "customer_id",
            },
            {
                "column": "product_id",
                "references_model": "dim_products",
                "references_column": "product_id",
            },
        ]

        builder = create_fact_tests(
            model_name="fct_order_items",
            primary_key="order_item_id",
            foreign_keys=foreign_keys,
            measures=[],
        )

        tests = builder.get_tests()

        # Primary key: unique + not_null = 2
        # Foreign keys: 2 * (not_null + relationships) = 4
        # Total = 6
        assert len(tests) == 6

        # Check foreign key tests
        fk_tests = [t for t in tests if t.column == "customer_id"]
        assert any(t.test_type == TestType.NOT_NULL for t in fk_tests)
        assert any(t.test_type == TestType.RELATIONSHIPS for t in fk_tests)

    def test_create_fact_tests_with_measures(self):
        """Test creating fact tests with measures."""
        measures = ["quantity", "unit_price", "total_amount"]

        builder = create_fact_tests(
            model_name="fct_orders",
            primary_key="order_id",
            foreign_keys=[],
            measures=measures,
        )

        tests = builder.get_tests()

        # Primary key: 2 tests
        # Measures: 3 * (not_null + expression) = 6 tests
        # Total = 8
        assert len(tests) == 8

        # Check measure tests
        for measure in measures:
            measure_tests = [t for t in tests if t.column == measure]
            assert any(t.test_type == TestType.NOT_NULL for t in measure_tests)
            assert any(t.test_type == TestType.EXPRESSION_IS_TRUE for t in measure_tests)

    def test_create_fact_tests_full(self):
        """Test creating complete fact test suite."""
        foreign_keys = [
            {
                "column": "customer_id",
                "references_model": "dim_customers",
                "references_column": "customer_id",
            },
        ]
        measures = ["order_total", "quantity"]

        builder = create_fact_tests(
            model_name="fct_orders",
            primary_key="order_id",
            foreign_keys=foreign_keys,
            measures=measures,
        )

        tests = builder.get_tests()

        # Primary key: 2 tests
        # Foreign keys: 1 * 2 = 2 tests
        # Measures: 2 * 2 = 4 tests
        # Total = 8
        assert len(tests) == 8

        # Verify all models are correct
        assert all(t.model == "fct_orders" for t in tests)


class TestIntegrationScenarios:
    """Integration tests for the testing module."""

    def test_complete_dimension_testing_workflow(self):
        """Test complete workflow for testing a dimension."""
        # Create tests for a customer dimension
        builder = TestBuilder()

        builder \
            .add_unique_test("dim_customers", "customer_id") \
            .add_not_null_test("dim_customers", "customer_id") \
            .add_not_null_test("dim_customers", "email") \
            .add_accepted_values_test(
                "dim_customers", "segment",
                ["prospect", "new", "active", "loyal", "churned"]
            ) \
            .add_expression_test(
                "dim_customers", "lifetime_value", ">= 0"
            )

        # Generate YAML
        yaml_output = builder.generate_yaml("dim_customers")

        # Verify structure
        assert "name: customer_id" in yaml_output
        assert "name: email" in yaml_output
        assert "name: segment" in yaml_output
        assert "name: lifetime_value" in yaml_output
        assert "unique" in yaml_output
        assert "not_null" in yaml_output
        assert "accepted_values" in yaml_output

    def test_complete_fact_testing_workflow(self):
        """Test complete workflow for testing a fact table."""
        # Create tests for an orders fact
        builder = create_fact_tests(
            model_name="fct_orders",
            primary_key="order_id",
            foreign_keys=[
                {
                    "column": "customer_id",
                    "references_model": "dim_customers",
                    "references_column": "customer_id",
                },
                {
                    "column": "product_id",
                    "references_model": "dim_products",
                    "references_column": "product_id",
                },
            ],
            measures=["quantity", "unit_price", "order_total"],
        )

        # Add additional custom tests
        builder.add_accepted_values_test(
            "fct_orders", "status",
            ["pending", "processing", "shipped", "delivered", "cancelled"]
        )
        builder.add_expression_test(
            "fct_orders", "order_date", "IS NOT NULL"
        )

        # Generate YAML
        yaml_output = builder.generate_yaml("fct_orders")

        # Verify complete test suite
        tests = builder.get_tests()
        assert len(tests) == 14  # 2 (PK) + 4 (FK) + 6 (measures) + 2 (custom)

    def test_data_quality_pipeline(self):
        """Test complete data quality checking pipeline."""
        checker = DataQualityChecker()

        # Add various quality checks
        checker.add_reconciliation_check(
            name="revenue_reconciliation",
            source_query="SELECT SUM(amount) as total FROM raw.orders",
            target_query="SELECT SUM(order_total) as total FROM fct_orders",
            tolerance=0.01,
        )

        checker.add_row_count_check(
            name="orders_volume",
            model="fct_orders",
            min_rows=1000,
            max_rows=1000000,
        )

        checker.add_freshness_check(
            name="orders_freshness",
            model="fct_orders",
            timestamp_column="updated_at",
            max_hours=24,
        )

        # Verify all checks are registered
        assert len(checker._checks) == 3

        # Verify reconciliation SQL can be generated
        recon_check = checker._checks[0]
        sql = checker.generate_reconciliation_sql(recon_check)
        assert "source_value" in sql
        assert "target_value" in sql
        assert "diff_ratio" in sql
