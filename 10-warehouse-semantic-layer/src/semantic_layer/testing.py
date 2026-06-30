"""Testing framework for dbt models."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of dbt tests."""

    UNIQUE = "unique"
    NOT_NULL = "not_null"
    ACCEPTED_VALUES = "accepted_values"
    RELATIONSHIPS = "relationships"
    EXPRESSION_IS_TRUE = "expression_is_true"
    CUSTOM = "custom"


@dataclass
class TestDefinition:
    """Definition of a test."""

    name: str
    test_type: TestType
    model: str
    column: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # error, warn


@dataclass
class TestResult:
    """Result of a test execution."""

    test_name: str
    status: str  # pass, fail, warn, error
    failures: int = 0
    message: str = ""
    sql: str = ""


class TestBuilder:
    """Builder for creating dbt tests."""

    def __init__(self):
        self._tests: List[TestDefinition] = []

    def add_unique_test(
        self, model: str, column: str, severity: str = "error"
    ) -> "TestBuilder":
        """Add a unique test."""
        self._tests.append(TestDefinition(
            name=f"unique_{model}_{column}",
            test_type=TestType.UNIQUE,
            model=model,
            column=column,
            severity=severity,
        ))
        return self

    def add_not_null_test(
        self, model: str, column: str, severity: str = "error"
    ) -> "TestBuilder":
        """Add a not_null test."""
        self._tests.append(TestDefinition(
            name=f"not_null_{model}_{column}",
            test_type=TestType.NOT_NULL,
            model=model,
            column=column,
            severity=severity,
        ))
        return self

    def add_accepted_values_test(
        self,
        model: str,
        column: str,
        values: List[Any],
        severity: str = "error",
    ) -> "TestBuilder":
        """Add an accepted_values test."""
        self._tests.append(TestDefinition(
            name=f"accepted_values_{model}_{column}",
            test_type=TestType.ACCEPTED_VALUES,
            model=model,
            column=column,
            config={"values": values},
            severity=severity,
        ))
        return self

    def add_relationships_test(
        self,
        model: str,
        column: str,
        to_model: str,
        to_column: str,
        severity: str = "error",
    ) -> "TestBuilder":
        """Add a relationships test."""
        self._tests.append(TestDefinition(
            name=f"relationships_{model}_{column}",
            test_type=TestType.RELATIONSHIPS,
            model=model,
            column=column,
            config={"to": to_model, "field": to_column},
            severity=severity,
        ))
        return self

    def add_expression_test(
        self,
        model: str,
        column: str,
        expression: str,
        severity: str = "error",
    ) -> "TestBuilder":
        """Add an expression_is_true test."""
        self._tests.append(TestDefinition(
            name=f"expression_{model}_{column}",
            test_type=TestType.EXPRESSION_IS_TRUE,
            model=model,
            column=column,
            config={"expression": expression},
            severity=severity,
        ))
        return self

    def add_custom_test(
        self,
        name: str,
        model: str,
        sql: str,
        severity: str = "error",
    ) -> "TestBuilder":
        """Add a custom test."""
        self._tests.append(TestDefinition(
            name=name,
            test_type=TestType.CUSTOM,
            model=model,
            config={"sql": sql},
            severity=severity,
        ))
        return self

    def get_tests(self) -> List[TestDefinition]:
        """Get all tests."""
        return self._tests

    def get_tests_for_model(self, model: str) -> List[TestDefinition]:
        """Get tests for a specific model."""
        return [t for t in self._tests if t.model == model]

    def generate_yaml(self, model: str) -> str:
        """Generate YAML test definitions for a model."""
        tests = self.get_tests_for_model(model)
        if not tests:
            return ""

        columns: Dict[str, List[str]] = {}

        for test in tests:
            if test.column:
                if test.column not in columns:
                    columns[test.column] = []

                if test.test_type == TestType.UNIQUE:
                    columns[test.column].append("- unique")
                elif test.test_type == TestType.NOT_NULL:
                    columns[test.column].append("- not_null")
                elif test.test_type == TestType.ACCEPTED_VALUES:
                    values = test.config.get("values", [])
                    values_str = ", ".join(f"'{v}'" for v in values)
                    columns[test.column].append(
                        f"- accepted_values:\n              values: [{values_str}]"
                    )
                elif test.test_type == TestType.RELATIONSHIPS:
                    columns[test.column].append(
                        f"- relationships:\n"
                        f"              to: ref('{test.config['to']}')\n"
                        f"              field: {test.config['field']}"
                    )
                elif test.test_type == TestType.EXPRESSION_IS_TRUE:
                    columns[test.column].append(
                        f"- dbt_utils.expression_is_true:\n"
                        f'              expression: "{test.config["expression"]}"'
                    )

        yaml_parts = []
        for col_name, col_tests in columns.items():
            yaml_parts.append(f"      - name: {col_name}")
            yaml_parts.append("        tests:")
            for test_yaml in col_tests:
                yaml_parts.append(f"          {test_yaml}")

        return "\n".join(yaml_parts)


class DataQualityChecker:
    """Run data quality checks on models."""

    def __init__(self):
        self._checks: List[Dict[str, Any]] = []

    def add_reconciliation_check(
        self,
        name: str,
        source_query: str,
        target_query: str,
        tolerance: float = 0.01,
    ) -> "DataQualityChecker":
        """Add a reconciliation check between source and target."""
        self._checks.append({
            "name": name,
            "type": "reconciliation",
            "source_query": source_query,
            "target_query": target_query,
            "tolerance": tolerance,
        })
        return self

    def add_row_count_check(
        self,
        name: str,
        model: str,
        min_rows: int = 0,
        max_rows: Optional[int] = None,
    ) -> "DataQualityChecker":
        """Add a row count check."""
        self._checks.append({
            "name": name,
            "type": "row_count",
            "model": model,
            "min_rows": min_rows,
            "max_rows": max_rows,
        })
        return self

    def add_freshness_check(
        self,
        name: str,
        model: str,
        timestamp_column: str,
        max_hours: int = 24,
    ) -> "DataQualityChecker":
        """Add a freshness check."""
        self._checks.append({
            "name": name,
            "type": "freshness",
            "model": model,
            "timestamp_column": timestamp_column,
            "max_hours": max_hours,
        })
        return self

    def generate_reconciliation_sql(self, check: Dict[str, Any]) -> str:
        """Generate SQL for reconciliation test."""
        return f"""-- {check['name']}
with source_value as (
    {check['source_query']}
),

target_value as (
    {check['target_query']}
)

select
    source_value.total as source_total,
    target_value.total as target_total,
    abs(source_value.total - target_value.total) as difference,
    abs(source_value.total - target_value.total) / nullif(source_value.total, 0) as diff_ratio
from source_value, target_value
where abs(source_value.total - target_value.total) / nullif(source_value.total, 0) > {check['tolerance']}"""


# Pre-built test suites for common patterns
def create_dimension_tests(model_name: str, primary_key: str) -> TestBuilder:
    """Create standard tests for a dimension table."""
    builder = TestBuilder()
    builder.add_unique_test(model_name, primary_key)
    builder.add_not_null_test(model_name, primary_key)
    return builder


def create_fact_tests(
    model_name: str,
    primary_key: str,
    foreign_keys: List[Dict[str, str]],
    measures: List[str],
) -> TestBuilder:
    """Create standard tests for a fact table."""
    builder = TestBuilder()

    # Primary key tests
    builder.add_unique_test(model_name, primary_key)
    builder.add_not_null_test(model_name, primary_key)

    # Foreign key tests
    for fk in foreign_keys:
        builder.add_not_null_test(model_name, fk["column"])
        builder.add_relationships_test(
            model_name,
            fk["column"],
            fk["references_model"],
            fk["references_column"],
        )

    # Measure tests (non-negative)
    for measure in measures:
        builder.add_not_null_test(model_name, measure)
        builder.add_expression_test(model_name, measure, ">= 0")

    return builder
