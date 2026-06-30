"""Data quality validation engine."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Lazy pyspark imports
PYSPARK_AVAILABLE = False
try:
    from pyspark.sql import DataFrame
    from pyspark.sql.functions import col, count, when
    PYSPARK_AVAILABLE = True
except ImportError:
    DataFrame = Any
    col = count = when = None


def _require_pyspark():
    """Raise an error if pyspark is not available."""
    if not PYSPARK_AVAILABLE:
        raise ImportError(
            "PySpark is required for data quality validation. "
            "Install it with: pip install pyspark>=3.4.0 delta-spark>=2.4.0"
        )


@dataclass
class ExpectationResult:
    """Result of a single expectation check."""

    expectation_type: str
    column: Optional[str]
    success: bool
    result: Dict[str, Any]
    exception_info: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating a DataFrame against expectations."""

    success: bool
    results: List[ExpectationResult]
    statistics: Dict[str, Any]

    @property
    def failed_expectations(self) -> List[ExpectationResult]:
        """Get list of failed expectations."""
        return [r for r in self.results if not r.success]


class QualityEngine:
    """Data quality validation engine with Great Expectations-like interface."""

    def __init__(self):
        self._expectations: List[Dict[str, Any]] = []

    def expect_column_to_exist(self, column: str) -> "QualityEngine":
        """Expect a column to exist in the DataFrame."""
        self._expectations.append({
            "type": "expect_column_to_exist",
            "column": column,
        })
        return self

    def expect_column_values_to_not_be_null(self, column: str) -> "QualityEngine":
        """Expect column values to not be null."""
        self._expectations.append({
            "type": "expect_column_values_to_not_be_null",
            "column": column,
        })
        return self

    def expect_column_values_to_be_unique(self, column: str) -> "QualityEngine":
        """Expect column values to be unique."""
        self._expectations.append({
            "type": "expect_column_values_to_be_unique",
            "column": column,
        })
        return self

    def expect_column_values_to_be_in_set(
        self, column: str, value_set: List[Any]
    ) -> "QualityEngine":
        """Expect column values to be in a given set."""
        self._expectations.append({
            "type": "expect_column_values_to_be_in_set",
            "column": column,
            "value_set": value_set,
        })
        return self

    def expect_column_values_to_be_between(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "QualityEngine":
        """Expect column values to be between min and max."""
        self._expectations.append({
            "type": "expect_column_values_to_be_between",
            "column": column,
            "min_value": min_value,
            "max_value": max_value,
        })
        return self

    def expect_table_row_count_to_be_between(
        self, min_value: int, max_value: int
    ) -> "QualityEngine":
        """Expect table row count to be between min and max."""
        self._expectations.append({
            "type": "expect_table_row_count_to_be_between",
            "min_value": min_value,
            "max_value": max_value,
        })
        return self

    def validate(self, df: "DataFrame") -> ValidationResult:
        """
        Validate DataFrame against all expectations.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with all expectation results
        """
        _require_pyspark()
        results = []
        row_count = df.count()

        for expectation in self._expectations:
            exp_type = expectation["type"]

            if exp_type == "expect_column_to_exist":
                result = self._check_column_exists(df, expectation["column"])
            elif exp_type == "expect_column_values_to_not_be_null":
                result = self._check_not_null(df, expectation["column"])
            elif exp_type == "expect_column_values_to_be_unique":
                result = self._check_unique(df, expectation["column"])
            elif exp_type == "expect_column_values_to_be_in_set":
                result = self._check_in_set(
                    df, expectation["column"], expectation["value_set"]
                )
            elif exp_type == "expect_column_values_to_be_between":
                result = self._check_between(
                    df,
                    expectation["column"],
                    expectation.get("min_value"),
                    expectation.get("max_value"),
                )
            elif exp_type == "expect_table_row_count_to_be_between":
                result = self._check_row_count(
                    row_count, expectation["min_value"], expectation["max_value"]
                )
            else:
                result = ExpectationResult(
                    expectation_type=exp_type,
                    column=expectation.get("column"),
                    success=False,
                    result={"error": f"Unknown expectation type: {exp_type}"},
                )

            results.append(result)

        all_success = all(r.success for r in results)

        return ValidationResult(
            success=all_success,
            results=results,
            statistics={
                "row_count": row_count,
                "total_expectations": len(results),
                "successful_expectations": sum(1 for r in results if r.success),
            },
        )

    def _check_column_exists(self, df: DataFrame, column: str) -> ExpectationResult:
        """Check if column exists in DataFrame."""
        exists = column in df.columns
        return ExpectationResult(
            expectation_type="expect_column_to_exist",
            column=column,
            success=exists,
            result={"column_exists": exists},
        )

    def _check_not_null(self, df: DataFrame, column: str) -> ExpectationResult:
        """Check if column has no null values."""
        if column not in df.columns:
            return ExpectationResult(
                expectation_type="expect_column_values_to_not_be_null",
                column=column,
                success=False,
                result={"error": f"Column {column} not found"},
            )

        null_count = df.filter(col(column).isNull()).count()
        total_count = df.count()

        return ExpectationResult(
            expectation_type="expect_column_values_to_not_be_null",
            column=column,
            success=null_count == 0,
            result={
                "null_count": null_count,
                "total_count": total_count,
                "null_percent": (null_count / total_count * 100) if total_count > 0 else 0,
            },
        )

    def _check_unique(self, df: DataFrame, column: str) -> ExpectationResult:
        """Check if column values are unique."""
        if column not in df.columns:
            return ExpectationResult(
                expectation_type="expect_column_values_to_be_unique",
                column=column,
                success=False,
                result={"error": f"Column {column} not found"},
            )

        total_count = df.count()
        distinct_count = df.select(column).distinct().count()

        return ExpectationResult(
            expectation_type="expect_column_values_to_be_unique",
            column=column,
            success=total_count == distinct_count,
            result={
                "total_count": total_count,
                "distinct_count": distinct_count,
                "duplicate_count": total_count - distinct_count,
            },
        )

    def _check_in_set(
        self, df: DataFrame, column: str, value_set: List[Any]
    ) -> ExpectationResult:
        """Check if column values are in a given set."""
        if column not in df.columns:
            return ExpectationResult(
                expectation_type="expect_column_values_to_be_in_set",
                column=column,
                success=False,
                result={"error": f"Column {column} not found"},
            )

        outside_count = df.filter(~col(column).isin(value_set)).count()
        total_count = df.count()

        return ExpectationResult(
            expectation_type="expect_column_values_to_be_in_set",
            column=column,
            success=outside_count == 0,
            result={
                "outside_count": outside_count,
                "total_count": total_count,
                "value_set": value_set,
            },
        )

    def _check_between(
        self,
        df: DataFrame,
        column: str,
        min_value: Optional[float],
        max_value: Optional[float],
    ) -> ExpectationResult:
        """Check if column values are between min and max."""
        if column not in df.columns:
            return ExpectationResult(
                expectation_type="expect_column_values_to_be_between",
                column=column,
                success=False,
                result={"error": f"Column {column} not found"},
            )

        conditions = []
        if min_value is not None:
            conditions.append(col(column) < min_value)
        if max_value is not None:
            conditions.append(col(column) > max_value)

        if not conditions:
            return ExpectationResult(
                expectation_type="expect_column_values_to_be_between",
                column=column,
                success=True,
                result={"note": "No bounds specified"},
            )

        filter_condition = conditions[0]
        for cond in conditions[1:]:
            filter_condition = filter_condition | cond

        outside_count = df.filter(filter_condition).count()
        total_count = df.count()

        return ExpectationResult(
            expectation_type="expect_column_values_to_be_between",
            column=column,
            success=outside_count == 0,
            result={
                "outside_count": outside_count,
                "total_count": total_count,
                "min_value": min_value,
                "max_value": max_value,
            },
        )

    def _check_row_count(
        self, row_count: int, min_value: int, max_value: int
    ) -> ExpectationResult:
        """Check if row count is between min and max."""
        success = min_value <= row_count <= max_value

        return ExpectationResult(
            expectation_type="expect_table_row_count_to_be_between",
            column=None,
            success=success,
            result={
                "row_count": row_count,
                "min_value": min_value,
                "max_value": max_value,
            },
        )

    def clear_expectations(self) -> None:
        """Clear all expectations."""
        self._expectations = []
