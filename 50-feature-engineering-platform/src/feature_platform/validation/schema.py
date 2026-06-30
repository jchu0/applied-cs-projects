"""Schema validation for features."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import numpy as np

from feature_platform.core.models import DataType, Feature, FeatureSchema


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    field: str
    message: str
    severity: ValidationSeverity
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    row_indices: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "expected": str(self.expected) if self.expected else None,
            "actual": str(self.actual) if self.actual else None,
            "affected_rows": len(self.row_indices) if self.row_indices else 0,
        }


@dataclass
class SchemaValidationResult:
    """Result of schema validation."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    row_count: int = 0
    column_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "validated_at": self.validated_at.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "issues": [i.to_dict() for i in self.issues],
        }


class SchemaValidator:
    """
    Validator for feature schema compliance.

    Validates:
    - Column presence
    - Data types
    - Null values
    - Value constraints
    """

    def __init__(
        self,
        schema: Optional[FeatureSchema] = None,
        strict: bool = False,
    ):
        self.schema = schema
        self.strict = strict
        self._column_validators: Dict[str, List[callable]] = {}

    def set_schema(self, schema: FeatureSchema) -> None:
        """Set the schema to validate against."""
        self.schema = schema

    def add_column_validator(
        self,
        column: str,
        validator: callable,
    ) -> None:
        """Add a custom validator for a column."""
        if column not in self._column_validators:
            self._column_validators[column] = []
        self._column_validators[column].append(validator)

    def validate(
        self,
        data: np.ndarray,
        columns: Optional[List[str]] = None,
    ) -> SchemaValidationResult:
        """
        Validate data against the schema.

        Parameters:
            data: Input data array
            columns: Column names

        Returns:
            SchemaValidationResult with validation details
        """
        issues = []

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_rows, n_cols = data.shape
        columns = columns or [f"col_{i}" for i in range(n_cols)]

        # Validate column count
        if self.schema:
            expected_cols = set(self.schema.get_feature_names())
            actual_cols = set(columns)

            # Check for missing columns
            missing = expected_cols - actual_cols
            for col in missing:
                feature = self.schema.get_feature(col)
                if feature and not feature.nullable:
                    issues.append(ValidationIssue(
                        field=col,
                        message=f"Required column missing: {col}",
                        severity=ValidationSeverity.ERROR,
                    ))
                else:
                    issues.append(ValidationIssue(
                        field=col,
                        message=f"Optional column missing: {col}",
                        severity=ValidationSeverity.WARNING,
                    ))

            # Check for extra columns in strict mode
            if self.strict:
                extra = actual_cols - expected_cols
                for col in extra:
                    issues.append(ValidationIssue(
                        field=col,
                        message=f"Unexpected column: {col}",
                        severity=ValidationSeverity.WARNING,
                    ))

            # Validate each column
            for i, col in enumerate(columns):
                if col not in expected_cols:
                    continue

                feature = self.schema.get_feature(col)
                if feature:
                    col_issues = self._validate_column(
                        data[:, i], col, feature
                    )
                    issues.extend(col_issues)

        # Run custom validators
        for col, validators in self._column_validators.items():
            if col in columns:
                col_idx = columns.index(col)
                for validator in validators:
                    try:
                        result = validator(data[:, col_idx])
                        if result is not True:
                            issues.append(ValidationIssue(
                                field=col,
                                message=str(result) if result else "Custom validation failed",
                                severity=ValidationSeverity.ERROR,
                            ))
                    except Exception as e:
                        issues.append(ValidationIssue(
                            field=col,
                            message=f"Validator error: {str(e)}",
                            severity=ValidationSeverity.ERROR,
                        ))

        is_valid = not any(i.severity == ValidationSeverity.ERROR for i in issues)

        return SchemaValidationResult(
            is_valid=is_valid,
            issues=issues,
            row_count=n_rows,
            column_count=n_cols,
        )

    def _validate_column(
        self,
        values: np.ndarray,
        column: str,
        feature: Feature,
    ) -> List[ValidationIssue]:
        """Validate a single column."""
        issues = []

        # Check for nulls
        if np.issubdtype(values.dtype, np.number):
            null_mask = np.isnan(values)
        else:
            null_mask = np.array([v is None or str(v) in ("", "None", "nan") for v in values])

        null_count = np.sum(null_mask)

        if not feature.nullable and null_count > 0:
            issues.append(ValidationIssue(
                field=column,
                message=f"Column {column} contains {null_count} null values but is not nullable",
                severity=ValidationSeverity.ERROR,
                expected="No nulls",
                actual=f"{null_count} nulls",
                row_indices=list(np.where(null_mask)[0][:10]),  # First 10 indices
            ))

        # Check data type
        type_issues = self._validate_dtype(values[~null_mask], column, feature.dtype)
        issues.extend(type_issues)

        return issues

    def _validate_dtype(
        self,
        values: np.ndarray,
        column: str,
        expected_dtype: DataType,
    ) -> List[ValidationIssue]:
        """Validate data type compatibility."""
        issues = []

        if len(values) == 0:
            return issues

        actual_dtype = values.dtype

        # Check type compatibility
        type_compatible = self._check_type_compatibility(actual_dtype, expected_dtype)

        if not type_compatible:
            issues.append(ValidationIssue(
                field=column,
                message=f"Type mismatch for column {column}",
                severity=ValidationSeverity.ERROR,
                expected=expected_dtype.value,
                actual=str(actual_dtype),
            ))

        return issues

    def _check_type_compatibility(
        self,
        actual: np.dtype,
        expected: DataType,
    ) -> bool:
        """Check if actual numpy dtype is compatible with expected DataType."""
        compatibility_map = {
            DataType.INT32: [np.int32, np.int64, np.int16, np.int8],
            DataType.INT64: [np.int64, np.int32],
            DataType.FLOAT32: [np.float32, np.float64, np.int32, np.int64],
            DataType.FLOAT64: [np.float64, np.float32, np.int32, np.int64],
            DataType.STRING: [np.object_, np.str_],
            DataType.BOOL: [np.bool_],
        }

        compatible_types = compatibility_map.get(expected, [])

        for compat_type in compatible_types:
            if np.issubdtype(actual, compat_type):
                return True

        return False

    def validate_value_ranges(
        self,
        data: np.ndarray,
        columns: List[str],
        ranges: Dict[str, tuple],
    ) -> List[ValidationIssue]:
        """
        Validate that values fall within specified ranges.

        Parameters:
            data: Input data array
            columns: Column names
            ranges: Dict mapping column name to (min, max) tuple

        Returns:
            List of validation issues
        """
        issues = []

        for col, (min_val, max_val) in ranges.items():
            if col not in columns:
                continue

            col_idx = columns.index(col)
            values = data[:, col_idx]

            if np.issubdtype(values.dtype, np.number):
                below_min = values < min_val
                above_max = values > max_val

                if np.any(below_min):
                    issues.append(ValidationIssue(
                        field=col,
                        message=f"Values below minimum {min_val}",
                        severity=ValidationSeverity.WARNING,
                        expected=f">= {min_val}",
                        actual=f"Min: {np.min(values)}",
                        row_indices=list(np.where(below_min)[0][:10]),
                    ))

                if np.any(above_max):
                    issues.append(ValidationIssue(
                        field=col,
                        message=f"Values above maximum {max_val}",
                        severity=ValidationSeverity.WARNING,
                        expected=f"<= {max_val}",
                        actual=f"Max: {np.max(values)}",
                        row_indices=list(np.where(above_max)[0][:10]),
                    ))

        return issues

    def validate_unique(
        self,
        data: np.ndarray,
        columns: List[str],
        unique_columns: List[str],
    ) -> List[ValidationIssue]:
        """
        Validate that specified columns contain unique values.

        Parameters:
            data: Input data array
            columns: Column names
            unique_columns: Columns that should be unique

        Returns:
            List of validation issues
        """
        issues = []

        for col in unique_columns:
            if col not in columns:
                continue

            col_idx = columns.index(col)
            values = data[:, col_idx]

            unique_values = np.unique(values)
            if len(unique_values) < len(values):
                duplicates = len(values) - len(unique_values)
                issues.append(ValidationIssue(
                    field=col,
                    message=f"Column {col} contains {duplicates} duplicate values",
                    severity=ValidationSeverity.ERROR,
                    expected="All unique values",
                    actual=f"{duplicates} duplicates",
                ))

        return issues

    def validate_allowed_values(
        self,
        data: np.ndarray,
        columns: List[str],
        allowed: Dict[str, Set[Any]],
    ) -> List[ValidationIssue]:
        """
        Validate that values are in an allowed set.

        Parameters:
            data: Input data array
            columns: Column names
            allowed: Dict mapping column name to set of allowed values

        Returns:
            List of validation issues
        """
        issues = []

        for col, allowed_values in allowed.items():
            if col not in columns:
                continue

            col_idx = columns.index(col)
            values = data[:, col_idx]

            invalid_mask = ~np.isin(values, list(allowed_values))
            if np.any(invalid_mask):
                invalid_values = set(values[invalid_mask])
                issues.append(ValidationIssue(
                    field=col,
                    message=f"Invalid values in column {col}",
                    severity=ValidationSeverity.ERROR,
                    expected=f"One of: {list(allowed_values)[:5]}...",
                    actual=f"Found: {list(invalid_values)[:5]}...",
                    row_indices=list(np.where(invalid_mask)[0][:10]),
                ))

        return issues
