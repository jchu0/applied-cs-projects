"""Statistical validation for features."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class StatisticalCheckType(Enum):
    """Types of statistical checks."""

    NULL_RATIO = "null_ratio"
    OUTLIERS = "outliers"
    DISTRIBUTION = "distribution"
    CARDINALITY = "cardinality"
    CORRELATION = "correlation"
    VARIANCE = "variance"


@dataclass
class StatisticalCheck:
    """Result of a single statistical check."""

    check_type: StatisticalCheckType
    column: str
    passed: bool
    message: str
    value: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalValidationResult:
    """Result of statistical validation."""

    is_valid: bool
    checks: List[StatisticalCheck] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def failed_checks(self) -> List[StatisticalCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def passed_checks(self) -> List[StatisticalCheck]:
        return [c for c in self.checks if c.passed]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "total_checks": len(self.checks),
            "passed_checks": len(self.passed_checks),
            "failed_checks": len(self.failed_checks),
            "validated_at": self.validated_at.isoformat(),
            "failed_details": [
                {
                    "check_type": c.check_type.value,
                    "column": c.column,
                    "message": c.message,
                    "value": c.value,
                    "threshold": c.threshold,
                }
                for c in self.failed_checks
            ],
        }


class StatisticalValidator:
    """
    Validator for statistical properties of features.

    Validates:
    - Null ratios
    - Outlier detection
    - Distribution checks
    - Cardinality checks
    - Variance checks
    """

    def __init__(
        self,
        null_threshold: float = 0.1,
        outlier_std_threshold: float = 3.0,
        min_variance: float = 0.0,
        max_cardinality_ratio: float = 0.95,
    ):
        self.null_threshold = null_threshold
        self.outlier_std_threshold = outlier_std_threshold
        self.min_variance = min_variance
        self.max_cardinality_ratio = max_cardinality_ratio
        self._reference_stats: Optional[Dict[str, Dict[str, float]]] = None

    def set_reference_statistics(
        self,
        data: np.ndarray,
        columns: Optional[List[str]] = None,
    ) -> None:
        """Set reference statistics for comparison."""
        self._reference_stats = self.compute_statistics(data, columns)

    def compute_statistics(
        self,
        data: np.ndarray,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistics for all columns."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_cols = data.shape[1]
        columns = columns or [f"col_{i}" for i in range(n_cols)]

        stats = {}
        for i, col in enumerate(columns):
            values = data[:, i]
            col_stats = self._compute_column_stats(values)
            stats[col] = col_stats

        return stats

    def _compute_column_stats(self, values: np.ndarray) -> Dict[str, float]:
        """Compute statistics for a single column."""
        if np.issubdtype(values.dtype, np.number):
            null_mask = np.isnan(values)
            valid_values = values[~null_mask]
        else:
            null_mask = np.array([v is None or str(v) in ("", "None", "nan") for v in values])
            valid_values = values[~null_mask]

        stats = {
            "count": len(values),
            "null_count": int(np.sum(null_mask)),
            "null_ratio": float(np.mean(null_mask)),
        }

        if np.issubdtype(values.dtype, np.number) and len(valid_values) > 0:
            stats.update({
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values)),
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "median": float(np.median(valid_values)),
                "variance": float(np.var(valid_values)),
                "q25": float(np.percentile(valid_values, 25)),
                "q75": float(np.percentile(valid_values, 75)),
            })
        else:
            stats.update({
                "unique_count": len(np.unique(valid_values)),
                "cardinality_ratio": len(np.unique(valid_values)) / len(values) if len(values) > 0 else 0,
            })

        return stats

    def validate(
        self,
        data: np.ndarray,
        columns: Optional[List[str]] = None,
    ) -> StatisticalValidationResult:
        """
        Validate statistical properties of the data.

        Parameters:
            data: Input data array
            columns: Column names

        Returns:
            StatisticalValidationResult with check details
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_cols = data.shape[1]
        columns = columns or [f"col_{i}" for i in range(n_cols)]

        checks = []
        statistics = {}

        for i, col in enumerate(columns):
            values = data[:, i]
            col_stats = self._compute_column_stats(values)
            statistics[col] = col_stats

            # Null ratio check
            null_check = self._check_null_ratio(col, col_stats)
            checks.append(null_check)

            # Numeric-specific checks
            if np.issubdtype(values.dtype, np.number):
                # Outlier check
                outlier_check = self._check_outliers(col, values, col_stats)
                checks.append(outlier_check)

                # Variance check
                variance_check = self._check_variance(col, col_stats)
                checks.append(variance_check)
            else:
                # Cardinality check
                cardinality_check = self._check_cardinality(col, col_stats, len(values))
                checks.append(cardinality_check)

        is_valid = all(c.passed for c in checks)

        return StatisticalValidationResult(
            is_valid=is_valid,
            checks=checks,
            statistics=statistics,
        )

    def _check_null_ratio(
        self,
        column: str,
        stats: Dict[str, float],
    ) -> StatisticalCheck:
        """Check null ratio against threshold."""
        null_ratio = stats.get("null_ratio", 0.0)
        passed = null_ratio <= self.null_threshold

        return StatisticalCheck(
            check_type=StatisticalCheckType.NULL_RATIO,
            column=column,
            passed=passed,
            message=f"Null ratio {null_ratio:.2%} {'<=' if passed else '>'} threshold {self.null_threshold:.2%}",
            value=null_ratio,
            threshold=self.null_threshold,
        )

    def _check_outliers(
        self,
        column: str,
        values: np.ndarray,
        stats: Dict[str, float],
    ) -> StatisticalCheck:
        """Check for outliers using z-score method."""
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)

        if std == 0:
            return StatisticalCheck(
                check_type=StatisticalCheckType.OUTLIERS,
                column=column,
                passed=True,
                message="No outliers (zero variance)",
                value=0.0,
                threshold=self.outlier_std_threshold,
            )

        # Filter out NaN values
        valid_values = values[~np.isnan(values)]
        z_scores = np.abs((valid_values - mean) / std)
        outlier_ratio = np.mean(z_scores > self.outlier_std_threshold)

        # Allow up to 1% outliers
        passed = outlier_ratio <= 0.01

        return StatisticalCheck(
            check_type=StatisticalCheckType.OUTLIERS,
            column=column,
            passed=passed,
            message=f"Outlier ratio {outlier_ratio:.2%} (>{self.outlier_std_threshold} std)",
            value=outlier_ratio,
            threshold=0.01,
            details={"outlier_count": int(np.sum(z_scores > self.outlier_std_threshold))},
        )

    def _check_variance(
        self,
        column: str,
        stats: Dict[str, float],
    ) -> StatisticalCheck:
        """Check that variance is above minimum."""
        variance = stats.get("variance", 0.0)
        passed = variance > self.min_variance

        return StatisticalCheck(
            check_type=StatisticalCheckType.VARIANCE,
            column=column,
            passed=passed,
            message=f"Variance {variance:.4f} {'>' if passed else '<='} threshold {self.min_variance}",
            value=variance,
            threshold=self.min_variance,
        )

    def _check_cardinality(
        self,
        column: str,
        stats: Dict[str, float],
        n_rows: int,
    ) -> StatisticalCheck:
        """Check cardinality ratio for categorical columns."""
        cardinality_ratio = stats.get("cardinality_ratio", 0.0)
        passed = cardinality_ratio <= self.max_cardinality_ratio

        return StatisticalCheck(
            check_type=StatisticalCheckType.CARDINALITY,
            column=column,
            passed=passed,
            message=f"Cardinality ratio {cardinality_ratio:.2%} {'<=' if passed else '>'} threshold {self.max_cardinality_ratio:.2%}",
            value=cardinality_ratio,
            threshold=self.max_cardinality_ratio,
            details={"unique_count": stats.get("unique_count", 0)},
        )

    def validate_distribution(
        self,
        data: np.ndarray,
        columns: Optional[List[str]] = None,
        expected_distribution: str = "normal",
    ) -> List[StatisticalCheck]:
        """
        Validate distribution of numeric columns.

        Parameters:
            data: Input data array
            columns: Column names
            expected_distribution: Expected distribution type

        Returns:
            List of distribution checks
        """
        from scipy import stats as scipy_stats

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_cols = data.shape[1]
        columns = columns or [f"col_{i}" for i in range(n_cols)]

        checks = []

        for i, col in enumerate(columns):
            values = data[:, i]

            if not np.issubdtype(values.dtype, np.number):
                continue

            valid_values = values[~np.isnan(values)]

            if len(valid_values) < 20:
                continue

            if expected_distribution == "normal":
                # Shapiro-Wilk test
                _, p_value = scipy_stats.shapiro(valid_values[:5000])  # Limit sample size
                passed = p_value > 0.05

                checks.append(StatisticalCheck(
                    check_type=StatisticalCheckType.DISTRIBUTION,
                    column=col,
                    passed=passed,
                    message=f"Shapiro-Wilk p-value: {p_value:.4f} ({'normal' if passed else 'non-normal'})",
                    value=p_value,
                    threshold=0.05,
                ))

        return checks

    def compare_with_reference(
        self,
        data: np.ndarray,
        columns: Optional[List[str]] = None,
        tolerance: float = 0.1,
    ) -> List[StatisticalCheck]:
        """
        Compare current statistics with reference statistics.

        Parameters:
            data: Input data array
            columns: Column names
            tolerance: Allowed relative difference

        Returns:
            List of comparison checks
        """
        if self._reference_stats is None:
            raise ValueError("No reference statistics set")

        current_stats = self.compute_statistics(data, columns)
        checks = []

        for col, ref_stats in self._reference_stats.items():
            if col not in current_stats:
                continue

            curr_stats = current_stats[col]

            # Compare mean
            if "mean" in ref_stats and "mean" in curr_stats:
                ref_mean = ref_stats["mean"]
                curr_mean = curr_stats["mean"]

                if ref_mean != 0:
                    rel_diff = abs(curr_mean - ref_mean) / abs(ref_mean)
                else:
                    rel_diff = abs(curr_mean)

                passed = rel_diff <= tolerance

                checks.append(StatisticalCheck(
                    check_type=StatisticalCheckType.DISTRIBUTION,
                    column=col,
                    passed=passed,
                    message=f"Mean drift: {rel_diff:.2%} ({'within' if passed else 'exceeds'} tolerance)",
                    value=rel_diff,
                    threshold=tolerance,
                    details={"reference_mean": ref_mean, "current_mean": curr_mean},
                ))

            # Compare null ratio
            ref_null = ref_stats.get("null_ratio", 0.0)
            curr_null = curr_stats.get("null_ratio", 0.0)
            null_diff = abs(curr_null - ref_null)
            passed = null_diff <= tolerance

            checks.append(StatisticalCheck(
                check_type=StatisticalCheckType.NULL_RATIO,
                column=col,
                passed=passed,
                message=f"Null ratio change: {null_diff:.2%}",
                value=null_diff,
                threshold=tolerance,
            ))

        return checks
