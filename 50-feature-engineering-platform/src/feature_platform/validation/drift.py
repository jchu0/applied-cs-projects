"""Drift detection for features."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class DriftType(Enum):
    """Types of drift."""

    FEATURE_DRIFT = "feature_drift"
    LABEL_DRIFT = "label_drift"
    CONCEPT_DRIFT = "concept_drift"
    PRIOR_DRIFT = "prior_drift"


class DriftMethod(Enum):
    """Methods for drift detection."""

    PSI = "psi"  # Population Stability Index
    KS = "ks"  # Kolmogorov-Smirnov
    CHI2 = "chi2"  # Chi-squared
    JS = "js"  # Jensen-Shannon divergence
    WASSERSTEIN = "wasserstein"  # Wasserstein/Earth Mover's distance


@dataclass
class DriftResult:
    """Result of drift detection."""

    column: str
    drift_type: DriftType
    method: DriftMethod
    score: float
    threshold: float
    is_drifted: bool
    p_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "drift_type": self.drift_type.value,
            "method": self.method.value,
            "score": self.score,
            "threshold": self.threshold,
            "is_drifted": self.is_drifted,
            "p_value": self.p_value,
        }


@dataclass
class DriftReport:
    """Complete drift detection report."""

    is_drifted: bool
    results: List[DriftResult]
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    reference_stats: Dict[str, Any] = field(default_factory=dict)
    current_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def drifted_features(self) -> List[str]:
        return [r.column for r in self.results if r.is_drifted]

    @property
    def drift_count(self) -> int:
        return len(self.drifted_features)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_drifted": self.is_drifted,
            "drift_count": self.drift_count,
            "drifted_features": self.drifted_features,
            "analyzed_at": self.analyzed_at.isoformat(),
            "results": [r.to_dict() for r in self.results],
        }


class DriftDetector:
    """
    Detector for data drift in features.

    Supports multiple drift detection methods:
    - PSI (Population Stability Index)
    - Kolmogorov-Smirnov test
    - Chi-squared test
    - Jensen-Shannon divergence
    - Wasserstein distance
    """

    def __init__(
        self,
        method: DriftMethod = DriftMethod.PSI,
        threshold: float = 0.1,
        n_bins: int = 10,
    ):
        self.method = method
        self.threshold = threshold
        self.n_bins = n_bins
        self._reference_data: Optional[np.ndarray] = None
        self._reference_columns: Optional[List[str]] = None
        self._reference_histograms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def set_reference(
        self,
        data: np.ndarray,
        columns: Optional[List[str]] = None,
    ) -> None:
        """
        Set reference data for drift comparison.

        Parameters:
            data: Reference data array
            columns: Column names
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._reference_data = data
        self._reference_columns = columns or [f"col_{i}" for i in range(data.shape[1])]

        # Pre-compute histograms for reference data
        self._reference_histograms = {}
        for i, col in enumerate(self._reference_columns):
            values = data[:, i]
            if np.issubdtype(values.dtype, np.number):
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    hist, edges = np.histogram(valid_values, bins=self.n_bins)
                    self._reference_histograms[col] = (hist / len(valid_values), edges)

    def detect(
        self,
        current_data: np.ndarray,
        columns: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Detect drift between reference and current data.

        Parameters:
            current_data: Current data array
            columns: Column names

        Returns:
            DriftReport with drift detection results
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set")

        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

        columns = columns or [f"col_{i}" for i in range(current_data.shape[1])]

        results = []
        for i, col in enumerate(columns):
            if i >= self._reference_data.shape[1]:
                continue

            ref_values = self._reference_data[:, i]
            curr_values = current_data[:, i]

            result = self._detect_column_drift(col, ref_values, curr_values)
            results.append(result)

        is_drifted = any(r.is_drifted for r in results)

        return DriftReport(
            is_drifted=is_drifted,
            results=results,
        )

    def _detect_column_drift(
        self,
        column: str,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> DriftResult:
        """Detect drift for a single column."""
        # Handle numeric vs categorical
        if np.issubdtype(reference.dtype, np.number):
            return self._detect_numeric_drift(column, reference, current)
        else:
            return self._detect_categorical_drift(column, reference, current)

    def _detect_numeric_drift(
        self,
        column: str,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> DriftResult:
        """Detect drift for numeric column."""
        # Filter out NaN values
        ref_valid = reference[~np.isnan(reference)]
        curr_valid = current[~np.isnan(current)]

        if len(ref_valid) == 0 or len(curr_valid) == 0:
            return DriftResult(
                column=column,
                drift_type=DriftType.FEATURE_DRIFT,
                method=self.method,
                score=0.0,
                threshold=self.threshold,
                is_drifted=False,
            )

        if self.method == DriftMethod.PSI:
            score = self._compute_psi(ref_valid, curr_valid)
            is_drifted = score > self.threshold
            return DriftResult(
                column=column,
                drift_type=DriftType.FEATURE_DRIFT,
                method=DriftMethod.PSI,
                score=score,
                threshold=self.threshold,
                is_drifted=is_drifted,
            )

        elif self.method == DriftMethod.KS:
            from scipy import stats
            statistic, p_value = stats.ks_2samp(ref_valid, curr_valid)
            is_drifted = p_value < 0.05
            return DriftResult(
                column=column,
                drift_type=DriftType.FEATURE_DRIFT,
                method=DriftMethod.KS,
                score=statistic,
                threshold=0.05,
                is_drifted=is_drifted,
                p_value=p_value,
            )

        elif self.method == DriftMethod.JS:
            score = self._compute_js_divergence(ref_valid, curr_valid)
            is_drifted = score > self.threshold
            return DriftResult(
                column=column,
                drift_type=DriftType.FEATURE_DRIFT,
                method=DriftMethod.JS,
                score=score,
                threshold=self.threshold,
                is_drifted=is_drifted,
            )

        elif self.method == DriftMethod.WASSERSTEIN:
            from scipy import stats
            score = stats.wasserstein_distance(ref_valid, curr_valid)
            # Normalize by range
            data_range = max(ref_valid.max(), curr_valid.max()) - min(ref_valid.min(), curr_valid.min())
            normalized_score = score / data_range if data_range > 0 else 0
            is_drifted = normalized_score > self.threshold
            return DriftResult(
                column=column,
                drift_type=DriftType.FEATURE_DRIFT,
                method=DriftMethod.WASSERSTEIN,
                score=normalized_score,
                threshold=self.threshold,
                is_drifted=is_drifted,
            )

        else:
            raise ValueError(f"Unsupported drift method: {self.method}")

    def _detect_categorical_drift(
        self,
        column: str,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> DriftResult:
        """Detect drift for categorical column."""
        # Get value counts
        ref_counts = {}
        for val in reference:
            val_str = str(val)
            ref_counts[val_str] = ref_counts.get(val_str, 0) + 1

        curr_counts = {}
        for val in current:
            val_str = str(val)
            curr_counts[val_str] = curr_counts.get(val_str, 0) + 1

        # All categories
        all_categories = set(ref_counts.keys()) | set(curr_counts.keys())

        if not all_categories:
            return DriftResult(
                column=column,
                drift_type=DriftType.FEATURE_DRIFT,
                method=DriftMethod.CHI2,
                score=0.0,
                threshold=self.threshold,
                is_drifted=False,
            )

        # Compute proportions
        n_ref = len(reference)
        n_curr = len(current)

        ref_props = [ref_counts.get(cat, 0) / n_ref for cat in all_categories]
        curr_props = [curr_counts.get(cat, 0) / n_curr for cat in all_categories]

        # Chi-squared test
        from scipy import stats

        # Add small value to avoid division by zero
        expected = np.array(ref_props) * n_curr + 1e-10
        observed = np.array(curr_props) * n_curr

        if len(expected) > 1:
            chi2, p_value = stats.chisquare(observed, expected)
            is_drifted = p_value < 0.05
        else:
            chi2 = 0.0
            p_value = 1.0
            is_drifted = False

        return DriftResult(
            column=column,
            drift_type=DriftType.FEATURE_DRIFT,
            method=DriftMethod.CHI2,
            score=chi2,
            threshold=0.05,
            is_drifted=is_drifted,
            p_value=p_value,
            details={"categories": list(all_categories)[:10]},
        )

    def _compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """Compute Population Stability Index."""
        # Create bins based on reference data
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())

        edges = np.linspace(min_val, max_val, self.n_bins + 1)

        ref_hist, _ = np.histogram(reference, bins=edges)
        curr_hist, _ = np.histogram(current, bins=edges)

        # Convert to proportions
        ref_props = ref_hist / len(reference)
        curr_props = curr_hist / len(current)

        # Add small value to avoid log(0) and division by zero
        epsilon = 1e-10
        ref_props = np.clip(ref_props, epsilon, 1 - epsilon)
        curr_props = np.clip(curr_props, epsilon, 1 - epsilon)

        # Compute PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))

        return psi

    def _compute_js_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """Compute Jensen-Shannon divergence."""
        # Create bins
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())

        edges = np.linspace(min_val, max_val, self.n_bins + 1)

        ref_hist, _ = np.histogram(reference, bins=edges)
        curr_hist, _ = np.histogram(current, bins=edges)

        # Convert to proportions
        ref_props = ref_hist / len(reference)
        curr_props = curr_hist / len(current)

        # Add small value to avoid log(0)
        epsilon = 1e-10
        ref_props = ref_props + epsilon
        curr_props = curr_props + epsilon

        # Normalize
        ref_props = ref_props / ref_props.sum()
        curr_props = curr_props / curr_props.sum()

        # Middle distribution
        m = 0.5 * (ref_props + curr_props)

        # KL divergences
        kl_ref = np.sum(ref_props * np.log(ref_props / m))
        kl_curr = np.sum(curr_props * np.log(curr_props / m))

        # JS divergence
        js = 0.5 * (kl_ref + kl_curr)

        return js

    def detect_label_drift(
        self,
        reference_labels: np.ndarray,
        current_labels: np.ndarray,
    ) -> DriftResult:
        """
        Detect drift in labels/targets.

        Parameters:
            reference_labels: Reference label array
            current_labels: Current label array

        Returns:
            DriftResult for label drift
        """
        # Use the same detection logic but mark as label drift
        result = self._detect_column_drift("label", reference_labels, current_labels)
        result.drift_type = DriftType.LABEL_DRIFT
        return result

    def get_drift_summary(self, report: DriftReport) -> Dict[str, Any]:
        """Get a summary of drift detection results."""
        return {
            "total_features": len(report.results),
            "drifted_features": report.drift_count,
            "drift_percentage": report.drift_count / len(report.results) * 100 if report.results else 0,
            "max_drift_score": max(r.score for r in report.results) if report.results else 0,
            "drifted_columns": report.drifted_features,
        }
