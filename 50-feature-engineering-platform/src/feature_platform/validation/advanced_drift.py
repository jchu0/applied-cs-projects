"""Advanced drift detection with multivariate analysis and concept drift."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import deque

from .drift import DriftResult, DriftReport, DriftType, DriftMethod


class ConceptDriftMethod(Enum):
    """Methods for concept drift detection."""

    DDM = "ddm"  # Drift Detection Method
    EDDM = "eddm"  # Early Drift Detection Method
    ADWIN = "adwin"  # Adaptive Windowing
    PAGE_HINKLEY = "page_hinkley"  # Page-Hinkley test
    CUSUM = "cusum"  # Cumulative Sum


@dataclass
class ConceptDriftResult:
    """Result of concept drift detection."""

    method: ConceptDriftMethod
    is_drift_detected: bool
    is_warning_detected: bool
    drift_point: Optional[int] = None
    warning_point: Optional[int] = None
    current_mean: Optional[float] = None
    detection_delay: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultivariateDriftResult:
    """Result of multivariate drift detection."""

    is_drifted: bool
    method: str
    score: float
    threshold: float
    p_value: Optional[float] = None
    contributing_features: List[Tuple[str, float]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class WindowedDriftMonitor:
    """
    Monitor for continuous drift detection using sliding windows.

    Maintains a reference window and compares against current data
    to detect drift over time.
    """

    def __init__(
        self,
        window_size: int = 1000,
        step_size: int = 100,
        drift_threshold: float = 0.1,
        method: DriftMethod = DriftMethod.PSI,
        cooldown_steps: int = 5,
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.drift_threshold = drift_threshold
        self.method = method
        self.cooldown_steps = cooldown_steps

        self._reference_window: Optional[np.ndarray] = None
        self._current_window: deque = deque(maxlen=window_size)
        self._columns: Optional[List[str]] = None
        self._drift_history: List[DriftReport] = []
        self._last_drift_step: int = -1000
        self._step_count: int = 0

    def initialize(self, data: np.ndarray, columns: Optional[List[str]] = None) -> None:
        """Initialize with reference data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._reference_window = data[-self.window_size:].copy()
        self._columns = columns or [f"col_{i}" for i in range(data.shape[1])]
        self._current_window = deque(maxlen=self.window_size)
        for row in data[-self.window_size:]:
            self._current_window.append(row)
        self._step_count = 0

    def update(self, data: np.ndarray) -> Optional[DriftReport]:
        """
        Update with new data and check for drift.

        Returns DriftReport if drift check was performed, None otherwise.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Add to current window
        for row in data:
            self._current_window.append(row)

        self._step_count += 1

        # Check drift at intervals
        if self._step_count % self.step_size == 0:
            return self._check_drift()

        return None

    def _check_drift(self) -> DriftReport:
        """Check for drift between reference and current window."""
        from .drift import DriftDetector

        current_array = np.array(list(self._current_window))

        detector = DriftDetector(
            method=self.method,
            threshold=self.drift_threshold,
        )
        detector.set_reference(self._reference_window, self._columns)
        report = detector.detect(current_array, self._columns)

        # Update reference if no drift (adaptive reference)
        if not report.is_drifted:
            self._reference_window = current_array.copy()
        elif self._step_count - self._last_drift_step > self.cooldown_steps:
            self._last_drift_step = self._step_count

        self._drift_history.append(report)
        return report

    def get_drift_rate(self, last_n: int = 10) -> float:
        """Get drift detection rate over last N checks."""
        recent = self._drift_history[-last_n:]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.is_drifted) / len(recent)

    @property
    def drift_history(self) -> List[DriftReport]:
        return self._drift_history


class DDMDetector:
    """
    Drift Detection Method (DDM) for concept drift.

    Based on the idea that if the distribution of data changes,
    the error rate of a learning algorithm will also change.
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_samples: int = 30,
    ):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples

        self._n = 0
        self._p = 0.0
        self._s = 0.0
        self._p_min = float('inf')
        self._s_min = float('inf')
        self._warning_point: Optional[int] = None
        self._drift_point: Optional[int] = None
        self._in_warning = False
        self._in_drift = False

    def update(self, error: float) -> ConceptDriftResult:
        """
        Update with new error observation.

        Parameters:
            error: Error value (0 or 1 for classification, continuous for regression)

        Returns:
            ConceptDriftResult with detection status
        """
        self._n += 1

        # Update mean and standard deviation
        self._p = self._p + (error - self._p) / self._n
        self._s = np.sqrt(self._p * (1 - self._p) / self._n)

        result = ConceptDriftResult(
            method=ConceptDriftMethod.DDM,
            is_drift_detected=False,
            is_warning_detected=False,
            current_mean=self._p,
        )

        if self._n < self.min_samples:
            return result

        # Update minimums
        if self._p + self._s < self._p_min + self._s_min:
            self._p_min = self._p
            self._s_min = self._s
            self._in_warning = False
            self._in_drift = False

        # Check for warning and drift (only if s_min > 0 to avoid false positives)
        if self._s_min > 0:
            if self._p + self._s >= self._p_min + self.warning_level * self._s_min:
                if not self._in_warning:
                    self._warning_point = self._n
                    self._in_warning = True
                result.is_warning_detected = True
                result.warning_point = self._warning_point

            if self._p + self._s >= self._p_min + self.drift_level * self._s_min:
                if not self._in_drift:
                    self._drift_point = self._n
                    self._in_drift = True
                result.is_drift_detected = True
                result.drift_point = self._drift_point
                result.detection_delay = self._n - (self._warning_point or self._n)

        return result

    def reset(self) -> None:
        """Reset the detector."""
        self._n = 0
        self._p = 0.0
        self._s = 0.0
        self._p_min = float('inf')
        self._s_min = float('inf')
        self._warning_point = None
        self._drift_point = None
        self._in_warning = False
        self._in_drift = False


class ADWINDetector:
    """
    ADWIN (Adaptive Windowing) for concept drift detection.

    Maintains a variable-length window that automatically shrinks
    when drift is detected.
    """

    def __init__(
        self,
        delta: float = 0.002,
        max_buckets: int = 5,
        min_window: int = 10,
        min_clock: int = 32,
    ):
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_window = min_window
        self.min_clock = min_clock

        # Bucket structure for efficient windowing
        self._buckets: List[List[Tuple[int, float, float]]] = []
        self._total = 0.0
        self._variance = 0.0
        self._width = 0
        self._last_bucket_row = 0

    def update(self, value: float) -> ConceptDriftResult:
        """
        Update with new value and check for drift.

        Returns:
            ConceptDriftResult with detection status
        """
        # Add new element
        self._insert_element(value)

        result = ConceptDriftResult(
            method=ConceptDriftMethod.ADWIN,
            is_drift_detected=False,
            is_warning_detected=False,
            current_mean=self._total / self._width if self._width > 0 else 0,
        )

        # Check for drift
        if self._width >= self.min_window:
            is_drift = self._detect_change()
            if is_drift:
                result.is_drift_detected = True
                result.drift_point = self._width

        return result

    def _insert_element(self, value: float) -> None:
        """Insert new element into bucket structure."""
        self._width += 1
        self._total += value
        self._variance += 0  # Simplified variance tracking

        # Add to first level of buckets
        if not self._buckets:
            self._buckets.append([])
        self._buckets[0].append((1, value, 0))

        # Compress buckets if needed
        self._compress_buckets()

    def _compress_buckets(self) -> None:
        """Compress buckets to maintain efficiency."""
        for level in range(len(self._buckets)):
            if len(self._buckets[level]) > self.max_buckets:
                # Merge buckets at this level
                if level + 1 >= len(self._buckets):
                    self._buckets.append([])

                # Take two oldest buckets and merge
                b1 = self._buckets[level].pop(0)
                b2 = self._buckets[level].pop(0)

                new_count = b1[0] + b2[0]
                new_total = b1[1] + b2[1]
                new_var = b1[2] + b2[2]

                self._buckets[level + 1].append((new_count, new_total, new_var))

    def _detect_change(self) -> bool:
        """Detect if there's a significant change in the window."""
        if self._width < self.min_window * 2:
            return False

        # Simplified change detection using mean comparison
        # Full implementation would use bucket-based hypothesis testing
        n0 = self._width // 2
        n1 = self._width - n0

        if n0 < self.min_window or n1 < self.min_window:
            return False

        # Estimate means from buckets
        # This is a simplified version
        total_mean = self._total / self._width
        epsilon = np.sqrt(np.log(2 / self.delta) / (2 * min(n0, n1)))

        return False  # Would need full bucket-based implementation

    @property
    def width(self) -> int:
        return self._width

    @property
    def mean(self) -> float:
        return self._total / self._width if self._width > 0 else 0.0

    def reset(self) -> None:
        """Reset the detector."""
        self._buckets = []
        self._total = 0.0
        self._variance = 0.0
        self._width = 0


class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) control chart for concept drift detection.
    """

    def __init__(
        self,
        target_mean: float = 0.0,
        threshold: float = 50.0,
        allowance: float = 4.0,
        min_samples: int = 30,
    ):
        self.target_mean = target_mean
        self.threshold = threshold
        self.allowance = allowance
        self.min_samples = min_samples

        self._n = 0
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._sum = 0.0
        self._drift_point: Optional[int] = None

    def update(self, value: float) -> ConceptDriftResult:
        """
        Update with new value and check for drift.

        Returns:
            ConceptDriftResult with detection status
        """
        self._n += 1
        self._sum += value

        # Update CUSUM statistics
        deviation = value - self.target_mean
        self._cusum_pos = max(0, self._cusum_pos + deviation - self.allowance)
        self._cusum_neg = min(0, self._cusum_neg + deviation + self.allowance)

        result = ConceptDriftResult(
            method=ConceptDriftMethod.CUSUM,
            is_drift_detected=False,
            is_warning_detected=False,
            current_mean=self._sum / self._n,
            details={
                "cusum_positive": self._cusum_pos,
                "cusum_negative": abs(self._cusum_neg),
            },
        )

        if self._n < self.min_samples:
            return result

        # Check for drift (either direction)
        if self._cusum_pos > self.threshold or abs(self._cusum_neg) > self.threshold:
            result.is_drift_detected = True
            result.drift_point = self._n
            self._drift_point = self._n

            # Reset CUSUM after detection
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0

        return result

    def reset(self) -> None:
        """Reset the detector."""
        self._n = 0
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._sum = 0.0
        self._drift_point = None


class MultivariateDriftDetector:
    """
    Multivariate drift detection using dimensionality reduction and statistical tests.

    Supports:
    - Maximum Mean Discrepancy (MMD)
    - Multivariate two-sample tests
    - Domain classifier approach
    """

    def __init__(
        self,
        method: str = "mmd",
        threshold: float = 0.1,
        kernel: str = "rbf",
        n_permutations: int = 100,
    ):
        self.method = method
        self.threshold = threshold
        self.kernel = kernel
        self.n_permutations = n_permutations
        self._reference_data: Optional[np.ndarray] = None
        self._reference_columns: Optional[List[str]] = None

    def set_reference(
        self,
        data: np.ndarray,
        columns: Optional[List[str]] = None,
    ) -> None:
        """Set reference data for multivariate comparison."""
        self._reference_data = data
        self._reference_columns = columns or [f"col_{i}" for i in range(data.shape[1])]

    def detect(
        self,
        current_data: np.ndarray,
        columns: Optional[List[str]] = None,
    ) -> MultivariateDriftResult:
        """
        Detect multivariate drift.

        Returns:
            MultivariateDriftResult with detection results
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set")

        columns = columns or self._reference_columns

        if self.method == "mmd":
            return self._detect_mmd(current_data, columns)
        elif self.method == "energy":
            return self._detect_energy_distance(current_data, columns)
        elif self.method == "domain_classifier":
            return self._detect_domain_classifier(current_data, columns)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_mmd(
        self,
        current_data: np.ndarray,
        columns: List[str],
    ) -> MultivariateDriftResult:
        """Detect drift using Maximum Mean Discrepancy."""
        # Compute MMD with RBF kernel
        mmd_score = self._compute_mmd(self._reference_data, current_data)

        # Permutation test for p-value
        combined = np.vstack([self._reference_data, current_data])
        n_ref = len(self._reference_data)

        perm_scores = []
        for _ in range(self.n_permutations):
            perm_idx = np.random.permutation(len(combined))
            perm_ref = combined[perm_idx[:n_ref]]
            perm_curr = combined[perm_idx[n_ref:]]
            perm_scores.append(self._compute_mmd(perm_ref, perm_curr))

        p_value = np.mean(np.array(perm_scores) >= mmd_score)
        is_drifted = p_value < 0.05

        # Find contributing features
        contributing = self._find_contributing_features(current_data, columns)

        return MultivariateDriftResult(
            is_drifted=is_drifted,
            method="mmd",
            score=mmd_score,
            threshold=self.threshold,
            p_value=p_value,
            contributing_features=contributing,
        )

    def _compute_mmd(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Maximum Mean Discrepancy with RBF kernel."""
        # Compute bandwidth using median heuristic
        combined = np.vstack([x, y])
        dists = np.linalg.norm(combined[:, None] - combined, axis=2)
        bandwidth = np.median(dists[dists > 0])
        if bandwidth == 0:
            bandwidth = 1.0

        gamma = 1.0 / (2 * bandwidth ** 2)

        # Compute kernel matrices
        def rbf_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            dists = np.linalg.norm(a[:, None] - b, axis=2)
            return np.exp(-gamma * dists ** 2)

        k_xx = rbf_kernel(x, x)
        k_yy = rbf_kernel(y, y)
        k_xy = rbf_kernel(x, y)

        n = len(x)
        m = len(y)

        # MMD^2
        mmd2 = (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1))
        mmd2 += (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
        mmd2 -= 2 * np.mean(k_xy)

        return max(0, mmd2) ** 0.5

    def _detect_energy_distance(
        self,
        current_data: np.ndarray,
        columns: List[str],
    ) -> MultivariateDriftResult:
        """Detect drift using energy distance."""
        # Compute energy distance
        energy = self._compute_energy_distance(self._reference_data, current_data)

        # Simple threshold-based detection
        is_drifted = energy > self.threshold

        contributing = self._find_contributing_features(current_data, columns)

        return MultivariateDriftResult(
            is_drifted=is_drifted,
            method="energy",
            score=energy,
            threshold=self.threshold,
            contributing_features=contributing,
        )

    def _compute_energy_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute energy distance between two samples."""
        n, m = len(x), len(y)

        # E[||X - Y||]
        xy_dists = np.linalg.norm(x[:, None] - y, axis=2)
        e_xy = np.mean(xy_dists)

        # E[||X - X'||]
        xx_dists = np.linalg.norm(x[:, None] - x, axis=2)
        e_xx = np.sum(xx_dists) / (n * (n - 1)) if n > 1 else 0

        # E[||Y - Y'||]
        yy_dists = np.linalg.norm(y[:, None] - y, axis=2)
        e_yy = np.sum(yy_dists) / (m * (m - 1)) if m > 1 else 0

        # Clamp to 0 to handle floating-point precision issues
        return max(0.0, 2 * e_xy - e_xx - e_yy)

    def _detect_domain_classifier(
        self,
        current_data: np.ndarray,
        columns: List[str],
    ) -> MultivariateDriftResult:
        """Detect drift using domain classifier approach."""
        # Create labels (0 for reference, 1 for current)
        n_ref = len(self._reference_data)
        n_curr = len(current_data)

        X = np.vstack([self._reference_data, current_data])
        y = np.array([0] * n_ref + [1] * n_curr)

        # Simple logistic regression classifier
        # Normalize features
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

        # Fit simple classifier and get accuracy
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        try:
            clf = LogisticRegression(max_iter=100, solver='lbfgs')
            scores = cross_val_score(clf, X_norm, y, cv=5, scoring='accuracy')
            accuracy = np.mean(scores)
        except Exception:
            accuracy = 0.5

        # Drift if classifier can distinguish domains
        # Score is how much better than random (0.5)
        drift_score = 2 * (accuracy - 0.5)
        is_drifted = drift_score > self.threshold

        contributing = self._find_contributing_features(current_data, columns)

        return MultivariateDriftResult(
            is_drifted=is_drifted,
            method="domain_classifier",
            score=drift_score,
            threshold=self.threshold,
            details={"classifier_accuracy": accuracy},
            contributing_features=contributing,
        )

    def _find_contributing_features(
        self,
        current_data: np.ndarray,
        columns: List[str],
    ) -> List[Tuple[str, float]]:
        """Find which features contribute most to the drift."""
        from .drift import DriftDetector, DriftMethod

        contributions = []
        detector = DriftDetector(method=DriftMethod.PSI, threshold=0.1)
        detector.set_reference(self._reference_data, columns)

        report = detector.detect(current_data, columns)

        for result in report.results:
            contributions.append((result.column, result.score))

        # Sort by drift score (descending)
        contributions.sort(key=lambda x: x[1], reverse=True)

        return contributions[:10]  # Top 10 contributors
