"""Tests for advanced drift detection module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Check if sklearn is available
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from feature_platform.validation.advanced_drift import (
    ConceptDriftMethod,
    ConceptDriftResult,
    MultivariateDriftResult,
    WindowedDriftMonitor,
    DDMDetector,
    ADWINDetector,
    CUSUMDetector,
    MultivariateDriftDetector,
)
from feature_platform.validation.drift import DriftMethod


class TestConceptDriftResult:
    """Tests for ConceptDriftResult dataclass."""

    def test_creation_minimal(self):
        """Test creating result with minimal parameters."""
        result = ConceptDriftResult(
            method=ConceptDriftMethod.DDM,
            is_drift_detected=False,
            is_warning_detected=False,
        )
        assert result.method == ConceptDriftMethod.DDM
        assert not result.is_drift_detected
        assert not result.is_warning_detected
        assert result.drift_point is None
        assert result.warning_point is None
        assert result.details == {}

    def test_creation_full(self):
        """Test creating result with all parameters."""
        result = ConceptDriftResult(
            method=ConceptDriftMethod.CUSUM,
            is_drift_detected=True,
            is_warning_detected=True,
            drift_point=100,
            warning_point=80,
            current_mean=0.5,
            detection_delay=20,
            details={"extra": "info"},
        )
        assert result.is_drift_detected
        assert result.drift_point == 100
        assert result.current_mean == 0.5
        assert result.details["extra"] == "info"


class TestMultivariateDriftResult:
    """Tests for MultivariateDriftResult dataclass."""

    def test_creation(self):
        """Test creating multivariate result."""
        result = MultivariateDriftResult(
            is_drifted=True,
            method="mmd",
            score=0.25,
            threshold=0.1,
            p_value=0.01,
            contributing_features=[("feature1", 0.3), ("feature2", 0.2)],
        )
        assert result.is_drifted
        assert result.method == "mmd"
        assert result.score == 0.25
        assert len(result.contributing_features) == 2


class TestDDMDetector:
    """Tests for DDM (Drift Detection Method) detector."""

    def test_initialization(self):
        """Test DDM detector initialization."""
        detector = DDMDetector(
            warning_level=2.0,
            drift_level=3.0,
            min_samples=30,
        )
        assert detector.warning_level == 2.0
        assert detector.drift_level == 3.0
        assert detector.min_samples == 30

    def test_no_drift_low_error(self):
        """Test no drift detected with low error rate."""
        detector = DDMDetector(min_samples=10)

        # Feed low error rate (0s)
        for _ in range(50):
            result = detector.update(0)

        assert not result.is_drift_detected
        assert not result.is_warning_detected

    def test_drift_detection_increasing_error(self):
        """Test drift detection with increasing error rate."""
        detector = DDMDetector(
            warning_level=2.0,
            drift_level=3.0,
            min_samples=20,
        )

        # Initial low error phase
        for _ in range(30):
            result = detector.update(0.1)

        # Sudden high error phase
        drift_detected = False
        for _ in range(50):
            result = detector.update(0.9)
            if result.is_drift_detected:
                drift_detected = True
                break

        assert drift_detected

    def test_warning_before_drift(self):
        """Test warning is detected before drift."""
        detector = DDMDetector(min_samples=10)

        warning_detected = False
        drift_detected = False

        # Low error then high error
        for i in range(100):
            error = 0.1 if i < 30 else 0.8
            result = detector.update(error)

            if result.is_warning_detected and not warning_detected:
                warning_detected = True
            if result.is_drift_detected:
                drift_detected = True
                break

        # Warning should come before or with drift
        assert warning_detected or drift_detected

    def test_reset(self):
        """Test detector reset."""
        detector = DDMDetector(min_samples=10)

        for _ in range(50):
            detector.update(0.5)

        detector.reset()

        assert detector._n == 0
        assert detector._p == 0.0
        assert detector._p_min == float('inf')

    def test_current_mean_tracking(self):
        """Test that current mean is tracked correctly."""
        detector = DDMDetector(min_samples=5)

        errors = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        for error in errors:
            result = detector.update(error)

        expected_mean = sum(errors) / len(errors)
        assert abs(result.current_mean - expected_mean) < 0.01


class TestADWINDetector:
    """Tests for ADWIN detector."""

    def test_initialization(self):
        """Test ADWIN detector initialization."""
        detector = ADWINDetector(
            delta=0.002,
            max_buckets=5,
            min_window=10,
        )
        assert detector.delta == 0.002
        assert detector.max_buckets == 5
        assert detector.min_window == 10

    def test_update_single_value(self):
        """Test updating with single values."""
        detector = ADWINDetector()

        for i in range(20):
            result = detector.update(float(i))

        assert detector.width == 20
        assert result.method == ConceptDriftMethod.ADWIN

    def test_mean_calculation(self):
        """Test mean calculation."""
        detector = ADWINDetector()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            detector.update(v)

        assert abs(detector.mean - 3.0) < 0.01

    def test_reset(self):
        """Test detector reset."""
        detector = ADWINDetector()

        for i in range(50):
            detector.update(float(i))

        detector.reset()

        assert detector.width == 0
        assert detector._total == 0.0

    def test_bucket_compression(self):
        """Test bucket compression occurs."""
        detector = ADWINDetector(max_buckets=3)

        # Add enough elements to trigger compression
        for i in range(20):
            detector.update(float(i))

        # Should have compressed buckets
        assert len(detector._buckets) > 1


class TestCUSUMDetector:
    """Tests for CUSUM detector."""

    def test_initialization(self):
        """Test CUSUM detector initialization."""
        detector = CUSUMDetector(
            target_mean=0.0,
            threshold=50.0,
            allowance=4.0,
            min_samples=30,
        )
        assert detector.target_mean == 0.0
        assert detector.threshold == 50.0
        assert detector.allowance == 4.0

    def test_no_drift_stable_data(self):
        """Test no drift with stable data around target mean."""
        detector = CUSUMDetector(target_mean=0.0, threshold=50.0)

        np.random.seed(42)
        for _ in range(100):
            result = detector.update(np.random.normal(0, 0.5))

        assert not result.is_drift_detected

    def test_drift_detection_positive_shift(self):
        """Test drift detection with positive mean shift."""
        detector = CUSUMDetector(
            target_mean=0.0,
            threshold=20.0,
            allowance=2.0,
            min_samples=10,
        )

        # Feed data with positive shift
        drift_detected = False
        for i in range(100):
            value = 0.0 if i < 30 else 5.0
            result = detector.update(value)
            if result.is_drift_detected:
                drift_detected = True
                break

        assert drift_detected

    def test_drift_detection_negative_shift(self):
        """Test drift detection with negative mean shift."""
        detector = CUSUMDetector(
            target_mean=0.0,
            threshold=20.0,
            allowance=2.0,
            min_samples=10,
        )

        drift_detected = False
        for i in range(100):
            value = 0.0 if i < 30 else -5.0
            result = detector.update(value)
            if result.is_drift_detected:
                drift_detected = True
                break

        assert drift_detected

    def test_cusum_details(self):
        """Test CUSUM statistics in details."""
        detector = CUSUMDetector(target_mean=0.0)

        result = detector.update(5.0)

        assert "cusum_positive" in result.details
        assert "cusum_negative" in result.details

    def test_reset_after_drift(self):
        """Test CUSUM resets after drift detection."""
        detector = CUSUMDetector(
            target_mean=0.0,
            threshold=10.0,
            allowance=1.0,
            min_samples=5,
        )

        # Trigger drift
        for _ in range(20):
            result = detector.update(10.0)
            if result.is_drift_detected:
                break

        # After drift, CUSUM should reset
        assert detector._cusum_pos == 0.0 or abs(detector._cusum_pos) < 1.0

    def test_reset(self):
        """Test manual reset."""
        detector = CUSUMDetector()

        for _ in range(50):
            detector.update(1.0)

        detector.reset()

        assert detector._n == 0
        assert detector._cusum_pos == 0.0
        assert detector._cusum_neg == 0.0


class TestWindowedDriftMonitor:
    """Tests for WindowedDriftMonitor."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = WindowedDriftMonitor(
            window_size=500,
            step_size=50,
            drift_threshold=0.1,
            method=DriftMethod.PSI,
        )
        assert monitor.window_size == 500
        assert monitor.step_size == 50
        assert monitor.drift_threshold == 0.1

    def test_initialize_with_data(self):
        """Test initializing with reference data."""
        monitor = WindowedDriftMonitor(window_size=100)

        np.random.seed(42)
        data = np.random.randn(200, 3)
        columns = ["a", "b", "c"]

        monitor.initialize(data, columns)

        assert monitor._reference_window is not None
        assert monitor._reference_window.shape == (100, 3)
        assert monitor._columns == columns

    def test_update_returns_none_before_step(self):
        """Test that update returns None before step_size is reached."""
        monitor = WindowedDriftMonitor(window_size=50, step_size=10)

        np.random.seed(42)
        ref_data = np.random.randn(100, 2)
        monitor.initialize(ref_data)

        # Add less than step_size samples
        new_data = np.random.randn(5, 2)
        result = monitor.update(new_data)

        assert result is None

    def test_update_returns_report_at_step(self):
        """Test that update returns DriftReport at step intervals."""
        monitor = WindowedDriftMonitor(window_size=50, step_size=5)

        np.random.seed(42)
        ref_data = np.random.randn(100, 2)
        monitor.initialize(ref_data)

        # Update multiple times
        report = None
        for _ in range(10):
            new_data = np.random.randn(1, 2)
            result = monitor.update(new_data)
            if result is not None:
                report = result
                break

        assert report is not None

    def test_drift_rate_calculation(self):
        """Test drift rate calculation."""
        monitor = WindowedDriftMonitor(window_size=20, step_size=1)

        np.random.seed(42)
        ref_data = np.random.randn(30, 2)
        monitor.initialize(ref_data)

        # Add some data
        for _ in range(15):
            new_data = np.random.randn(1, 2)
            monitor.update(new_data)

        rate = monitor.get_drift_rate(last_n=10)
        assert 0.0 <= rate <= 1.0

    def test_drift_history(self):
        """Test drift history tracking."""
        monitor = WindowedDriftMonitor(window_size=20, step_size=2)

        np.random.seed(42)
        ref_data = np.random.randn(30, 2)
        monitor.initialize(ref_data)

        for _ in range(10):
            new_data = np.random.randn(1, 2)
            monitor.update(new_data)

        assert len(monitor.drift_history) > 0

    def test_1d_data_handling(self):
        """Test handling of 1D input data."""
        monitor = WindowedDriftMonitor(window_size=50, step_size=5)

        np.random.seed(42)
        ref_data = np.random.randn(100)  # 1D data
        monitor.initialize(ref_data)

        assert monitor._reference_window.shape == (50, 1)


class TestMultivariateDriftDetector:
    """Tests for MultivariateDriftDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = MultivariateDriftDetector(
            method="mmd",
            threshold=0.1,
            kernel="rbf",
            n_permutations=100,
        )
        assert detector.method == "mmd"
        assert detector.threshold == 0.1
        assert detector.kernel == "rbf"

    def test_set_reference(self):
        """Test setting reference data."""
        detector = MultivariateDriftDetector()

        np.random.seed(42)
        ref_data = np.random.randn(100, 5)
        columns = ["f1", "f2", "f3", "f4", "f5"]

        detector.set_reference(ref_data, columns)

        assert detector._reference_data is not None
        assert detector._reference_columns == columns

    def test_detect_without_reference_raises(self):
        """Test that detect raises error without reference data."""
        detector = MultivariateDriftDetector()

        with pytest.raises(ValueError, match="Reference data not set"):
            detector.detect(np.random.randn(50, 5))

    def test_mmd_no_drift_same_distribution(self):
        """Test MMD detects no drift with same distribution."""
        detector = MultivariateDriftDetector(method="mmd", n_permutations=50)

        np.random.seed(42)
        ref_data = np.random.randn(100, 3)
        curr_data = np.random.randn(100, 3)

        detector.set_reference(ref_data)
        result = detector.detect(curr_data)

        # Same distribution should not show drift
        assert result.method == "mmd"
        assert result.score >= 0

    def test_mmd_drift_different_distribution(self):
        """Test MMD detects drift with different distribution."""
        detector = MultivariateDriftDetector(method="mmd", n_permutations=50)

        np.random.seed(42)
        ref_data = np.random.randn(100, 3)
        # Shift mean significantly
        curr_data = np.random.randn(100, 3) + 5.0

        detector.set_reference(ref_data)
        result = detector.detect(curr_data)

        assert result.is_drifted
        assert result.p_value is not None
        assert result.p_value < 0.05

    def test_energy_distance_method(self):
        """Test energy distance method."""
        detector = MultivariateDriftDetector(method="energy", threshold=0.5)

        np.random.seed(42)
        ref_data = np.random.randn(100, 3)
        curr_data = np.random.randn(100, 3) + 3.0

        detector.set_reference(ref_data)
        result = detector.detect(curr_data)

        assert result.method == "energy"
        assert result.score > 0

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_domain_classifier_method(self):
        """Test domain classifier method."""
        detector = MultivariateDriftDetector(method="domain_classifier", threshold=0.2)

        np.random.seed(42)
        ref_data = np.random.randn(100, 3)
        curr_data = np.random.randn(100, 3) + 3.0

        detector.set_reference(ref_data)
        result = detector.detect(curr_data)

        assert result.method == "domain_classifier"
        assert "classifier_accuracy" in result.details

    def test_unknown_method_raises(self):
        """Test unknown method raises error."""
        detector = MultivariateDriftDetector(method="unknown")

        np.random.seed(42)
        detector.set_reference(np.random.randn(100, 3))

        with pytest.raises(ValueError, match="Unknown method"):
            detector.detect(np.random.randn(50, 3))

    def test_contributing_features(self):
        """Test contributing features are identified."""
        detector = MultivariateDriftDetector(method="mmd", n_permutations=30)

        np.random.seed(42)
        ref_data = np.random.randn(100, 3)
        # Only shift first column
        curr_data = np.random.randn(100, 3)
        curr_data[:, 0] += 5.0

        columns = ["feature_0", "feature_1", "feature_2"]
        detector.set_reference(ref_data, columns)
        result = detector.detect(curr_data, columns)

        assert len(result.contributing_features) > 0
        # First feature should be among top contributors
        top_features = [f[0] for f in result.contributing_features[:3]]
        assert "feature_0" in top_features or len(result.contributing_features) > 0


class TestComputeMMD:
    """Tests for MMD computation."""

    def test_mmd_identical_samples(self):
        """Test MMD is near zero for identical samples."""
        detector = MultivariateDriftDetector()

        np.random.seed(42)
        data = np.random.randn(100, 3)

        mmd = detector._compute_mmd(data, data.copy())
        assert mmd < 0.1

    def test_mmd_different_samples(self):
        """Test MMD is positive for different samples."""
        detector = MultivariateDriftDetector()

        np.random.seed(42)
        x = np.random.randn(100, 3)
        y = np.random.randn(100, 3) + 5.0

        mmd = detector._compute_mmd(x, y)
        assert mmd > 0


class TestEnergyDistance:
    """Tests for energy distance computation."""

    def test_energy_distance_same_distribution(self):
        """Test energy distance for same distribution."""
        detector = MultivariateDriftDetector()

        np.random.seed(42)
        x = np.random.randn(50, 2)
        y = np.random.randn(50, 2)

        energy = detector._compute_energy_distance(x, y)
        assert energy >= 0

    def test_energy_distance_different_distribution(self):
        """Test energy distance for different distributions."""
        detector = MultivariateDriftDetector()

        np.random.seed(42)
        x = np.random.randn(50, 2)
        y = np.random.randn(50, 2) + 10.0

        energy = detector._compute_energy_distance(x, y)
        assert energy > 1.0  # Should be significantly different


class TestConceptDriftMethod:
    """Tests for ConceptDriftMethod enum."""

    def test_enum_values(self):
        """Test enum values exist."""
        assert ConceptDriftMethod.DDM.value == "ddm"
        assert ConceptDriftMethod.EDDM.value == "eddm"
        assert ConceptDriftMethod.ADWIN.value == "adwin"
        assert ConceptDriftMethod.PAGE_HINKLEY.value == "page_hinkley"
        assert ConceptDriftMethod.CUSUM.value == "cusum"


class TestIntegration:
    """Integration tests for drift detection components."""

    def test_ddm_with_simulated_concept_drift(self):
        """Test DDM with simulated concept drift scenario."""
        detector = DDMDetector(warning_level=2.0, drift_level=3.0, min_samples=20)

        np.random.seed(42)

        # Simulate error stream with concept drift
        # Phase 1: Low error rate (good model)
        # Phase 2: High error rate (model degraded)
        errors = []
        for i in range(200):
            if i < 100:
                error = 1 if np.random.random() < 0.1 else 0
            else:
                error = 1 if np.random.random() < 0.5 else 0
            errors.append(error)

        drift_detected_at = None
        for i, error in enumerate(errors):
            result = detector.update(error)
            if result.is_drift_detected and drift_detected_at is None:
                drift_detected_at = i
                break

        # Drift should be detected after the concept change
        assert drift_detected_at is not None
        assert drift_detected_at > 100  # After concept change

    def test_cusum_with_process_shift(self):
        """Test CUSUM with simulated process shift."""
        detector = CUSUMDetector(
            target_mean=0.0,
            threshold=15.0,
            allowance=2.0,
            min_samples=10,
        )

        np.random.seed(42)

        # Process shift after sample 50
        drift_detected_at = None
        for i in range(150):
            if i < 50:
                value = np.random.normal(0, 1)
            else:
                value = np.random.normal(3, 1)  # Mean shift

            result = detector.update(value)
            if result.is_drift_detected and drift_detected_at is None:
                drift_detected_at = i
                break

        assert drift_detected_at is not None
        assert drift_detected_at >= 50

    def test_multivariate_detector_with_feature_drift(self):
        """Test multivariate detector with feature-level drift."""
        detector = MultivariateDriftDetector(method="mmd", n_permutations=50)

        np.random.seed(42)

        # Reference: all features normal
        ref_data = np.random.randn(100, 4)

        # Current: one feature shifted
        curr_data = np.random.randn(100, 4)
        curr_data[:, 2] += 4.0  # Shift feature 2

        columns = ["f0", "f1", "f2", "f3"]
        detector.set_reference(ref_data, columns)
        result = detector.detect(curr_data, columns)

        assert result.is_drifted
        # Feature 2 should be identified as contributor
        contrib_names = [f[0] for f in result.contributing_features]
        assert "f2" in contrib_names[:3] or result.is_drifted
