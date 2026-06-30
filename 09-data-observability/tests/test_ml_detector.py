"""Tests for ML-based anomaly detection."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from observability.config import DetectorConfig
from observability.models import (
    Anomaly,
    AnomalyType,
    ColumnMetadata,
    ColumnStats,
    TableMetadata,
)
from observability.ml_detector import (
    AnomalyCorrelator,
    DetectionResult,
    EnsembleDetector,
    FeatureVector,
    StatisticalDetector,
)

# Check if sklearn is available
try:
    import sklearn
    from observability.ml_detector import IsolationForestDetector
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    IsolationForestDetector = None


# Fixtures
@pytest.fixture
def detector_config():
    """Create detector configuration."""
    return DetectorConfig(
        volume_threshold=3.0,
        null_rate_threshold=3.0,
        min_history_points=5,
    )


@pytest.fixture
def sample_table_metadata():
    """Create sample table metadata."""
    return TableMetadata(
        table_id="db.schema.test_table",
        database="db",
        schema="schema",
        table_name="test_table",
        columns=[
            ColumnMetadata(name="id", data_type="bigint", nullable=False),
            ColumnMetadata(name="name", data_type="varchar", nullable=True),
            ColumnMetadata(name="value", data_type="numeric", nullable=True),
        ],
        row_count=10000,
        size_bytes=5000000,
        last_modified=datetime.now(),
    )


@pytest.fixture
def sample_column_stats():
    """Create sample column statistics."""
    return {
        "id": ColumnStats(
            null_count=0, null_ratio=0.0, distinct_count=10000
        ),
        "name": ColumnStats(
            null_count=100, null_ratio=0.01, distinct_count=5000
        ),
        "value": ColumnStats(
            null_count=500, null_ratio=0.05, distinct_count=1000,
            mean=50.0, stddev=10.0, min_value=0.0, max_value=100.0,
        ),
    }


# Statistical Detector Tests
class TestStatisticalDetector:
    """Tests for StatisticalDetector."""

    def test_init(self, detector_config):
        """Test detector initialization."""
        detector = StatisticalDetector(detector_config)
        assert detector.config == detector_config

    def test_record_volume(self, detector_config):
        """Test recording volume history."""
        detector = StatisticalDetector(detector_config)

        for i in range(10):
            detector.record_volume("test_table", 1000 + i * 10)

        assert len(detector._volume_history["test_table"]) == 10

    def test_record_volume_max_history(self, detector_config):
        """Test volume history is capped at 100."""
        detector = StatisticalDetector(detector_config)

        for i in range(150):
            detector.record_volume("test_table", 1000 + i)

        assert len(detector._volume_history["test_table"]) == 100

    def test_record_null_rate(self, detector_config):
        """Test recording null rate history."""
        detector = StatisticalDetector(detector_config)

        for i in range(10):
            detector.record_null_rate("test_table", "col1", 0.01 + i * 0.001)

        assert len(detector._null_rate_history["test_table"]["col1"]) == 10

    @pytest.mark.asyncio
    async def test_detect_volume_anomaly(self, detector_config):
        """Test volume anomaly detection."""
        detector = StatisticalDetector(detector_config)

        # Build history with some variation (IQR needs non-zero values)
        for i in range(10):
            detector.record_volume("test_table", 1000 + (i % 3) * 50)  # 1000, 1050, 1100, ...

        # Test normal value (within IQR bounds)
        anomaly = await detector._detect_volume("test_table", 1050)
        assert anomaly is None

        # Test anomalous value (far outside IQR bounds)
        anomaly = await detector._detect_volume("test_table", 5000)
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.VOLUME

    @pytest.mark.asyncio
    async def test_detect_volume_insufficient_history(self, detector_config):
        """Test volume detection with insufficient history."""
        detector = StatisticalDetector(detector_config)

        detector.record_volume("test_table", 1000)
        detector.record_volume("test_table", 1100)

        anomaly = await detector._detect_volume("test_table", 5000)
        assert anomaly is None  # Not enough history

    @pytest.mark.asyncio
    async def test_detect_null_rate_anomaly(self, detector_config):
        """Test null rate anomaly detection."""
        detector = StatisticalDetector(detector_config)

        # Build history with low null rate
        for _ in range(10):
            detector.record_null_rate("test_table", "col1", 0.01)

        # Test normal value
        anomaly = await detector._detect_null_rate("test_table", "col1", 0.015)
        assert anomaly is None

        # Test anomalous value
        anomaly = await detector._detect_null_rate("test_table", "col1", 0.5)
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.NULL_RATE

    @pytest.mark.asyncio
    async def test_detect_all(
        self, detector_config, sample_table_metadata, sample_column_stats
    ):
        """Test running all detections."""
        detector = StatisticalDetector(detector_config)

        # Build history
        for _ in range(10):
            detector.record_volume("db.schema.test_table", 10000)
            for col, stats in sample_column_stats.items():
                detector.record_null_rate(
                    "db.schema.test_table", col, stats.null_ratio
                )

        # Run detection
        anomalies = await detector.detect_all(
            "db.schema.test_table",
            sample_table_metadata,
            sample_column_stats,
        )

        assert isinstance(anomalies, list)


# Feature Vector Tests
class TestFeatureVector:
    """Tests for FeatureVector."""

    def test_feature_vector_creation(self):
        """Test creating feature vector."""
        features = np.array([1.0, 2.0, 3.0])
        fv = FeatureVector(
            table_id="test_table",
            features=features,
            feature_names=["f1", "f2", "f3"],
        )

        assert fv.table_id == "test_table"
        assert len(fv.features) == 3
        assert len(fv.feature_names) == 3


# Isolation Forest Tests (only if sklearn available)
@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestIsolationForestDetector:
    """Tests for IsolationForestDetector."""

    def test_init(self):
        """Test Isolation Forest initialization."""
        detector = IsolationForestDetector(
            contamination=0.1,
            n_estimators=50,
        )
        assert detector.contamination == 0.1
        assert detector.n_estimators == 50

    def test_add_training_sample(self):
        """Test adding training samples."""
        detector = IsolationForestDetector()

        fv = FeatureVector(
            table_id="table1",
            features=np.array([1.0, 2.0, 3.0]),
            feature_names=["f1", "f2", "f3"],
        )

        detector.add_training_sample(fv)
        assert len(detector._training_data) == 1

    def test_add_training_sample_mismatched_features(self):
        """Test error on mismatched feature names."""
        detector = IsolationForestDetector()

        fv1 = FeatureVector(
            table_id="table1",
            features=np.array([1.0, 2.0]),
            feature_names=["f1", "f2"],
        )
        fv2 = FeatureVector(
            table_id="table2",
            features=np.array([1.0, 2.0]),
            feature_names=["f1", "f3"],  # Different names
        )

        detector.add_training_sample(fv1)

        with pytest.raises(ValueError):
            detector.add_training_sample(fv2)

    def test_fit_insufficient_samples(self):
        """Test fitting with insufficient samples."""
        detector = IsolationForestDetector()

        for i in range(10):
            fv = FeatureVector(
                table_id=f"table{i}",
                features=np.array([float(i), float(i * 2)]),
                feature_names=["f1", "f2"],
            )
            detector.add_training_sample(fv)

        success = detector.fit(min_samples=30)
        assert success is False

    def test_fit_success(self):
        """Test successful model fitting."""
        detector = IsolationForestDetector()

        # Add enough training data
        np.random.seed(42)
        for i in range(50):
            fv = FeatureVector(
                table_id=f"table{i}",
                features=np.random.randn(5),
                feature_names=["f1", "f2", "f3", "f4", "f5"],
            )
            detector.add_training_sample(fv)

        success = detector.fit(min_samples=30)
        assert success is True
        assert detector._is_fitted

    def test_predict(self):
        """Test prediction on fitted model."""
        detector = IsolationForestDetector()

        # Generate normal training data
        np.random.seed(42)
        for i in range(50):
            fv = FeatureVector(
                table_id=f"table{i}",
                features=np.random.randn(5) * 0.1,  # Small variance
                feature_names=["f1", "f2", "f3", "f4", "f5"],
            )
            detector.add_training_sample(fv)

        detector.fit(min_samples=30)

        # Test normal sample
        normal_fv = FeatureVector(
            table_id="normal",
            features=np.random.randn(5) * 0.1,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )
        result = detector.predict(normal_fv)
        assert isinstance(result, DetectionResult)

        # Test anomalous sample
        anomaly_fv = FeatureVector(
            table_id="anomaly",
            features=np.array([10.0, -10.0, 15.0, -20.0, 25.0]),  # Far from training
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )
        result = detector.predict(anomaly_fv)
        assert result.is_anomaly == True
        assert result.anomaly_score > 0.5

    def test_predict_unfitted_raises(self):
        """Test prediction on unfitted model raises error."""
        detector = IsolationForestDetector()

        fv = FeatureVector(
            table_id="test",
            features=np.array([1.0, 2.0]),
            feature_names=["f1", "f2"],
        )

        with pytest.raises(ValueError):
            detector.predict(fv)

    def test_feature_contributions(self):
        """Test feature contributions are calculated."""
        detector = IsolationForestDetector()

        np.random.seed(42)
        for i in range(50):
            fv = FeatureVector(
                table_id=f"table{i}",
                features=np.random.randn(3),
                feature_names=["f1", "f2", "f3"],
            )
            detector.add_training_sample(fv)

        detector.fit(min_samples=30)

        fv = FeatureVector(
            table_id="test",
            features=np.array([5.0, 0.0, -5.0]),
            feature_names=["f1", "f2", "f3"],
        )
        result = detector.predict(fv)

        assert "f1" in result.feature_contributions
        assert "f2" in result.feature_contributions
        assert "f3" in result.feature_contributions

        # Contributions should sum to approximately 1
        total = sum(result.feature_contributions.values())
        assert abs(total - 1.0) < 0.01


# Ensemble Detector Tests
class TestEnsembleDetector:
    """Tests for EnsembleDetector."""

    def test_init(self, detector_config):
        """Test ensemble detector initialization."""
        detector = EnsembleDetector(detector_config)
        assert detector._statistical_detector is not None

    def test_extract_features(
        self, detector_config, sample_table_metadata, sample_column_stats
    ):
        """Test feature extraction."""
        detector = EnsembleDetector(detector_config)

        features = detector.extract_features(
            "test_table", sample_table_metadata, sample_column_stats
        )

        assert isinstance(features, FeatureVector)
        assert features.table_id == "test_table"
        assert len(features.features) > 0
        assert "row_count" in features.feature_names
        assert "mean_null_ratio" in features.feature_names

    def test_record_sample(
        self, detector_config, sample_table_metadata, sample_column_stats
    ):
        """Test recording samples."""
        detector = EnsembleDetector(detector_config)

        detector.record_sample(
            "test_table", sample_table_metadata, sample_column_stats
        )

        assert "test_table" in detector._feature_history
        assert len(detector._feature_history["test_table"]) == 1

    @pytest.mark.asyncio
    async def test_detect_anomalies(
        self, detector_config, sample_table_metadata, sample_column_stats
    ):
        """Test anomaly detection with ensemble."""
        detector = EnsembleDetector(detector_config)

        # Build some history
        for _ in range(10):
            detector._statistical_detector.record_volume("db.schema.test_table", 10000)
            for col, stats in sample_column_stats.items():
                detector._statistical_detector.record_null_rate(
                    "db.schema.test_table", col, stats.null_ratio
                )

        anomalies = await detector.detect_anomalies(
            "db.schema.test_table",
            sample_table_metadata,
            sample_column_stats,
        )

        assert isinstance(anomalies, list)


# Anomaly Correlator Tests
class TestAnomalyCorrelator:
    """Tests for AnomalyCorrelator."""

    def test_init(self):
        """Test correlator initialization."""
        correlator = AnomalyCorrelator(window_minutes=30)
        assert correlator.window_minutes == 30

    def test_add_anomaly(self):
        """Test adding anomalies."""
        correlator = AnomalyCorrelator()

        anomaly = Anomaly(
            anomaly_id="test-1",
            table_id="table1",
            column_name=None,
            anomaly_type=AnomalyType.VOLUME,
            severity="warning",
            detected_at=datetime.now(),
            metric_value=100,
            expected_range=(0, 50),
            description="Test anomaly",
        )

        correlator.add_anomaly(anomaly)
        assert len(correlator._recent_anomalies) == 1

    def test_prune_old_anomalies(self):
        """Test old anomalies are pruned."""
        correlator = AnomalyCorrelator(window_minutes=5)

        old_anomaly = Anomaly(
            anomaly_id="old-1",
            table_id="table1",
            column_name=None,
            anomaly_type=AnomalyType.VOLUME,
            severity="warning",
            detected_at=datetime.now() - timedelta(minutes=10),
            metric_value=100,
            expected_range=(0, 50),
            description="Old anomaly",
        )

        new_anomaly = Anomaly(
            anomaly_id="new-1",
            table_id="table1",
            column_name=None,
            anomaly_type=AnomalyType.VOLUME,
            severity="warning",
            detected_at=datetime.now(),
            metric_value=100,
            expected_range=(0, 50),
            description="New anomaly",
        )

        correlator._recent_anomalies = [old_anomaly]
        correlator.add_anomaly(new_anomaly)

        assert len(correlator._recent_anomalies) == 1
        assert correlator._recent_anomalies[0].anomaly_id == "new-1"

    def test_find_correlated_same_table(self):
        """Test finding correlations on same table."""
        correlator = AnomalyCorrelator()

        anomaly1 = Anomaly(
            anomaly_id="a1",
            table_id="table1",
            column_name=None,
            anomaly_type=AnomalyType.VOLUME,
            severity="warning",
            detected_at=datetime.now(),
            metric_value=100,
            expected_range=(0, 50),
            description="Anomaly 1",
        )

        anomaly2 = Anomaly(
            anomaly_id="a2",
            table_id="table1",
            column_name="col1",
            anomaly_type=AnomalyType.NULL_RATE,
            severity="warning",
            detected_at=datetime.now(),
            metric_value=0.5,
            expected_range=(0, 0.1),
            description="Anomaly 2",
        )

        correlator.add_anomaly(anomaly1)
        correlator.add_anomaly(anomaly2)

        lineage = {"upstream": [], "downstream": []}
        correlated = correlator.find_correlated(anomaly1, lineage)

        assert len(correlated) == 1
        assert correlated[0].anomaly_id == "a2"

    def test_find_correlated_by_lineage(self):
        """Test finding correlations via lineage."""
        correlator = AnomalyCorrelator()

        anomaly1 = Anomaly(
            anomaly_id="a1",
            table_id="table1",
            column_name=None,
            anomaly_type=AnomalyType.VOLUME,
            severity="warning",
            detected_at=datetime.now(),
            metric_value=100,
            expected_range=(0, 50),
            description="Anomaly 1",
        )

        anomaly2 = Anomaly(
            anomaly_id="a2",
            table_id="upstream_table",
            column_name=None,
            anomaly_type=AnomalyType.FRESHNESS,
            severity="warning",
            detected_at=datetime.now(),
            metric_value=3600,
            expected_range=(0, 1800),
            description="Upstream anomaly",
        )

        correlator.add_anomaly(anomaly1)
        correlator.add_anomaly(anomaly2)

        lineage = {"upstream": ["upstream_table"], "downstream": []}
        correlated = correlator.find_correlated(anomaly1, lineage)

        assert len(correlated) == 1
        assert correlated[0].anomaly_id == "a2"

    def test_get_incident_groups(self):
        """Test grouping anomalies into incidents."""
        correlator = AnomalyCorrelator()

        now = datetime.now()

        # Group 1: Same table
        anomalies = [
            Anomaly(
                anomaly_id="g1-a1",
                table_id="table1",
                column_name=None,
                anomaly_type=AnomalyType.VOLUME,
                severity="warning",
                detected_at=now,
                metric_value=100,
                expected_range=(0, 50),
                description="Group 1 Anomaly 1",
            ),
            Anomaly(
                anomaly_id="g1-a2",
                table_id="table1",
                column_name="col1",
                anomaly_type=AnomalyType.NULL_RATE,
                severity="warning",
                detected_at=now,
                metric_value=0.5,
                expected_range=(0, 0.1),
                description="Group 1 Anomaly 2",
            ),
            # Group 2: Different table, same type, close in time
            Anomaly(
                anomaly_id="g2-a1",
                table_id="table2",
                column_name=None,
                anomaly_type=AnomalyType.FRESHNESS,
                severity="warning",
                detected_at=now,
                metric_value=3600,
                expected_range=(0, 1800),
                description="Group 2 Anomaly 1",
            ),
            Anomaly(
                anomaly_id="g2-a2",
                table_id="table3",
                column_name=None,
                anomaly_type=AnomalyType.FRESHNESS,
                severity="warning",
                detected_at=now + timedelta(seconds=60),
                metric_value=4000,
                expected_range=(0, 1800),
                description="Group 2 Anomaly 2",
            ),
        ]

        for a in anomalies:
            correlator.add_anomaly(a)

        groups = correlator.get_incident_groups()

        # Should have 2 groups
        assert len(groups) >= 2

    def test_get_incident_groups_empty(self):
        """Test getting incident groups with no anomalies."""
        correlator = AnomalyCorrelator()
        groups = correlator.get_incident_groups()
        assert groups == []


# Detection Result Tests
class TestDetectionResult:
    """Tests for DetectionResult."""

    def test_detection_result_creation(self):
        """Test creating detection result."""
        result = DetectionResult(
            is_anomaly=True,
            anomaly_score=0.85,
            feature_contributions={"f1": 0.5, "f2": 0.5},
            confidence=0.9,
        )

        assert result.is_anomaly is True
        assert result.anomaly_score == 0.85
        assert len(result.feature_contributions) == 2
        assert result.confidence == 0.9
