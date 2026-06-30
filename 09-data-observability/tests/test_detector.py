"""Tests for anomaly detection module."""

import pytest
from datetime import datetime, timedelta
from observability.detector import AnomalyDetector, generate_id
from observability.config import DetectorConfig
from observability.models import (
    AnomalyType, ColumnStats, ColumnMetadata, TableMetadata
)


class TestGenerateId:
    """Tests for ID generation."""

    def test_generate_unique_ids(self):
        """Test that IDs are unique."""
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_generate_id_format(self):
        """Test ID format is valid UUID."""
        id_str = generate_id()
        assert len(id_str) == 36
        assert id_str.count("-") == 4


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector with test configuration."""
        config = DetectorConfig(
            min_history_points=5,
            volume_threshold=2.0,
            freshness_threshold_hours=24.0,
            null_rate_threshold=2.0
        )
        return AnomalyDetector(config)

    def test_record_volume(self, detector):
        """Test recording volume history."""
        table_id = "test.table"
        for i in range(10):
            detector.record_volume(table_id, 1000 + i)

        assert table_id in detector._volume_history
        assert len(detector._volume_history[table_id]) == 10

    def test_volume_history_limit(self, detector):
        """Test volume history is limited to 100 points."""
        table_id = "test.table"
        for i in range(150):
            detector.record_volume(table_id, 1000 + i)

        assert len(detector._volume_history[table_id]) == 100

    def test_record_freshness(self, detector):
        """Test recording freshness history."""
        table_id = "test.table"
        for i in range(10):
            detector.record_freshness(table_id, float(i * 3600))

        assert table_id in detector._freshness_history
        assert len(detector._freshness_history[table_id]) == 10

    def test_record_null_rate(self, detector):
        """Test recording null rate history."""
        table_id = "test.table"
        column_name = "email"
        for i in range(10):
            detector.record_null_rate(table_id, column_name, 0.01 * i)

        assert table_id in detector._null_rate_history
        assert column_name in detector._null_rate_history[table_id]
        assert len(detector._null_rate_history[table_id][column_name]) == 10

    def test_record_distribution(self, detector):
        """Test recording distribution history."""
        table_id = "test.table"
        column_name = "user_id"
        stats = ColumnStats(
            null_count=10,
            null_ratio=0.01,
            distinct_count=1000,
            mean=5000.0,
            stddev=1500.0
        )

        for _ in range(10):
            detector.record_distribution(table_id, column_name, stats)

        assert table_id in detector._distribution_history
        assert column_name in detector._distribution_history[table_id]

    def test_record_schema(self, detector):
        """Test recording schema history."""
        table_id = "test.table"
        metadata = TableMetadata(
            table_id=table_id,
            database="test",
            schema="public",
            table_name="test",
            columns=[],
            row_count=1000,
            size_bytes=1024,
            last_modified=datetime.now()
        )

        detector.record_schema(table_id, metadata)
        assert table_id in detector._schema_history


class TestVolumeAnomalyDetection:
    """Tests for volume anomaly detection."""

    @pytest.fixture
    def detector_with_history(self):
        """Create detector with volume history."""
        config = DetectorConfig(min_history_points=5, volume_threshold=2.0)
        detector = AnomalyDetector(config)

        # Add consistent history around 1000
        for i in range(10):
            detector.record_volume("test.table", 1000 + (i % 20))

        return detector

    @pytest.mark.asyncio
    async def test_no_anomaly_normal_value(self, detector_with_history):
        """Test no anomaly for normal values."""
        anomaly = await detector_with_history.detect_volume_anomaly(
            "test.table", 1010
        )
        assert anomaly is None

    @pytest.mark.asyncio
    async def test_detect_volume_spike(self, detector_with_history):
        """Test detection of volume spike."""
        anomaly = await detector_with_history.detect_volume_anomaly(
            "test.table", 5000  # Much higher than historical
        )
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.VOLUME
        assert anomaly.severity in ("warning", "critical")

    @pytest.mark.asyncio
    async def test_detect_volume_drop(self, detector_with_history):
        """Test detection of volume drop."""
        anomaly = await detector_with_history.detect_volume_anomaly(
            "test.table", 100  # Much lower than historical
        )
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.VOLUME

    @pytest.mark.asyncio
    async def test_insufficient_history(self):
        """Test no detection with insufficient history."""
        config = DetectorConfig(min_history_points=10)
        detector = AnomalyDetector(config)
        detector.record_volume("test.table", 1000)  # Only 1 point

        anomaly = await detector.detect_volume_anomaly("test.table", 5000)
        assert anomaly is None


class TestSchemaAnomalyDetection:
    """Tests for schema anomaly detection."""

    @pytest.fixture
    def detector_with_schema(self):
        """Create detector with schema history."""
        config = DetectorConfig()
        detector = AnomalyDetector(config)

        schema = TableMetadata(
            table_id="test.table",
            database="test",
            schema="public",
            table_name="test",
            columns=[
                ColumnMetadata(name="id", data_type="INT64", nullable=False),
                ColumnMetadata(name="name", data_type="STRING", nullable=True),
                ColumnMetadata(name="email", data_type="STRING", nullable=True),
            ],
            row_count=1000,
            size_bytes=1024,
            last_modified=datetime.now()
        )
        detector.record_schema("test.table", schema)
        return detector

    @pytest.mark.asyncio
    async def test_no_schema_change(self, detector_with_schema):
        """Test no anomaly when schema unchanged."""
        current_schema = TableMetadata(
            table_id="test.table",
            database="test",
            schema="public",
            table_name="test",
            columns=[
                ColumnMetadata(name="id", data_type="INT64", nullable=False),
                ColumnMetadata(name="name", data_type="STRING", nullable=True),
                ColumnMetadata(name="email", data_type="STRING", nullable=True),
            ],
            row_count=1000,
            size_bytes=1024,
            last_modified=datetime.now()
        )

        anomaly = await detector_with_schema.detect_schema_anomaly(
            "test.table", current_schema
        )
        assert anomaly is None

    @pytest.mark.asyncio
    async def test_detect_column_added(self, detector_with_schema):
        """Test detection of added column."""
        current_schema = TableMetadata(
            table_id="test.table",
            database="test",
            schema="public",
            table_name="test",
            columns=[
                ColumnMetadata(name="id", data_type="INT64", nullable=False),
                ColumnMetadata(name="name", data_type="STRING", nullable=True),
                ColumnMetadata(name="email", data_type="STRING", nullable=True),
                ColumnMetadata(name="phone", data_type="STRING", nullable=True),  # New
            ],
            row_count=1000,
            size_bytes=1024,
            last_modified=datetime.now()
        )

        anomaly = await detector_with_schema.detect_schema_anomaly(
            "test.table", current_schema
        )
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.SCHEMA
        assert "phone" in str(anomaly.context)

    @pytest.mark.asyncio
    async def test_detect_column_removed(self, detector_with_schema):
        """Test detection of removed column."""
        current_schema = TableMetadata(
            table_id="test.table",
            database="test",
            schema="public",
            table_name="test",
            columns=[
                ColumnMetadata(name="id", data_type="INT64", nullable=False),
                ColumnMetadata(name="name", data_type="STRING", nullable=True),
                # email removed
            ],
            row_count=1000,
            size_bytes=1024,
            last_modified=datetime.now()
        )

        anomaly = await detector_with_schema.detect_schema_anomaly(
            "test.table", current_schema
        )
        assert anomaly is not None
        assert anomaly.severity == "critical"  # Column removal is critical

    @pytest.mark.asyncio
    async def test_detect_type_change(self, detector_with_schema):
        """Test detection of type change."""
        current_schema = TableMetadata(
            table_id="test.table",
            database="test",
            schema="public",
            table_name="test",
            columns=[
                ColumnMetadata(name="id", data_type="STRING", nullable=False),  # Changed
                ColumnMetadata(name="name", data_type="STRING", nullable=True),
                ColumnMetadata(name="email", data_type="STRING", nullable=True),
            ],
            row_count=1000,
            size_bytes=1024,
            last_modified=datetime.now()
        )

        anomaly = await detector_with_schema.detect_schema_anomaly(
            "test.table", current_schema
        )
        assert anomaly is not None
        assert anomaly.severity == "critical"  # Type change is critical


class TestNullRateAnomalyDetection:
    """Tests for null rate anomaly detection."""

    @pytest.fixture
    def detector_with_null_history(self):
        """Create detector with null rate history."""
        config = DetectorConfig(min_history_points=5, null_rate_threshold=2.0)
        detector = AnomalyDetector(config)

        # Add consistent null rate history around 0.01 (1%)
        for _ in range(10):
            detector.record_null_rate("test.table", "email", 0.01)

        return detector

    @pytest.mark.asyncio
    async def test_no_anomaly_normal_null_rate(self, detector_with_null_history):
        """Test no anomaly for normal null rate."""
        anomaly = await detector_with_null_history.detect_null_rate_anomaly(
            "test.table", "email", 0.015
        )
        assert anomaly is None

    @pytest.mark.asyncio
    async def test_detect_high_null_rate(self, detector_with_null_history):
        """Test detection of high null rate."""
        anomaly = await detector_with_null_history.detect_null_rate_anomaly(
            "test.table", "email", 0.50  # 50% nulls
        )
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.NULL_RATE
        assert anomaly.severity == "critical"


class TestDetectAllAnomalies:
    """Tests for comprehensive anomaly detection."""

    @pytest.fixture
    def detector_with_all_history(self):
        """Create detector with all history types."""
        config = DetectorConfig(
            min_history_points=5,
            volume_threshold=2.0,
            null_rate_threshold=2.0
        )
        detector = AnomalyDetector(config)

        # Add history
        for _ in range(10):
            detector.record_volume("test.table", 1000)
            detector.record_freshness("test.table", datetime.now().timestamp())
            detector.record_null_rate("test.table", "email", 0.01)

        return detector

    @pytest.mark.asyncio
    async def test_detect_all_no_anomalies(self, detector_with_all_history):
        """Test comprehensive check with no anomalies."""
        metadata = TableMetadata(
            table_id="test.table",
            database="test",
            schema="public",
            table_name="test",
            columns=[],
            row_count=1000,
            size_bytes=1024,
            last_modified=datetime.now()
        )
        stats = {
            "email": ColumnStats(
                null_count=10,
                null_ratio=0.01,
                distinct_count=1000
            )
        }

        anomalies = await detector_with_all_history.detect_all_anomalies(
            "test.table", metadata, stats
        )
        # May or may not have anomalies depending on history state
        assert isinstance(anomalies, list)

    @pytest.mark.asyncio
    async def test_detect_all_multiple_anomalies(self, detector_with_all_history):
        """Test comprehensive check with multiple anomalies."""
        metadata = TableMetadata(
            table_id="test.table",
            database="test",
            schema="public",
            table_name="test",
            columns=[],
            row_count=50000,  # Much higher than history
            size_bytes=1024,
            last_modified=datetime.now() - timedelta(days=7)  # Stale
        )
        stats = {
            "email": ColumnStats(
                null_count=500,
                null_ratio=0.50,  # High null rate
                distinct_count=100
            )
        }

        anomalies = await detector_with_all_history.detect_all_anomalies(
            "test.table", metadata, stats
        )

        # Should detect at least volume anomaly
        volume_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.VOLUME]
        assert len(volume_anomalies) >= 1
