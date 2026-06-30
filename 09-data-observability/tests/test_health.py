"""Tests for health scoring module."""

import pytest
from datetime import datetime, timedelta
from observability.health import HealthScorer
from observability.models import DataHealthScore, TableMetadata, ColumnMetadata, Anomaly, AnomalyType


class TestHealthScorer:
    """Tests for HealthScorer class."""

    @pytest.fixture
    def calculator(self):
        """Create health calculator."""
        return HealthScorer()

    @pytest.fixture
    def healthy_table(self):
        """Create metadata for a healthy table."""
        return TableMetadata(
            table_id="healthy.table",
            database="production",
            schema="public",
            table_name="users",
            columns=[
                ColumnMetadata(name="id", data_type="INT64", nullable=False),
                ColumnMetadata(name="email", data_type="STRING", nullable=False),
            ],
            row_count=100000,
            size_bytes=1024 * 1024 * 100,
            last_modified=datetime.now() - timedelta(hours=1),  # Fresh
        )

    @pytest.fixture
    def stale_table(self):
        """Create metadata for a stale table."""
        return TableMetadata(
            table_id="stale.table",
            database="production",
            schema="public",
            table_name="old_data",
            columns=[],
            row_count=100000,
            size_bytes=1024 * 1024 * 100,
            last_modified=datetime.now() - timedelta(days=7),  # Very stale
        )

    def test_calculate_freshness_score_fresh(self, calculator, healthy_table):
        """Test freshness score for fresh data."""
        score = calculator.calculate_freshness_score(
            healthy_table,
            expected_interval_hours=24
        )
        assert score >= 90.0  # Should be high for fresh data

    def test_calculate_freshness_score_stale(self, calculator, stale_table):
        """Test freshness score for stale data."""
        score = calculator.calculate_freshness_score(
            stale_table,
            expected_interval_hours=24
        )
        assert score < 50.0  # Should be low for stale data

    def test_calculate_volume_score_normal(self, calculator, healthy_table):
        """Test volume score for normal data."""
        score = calculator.calculate_volume_score(
            healthy_table,
            historical_counts=[95000, 100000, 105000, 100000, 98000]
        )
        assert score >= 80.0

    def test_calculate_volume_score_anomaly(self, calculator, healthy_table):
        """Test volume score with anomalous data."""
        # Modify to have very different row count
        healthy_table.row_count = 10000  # Much lower
        score = calculator.calculate_volume_score(
            healthy_table,
            historical_counts=[100000, 100000, 100000, 100000, 100000]
        )
        assert score < 80.0

    def test_calculate_schema_score_no_changes(self, calculator):
        """Test schema score with no recent changes."""
        score = calculator.calculate_schema_score(
            recent_changes=[],
            days_since_last_change=30
        )
        assert score >= 90.0

    def test_calculate_schema_score_with_changes(self, calculator):
        """Test schema score with recent changes."""
        score = calculator.calculate_schema_score(
            recent_changes=["column_removed", "type_changed"],
            days_since_last_change=1
        )
        assert score < 80.0

    def test_calculate_quality_score_good(self, calculator):
        """Test quality score for good data."""
        score = calculator.calculate_quality_score(
            null_ratio=0.01,
            duplicate_ratio=0.001,
            invalid_ratio=0.0
        )
        assert score >= 90.0

    def test_calculate_quality_score_poor(self, calculator):
        """Test quality score for poor data."""
        score = calculator.calculate_quality_score(
            null_ratio=0.30,
            duplicate_ratio=0.10,
            invalid_ratio=0.05
        )
        assert score < 70.0

    def test_calculate_overall_score(self, calculator, healthy_table):
        """Test overall health score calculation."""
        score = calculator.calculate_overall_score(
            healthy_table,
            freshness_score=90.0,
            volume_score=85.0,
            schema_score=95.0,
            quality_score=88.0
        )
        assert isinstance(score, DataHealthScore)
        assert 80.0 <= score.overall_score <= 100.0
        assert score.freshness_score == 90.0
        assert score.volume_score == 85.0

    def test_health_score_factors(self, calculator, healthy_table):
        """Test that health score includes factors."""
        score = calculator.calculate_overall_score(
            healthy_table,
            freshness_score=90.0,
            volume_score=85.0,
            schema_score=95.0,
            quality_score=88.0,
            factors=[
                {"name": "high_null_rate", "impact": -5},
                {"name": "stale_data", "impact": -3}
            ]
        )
        assert len(score.factors) == 2

    def test_weighted_score(self, calculator, healthy_table):
        """Test weighted overall score calculation."""
        # With custom weights
        score = calculator.calculate_overall_score(
            healthy_table,
            freshness_score=50.0,  # Low freshness
            volume_score=100.0,
            schema_score=100.0,
            quality_score=100.0,
            weights={
                "freshness": 0.5,  # High weight on freshness
                "volume": 0.2,
                "schema": 0.1,
                "quality": 0.2
            }
        )
        # Overall should be lower due to freshness weight
        assert score.overall_score < 90.0


class TestHealthScoreModel:
    """Tests for DataHealthScore model."""

    def test_health_score_creation(self):
        """Test creating health score."""
        score = DataHealthScore(
            table_id="test.table",
            overall_score=85.5,
            freshness_score=90.0,
            volume_score=80.0,
            schema_score=95.0,
            quality_score=77.0,
            calculated_at=datetime.now(),
            factors=[{"name": "test", "impact": -5}]
        )
        assert score.overall_score == 85.5
        assert len(score.factors) == 1

    def test_health_score_valid_range(self):
        """Test health scores are in valid range."""
        score = DataHealthScore(
            table_id="test",
            overall_score=100.0,
            freshness_score=100.0,
            volume_score=100.0,
            schema_score=100.0,
            quality_score=100.0,
            calculated_at=datetime.now()
        )
        assert 0 <= score.overall_score <= 100
        assert 0 <= score.freshness_score <= 100


class TestHealthTrends:
    """Tests for health trend analysis."""

    @pytest.fixture
    def calculator(self):
        """Create health calculator."""
        return HealthScorer()

    def test_calculate_trend(self, calculator):
        """Test calculating health score trend."""
        scores = [
            DataHealthScore(
                table_id="test",
                overall_score=90 - i * 5,  # Declining
                freshness_score=90.0,
                volume_score=85.0,
                schema_score=95.0,
                quality_score=88.0,
                calculated_at=datetime.now() - timedelta(days=i)
            )
            for i in range(7)
        ]

        trend = calculator.calculate_trend(scores)
        assert trend["direction"] == "declining"
        assert trend["change_rate"] < 0

    def test_stable_trend(self, calculator):
        """Test detecting stable health trend."""
        scores = [
            DataHealthScore(
                table_id="test",
                overall_score=85.0,  # Stable
                freshness_score=90.0,
                volume_score=85.0,
                schema_score=95.0,
                quality_score=88.0,
                calculated_at=datetime.now() - timedelta(days=i)
            )
            for i in range(7)
        ]

        trend = calculator.calculate_trend(scores)
        assert trend["direction"] == "stable"
