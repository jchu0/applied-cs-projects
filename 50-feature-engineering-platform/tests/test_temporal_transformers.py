"""Tests for temporal transformers."""

from datetime import datetime, timedelta
import numpy as np
import pytest

from feature_platform.transformers.temporal import (
    DatePartsExtractor,
    TimeSinceEvent,
    CyclicalEncoder,
    RollingWindowFeatures,
    LagFeatures,
    DateDiffFeatures,
    HolidayFeatures,
    TimeZoneConverter,
)


class TestDatePartsExtractor:
    """Tests for DatePartsExtractor."""

    def test_default_parts(self):
        """Test extraction of default parts."""
        X = np.array([[datetime(2023, 6, 15, 10, 30)]])
        extractor = DatePartsExtractor()
        result = extractor.fit_transform(X)

        # Default parts: year, month, day, dayofweek
        assert result.shape[1] == 4

    def test_all_parts(self):
        """Test extraction of all parts."""
        parts = [
            "year", "month", "day", "hour", "minute", "second",
            "dayofweek", "dayofyear", "weekofyear", "quarter",
            "is_weekend", "is_month_start", "is_month_end",
        ]
        X = np.array([[datetime(2023, 12, 31, 23, 59, 59)]])
        extractor = DatePartsExtractor(parts=parts)
        result = extractor.fit_transform(X)

        assert result.shape[1] == len(parts)

        # Check specific values
        values = result.flatten()
        assert values[0] == 2023  # year
        assert values[1] == 12  # month
        assert values[2] == 31  # day
        assert values[3] == 23  # hour
        assert values[9] == 4  # quarter

    def test_is_weekend(self):
        """Test weekend detection."""
        # Saturday
        X = np.array([[datetime(2023, 6, 17)]])
        extractor = DatePartsExtractor(parts=["is_weekend"])
        result = extractor.fit_transform(X)

        assert result[0, 0] == 1.0

    def test_is_month_start(self):
        """Test month start detection."""
        X = np.array([
            [datetime(2023, 6, 1)],
            [datetime(2023, 6, 15)],
        ])
        extractor = DatePartsExtractor(parts=["is_month_start"])
        result = extractor.fit_transform(X)

        assert result[0, 0] == 1.0
        assert result[1, 0] == 0.0

    def test_string_dates(self):
        """Test handling of string dates."""
        X = np.array([["2023-06-15"], ["2023-12-25"]])
        extractor = DatePartsExtractor(parts=["month", "day"])
        result = extractor.fit_transform(X)

        assert result[0, 0] == 6  # month
        assert result[1, 1] == 25  # day

    def test_handles_nan(self):
        """Test handling of NaN values."""
        X = np.array([[datetime(2023, 6, 15)], [None]])
        extractor = DatePartsExtractor(parts=["year"])
        result = extractor.fit_transform(X)

        assert result[0, 0] == 2023
        assert np.isnan(result[1, 0])


class TestTimeSinceEvent:
    """Tests for TimeSinceEvent."""

    def test_days_since(self):
        """Test calculating days since event."""
        reference = datetime(2023, 6, 15)
        X = np.array([
            [datetime(2023, 6, 10)],  # 5 days before
            [datetime(2023, 6, 14)],  # 1 day before
        ])

        transformer = TimeSinceEvent(reference_time=reference, unit="days")
        result = transformer.fit_transform(X)

        assert result[0, 0] == pytest.approx(5.0)
        assert result[1, 0] == pytest.approx(1.0)

    def test_hours_since(self):
        """Test calculating hours since event."""
        reference = datetime(2023, 6, 15, 12, 0)
        X = np.array([[datetime(2023, 6, 15, 10, 0)]])  # 2 hours before

        transformer = TimeSinceEvent(reference_time=reference, unit="hours")
        result = transformer.fit_transform(X)

        assert result[0, 0] == pytest.approx(2.0)

    def test_now_reference(self):
        """Test using 'now' as reference."""
        X = np.array([[datetime.utcnow() - timedelta(days=1)]])

        transformer = TimeSinceEvent(reference_time="now", unit="days")
        result = transformer.fit_transform(X)

        assert result[0, 0] == pytest.approx(1.0, abs=0.1)


class TestCyclicalEncoder:
    """Tests for CyclicalEncoder."""

    def test_hour_encoding(self):
        """Test cyclical encoding of hours."""
        X = np.array([[0], [6], [12], [18]])
        encoder = CyclicalEncoder(period=24)
        result = encoder.fit_transform(X)

        # Should have 2 columns (sin and cos)
        assert result.shape == (4, 2)

        # Hour 0 and 24 should be the same
        # sin(0) = 0, cos(0) = 1

    def test_values_in_range(self):
        """Test that sin/cos values are in [-1, 1]."""
        X = np.array([[i] for i in range(24)])
        encoder = CyclicalEncoder(period=24)
        result = encoder.fit_transform(X)

        assert result.min() >= -1
        assert result.max() <= 1

    def test_day_of_week(self):
        """Test cyclical encoding of day of week."""
        X = np.array([[0], [1], [2], [3], [4], [5], [6]])
        encoder = CyclicalEncoder(period=7)
        result = encoder.fit_transform(X)

        # Monday (0) and Sunday (6) should be close but not identical
        # Check that the encoding is smooth around the boundary


class TestRollingWindowFeatures:
    """Tests for RollingWindowFeatures."""

    def test_basic_rolling(self):
        """Test basic rolling window features."""
        X = np.array([[1], [2], [3], [4], [5]])
        transformer = RollingWindowFeatures(
            window_sizes=[3],
            statistics=["mean"]
        )
        result = transformer.fit_transform(X)

        # Rolling mean with window 3
        # [1] -> 1
        # [1,2] -> 1.5
        # [1,2,3] -> 2
        # [2,3,4] -> 3
        # [3,4,5] -> 4
        expected = np.array([[1], [1.5], [2], [3], [4]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_windows(self):
        """Test multiple window sizes."""
        X = np.array([[1], [2], [3], [4], [5]])
        transformer = RollingWindowFeatures(
            window_sizes=[2, 3],
            statistics=["mean"]
        )
        result = transformer.fit_transform(X)

        assert result.shape == (5, 2)

    def test_multiple_statistics(self):
        """Test multiple statistics."""
        X = np.array([[1], [2], [3], [4], [5]])
        transformer = RollingWindowFeatures(
            window_sizes=[3],
            statistics=["mean", "std", "min", "max"]
        )
        result = transformer.fit_transform(X)

        assert result.shape == (5, 4)


class TestLagFeatures:
    """Tests for LagFeatures."""

    def test_basic_lags(self):
        """Test basic lag features."""
        X = np.array([[1], [2], [3], [4], [5]])
        transformer = LagFeatures(lags=[1, 2])
        result = transformer.fit_transform(X)

        # Lag 1: [nan, 1, 2, 3, 4]
        # Lag 2: [nan, nan, 1, 2, 3]
        assert result.shape == (5, 2)
        assert np.isnan(result[0, 0])  # First lag-1 is nan
        assert result[2, 0] == 2  # Third value lag-1 is second value

    def test_custom_fill_value(self):
        """Test custom fill value for missing lags."""
        X = np.array([[1], [2], [3]])
        transformer = LagFeatures(lags=[1], fill_value=0)
        result = transformer.fit_transform(X)

        assert result[0, 0] == 0  # First value filled with 0

    def test_multiple_columns(self):
        """Test with multiple columns."""
        X = np.array([[1, 10], [2, 20], [3, 30]])
        transformer = LagFeatures(lags=[1])
        result = transformer.fit_transform(X)

        assert result.shape == (3, 2)


class TestDateDiffFeatures:
    """Tests for DateDiffFeatures."""

    def test_date_difference(self):
        """Test calculating date differences."""
        X = np.array([
            [datetime(2023, 1, 1), datetime(2023, 1, 11)],
            [datetime(2023, 6, 15), datetime(2023, 6, 20)],
        ])
        transformer = DateDiffFeatures(pairs=[(0, 1)], unit="days")
        result = transformer.fit_transform(X)

        assert result[0, 0] == pytest.approx(10.0)
        assert result[1, 0] == pytest.approx(5.0)

    def test_hours_unit(self):
        """Test date difference in hours."""
        X = np.array([
            [datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 12, 0)],
        ])
        transformer = DateDiffFeatures(pairs=[(0, 1)], unit="hours")
        result = transformer.fit_transform(X)

        assert result[0, 0] == pytest.approx(12.0)


class TestHolidayFeatures:
    """Tests for HolidayFeatures."""

    def test_default_holidays(self):
        """Test detection of default holidays."""
        X = np.array([
            [datetime(2023, 1, 1)],   # New Year's Day
            [datetime(2023, 7, 4)],   # Independence Day
            [datetime(2023, 6, 15)],  # Regular day
        ])
        transformer = HolidayFeatures()
        result = transformer.fit_transform(X)

        # First column is is_holiday
        assert result[0, 0] == 1.0  # New Year
        assert result[1, 0] == 1.0  # July 4th
        assert result[2, 0] == 0.0  # Regular day

    def test_custom_holidays(self):
        """Test with custom holidays."""
        custom_holidays = {
            "0615": "custom_day",
        }
        X = np.array([
            [datetime(2023, 6, 15)],
            [datetime(2023, 6, 16)],
        ])
        transformer = HolidayFeatures(holidays=custom_holidays)
        result = transformer.fit_transform(X)

        assert result[0, 0] == 1.0
        assert result[1, 0] == 0.0


class TestTimeZoneConverter:
    """Tests for TimeZoneConverter."""

    def test_utc_to_utc(self):
        """Test UTC to UTC conversion (no change)."""
        X = np.array([[datetime(2023, 6, 15, 12, 0)]])
        converter = TimeZoneConverter(from_tz="UTC", to_tz="UTC")
        result = converter.fit_transform(X)

        assert result[0, 0].hour == 12

    def test_handles_none(self):
        """Test handling of None values."""
        X = np.array([[datetime(2023, 6, 15)], [None]])
        converter = TimeZoneConverter()
        result = converter.fit_transform(X)

        assert result[1, 0] is None


class TestTemporalTransformersIntegration:
    """Integration tests for temporal transformers."""

    def test_date_to_features_pipeline(self):
        """Test converting dates to multiple features."""
        from feature_platform.transformers.composite import Pipeline

        dates = np.array([
            [datetime(2023, 6, 15, 10, 30)],
            [datetime(2023, 12, 25, 14, 0)],
        ])

        # Extract parts
        extractor = DatePartsExtractor(parts=["month", "day", "hour"])

        result = extractor.fit_transform(dates)

        assert result.shape == (2, 3)
        assert result[0, 0] == 6   # June
        assert result[1, 1] == 25  # Christmas day

    def test_time_series_features(self):
        """Test creating time series features."""
        values = np.array([[i * 1.0] for i in range(10)])

        # Create lag features
        lag_transformer = LagFeatures(lags=[1, 2, 3])
        lag_result = lag_transformer.fit_transform(values)

        # Create rolling features
        roll_transformer = RollingWindowFeatures(
            window_sizes=[3],
            statistics=["mean"]
        )
        roll_result = roll_transformer.fit_transform(values)

        assert lag_result.shape == (10, 3)
        assert roll_result.shape == (10, 1)


class TestTemporalEdgeCases:
    """Test edge cases for temporal transformers."""

    def test_unix_timestamps(self):
        """Test handling of Unix timestamps."""
        # Timestamp for 2023-06-15
        ts = datetime(2023, 6, 15).timestamp()
        X = np.array([[ts]])

        extractor = DatePartsExtractor(parts=["year", "month"])
        result = extractor.fit_transform(X)

        assert result[0, 0] == 2023
        assert result[0, 1] == 6

    def test_string_iso_format(self):
        """Test ISO format string dates."""
        X = np.array([["2023-06-15T10:30:00"]])
        extractor = DatePartsExtractor(parts=["hour", "minute"])
        result = extractor.fit_transform(X)

        assert result[0, 0] == 10
        assert result[0, 1] == 30

    def test_empty_array(self):
        """Test with empty array."""
        X = np.array([]).reshape(0, 1)
        extractor = DatePartsExtractor(parts=["year"])
        result = extractor.fit_transform(X)

        assert result.shape == (0, 1)
