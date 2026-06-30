"""Temporal feature transformers."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import numpy as np

from feature_platform.transformers.base import BaseTransformer


def _to_datetime(value: Any) -> Optional[datetime]:
    """Convert various types to datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, np.datetime64):
        # Handle NaT
        if np.isnat(value):
            return None
        return value.astype("datetime64[us]").astype(datetime)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
        # Try other common formats
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    if isinstance(value, (int, float)):
        # Assume Unix timestamp
        try:
            return datetime.fromtimestamp(value)
        except (ValueError, OSError):
            pass
    return None


class DatePartsExtractor(BaseTransformer):
    """
    Extract date/time parts from datetime features.

    Parameters:
        parts: List of parts to extract ('year', 'month', 'day', 'hour', 'minute',
               'second', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter')
        drop_original: If True, don't include original column in output
    """

    VALID_PARTS = [
        "year", "month", "day", "hour", "minute", "second",
        "dayofweek", "dayofyear", "weekofyear", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
        "is_year_start", "is_year_end"
    ]

    def __init__(
        self,
        parts: Optional[List[str]] = None,
        drop_original: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.parts = parts or ["year", "month", "day", "dayofweek"]
        self.drop_original = drop_original
        self.n_features_: int = 0

        # Validate parts
        for part in self.parts:
            if part not in self.VALID_PARTS:
                raise ValueError(f"Invalid date part: {part}")

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "DatePartsExtractor":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # Generate output column names
        self._output_columns = []
        for col in self._input_columns:
            for part in self.parts:
                self._output_columns.append(f"{col}_{part}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def _extract_part(self, dt: Optional[datetime], part: str) -> float:
        """Extract a specific part from a datetime."""
        if dt is None:
            return np.nan

        if part == "year":
            return dt.year
        elif part == "month":
            return dt.month
        elif part == "day":
            return dt.day
        elif part == "hour":
            return dt.hour
        elif part == "minute":
            return dt.minute
        elif part == "second":
            return dt.second
        elif part == "dayofweek":
            return dt.weekday()
        elif part == "dayofyear":
            return dt.timetuple().tm_yday
        elif part == "weekofyear":
            return dt.isocalendar()[1]
        elif part == "quarter":
            return (dt.month - 1) // 3 + 1
        elif part == "is_weekend":
            return 1.0 if dt.weekday() >= 5 else 0.0
        elif part == "is_month_start":
            return 1.0 if dt.day == 1 else 0.0
        elif part == "is_month_end":
            next_day = dt + timedelta(days=1)
            return 1.0 if next_day.day == 1 else 0.0
        elif part == "is_year_start":
            return 1.0 if dt.month == 1 and dt.day == 1 else 0.0
        elif part == "is_year_end":
            return 1.0 if dt.month == 12 and dt.day == 31 else 0.0
        else:
            return np.nan

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        n_output_cols = self.n_features_ * len(self.parts)
        result = np.zeros((X.shape[0], n_output_cols), dtype=np.float64)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                dt = _to_datetime(X[j, i])
                for k, part in enumerate(self.parts):
                    col_idx = i * len(self.parts) + k
                    result[j, col_idx] = self._extract_part(dt, part)

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {"parts": self.parts, "drop_original": self.drop_original}


class TimeSinceEvent(BaseTransformer):
    """
    Calculate time since a reference event.

    Parameters:
        reference_time: Reference datetime or 'now' for current time
        unit: Time unit for output ('seconds', 'minutes', 'hours', 'days', 'weeks')
    """

    def __init__(
        self,
        reference_time: Union[datetime, str] = "now",
        unit: str = "days",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.reference_time = reference_time
        self.unit = unit
        self.reference_time_: Optional[datetime] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "TimeSinceEvent":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = [f"{col}_time_since" for col in self._input_columns]

        if self.reference_time == "now":
            self.reference_time_ = datetime.utcnow()
        elif isinstance(self.reference_time, datetime):
            self.reference_time_ = self.reference_time
        else:
            self.reference_time_ = datetime.fromisoformat(self.reference_time)

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def _convert_timedelta(self, td: timedelta) -> float:
        """Convert timedelta to the specified unit."""
        total_seconds = td.total_seconds()

        if self.unit == "seconds":
            return total_seconds
        elif self.unit == "minutes":
            return total_seconds / 60
        elif self.unit == "hours":
            return total_seconds / 3600
        elif self.unit == "days":
            return total_seconds / 86400
        elif self.unit == "weeks":
            return total_seconds / 604800
        else:
            return total_seconds

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.zeros((X.shape[0], self.n_features_), dtype=np.float64)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                dt = _to_datetime(X[j, i])
                if dt is None:
                    result[j, i] = np.nan
                else:
                    td = self.reference_time_ - dt
                    result[j, i] = self._convert_timedelta(td)

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {"unit": self.unit}

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "reference_time": self.reference_time_.isoformat() if self.reference_time_ else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        if stats.get("reference_time"):
            self.reference_time_ = datetime.fromisoformat(stats["reference_time"])
        self.n_features_ = stats.get("n_features", 0)


class CyclicalEncoder(BaseTransformer):
    """
    Encode cyclical features using sine and cosine transformations.

    Useful for features like hour of day, day of week, month of year.

    Parameters:
        period: The period of the cycle (e.g., 24 for hours, 7 for days)
    """

    def __init__(
        self,
        period: float = 24.0,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.period = period
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "CyclicalEncoder":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # Generate output column names (sin and cos for each input)
        self._output_columns = []
        for col in self._input_columns:
            self._output_columns.append(f"{col}_sin")
            self._output_columns.append(f"{col}_cos")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)
        X = X.astype(np.float64)

        result = np.zeros((X.shape[0], self.n_features_ * 2), dtype=np.float64)

        for i in range(self.n_features_):
            angle = 2 * np.pi * X[:, i] / self.period
            result[:, i * 2] = np.sin(angle)
            result[:, i * 2 + 1] = np.cos(angle)

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {"period": self.period}


class RollingWindowFeatures(BaseTransformer):
    """
    Compute rolling window statistics over time series data.

    Parameters:
        window_sizes: List of window sizes
        statistics: Statistics to compute ('mean', 'std', 'min', 'max', 'sum')
        min_periods: Minimum number of observations required
    """

    VALID_STATS = ["mean", "std", "min", "max", "sum", "median", "var", "count"]

    def __init__(
        self,
        window_sizes: List[int] = None,
        statistics: List[str] = None,
        min_periods: int = 1,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.window_sizes = window_sizes or [3, 7, 14]
        self.statistics = statistics or ["mean", "std"]
        self.min_periods = min_periods
        self.n_features_: int = 0

        for stat in self.statistics:
            if stat not in self.VALID_STATS:
                raise ValueError(f"Invalid statistic: {stat}")

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "RollingWindowFeatures":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # Generate output column names
        self._output_columns = []
        for col in self._input_columns:
            for window in self.window_sizes:
                for stat in self.statistics:
                    self._output_columns.append(f"{col}_rolling_{window}_{stat}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def _compute_stat(self, window_data: np.ndarray, stat: str) -> float:
        """Compute a single statistic on window data."""
        valid_data = window_data[~np.isnan(window_data)]

        if len(valid_data) < self.min_periods:
            return np.nan

        if stat == "mean":
            return np.mean(valid_data)
        elif stat == "std":
            return np.std(valid_data) if len(valid_data) > 1 else 0.0
        elif stat == "min":
            return np.min(valid_data)
        elif stat == "max":
            return np.max(valid_data)
        elif stat == "sum":
            return np.sum(valid_data)
        elif stat == "median":
            return np.median(valid_data)
        elif stat == "var":
            return np.var(valid_data) if len(valid_data) > 1 else 0.0
        elif stat == "count":
            return len(valid_data)
        else:
            return np.nan

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)
        X = X.astype(np.float64)

        n_output_cols = self.n_features_ * len(self.window_sizes) * len(self.statistics)
        result = np.zeros((X.shape[0], n_output_cols), dtype=np.float64)

        col_idx = 0
        for i in range(self.n_features_):
            for window in self.window_sizes:
                for stat in self.statistics:
                    for j in range(X.shape[0]):
                        start_idx = max(0, j - window + 1)
                        window_data = X[start_idx:j + 1, i]
                        result[j, col_idx] = self._compute_stat(window_data, stat)
                    col_idx += 1

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "window_sizes": self.window_sizes,
            "statistics": self.statistics,
            "min_periods": self.min_periods,
        }


class LagFeatures(BaseTransformer):
    """
    Create lagged versions of features.

    Parameters:
        lags: List of lag values (positive integers)
        fill_value: Value to use for missing lags (default: NaN)
    """

    def __init__(
        self,
        lags: List[int] = None,
        fill_value: float = np.nan,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.lags = lags or [1, 2, 3]
        self.fill_value = fill_value
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "LagFeatures":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # Generate output column names
        self._output_columns = []
        for col in self._input_columns:
            for lag in self.lags:
                self._output_columns.append(f"{col}_lag_{lag}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        n_output_cols = self.n_features_ * len(self.lags)
        result = np.full((X.shape[0], n_output_cols), self.fill_value, dtype=np.float64)

        col_idx = 0
        for i in range(self.n_features_):
            for lag in self.lags:
                if lag < X.shape[0]:
                    result[lag:, col_idx] = X[:-lag, i] if lag > 0 else X[:, i]
                col_idx += 1

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {"lags": self.lags, "fill_value": self.fill_value}


class DateDiffFeatures(BaseTransformer):
    """
    Calculate differences between date columns.

    Parameters:
        pairs: List of (start_col_idx, end_col_idx) pairs
        unit: Time unit for output ('seconds', 'minutes', 'hours', 'days')
    """

    def __init__(
        self,
        pairs: Optional[List[tuple]] = None,
        unit: str = "days",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.pairs = pairs
        self.unit = unit
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "DateDiffFeatures":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # If pairs not specified, compute diff between consecutive columns
        if self.pairs is None:
            self.pairs = [(i, i + 1) for i in range(0, self.n_features_ - 1, 2)]

        # Generate output column names
        self._output_columns = []
        for start_idx, end_idx in self.pairs:
            start_col = self._input_columns[start_idx]
            end_col = self._input_columns[end_idx]
            self._output_columns.append(f"{end_col}_minus_{start_col}_{self.unit}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def _convert_timedelta(self, td: timedelta) -> float:
        """Convert timedelta to the specified unit."""
        total_seconds = td.total_seconds()

        if self.unit == "seconds":
            return total_seconds
        elif self.unit == "minutes":
            return total_seconds / 60
        elif self.unit == "hours":
            return total_seconds / 3600
        elif self.unit == "days":
            return total_seconds / 86400
        else:
            return total_seconds

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        n_output_cols = len(self.pairs)
        result = np.zeros((X.shape[0], n_output_cols), dtype=np.float64)

        for col_idx, (start_idx, end_idx) in enumerate(self.pairs):
            for j in range(X.shape[0]):
                start_dt = _to_datetime(X[j, start_idx])
                end_dt = _to_datetime(X[j, end_idx])

                if start_dt is None or end_dt is None:
                    result[j, col_idx] = np.nan
                else:
                    td = end_dt - start_dt
                    result[j, col_idx] = self._convert_timedelta(td)

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {"pairs": self.pairs, "unit": self.unit}


class HolidayFeatures(BaseTransformer):
    """
    Add holiday indicator features.

    Parameters:
        holidays: Dict mapping dates (MMDD format) to holiday names,
                  or list of holiday dates
        country: Country code for built-in holidays (not implemented)
    """

    # Common US holidays (month-day format)
    DEFAULT_HOLIDAYS = {
        "0101": "new_years_day",
        "0704": "independence_day",
        "1225": "christmas",
        "1231": "new_years_eve",
    }

    def __init__(
        self,
        holidays: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.holidays = holidays or self.DEFAULT_HOLIDAYS
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "HolidayFeatures":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # Generate output column names
        self._output_columns = []
        for col in self._input_columns:
            self._output_columns.append(f"{col}_is_holiday")
            for holiday_name in set(self.holidays.values()):
                self._output_columns.append(f"{col}_is_{holiday_name}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        holiday_names = list(set(self.holidays.values()))
        n_holiday_features = 1 + len(holiday_names)  # is_holiday + individual holidays
        n_output_cols = self.n_features_ * n_holiday_features
        result = np.zeros((X.shape[0], n_output_cols), dtype=np.float64)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                dt = _to_datetime(X[j, i])
                if dt is None:
                    continue

                date_key = f"{dt.month:02d}{dt.day:02d}"
                base_idx = i * n_holiday_features

                if date_key in self.holidays:
                    result[j, base_idx] = 1.0  # is_holiday
                    holiday_name = self.holidays[date_key]
                    if holiday_name in holiday_names:
                        holiday_idx = holiday_names.index(holiday_name)
                        result[j, base_idx + 1 + holiday_idx] = 1.0

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {"holidays": self.holidays}


class TimeZoneConverter(BaseTransformer):
    """
    Convert datetime features between time zones.

    Parameters:
        from_tz: Source timezone (e.g., 'UTC', 'US/Eastern')
        to_tz: Target timezone
    """

    def __init__(
        self,
        from_tz: str = "UTC",
        to_tz: str = "UTC",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.from_tz = from_tz
        self.to_tz = to_tz
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "TimeZoneConverter":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = [f"{col}_{self.to_tz}" for col in self._input_columns]

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.empty((X.shape[0], self.n_features_), dtype=object)

        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            # Fallback for Python < 3.9
            from datetime import timezone
            # Simple offset-based conversion for UTC
            if self.from_tz == "UTC" and self.to_tz == "UTC":
                return X
            else:
                # Return unchanged if timezone conversion not available
                return X

        from_zone = ZoneInfo(self.from_tz)
        to_zone = ZoneInfo(self.to_tz)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                dt = _to_datetime(X[j, i])
                if dt is None:
                    result[j, i] = None
                else:
                    # Localize to source timezone
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=from_zone)
                    # Convert to target timezone
                    result[j, i] = dt.astimezone(to_zone)

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {"from_tz": self.from_tz, "to_tz": self.to_tz}
