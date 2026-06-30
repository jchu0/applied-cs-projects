"""Time-series forecasting for proactive monitoring."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from observability.models import Anomaly, AnomalyType
from observability.detector import generate_id

logger = logging.getLogger(__name__)


@dataclass
class ForecastPoint:
    """A single forecast point."""

    timestamp: datetime
    predicted: float
    lower_bound: float
    upper_bound: float


@dataclass
class Forecast:
    """Forecast result."""

    metric_name: str
    table_id: str
    points: List[ForecastPoint]
    confidence_level: float = 0.95


class MetricForecaster:
    """Forecast metrics for proactive alerting."""

    def __init__(
        self,
        seasonality_period: int = 24,  # hours
        trend_weight: float = 0.3,
        seasonal_weight: float = 0.5,
    ):
        self.seasonality_period = seasonality_period
        self.trend_weight = trend_weight
        self.seasonal_weight = seasonal_weight
        self._history: Dict[str, Dict[str, List[Tuple[datetime, float]]]] = {}

    def record_metric(
        self, table_id: str, metric_name: str, value: float, timestamp: datetime = None
    ) -> None:
        """Record a metric value."""
        if timestamp is None:
            timestamp = datetime.now()

        if table_id not in self._history:
            self._history[table_id] = {}
        if metric_name not in self._history[table_id]:
            self._history[table_id][metric_name] = []

        self._history[table_id][metric_name].append((timestamp, value))

        # Keep last 1000 points
        if len(self._history[table_id][metric_name]) > 1000:
            self._history[table_id][metric_name] = (
                self._history[table_id][metric_name][-1000:]
            )

    def forecast(
        self,
        table_id: str,
        metric_name: str,
        periods: int = 24,
        interval_hours: int = 1,
    ) -> Optional[Forecast]:
        """Generate forecast for a metric."""
        if table_id not in self._history:
            return None
        if metric_name not in self._history[table_id]:
            return None

        history = self._history[table_id][metric_name]
        if len(history) < self.seasonality_period * 2:
            return None

        # Extract values
        values = np.array([v for _, v in history])

        # Simple forecasting using decomposition
        trend = self._calculate_trend(values)
        seasonal = self._calculate_seasonality(values)
        residual_std = self._calculate_residual_std(values, trend, seasonal)

        # Generate forecast points
        points = []
        last_timestamp = history[-1][0]

        for i in range(1, periods + 1):
            future_timestamp = last_timestamp + timedelta(hours=i * interval_hours)

            # Combine trend and seasonality
            trend_value = trend[-1] + (trend[-1] - trend[-2]) * i
            seasonal_index = (len(values) + i) % len(seasonal)
            seasonal_value = seasonal[seasonal_index]

            predicted = trend_value * self.trend_weight + seasonal_value * self.seasonal_weight
            predicted += values[-1] * (1 - self.trend_weight - self.seasonal_weight)

            # Confidence bounds
            z_score = 1.96  # 95% confidence
            lower = predicted - z_score * residual_std * np.sqrt(i)
            upper = predicted + z_score * residual_std * np.sqrt(i)

            points.append(ForecastPoint(
                timestamp=future_timestamp,
                predicted=float(predicted),
                lower_bound=float(lower),
                upper_bound=float(upper),
            ))

        return Forecast(
            metric_name=metric_name,
            table_id=table_id,
            points=points,
        )

    def _calculate_trend(self, values: np.ndarray) -> np.ndarray:
        """Calculate trend component using moving average."""
        window = min(self.seasonality_period, len(values) // 2)
        if window < 2:
            return values

        trend = np.convolve(values, np.ones(window) / window, mode="same")
        return trend

    def _calculate_seasonality(self, values: np.ndarray) -> np.ndarray:
        """Calculate seasonal component."""
        period = min(self.seasonality_period, len(values) // 2)

        # Reshape to periods and calculate average for each position
        n_complete_periods = len(values) // period
        if n_complete_periods < 1:
            return np.zeros(period)

        truncated = values[:n_complete_periods * period]
        reshaped = truncated.reshape(-1, period)
        seasonal = reshaped.mean(axis=0)

        return seasonal

    def _calculate_residual_std(
        self, values: np.ndarray, trend: np.ndarray, seasonal: np.ndarray
    ) -> float:
        """Calculate standard deviation of residuals."""
        # Reconstruct signal
        reconstructed = np.zeros_like(values)
        for i in range(len(values)):
            seasonal_index = i % len(seasonal)
            reconstructed[i] = (
                trend[i] * self.trend_weight
                + seasonal[seasonal_index] * self.seasonal_weight
            )

        residuals = values - reconstructed
        return float(np.std(residuals))

    async def detect_future_anomalies(
        self,
        table_id: str,
        metric_name: str,
        hours_ahead: int = 24,
    ) -> List[Anomaly]:
        """Detect potential future anomalies based on forecast."""
        forecast = self.forecast(table_id, metric_name, periods=hours_ahead)

        if not forecast:
            return []

        # Get current value
        if table_id not in self._history:
            return []
        if metric_name not in self._history[table_id]:
            return []

        history = self._history[table_id][metric_name]
        if not history:
            return []

        current_value = history[-1][1]
        anomalies = []

        # Check if current value is outside forecast bounds
        for point in forecast.points[:3]:  # Check next 3 points
            if current_value < point.lower_bound or current_value > point.upper_bound:
                deviation = abs(
                    current_value - point.predicted
                ) / max(abs(point.predicted), 1)

                anomalies.append(Anomaly(
                    anomaly_id=generate_id(),
                    table_id=table_id,
                    column_name=None,
                    anomaly_type=AnomalyType.VOLUME,
                    severity="warning" if deviation < 0.5 else "critical",
                    detected_at=datetime.now(),
                    metric_value=current_value,
                    expected_range=(point.lower_bound, point.upper_bound),
                    description=(
                        f"Forecasted anomaly: {metric_name} ({current_value:.2f}) "
                        f"trending outside expected range "
                        f"({point.lower_bound:.2f}, {point.upper_bound:.2f})"
                    ),
                    context={
                        "forecast_timestamp": point.timestamp.isoformat(),
                        "predicted_value": point.predicted,
                        "deviation": deviation,
                    },
                ))
                break  # One anomaly per check

        return anomalies

    def get_trend_direction(
        self, table_id: str, metric_name: str, periods: int = 5
    ) -> Optional[str]:
        """Get trend direction: increasing, decreasing, or stable."""
        if table_id not in self._history:
            return None
        if metric_name not in self._history[table_id]:
            return None

        history = self._history[table_id][metric_name]
        if len(history) < periods:
            return None

        recent = [v for _, v in history[-periods:]]

        # Calculate slope
        x = np.arange(len(recent))
        coeffs = np.polyfit(x, recent, 1)
        slope = coeffs[0]

        # Normalize by mean to get relative slope
        mean = np.mean(recent)
        if mean == 0:
            return "stable"

        relative_slope = slope / mean

        if relative_slope > 0.05:
            return "increasing"
        elif relative_slope < -0.05:
            return "decreasing"
        else:
            return "stable"


class AnomalyForecaster:
    """Forecast anomaly likelihood based on patterns."""

    def __init__(self):
        self._anomaly_history: Dict[str, List[Anomaly]] = {}

    def record_anomaly(self, anomaly: Anomaly) -> None:
        """Record an anomaly for pattern learning."""
        if anomaly.table_id not in self._anomaly_history:
            self._anomaly_history[anomaly.table_id] = []

        self._anomaly_history[anomaly.table_id].append(anomaly)

        # Keep last 100 anomalies per table
        if len(self._anomaly_history[anomaly.table_id]) > 100:
            self._anomaly_history[anomaly.table_id] = (
                self._anomaly_history[anomaly.table_id][-100:]
            )

    def predict_anomaly_likelihood(
        self, table_id: str, hours_ahead: int = 24
    ) -> float:
        """Predict likelihood of anomaly in next N hours."""
        if table_id not in self._anomaly_history:
            return 0.0

        anomalies = self._anomaly_history[table_id]
        if len(anomalies) < 3:
            return 0.0

        # Calculate average interval between anomalies
        timestamps = [a.detected_at for a in anomalies]
        intervals = []

        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
            intervals.append(interval)

        if not intervals:
            return 0.0

        avg_interval = np.mean(intervals)

        # Time since last anomaly
        time_since_last = (
            datetime.now() - timestamps[-1]
        ).total_seconds() / 3600

        # Probability increases as we approach average interval
        if avg_interval > 0:
            progress = time_since_last / avg_interval
            # Sigmoid-like probability
            likelihood = 1 / (1 + np.exp(-2 * (progress - 0.5)))
            return float(min(1.0, likelihood))

        return 0.0

    def get_high_risk_tables(self, threshold: float = 0.7) -> List[str]:
        """Get tables with high anomaly likelihood."""
        high_risk = []

        for table_id in self._anomaly_history:
            likelihood = self.predict_anomaly_likelihood(table_id)
            if likelihood >= threshold:
                high_risk.append(table_id)

        return high_risk
