"""Anomaly detection for real-time streaming data.

This module provides:
- Statistical anomaly detection
- Rolling statistics computation
- Z-score based detection
- MAD (Median Absolute Deviation) detection
- Anomaly alerting with configurable thresholds
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, Generic, List, Optional, Tuple, TypeVar
import math

T = TypeVar("T")


class AnomalyType(Enum):
    """Types of anomalies."""
    POINT = "point"  # Single outlier point
    CONTEXTUAL = "contextual"  # Outlier in specific context
    COLLECTIVE = "collective"  # Group of points forming anomaly
    TREND = "trend"  # Anomalous trend change
    SEASONAL = "seasonal"  # Seasonal pattern violation


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """Represents a detected anomaly."""

    timestamp: datetime
    value: float
    expected_value: float
    anomaly_type: AnomalyType
    score: float  # Anomaly score (e.g., z-score)
    severity: AlertSeverity
    metric_name: str
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def deviation(self) -> float:
        """Deviation from expected value."""
        return abs(self.value - self.expected_value)

    @property
    def deviation_percentage(self) -> float:
        """Deviation as percentage of expected value."""
        if self.expected_value == 0:
            return float('inf') if self.value != 0 else 0.0
        return abs(self.deviation / self.expected_value) * 100


@dataclass
class RollingStatistics:
    """Rolling statistics for a metric."""

    count: int = 0
    sum: float = 0.0
    sum_squares: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')

    # For median calculation
    _values: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    @property
    def mean(self) -> float:
        """Calculate mean."""
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    @property
    def variance(self) -> float:
        """Calculate variance using Welford's online algorithm."""
        if self.count < 2:
            return 0.0
        return (self.sum_squares - (self.sum ** 2) / self.count) / (self.count - 1)

    @property
    def std_dev(self) -> float:
        """Calculate standard deviation."""
        return math.sqrt(self.variance)

    @property
    def median(self) -> float:
        """Calculate median from recent values."""
        if len(self._values) == 0:
            return 0.0
        sorted_values = sorted(self._values)
        n = len(sorted_values)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2
        return sorted_values[mid]

    @property
    def mad(self) -> float:
        """Calculate Median Absolute Deviation."""
        if len(self._values) == 0:
            return 0.0
        med = self.median
        deviations = sorted(abs(v - med) for v in self._values)
        n = len(deviations)
        mid = n // 2
        if n % 2 == 0:
            return (deviations[mid - 1] + deviations[mid]) / 2
        return deviations[mid]

    def update(self, value: float) -> None:
        """Update statistics with a new value."""
        self.count += 1
        self.sum += value
        self.sum_squares += value ** 2
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self._values.append(value)


class StatisticalDetector:
    """Statistical anomaly detector using Z-score and MAD."""

    def __init__(
        self,
        z_threshold: float = 3.0,
        mad_threshold: float = 3.5,
        min_samples: int = 30,
        window_size: int = 1000
    ):
        """Initialize the detector.

        Args:
            z_threshold: Z-score threshold for anomaly detection
            mad_threshold: MAD threshold (scaled by 0.6745 for normal distribution)
            min_samples: Minimum samples before detection starts
            window_size: Size of rolling window for statistics
        """
        self.z_threshold = z_threshold
        self.mad_threshold = mad_threshold
        self.min_samples = min_samples
        self.window_size = window_size
        self._stats: Dict[str, RollingStatistics] = {}

    def get_or_create_stats(self, metric_name: str) -> RollingStatistics:
        """Get or create statistics for a metric."""
        if metric_name not in self._stats:
            self._stats[metric_name] = RollingStatistics(
                _values=deque(maxlen=self.window_size)
            )
        return self._stats[metric_name]

    def detect(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> Optional[AnomalyEvent]:
        """Detect if a value is anomalous.

        Args:
            metric_name: Name of the metric
            value: The value to check
            timestamp: Optional timestamp for the event

        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        timestamp = timestamp or datetime.now()
        stats = self.get_or_create_stats(metric_name)

        # Always update statistics
        old_count = stats.count
        stats.update(value)

        # Need minimum samples before detection
        if old_count < self.min_samples:
            return None

        # Calculate scores
        z_score = self._calculate_z_score(stats, value)
        mad_score = self._calculate_mad_score(stats, value)

        # Check for anomaly
        is_anomaly = False
        score = 0.0
        anomaly_type = AnomalyType.POINT

        if abs(z_score) > self.z_threshold:
            is_anomaly = True
            score = z_score
        elif abs(mad_score) > self.mad_threshold:
            is_anomaly = True
            score = mad_score

        if not is_anomaly:
            return None

        # Determine severity
        severity = self._determine_severity(abs(score))

        return AnomalyEvent(
            timestamp=timestamp,
            value=value,
            expected_value=stats.mean,
            anomaly_type=anomaly_type,
            score=score,
            severity=severity,
            metric_name=metric_name,
            context={
                "z_score": z_score,
                "mad_score": mad_score,
                "mean": stats.mean,
                "std_dev": stats.std_dev,
                "median": stats.median,
                "mad": stats.mad,
            }
        )

    def _calculate_z_score(self, stats: RollingStatistics, value: float) -> float:
        """Calculate Z-score for a value."""
        if stats.std_dev == 0:
            return 0.0
        return (value - stats.mean) / stats.std_dev

    def _calculate_mad_score(self, stats: RollingStatistics, value: float) -> float:
        """Calculate MAD-based score (modified Z-score)."""
        mad = stats.mad
        if mad == 0:
            return 0.0
        # Scale factor for normal distribution
        k = 0.6745
        return (value - stats.median) / (mad / k)

    def _determine_severity(self, score: float) -> AlertSeverity:
        """Determine severity based on anomaly score."""
        if score >= 5.0:
            return AlertSeverity.CRITICAL
        elif score >= 4.0:
            return AlertSeverity.WARNING
        return AlertSeverity.INFO


class ExponentialMovingAverageDetector:
    """Anomaly detector using Exponential Moving Average."""

    def __init__(
        self,
        alpha: float = 0.1,
        threshold_factor: float = 3.0,
        min_samples: int = 10
    ):
        """Initialize the EMA detector.

        Args:
            alpha: Smoothing factor (0 < alpha < 1)
            threshold_factor: Factor to multiply EMA std dev for threshold
            min_samples: Minimum samples before detection
        """
        self.alpha = alpha
        self.threshold_factor = threshold_factor
        self.min_samples = min_samples
        self._ema: Dict[str, float] = {}
        self._ema_variance: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def detect(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> Optional[AnomalyEvent]:
        """Detect anomalies using EMA."""
        timestamp = timestamp or datetime.now()

        # Initialize if needed
        if metric_name not in self._ema:
            self._ema[metric_name] = value
            self._ema_variance[metric_name] = 0.0
            self._counts[metric_name] = 1
            return None

        self._counts[metric_name] += 1

        # Get current values
        ema = self._ema[metric_name]
        ema_var = self._ema_variance[metric_name]

        # Calculate deviation
        deviation = value - ema
        ema_std = math.sqrt(ema_var)

        # Update EMA and variance
        self._ema[metric_name] = ema + self.alpha * deviation
        self._ema_variance[metric_name] = (1 - self.alpha) * (ema_var + self.alpha * deviation ** 2)

        # Check for anomaly
        if self._counts[metric_name] < self.min_samples:
            return None

        if ema_std == 0:
            return None

        score = abs(deviation) / ema_std

        if score < self.threshold_factor:
            return None

        severity = AlertSeverity.WARNING if score < 4.0 else AlertSeverity.CRITICAL

        return AnomalyEvent(
            timestamp=timestamp,
            value=value,
            expected_value=ema,
            anomaly_type=AnomalyType.POINT,
            score=score,
            severity=severity,
            metric_name=metric_name,
            context={
                "ema": ema,
                "ema_std": ema_std,
                "deviation": deviation,
            }
        )


class ThresholdDetector:
    """Simple threshold-based anomaly detector."""

    def __init__(
        self,
        thresholds: Dict[str, Tuple[Optional[float], Optional[float]]]
    ):
        """Initialize with metric thresholds.

        Args:
            thresholds: Dict of metric_name -> (min_threshold, max_threshold)
        """
        self.thresholds = thresholds

    def detect(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> Optional[AnomalyEvent]:
        """Check if value violates thresholds."""
        if metric_name not in self.thresholds:
            return None

        timestamp = timestamp or datetime.now()
        min_thresh, max_thresh = self.thresholds[metric_name]

        violation = None
        if min_thresh is not None and value < min_thresh:
            violation = ("below", min_thresh)
        elif max_thresh is not None and value > max_thresh:
            violation = ("above", max_thresh)

        if violation is None:
            return None

        direction, threshold = violation
        expected = (min_thresh or 0 + (max_thresh or 0)) / 2 if min_thresh and max_thresh else threshold

        return AnomalyEvent(
            timestamp=timestamp,
            value=value,
            expected_value=expected,
            anomaly_type=AnomalyType.POINT,
            score=abs(value - threshold),
            severity=AlertSeverity.WARNING,
            metric_name=metric_name,
            context={
                "violation": direction,
                "threshold": threshold,
            }
        )


class AnomalyAlerter:
    """Manages anomaly alerts with rate limiting and aggregation."""

    def __init__(
        self,
        min_alert_interval_seconds: float = 60.0,
        aggregation_window_seconds: float = 300.0,
        alert_callback: Optional[Callable[[AnomalyEvent], None]] = None
    ):
        """Initialize the alerter.

        Args:
            min_alert_interval_seconds: Minimum time between alerts for same metric
            aggregation_window_seconds: Window for aggregating similar anomalies
            alert_callback: Optional callback for when alerts fire
        """
        self.min_alert_interval = min_alert_interval_seconds
        self.aggregation_window = aggregation_window_seconds
        self.alert_callback = alert_callback
        self._last_alert_time: Dict[str, datetime] = {}
        self._aggregated_events: Dict[str, List[AnomalyEvent]] = {}
        self._alert_history: Deque[AnomalyEvent] = deque(maxlen=1000)

    def process_event(self, event: AnomalyEvent) -> bool:
        """Process an anomaly event, potentially firing an alert.

        Args:
            event: The anomaly event to process

        Returns:
            True if an alert was fired, False otherwise
        """
        key = f"{event.metric_name}:{event.anomaly_type.value}"

        # Check rate limiting
        if key in self._last_alert_time:
            time_since_last = (event.timestamp - self._last_alert_time[key]).total_seconds()
            if time_since_last < self.min_alert_interval:
                # Aggregate instead of firing
                if key not in self._aggregated_events:
                    self._aggregated_events[key] = []
                self._aggregated_events[key].append(event)
                return False

        # Fire alert
        self._last_alert_time[key] = event.timestamp
        self._alert_history.append(event)

        if self.alert_callback:
            self.alert_callback(event)

        # Clear aggregated events for this key
        if key in self._aggregated_events:
            del self._aggregated_events[key]

        return True

    def get_recent_alerts(self, limit: int = 100) -> List[AnomalyEvent]:
        """Get recent alerts."""
        return list(self._alert_history)[-limit:]

    def get_aggregated_counts(self) -> Dict[str, int]:
        """Get counts of aggregated (suppressed) events."""
        return {k: len(v) for k, v in self._aggregated_events.items()}


class StreamingAnomalyDetector:
    """Complete anomaly detection system for streaming data."""

    def __init__(
        self,
        z_threshold: float = 3.0,
        mad_threshold: float = 3.5,
        min_samples: int = 30,
        window_size: int = 1000,
        alert_interval_seconds: float = 60.0,
        alert_callback: Optional[Callable[[AnomalyEvent], None]] = None
    ):
        """Initialize the streaming anomaly detector.

        Args:
            z_threshold: Z-score threshold for statistical detection
            mad_threshold: MAD threshold for robust detection
            min_samples: Minimum samples before detection starts
            window_size: Rolling window size for statistics
            alert_interval_seconds: Minimum interval between alerts
            alert_callback: Optional callback for alerts
        """
        self.stat_detector = StatisticalDetector(
            z_threshold=z_threshold,
            mad_threshold=mad_threshold,
            min_samples=min_samples,
            window_size=window_size
        )
        self.ema_detector = ExponentialMovingAverageDetector()
        self.alerter = AnomalyAlerter(
            min_alert_interval_seconds=alert_interval_seconds,
            alert_callback=alert_callback
        )
        self._threshold_detector: Optional[ThresholdDetector] = None

    def add_thresholds(self, thresholds: Dict[str, Tuple[Optional[float], Optional[float]]]) -> None:
        """Add threshold-based detection for specific metrics."""
        self._threshold_detector = ThresholdDetector(thresholds)

    def process(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> Optional[AnomalyEvent]:
        """Process a metric value and detect anomalies.

        Args:
            metric_name: Name of the metric
            value: The value to check
            timestamp: Optional timestamp

        Returns:
            AnomalyEvent if detected and alert fired, None otherwise
        """
        timestamp = timestamp or datetime.now()

        # Check all detectors
        events = []

        # Statistical detector (primary)
        stat_event = self.stat_detector.detect(metric_name, value, timestamp)
        if stat_event:
            events.append(stat_event)

        # EMA detector (secondary confirmation)
        ema_event = self.ema_detector.detect(metric_name, value, timestamp)
        if ema_event:
            events.append(ema_event)

        # Threshold detector (if configured)
        if self._threshold_detector:
            thresh_event = self._threshold_detector.detect(metric_name, value, timestamp)
            if thresh_event:
                events.append(thresh_event)

        # Return the most severe event
        if not events:
            return None

        # Sort by severity and score
        events.sort(key=lambda e: (e.severity.value, e.score), reverse=True)
        best_event = events[0]

        # Process through alerter
        if self.alerter.process_event(best_event):
            return best_event

        return None

    def get_statistics(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Get current statistics for a metric."""
        stats = self.stat_detector._stats.get(metric_name)
        if not stats:
            return None

        return {
            "count": stats.count,
            "mean": stats.mean,
            "std_dev": stats.std_dev,
            "median": stats.median,
            "mad": stats.mad,
            "min": stats.min_value,
            "max": stats.max_value,
        }

    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {
            name: self.get_statistics(name)
            for name in self.stat_detector._stats.keys()
        }


def create_detector_from_config(config: Dict[str, Any]) -> StreamingAnomalyDetector:
    """Create an anomaly detector from configuration.

    Args:
        config: Configuration dictionary with optional keys:
            - z_threshold: float (default 3.0)
            - mad_threshold: float (default 3.5)
            - min_samples: int (default 30)
            - window_size: int (default 1000)
            - alert_interval_seconds: float (default 60.0)
            - thresholds: Dict[str, Tuple[float, float]]

    Returns:
        Configured StreamingAnomalyDetector
    """
    detector = StreamingAnomalyDetector(
        z_threshold=config.get("z_threshold", 3.0),
        mad_threshold=config.get("mad_threshold", 3.5),
        min_samples=config.get("min_samples", 30),
        window_size=config.get("window_size", 1000),
        alert_interval_seconds=config.get("alert_interval_seconds", 60.0),
    )

    if "thresholds" in config:
        detector.add_thresholds(config["thresholds"])

    return detector
