"""Congestion prediction for model routing layer."""

import time
import logging
import numpy as np
from collections import defaultdict
from typing import Optional

from ..schemas import CongestionMetrics

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Rolling window collector of timestamped congestion observations per model."""

    def __init__(self, window_minutes: int = 30):
        self.window_seconds = window_minutes * 60
        self._arrivals: dict[str, list[float]] = defaultdict(list)
        self._completions: dict[str, list[float]] = defaultdict(list)
        self._snapshots: dict[str, list[CongestionMetrics]] = defaultdict(list)

    def record_arrival(self, model: str, timestamp: float = None):
        """Record an arrival event."""
        self._arrivals[model].append(timestamp or time.time())
        self._cleanup(model)

    def record_completion(self, model: str, timestamp: float = None):
        """Record a completion event."""
        self._completions[model].append(timestamp or time.time())
        self._cleanup(model)

    def record_snapshot(self, metrics: CongestionMetrics):
        """Record a congestion snapshot."""
        self._snapshots[metrics.model].append(metrics)
        self._cleanup(metrics.model)

    def get_metrics(self, model: str, window_seconds: float = None) -> list[CongestionMetrics]:
        """Get recent metrics within window."""
        window = window_seconds or self.window_seconds
        cutoff = time.time() - window
        return [m for m in self._snapshots[model] if m.timestamp >= cutoff]

    def get_arrival_rate(self, model: str, window_seconds: float = 60.0) -> float:
        """Compute arrival rate (requests/second) over window."""
        cutoff = time.time() - window_seconds
        arrivals = [t for t in self._arrivals[model] if t >= cutoff]
        if not arrivals or window_seconds <= 0:
            return 0.0
        return len(arrivals) / window_seconds

    def get_service_rate(self, model: str, window_seconds: float = 60.0) -> float:
        """Compute service rate (completions/second) over window."""
        cutoff = time.time() - window_seconds
        completions = [t for t in self._completions[model] if t >= cutoff]
        if not completions or window_seconds <= 0:
            return 0.0
        return len(completions) / window_seconds

    def get_observation_count(self, model: str) -> int:
        """Get number of snapshots available for model."""
        cutoff = time.time() - self.window_seconds
        return len([m for m in self._snapshots[model] if m.timestamp >= cutoff])

    def _cleanup(self, model: str):
        """Remove expired entries outside window."""
        cutoff = time.time() - self.window_seconds
        self._arrivals[model] = [t for t in self._arrivals[model] if t >= cutoff]
        self._completions[model] = [t for t in self._completions[model] if t >= cutoff]
        self._snapshots[model] = [m for m in self._snapshots[model] if m.timestamp >= cutoff]


class CongestionModel:
    """Lightweight numpy neural net with sigmoid output for congestion prediction.

    Architecture: feature_dim -> hidden_dim (ReLU) -> 1 (sigmoid)
    Trained with binary cross-entropy.
    """

    def __init__(self, feature_dim: int = 6, hidden_dim: int = 16):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W1 = np.random.randn(feature_dim, hidden_dim) * np.sqrt(2.0 / (feature_dim + hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / (hidden_dim + 1))
        self.b2 = np.zeros(1)

        self.lr = 0.01

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def predict(self, features: np.ndarray) -> float:
        """Predict congestion probability.

        Args:
            features: Feature vector of shape (feature_dim,) or (batch, feature_dim)

        Returns:
            Probability in [0, 1]
        """
        features = np.atleast_2d(features).astype(np.float64)
        h = np.maximum(0, features @ self.W1 + self.b1)
        logit = h @ self.W2 + self.b2
        prob = self._sigmoid(logit)
        if prob.size == 1:
            return float(prob.flat[0])
        return prob.flatten()

    def train(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Train on a batch with binary cross-entropy.

        Args:
            features: Shape (batch, feature_dim)
            labels: Shape (batch,) with values 0 or 1

        Returns:
            Mean loss
        """
        features = np.atleast_2d(features).astype(np.float64)
        labels = np.atleast_1d(labels).astype(np.float64).reshape(-1, 1)
        batch_size = features.shape[0]

        # Forward
        h = np.maximum(0, features @ self.W1 + self.b1)
        logit = h @ self.W2 + self.b2
        prob = self._sigmoid(logit)

        # Binary cross-entropy loss
        eps = 1e-7
        prob_clipped = np.clip(prob, eps, 1 - eps)
        loss = -np.mean(labels * np.log(prob_clipped) + (1 - labels) * np.log(1 - prob_clipped))

        # Backprop
        # dL/dlogit = (prob - label) / batch_size
        dlogit = (prob - labels) / batch_size

        # Layer 2
        dW2 = h.T @ dlogit
        db2 = np.sum(dlogit, axis=0)
        dh = dlogit @ self.W2.T

        # ReLU derivative
        dh = dh * (h > 0)

        # Layer 1
        dW1 = features.T @ dh
        db1 = np.sum(dh, axis=0)

        # Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return float(loss)


class CongestionPredictor:
    """Predicts congestion and decides whether to throttle.

    Uses MetricsCollector to gather observations and CongestionModel to predict.
    Returns 0.0 when insufficient data (<3 observations).
    """

    MIN_OBSERVATIONS = 3

    def __init__(self, model: CongestionModel, metrics_collector: MetricsCollector):
        self.model = model
        self.metrics_collector = metrics_collector

    def predict_congestion(self, model_name: str, horizon_minutes: float = 5) -> float:
        """Predict congestion probability for a model.

        Args:
            model_name: Model to predict for
            horizon_minutes: Prediction horizon

        Returns:
            Probability [0, 1], or 0.0 if insufficient data
        """
        if self.metrics_collector.get_observation_count(model_name) < self.MIN_OBSERVATIONS:
            return 0.0

        features = self._extract_features(model_name)
        return self.model.predict(features)

    def should_throttle(self, model_name: str) -> bool:
        """Check if model should be throttled (prediction > 0.8)."""
        return self.predict_congestion(model_name) > 0.8

    def train_on_observation(self, model_name: str, was_congested: bool):
        """Online learning: train model on a single observation.

        Args:
            model_name: Model that was observed
            was_congested: Whether congestion was actually observed
        """
        if self.metrics_collector.get_observation_count(model_name) < self.MIN_OBSERVATIONS:
            return

        features = self._extract_features(model_name)
        label = np.array([1.0 if was_congested else 0.0])
        self.model.train(np.atleast_2d(features), label)

    def _extract_features(self, model_name: str) -> np.ndarray:
        """Extract feature vector for congestion prediction.

        Features:
          [0] queue_depth_normalized (latest snapshot / 100)
          [1] queue_depth_trend (slope over recent snapshots)
          [2] arrival_rate
          [3] service_rate
          [4] arrival/service ratio
          [5] avg_latency_normalized (latest / 10000)
        """
        metrics = self.metrics_collector.get_metrics(model_name)
        features = np.zeros(6, dtype=np.float64)

        if not metrics:
            return features

        latest = metrics[-1]
        features[0] = latest.queue_depth / 100.0

        # Queue depth trend (slope)
        if len(metrics) >= 2:
            depths = [m.queue_depth for m in metrics[-5:]]
            if len(depths) >= 2:
                x = np.arange(len(depths), dtype=np.float64)
                y = np.array(depths, dtype=np.float64)
                # Simple linear regression slope
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                denom = np.sum((x - x_mean) ** 2)
                if denom > 0:
                    features[1] = np.sum((x - x_mean) * (y - y_mean)) / denom / 100.0

        features[2] = self.metrics_collector.get_arrival_rate(model_name)
        features[3] = self.metrics_collector.get_service_rate(model_name)

        service = max(features[3], 1e-6)
        features[4] = features[2] / service

        features[5] = latest.avg_latency_ms / 10000.0

        return features
