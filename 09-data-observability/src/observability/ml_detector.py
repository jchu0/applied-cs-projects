"""ML-based anomaly detection with Isolation Forest and ensemble methods."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from observability.config import DetectorConfig
from observability.models import (
    Anomaly,
    AnomalyType,
    ColumnStats,
    TableMetadata,
)
from observability.detector import generate_id

logger = logging.getLogger(__name__)

# Optional imports for sklearn
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    IsolationForest = None
    StandardScaler = None


@dataclass
class FeatureVector:
    """Feature vector for ML models."""

    table_id: str
    features: np.ndarray
    feature_names: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DetectionResult:
    """Result from ML detector."""

    is_anomaly: bool
    anomaly_score: float  # Higher = more anomalous
    feature_contributions: Dict[str, float]
    confidence: float


class IsolationForestDetector:
    """Isolation Forest based anomaly detector."""

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str = "auto",
        random_state: int = 42,
    ):
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for IsolationForestDetector. "
                "Install with: pip install scikit-learn"
            )

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._training_data: List[np.ndarray] = []

    def _create_model(self) -> IsolationForest:
        """Create a new Isolation Forest model."""
        return IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def add_training_sample(self, features: FeatureVector) -> None:
        """Add a training sample to the buffer."""
        if not self._feature_names:
            self._feature_names = features.feature_names
        elif features.feature_names != self._feature_names:
            raise ValueError("Feature names must match across samples")

        self._training_data.append(features.features)

    def fit(self, min_samples: int = 30) -> bool:
        """Fit the model on collected training data."""
        if len(self._training_data) < min_samples:
            logger.warning(
                f"Not enough training samples ({len(self._training_data)}/{min_samples})"
            )
            return False

        X = np.array(self._training_data)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Fit Isolation Forest
        self._model = self._create_model()
        self._model.fit(X_scaled)
        self._is_fitted = True

        logger.info(
            f"IsolationForest fitted on {len(self._training_data)} samples "
            f"with {len(self._feature_names)} features"
        )
        return True

    def predict(self, features: FeatureVector) -> DetectionResult:
        """Predict if a sample is anomalous."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Validate features
        if features.feature_names != self._feature_names:
            raise ValueError("Feature names must match training data")

        # Scale and predict
        X = features.features.reshape(1, -1)
        X_scaled = self._scaler.transform(X)

        # Get prediction (-1 for anomaly, 1 for normal)
        prediction = self._model.predict(X_scaled)[0]
        is_anomaly = prediction == -1

        # Get anomaly score (negative for anomalies)
        # Convert to 0-1 scale where higher = more anomalous
        raw_score = self._model.score_samples(X_scaled)[0]
        anomaly_score = 1 - (raw_score + 0.5)  # Normalize to ~[0, 1]
        anomaly_score = max(0, min(1, anomaly_score))

        # Calculate feature contributions (approximate)
        contributions = self._calculate_feature_contributions(X_scaled[0])

        # Calculate confidence based on distance from decision boundary
        decision_value = self._model.decision_function(X_scaled)[0]
        confidence = min(1.0, abs(decision_value) * 2)

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            feature_contributions=contributions,
            confidence=confidence,
        )

    def _calculate_feature_contributions(
        self, x_scaled: np.ndarray
    ) -> Dict[str, float]:
        """Calculate approximate feature contributions to anomaly score."""
        contributions = {}

        # Use feature deviation from mean as proxy for contribution
        # In a production system, you might use SHAP values
        for i, name in enumerate(self._feature_names):
            # Deviation from 0 (scaled mean) indicates contribution
            contributions[name] = abs(x_scaled[i])

        # Normalize contributions
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions


class EnsembleDetector:
    """Ensemble anomaly detector combining multiple methods."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._statistical_detector = StatisticalDetector(config)
        self._isolation_forest: Optional[IsolationForestDetector] = None
        self._feature_history: Dict[str, List[FeatureVector]] = {}

        # Initialize Isolation Forest if sklearn available
        if HAS_SKLEARN:
            self._isolation_forest = IsolationForestDetector(
                contamination=0.1,
                n_estimators=100,
            )

    def extract_features(
        self,
        table_id: str,
        metadata: TableMetadata,
        stats: Dict[str, ColumnStats],
    ) -> FeatureVector:
        """Extract features from table metadata and stats."""
        features = []
        feature_names = []

        # Table-level features
        features.append(float(metadata.row_count))
        feature_names.append("row_count")

        features.append(float(metadata.size_bytes))
        feature_names.append("size_bytes")

        features.append(float(len(metadata.columns)))
        feature_names.append("column_count")

        # Aggregate column stats
        null_ratios = [s.null_ratio for s in stats.values()]
        if null_ratios:
            features.append(float(np.mean(null_ratios)))
            feature_names.append("mean_null_ratio")

            features.append(float(np.max(null_ratios)))
            feature_names.append("max_null_ratio")

            features.append(float(np.std(null_ratios)))
            feature_names.append("std_null_ratio")
        else:
            features.extend([0.0, 0.0, 0.0])
            feature_names.extend(["mean_null_ratio", "max_null_ratio", "std_null_ratio"])

        distinct_counts = [s.distinct_count for s in stats.values()]
        if distinct_counts:
            features.append(float(np.mean(distinct_counts)))
            feature_names.append("mean_distinct_count")

            # Cardinality ratio (distinct / row_count)
            if metadata.row_count > 0:
                avg_cardinality = np.mean(distinct_counts) / metadata.row_count
            else:
                avg_cardinality = 0.0
            features.append(avg_cardinality)
            feature_names.append("avg_cardinality_ratio")
        else:
            features.extend([0.0, 0.0])
            feature_names.extend(["mean_distinct_count", "avg_cardinality_ratio"])

        return FeatureVector(
            table_id=table_id,
            features=np.array(features),
            feature_names=feature_names,
        )

    def record_sample(
        self,
        table_id: str,
        metadata: TableMetadata,
        stats: Dict[str, ColumnStats],
    ) -> None:
        """Record a sample for training the ML model."""
        features = self.extract_features(table_id, metadata, stats)

        if table_id not in self._feature_history:
            self._feature_history[table_id] = []

        self._feature_history[table_id].append(features)

        # Keep last 1000 samples per table
        if len(self._feature_history[table_id]) > 1000:
            self._feature_history[table_id] = self._feature_history[table_id][-1000:]

        # Add to Isolation Forest training data
        if self._isolation_forest:
            self._isolation_forest.add_training_sample(features)

    def train_ml_model(self, min_samples: int = 50) -> bool:
        """Train the ML model on collected data."""
        if not self._isolation_forest:
            logger.warning("Isolation Forest not available (sklearn not installed)")
            return False

        return self._isolation_forest.fit(min_samples=min_samples)

    async def detect_anomalies(
        self,
        table_id: str,
        metadata: TableMetadata,
        stats: Dict[str, ColumnStats],
    ) -> List[Anomaly]:
        """Detect anomalies using ensemble of methods."""
        anomalies = []

        # Statistical detection (always available)
        stat_anomalies = await self._statistical_detector.detect_all(
            table_id, metadata, stats
        )
        anomalies.extend(stat_anomalies)

        # ML detection (if model is fitted)
        if self._isolation_forest and self._isolation_forest._is_fitted:
            features = self.extract_features(table_id, metadata, stats)
            ml_result = self._isolation_forest.predict(features)

            if ml_result.is_anomaly and ml_result.confidence > 0.6:
                # Create anomaly from ML detection
                top_contributors = sorted(
                    ml_result.feature_contributions.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]

                anomalies.append(
                    Anomaly(
                        anomaly_id=generate_id(),
                        table_id=table_id,
                        column_name=None,
                        anomaly_type=AnomalyType.DISTRIBUTION,
                        severity="warning" if ml_result.anomaly_score < 0.7 else "critical",
                        detected_at=datetime.now(),
                        metric_value=ml_result.anomaly_score,
                        expected_range=(0.0, 0.5),
                        description=(
                            f"ML-detected anomaly with score {ml_result.anomaly_score:.2f}. "
                            f"Top contributing features: {', '.join(f'{k}' for k, v in top_contributors)}"
                        ),
                        context={
                            "detection_method": "isolation_forest",
                            "anomaly_score": ml_result.anomaly_score,
                            "confidence": ml_result.confidence,
                            "feature_contributions": ml_result.feature_contributions,
                        },
                    )
                )

        # Record sample for future training
        self.record_sample(table_id, metadata, stats)

        return anomalies


class StatisticalDetector:
    """Statistical anomaly detection using Z-score and percentile methods."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._volume_history: Dict[str, List[int]] = {}
        self._null_rate_history: Dict[str, Dict[str, List[float]]] = {}

    def record_volume(self, table_id: str, row_count: int) -> None:
        """Record volume for history."""
        if table_id not in self._volume_history:
            self._volume_history[table_id] = []
        self._volume_history[table_id].append(row_count)
        if len(self._volume_history[table_id]) > 100:
            self._volume_history[table_id] = self._volume_history[table_id][-100:]

    def record_null_rate(
        self, table_id: str, column_name: str, null_ratio: float
    ) -> None:
        """Record null rate for history."""
        if table_id not in self._null_rate_history:
            self._null_rate_history[table_id] = {}
        if column_name not in self._null_rate_history[table_id]:
            self._null_rate_history[table_id][column_name] = []
        self._null_rate_history[table_id][column_name].append(null_ratio)
        if len(self._null_rate_history[table_id][column_name]) > 100:
            self._null_rate_history[table_id][column_name] = (
                self._null_rate_history[table_id][column_name][-100:]
            )

    async def detect_all(
        self,
        table_id: str,
        metadata: TableMetadata,
        stats: Dict[str, ColumnStats],
    ) -> List[Anomaly]:
        """Run all statistical detection methods."""
        anomalies = []

        # Volume anomaly detection
        volume_anomaly = await self._detect_volume(table_id, metadata.row_count)
        if volume_anomaly:
            anomalies.append(volume_anomaly)

        # Null rate anomalies
        for col_name, col_stats in stats.items():
            null_anomaly = await self._detect_null_rate(
                table_id, col_name, col_stats.null_ratio
            )
            if null_anomaly:
                anomalies.append(null_anomaly)

        # Update history
        self.record_volume(table_id, metadata.row_count)
        for col_name, col_stats in stats.items():
            self.record_null_rate(table_id, col_name, col_stats.null_ratio)

        return anomalies

    async def _detect_volume(
        self, table_id: str, current_count: int
    ) -> Optional[Anomaly]:
        """Detect volume anomalies using IQR method."""
        history = self._volume_history.get(table_id, [])

        if len(history) < self.config.min_history_points:
            return None

        # Use IQR method (robust to outliers)
        q1 = np.percentile(history, 25)
        q3 = np.percentile(history, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        if current_count < lower_bound or current_count > upper_bound:
            # Calculate z-score for severity
            mean = float(np.mean(history))
            std = float(np.std(history))
            z_score = abs(current_count - mean) / std if std > 0 else 0

            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=None,
                anomaly_type=AnomalyType.VOLUME,
                severity="critical" if z_score > 5 else "warning",
                detected_at=datetime.now(),
                metric_value=float(current_count),
                expected_range=(lower_bound, upper_bound),
                description=(
                    f"Row count {current_count} is outside IQR bounds "
                    f"[{lower_bound:.0f}, {upper_bound:.0f}]"
                ),
                context={
                    "detection_method": "iqr",
                    "z_score": z_score,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                },
            )

        return None

    async def _detect_null_rate(
        self, table_id: str, column_name: str, current_ratio: float
    ) -> Optional[Anomaly]:
        """Detect null rate anomalies using modified Z-score."""
        if table_id not in self._null_rate_history:
            return None
        if column_name not in self._null_rate_history[table_id]:
            return None

        history = self._null_rate_history[table_id][column_name]

        if len(history) < self.config.min_history_points:
            return None

        # Use modified Z-score (robust to outliers)
        median = float(np.median(history))
        mad = float(np.median(np.abs(np.array(history) - median)))

        # MAD-based threshold
        if mad > 0:
            modified_z = 0.6745 * (current_ratio - median) / mad
        else:
            # If MAD is 0, use percentage change
            if median > 0:
                modified_z = (current_ratio - median) / median * 3
            else:
                modified_z = current_ratio * 10 if current_ratio > 0.1 else 0

        if abs(modified_z) > 3.5:  # Modified Z-score threshold
            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=column_name,
                anomaly_type=AnomalyType.NULL_RATE,
                severity="critical" if current_ratio >= 0.5 else "warning",
                detected_at=datetime.now(),
                metric_value=current_ratio,
                expected_range=(0, median + 3 * mad),
                description=(
                    f"Null rate {current_ratio:.1%} for {column_name} "
                    f"significantly deviates from median {median:.1%}"
                ),
                context={
                    "detection_method": "modified_z_score",
                    "modified_z_score": modified_z,
                    "median": median,
                    "mad": mad,
                },
            )

        return None


class AnomalyCorrelator:
    """Correlate anomalies across tables and time."""

    def __init__(self, window_minutes: int = 30):
        self.window_minutes = window_minutes
        self._recent_anomalies: List[Anomaly] = []

    def add_anomaly(self, anomaly: Anomaly) -> None:
        """Add anomaly to correlation buffer."""
        self._recent_anomalies.append(anomaly)

        # Prune old anomalies
        cutoff = datetime.now().timestamp() - (self.window_minutes * 60)
        self._recent_anomalies = [
            a for a in self._recent_anomalies
            if a.detected_at.timestamp() > cutoff
        ]

    def find_correlated(
        self, anomaly: Anomaly, lineage: Dict[str, List[str]]
    ) -> List[Anomaly]:
        """Find anomalies correlated with the given anomaly."""
        correlated = []

        upstream_tables = set(lineage.get("upstream", []))
        downstream_tables = set(lineage.get("downstream", []))

        for other in self._recent_anomalies:
            if other.anomaly_id == anomaly.anomaly_id:
                continue

            # Check for correlation by:
            # 1. Same table
            # 2. Upstream/downstream relationship
            # 3. Similar anomaly type within time window
            if (
                other.table_id == anomaly.table_id
                or other.table_id in upstream_tables
                or other.table_id in downstream_tables
            ):
                correlated.append(other)
            elif (
                other.anomaly_type == anomaly.anomaly_type
                and abs(
                    (anomaly.detected_at - other.detected_at).total_seconds()
                ) < 300  # 5 minute window
            ):
                correlated.append(other)

        return correlated

    def get_incident_groups(self) -> List[List[Anomaly]]:
        """Group related anomalies into potential incidents."""
        if not self._recent_anomalies:
            return []

        # Simple clustering by table and time proximity
        groups: List[List[Anomaly]] = []
        used = set()

        for anomaly in self._recent_anomalies:
            if anomaly.anomaly_id in used:
                continue

            group = [anomaly]
            used.add(anomaly.anomaly_id)

            # Find similar anomalies
            for other in self._recent_anomalies:
                if other.anomaly_id in used:
                    continue

                # Group if same table or close in time with same type
                if (
                    other.table_id == anomaly.table_id
                    or (
                        other.anomaly_type == anomaly.anomaly_type
                        and abs(
                            (anomaly.detected_at - other.detected_at).total_seconds()
                        ) < 600
                    )
                ):
                    group.append(other)
                    used.add(other.anomaly_id)

            if len(group) >= 1:
                groups.append(group)

        # Sort groups by size (largest incidents first)
        groups.sort(key=len, reverse=True)
        return groups
