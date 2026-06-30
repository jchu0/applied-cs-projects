"""Health scoring for data assets."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from observability.models import (
    Anomaly,
    AnomalyType,
    ColumnStats,
    DataHealthScore,
    TableMetadata,
)

logger = logging.getLogger(__name__)


class HealthScorer:
    """Calculate health scores for data assets."""

    def __init__(
        self,
        freshness_threshold_hours: float = 24.0,
        volume_variance_threshold: float = 0.2,
        quality_null_threshold: float = 0.1,
    ):
        self.freshness_threshold_hours = freshness_threshold_hours
        self.volume_variance_threshold = volume_variance_threshold
        self.quality_null_threshold = quality_null_threshold

    def calculate_freshness_score(
        self, table: TableMetadata, expected_interval_hours: float = 24.0
    ) -> float:
        """Calculate freshness score (0-100) for a table."""
        if table.last_modified is None:
            return 0.0

        age_hours = (datetime.now() - table.last_modified).total_seconds() / 3600

        if age_hours <= expected_interval_hours:
            return 100.0
        elif age_hours <= expected_interval_hours * 2:
            ratio = (age_hours - expected_interval_hours) / expected_interval_hours
            return max(0.0, 100.0 - (ratio * 50))
        else:
            # Beyond 2x expected interval, rapid decay
            ratio = min(1.0, (age_hours - expected_interval_hours * 2) / (expected_interval_hours * 5))
            return max(0.0, 50.0 - (ratio * 50))

    def calculate_volume_score(
        self, table: TableMetadata, historical_counts: List[int]
    ) -> float:
        """Calculate volume score (0-100) based on historical comparison."""
        if not historical_counts or len(historical_counts) < 2:
            return 100.0

        import numpy as np

        mean = np.mean(historical_counts)
        std = np.std(historical_counts)

        if mean == 0:
            return 100.0

        # Calculate deviation from mean
        deviation = abs(table.row_count - mean) / mean

        if deviation <= 0.1:  # Within 10%
            return 100.0
        elif deviation <= 0.25:  # Within 25%
            return max(80.0, 100.0 - (deviation * 100))
        elif deviation <= 0.5:
            return max(50.0, 80.0 - ((deviation - 0.25) * 120))
        else:
            return max(0.0, 50.0 - ((deviation - 0.5) * 100))

    def calculate_schema_score(
        self, recent_changes: List[str], days_since_last_change: int = 30
    ) -> float:
        """Calculate schema score (0-100) based on recent changes."""
        if not recent_changes:
            return 100.0

        # Base penalty for each change
        penalty = len(recent_changes) * 5

        # More recent changes are worse
        if days_since_last_change < 7:
            penalty += 20
        elif days_since_last_change < 14:
            penalty += 10

        return max(0.0, 100.0 - penalty)

    def calculate_quality_score(
        self, null_ratio: float, duplicate_ratio: float, invalid_ratio: float
    ) -> float:
        """Calculate quality score (0-100) based on data quality metrics."""
        # Weight each metric
        null_penalty = min(40.0, null_ratio * 100)
        dup_penalty = min(30.0, duplicate_ratio * 200)
        invalid_penalty = min(30.0, invalid_ratio * 300)

        return max(0.0, 100.0 - null_penalty - dup_penalty - invalid_penalty)

    def calculate_overall_score(
        self,
        table: TableMetadata,
        freshness_score: float,
        volume_score: float,
        schema_score: float,
        quality_score: float,
        weights: Optional[Dict[str, float]] = None,
        factors: Optional[List[Dict[str, Any]]] = None,
    ) -> DataHealthScore:
        """Calculate overall health score from component scores."""
        if weights is None:
            weights = {
                "freshness": 0.25,
                "volume": 0.25,
                "schema": 0.20,
                "quality": 0.30,
            }

        overall = (
            freshness_score * weights.get("freshness", 0.25)
            + volume_score * weights.get("volume", 0.25)
            + schema_score * weights.get("schema", 0.20)
            + quality_score * weights.get("quality", 0.30)
        )

        return DataHealthScore(
            table_id=table.table_id,
            overall_score=overall,
            freshness_score=freshness_score,
            volume_score=volume_score,
            schema_score=schema_score,
            quality_score=quality_score,
            calculated_at=datetime.now(),
            factors=factors or [],
        )

    def calculate_trend(self, scores: List[DataHealthScore]) -> Dict[str, Any]:
        """Calculate health score trend from a series of scores."""
        if not scores or len(scores) < 2:
            return {"direction": "stable", "change_rate": 0.0}

        import numpy as np

        # Use scores in input order (assume most recent first)
        values = [s.overall_score for s in scores]
        n = len(values)

        if n < 2:
            return {"direction": "stable", "change_rate": 0.0}

        x = np.arange(n)
        y = np.array(values)

        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]

        # Determine direction based on slope
        # Negative slope = scores declining as index increases (most recent to oldest)
        if slope < -1.0:
            direction = "declining"
        elif slope > 1.0:
            direction = "improving"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "change_rate": float(slope),
            "start_score": values[0],
            "end_score": values[-1],
        }

    def calculate_health_score(
        self,
        table_id: str,
        metadata: TableMetadata,
        stats: Dict[str, ColumnStats],
        recent_anomalies: List[Anomaly],
        volume_history: Optional[List[int]] = None,
    ) -> DataHealthScore:
        """Calculate overall health score for a table."""
        factors = []

        # Calculate individual scores
        freshness_score = self._calculate_freshness_score(
            metadata.last_modified, factors
        )

        volume_score = self._calculate_volume_score(
            metadata.row_count, volume_history, factors
        )

        schema_score = self._calculate_schema_score(
            recent_anomalies, factors
        )

        quality_score = self._calculate_quality_score(
            stats, recent_anomalies, factors
        )

        # Calculate weighted overall score
        weights = {
            "freshness": 0.25,
            "volume": 0.25,
            "schema": 0.20,
            "quality": 0.30,
        }

        overall_score = (
            freshness_score * weights["freshness"]
            + volume_score * weights["volume"]
            + schema_score * weights["schema"]
            + quality_score * weights["quality"]
        )

        return DataHealthScore(
            table_id=table_id,
            overall_score=overall_score,
            freshness_score=freshness_score,
            volume_score=volume_score,
            schema_score=schema_score,
            quality_score=quality_score,
            calculated_at=datetime.now(),
            factors=factors,
        )

    def _calculate_freshness_score(
        self,
        last_modified: datetime,
        factors: List[dict],
    ) -> float:
        """Calculate freshness score (0-100)."""
        age_hours = (datetime.now() - last_modified).total_seconds() / 3600

        if age_hours <= self.freshness_threshold_hours:
            score = 100.0
        elif age_hours <= self.freshness_threshold_hours * 2:
            # Linear decay
            ratio = (age_hours - self.freshness_threshold_hours) / self.freshness_threshold_hours
            score = 100.0 - (ratio * 50)
        else:
            # Severe decay
            ratio = min(1.0, (age_hours - self.freshness_threshold_hours * 2) / (self.freshness_threshold_hours * 2))
            score = max(0.0, 50.0 - (ratio * 50))

        factors.append({
            "type": "freshness",
            "description": f"Last updated {age_hours:.1f} hours ago",
            "impact": "positive" if score >= 80 else "negative",
            "value": age_hours,
        })

        return score

    def _calculate_volume_score(
        self,
        current_count: int,
        history: Optional[List[int]],
        factors: List[dict],
    ) -> float:
        """Calculate volume score (0-100)."""
        if not history or len(history) < 3:
            factors.append({
                "type": "volume",
                "description": "Insufficient history for volume scoring",
                "impact": "neutral",
                "value": current_count,
            })
            return 100.0  # No history, assume healthy

        import numpy as np

        mean = np.mean(history)
        std = np.std(history)

        if std == 0:
            variance = 0.0
        else:
            variance = abs(current_count - mean) / mean if mean > 0 else 0.0

        if variance <= self.volume_variance_threshold:
            score = 100.0
        elif variance <= self.volume_variance_threshold * 2:
            ratio = (variance - self.volume_variance_threshold) / self.volume_variance_threshold
            score = 100.0 - (ratio * 40)
        else:
            score = max(0.0, 60.0 - ((variance - self.volume_variance_threshold * 2) * 100))

        description = (
            f"Row count {current_count} "
            f"({variance * 100:.1f}% variance from mean {mean:.0f})"
        )

        factors.append({
            "type": "volume",
            "description": description,
            "impact": "positive" if score >= 80 else "negative",
            "value": current_count,
        })

        return score

    def _calculate_schema_score(
        self,
        recent_anomalies: List[Anomaly],
        factors: List[dict],
    ) -> float:
        """Calculate schema score (0-100)."""
        schema_anomalies = [
            a for a in recent_anomalies
            if a.anomaly_type == AnomalyType.SCHEMA
        ]

        if not schema_anomalies:
            factors.append({
                "type": "schema",
                "description": "No schema changes detected",
                "impact": "positive",
                "value": 0,
            })
            return 100.0

        # Penalize based on number and severity of schema changes
        penalty = 0.0
        for anomaly in schema_anomalies:
            if anomaly.severity == "critical":
                penalty += 30.0
            elif anomaly.severity == "warning":
                penalty += 15.0
            else:
                penalty += 5.0

        score = max(0.0, 100.0 - penalty)

        factors.append({
            "type": "schema",
            "description": f"{len(schema_anomalies)} schema change(s) detected",
            "impact": "negative",
            "value": len(schema_anomalies),
        })

        return score

    def _calculate_quality_score(
        self,
        stats: Dict[str, ColumnStats],
        recent_anomalies: List[Anomaly],
        factors: List[dict],
    ) -> float:
        """Calculate quality score (0-100)."""
        if not stats:
            return 100.0

        # Check null rates
        high_null_columns = []
        for col_name, col_stats in stats.items():
            if col_stats.null_ratio > self.quality_null_threshold:
                high_null_columns.append((col_name, col_stats.null_ratio))

        # Check for quality-related anomalies
        quality_anomalies = [
            a for a in recent_anomalies
            if a.anomaly_type in (AnomalyType.NULL_RATE, AnomalyType.DISTRIBUTION)
        ]

        # Calculate score
        score = 100.0

        # Penalize for high null rates
        if high_null_columns:
            avg_null_rate = sum(r for _, r in high_null_columns) / len(high_null_columns)
            penalty = min(40.0, avg_null_rate * 100)
            score -= penalty

            factors.append({
                "type": "quality_nulls",
                "description": f"{len(high_null_columns)} column(s) with high null rates",
                "impact": "negative",
                "value": [{"column": c, "null_rate": r} for c, r in high_null_columns],
            })

        # Penalize for anomalies
        if quality_anomalies:
            for anomaly in quality_anomalies:
                if anomaly.severity == "critical":
                    score -= 20.0
                else:
                    score -= 10.0

            factors.append({
                "type": "quality_anomalies",
                "description": f"{len(quality_anomalies)} quality anomaly/anomalies detected",
                "impact": "negative",
                "value": len(quality_anomalies),
            })

        return max(0.0, score)


class HealthMonitor:
    """Monitor health scores over time."""

    def __init__(self, scorer: HealthScorer):
        self.scorer = scorer
        self._health_history: Dict[str, List[DataHealthScore]] = {}

    def record_health(self, score: DataHealthScore) -> None:
        """Record a health score."""
        if score.table_id not in self._health_history:
            self._health_history[score.table_id] = []

        self._health_history[score.table_id].append(score)

        # Keep last 100 scores
        if len(self._health_history[score.table_id]) > 100:
            self._health_history[score.table_id] = (
                self._health_history[score.table_id][-100:]
            )

    def get_health_trend(
        self, table_id: str, hours: int = 24
    ) -> List[DataHealthScore]:
        """Get health scores for the last N hours."""
        if table_id not in self._health_history:
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            score for score in self._health_history[table_id]
            if score.calculated_at >= cutoff
        ]

    def get_degraded_tables(self, threshold: float = 70.0) -> List[str]:
        """Get tables with health score below threshold."""
        degraded = []

        for table_id, history in self._health_history.items():
            if history:
                latest = history[-1]
                if latest.overall_score < threshold:
                    degraded.append(table_id)

        return degraded

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of all health scores."""
        total_tables = len(self._health_history)
        healthy = 0
        warning = 0
        critical = 0

        for history in self._health_history.values():
            if history:
                score = history[-1].overall_score
                if score >= 80:
                    healthy += 1
                elif score >= 50:
                    warning += 1
                else:
                    critical += 1

        return {
            "total_tables": total_tables,
            "healthy": healthy,
            "warning": warning,
            "critical": critical,
            "health_percentage": (healthy / total_tables * 100) if total_tables > 0 else 100,
        }
