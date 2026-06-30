"""Anomaly detection engine for data quality."""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from observability.config import DetectorConfig
from observability.models import (
    Anomaly,
    AnomalyType,
    ColumnStats,
    SchemaChange,
    TableMetadata,
)

logger = logging.getLogger(__name__)


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


class AnomalyDetector:
    """ML-based anomaly detection for data quality."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._volume_history: Dict[str, List[int]] = {}
        self._freshness_history: Dict[str, List[float]] = {}
        self._null_rate_history: Dict[str, Dict[str, List[float]]] = {}
        self._distribution_history: Dict[str, Dict[str, List[Dict]]] = {}
        self._schema_history: Dict[str, TableMetadata] = {}

    def record_volume(self, table_id: str, row_count: int) -> None:
        """Record volume metric for history."""
        if table_id not in self._volume_history:
            self._volume_history[table_id] = []
        self._volume_history[table_id].append(row_count)
        # Keep last 100 points
        if len(self._volume_history[table_id]) > 100:
            self._volume_history[table_id] = self._volume_history[table_id][-100:]

    def record_freshness(self, table_id: str, timestamp: float) -> None:
        """Record freshness metric for history."""
        if table_id not in self._freshness_history:
            self._freshness_history[table_id] = []
        self._freshness_history[table_id].append(timestamp)
        if len(self._freshness_history[table_id]) > 100:
            self._freshness_history[table_id] = self._freshness_history[table_id][-100:]

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

    def record_distribution(
        self, table_id: str, column_name: str, stats: ColumnStats
    ) -> None:
        """Record distribution stats for history."""
        if table_id not in self._distribution_history:
            self._distribution_history[table_id] = {}
        if column_name not in self._distribution_history[table_id]:
            self._distribution_history[table_id][column_name] = []

        self._distribution_history[table_id][column_name].append({
            "null_ratio": stats.null_ratio,
            "distinct_count": stats.distinct_count,
            "mean": stats.mean,
            "stddev": stats.stddev,
        })

        if len(self._distribution_history[table_id][column_name]) > 100:
            self._distribution_history[table_id][column_name] = (
                self._distribution_history[table_id][column_name][-100:]
            )

    def record_schema(self, table_id: str, schema: TableMetadata) -> None:
        """Record schema for change detection."""
        self._schema_history[table_id] = schema

    async def detect_volume_anomaly(
        self, table_id: str, current_count: int
    ) -> Optional[Anomaly]:
        """Detect anomalies in row count."""
        history = self._volume_history.get(table_id, [])

        if len(history) < self.config.min_history_points:
            return None

        # Calculate statistics
        mean = float(np.mean(history))
        std = float(np.std(history))

        # Use minimum effective std (100% of mean) when std is 0 to still detect large changes
        min_tolerance = mean if mean > 0 else 1
        effective_std = max(std, min_tolerance / self.config.volume_threshold) if std == 0 else std

        # Z-score based detection
        z_score = (current_count - mean) / effective_std

        if abs(z_score) > self.config.volume_threshold:
            severity = "critical" if abs(z_score) > 5 else "warning"

            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=None,
                anomaly_type=AnomalyType.VOLUME,
                severity=severity,
                detected_at=datetime.now(),
                metric_value=float(current_count),
                expected_range=(mean - 3 * std, mean + 3 * std),
                description=(
                    f"Row count {current_count} is {abs(z_score):.1f} "
                    f"standard deviations from expected {mean:.0f}"
                ),
                context={
                    "z_score": z_score,
                    "historical_mean": mean,
                    "historical_std": std,
                    "history": history[-10:],
                },
            )

        return None

    async def detect_freshness_anomaly(
        self, table_id: str, last_updated: datetime
    ) -> Optional[Anomaly]:
        """Detect anomalies in data freshness."""
        history = self._freshness_history.get(table_id, [])

        if len(history) < self.config.min_history_points:
            return None

        # Calculate typical update interval
        intervals = np.diff(history)
        if len(intervals) == 0:
            return None

        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))

        current_interval = (datetime.now() - last_updated).total_seconds()

        threshold = mean_interval + 3 * std_interval
        if current_interval > threshold:
            delay_hours = (current_interval - mean_interval) / 3600

            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=None,
                anomaly_type=AnomalyType.FRESHNESS,
                severity="critical" if delay_hours > 24 else "warning",
                detected_at=datetime.now(),
                metric_value=current_interval,
                expected_range=(0, threshold),
                description=(
                    f"Table has not been updated for {delay_hours:.1f} "
                    f"hours longer than expected"
                ),
                context={
                    "last_updated": last_updated.isoformat(),
                    "expected_interval_hours": mean_interval / 3600,
                    "actual_interval_hours": current_interval / 3600,
                },
            )

        return None

    async def detect_schema_anomaly(
        self, table_id: str, current_schema: TableMetadata
    ) -> Optional[Anomaly]:
        """Detect schema changes."""
        previous_schema = self._schema_history.get(table_id)

        if not previous_schema:
            return None

        changes = self._diff_schemas(previous_schema, current_schema)

        if changes:
            severity = "critical" if any(
                c.change_type in ("column_removed", "type_changed")
                for c in changes
            ) else "warning"

            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=None,
                anomaly_type=AnomalyType.SCHEMA,
                severity=severity,
                detected_at=datetime.now(),
                metric_value=float(len(changes)),
                expected_range=(0, 0),
                description=f"Schema change detected: {len(changes)} modification(s)",
                context={
                    "changes": [
                        {
                            "type": c.change_type,
                            "column": c.column_name,
                            "old": c.old_value,
                            "new": c.new_value,
                        }
                        for c in changes
                    ],
                    "previous_column_count": len(previous_schema.columns),
                    "current_column_count": len(current_schema.columns),
                },
            )

        return None

    def _diff_schemas(
        self, old: TableMetadata, new: TableMetadata
    ) -> List[SchemaChange]:
        """Compare two schemas and return differences."""
        changes = []

        old_cols = {col.name: col for col in old.columns}
        new_cols = {col.name: col for col in new.columns}

        # Check for removed columns
        for name in old_cols:
            if name not in new_cols:
                changes.append(SchemaChange(
                    change_type="column_removed",
                    column_name=name,
                    old_value=old_cols[name].data_type,
                ))

        # Check for added columns
        for name in new_cols:
            if name not in old_cols:
                changes.append(SchemaChange(
                    change_type="column_added",
                    column_name=name,
                    new_value=new_cols[name].data_type,
                ))

        # Check for type changes
        for name in old_cols:
            if name in new_cols:
                if old_cols[name].data_type != new_cols[name].data_type:
                    changes.append(SchemaChange(
                        change_type="type_changed",
                        column_name=name,
                        old_value=old_cols[name].data_type,
                        new_value=new_cols[name].data_type,
                    ))

        return changes

    async def detect_null_rate_anomaly(
        self, table_id: str, column_name: str, current_null_ratio: float
    ) -> Optional[Anomaly]:
        """Detect anomalies in null rate."""
        if table_id not in self._null_rate_history:
            return None
        if column_name not in self._null_rate_history[table_id]:
            return None

        history = self._null_rate_history[table_id][column_name]

        if len(history) < self.config.min_history_points:
            return None

        mean = float(np.mean(history))
        std = float(np.std(history))

        # Check for significant increase in null rate
        # Use a minimum tolerance of 100% of mean when std is very small
        # This allows up to 2x the mean before flagging an anomaly
        min_tolerance = mean if mean > 0 else 0.02
        effective_std = max(std, min_tolerance / self.config.null_rate_threshold)
        threshold = mean + self.config.null_rate_threshold * effective_std
        if current_null_ratio > threshold:
            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=column_name,
                anomaly_type=AnomalyType.NULL_RATE,
                severity="critical" if current_null_ratio >= 0.5 else "warning",
                detected_at=datetime.now(),
                metric_value=current_null_ratio,
                expected_range=(max(0, mean - 3 * std), min(1, mean + 3 * std)),
                description=(
                    f"Null rate for {column_name} increased to "
                    f"{current_null_ratio:.1%} (expected ~{mean:.1%})"
                ),
                context={
                    "historical_mean": mean,
                    "historical_std": std,
                    "increase_factor": (
                        current_null_ratio / mean if mean > 0 else float("inf")
                    ),
                },
            )

        return None

    async def detect_distribution_anomaly(
        self, table_id: str, column_name: str, current_stats: ColumnStats
    ) -> Optional[Anomaly]:
        """Detect anomalies in column value distribution."""
        if table_id not in self._distribution_history:
            return None
        if column_name not in self._distribution_history[table_id]:
            return None

        history = self._distribution_history[table_id][column_name]

        if len(history) < self.config.min_history_points:
            return None

        # Simple statistical approach (would use Isolation Forest in production)
        # Check if current stats are outliers in any dimension

        anomaly_score = 0.0
        details = []

        # Check null ratio
        null_ratios = [h["null_ratio"] for h in history]
        mean_null = float(np.mean(null_ratios))
        std_null = float(np.std(null_ratios))
        if std_null > 0:
            z_null = abs(current_stats.null_ratio - mean_null) / std_null
            if z_null > 2:
                anomaly_score += z_null
                details.append(f"null_ratio z-score: {z_null:.2f}")

        # Check distinct count
        distinct_counts = [h["distinct_count"] for h in history]
        mean_distinct = float(np.mean(distinct_counts))
        std_distinct = float(np.std(distinct_counts))
        if std_distinct > 0:
            z_distinct = abs(current_stats.distinct_count - mean_distinct) / std_distinct
            if z_distinct > 2:
                anomaly_score += z_distinct
                details.append(f"distinct_count z-score: {z_distinct:.2f}")

        # Check mean if numeric
        if current_stats.mean is not None:
            means = [h["mean"] for h in history if h["mean"] is not None]
            if means:
                mean_mean = float(np.mean(means))
                std_mean = float(np.std(means))
                if std_mean > 0:
                    z_mean = abs(current_stats.mean - mean_mean) / std_mean
                    if z_mean > 2:
                        anomaly_score += z_mean
                        details.append(f"mean z-score: {z_mean:.2f}")

        if anomaly_score > 3:  # Threshold for anomaly
            return Anomaly(
                anomaly_id=generate_id(),
                table_id=table_id,
                column_name=column_name,
                anomaly_type=AnomalyType.DISTRIBUTION,
                severity="warning",
                detected_at=datetime.now(),
                metric_value=anomaly_score,
                expected_range=(0, 3),
                description=f"Unusual distribution detected for column {column_name}",
                context={
                    "current_stats": {
                        "null_ratio": current_stats.null_ratio,
                        "distinct_count": current_stats.distinct_count,
                        "mean": current_stats.mean,
                        "stddev": current_stats.stddev,
                    },
                    "anomaly_score": anomaly_score,
                    "details": details,
                },
            )

        return None

    async def detect_all_anomalies(
        self, table_id: str, metadata: TableMetadata, stats: Dict[str, ColumnStats]
    ) -> List[Anomaly]:
        """Run all anomaly detection checks for a table."""
        anomalies = []

        # Volume check
        volume_anomaly = await self.detect_volume_anomaly(
            table_id, metadata.row_count
        )
        if volume_anomaly:
            anomalies.append(volume_anomaly)

        # Freshness check
        freshness_anomaly = await self.detect_freshness_anomaly(
            table_id, metadata.last_modified
        )
        if freshness_anomaly:
            anomalies.append(freshness_anomaly)

        # Schema check
        schema_anomaly = await self.detect_schema_anomaly(table_id, metadata)
        if schema_anomaly:
            anomalies.append(schema_anomaly)

        # Column-level checks
        for col_name, col_stats in stats.items():
            # Null rate check
            null_anomaly = await self.detect_null_rate_anomaly(
                table_id, col_name, col_stats.null_ratio
            )
            if null_anomaly:
                anomalies.append(null_anomaly)

            # Distribution check
            dist_anomaly = await self.detect_distribution_anomaly(
                table_id, col_name, col_stats
            )
            if dist_anomaly:
                anomalies.append(dist_anomaly)

        # Update history
        self.record_volume(table_id, metadata.row_count)
        self.record_freshness(table_id, metadata.last_modified.timestamp())
        self.record_schema(table_id, metadata)
        for col_name, col_stats in stats.items():
            self.record_null_rate(table_id, col_name, col_stats.null_ratio)
            self.record_distribution(table_id, col_name, col_stats)

        return anomalies
