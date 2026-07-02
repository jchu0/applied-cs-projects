"""Offline feature store implementations."""

from __future__ import annotations

import os
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Feature view names (and derived column names) are interpolated into SQL
# statements by DuckDBOfflineStore, so they must be restricted to a safe
# character set to prevent SQL injection.
FEATURE_VIEW_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def validate_feature_view_name(name: Any) -> str:
    """Validate a feature view name against ``^[A-Za-z0-9_]+$``.

    Raises ValueError for anything else, preventing SQL injection through
    view names that get interpolated into table names.
    """
    if not isinstance(name, str) or not FEATURE_VIEW_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid feature view name: {name!r} "
            "(must contain only letters, digits, and underscores)"
        )
    return name


def _validate_sql_column(name: Any) -> str:
    """Validate an entity/feature column name interpolated into SQL."""
    if not isinstance(name, str) or not FEATURE_VIEW_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid column name: {name!r} "
            "(must contain only letters, digits, and underscores)"
        )
    return name


@dataclass
class FeatureData:
    """Container for feature data."""

    entity_ids: Dict[str, List[Any]]
    features: Dict[str, np.ndarray]
    timestamps: Optional[np.ndarray] = None
    feature_view: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = dict(self.entity_ids)
        for name, values in self.features.items():
            result[name] = values.tolist() if isinstance(values, np.ndarray) else values
        if self.timestamps is not None:
            result["_timestamp"] = self.timestamps.tolist()
        return result

    def __len__(self) -> int:
        if self.features:
            first_key = next(iter(self.features))
            return len(self.features[first_key])
        return 0


class OfflineStore(ABC):
    """
    Abstract base class for offline feature stores.

    Offline stores are used for:
    - Training data generation
    - Batch feature computation
    - Historical feature retrieval
    """

    @abstractmethod
    def write_features(
        self,
        feature_view: str,
        data: FeatureData,
        mode: str = "append",
    ) -> None:
        """
        Write feature data to the offline store.

        Parameters:
            feature_view: Name of the feature view
            data: Feature data to write
            mode: Write mode ('append' or 'overwrite')
        """
        pass

    @abstractmethod
    def read_features(
        self,
        feature_view: str,
        entity_ids: Optional[Dict[str, List[Any]]] = None,
        feature_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> FeatureData:
        """
        Read feature data from the offline store.

        Parameters:
            feature_view: Name of the feature view
            entity_ids: Filter by entity IDs
            feature_names: List of features to retrieve
            start_time: Start of time range
            end_time: End of time range

        Returns:
            FeatureData containing the requested features
        """
        pass

    @abstractmethod
    def get_historical_features(
        self,
        entity_df: Dict[str, Any],
        feature_refs: List[str],
        timestamp_column: str = "_timestamp",
    ) -> FeatureData:
        """
        Retrieve historical features with point-in-time correctness.

        Parameters:
            entity_df: Dictionary with entity IDs and timestamps
            feature_refs: List of feature references (view_name:feature_name)
            timestamp_column: Name of the timestamp column in entity_df

        Returns:
            FeatureData with point-in-time correct features
        """
        pass

    @abstractmethod
    def delete_features(
        self,
        feature_view: str,
        entity_ids: Optional[Dict[str, List[Any]]] = None,
        before_time: Optional[datetime] = None,
    ) -> int:
        """
        Delete feature data.

        Parameters:
            feature_view: Name of the feature view
            entity_ids: Specific entity IDs to delete (None for all)
            before_time: Delete data before this time

        Returns:
            Number of rows deleted
        """
        pass

    @abstractmethod
    def list_feature_views(self) -> List[str]:
        """List all feature views in the store."""
        pass


class ParquetOfflineStore(OfflineStore):
    """
    Parquet-based offline feature store.

    Features are stored as partitioned Parquet files.
    """

    def __init__(
        self,
        path: str = "./feature_store",
        partition_by: List[str] = None,
    ):
        if not PYARROW_AVAILABLE:
            raise ImportError("pyarrow is required for ParquetOfflineStore")

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.partition_by = partition_by or []

    def _get_feature_view_path(self, feature_view: str) -> Path:
        """Get path for a feature view."""
        return self.path / feature_view

    def _data_to_table(self, data: FeatureData) -> pa.Table:
        """Convert FeatureData to PyArrow Table."""
        columns = {}

        # Add entity columns
        for key, values in data.entity_ids.items():
            columns[key] = values

        # Add feature columns
        for name, values in data.features.items():
            if isinstance(values, np.ndarray):
                columns[name] = values.tolist()
            else:
                columns[name] = values

        # Add timestamp
        if data.timestamps is not None:
            columns["_timestamp"] = data.timestamps.tolist()
        else:
            columns["_timestamp"] = [datetime.utcnow().isoformat()] * len(data)

        return pa.Table.from_pydict(columns)

    def _table_to_data(self, table: pa.Table, feature_view: str) -> FeatureData:
        """Convert PyArrow Table to FeatureData."""
        entity_ids = {}
        features = {}
        timestamps = None

        for column_name in table.column_names:
            column = table.column(column_name).to_pylist()

            if column_name == "_timestamp":
                timestamps = np.array([
                    datetime.fromisoformat(t) if isinstance(t, str) else t
                    for t in column
                ])
            elif column_name.startswith("_"):
                continue
            elif self._is_entity_column(column_name, feature_view):
                entity_ids[column_name] = column
            else:
                features[column_name] = np.array(column)

        return FeatureData(
            entity_ids=entity_ids,
            features=features,
            timestamps=timestamps,
            feature_view=feature_view,
        )

    def _is_entity_column(self, column_name: str, feature_view: str) -> bool:
        """Check if a column is an entity column."""
        # Simple heuristic - columns ending with _id are likely entity columns
        return column_name.endswith("_id") or column_name == "id"

    def write_features(
        self,
        feature_view: str,
        data: FeatureData,
        mode: str = "append",
    ) -> None:
        view_path = self._get_feature_view_path(feature_view)
        view_path.mkdir(parents=True, exist_ok=True)

        table = self._data_to_table(data)

        if mode == "overwrite":
            # Remove existing data
            if view_path.exists():
                for f in view_path.glob("*.parquet"):
                    f.unlink()

        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        file_path = view_path / f"data_{timestamp}.parquet"

        pq.write_table(table, file_path)

    def read_features(
        self,
        feature_view: str,
        entity_ids: Optional[Dict[str, List[Any]]] = None,
        feature_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> FeatureData:
        view_path = self._get_feature_view_path(feature_view)

        if not view_path.exists():
            return FeatureData(
                entity_ids={},
                features={},
                feature_view=feature_view,
            )

        # Read all parquet files
        tables = []
        for file_path in view_path.glob("*.parquet"):
            table = pq.read_table(file_path)
            tables.append(table)

        if not tables:
            return FeatureData(
                entity_ids={},
                features={},
                feature_view=feature_view,
            )

        combined_table = pa.concat_tables(tables)

        # Apply filters
        if feature_names:
            columns_to_keep = []
            for col in combined_table.column_names:
                if col in feature_names or col.endswith("_id") or col == "_timestamp":
                    columns_to_keep.append(col)
            combined_table = combined_table.select(columns_to_keep)

        data = self._table_to_data(combined_table, feature_view)

        # Filter by entity IDs
        if entity_ids:
            mask = np.ones(len(data), dtype=bool)
            for key, values in entity_ids.items():
                if key in data.entity_ids:
                    col_mask = np.isin(data.entity_ids[key], values)
                    mask &= col_mask

            data.entity_ids = {k: [v for i, v in enumerate(vals) if mask[i]]
                              for k, vals in data.entity_ids.items()}
            data.features = {k: v[mask] for k, v in data.features.items()}
            if data.timestamps is not None:
                data.timestamps = data.timestamps[mask]

        # Filter by time range
        if data.timestamps is not None and (start_time or end_time):
            mask = np.ones(len(data.timestamps), dtype=bool)
            if start_time:
                mask &= data.timestamps >= start_time
            if end_time:
                mask &= data.timestamps <= end_time

            data.entity_ids = {k: [v for i, v in enumerate(vals) if mask[i]]
                              for k, vals in data.entity_ids.items()}
            data.features = {k: v[mask] for k, v in data.features.items()}
            data.timestamps = data.timestamps[mask]

        return data

    def get_historical_features(
        self,
        entity_df: Dict[str, Any],
        feature_refs: List[str],
        timestamp_column: str = "_timestamp",
    ) -> FeatureData:
        # Parse feature refs
        feature_views: Dict[str, List[str]] = {}
        for ref in feature_refs:
            view_name, feature_name = ref.split(":", 1)
            if view_name not in feature_views:
                feature_views[view_name] = []
            feature_views[view_name].append(feature_name)

        # Get entity timestamps
        timestamps = entity_df.get(timestamp_column, [])
        if isinstance(timestamps[0], str):
            timestamps = [datetime.fromisoformat(t) for t in timestamps]

        # Get entity IDs (all columns except timestamp)
        entity_cols = {k: v for k, v in entity_df.items() if k != timestamp_column}

        # Collect features from each view
        result_features = {}
        result_entity_ids = entity_cols.copy()

        for view_name, feature_names in feature_views.items():
            # Read all data for this view
            view_data = self.read_features(view_name, feature_names=feature_names)

            if not view_data.features:
                # Fill with NaN if no data
                for feature_name in feature_names:
                    result_features[f"{view_name}:{feature_name}"] = np.full(
                        len(timestamps), np.nan
                    )
                continue

            # Point-in-time join
            for i, (ts, *entity_values) in enumerate(zip(timestamps, *entity_cols.values())):
                entity_dict = dict(zip(entity_cols.keys(), [[v] for v in entity_values]))

                # Find matching rows before timestamp
                matches = self._find_pit_matches(view_data, entity_dict, ts)

                for feature_name in feature_names:
                    key = f"{view_name}:{feature_name}"
                    if key not in result_features:
                        result_features[key] = np.full(len(timestamps), np.nan)

                    if matches is not None and feature_name in view_data.features:
                        result_features[key][i] = matches.get(feature_name, np.nan)

        return FeatureData(
            entity_ids=result_entity_ids,
            features=result_features,
            timestamps=np.array(timestamps),
            feature_view="historical",
        )

    def _find_pit_matches(
        self,
        data: FeatureData,
        entity_ids: Dict[str, List[Any]],
        timestamp: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Find the most recent feature values before a timestamp."""
        if not data.features:
            return None

        # Find matching entity rows
        mask = np.ones(len(data), dtype=bool)
        for key, values in entity_ids.items():
            if key in data.entity_ids:
                col_mask = np.isin(data.entity_ids[key], values)
                mask &= col_mask

        if not mask.any():
            return None

        # Filter by timestamp
        if data.timestamps is not None:
            time_mask = data.timestamps <= timestamp
            mask &= time_mask

            if not mask.any():
                return None

            # Get the most recent row
            indices = np.where(mask)[0]
            most_recent_idx = indices[np.argmax(data.timestamps[indices])]
        else:
            indices = np.where(mask)[0]
            most_recent_idx = indices[-1]

        return {
            name: values[most_recent_idx]
            for name, values in data.features.items()
        }

    def delete_features(
        self,
        feature_view: str,
        entity_ids: Optional[Dict[str, List[Any]]] = None,
        before_time: Optional[datetime] = None,
    ) -> int:
        view_path = self._get_feature_view_path(feature_view)

        if not view_path.exists():
            return 0

        if entity_ids is None and before_time is None:
            # Delete all files
            count = 0
            for f in view_path.glob("*.parquet"):
                table = pq.read_table(f)
                count += len(table)
                f.unlink()
            return count

        # Read, filter, and rewrite
        data = self.read_features(feature_view)
        original_len = len(data)

        if entity_ids:
            mask = np.ones(original_len, dtype=bool)
            for key, values in entity_ids.items():
                if key in data.entity_ids:
                    mask &= ~np.isin(data.entity_ids[key], values)
        else:
            mask = np.ones(original_len, dtype=bool)

        if before_time and data.timestamps is not None:
            mask &= data.timestamps >= before_time

        # Keep only non-deleted rows
        new_data = FeatureData(
            entity_ids={k: [v for i, v in enumerate(vals) if mask[i]]
                       for k, vals in data.entity_ids.items()},
            features={k: v[mask] for k, v in data.features.items()},
            timestamps=data.timestamps[mask] if data.timestamps is not None else None,
            feature_view=feature_view,
        )

        # Overwrite with filtered data
        self.write_features(feature_view, new_data, mode="overwrite")

        return original_len - len(new_data)

    def list_feature_views(self) -> List[str]:
        views = []
        for item in self.path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                views.append(item.name)
        return views


class DuckDBOfflineStore(OfflineStore):
    """
    DuckDB-based offline feature store.

    Uses DuckDB for efficient analytical queries on feature data.
    """

    def __init__(
        self,
        path: str = "./feature_store.duckdb",
    ):
        try:
            import duckdb
            self.duckdb = duckdb
        except ImportError:
            raise ImportError("duckdb is required for DuckDBOfflineStore")

        self.path = path
        self.conn = duckdb.connect(path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_view VARCHAR,
                feature_name VARCHAR,
                dtype VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (feature_view, feature_name)
            )
        """)

    def _get_table_name(self, feature_view: str) -> str:
        """Get table name for a feature view (validated to prevent SQL injection)."""
        return f"features_{validate_feature_view_name(feature_view)}"

    def _create_table(self, feature_view: str, data: FeatureData) -> None:
        """Create table for a feature view."""
        table_name = self._get_table_name(feature_view)

        columns = []

        # Entity columns
        for key in data.entity_ids.keys():
            columns.append(f"{_validate_sql_column(key)} VARCHAR")

        # Feature columns
        for name, values in data.features.items():
            _validate_sql_column(name)
            if isinstance(values, np.ndarray):
                if np.issubdtype(values.dtype, np.integer):
                    columns.append(f"{name} BIGINT")
                elif np.issubdtype(values.dtype, np.floating):
                    columns.append(f"{name} DOUBLE")
                else:
                    columns.append(f"{name} VARCHAR")
            else:
                columns.append(f"{name} VARCHAR")

        columns.append("_timestamp TIMESTAMP")

        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
        self.conn.execute(create_sql)

    def write_features(
        self,
        feature_view: str,
        data: FeatureData,
        mode: str = "append",
    ) -> None:
        table_name = self._get_table_name(feature_view)

        if mode == "overwrite":
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Create table if needed
        try:
            self.conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
        except Exception:
            self._create_table(feature_view, data)

        # Prepare data for insertion
        rows = []
        n_rows = len(data)

        for i in range(n_rows):
            row = []

            # Entity values
            for values in data.entity_ids.values():
                row.append(values[i])

            # Feature values
            for values in data.features.values():
                row.append(values[i] if isinstance(values, np.ndarray) else values)

            # Timestamp
            if data.timestamps is not None:
                row.append(data.timestamps[i])
            else:
                row.append(datetime.utcnow())

            rows.append(row)

        # Build column names (validated: they are interpolated into SQL)
        columns = [
            _validate_sql_column(c)
            for c in list(data.entity_ids.keys()) + list(data.features.keys())
        ] + ["_timestamp"]
        placeholders = ", ".join(["?" for _ in columns])

        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        self.conn.executemany(insert_sql, rows)

    def read_features(
        self,
        feature_view: str,
        entity_ids: Optional[Dict[str, List[Any]]] = None,
        feature_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> FeatureData:
        table_name = self._get_table_name(feature_view)

        try:
            self.conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
        except Exception:
            return FeatureData(entity_ids={}, features={}, feature_view=feature_view)

        # Build query
        select_cols = "*"
        if feature_names:
            # Get entity columns
            result = self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            entity_cols = [r[1] for r in result if r[1].endswith("_id") or r[1] == "id"]
            select_cols = ", ".join(
                entity_cols
                + [_validate_sql_column(f) for f in feature_names]
                + ["_timestamp"]
            )

        where_clauses = []
        params = []

        if entity_ids:
            for key, values in entity_ids.items():
                placeholders = ", ".join(["?" for _ in values])
                where_clauses.append(f"{_validate_sql_column(key)} IN ({placeholders})")
                params.extend(values)

        if start_time:
            where_clauses.append("_timestamp >= ?")
            params.append(start_time)

        if end_time:
            where_clauses.append("_timestamp <= ?")
            params.append(end_time)

        sql = f"SELECT {select_cols} FROM {table_name}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        result = self.conn.execute(sql, params).fetchall()
        columns = [desc[0] for desc in self.conn.description]

        # Convert to FeatureData
        entity_id_data: Dict[str, List] = {}
        feature_data: Dict[str, List] = {}
        timestamps = []

        for row in result:
            for col_name, value in zip(columns, row):
                if col_name == "_timestamp":
                    timestamps.append(value)
                elif col_name.endswith("_id") or col_name == "id":
                    if col_name not in entity_id_data:
                        entity_id_data[col_name] = []
                    entity_id_data[col_name].append(value)
                else:
                    if col_name not in feature_data:
                        feature_data[col_name] = []
                    feature_data[col_name].append(value)

        return FeatureData(
            entity_ids=entity_id_data,
            features={k: np.array(v) for k, v in feature_data.items()},
            timestamps=np.array(timestamps) if timestamps else None,
            feature_view=feature_view,
        )

    def get_historical_features(
        self,
        entity_df: Dict[str, Any],
        feature_refs: List[str],
        timestamp_column: str = "_timestamp",
    ) -> FeatureData:
        # Similar implementation to ParquetOfflineStore
        # Parse feature refs
        feature_views: Dict[str, List[str]] = {}
        for ref in feature_refs:
            view_name, feature_name = ref.split(":", 1)
            if view_name not in feature_views:
                feature_views[view_name] = []
            feature_views[view_name].append(feature_name)

        timestamps = entity_df.get(timestamp_column, [])
        entity_cols = {k: v for k, v in entity_df.items() if k != timestamp_column}

        result_features: Dict[str, np.ndarray] = {}

        for view_name, feature_names in feature_views.items():
            table_name = self._get_table_name(view_name)

            for i, ts in enumerate(timestamps):
                entity_values = {k: [v[i]] for k, v in entity_cols.items()}

                # Query for point-in-time features
                where_clauses = ["_timestamp <= ?"]
                params = [ts]

                for key, values in entity_values.items():
                    where_clauses.append(f"{_validate_sql_column(key)} = ?")
                    params.extend(values)

                sql = f"""
                    SELECT {', '.join(_validate_sql_column(f) for f in feature_names)}
                    FROM {table_name}
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY _timestamp DESC
                    LIMIT 1
                """

                try:
                    row = self.conn.execute(sql, params).fetchone()
                except Exception:
                    row = None

                for j, feature_name in enumerate(feature_names):
                    key = f"{view_name}:{feature_name}"
                    if key not in result_features:
                        result_features[key] = np.full(len(timestamps), np.nan)

                    if row:
                        result_features[key][i] = row[j]

        return FeatureData(
            entity_ids=entity_cols,
            features=result_features,
            timestamps=np.array(timestamps) if timestamps else None,
            feature_view="historical",
        )

    def delete_features(
        self,
        feature_view: str,
        entity_ids: Optional[Dict[str, List[Any]]] = None,
        before_time: Optional[datetime] = None,
    ) -> int:
        table_name = self._get_table_name(feature_view)

        try:
            count_before = self.conn.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]
        except Exception:
            return 0

        if entity_ids is None and before_time is None:
            self.conn.execute(f"DROP TABLE {table_name}")
            return count_before

        where_clauses = []
        params = []

        if entity_ids:
            for key, values in entity_ids.items():
                placeholders = ", ".join(["?" for _ in values])
                where_clauses.append(f"{_validate_sql_column(key)} IN ({placeholders})")
                params.extend(values)

        if before_time:
            where_clauses.append("_timestamp < ?")
            params.append(before_time)

        sql = f"DELETE FROM {table_name} WHERE {' AND '.join(where_clauses)}"
        self.conn.execute(sql, params)

        count_after = self.conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]

        return count_before - count_after

    def list_feature_views(self) -> List[str]:
        result = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'features_%'"
        ).fetchall()
        return [r[0].replace("features_", "") for r in result]

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
