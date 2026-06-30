"""Main feature store facade."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import numpy as np

from feature_platform.core.config import FeatureStoreConfig, OfflineStoreConfig, OnlineStoreConfig
from feature_platform.core.models import Entity, Feature, FeatureView, FeatureVector, FeatureValue
from feature_platform.store.registry import FeatureRegistry, FeatureDefinition
from feature_platform.store.offline import OfflineStore, ParquetOfflineStore, FeatureData
from feature_platform.store.online import OnlineStore, InMemoryOnlineStore


class FeatureStore:
    """
    Main feature store interface.

    Provides a unified API for:
    - Registering feature views
    - Writing features to offline/online stores
    - Reading features for training and serving
    - Managing feature metadata
    """

    def __init__(
        self,
        config: Optional[FeatureStoreConfig] = None,
        registry: Optional[FeatureRegistry] = None,
        offline_store: Optional[OfflineStore] = None,
        online_store: Optional[OnlineStore] = None,
    ):
        self.config = config or FeatureStoreConfig()

        # Initialize components
        self.registry = registry or FeatureRegistry(
            path=self.config.registry.path
        )

        self.offline_store = offline_store or self._create_offline_store()
        self.online_store = online_store or self._create_online_store()

        # Cache for feature views
        self._feature_view_cache: Dict[str, FeatureView] = {}

    def _create_offline_store(self) -> OfflineStore:
        """Create offline store based on config."""
        config = self.config.offline_store
        return ParquetOfflineStore(path=config.path)

    def _create_online_store(self) -> OnlineStore:
        """Create online store based on config."""
        config = self.config.online_store
        return InMemoryOnlineStore(default_ttl=config.ttl)

    def apply(self, objects: List[Union[Entity, FeatureView]]) -> None:
        """
        Apply feature definitions.

        Registers entities and feature views with the registry.
        """
        for obj in objects:
            if isinstance(obj, Entity):
                self.registry.register_entity(obj)
            elif isinstance(obj, FeatureView):
                self.registry.register_feature_view(obj)
                self._feature_view_cache[obj.name] = obj

    def get_feature_view(self, name: str) -> Optional[FeatureView]:
        """Get a feature view by name."""
        if name in self._feature_view_cache:
            return self._feature_view_cache[name]

        view = self.registry.get_feature_view(name)
        if view:
            self._feature_view_cache[name] = view
        return view

    def list_feature_views(self) -> List[FeatureView]:
        """List all registered feature views."""
        return self.registry.list_feature_views()

    def write_to_offline_store(
        self,
        feature_view: str,
        data: Dict[str, Any],
        timestamp_column: str = "_timestamp",
        mode: str = "append",
    ) -> None:
        """
        Write features to the offline store.

        Parameters:
            feature_view: Name of the feature view
            data: Dictionary with columns of data
            timestamp_column: Name of the timestamp column
            mode: Write mode ('append' or 'overwrite')
        """
        view = self.get_feature_view(feature_view)
        if not view:
            raise ValueError(f"Feature view not found: {feature_view}")

        # Extract entity columns
        entity_keys = view.get_entity_keys()
        entity_ids = {k: data[k] for k in entity_keys if k in data}

        # Extract feature columns
        feature_names = view.get_feature_names()
        features = {}
        for name in feature_names:
            if name in data:
                values = data[name]
                if not isinstance(values, np.ndarray):
                    values = np.array(values)
                features[name] = values

        # Extract timestamps
        timestamps = None
        if timestamp_column in data:
            timestamps = data[timestamp_column]
            if not isinstance(timestamps, np.ndarray):
                timestamps = np.array(timestamps)

        feature_data = FeatureData(
            entity_ids=entity_ids,
            features=features,
            timestamps=timestamps,
            feature_view=feature_view,
        )

        self.offline_store.write_features(feature_view, feature_data, mode)

    def write_to_online_store(
        self,
        feature_view: str,
        entity_id: Dict[str, Any],
        features: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Write features to the online store.

        Parameters:
            feature_view: Name of the feature view
            entity_id: Entity identifier
            features: Feature name -> value mapping
            timestamp: Timestamp for the features
        """
        view = self.get_feature_view(feature_view)
        if not view:
            raise ValueError(f"Feature view not found: {feature_view}")

        self.online_store.write_features(
            feature_view=feature_view,
            entity_id=entity_id,
            features=features,
            timestamp=timestamp,
            ttl=view.ttl,
        )

    def materialize(
        self,
        feature_view: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        Materialize features from offline to online store.

        Parameters:
            feature_view: Name of the feature view
            start_time: Start of time range to materialize
            end_time: End of time range to materialize

        Returns:
            Number of rows materialized
        """
        view = self.get_feature_view(feature_view)
        if not view:
            raise ValueError(f"Feature view not found: {feature_view}")

        # Read from offline store
        data = self.offline_store.read_features(
            feature_view=feature_view,
            start_time=start_time,
            end_time=end_time,
        )

        if not data.features:
            return 0

        # Group by entity and get latest values
        entity_keys = view.get_entity_keys()
        entity_groups: Dict[str, tuple] = {}

        n_rows = len(data)
        for i in range(n_rows):
            entity_id = {k: data.entity_ids[k][i] for k in entity_keys if k in data.entity_ids}
            entity_key = str(sorted(entity_id.items()))

            timestamp = data.timestamps[i] if data.timestamps is not None else datetime.utcnow()

            if entity_key not in entity_groups or timestamp > entity_groups[entity_key][1]:
                features = {name: values[i] for name, values in data.features.items()}
                entity_groups[entity_key] = (entity_id, timestamp, features)

        # Write to online store
        count = 0
        for entity_id, timestamp, features in entity_groups.values():
            self.online_store.write_features(
                feature_view=feature_view,
                entity_id=entity_id,
                features=features,
                timestamp=timestamp,
                ttl=view.ttl,
            )
            count += 1

        return count

    def get_online_features(
        self,
        feature_refs: List[str],
        entity_ids: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        """
        Get features for online serving.

        Parameters:
            feature_refs: List of feature references (view_name:feature_name)
            entity_ids: Dictionary of entity column -> list of values

        Returns:
            Dictionary mapping feature ref to list of values
        """
        return self.online_store.get_online_features(feature_refs, entity_ids)

    def get_historical_features(
        self,
        entity_df: Dict[str, Any],
        feature_refs: List[str],
        timestamp_column: str = "_timestamp",
    ) -> FeatureData:
        """
        Get historical features with point-in-time correctness.

        Parameters:
            entity_df: Dictionary with entity IDs and timestamps
            feature_refs: List of feature references
            timestamp_column: Name of the timestamp column

        Returns:
            FeatureData with historical features
        """
        return self.offline_store.get_historical_features(
            entity_df=entity_df,
            feature_refs=feature_refs,
            timestamp_column=timestamp_column,
        )

    def get_feature_vector(
        self,
        feature_view: str,
        entity_id: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
    ) -> FeatureVector:
        """
        Get a feature vector for a single entity.

        Parameters:
            feature_view: Name of the feature view
            entity_id: Entity identifier
            feature_names: Specific features to retrieve

        Returns:
            FeatureVector containing the features
        """
        view = self.get_feature_view(feature_view)
        if not view:
            raise ValueError(f"Feature view not found: {feature_view}")

        # Try online store first
        features_list = self.online_store.read_features(
            feature_view=feature_view,
            entity_ids=[entity_id],
            feature_names=feature_names,
        )

        if features_list and features_list[0]:
            features = features_list[0]
            feature_values = [
                FeatureValue(
                    name=name,
                    value=value,
                    timestamp=datetime.utcnow(),
                )
                for name, value in features.items()
                if not name.startswith("_")
            ]
            return FeatureVector(
                entity_id=entity_id,
                features=feature_values,
                timestamp=datetime.utcnow(),
            )

        # Fall back to offline store
        data = self.offline_store.read_features(
            feature_view=feature_view,
            entity_ids={k: [v] for k, v in entity_id.items()},
            feature_names=feature_names,
        )

        if data.features:
            feature_values = [
                FeatureValue(
                    name=name,
                    value=values[0] if len(values) > 0 else None,
                    timestamp=data.timestamps[0] if data.timestamps is not None else None,
                )
                for name, values in data.features.items()
            ]
            return FeatureVector(
                entity_id=entity_id,
                features=feature_values,
                timestamp=data.timestamps[0] if data.timestamps is not None and len(data.timestamps) > 0 else None,
            )

        return FeatureVector(entity_id=entity_id, features=[])

    def search_features(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for features by name or description."""
        return self.registry.search_features(query, limit)

    def delete_feature_view(self, name: str) -> bool:
        """Delete a feature view and its data."""
        # Delete from stores
        self.offline_store.delete_features(name)
        self.online_store.delete_features(name)

        # Delete from registry
        result = self.registry.delete_feature_view(name)

        # Clear cache
        if name in self._feature_view_cache:
            del self._feature_view_cache[name]

        return result

    def get_feature_statistics(
        self,
        feature_view: str,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for features in a feature view.

        Returns statistics like count, null_count, mean, std, min, max.
        """
        data = self.offline_store.read_features(
            feature_view=feature_view,
            feature_names=feature_names,
        )

        stats = {}
        for name, values in data.features.items():
            if isinstance(values, np.ndarray):
                values_clean = values[~np.isnan(values)] if np.issubdtype(values.dtype, np.number) else values

                feature_stats = {
                    "count": len(values),
                    "null_count": int(np.sum(np.isnan(values))) if np.issubdtype(values.dtype, np.number) else 0,
                }

                if np.issubdtype(values.dtype, np.number) and len(values_clean) > 0:
                    feature_stats.update({
                        "mean": float(np.mean(values_clean)),
                        "std": float(np.std(values_clean)),
                        "min": float(np.min(values_clean)),
                        "max": float(np.max(values_clean)),
                    })

                stats[name] = feature_stats

        return stats

    def validate_feature_view(
        self,
        feature_view: str,
    ) -> List[str]:
        """
        Validate a feature view configuration.

        Returns list of validation errors.
        """
        errors = []
        view = self.get_feature_view(feature_view)

        if not view:
            return [f"Feature view not found: {feature_view}"]

        # Check entities exist
        for entity in view.entities:
            if not self.registry.get_entity(entity.name):
                errors.append(f"Entity not registered: {entity.name}")

        # Check source
        if view.source:
            source_errors = view.source.validate()
            errors.extend(source_errors)

        return errors
