"""Online feature store implementations."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import hashlib


@dataclass
class OnlineFeatureValue:
    """A single feature value for online serving."""

    value: Any
    timestamp: datetime
    feature_view: str
    feature_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "feature_view": self.feature_view,
            "feature_name": self.feature_name,
        }


class OnlineStore(ABC):
    """
    Abstract base class for online feature stores.

    Online stores are used for:
    - Low-latency feature serving
    - Real-time inference
    - Feature caching
    """

    @abstractmethod
    def write_features(
        self,
        feature_view: str,
        entity_id: Dict[str, Any],
        features: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        ttl: Optional[timedelta] = None,
    ) -> None:
        """
        Write feature values to the online store.

        Parameters:
            feature_view: Name of the feature view
            entity_id: Entity identifier (e.g., {"user_id": 123})
            features: Feature name -> value mapping
            timestamp: Timestamp for the features
            ttl: Time-to-live for the features
        """
        pass

    @abstractmethod
    def read_features(
        self,
        feature_view: str,
        entity_ids: List[Dict[str, Any]],
        feature_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read feature values from the online store.

        Parameters:
            feature_view: Name of the feature view
            entity_ids: List of entity identifiers
            feature_names: Specific features to retrieve (None for all)

        Returns:
            List of feature dictionaries for each entity
        """
        pass

    @abstractmethod
    def delete_features(
        self,
        feature_view: str,
        entity_ids: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Delete feature values.

        Parameters:
            feature_view: Name of the feature view
            entity_ids: Specific entities to delete (None for all)

        Returns:
            Number of entities deleted
        """
        pass

    @abstractmethod
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
        pass


class InMemoryOnlineStore(OnlineStore):
    """
    In-memory online feature store.

    Useful for testing and development.
    """

    def __init__(self, default_ttl: timedelta = timedelta(days=1)):
        self.default_ttl = default_ttl
        self._store: Dict[str, Dict[str, Tuple[Dict[str, Any], datetime, datetime]]] = {}
        # Structure: {feature_view: {entity_key: (features, timestamp, expiry)}}

    def _make_entity_key(self, entity_id: Dict[str, Any]) -> str:
        """Create a unique key for an entity."""
        sorted_items = sorted(entity_id.items())
        return hashlib.md5(
            json.dumps(sorted_items).encode()
        ).hexdigest()

    def write_features(
        self,
        feature_view: str,
        entity_id: Dict[str, Any],
        features: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        ttl: Optional[timedelta] = None,
    ) -> None:
        if feature_view not in self._store:
            self._store[feature_view] = {}

        entity_key = self._make_entity_key(entity_id)
        timestamp = timestamp or datetime.utcnow()
        expiry = timestamp + (ttl or self.default_ttl)

        # Merge with existing features if present
        existing = self._store[feature_view].get(entity_key)
        if existing:
            existing_features, _, _ = existing
            existing_features.update(features)
            features = existing_features

        # Store entity_id with features for retrieval
        features["_entity_id"] = entity_id

        self._store[feature_view][entity_key] = (features, timestamp, expiry)

    def read_features(
        self,
        feature_view: str,
        entity_ids: List[Dict[str, Any]],
        feature_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        now = datetime.utcnow()

        view_store = self._store.get(feature_view, {})

        for entity_id in entity_ids:
            entity_key = self._make_entity_key(entity_id)

            if entity_key in view_store:
                features, timestamp, expiry = view_store[entity_key]

                # Check expiry
                if expiry < now:
                    results.append({})
                    continue

                if feature_names:
                    filtered = {k: v for k, v in features.items()
                               if k in feature_names or k == "_entity_id"}
                    results.append(filtered)
                else:
                    results.append(features.copy())
            else:
                results.append({})

        return results

    def delete_features(
        self,
        feature_view: str,
        entity_ids: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        if feature_view not in self._store:
            return 0

        if entity_ids is None:
            count = len(self._store[feature_view])
            del self._store[feature_view]
            return count

        count = 0
        for entity_id in entity_ids:
            entity_key = self._make_entity_key(entity_id)
            if entity_key in self._store[feature_view]:
                del self._store[feature_view][entity_key]
                count += 1

        return count

    def get_online_features(
        self,
        feature_refs: List[str],
        entity_ids: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        # Parse feature refs
        feature_views: Dict[str, List[str]] = {}
        for ref in feature_refs:
            view_name, feature_name = ref.split(":", 1)
            if view_name not in feature_views:
                feature_views[view_name] = []
            feature_views[view_name].append(feature_name)

        # Determine number of entities
        n_entities = len(next(iter(entity_ids.values())))

        # Build entity ID list
        entity_id_list = []
        for i in range(n_entities):
            entity_id = {k: v[i] for k, v in entity_ids.items()}
            entity_id_list.append(entity_id)

        # Collect features
        result: Dict[str, List[Any]] = {ref: [None] * n_entities for ref in feature_refs}

        for view_name, feature_names in feature_views.items():
            features_list = self.read_features(view_name, entity_id_list, feature_names)

            for i, features in enumerate(features_list):
                for feature_name in feature_names:
                    ref = f"{view_name}:{feature_name}"
                    result[ref][i] = features.get(feature_name)

        return result

    def clear_expired(self) -> int:
        """Clear expired entries."""
        now = datetime.utcnow()
        count = 0

        for feature_view in list(self._store.keys()):
            for entity_key in list(self._store[feature_view].keys()):
                _, _, expiry = self._store[feature_view][entity_key]
                if expiry < now:
                    del self._store[feature_view][entity_key]
                    count += 1

        return count


class RedisOnlineStore(OnlineStore):
    """
    Redis-based online feature store.

    Provides low-latency feature serving with TTL support.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "feature:",
        default_ttl: timedelta = timedelta(days=1),
    ):
        try:
            import redis
            self.redis = redis
        except ImportError:
            raise ImportError("redis is required for RedisOnlineStore")

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
        )
        self.prefix = prefix
        self.default_ttl = default_ttl

    def _make_key(self, feature_view: str, entity_id: Dict[str, Any]) -> str:
        """Create a Redis key for an entity."""
        sorted_items = sorted(entity_id.items())
        entity_str = json.dumps(sorted_items)
        entity_hash = hashlib.md5(entity_str.encode()).hexdigest()[:12]
        return f"{self.prefix}{feature_view}:{entity_hash}"

    def _serialize_value(self, value: Any) -> str:
        """Serialize a value for Redis storage."""
        return json.dumps(value)

    def _deserialize_value(self, value: str) -> Any:
        """Deserialize a value from Redis storage."""
        if value is None:
            return None
        return json.loads(value)

    def write_features(
        self,
        feature_view: str,
        entity_id: Dict[str, Any],
        features: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        ttl: Optional[timedelta] = None,
    ) -> None:
        key = self._make_key(feature_view, entity_id)
        timestamp = timestamp or datetime.utcnow()
        ttl = ttl or self.default_ttl

        # Prepare data
        data = {
            "_entity_id": self._serialize_value(entity_id),
            "_timestamp": timestamp.isoformat(),
        }
        for name, value in features.items():
            data[name] = self._serialize_value(value)

        # Store as hash
        pipeline = self.client.pipeline()
        pipeline.hset(key, mapping=data)
        pipeline.expire(key, int(ttl.total_seconds()))
        pipeline.execute()

    def read_features(
        self,
        feature_view: str,
        entity_ids: List[Dict[str, Any]],
        feature_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        results = []

        # Batch read using pipeline
        pipeline = self.client.pipeline()
        keys = []

        for entity_id in entity_ids:
            key = self._make_key(feature_view, entity_id)
            keys.append(key)
            if feature_names:
                pipeline.hmget(key, *feature_names)
            else:
                pipeline.hgetall(key)

        responses = pipeline.execute()

        for i, response in enumerate(responses):
            if feature_names:
                # hmget returns a list
                if response and any(v is not None for v in response):
                    features = {
                        name: self._deserialize_value(val)
                        for name, val in zip(feature_names, response)
                        if val is not None
                    }
                    results.append(features)
                else:
                    results.append({})
            else:
                # hgetall returns a dict
                if response:
                    features = {
                        k: self._deserialize_value(v)
                        for k, v in response.items()
                        if not k.startswith("_") or k == "_entity_id"
                    }
                    results.append(features)
                else:
                    results.append({})

        return results

    def delete_features(
        self,
        feature_view: str,
        entity_ids: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        if entity_ids is None:
            # Delete all keys for this feature view
            pattern = f"{self.prefix}{feature_view}:*"
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                return self.client.delete(*keys)
            return 0

        count = 0
        keys = []
        for entity_id in entity_ids:
            key = self._make_key(feature_view, entity_id)
            keys.append(key)

        if keys:
            count = self.client.delete(*keys)

        return count

    def get_online_features(
        self,
        feature_refs: List[str],
        entity_ids: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        # Parse feature refs
        feature_views: Dict[str, List[str]] = {}
        for ref in feature_refs:
            view_name, feature_name = ref.split(":", 1)
            if view_name not in feature_views:
                feature_views[view_name] = []
            feature_views[view_name].append(feature_name)

        # Determine number of entities
        n_entities = len(next(iter(entity_ids.values())))

        # Build entity ID list
        entity_id_list = []
        for i in range(n_entities):
            entity_id = {k: v[i] for k, v in entity_ids.items()}
            entity_id_list.append(entity_id)

        # Collect features
        result: Dict[str, List[Any]] = {ref: [None] * n_entities for ref in feature_refs}

        for view_name, feature_names in feature_views.items():
            features_list = self.read_features(view_name, entity_id_list, feature_names)

            for i, features in enumerate(features_list):
                for feature_name in feature_names:
                    ref = f"{view_name}:{feature_name}"
                    result[ref][i] = features.get(feature_name)

        return result

    def ping(self) -> bool:
        """Check if Redis is available."""
        try:
            return self.client.ping()
        except Exception:
            return False

    def close(self) -> None:
        """Close the Redis connection."""
        self.client.close()
