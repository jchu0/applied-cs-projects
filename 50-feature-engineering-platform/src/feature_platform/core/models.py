"""Core data models for Feature Engineering Platform."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np


class DataType(Enum):
    """Supported data types for features."""

    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    BOOL = "bool"
    DATETIME = "datetime"
    BYTES = "bytes"
    ARRAY_INT32 = "array_int32"
    ARRAY_INT64 = "array_int64"
    ARRAY_FLOAT32 = "array_float32"
    ARRAY_FLOAT64 = "array_float64"
    ARRAY_STRING = "array_string"

    @classmethod
    def from_numpy_dtype(cls, dtype: np.dtype) -> "DataType":
        """Convert numpy dtype to DataType."""
        dtype_map = {
            np.int32: cls.INT32,
            np.int64: cls.INT64,
            np.float32: cls.FLOAT32,
            np.float64: cls.FLOAT64,
            np.bool_: cls.BOOL,
            np.object_: cls.STRING,
        }
        for np_type, data_type in dtype_map.items():
            if np.issubdtype(dtype, np_type):
                return data_type
        return cls.STRING

    def to_numpy_dtype(self) -> np.dtype:
        """Convert DataType to numpy dtype."""
        type_map = {
            DataType.INT32: np.int32,
            DataType.INT64: np.int64,
            DataType.FLOAT32: np.float32,
            DataType.FLOAT64: np.float64,
            DataType.STRING: np.object_,
            DataType.BOOL: np.bool_,
            DataType.DATETIME: "datetime64[ns]",
        }
        return np.dtype(type_map.get(self, np.object_))


@dataclass
class Entity:
    """
    Entity represents an object in the real world that features are associated with.

    Examples: user, product, transaction, session
    """

    name: str
    join_keys: List[str]
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.join_keys:
            raise ValueError("Entity must have at least one join key")

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.join_keys)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.join_keys == other.join_keys


@dataclass
class Feature:
    """
    Feature represents a single feature with its metadata.
    """

    name: str
    dtype: Union[DataType, str]
    description: str = ""
    nullable: bool = True
    default_value: Any = None
    tags: Dict[str, str] = field(default_factory=dict)
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = DataType(self.dtype)

    def __hash__(self) -> int:
        return hash((self.name, self.dtype))

    def validate_value(self, value: Any) -> bool:
        """Validate that a value matches this feature's dtype."""
        if value is None:
            return self.nullable

        try:
            np_dtype = self.dtype.to_numpy_dtype()
            np.array([value], dtype=np_dtype)
            return True
        except (ValueError, TypeError):
            return False


@dataclass
class FeatureSchema:
    """Schema definition for a collection of features."""

    features: List[Feature]
    entity_columns: List[str] = field(default_factory=list)
    timestamp_column: Optional[str] = None

    def __post_init__(self):
        self._feature_map = {f.name: f for f in self.features}

    def get_feature(self, name: str) -> Optional[Feature]:
        """Get a feature by name."""
        return self._feature_map.get(name)

    def get_feature_names(self) -> List[str]:
        """Get all feature names."""
        return list(self._feature_map.keys())

    def validate(self, data: Dict[str, Any]) -> List[str]:
        """Validate data against schema, return list of errors."""
        errors = []
        for feature in self.features:
            if feature.name not in data:
                if not feature.nullable and feature.default_value is None:
                    errors.append(f"Missing required feature: {feature.name}")
            elif not feature.validate_value(data[feature.name]):
                errors.append(
                    f"Invalid value for feature {feature.name}: "
                    f"expected {feature.dtype.value}"
                )
        return errors


@dataclass
class FeatureValue:
    """A single feature value with metadata."""

    name: str
    value: Any
    timestamp: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class FeatureVector:
    """A collection of feature values for an entity."""

    entity_id: Dict[str, Any]
    features: List[FeatureValue]
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with feature name -> value mapping."""
        result = dict(self.entity_id)
        for fv in self.features:
            result[fv.name] = fv.value
        if self.timestamp:
            result["_timestamp"] = self.timestamp
        return result

    def get_value(self, feature_name: str) -> Optional[Any]:
        """Get a specific feature value."""
        for fv in self.features:
            if fv.name == feature_name:
                return fv.value
        return None


@dataclass
class FeatureSource:
    """Source configuration for feature data."""

    source_type: str  # "table", "file", "stream", "api"
    path: str  # Table name, file path, stream topic, or API endpoint
    query: Optional[str] = None
    timestamp_field: Optional[str] = None
    created_timestamp_field: Optional[str] = None
    field_mapping: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate source configuration."""
        errors = []
        if not self.path:
            errors.append("Source path is required")
        if self.source_type not in ("table", "file", "stream", "api"):
            errors.append(f"Invalid source type: {self.source_type}")
        return errors


@dataclass
class FeatureView:
    """
    FeatureView defines a collection of features and how to compute them.

    A FeatureView ties together:
    - The entities the features describe
    - The schema of the features
    - The source of the feature data
    - The transformations to apply
    - The TTL for online serving
    """

    name: str
    entities: List[Entity]
    schema: List[Feature]
    source: Optional[FeatureSource] = None
    ttl: timedelta = field(default_factory=lambda: timedelta(days=1))
    online: bool = True
    offline: bool = True
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    transformations: List[Callable] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureView):
            return False
        return self.name == other.name

    def get_entity_keys(self) -> List[str]:
        """Get all entity join keys."""
        keys = []
        for entity in self.entities:
            keys.extend(entity.join_keys)
        return keys

    def get_feature_names(self) -> List[str]:
        """Get all feature names."""
        return [f.name for f in self.schema]

    def get_feature_refs(self) -> List[str]:
        """Get full feature references (view_name:feature_name)."""
        return [f"{self.name}:{f.name}" for f in self.schema]


@dataclass
class MaterializationConfig:
    """Configuration for feature materialization."""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    incremental: bool = True
    batch_size: int = 10000
    parallelism: int = 4


@dataclass
class FeatureServiceConfig:
    """Configuration for a feature service."""

    name: str
    feature_views: List[str]
    features: List[str] = field(default_factory=list)
    description: str = ""
    owner: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingDataConfig:
    """Configuration for generating training data."""

    entity_df: Any  # DataFrame with entity keys and timestamps
    feature_refs: List[str]
    label_column: Optional[str] = None
    label_source: Optional[FeatureSource] = None
