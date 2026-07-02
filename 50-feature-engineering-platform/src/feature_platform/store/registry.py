"""Feature registry for managing feature definitions and versions."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import hashlib

from feature_platform.core.models import Feature, FeatureView, Entity, FeatureSource
from feature_platform.store.offline import validate_feature_view_name


@dataclass
class FeatureVersion:
    """Version information for a feature or feature view."""

    version: str
    created_at: datetime
    description: str = ""
    schema_hash: str = ""
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "schema_hash": self.schema_hash,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureVersion":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class FeatureDefinition:
    """Complete definition of a feature view with metadata."""

    name: str
    features: List[Feature]
    entities: List[Entity]
    source: Optional[FeatureSource] = None
    description: str = ""
    owner: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    versions: List[FeatureVersion] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def get_current_version(self) -> Optional[FeatureVersion]:
        """Get the current active version."""
        for version in reversed(self.versions):
            if version.is_active:
                return version
        return None

    def add_version(self, description: str = "") -> FeatureVersion:
        """Add a new version."""
        version_num = len(self.versions) + 1
        schema_hash = self._compute_schema_hash()

        version = FeatureVersion(
            version=f"v{version_num}",
            created_at=datetime.utcnow(),
            description=description,
            schema_hash=schema_hash,
        )
        self.versions.append(version)
        self.updated_at = datetime.utcnow()
        return version

    def _compute_schema_hash(self) -> str:
        """Compute hash of feature schema."""
        schema_str = json.dumps(
            [{"name": f.name, "dtype": f.dtype.value} for f in self.features],
            sort_keys=True,
        )
        return hashlib.md5(schema_str.encode()).hexdigest()[:8]

    def to_feature_view(self) -> FeatureView:
        """Convert to FeatureView."""
        return FeatureView(
            name=self.name,
            entities=self.entities,
            schema=self.features,
            source=self.source,
            description=self.description,
            owner=self.owner,
            tags=self.tags,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "features": [
                {
                    "name": f.name,
                    "dtype": f.dtype.value,
                    "description": f.description,
                    "nullable": f.nullable,
                }
                for f in self.features
            ],
            "entities": [
                {"name": e.name, "join_keys": e.join_keys, "description": e.description}
                for e in self.entities
            ],
            "source": {
                "source_type": self.source.source_type,
                "path": self.source.path,
            } if self.source else None,
            "description": self.description,
            "owner": self.owner,
            "tags": self.tags,
            "versions": [v.to_dict() for v in self.versions],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureDefinition":
        """Create from dictionary."""
        from feature_platform.core.models import DataType

        features = [
            Feature(
                name=f["name"],
                dtype=DataType(f["dtype"]),
                description=f.get("description", ""),
                nullable=f.get("nullable", True),
            )
            for f in data["features"]
        ]

        entities = [
            Entity(
                name=e["name"],
                join_keys=e["join_keys"],
                description=e.get("description", ""),
            )
            for e in data["entities"]
        ]

        source = None
        if data.get("source"):
            source = FeatureSource(
                source_type=data["source"]["source_type"],
                path=data["source"]["path"],
            )

        versions = [FeatureVersion.from_dict(v) for v in data.get("versions", [])]

        return cls(
            name=data["name"],
            features=features,
            entities=entities,
            source=source,
            description=data.get("description", ""),
            owner=data.get("owner", ""),
            tags=data.get("tags", {}),
            versions=versions,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class FeatureRegistry:
    """
    Registry for managing feature definitions.

    Provides:
    - Feature view registration and discovery
    - Version management
    - Metadata storage
    - Lineage tracking
    """

    def __init__(self, path: str = "./feature_registry"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self._definitions: Dict[str, FeatureDefinition] = {}
        self._entities: Dict[str, Entity] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        # Load feature definitions
        definitions_path = self.path / "definitions"
        if definitions_path.exists():
            for file_path in definitions_path.glob("*.json"):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    definition = FeatureDefinition.from_dict(data)
                    self._definitions[definition.name] = definition

        # Load entities
        entities_path = self.path / "entities.json"
        if entities_path.exists():
            with open(entities_path, "r") as f:
                entities_data = json.load(f)
                for name, data in entities_data.items():
                    self._entities[name] = Entity(
                        name=data["name"],
                        join_keys=data["join_keys"],
                        description=data.get("description", ""),
                    )

    def _save_definition(self, definition: FeatureDefinition) -> None:
        """Save a feature definition to disk."""
        definitions_path = self.path / "definitions"
        definitions_path.mkdir(exist_ok=True)

        file_path = definitions_path / f"{definition.name}.json"
        with open(file_path, "w") as f:
            json.dump(definition.to_dict(), f, indent=2)

    def _save_entities(self) -> None:
        """Save entities to disk."""
        entities_path = self.path / "entities.json"
        entities_data = {
            name: {
                "name": entity.name,
                "join_keys": entity.join_keys,
                "description": entity.description,
            }
            for name, entity in self._entities.items()
        }
        with open(entities_path, "w") as f:
            json.dump(entities_data, f, indent=2)

    def register_entity(self, entity: Entity) -> None:
        """Register an entity."""
        self._entities[entity.name] = entity
        self._save_entities()

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        return self._entities.get(name)

    def list_entities(self) -> List[Entity]:
        """List all registered entities."""
        return list(self._entities.values())

    def register_feature_view(
        self,
        feature_view: FeatureView,
        description: str = "",
    ) -> FeatureDefinition:
        """
        Register a feature view.

        If the feature view already exists, creates a new version.

        The view name is validated (letters, digits, underscores only)
        because offline stores interpolate it into SQL table names.
        """
        validate_feature_view_name(feature_view.name)

        # Register entities first
        for entity in feature_view.entities:
            if entity.name not in self._entities:
                self.register_entity(entity)

        if feature_view.name in self._definitions:
            # Update existing definition
            definition = self._definitions[feature_view.name]
            definition.features = feature_view.schema
            definition.entities = feature_view.entities
            definition.source = feature_view.source
            definition.add_version(description)
        else:
            # Create new definition
            definition = FeatureDefinition(
                name=feature_view.name,
                features=feature_view.schema,
                entities=feature_view.entities,
                source=feature_view.source,
                description=feature_view.description,
                owner=feature_view.owner,
                tags=feature_view.tags,
            )
            definition.add_version(description)
            self._definitions[feature_view.name] = definition

        self._save_definition(definition)
        return definition

    def get_feature_view(self, name: str) -> Optional[FeatureView]:
        """Get a feature view by name."""
        definition = self._definitions.get(name)
        if definition:
            return definition.to_feature_view()
        return None

    def get_definition(self, name: str) -> Optional[FeatureDefinition]:
        """Get a feature definition by name."""
        return self._definitions.get(name)

    def list_feature_views(
        self,
        tags: Optional[Dict[str, str]] = None,
        owner: Optional[str] = None,
    ) -> List[FeatureView]:
        """List all registered feature views with optional filtering."""
        results = []

        for definition in self._definitions.values():
            # Filter by tags
            if tags:
                if not all(
                    definition.tags.get(k) == v for k, v in tags.items()
                ):
                    continue

            # Filter by owner
            if owner and definition.owner != owner:
                continue

            results.append(definition.to_feature_view())

        return results

    def search_features(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for features by name or description."""
        results = []
        query_lower = query.lower()

        for definition in self._definitions.values():
            for feature in definition.features:
                if query_lower in feature.name.lower() or query_lower in feature.description.lower():
                    results.append({
                        "feature_view": definition.name,
                        "feature_name": feature.name,
                        "feature_ref": f"{definition.name}:{feature.name}",
                        "dtype": feature.dtype.value,
                        "description": feature.description,
                    })
                    if len(results) >= limit:
                        return results

        return results

    def delete_feature_view(self, name: str) -> bool:
        """Delete a feature view."""
        if name in self._definitions:
            del self._definitions[name]
            file_path = self.path / "definitions" / f"{name}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    def get_feature_refs(self, feature_view_name: str) -> List[str]:
        """Get all feature references for a feature view."""
        definition = self._definitions.get(feature_view_name)
        if not definition:
            return []
        return [f"{feature_view_name}:{f.name}" for f in definition.features]

    def parse_feature_ref(self, feature_ref: str) -> tuple:
        """Parse a feature reference into (feature_view, feature_name)."""
        if ":" not in feature_ref:
            raise ValueError(f"Invalid feature ref: {feature_ref}")
        parts = feature_ref.split(":", 1)
        return parts[0], parts[1]

    def get_feature_by_ref(self, feature_ref: str) -> Optional[Feature]:
        """Get a feature by its reference."""
        view_name, feature_name = self.parse_feature_ref(feature_ref)
        definition = self._definitions.get(view_name)
        if not definition:
            return None
        for feature in definition.features:
            if feature.name == feature_name:
                return feature
        return None

    def get_lineage(self, feature_view_name: str) -> Dict[str, Any]:
        """Get lineage information for a feature view."""
        definition = self._definitions.get(feature_view_name)
        if not definition:
            return {}

        return {
            "name": definition.name,
            "source": {
                "type": definition.source.source_type,
                "path": definition.source.path,
            } if definition.source else None,
            "entities": [e.name for e in definition.entities],
            "features": [f.name for f in definition.features],
            "versions": [v.version for v in definition.versions],
        }

    def export_registry(self) -> Dict[str, Any]:
        """Export the entire registry as a dictionary."""
        return {
            "entities": {
                name: {
                    "name": e.name,
                    "join_keys": e.join_keys,
                    "description": e.description,
                }
                for name, e in self._entities.items()
            },
            "feature_views": {
                name: d.to_dict() for name, d in self._definitions.items()
            },
        }

    def import_registry(self, data: Dict[str, Any]) -> None:
        """Import registry from a dictionary."""
        # Import entities
        for name, entity_data in data.get("entities", {}).items():
            entity = Entity(
                name=entity_data["name"],
                join_keys=entity_data["join_keys"],
                description=entity_data.get("description", ""),
            )
            self._entities[name] = entity

        # Import feature views
        for name, view_data in data.get("feature_views", {}).items():
            definition = FeatureDefinition.from_dict(view_data)
            self._definitions[name] = definition
            self._save_definition(definition)

        self._save_entities()
