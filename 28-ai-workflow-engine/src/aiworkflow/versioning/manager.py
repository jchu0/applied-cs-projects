"""Flow versioning and migration management."""

import json
import hashlib
from typing import Any
from datetime import datetime
from dataclasses import dataclass, field

from ..schemas import FlowDefinition


@dataclass
class FlowVersion:
    """Version metadata for a flow."""

    version: str
    flow_hash: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    parent_version: str = None
    changes: list[str] = field(default_factory=list)


class FlowVersionManager:
    """Manages flow versions and migrations."""

    def __init__(self, storage_backend=None):
        """Initialize version manager.

        Args:
            storage_backend: Storage for versions (default: in-memory)
        """
        self._versions: dict[str, dict[str, FlowVersion]] = {}
        self._flows: dict[str, dict[str, FlowDefinition]] = {}
        self.storage = storage_backend

    def save_version(
        self,
        flow: FlowDefinition,
        description: str = ""
    ) -> FlowVersion:
        """Save a new version of a flow.

        Args:
            flow: Flow definition
            description: Version description

        Returns:
            Created version
        """
        flow_name = flow.name
        flow_hash = self._compute_hash(flow)

        # Initialize storage for flow
        if flow_name not in self._versions:
            self._versions[flow_name] = {}
            self._flows[flow_name] = {}

        # Check if this exact version exists
        for existing_version in self._versions[flow_name].values():
            if existing_version.flow_hash == flow_hash:
                return existing_version

        # Get parent version
        parent = None
        if self._versions[flow_name]:
            parent = max(
                self._versions[flow_name].keys(),
                key=lambda v: self._versions[flow_name][v].created_at
            )

        # Create new version
        version = FlowVersion(
            version=flow.version,
            flow_hash=flow_hash,
            description=description,
            parent_version=parent
        )

        self._versions[flow_name][flow.version] = version
        self._flows[flow_name][flow.version] = flow

        return version

    def get_version(
        self,
        flow_name: str,
        version: str = None
    ) -> FlowDefinition:
        """Get a specific flow version.

        Args:
            flow_name: Flow name
            version: Version string (None for latest)

        Returns:
            Flow definition
        """
        if flow_name not in self._flows:
            raise ValueError(f"Flow not found: {flow_name}")

        if version is None:
            # Get latest
            version = self.get_latest_version(flow_name)

        if version not in self._flows[flow_name]:
            raise ValueError(f"Version not found: {flow_name}@{version}")

        return self._flows[flow_name][version]

    def get_latest_version(self, flow_name: str) -> str:
        """Get latest version string for a flow.

        Args:
            flow_name: Flow name

        Returns:
            Version string
        """
        if flow_name not in self._versions:
            raise ValueError(f"Flow not found: {flow_name}")

        return max(
            self._versions[flow_name].keys(),
            key=lambda v: self._versions[flow_name][v].created_at
        )

    def list_versions(self, flow_name: str) -> list[FlowVersion]:
        """List all versions of a flow.

        Args:
            flow_name: Flow name

        Returns:
            List of versions
        """
        if flow_name not in self._versions:
            return []

        versions = list(self._versions[flow_name].values())
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def compare_versions(
        self,
        flow_name: str,
        version_a: str,
        version_b: str
    ) -> dict[str, Any]:
        """Compare two versions of a flow.

        Args:
            flow_name: Flow name
            version_a: First version
            version_b: Second version

        Returns:
            Comparison result
        """
        flow_a = self.get_version(flow_name, version_a)
        flow_b = self.get_version(flow_name, version_b)

        # Compare nodes
        nodes_a = {n.id for n in flow_a.nodes}
        nodes_b = {n.id for n in flow_b.nodes}

        added_nodes = nodes_b - nodes_a
        removed_nodes = nodes_a - nodes_b
        common_nodes = nodes_a & nodes_b

        # Check for modified nodes
        modified_nodes = []
        for node_id in common_nodes:
            node_a = flow_a.node_map[node_id]
            node_b = flow_b.node_map[node_id]

            if self._nodes_differ(node_a, node_b):
                modified_nodes.append(node_id)

        return {
            "added_nodes": list(added_nodes),
            "removed_nodes": list(removed_nodes),
            "modified_nodes": modified_nodes,
            "version_a": version_a,
            "version_b": version_b
        }

    def _compute_hash(self, flow: FlowDefinition) -> str:
        """Compute content hash for flow.

        Args:
            flow: Flow definition

        Returns:
            Hash string
        """
        # Create deterministic representation
        content = {
            "name": flow.name,
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "config": n.config if isinstance(n.config, dict) else {
                        "prompt_template": n.config.prompt_template,
                        "model": n.config.model,
                        "temperature": n.config.temperature,
                        "max_tokens": n.config.max_tokens
                    },
                    "inputs": n.inputs,
                    "dependencies": n.dependencies
                }
                for n in sorted(flow.nodes, key=lambda n: n.id)
            ],
            "outputs": flow.outputs
        }

        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:12]

    def _nodes_differ(self, node_a, node_b) -> bool:
        """Check if two nodes differ.

        Args:
            node_a: First node
            node_b: Second node

        Returns:
            True if nodes differ
        """
        if node_a.type != node_b.type:
            return True
        if node_a.config != node_b.config:
            return True
        if node_a.inputs != node_b.inputs:
            return True
        if node_a.dependencies != node_b.dependencies:
            return True
        return False


class MigrationManager:
    """Manages flow migrations between versions."""

    def __init__(self):
        self._migrations: dict[tuple[str, str], callable] = {}

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migration_fn: callable
    ):
        """Register a migration function.

        Args:
            from_version: Source version
            to_version: Target version
            migration_fn: Migration function
        """
        self._migrations[(from_version, to_version)] = migration_fn

    def migrate(
        self,
        flow: FlowDefinition,
        target_version: str
    ) -> FlowDefinition:
        """Migrate flow to target version.

        Args:
            flow: Flow to migrate
            target_version: Target version

        Returns:
            Migrated flow
        """
        current = flow.version

        if current == target_version:
            return flow

        # Find migration path
        path = self._find_migration_path(current, target_version)

        if not path:
            raise ValueError(
                f"No migration path from {current} to {target_version}"
            )

        # Apply migrations
        result = flow
        for from_ver, to_ver in path:
            migration = self._migrations[(from_ver, to_ver)]
            result = migration(result)
            result.version = to_ver

        return result

    def _find_migration_path(
        self,
        from_version: str,
        to_version: str
    ) -> list[tuple[str, str]]:
        """Find migration path between versions.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            List of (from, to) tuples
        """
        # Simple BFS for migration path
        from collections import deque

        queue = deque([(from_version, [])])
        visited = {from_version}

        while queue:
            current, path = queue.popleft()

            if current == to_version:
                return path

            # Find next versions
            for (f, t), _ in self._migrations.items():
                if f == current and t not in visited:
                    visited.add(t)
                    queue.append((t, path + [(f, t)]))

        return []
