"""DAG builder for workflow execution."""

import copy
from typing import Any

from ..schemas import FlowDefinition, Node, NodeType


class CircularDependencyError(Exception):
    """Error raised when a circular dependency is detected."""
    pass


class DependencySet(set):
    """A set that also compares equal to equivalent lists/tuples."""

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return set(self) == set(other)
        return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return super().__hash__()


class DAGBuilder:
    """Builds and analyzes DAG from flow definition."""

    def __init__(self, flow: FlowDefinition = None):
        """Initialize DAG builder.

        Args:
            flow: Flow definition (optional, can use build() later)
        """
        self.flow = flow
        self.adjacency: dict[str, set[str]] = {}
        self.reverse_adjacency: dict[str, set[str]] = {}
        if flow is not None:
            self._build_graph()

    def build(self, flow: FlowDefinition) -> "DAGBuilder":
        """Build DAG from flow definition.

        Args:
            flow: Flow definition

        Returns:
            Self for chaining

        Raises:
            CircularDependencyError: If cyclic dependency detected
        """
        self.flow = flow
        self.adjacency = {}
        self.reverse_adjacency = {}
        self._build_graph()

        # Check for cycles
        if self._has_cycle():
            raise CircularDependencyError("Circular dependency detected in flow")

        return self

    def _build_graph(self):
        """Build adjacency lists from flow definition."""
        # Initialize
        for node in self.flow.nodes:
            self.adjacency[node.id] = set()
            self.reverse_adjacency[node.id] = set()

        # Add edges from explicit dependencies
        for node in self.flow.nodes:
            for dep in node.dependencies:
                self.adjacency[dep].add(node.id)
                self.reverse_adjacency[node.id].add(dep)

        # Add edges from explicit edge definitions
        for edge in self.flow.edges:
            self.adjacency[edge.from_node].add(edge.to_node)
            self.reverse_adjacency[edge.to_node].add(edge.from_node)

    def _has_cycle(self) -> bool:
        """Check if graph has cycles."""
        visited = set()
        rec_stack = set()

        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            for neighbor in self.adjacency.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node_id)
            return False

        for node_id in self.adjacency:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        return False

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID.

        Args:
            node_id: Node identifier

        Returns:
            Node or None
        """
        if self.flow:
            return self.flow.node_map.get(node_id)
        return None

    def topological_sort(self) -> list[str]:
        """Get nodes in topological order.

        Returns:
            List of node IDs in execution order
        """
        in_degree = {node.id: 0 for node in self.flow.nodes}

        # Calculate in-degrees
        for node_id, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                in_degree[neighbor] += 1

        # Start with nodes having no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for neighbor in self.adjacency.get(node_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def get_execution_levels(self) -> list[list[str]]:
        """Get nodes grouped by execution level.

        Nodes in the same level can be executed in parallel.

        Returns:
            List of levels, each containing node IDs
        """
        levels = []
        remaining = set(node.id for node in self.flow.nodes)
        in_degree = {node.id: len(self.reverse_adjacency.get(node.id, set()))
                     for node in self.flow.nodes}

        while remaining:
            # Find nodes with no remaining dependencies (sorted for determinism)
            current_level = sorted([
                node_id for node_id in remaining
                if in_degree[node_id] == 0
            ])

            if not current_level:
                # Cycle detected or error
                break

            levels.append(current_level)

            # Remove nodes from consideration
            for node_id in current_level:
                remaining.remove(node_id)
                # Decrease in-degree of neighbors
                for neighbor in self.adjacency.get(node_id, []):
                    in_degree[neighbor] -= 1

        return levels

    def get_dependencies(self, node_id: str) -> list[str]:
        """Get all dependencies of a node.

        Args:
            node_id: Node identifier

        Returns:
            List of dependency node IDs
        """
        return DependencySet(self.reverse_adjacency.get(node_id, set()))

    def get_dependents(self, node_id: str) -> set[str]:
        """Get all nodes that depend on this node.

        Args:
            node_id: Node identifier

        Returns:
            Set of dependent node IDs
        """
        return self.adjacency.get(node_id, set())

    def get_ancestors(self, node_id: str) -> set[str]:
        """Get all ancestors (transitive dependencies) of a node.

        Args:
            node_id: Node identifier

        Returns:
            Set of ancestor node IDs
        """
        ancestors = set()
        stack = list(self.reverse_adjacency.get(node_id, set()))

        while stack:
            current = stack.pop()
            if current not in ancestors:
                ancestors.add(current)
                stack.extend(self.reverse_adjacency.get(current, set()))

        return ancestors

    def get_parallel_groups(self) -> list[set[str]]:
        """Get groups of nodes that can be executed in parallel.

        Returns:
            List of sets of node IDs
        """
        levels = self.get_execution_levels()
        return [set(level) for level in levels]

    def can_execute(self, node_id: str, completed: set[str]) -> bool:
        """Check if a node can be executed given completed nodes.

        Args:
            node_id: Node to check
            completed: Set of completed node IDs

        Returns:
            True if node can execute
        """
        deps = self.get_dependencies(node_id)
        return deps.issubset(completed)


class FlowOptimizer:
    """Optimizes flow for execution."""

    def optimize(self, flow: FlowDefinition) -> FlowDefinition:
        """Optimize a flow definition.

        Args:
            flow: Flow to optimize

        Returns:
            Optimized flow
        """
        if not flow.nodes:
            return flow

        # Deep copy to avoid mutating original
        flow = copy.deepcopy(flow)

        # Merge redundant nodes
        flow = self._merge_redundant_nodes(flow)

        # Find parallel opportunities
        flow = self._optimize_parallelism(flow)

        # Mark cacheable nodes
        flow = self._mark_cacheable(flow)

        # Remove dead code
        flow = self._remove_dead_code(flow)

        return flow

    def _optimize_parallelism(self, flow: FlowDefinition) -> FlowDefinition:
        """Mark nodes that can run in parallel."""
        if not flow.nodes:
            return flow

        dag = DAGBuilder(flow)
        levels = dag.get_execution_levels()

        # Store parallel groups in flow metadata
        parallel_groups = [list(level) for level in levels if len(level) > 1]
        if parallel_groups:
            flow.metadata["parallel_groups"] = parallel_groups

        for level in levels:
            for node_id in level:
                node = flow.node_map.get(node_id)
                if node:
                    node.metadata['parallel_level'] = levels.index(level)

        return flow

    def _merge_redundant_nodes(self, flow: FlowDefinition) -> FlowDefinition:
        """Merge redundant DATA nodes with identical configs and no dependencies."""
        if not flow.nodes:
            return flow

        # Only merge DATA nodes with no dependencies and identical configs
        seen = {}
        duplicates = {}
        for node in flow.nodes:
            if node.type != NodeType.DATA or node.dependencies:
                continue
            config = node.config if isinstance(node.config, dict) else {}
            extra = getattr(node.config, 'extra', config)
            key = (node.type.value, str(sorted(extra.items()) if isinstance(extra, dict) else str(extra)))
            if key in seen:
                duplicates[node.id] = seen[key]
            else:
                seen[key] = node.id

        if not duplicates:
            return flow

        # Remap dependencies and edges
        for node in flow.nodes:
            node.dependencies = [
                duplicates.get(d, d) for d in node.dependencies
            ]
        for edge in flow.edges:
            edge.from_node = duplicates.get(edge.from_node, edge.from_node)
            edge.to_node = duplicates.get(edge.to_node, edge.to_node)

        # Remove duplicate nodes
        flow.nodes = [n for n in flow.nodes if n.id not in duplicates]

        return flow

    def _mark_cacheable(self, flow: FlowDefinition) -> FlowDefinition:
        """Mark deterministic expensive nodes as cacheable."""
        for node in flow.nodes:
            config = node.config if isinstance(node.config, dict) else {}
            extra = getattr(node.config, 'extra', config)
            if isinstance(extra, dict):
                if extra.get("deterministic") and extra.get("cost") == "high":
                    node.metadata["cache_enabled"] = True
        return flow

    def _remove_dead_code(self, flow: FlowDefinition) -> FlowDefinition:
        """Remove unreachable nodes."""
        # Find all nodes reachable from inputs
        reachable = set()
        dag = DAGBuilder(flow)

        # Start from nodes with no dependencies
        to_visit = [n.id for n in flow.nodes if not dag.get_dependencies(n.id)]

        while to_visit:
            node_id = to_visit.pop()
            if node_id not in reachable:
                reachable.add(node_id)
                to_visit.extend(dag.get_dependents(node_id))

        # Filter nodes
        flow.nodes = [n for n in flow.nodes if n.id in reachable]
        flow.edges = [e for e in flow.edges
                      if e.from_node in reachable and e.to_node in reachable]

        return flow
