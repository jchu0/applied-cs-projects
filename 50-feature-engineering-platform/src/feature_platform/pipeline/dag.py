"""DAG (Directed Acyclic Graph) for feature pipelines."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import concurrent.futures
import threading
import time


class NodeStatus(Enum):
    """Status of a DAG node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    """
    A node in the DAG representing a computation step.

    Parameters:
        name: Unique name for the node
        func: Function to execute
        inputs: List of input node names
        outputs: List of output names
        config: Configuration parameters
    """

    name: str
    func: Callable
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DAGNode):
            return False
        return self.name == other.name

    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class DAGEdge:
    """An edge connecting two nodes in the DAG."""

    source: str
    target: str
    data_key: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.source, self.target))


class DAG:
    """
    Directed Acyclic Graph for feature pipeline execution.

    Provides:
    - Topological ordering for execution
    - Dependency tracking
    - Parallel execution support
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.nodes: Dict[str, DAGNode] = {}
        self.edges: Set[DAGEdge] = set()
        self._adjacency: Dict[str, Set[str]] = {}
        self._reverse_adjacency: Dict[str, Set[str]] = {}

    def add_node(self, node: DAGNode) -> None:
        """Add a node to the DAG."""
        if node.name in self.nodes:
            raise ValueError(f"Node already exists: {node.name}")

        self.nodes[node.name] = node
        self._adjacency[node.name] = set()
        self._reverse_adjacency[node.name] = set()

        # Add edges from inputs
        for input_name in node.inputs:
            if input_name in self.nodes:
                self.add_edge(input_name, node.name)

    def add_edge(self, source: str, target: str, data_key: Optional[str] = None) -> None:
        """Add an edge between two nodes."""
        if source not in self.nodes:
            raise ValueError(f"Source node not found: {source}")
        if target not in self.nodes:
            raise ValueError(f"Target node not found: {target}")

        edge = DAGEdge(source=source, target=target, data_key=data_key)
        self.edges.add(edge)
        self._adjacency[source].add(target)
        self._reverse_adjacency[target].add(source)

        # Check for cycles
        if self._has_cycle():
            self.edges.discard(edge)
            self._adjacency[source].discard(target)
            self._reverse_adjacency[target].discard(source)
            raise ValueError(f"Adding edge {source} -> {target} would create a cycle")

    def _has_cycle(self) -> bool:
        """Check if the DAG has a cycle using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self._adjacency.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.discard(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def get_upstream(self, node_name: str) -> Set[str]:
        """Get all upstream dependencies of a node."""
        return self._reverse_adjacency.get(node_name, set())

    def get_downstream(self, node_name: str) -> Set[str]:
        """Get all downstream dependents of a node."""
        return self._adjacency.get(node_name, set())

    def get_roots(self) -> List[str]:
        """Get nodes with no upstream dependencies."""
        return [
            name for name, deps in self._reverse_adjacency.items()
            if not deps
        ]

    def get_leaves(self) -> List[str]:
        """Get nodes with no downstream dependents."""
        return [
            name for name, deps in self._adjacency.items()
            if not deps
        ]

    def topological_sort(self) -> List[str]:
        """Get topologically sorted list of node names."""
        in_degree = {name: len(deps) for name, deps in self._reverse_adjacency.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for downstream in self._adjacency.get(node, set()):
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)

        if len(result) != len(self.nodes):
            raise ValueError("DAG has a cycle")

        return result

    def get_execution_levels(self) -> List[List[str]]:
        """
        Get nodes grouped by execution level.

        Nodes at the same level can be executed in parallel.
        """
        levels = []
        remaining = set(self.nodes.keys())
        completed = set()

        while remaining:
            # Find nodes whose dependencies are all completed
            current_level = [
                name for name in remaining
                if self._reverse_adjacency[name].issubset(completed)
            ]

            if not current_level:
                raise ValueError("DAG has a cycle")

            levels.append(current_level)
            completed.update(current_level)
            remaining -= set(current_level)

        return levels

    def reset(self) -> None:
        """Reset all node statuses."""
        for node in self.nodes.values():
            node.status = NodeStatus.PENDING
            node.result = None
            node.error = None
            node.start_time = None
            node.end_time = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert DAG to dictionary."""
        return {
            "name": self.name,
            "nodes": [
                {
                    "name": n.name,
                    "inputs": n.inputs,
                    "outputs": n.outputs,
                    "status": n.status.value,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target}
                for e in self.edges
            ],
        }


class DAGExecutor:
    """
    Executor for running DAG pipelines.

    Supports:
    - Sequential execution
    - Parallel execution with thread pool
    - Error handling and retry
    - Progress tracking
    """

    def __init__(
        self,
        dag: DAG,
        max_workers: int = 4,
        retry_count: int = 0,
        retry_delay: float = 1.0,
    ):
        self.dag = dag
        self.max_workers = max_workers
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self._results: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def execute(
        self,
        initial_data: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the DAG.

        Parameters:
            initial_data: Initial data to pass to root nodes
            parallel: If True, execute independent nodes in parallel

        Returns:
            Dictionary mapping node names to their results
        """
        self.dag.reset()
        self._results = initial_data.copy() if initial_data else {}

        if parallel:
            return self._execute_parallel()
        else:
            return self._execute_sequential()

    def _execute_sequential(self) -> Dict[str, Any]:
        """Execute nodes sequentially in topological order."""
        for node_name in self.dag.topological_sort():
            node = self.dag.nodes[node_name]
            self._execute_node(node)

            if node.status == NodeStatus.FAILED:
                # Mark downstream as skipped
                self._skip_downstream(node_name)

        return self._results

    def _execute_parallel(self) -> Dict[str, Any]:
        """Execute nodes in parallel by level."""
        levels = self.dag.get_execution_levels()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for level in levels:
                # Submit all nodes at this level
                futures = {}
                for node_name in level:
                    node = self.dag.nodes[node_name]

                    # Check if upstream failed
                    upstream_failed = any(
                        self.dag.nodes[up].status == NodeStatus.FAILED
                        for up in self.dag.get_upstream(node_name)
                    )

                    if upstream_failed:
                        node.status = NodeStatus.SKIPPED
                        continue

                    future = executor.submit(self._execute_node, node)
                    futures[future] = node_name

                # Wait for all nodes at this level to complete
                for future in concurrent.futures.as_completed(futures):
                    node_name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.dag.nodes[node_name].error = e
                        self.dag.nodes[node_name].status = NodeStatus.FAILED

        return self._results

    def _execute_node(self, node: DAGNode) -> Any:
        """Execute a single node."""
        node.status = NodeStatus.RUNNING
        node.start_time = datetime.utcnow()

        # Gather inputs
        inputs = {}
        for input_name in node.inputs:
            if input_name in self._results:
                inputs[input_name] = self._results[input_name]

        # Add config
        inputs.update(node.config)

        # Execute with retry
        last_error = None
        for attempt in range(self.retry_count + 1):
            try:
                result = node.func(**inputs)

                with self._lock:
                    self._results[node.name] = result

                    # Store outputs
                    if isinstance(result, dict) and node.outputs:
                        for output_name in node.outputs:
                            if output_name in result:
                                self._results[output_name] = result[output_name]

                node.result = result
                node.status = NodeStatus.COMPLETED
                node.end_time = datetime.utcnow()
                return result

            except Exception as e:
                last_error = e
                if attempt < self.retry_count:
                    time.sleep(self.retry_delay)

        node.error = last_error
        node.status = NodeStatus.FAILED
        node.end_time = datetime.utcnow()
        raise last_error

    def _skip_downstream(self, node_name: str) -> None:
        """Skip all downstream nodes."""
        for downstream in self.dag.get_downstream(node_name):
            node = self.dag.nodes[downstream]
            if node.status == NodeStatus.PENDING:
                node.status = NodeStatus.SKIPPED
                self._skip_downstream(downstream)

    def get_status(self) -> Dict[str, str]:
        """Get status of all nodes."""
        return {
            name: node.status.value
            for name, node in self.dag.nodes.items()
        }

    def get_results(self) -> Dict[str, Any]:
        """Get all results."""
        return self._results.copy()

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = {
            "total_nodes": len(self.dag.nodes),
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "total_duration": 0.0,
            "node_durations": {},
        }

        for name, node in self.dag.nodes.items():
            if node.status == NodeStatus.COMPLETED:
                stats["completed"] += 1
            elif node.status == NodeStatus.FAILED:
                stats["failed"] += 1
            elif node.status == NodeStatus.SKIPPED:
                stats["skipped"] += 1

            if node.duration:
                stats["node_durations"][name] = node.duration
                stats["total_duration"] += node.duration

        return stats


def create_pipeline(*steps: tuple) -> DAG:
    """
    Create a simple linear pipeline from steps.

    Parameters:
        steps: Tuples of (name, function)

    Returns:
        DAG with linear execution order
    """
    dag = DAG(name="pipeline")

    previous_name = None
    for name, func in steps:
        inputs = [previous_name] if previous_name else []
        node = DAGNode(name=name, func=func, inputs=inputs)
        dag.add_node(node)

        if previous_name:
            dag.add_edge(previous_name, name)

        previous_name = name

    return dag
