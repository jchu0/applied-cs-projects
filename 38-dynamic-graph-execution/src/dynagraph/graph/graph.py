"""Dynamic computation graph implementation."""

import logging
from typing import Any, Callable, List, Dict, Set, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Base node in computation graph."""
    id: int
    name: str
    inputs: List['Node'] = field(default_factory=list)
    outputs: List['Node'] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id


@dataclass
class InputNode(Node):
    """Input node representing graph inputs."""
    shape: Tuple[int, ...] = ()
    dtype: str = "float32"


@dataclass
class OutputNode(Node):
    """Output node representing graph outputs."""
    pass


@dataclass
class OperationNode(Node):
    """Node representing an operation."""
    op_type: str = "unknown"
    compute_fn: Optional[Callable] = None
    grad_fn: Optional[Callable] = None


class Operation:
    """Represents a computation operation."""

    def __init__(self, op_type: str, name: str = None):
        self.op_type = op_type
        self.name = name or op_type
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []

    def __repr__(self):
        return f"Operation({self.op_type}, inputs={len(self.inputs)}, outputs={len(self.outputs)})"


class Graph:
    """
    Dynamic computation graph.

    Features:
    - Dynamic construction during forward pass
    - Topological sorting for execution
    - Subgraph extraction
    - Graph optimization
    """

    def __init__(self, name: str = "graph"):
        self.name = name
        self._nodes: Dict[int, Node] = {}
        self._inputs: List[InputNode] = []
        self._outputs: List[OutputNode] = []
        self._operations: List[OperationNode] = []
        self._next_id = 0
        self._lock = threading.Lock()

    def _get_next_id(self) -> int:
        """Get next unique node ID."""
        with self._lock:
            id = self._next_id
            self._next_id += 1
            return id

    def add_input(self, name: str, shape: Tuple[int, ...], dtype: str = "float32") -> InputNode:
        """Add an input node."""
        node = InputNode(
            id=self._get_next_id(),
            name=name,
            shape=shape,
            dtype=dtype
        )
        self._nodes[node.id] = node
        self._inputs.append(node)
        return node

    def add_output(self, name: str, input_node: Node) -> OutputNode:
        """Add an output node."""
        node = OutputNode(
            id=self._get_next_id(),
            name=name,
            inputs=[input_node]
        )
        input_node.outputs.append(node)
        self._nodes[node.id] = node
        self._outputs.append(node)
        return node

    def add_operation(
        self,
        op_type: str,
        name: str,
        inputs: List[Node],
        compute_fn: Callable = None,
        grad_fn: Callable = None,
        attrs: Dict[str, Any] = None
    ) -> OperationNode:
        """Add an operation node."""
        node = OperationNode(
            id=self._get_next_id(),
            name=name,
            op_type=op_type,
            inputs=inputs,
            compute_fn=compute_fn,
            grad_fn=grad_fn,
            attrs=attrs or {}
        )

        # Update connections
        for inp in inputs:
            inp.outputs.append(node)

        self._nodes[node.id] = node
        self._operations.append(node)
        return node

    def get_node(self, node_id: int) -> Optional[Node]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def get_nodes(self) -> List[Node]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_inputs(self) -> List[InputNode]:
        """Get input nodes."""
        return self._inputs

    def get_outputs(self) -> List[OutputNode]:
        """Get output nodes."""
        return self._outputs

    def topological_sort(self) -> List[Node]:
        """Sort nodes in topological order."""
        in_degree = defaultdict(int)
        for node in self._nodes.values():
            for inp in node.inputs:
                in_degree[node.id] += 1

        # Start with nodes having no inputs
        queue = [node for node in self._nodes.values() if in_degree[node.id] == 0]
        sorted_nodes = []

        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)

            for out in node.outputs:
                in_degree[out.id] -= 1
                if in_degree[out.id] == 0:
                    queue.append(out)

        if len(sorted_nodes) != len(self._nodes):
            raise RuntimeError("Graph contains cycles")

        return sorted_nodes

    def reverse_topological_sort(self) -> List[Node]:
        """Sort nodes in reverse topological order (for backward pass)."""
        return list(reversed(self.topological_sort()))

    def subgraph(self, output_nodes: List[Node]) -> 'Graph':
        """Extract subgraph ending at given output nodes."""
        subgraph = Graph(f"{self.name}_subgraph")

        # Find all nodes needed for outputs
        needed = set()
        queue = list(output_nodes)

        while queue:
            node = queue.pop(0)
            if node.id in needed:
                continue
            needed.add(node.id)
            queue.extend(node.inputs)

        # Copy needed nodes
        node_map = {}
        for node_id in needed:
            old_node = self._nodes[node_id]
            if isinstance(old_node, InputNode):
                new_node = subgraph.add_input(old_node.name, old_node.shape, old_node.dtype)
            elif isinstance(old_node, OperationNode):
                new_inputs = [node_map[inp.id] for inp in old_node.inputs if inp.id in node_map]
                new_node = subgraph.add_operation(
                    old_node.op_type,
                    old_node.name,
                    new_inputs,
                    old_node.compute_fn,
                    old_node.grad_fn,
                    old_node.attrs
                )
            else:
                continue
            node_map[node_id] = new_node

        # Add outputs
        for out in output_nodes:
            if out.id in node_map:
                subgraph.add_output(out.name, node_map[out.id])

        return subgraph

    def num_nodes(self) -> int:
        """Get number of nodes."""
        return len(self._nodes)

    def num_operations(self) -> int:
        """Get number of operation nodes."""
        return len(self._operations)

    def __repr__(self):
        return f"Graph({self.name}, nodes={self.num_nodes()}, ops={self.num_operations()})"


class GraphContext:
    """
    Context manager for graph construction.

    Captures operations and builds graph dynamically.
    """

    _current: 'GraphContext' = None
    _lock = threading.Lock()

    def __init__(self, graph: Graph = None):
        self.graph = graph or Graph()
        self._prev_context = None
        self._tensor_to_node: Dict[int, Node] = {}
        self._recording = False

    @classmethod
    def get_current(cls) -> Optional['GraphContext']:
        """Get current graph context."""
        return cls._current

    def __enter__(self):
        with self._lock:
            self._prev_context = GraphContext._current
            GraphContext._current = self
            self._recording = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            GraphContext._current = self._prev_context
            self._recording = False
        return False

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def add_input(self, tensor, name: str = None) -> Node:
        """Register a tensor as graph input."""
        tensor_id = id(tensor)
        if tensor_id in self._tensor_to_node:
            return self._tensor_to_node[tensor_id]

        name = name or f"input_{len(self.graph.get_inputs())}"
        node = self.graph.add_input(name, tensor.shape)
        self._tensor_to_node[tensor_id] = node
        return node

    def add_operation(
        self,
        op_type: str,
        inputs: List[Any],
        output: Any,
        name: str = None,
        compute_fn: Callable = None,
        grad_fn: Callable = None,
        attrs: Dict[str, Any] = None
    ) -> Node:
        """Record an operation."""
        if not self._recording:
            return None

        # Get input nodes
        input_nodes = []
        for inp in inputs:
            tensor_id = id(inp)
            if tensor_id in self._tensor_to_node:
                input_nodes.append(self._tensor_to_node[tensor_id])
            else:
                # Create input node for unregistered tensor
                node = self.add_input(inp)
                input_nodes.append(node)

        # Add operation
        name = name or f"{op_type}_{len(self.graph.get_nodes())}"
        op_node = self.graph.add_operation(
            op_type, name, input_nodes, compute_fn, grad_fn, attrs
        )

        # Map output tensor to node
        self._tensor_to_node[id(output)] = op_node
        return op_node

    def add_output(self, tensor, name: str = None) -> Node:
        """Register a tensor as graph output."""
        tensor_id = id(tensor)
        if tensor_id not in self._tensor_to_node:
            self.add_input(tensor)

        input_node = self._tensor_to_node[tensor_id]
        name = name or f"output_{len(self.graph.get_outputs())}"
        return self.graph.add_output(name, input_node)

    def get_node(self, tensor) -> Optional[Node]:
        """Get node for tensor."""
        return self._tensor_to_node.get(id(tensor))

    def get_graph(self) -> Graph:
        """Get constructed graph."""
        return self.graph


def trace_graph(func: Callable, *args, **kwargs) -> Tuple[Graph, Any]:
    """
    Trace a function to build computation graph.

    Args:
        func: Function to trace
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Tuple of (graph, outputs)
    """
    context = GraphContext()

    with context:
        # Register inputs
        for i, arg in enumerate(args):
            context.add_input(arg, f"arg_{i}")

        # Execute function
        outputs = func(*args, **kwargs)

        # Register outputs
        if isinstance(outputs, tuple):
            for i, out in enumerate(outputs):
                context.add_output(out, f"output_{i}")
        else:
            context.add_output(outputs, "output")

    return context.get_graph(), outputs
