"""Core graph data structures for dynamic execution."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto


class OpType(Enum):
    """Types of operations in the graph."""
    # Tensor ops
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MATMUL = auto()

    # Neural network ops
    LINEAR = auto()
    CONV2D = auto()
    RELU = auto()
    SIGMOID = auto()
    SOFTMAX = auto()
    BATCHNORM = auto()
    DROPOUT = auto()

    # Shape ops
    RESHAPE = auto()
    TRANSPOSE = auto()
    PERMUTE = auto()
    SQUEEZE = auto()
    UNSQUEEZE = auto()

    # Reduction ops
    SUM = auto()
    MEAN = auto()
    MAX = auto()
    MIN = auto()

    # Control flow
    IF_THEN_ELSE = auto()
    WHILE_LOOP = auto()
    FOR_LOOP = auto()

    # Memory ops
    COPY = auto()
    CLONE = auto()
    DETACH = auto()

    # Placeholder
    INPUT = auto()
    OUTPUT = auto()
    CONSTANT = auto()
    PARAMETER = auto()
    BUFFER = auto()

    # Custom
    CUSTOM = auto()


@dataclass
class NodeMetadata:
    """Metadata associated with a graph node."""
    dtype: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None
    device: Optional[str] = None
    requires_grad: bool = False
    is_parameter: bool = False
    is_buffer: bool = False
    source_location: Optional[str] = None
    original_name: Optional[str] = None
    compile_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    """Represents a node in the computation graph."""
    id: str = field(default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}")
    op_type: OpType = OpType.CUSTOM
    name: Optional[str] = None
    inputs: List[str] = field(default_factory=list)  # Node IDs
    outputs: List[str] = field(default_factory=list)  # Node IDs
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: NodeMetadata = field(default_factory=NodeMetadata)

    def add_input(self, node_id: str) -> None:
        """Add an input to this node."""
        if node_id not in self.inputs:
            self.inputs.append(node_id)

    def add_output(self, node_id: str) -> None:
        """Add an output from this node."""
        if node_id not in self.outputs:
            self.outputs.append(node_id)

    def remove_input(self, node_id: str) -> None:
        """Remove an input from this node."""
        if node_id in self.inputs:
            self.inputs.remove(node_id)

    def remove_output(self, node_id: str) -> None:
        """Remove an output from this node."""
        if node_id in self.outputs:
            self.outputs.remove(node_id)

    def __repr__(self) -> str:
        return f"Node(id={self.id}, op={self.op_type.name}, name={self.name})"


@dataclass
class Edge:
    """Represents an edge between nodes in the graph."""
    source: str  # Source node ID
    target: str  # Target node ID
    index: int = 0  # Input index at target node
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Edge({self.source} -> {self.target}[{self.index}])"


class Graph:
    """Dynamic computation graph."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or f"graph_{uuid.uuid4().hex[:8]}"
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.input_nodes: List[str] = []
        self.output_nodes: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self._topological_order: Optional[List[str]] = None

    def add_node(self, node: Node) -> str:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self._topological_order = None  # Invalidate cache

        # Track inputs and outputs
        if node.op_type == OpType.INPUT:
            self.input_nodes.append(node.id)
        elif node.op_type == OpType.OUTPUT:
            self.output_nodes.append(node.id)

        return node.id

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph."""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Remove edges
        self.edges = [e for e in self.edges
                      if e.source != node_id and e.target != node_id]

        # Update connected nodes
        for input_id in node.inputs:
            if input_id in self.nodes:
                self.nodes[input_id].remove_output(node_id)

        for output_id in node.outputs:
            if output_id in self.nodes:
                self.nodes[output_id].remove_input(node_id)

        # Remove from special lists
        if node_id in self.input_nodes:
            self.input_nodes.remove(node_id)
        if node_id in self.output_nodes:
            self.output_nodes.remove(node_id)

        del self.nodes[node_id]
        self._topological_order = None

    def add_edge(self, source: str, target: str, index: int = 0,
                 attributes: Optional[Dict] = None) -> Edge:
        """Add an edge between two nodes."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Both nodes must exist: {source}, {target}")

        edge = Edge(source=source, target=target, index=index,
                   attributes=attributes or {})
        self.edges.append(edge)

        # Update node connections
        self.nodes[source].add_output(target)
        self.nodes[target].add_input(source)

        self._topological_order = None
        return edge

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all predecessor nodes."""
        if node_id not in self.nodes:
            return []
        return self.nodes[node_id].inputs

    def get_successors(self, node_id: str) -> List[str]:
        """Get all successor nodes."""
        if node_id not in self.nodes:
            return []
        return self.nodes[node_id].outputs

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order."""
        if self._topological_order is not None:
            return self._topological_order

        # Kahn's algorithm
        in_degree = {node_id: len(self.nodes[node_id].inputs)
                    for node_id in self.nodes}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for successor in self.nodes[node_id].outputs:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles")

        self._topological_order = result
        return result

    def has_cycle(self) -> bool:
        """Check if the graph contains cycles."""
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True

    def subgraph(self, node_ids: Set[str]) -> 'Graph':
        """Extract a subgraph containing only specified nodes."""
        subg = Graph(name=f"{self.name}_sub")

        # Copy nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                new_node = Node(
                    id=node_id,
                    op_type=node.op_type,
                    name=node.name,
                    attributes=dict(node.attributes),
                    metadata=node.metadata
                )
                subg.add_node(new_node)

        # Copy edges
        for edge in self.edges:
            if edge.source in node_ids and edge.target in node_ids:
                subg.add_edge(edge.source, edge.target,
                            edge.index, dict(edge.attributes))

        # Preserve the declared input/output ordering. ``add_node`` appends to
        # these lists in the (unordered) node-id iteration order, so restore the
        # original relative order for the nodes that survived into the subgraph.
        subg.input_nodes = [n for n in self.input_nodes if n in node_ids]
        subg.output_nodes = [n for n in self.output_nodes if n in node_ids]

        return subg

    def clone(self) -> 'Graph':
        """Create a deep copy of the graph."""
        return self.subgraph(set(self.nodes.keys()))

    def validate(self) -> List[str]:
        """Validate graph structure and return any issues."""
        issues = []

        # Check for cycles
        if self.has_cycle():
            issues.append("Graph contains cycles")

        # Check edge consistency
        for edge in self.edges:
            if edge.source not in self.nodes:
                issues.append(f"Edge source {edge.source} not in graph")
            if edge.target not in self.nodes:
                issues.append(f"Edge target {edge.target} not in graph")

        # Check node connection consistency
        for node_id, node in self.nodes.items():
            for input_id in node.inputs:
                if input_id not in self.nodes:
                    issues.append(f"Node {node_id} has invalid input {input_id}")
            for output_id in node.outputs:
                if output_id not in self.nodes:
                    issues.append(f"Node {node_id} has invalid output {output_id}")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "name": self.name,
            "nodes": {
                node_id: {
                    "id": node.id,
                    "op_type": node.op_type.name,
                    "name": node.name,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "attributes": node.attributes,
                    "metadata": {
                        "dtype": node.metadata.dtype,
                        "shape": node.metadata.shape,
                        "device": node.metadata.device,
                        "requires_grad": node.metadata.requires_grad,
                    }
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "index": edge.index,
                    "attributes": edge.attributes
                }
                for edge in self.edges
            ],
            "input_nodes": self.input_nodes,
            "output_nodes": self.output_nodes,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Graph':
        """Deserialize graph from dictionary."""
        graph = cls(name=data["name"])
        graph.metadata = data.get("metadata", {})

        # Create nodes
        for node_data in data["nodes"].values():
            metadata = NodeMetadata(
                dtype=node_data["metadata"].get("dtype"),
                shape=tuple(node_data["metadata"]["shape"])
                      if node_data["metadata"].get("shape") else None,
                device=node_data["metadata"].get("device"),
                requires_grad=node_data["metadata"].get("requires_grad", False),
            )
            node = Node(
                id=node_data["id"],
                op_type=OpType[node_data["op_type"]],
                name=node_data.get("name"),
                attributes=node_data.get("attributes", {}),
                metadata=metadata
            )
            graph.nodes[node.id] = node

        # Create edges
        for edge_data in data["edges"]:
            graph.add_edge(
                source=edge_data["source"],
                target=edge_data["target"],
                index=edge_data.get("index", 0),
                attributes=edge_data.get("attributes", {})
            )

        graph.input_nodes = data.get("input_nodes", [])
        graph.output_nodes = data.get("output_nodes", [])

        return graph

    def __repr__(self) -> str:
        return f"Graph(name={self.name}, nodes={len(self.nodes)}, edges={len(self.edges)})"

    def __str__(self) -> str:
        """Pretty print graph structure."""
        lines = [f"Graph: {self.name}"]
        lines.append(f"  Nodes: {len(self.nodes)}")
        lines.append(f"  Edges: {len(self.edges)}")

        if self.input_nodes:
            lines.append(f"  Inputs: {', '.join(self.input_nodes)}")
        if self.output_nodes:
            lines.append(f"  Outputs: {', '.join(self.output_nodes)}")

        # Show first few nodes
        for i, (node_id, node) in enumerate(self.nodes.items()):
            if i >= 5:
                lines.append(f"  ... and {len(self.nodes) - 5} more nodes")
                break
            lines.append(f"  {node}")

        return "\n".join(lines)