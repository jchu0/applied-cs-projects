"""Tests for core graph data structures."""

import unittest
import json
from typing import Set

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynamicgraph.core.graph import Graph, Node, Edge, OpType, NodeMetadata


class TestNode(unittest.TestCase):
    """Tests for Node class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(op_type=OpType.ADD, name="add_node")
        self.assertIsNotNone(node.id)
        self.assertEqual(node.op_type, OpType.ADD)
        self.assertEqual(node.name, "add_node")
        self.assertEqual(len(node.inputs), 0)
        self.assertEqual(len(node.outputs), 0)

    def test_node_connections(self):
        """Test adding/removing connections."""
        node = Node()

        # Add inputs
        node.add_input("input1")
        node.add_input("input2")
        self.assertEqual(len(node.inputs), 2)
        self.assertIn("input1", node.inputs)

        # Add duplicate - should not add
        node.add_input("input1")
        self.assertEqual(len(node.inputs), 2)

        # Remove input
        node.remove_input("input1")
        self.assertEqual(len(node.inputs), 1)
        self.assertNotIn("input1", node.inputs)

        # Add outputs
        node.add_output("output1")
        self.assertEqual(len(node.outputs), 1)

    def test_node_metadata(self):
        """Test node metadata."""
        metadata = NodeMetadata(
            dtype="float32",
            shape=(10, 20),
            device="cuda:0",
            requires_grad=True
        )
        node = Node(metadata=metadata)

        self.assertEqual(node.metadata.dtype, "float32")
        self.assertEqual(node.metadata.shape, (10, 20))
        self.assertEqual(node.metadata.device, "cuda:0")
        self.assertTrue(node.metadata.requires_grad)

    def test_node_attributes(self):
        """Test node attributes."""
        node = Node()
        node.attributes["kernel_size"] = 3
        node.attributes["padding"] = 1

        self.assertEqual(node.attributes["kernel_size"], 3)
        self.assertEqual(node.attributes["padding"], 1)


class TestEdge(unittest.TestCase):
    """Tests for Edge class."""

    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = Edge(source="node1", target="node2", index=0)
        self.assertEqual(edge.source, "node1")
        self.assertEqual(edge.target, "node2")
        self.assertEqual(edge.index, 0)

    def test_edge_attributes(self):
        """Test edge attributes."""
        edge = Edge(
            source="node1",
            target="node2",
            attributes={"weight": 1.0}
        )
        self.assertEqual(edge.attributes["weight"], 1.0)


class TestGraph(unittest.TestCase):
    """Tests for Graph class."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph = Graph(name="test_graph")

    def test_graph_creation(self):
        """Test basic graph creation."""
        self.assertEqual(self.graph.name, "test_graph")
        self.assertEqual(len(self.graph.nodes), 0)
        self.assertEqual(len(self.graph.edges), 0)

    def test_add_remove_nodes(self):
        """Test adding and removing nodes."""
        # Add nodes
        node1 = Node(op_type=OpType.INPUT, name="input")
        node2 = Node(op_type=OpType.ADD, name="add")
        node3 = Node(op_type=OpType.OUTPUT, name="output")

        id1 = self.graph.add_node(node1)
        id2 = self.graph.add_node(node2)
        id3 = self.graph.add_node(node3)

        self.assertEqual(len(self.graph.nodes), 3)
        self.assertIn(id1, self.graph.nodes)
        self.assertIn(id1, self.graph.input_nodes)
        self.assertIn(id3, self.graph.output_nodes)

        # Remove node
        self.graph.remove_node(id2)
        self.assertEqual(len(self.graph.nodes), 2)
        self.assertNotIn(id2, self.graph.nodes)

    def test_add_edges(self):
        """Test adding edges between nodes."""
        # Create nodes
        node1 = Node(op_type=OpType.INPUT)
        node2 = Node(op_type=OpType.ADD)
        node3 = Node(op_type=OpType.OUTPUT)

        id1 = self.graph.add_node(node1)
        id2 = self.graph.add_node(node2)
        id3 = self.graph.add_node(node3)

        # Add edges
        edge1 = self.graph.add_edge(id1, id2)
        edge2 = self.graph.add_edge(id2, id3)

        self.assertEqual(len(self.graph.edges), 2)
        self.assertIn(id2, self.graph.nodes[id1].outputs)
        self.assertIn(id1, self.graph.nodes[id2].inputs)

        # Test invalid edge
        with self.assertRaises(ValueError):
            self.graph.add_edge("invalid", id2)

    def test_predecessors_successors(self):
        """Test getting predecessors and successors."""
        # Build a simple graph: n1 -> n2 -> n3
        nodes = [Node() for _ in range(3)]
        ids = [self.graph.add_node(n) for n in nodes]

        self.graph.add_edge(ids[0], ids[1])
        self.graph.add_edge(ids[1], ids[2])

        # Check predecessors
        self.assertEqual(self.graph.get_predecessors(ids[0]), [])
        self.assertEqual(self.graph.get_predecessors(ids[1]), [ids[0]])
        self.assertEqual(self.graph.get_predecessors(ids[2]), [ids[1]])

        # Check successors
        self.assertEqual(self.graph.get_successors(ids[0]), [ids[1]])
        self.assertEqual(self.graph.get_successors(ids[1]), [ids[2]])
        self.assertEqual(self.graph.get_successors(ids[2]), [])

    def test_topological_sort(self):
        """Test topological sorting."""
        # Create DAG: n1 -> n2 -> n4
        #             n1 -> n3 -> n4
        nodes = [Node() for _ in range(4)]
        ids = [self.graph.add_node(n) for n in nodes]

        self.graph.add_edge(ids[0], ids[1])
        self.graph.add_edge(ids[0], ids[2])
        self.graph.add_edge(ids[1], ids[3])
        self.graph.add_edge(ids[2], ids[3])

        topo_order = self.graph.topological_sort()

        # Check ordering constraints
        self.assertEqual(len(topo_order), 4)
        self.assertLess(topo_order.index(ids[0]), topo_order.index(ids[1]))
        self.assertLess(topo_order.index(ids[0]), topo_order.index(ids[2]))
        self.assertLess(topo_order.index(ids[1]), topo_order.index(ids[3]))
        self.assertLess(topo_order.index(ids[2]), topo_order.index(ids[3]))

    def test_cycle_detection(self):
        """Test cycle detection."""
        # Create graph without cycle
        nodes = [Node() for _ in range(3)]
        ids = [self.graph.add_node(n) for n in nodes]

        self.graph.add_edge(ids[0], ids[1])
        self.graph.add_edge(ids[1], ids[2])

        self.assertFalse(self.graph.has_cycle())

        # Add edge to create cycle
        self.graph.add_edge(ids[2], ids[0])
        self.assertTrue(self.graph.has_cycle())

    def test_subgraph(self):
        """Test subgraph extraction."""
        # Create graph with 4 nodes
        nodes = [Node(name=f"n{i}") for i in range(4)]
        ids = [self.graph.add_node(n) for n in nodes]

        self.graph.add_edge(ids[0], ids[1])
        self.graph.add_edge(ids[1], ids[2])
        self.graph.add_edge(ids[2], ids[3])

        # Extract subgraph with first 3 nodes
        subgraph = self.graph.subgraph(set(ids[:3]))

        self.assertEqual(len(subgraph.nodes), 3)
        self.assertEqual(len(subgraph.edges), 2)
        self.assertNotIn(ids[3], subgraph.nodes)

    def test_graph_clone(self):
        """Test graph cloning."""
        # Create graph
        nodes = [Node(name=f"n{i}") for i in range(3)]
        ids = [self.graph.add_node(n) for n in nodes]
        self.graph.add_edge(ids[0], ids[1])
        self.graph.add_edge(ids[1], ids[2])

        # Clone
        cloned = self.graph.clone()

        self.assertEqual(len(cloned.nodes), len(self.graph.nodes))
        self.assertEqual(len(cloned.edges), len(self.graph.edges))
        self.assertIsNot(cloned, self.graph)

        # Modify clone shouldn't affect original
        cloned.remove_node(ids[0])
        self.assertIn(ids[0], self.graph.nodes)
        self.assertNotIn(ids[0], cloned.nodes)

    def test_graph_validation(self):
        """Test graph validation."""
        # Create valid graph
        node1 = Node()
        node2 = Node()
        id1 = self.graph.add_node(node1)
        id2 = self.graph.add_node(node2)
        self.graph.add_edge(id1, id2)

        issues = self.graph.validate()
        self.assertEqual(len(issues), 0)

        # Create invalid references
        self.graph.nodes[id1].inputs.append("invalid_id")
        issues = self.graph.validate()
        self.assertGreater(len(issues), 0)

    def test_serialization(self):
        """Test graph serialization."""
        # Create graph with metadata
        node1 = Node(
            op_type=OpType.INPUT,
            name="input",
            metadata=NodeMetadata(dtype="float32", shape=(10, 20))
        )
        node2 = Node(op_type=OpType.ADD, name="add")
        node3 = Node(op_type=OpType.OUTPUT, name="output")

        id1 = self.graph.add_node(node1)
        id2 = self.graph.add_node(node2)
        id3 = self.graph.add_node(node3)

        self.graph.add_edge(id1, id2)
        self.graph.add_edge(id2, id3)
        self.graph.metadata["version"] = "1.0"

        # Serialize
        data = self.graph.to_dict()

        # Verify structure
        self.assertEqual(data["name"], "test_graph")
        self.assertEqual(len(data["nodes"]), 3)
        self.assertEqual(len(data["edges"]), 2)
        self.assertEqual(data["metadata"]["version"], "1.0")

        # Deserialize
        restored = Graph.from_dict(data)

        self.assertEqual(restored.name, self.graph.name)
        self.assertEqual(len(restored.nodes), len(self.graph.nodes))
        self.assertEqual(len(restored.edges), len(self.graph.edges))

        # Check node metadata preserved
        restored_node = restored.nodes[id1]
        self.assertEqual(restored_node.metadata.dtype, "float32")
        self.assertEqual(restored_node.metadata.shape, (10, 20))


class TestGraphOperations(unittest.TestCase):
    """Tests for complex graph operations."""

    def test_diamond_graph(self):
        """Test diamond-shaped graph pattern."""
        graph = Graph()

        # Create diamond: n1 -> n2 -> n4
        #                 n1 -> n3 -> n4
        nodes = [Node(name=f"n{i}") for i in range(4)]
        ids = [graph.add_node(n) for n in nodes]

        graph.add_edge(ids[0], ids[1])
        graph.add_edge(ids[0], ids[2])
        graph.add_edge(ids[1], ids[3])
        graph.add_edge(ids[2], ids[3])

        # Verify structure
        self.assertEqual(len(graph.get_successors(ids[0])), 2)
        self.assertEqual(len(graph.get_predecessors(ids[3])), 2)

        # Topological sort should respect dependencies
        order = graph.topological_sort()
        self.assertEqual(order[0], ids[0])
        self.assertEqual(order[-1], ids[3])

    def test_linear_chain(self):
        """Test linear chain of operations."""
        graph = Graph()

        # Create chain of 10 nodes
        num_nodes = 10
        nodes = [Node(op_type=OpType.ADD, name=f"add_{i}") for i in range(num_nodes)]
        ids = [graph.add_node(n) for n in nodes]

        for i in range(num_nodes - 1):
            graph.add_edge(ids[i], ids[i + 1])

        # Verify chain structure
        self.assertEqual(len(graph.topological_sort()), num_nodes)

        # First node has no predecessors
        self.assertEqual(len(graph.get_predecessors(ids[0])), 0)

        # Last node has no successors
        self.assertEqual(len(graph.get_successors(ids[-1])), 0)

        # Middle nodes have exactly one predecessor and successor
        for i in range(1, num_nodes - 1):
            self.assertEqual(len(graph.get_predecessors(ids[i])), 1)
            self.assertEqual(len(graph.get_successors(ids[i])), 1)

    def test_parallel_branches(self):
        """Test parallel execution branches."""
        graph = Graph()

        # Create structure:
        # input -> [branch1_1 -> branch1_2] -> output
        #       -> [branch2_1 -> branch2_2] ->
        input_node = Node(op_type=OpType.INPUT, name="input")
        output_node = Node(op_type=OpType.OUTPUT, name="output")

        input_id = graph.add_node(input_node)
        output_id = graph.add_node(output_node)

        # Branch 1
        b1_nodes = [Node(op_type=OpType.ADD, name=f"b1_{i}") for i in range(2)]
        b1_ids = [graph.add_node(n) for n in b1_nodes]
        graph.add_edge(input_id, b1_ids[0])
        graph.add_edge(b1_ids[0], b1_ids[1])
        graph.add_edge(b1_ids[1], output_id)

        # Branch 2
        b2_nodes = [Node(op_type=OpType.MUL, name=f"b2_{i}") for i in range(2)]
        b2_ids = [graph.add_node(n) for n in b2_nodes]
        graph.add_edge(input_id, b2_ids[0])
        graph.add_edge(b2_ids[0], b2_ids[1])
        graph.add_edge(b2_ids[1], output_id)

        # Verify structure
        self.assertEqual(len(graph.nodes), 6)
        self.assertEqual(len(graph.get_successors(input_id)), 2)
        self.assertEqual(len(graph.get_predecessors(output_id)), 2)

        # Should not have cycles
        self.assertFalse(graph.has_cycle())


class TestOpTypes(unittest.TestCase):
    """Tests for operation types."""

    def test_op_type_categories(self):
        """Test operation type categorization."""
        # Tensor ops
        tensor_ops = [OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV, OpType.MATMUL]
        for op in tensor_ops:
            self.assertIsInstance(op, OpType)

        # NN ops
        nn_ops = [OpType.LINEAR, OpType.CONV2D, OpType.RELU, OpType.SOFTMAX]
        for op in nn_ops:
            self.assertIsInstance(op, OpType)

        # Shape ops
        shape_ops = [OpType.RESHAPE, OpType.TRANSPOSE, OpType.SQUEEZE]
        for op in shape_ops:
            self.assertIsInstance(op, OpType)

    def test_op_type_node_creation(self):
        """Test creating nodes with different op types."""
        graph = Graph()

        # Create nodes with different op types
        op_types = [
            OpType.INPUT, OpType.ADD, OpType.RELU,
            OpType.CONV2D, OpType.OUTPUT
        ]

        for op_type in op_types:
            node = Node(op_type=op_type, name=op_type.name.lower())
            node_id = graph.add_node(node)
            self.assertEqual(graph.nodes[node_id].op_type, op_type)


if __name__ == "__main__":
    unittest.main()