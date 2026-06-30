"""Pytest configuration and fixtures for DynaGraph tests."""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield 42


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    from dynagraph import Tensor
    return Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)


@pytest.fixture
def sample_graph():
    """Create a sample computation graph for testing."""
    from dynagraph import Graph

    graph = Graph("test_graph")
    x = graph.add_input("x", (2, 3), "float32")
    y = graph.add_input("y", (3, 2), "float32")
    z = graph.add_operation("matmul", "matmul_0", [x, y])
    out = graph.add_output("output", z)

    return graph


@pytest.fixture
def simple_mlp_graph():
    """Create a simple MLP-style graph for optimization tests."""
    from dynagraph import Graph

    graph = Graph("mlp")
    x = graph.add_input("x", (4, 8), "float32")
    w1 = graph.add_input("w1", (8, 16), "float32")
    b1 = graph.add_input("b1", (16,), "float32")
    w2 = graph.add_input("w2", (16, 4), "float32")

    # Layer 1: matmul + bias + relu
    mm1 = graph.add_operation("matmul", "mm1", [x, w1])
    add1 = graph.add_operation("add", "add_bias1", [mm1, b1])
    relu1 = graph.add_operation("relu", "relu1", [add1])

    # Layer 2: matmul
    mm2 = graph.add_operation("matmul", "mm2", [relu1, w2])

    out = graph.add_output("output", mm2)

    return graph


@pytest.fixture
def cse_candidate_graph():
    """Create a graph with common subexpressions for CSE testing."""
    from dynagraph import Graph

    graph = Graph("cse_test")
    x = graph.add_input("x", (4, 4), "float32")
    y = graph.add_input("y", (4, 4), "float32")

    # Create two identical operations
    add1 = graph.add_operation("add", "add1", [x, y])
    add2 = graph.add_operation("add", "add2", [x, y])  # Duplicate

    # Use both results
    mul1 = graph.add_operation("mul", "mul1", [add1, x])
    mul2 = graph.add_operation("mul", "mul2", [add2, y])

    # Combine
    result = graph.add_operation("add", "final_add", [mul1, mul2])
    out = graph.add_output("output", result)

    return graph


@pytest.fixture
def dead_code_graph():
    """Create a graph with dead code for DCE testing."""
    from dynagraph import Graph

    graph = Graph("dce_test")
    x = graph.add_input("x", (4, 4), "float32")
    y = graph.add_input("y", (4, 4), "float32")

    # Used operations
    add1 = graph.add_operation("add", "used_add", [x, y])
    mul1 = graph.add_operation("mul", "used_mul", [add1, x])

    # Dead operations (not connected to output)
    dead_sub = graph.add_operation("sub", "dead_sub", [x, y])
    dead_div = graph.add_operation("div", "dead_div", [dead_sub, y])

    out = graph.add_output("output", mul1)

    return graph
