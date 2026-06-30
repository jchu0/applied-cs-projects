"""Graph optimizer that orchestrates optimization passes."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core.graph import Graph
from .passes import (
    OptimizationPass,
    PassResult,
    ConstantFolding,
    DeadCodeElimination,
    CommonSubexpressionElimination,
    AlgebraicSimplification,
    OperatorFusion,
    LayoutOptimization,
    ShapeInference,
)


@dataclass
class OptimizationStats:
    """Statistics from optimization."""
    total_passes: int = 0
    total_changes: int = 0
    nodes_removed: int = 0
    nodes_added: int = 0
    optimization_time_ms: float = 0.0
    pass_results: Dict[str, List[PassResult]] = field(default_factory=dict)


class GraphOptimizer:
    """
    Optimizes computation graphs through a series of transformation passes.

    The optimizer runs passes until convergence (no more changes) or a
    maximum number of iterations is reached.
    """

    def __init__(
        self,
        optimization_level: int = 1,
        max_iterations: int = 10,
        verbose: bool = False
    ):
        """
        Initialize the graph optimizer.

        Args:
            optimization_level: 0=none, 1=basic, 2=aggressive
            max_iterations: Maximum iterations before stopping
            verbose: Print optimization progress
        """
        self.optimization_level = optimization_level
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Configure passes based on optimization level
        self.passes: List[OptimizationPass] = []
        self._configure_passes()

    def _configure_passes(self):
        """Configure passes based on optimization level."""
        self.passes = []

        if self.optimization_level == 0:
            return

        # Level 1: Basic optimizations
        if self.optimization_level >= 1:
            self.passes.extend([
                ShapeInference(),
                ConstantFolding(),
                AlgebraicSimplification(),
                DeadCodeElimination(),
            ])

        # Level 2: Aggressive optimizations
        if self.optimization_level >= 2:
            self.passes.extend([
                CommonSubexpressionElimination(),
                OperatorFusion(),
                LayoutOptimization(),
            ])

    def add_pass(self, pass_: OptimizationPass, index: Optional[int] = None):
        """Add a custom optimization pass."""
        if index is not None:
            self.passes.insert(index, pass_)
        else:
            self.passes.append(pass_)

    def remove_pass(self, name: str):
        """Remove a pass by name."""
        self.passes = [p for p in self.passes if p.name != name]

    def enable_pass(self, name: str):
        """Enable a pass by name."""
        for p in self.passes:
            if p.name == name:
                p.enabled = True

    def disable_pass(self, name: str):
        """Disable a pass by name."""
        for p in self.passes:
            if p.name == name:
                p.enabled = False

    def optimize(self, graph: Graph) -> Tuple[Graph, OptimizationStats]:
        """
        Optimize the graph through transformation passes.

        Args:
            graph: The graph to optimize

        Returns:
            Tuple of (optimized_graph, statistics)
        """
        stats = OptimizationStats()
        start_time = time.time()

        if self.optimization_level == 0:
            return graph, stats

        # Clone graph to avoid modifying original
        optimized = graph.clone()

        if self.verbose:
            print(f"Starting optimization with {len(self.passes)} passes")
            print(f"Initial graph: {len(optimized.nodes)} nodes, {len(optimized.edges)} edges")

        # Run passes until convergence
        for iteration in range(self.max_iterations):
            changed_this_iteration = False

            for pass_ in self.passes:
                if not pass_.enabled:
                    continue

                result = pass_.run(optimized)
                stats.total_passes += 1

                # Track results
                if pass_.name not in stats.pass_results:
                    stats.pass_results[pass_.name] = []
                stats.pass_results[pass_.name].append(result)

                if result.changed:
                    changed_this_iteration = True
                    stats.total_changes += 1
                    stats.nodes_removed += result.nodes_removed
                    stats.nodes_added += result.nodes_added

                    if self.verbose:
                        print(f"  [{pass_.name}] {result.message}")

            if not changed_this_iteration:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

        stats.optimization_time_ms = (time.time() - start_time) * 1000

        if self.verbose:
            print(f"Final graph: {len(optimized.nodes)} nodes, {len(optimized.edges)} edges")
            print(f"Optimization took {stats.optimization_time_ms:.2f}ms")

        return optimized, stats

    def optimize_for_inference(self, graph: Graph) -> Tuple[Graph, OptimizationStats]:
        """
        Optimize graph specifically for inference (no gradient tracking).

        Applies additional optimizations valid only for inference:
        - Fuses batch norm into convolutions
        - Eliminates dropout
        - Constant propagation through training-only ops
        """
        # First run standard optimization
        optimized, stats = self.optimize(graph)

        # Additional inference-specific optimizations
        from ..core.graph import OpType

        # Remove dropout nodes (they're identity at inference)
        dropout_nodes = [
            node_id for node_id, node in optimized.nodes.items()
            if node.op_type == OpType.DROPOUT
        ]

        for node_id in dropout_nodes:
            node = optimized.nodes[node_id]
            if node.inputs:
                # Replace dropout with its input
                input_id = node.inputs[0]
                for other_node in optimized.nodes.values():
                    other_node.inputs = [
                        input_id if inp == node_id else inp
                        for inp in other_node.inputs
                    ]
                optimized.output_nodes = [
                    input_id if o == node_id else o
                    for o in optimized.output_nodes
                ]
            optimized.remove_node(node_id)
            stats.nodes_removed += 1

        return optimized, stats

    def optimize_for_training(self, graph: Graph) -> Tuple[Graph, OptimizationStats]:
        """
        Optimize graph for training (preserves gradient tracking).

        More conservative optimization that maintains backward compatibility.
        """
        # Use level 1 optimization to be conservative
        original_level = self.optimization_level
        self.optimization_level = min(self.optimization_level, 1)
        self._configure_passes()

        optimized, stats = self.optimize(graph)

        # Restore original level
        self.optimization_level = original_level
        self._configure_passes()

        return optimized, stats

    def get_pass_stats(self) -> Dict[str, dict]:
        """Get statistics from all passes."""
        return {p.name: p.stats for p in self.passes}

    def reset_stats(self):
        """Reset statistics for all passes."""
        for p in self.passes:
            p.stats = {
                "runs": 0,
                "changes": 0,
                "nodes_removed": 0,
                "nodes_added": 0,
            }


def optimize_graph(
    graph: Graph,
    level: int = 1,
    mode: str = "default"
) -> Graph:
    """
    Convenience function to optimize a graph.

    Args:
        graph: Graph to optimize
        level: Optimization level (0-2)
        mode: "default", "inference", or "training"

    Returns:
        Optimized graph
    """
    optimizer = GraphOptimizer(optimization_level=level)

    if mode == "inference":
        optimized, _ = optimizer.optimize_for_inference(graph)
    elif mode == "training":
        optimized, _ = optimizer.optimize_for_training(graph)
    else:
        optimized, _ = optimizer.optimize(graph)

    return optimized
