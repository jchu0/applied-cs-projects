"""Graph optimization and kernel fusion."""

from dataclasses import dataclass
from enum import Enum
from typing import Any
import uuid

from ..core.kernel import (
    Kernel, ComputeGraph, KernelType, TensorDescriptor, DataType
)


class OptimizationPass(Enum):
    """Types of optimization passes."""
    FUSION = "fusion"
    CONSTANT_FOLDING = "constant_folding"
    DEAD_CODE_ELIMINATION = "dead_code"
    LAYOUT_OPTIMIZATION = "layout"
    MEMORY_PLANNING = "memory"
    KERNEL_SELECTION = "kernel_selection"


@dataclass
class OptimizationResult:
    """Result of optimization."""
    original_kernels: int
    optimized_kernels: int
    estimated_speedup: float
    passes_applied: list[str]
    fused_patterns: list[str]


class GraphOptimizer:
    """Base class for graph optimizers."""

    def optimize(self, graph: ComputeGraph) -> ComputeGraph:
        """Apply optimization to graph."""
        raise NotImplementedError


class KernelFuser(GraphOptimizer):
    """Fuse kernels for better performance."""

    def __init__(self):
        self.fusion_patterns = [
            self._fuse_gemm_bias_activation,
            self._fuse_layer_norm,
            self._fuse_attention,
            self._fuse_elementwise_chain,
        ]
        self.fused_count = 0
        self.fused_patterns: list[str] = []

    def optimize(self, graph: ComputeGraph) -> ComputeGraph:
        """Apply kernel fusion."""
        self.fused_count = 0
        self.fused_patterns = []

        optimized = self._copy_graph(graph)

        # Apply each fusion pattern
        for pattern_fn in self.fusion_patterns:
            changed = True
            while changed:
                changed = pattern_fn(optimized)

        return optimized

    def _copy_graph(self, graph: ComputeGraph) -> ComputeGraph:
        """Create a copy of the graph."""
        new_graph = ComputeGraph()
        new_graph.input_tensors = graph.input_tensors.copy()
        new_graph.output_tensors = graph.output_tensors.copy()

        for kernel_id, kernel in graph.kernels.items():
            new_graph.kernels[kernel_id] = kernel

        new_graph.dependencies = graph.dependencies.copy()
        return new_graph

    def _fuse_gemm_bias_activation(self, graph: ComputeGraph) -> bool:
        """Fuse GEMM + Bias + Activation."""
        fused = False

        for kernel_id in list(graph.kernels.keys()):
            if kernel_id not in graph.kernels:
                continue

            kernel = graph.kernels[kernel_id]
            if kernel.kernel_type != KernelType.GEMM:
                continue

            # Find dependent elementwise (bias add)
            dependents = graph.get_dependents(kernel_id)
            for dep_id in dependents:
                if dep_id not in graph.kernels:
                    continue

                dep_kernel = graph.kernels[dep_id]
                if dep_kernel.kernel_type != KernelType.ELEMENTWISE:
                    continue

                # Check for activation after bias
                activation_id = None
                for act_dep in graph.get_dependents(dep_id):
                    if act_dep in graph.kernels:
                        act_kernel = graph.kernels[act_dep]
                        if act_kernel.kernel_type == KernelType.ELEMENTWISE:
                            if act_kernel.attributes.get("op") in ["relu", "gelu", "silu"]:
                                activation_id = act_dep
                                break

                # Create fused kernel
                fused_kernel = Kernel(
                    kernel_id=str(uuid.uuid4())[:8],
                    name=f"fused_gemm_bias_act",
                    kernel_type=KernelType.GEMM,
                    inputs=kernel.inputs.copy(),
                    outputs=dep_kernel.outputs.copy() if not activation_id else graph.kernels[activation_id].outputs.copy(),
                    device_id=kernel.device_id,
                    estimated_time_us=kernel.estimated_time_us * 0.9,  # Fusion speedup
                    attributes={
                        "fused": True,
                        "has_bias": True,
                        "activation": graph.kernels[activation_id].attributes.get("op") if activation_id else None
                    }
                )

                # Update graph
                self._replace_kernels(
                    graph,
                    [kernel_id, dep_id] + ([activation_id] if activation_id else []),
                    fused_kernel
                )

                self.fused_count += 1
                self.fused_patterns.append("gemm_bias_activation")
                fused = True
                break

        return fused

    def _fuse_layer_norm(self, graph: ComputeGraph) -> bool:
        """Fuse LayerNorm components."""
        # Simplified: just mark as done
        return False

    def _fuse_attention(self, graph: ComputeGraph) -> bool:
        """Fuse attention pattern (Q*K -> Softmax -> V)."""
        # Would fuse QK matmul + softmax + V matmul
        return False

    def _fuse_elementwise_chain(self, graph: ComputeGraph) -> bool:
        """Fuse chain of elementwise operations."""
        fused = False

        for kernel_id in list(graph.kernels.keys()):
            if kernel_id not in graph.kernels:
                continue

            kernel = graph.kernels[kernel_id]
            if kernel.kernel_type != KernelType.ELEMENTWISE:
                continue

            # Find chain of elementwise ops
            chain = [kernel_id]
            current = kernel_id

            while True:
                dependents = graph.get_dependents(current)
                if len(dependents) != 1:
                    break

                dep_id = dependents[0]
                if dep_id not in graph.kernels:
                    break

                dep_kernel = graph.kernels[dep_id]
                if dep_kernel.kernel_type != KernelType.ELEMENTWISE:
                    break

                # Check if this is the only dependency
                if len(graph.get_dependencies(dep_id)) != 1:
                    break

                chain.append(dep_id)
                current = dep_id

            if len(chain) >= 3:
                # Fuse the chain
                first_kernel = graph.kernels[chain[0]]
                last_kernel = graph.kernels[chain[-1]]

                fused_kernel = Kernel(
                    kernel_id=str(uuid.uuid4())[:8],
                    name=f"fused_elementwise_x{len(chain)}",
                    kernel_type=KernelType.ELEMENTWISE,
                    inputs=first_kernel.inputs.copy(),
                    outputs=last_kernel.outputs.copy(),
                    device_id=first_kernel.device_id,
                    estimated_time_us=first_kernel.estimated_time_us * len(chain) * 0.5,
                    attributes={"fused": True, "num_ops": len(chain)}
                )

                self._replace_kernels(graph, chain, fused_kernel)
                self.fused_count += len(chain) - 1
                self.fused_patterns.append(f"elementwise_x{len(chain)}")
                fused = True
                break

        return fused

    def _replace_kernels(
        self,
        graph: ComputeGraph,
        old_ids: list[str],
        new_kernel: Kernel
    ) -> None:
        """Replace multiple kernels with a single kernel."""
        # Add new kernel
        graph.add_kernel(new_kernel)

        # Update dependencies
        first_id = old_ids[0]
        last_id = old_ids[-1]

        new_deps = []
        for dep in graph.dependencies:
            if dep.source_id in old_ids and dep.target_id in old_ids:
                # Internal dependency, skip
                continue
            elif dep.target_id == first_id:
                # Incoming dependency
                new_deps.append(type(dep)(
                    source_id=dep.source_id,
                    target_id=new_kernel.kernel_id,
                    tensor_id=dep.tensor_id
                ))
            elif dep.source_id == last_id:
                # Outgoing dependency
                new_deps.append(type(dep)(
                    source_id=new_kernel.kernel_id,
                    target_id=dep.target_id,
                    tensor_id=dep.tensor_id
                ))
            elif dep.source_id not in old_ids and dep.target_id not in old_ids:
                new_deps.append(dep)

        graph.dependencies = new_deps

        # Remove old kernels
        for old_id in old_ids:
            if old_id in graph.kernels:
                del graph.kernels[old_id]


class MemoryOptimizer(GraphOptimizer):
    """Optimize memory usage and layout."""

    def __init__(self, max_memory_mb: float = 16000):
        self.max_memory = max_memory_mb * 1024 * 1024

    def optimize(self, graph: ComputeGraph) -> ComputeGraph:
        """Plan memory allocation."""
        optimized = ComputeGraph()
        optimized.kernels = graph.kernels.copy()
        optimized.dependencies = graph.dependencies.copy()

        # Compute tensor lifetimes
        lifetimes = self._compute_lifetimes(graph)

        # Plan memory allocation
        allocation = self._plan_allocation(graph, lifetimes)

        # Update tensor offsets
        for kernel_id, kernel in optimized.kernels.items():
            for tensor in kernel.inputs + kernel.outputs:
                if tensor.tensor_id in allocation:
                    tensor.memory_offset = allocation[tensor.tensor_id]

        return optimized

    def _compute_lifetimes(self, graph: ComputeGraph) -> dict[str, tuple[int, int]]:
        """Compute tensor lifetimes (start, end indices)."""
        topo_order = graph.topological_sort()
        kernel_index = {k: i for i, k in enumerate(topo_order)}

        lifetimes: dict[str, tuple[int, int]] = {}

        for kernel_id in topo_order:
            kernel = graph.kernels[kernel_id]
            idx = kernel_index[kernel_id]

            # Outputs are created here
            for tensor in kernel.outputs:
                if tensor.tensor_id not in lifetimes:
                    lifetimes[tensor.tensor_id] = (idx, idx)
                else:
                    start, _ = lifetimes[tensor.tensor_id]
                    lifetimes[tensor.tensor_id] = (start, idx)

            # Inputs are used here
            for tensor in kernel.inputs:
                if tensor.tensor_id in lifetimes:
                    start, _ = lifetimes[tensor.tensor_id]
                    lifetimes[tensor.tensor_id] = (start, idx)

        # Extend lifetime based on dependencies
        for dep in graph.dependencies:
            tensor_id = dep.tensor_id
            if tensor_id in lifetimes:
                start, end = lifetimes[tensor_id]
                target_idx = kernel_index.get(dep.target_id, end)
                lifetimes[tensor_id] = (start, max(end, target_idx))

        return lifetimes

    def _plan_allocation(
        self,
        graph: ComputeGraph,
        lifetimes: dict[str, tuple[int, int]]
    ) -> dict[str, int]:
        """Plan memory offsets using linear scan."""
        allocation: dict[str, int] = {}

        # Get tensor sizes
        tensor_sizes: dict[str, int] = {}
        for kernel in graph.kernels.values():
            for tensor in kernel.inputs + kernel.outputs:
                tensor_sizes[tensor.tensor_id] = tensor.size_bytes

        # Sort tensors by start time
        sorted_tensors = sorted(
            lifetimes.items(),
            key=lambda x: (x[1][0], -tensor_sizes.get(x[0], 0))
        )

        # Simple first-fit allocation
        free_regions: list[tuple[int, int]] = [(0, self.max_memory)]

        for tensor_id, (start, end) in sorted_tensors:
            size = tensor_sizes.get(tensor_id, 0)
            if size == 0:
                continue

            # Find first fitting region
            allocated = False
            for i, (region_start, region_end) in enumerate(free_regions):
                if region_end - region_start >= size:
                    allocation[tensor_id] = region_start

                    # Update free regions
                    new_start = region_start + size
                    if new_start < region_end:
                        free_regions[i] = (new_start, region_end)
                    else:
                        free_regions.pop(i)

                    allocated = True
                    break

            if not allocated:
                # Out of memory, use simple offset
                allocation[tensor_id] = 0

        return allocation


class ConstantFolder(GraphOptimizer):
    """Fold constant expressions."""

    def optimize(self, graph: ComputeGraph) -> ComputeGraph:
        """Fold constants in graph."""
        # Simplified: return as-is
        return graph


class DeadCodeEliminator(GraphOptimizer):
    """Remove unused kernels."""

    def optimize(self, graph: ComputeGraph) -> ComputeGraph:
        """Remove kernels whose outputs are not used."""
        optimized = ComputeGraph()

        # Find all used tensor IDs
        used_tensors = set()
        for tensor in graph.output_tensors:
            used_tensors.add(tensor.tensor_id)

        for dep in graph.dependencies:
            used_tensors.add(dep.tensor_id)

        # Keep kernels that produce used tensors
        for kernel_id, kernel in graph.kernels.items():
            output_ids = {t.tensor_id for t in kernel.outputs}
            if output_ids & used_tensors or not graph.get_dependents(kernel_id):
                optimized.kernels[kernel_id] = kernel

        # Keep relevant dependencies
        kept_ids = set(optimized.kernels.keys())
        optimized.dependencies = [
            dep for dep in graph.dependencies
            if dep.source_id in kept_ids and dep.target_id in kept_ids
        ]

        optimized.input_tensors = graph.input_tensors
        optimized.output_tensors = graph.output_tensors

        return optimized


class OptimizationPipeline:
    """Pipeline of optimizations."""

    def __init__(self):
        self.passes: list[tuple[str, GraphOptimizer]] = []

    def add_pass(self, name: str, optimizer: GraphOptimizer) -> None:
        """Add optimization pass."""
        self.passes.append((name, optimizer))

    def optimize(self, graph: ComputeGraph) -> tuple[ComputeGraph, OptimizationResult]:
        """Run all optimization passes."""
        original_kernels = len(graph.kernels)
        passes_applied = []
        fused_patterns: list[str] = []

        current = graph
        for name, optimizer in self.passes:
            current = optimizer.optimize(current)
            passes_applied.append(name)

            if isinstance(optimizer, KernelFuser):
                fused_patterns.extend(optimizer.fused_patterns)

        optimized_kernels = len(current.kernels)

        # Estimate speedup
        original_time = sum(k.estimated_time_us for k in graph.kernels.values())
        optimized_time = sum(k.estimated_time_us for k in current.kernels.values())
        speedup = original_time / optimized_time if optimized_time > 0 else 1.0

        result = OptimizationResult(
            original_kernels=original_kernels,
            optimized_kernels=optimized_kernels,
            estimated_speedup=speedup,
            passes_applied=passes_applied,
            fused_patterns=fused_patterns
        )

        return current, result


def create_default_pipeline() -> OptimizationPipeline:
    """Create default optimization pipeline."""
    pipeline = OptimizationPipeline()
    pipeline.add_pass("dead_code", DeadCodeEliminator())
    pipeline.add_pass("fusion", KernelFuser())
    pipeline.add_pass("memory", MemoryOptimizer())
    return pipeline


def optimize_graph(graph: ComputeGraph) -> tuple[ComputeGraph, OptimizationResult]:
    """Convenience function to optimize graph."""
    pipeline = create_default_pipeline()
    return pipeline.optimize(graph)
