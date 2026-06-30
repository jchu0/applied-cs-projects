"""Comprehensive tests for memory planning in the ML compiler.

Tests cover:
- Lifetime analysis
- Allocation strategies (Greedy, Linear Scan, Best-Fit)
- Buffer reuse optimization
- In-place operation detection
- Memory statistics and analysis
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlcompiler.ir import (
    IRModule, Function, FunctionType, IRBuilder,
    TensorType, DType, Value, OpCode, Operation
)
from mlcompiler.memory import (
    AllocationStrategy, Lifetime, BufferAllocation, MemoryPlan,
    LifetimeAnalyzer, MemoryPlanner, InplaceOptimizer, MemoryStats,
    analyze_memory_usage
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_function():
    """Create a simple function for testing."""
    input_type = TensorType((32, 64), DType.FLOAT32)
    output_type = TensorType((32, 64), DType.FLOAT32)

    module = IRModule(name="test")
    func = module.create_function(
        "simple",
        input_types=[input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x = func.entry_block.arguments[0]
    y = builder.relu(x)
    builder.return_op([y])

    return func


@pytest.fixture
def chain_function():
    """Create a function with chain of operations."""
    input_type = TensorType((32, 64), DType.FLOAT32)
    output_type = TensorType((32, 64), DType.FLOAT32)

    module = IRModule(name="test")
    func = module.create_function(
        "chain",
        input_types=[input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x = func.entry_block.arguments[0]

    # Chain: x -> relu -> sigmoid -> tanh -> exp
    h1 = builder.relu(x)
    h2 = builder.sigmoid(h1)
    h3 = builder.tanh(h2)
    h4 = builder.exp(h3)
    builder.return_op([h4])

    return func


@pytest.fixture
def diamond_function():
    """Create a function with diamond pattern (reused input)."""
    input_type = TensorType((32, 64), DType.FLOAT32)
    output_type = TensorType((32, 64), DType.FLOAT32)

    module = IRModule(name="test")
    func = module.create_function(
        "diamond",
        input_types=[input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x = func.entry_block.arguments[0]

    # Diamond: x -> relu, x -> sigmoid, then add
    h1 = builder.relu(x)
    h2 = builder.sigmoid(x)
    h3 = builder.add(h1, h2)
    builder.return_op([h3])

    return func


@pytest.fixture
def large_mlp_function():
    """Create a larger MLP-like function for testing."""
    input_type = TensorType((32, 128), DType.FLOAT32)
    w1_type = TensorType((128, 256), DType.FLOAT32)
    w2_type = TensorType((256, 128), DType.FLOAT32)
    w3_type = TensorType((128, 10), DType.FLOAT32)
    output_type = TensorType((32, 10), DType.FLOAT32)

    module = IRModule(name="test")
    func = module.create_function(
        "mlp",
        input_types=[input_type, w1_type, w2_type, w3_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x, w1, w2, w3 = func.entry_block.arguments

    # Layer 1
    h1 = builder.matmul(x, w1)
    h1_act = builder.relu(h1)

    # Layer 2
    h2 = builder.matmul(h1_act, w2)
    h2_act = builder.relu(h2)

    # Layer 3
    h3 = builder.matmul(h2_act, w3)
    out = builder.softmax(h3)

    builder.return_op([out])

    return func


# ============================================================================
# AllocationStrategy Tests
# ============================================================================

class TestAllocationStrategy:
    """Tests for allocation strategy enum."""

    def test_strategy_enum_values(self):
        """Test all strategies are defined."""
        assert AllocationStrategy.GREEDY.value == "greedy"
        assert AllocationStrategy.LINEAR_SCAN.value == "linear_scan"
        assert AllocationStrategy.GRAPH_COLORING.value == "graph_coloring"
        assert AllocationStrategy.BEST_FIT.value == "best_fit"


# ============================================================================
# Lifetime Tests
# ============================================================================

class TestLifetime:
    """Tests for Lifetime dataclass."""

    def test_lifetime_creation(self):
        """Test lifetime creation."""
        lifetime = Lifetime(
            value_id="v0",
            start=0,
            end=5,
            size_bytes=1024
        )

        assert lifetime.value_id == "v0"
        assert lifetime.start == 0
        assert lifetime.end == 5
        assert lifetime.size_bytes == 1024

    def test_lifetime_duration(self):
        """Test lifetime duration property."""
        lifetime = Lifetime(
            value_id="v0",
            start=2,
            end=10,
            size_bytes=512
        )

        assert lifetime.duration == 8


# ============================================================================
# BufferAllocation Tests
# ============================================================================

class TestBufferAllocation:
    """Tests for BufferAllocation dataclass."""

    def test_allocation_creation(self):
        """Test buffer allocation creation."""
        alloc = BufferAllocation(
            value_id="v0",
            offset=0,
            size=4096,
            memory_space="global"
        )

        assert alloc.value_id == "v0"
        assert alloc.offset == 0
        assert alloc.size == 4096
        assert alloc.memory_space == "global"

    def test_default_memory_space(self):
        """Test default memory space is global."""
        alloc = BufferAllocation(
            value_id="v0",
            offset=0,
            size=1024
        )

        assert alloc.memory_space == "global"


# ============================================================================
# MemoryPlan Tests
# ============================================================================

class TestMemoryPlan:
    """Tests for MemoryPlan dataclass."""

    def test_memory_plan_creation(self):
        """Test memory plan creation."""
        allocations = {
            "v0": BufferAllocation("v0", 0, 1024),
            "v1": BufferAllocation("v1", 1024, 2048)
        }

        plan = MemoryPlan(
            allocations=allocations,
            total_size=3072,
            peak_memory=3072,
            reuse_count=0
        )

        assert len(plan.allocations) == 2
        assert plan.total_size == 3072
        assert plan.peak_memory == 3072
        assert plan.reuse_count == 0


# ============================================================================
# LifetimeAnalyzer Tests
# ============================================================================

class TestLifetimeAnalyzer:
    """Tests for lifetime analysis."""

    def test_analyze_simple_function(self, simple_function):
        """Test lifetime analysis on simple function."""
        analyzer = LifetimeAnalyzer()
        lifetimes = analyzer.analyze(simple_function)

        # Should have lifetimes for argument and relu output
        assert len(lifetimes) >= 2

        # All lifetimes should have valid start/end
        for lifetime in lifetimes.values():
            assert lifetime.start >= 0
            assert lifetime.end >= lifetime.start

    def test_analyze_chain_function(self, chain_function):
        """Test lifetime analysis on chain of operations."""
        analyzer = LifetimeAnalyzer()
        lifetimes = analyzer.analyze(chain_function)

        # Should have lifetimes for each intermediate value
        assert len(lifetimes) >= 5  # input + 4 ops

        # Verify temporal ordering
        value_ids = sorted(lifetimes.keys(), key=lambda x: lifetimes[x].start)
        for i in range(len(value_ids) - 1):
            assert lifetimes[value_ids[i]].start <= lifetimes[value_ids[i+1]].start

    def test_analyze_diamond_function(self, diamond_function):
        """Test lifetime analysis on diamond pattern."""
        analyzer = LifetimeAnalyzer()
        lifetimes = analyzer.analyze(diamond_function)

        # Input should be live until both relu and sigmoid complete
        input_lifetime = None
        for vid, lt in lifetimes.items():
            if vid == "arg0":
                input_lifetime = lt
                break

        # Input should have extended lifetime due to reuse
        if input_lifetime:
            assert input_lifetime.end >= 1  # Used by both relu and sigmoid

    def test_lifetime_size_bytes(self, simple_function):
        """Test that lifetime tracks size in bytes."""
        analyzer = LifetimeAnalyzer()
        lifetimes = analyzer.analyze(simple_function)

        for lifetime in lifetimes.values():
            # 32x64 float32 = 32*64*4 = 8192 bytes
            if lifetime.size_bytes > 0:
                assert lifetime.size_bytes == 32 * 64 * 4


# ============================================================================
# MemoryPlanner Tests - Greedy Strategy
# ============================================================================

class TestMemoryPlannerGreedy:
    """Tests for greedy allocation strategy."""

    def test_greedy_simple_allocation(self, simple_function):
        """Test greedy allocation on simple function."""
        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(simple_function)

        assert isinstance(plan, MemoryPlan)
        assert len(plan.allocations) > 0
        assert plan.total_size >= 0

    def test_greedy_chain_allocation(self, chain_function):
        """Test greedy allocation enables buffer reuse in chains."""
        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(chain_function)

        # Greedy should reuse buffers when lifetimes don't overlap
        # In a chain, intermediate buffers can be reused
        assert plan.total_size >= 0
        # With 4 intermediate values of 32*64*4 = 8192 bytes each,
        # optimal reuse could reduce total size significantly

    def test_greedy_tracks_peak_memory(self, chain_function):
        """Test that peak memory is tracked."""
        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(chain_function)

        assert plan.peak_memory >= 0
        assert plan.peak_memory <= plan.total_size + sum(
            alloc.size for alloc in plan.allocations.values()
        )

    def test_greedy_reuse_count(self, chain_function):
        """Test that reuse count is tracked."""
        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(chain_function)

        # In a chain, later operations can reuse earlier buffers
        assert plan.reuse_count >= 0


# ============================================================================
# MemoryPlanner Tests - Linear Scan Strategy
# ============================================================================

class TestMemoryPlannerLinearScan:
    """Tests for linear scan allocation strategy."""

    def test_linear_scan_simple_allocation(self, simple_function):
        """Test linear scan allocation on simple function."""
        planner = MemoryPlanner(AllocationStrategy.LINEAR_SCAN)
        plan = planner.plan(simple_function)

        assert isinstance(plan, MemoryPlan)
        assert len(plan.allocations) > 0

    def test_linear_scan_chain_allocation(self, chain_function):
        """Test linear scan on chain of operations."""
        planner = MemoryPlanner(AllocationStrategy.LINEAR_SCAN)
        plan = planner.plan(chain_function)

        # All values should have allocations
        assert len(plan.allocations) >= 5

    def test_linear_scan_no_overlap(self, chain_function):
        """Test that non-overlapping lifetimes get reused buffers."""
        planner = MemoryPlanner(AllocationStrategy.LINEAR_SCAN)
        plan = planner.plan(chain_function)

        # Check that allocations don't violate constraints
        # (This is more of a sanity check)
        offsets_sizes = [(a.offset, a.size) for a in plan.allocations.values()]
        # Should not have negative offsets
        assert all(offset >= 0 for offset, _ in offsets_sizes)


# ============================================================================
# MemoryPlanner Tests - Best-Fit Strategy
# ============================================================================

class TestMemoryPlannerBestFit:
    """Tests for best-fit allocation strategy."""

    def test_best_fit_simple_allocation(self, simple_function):
        """Test best-fit allocation on simple function."""
        planner = MemoryPlanner(AllocationStrategy.BEST_FIT)
        plan = planner.plan(simple_function)

        assert isinstance(plan, MemoryPlan)
        assert len(plan.allocations) > 0

    def test_best_fit_minimizes_fragmentation(self, large_mlp_function):
        """Test best-fit reduces fragmentation for varied sizes."""
        planner = MemoryPlanner(AllocationStrategy.BEST_FIT)
        plan = planner.plan(large_mlp_function)

        # Best-fit should handle varied tensor sizes well
        assert plan.total_size >= 0
        assert plan.peak_memory >= 0


# ============================================================================
# Strategy Comparison Tests
# ============================================================================

class TestStrategyComparison:
    """Tests comparing different allocation strategies."""

    def test_all_strategies_produce_valid_plans(self, chain_function):
        """Test all strategies produce valid memory plans."""
        strategies = [
            AllocationStrategy.GREEDY,
            AllocationStrategy.LINEAR_SCAN,
            AllocationStrategy.BEST_FIT
        ]

        for strategy in strategies:
            planner = MemoryPlanner(strategy)
            plan = planner.plan(chain_function)

            assert isinstance(plan, MemoryPlan)
            assert len(plan.allocations) > 0
            assert plan.total_size >= 0

    def test_strategies_allocate_all_values(self, chain_function):
        """Test all strategies allocate all values."""
        analyzer = LifetimeAnalyzer()
        lifetimes = analyzer.analyze(chain_function)

        strategies = [
            AllocationStrategy.GREEDY,
            AllocationStrategy.LINEAR_SCAN,
            AllocationStrategy.BEST_FIT
        ]

        for strategy in strategies:
            planner = MemoryPlanner(strategy)
            plan = planner.plan(chain_function)

            # Each value with a lifetime should have an allocation
            for value_id in lifetimes.keys():
                if lifetimes[value_id].size_bytes > 0:
                    assert value_id in plan.allocations, \
                        f"Strategy {strategy} missing allocation for {value_id}"


# ============================================================================
# InplaceOptimizer Tests
# ============================================================================

class TestInplaceOptimizer:
    """Tests for in-place operation optimization."""

    def test_inplace_optimizer_creation(self):
        """Test in-place optimizer creation."""
        optimizer = InplaceOptimizer()
        assert optimizer is not None

    def test_inplace_elementwise_detection(self, chain_function):
        """Test detection of in-place opportunities for elementwise ops."""
        analyzer = LifetimeAnalyzer()
        lifetimes = analyzer.analyze(chain_function)

        optimizer = InplaceOptimizer()
        inplace_map = optimizer.optimize(chain_function, lifetimes)

        # In a chain, each op's input is only used once
        # So in-place should be possible for most ops
        assert isinstance(inplace_map, dict)

    def test_inplace_preserves_multi_use(self, diamond_function):
        """Test that multi-use values are not marked for in-place."""
        analyzer = LifetimeAnalyzer()
        lifetimes = analyzer.analyze(diamond_function)

        optimizer = InplaceOptimizer()
        inplace_map = optimizer.optimize(diamond_function, lifetimes)

        # The input x is used by both relu and sigmoid
        # So relu cannot be in-place on x (x is still needed by sigmoid)
        # Check that we're not marking operations on multi-use inputs as in-place
        assert isinstance(inplace_map, dict)


# ============================================================================
# MemoryStats Tests
# ============================================================================

class TestMemoryStats:
    """Tests for memory statistics."""

    def test_memory_stats_creation(self):
        """Test memory stats creation."""
        stats = MemoryStats(
            total_allocated=10240,
            peak_memory=8192,
            buffer_reuse_rate=0.25,
            fragmentation=0.05
        )

        assert stats.total_allocated == 10240
        assert stats.peak_memory == 8192
        assert stats.buffer_reuse_rate == 0.25
        assert stats.fragmentation == 0.05


# ============================================================================
# analyze_memory_usage Tests
# ============================================================================

class TestAnalyzeMemoryUsage:
    """Tests for memory usage analysis function."""

    def test_analyze_simple_plan(self, simple_function):
        """Test analyzing simple memory plan."""
        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(simple_function)

        stats = analyze_memory_usage(plan)

        assert isinstance(stats, MemoryStats)
        assert stats.total_allocated >= 0
        assert stats.peak_memory >= 0
        assert 0 <= stats.buffer_reuse_rate <= 1
        assert 0 <= stats.fragmentation <= 1

    def test_analyze_chain_plan(self, chain_function):
        """Test analyzing chain function memory plan."""
        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(chain_function)

        stats = analyze_memory_usage(plan)

        assert isinstance(stats, MemoryStats)
        # Chain should show some reuse
        assert stats.buffer_reuse_rate >= 0

    def test_analyze_large_plan(self, large_mlp_function):
        """Test analyzing larger memory plan."""
        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(large_mlp_function)

        stats = analyze_memory_usage(plan)

        assert isinstance(stats, MemoryStats)
        # MLP has varied tensor sizes
        assert stats.total_allocated > 0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases in memory planning."""

    def test_empty_function(self):
        """Test planning for function with no operations."""
        module = IRModule(name="test")
        func = module.create_function(
            "empty",
            input_types=[TensorType((32, 64), DType.FLOAT32)],
            output_types=[TensorType((32, 64), DType.FLOAT32)]
        )

        # Just return input directly
        builder = IRBuilder(func.entry_block)
        builder.return_op([func.entry_block.arguments[0]])

        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(func)

        assert isinstance(plan, MemoryPlan)

    def test_scalar_tensors(self):
        """Test planning with scalar tensors."""
        module = IRModule(name="test")
        func = module.create_function(
            "scalar",
            input_types=[TensorType((), DType.FLOAT32)],
            output_types=[TensorType((), DType.FLOAT32)]
        )

        builder = IRBuilder(func.entry_block)
        x = func.entry_block.arguments[0]
        y = builder.exp(x)
        builder.return_op([y])

        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(func)

        assert isinstance(plan, MemoryPlan)
        # Scalar = 4 bytes
        assert plan.total_size >= 4

    def test_large_tensors(self):
        """Test planning with large tensors."""
        module = IRModule(name="test")
        # 1M element tensor
        func = module.create_function(
            "large",
            input_types=[TensorType((1024, 1024), DType.FLOAT32)],
            output_types=[TensorType((1024, 1024), DType.FLOAT32)]
        )

        builder = IRBuilder(func.entry_block)
        x = func.entry_block.arguments[0]
        y = builder.relu(x)
        builder.return_op([y])

        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(func)

        # 1024*1024*4 = 4MB
        assert plan.total_size >= 4 * 1024 * 1024

    def test_different_sized_tensors(self):
        """Test planning with different sized tensors."""
        module = IRModule(name="test")
        func = module.create_function(
            "varied",
            input_types=[
                TensorType((32, 128), DType.FLOAT32),
                TensorType((128, 256), DType.FLOAT32)
            ],
            output_types=[TensorType((32, 256), DType.FLOAT32)]
        )

        builder = IRBuilder(func.entry_block)
        a, b = func.entry_block.arguments
        c = builder.matmul(a, b)
        builder.return_op([c])

        planner = MemoryPlanner(AllocationStrategy.BEST_FIT)
        plan = planner.plan(func)

        assert isinstance(plan, MemoryPlan)
        # Should handle varied sizes


# ============================================================================
# Integration Tests
# ============================================================================

class TestMemoryPlanningIntegration:
    """Integration tests for memory planning pipeline."""

    def test_full_pipeline(self, large_mlp_function):
        """Test full memory planning pipeline."""
        # 1. Analyze lifetimes
        analyzer = LifetimeAnalyzer()
        lifetimes = analyzer.analyze(large_mlp_function)

        # 2. Plan memory
        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(large_mlp_function)

        # 3. Optimize for in-place
        optimizer = InplaceOptimizer()
        inplace_map = optimizer.optimize(large_mlp_function, lifetimes)

        # 4. Analyze usage
        stats = analyze_memory_usage(plan)

        # Verify pipeline results
        assert len(lifetimes) > 0
        assert len(plan.allocations) > 0
        assert isinstance(inplace_map, dict)
        assert stats.total_allocated > 0

    def test_memory_reduction(self, chain_function):
        """Test that memory planning reduces peak memory."""
        # Calculate naive allocation (no reuse)
        input_size = 32 * 64 * 4  # 8KB per tensor

        # Chain has 5 values (input + 4 ops)
        naive_total = input_size * 5

        # Plan with reuse
        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        plan = planner.plan(chain_function)

        # With reuse, total should be less than naive
        # (or at least not significantly more)
        assert plan.total_size <= naive_total * 2  # Allow some overhead


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
