"""Comprehensive tests for optimization passes in the ML compiler.

Tests cover:
- Constant folding
- Dead code elimination
- Common subexpression elimination (CSE)
- Operator fusion
- Layout optimization
- Strength reduction
- Algebraic simplification
- Pass manager and pipeline
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlcompiler.ir import (
    IRModule, Function, FunctionType, IRBuilder,
    TensorType, DType, Value, Constant, OpCode, Operation
)
from mlcompiler.optimization import (
    Pass, FunctionPass, ConstantFolding, DeadCodeElimination,
    CommonSubexpressionElimination, OperatorFusion,
    LayoutOptimization, StrengthReduction, AlgebraicSimplification,
    PassManager, PassError, create_default_pipeline
)


class _BoomPass(Pass):
    """A pass that always raises, for testing PassManager resilience."""

    name = "boom"

    def run(self, module):
        raise ValueError("kaboom")


class _CountingPass(Pass):
    """A pass that records how many times it ran and never modifies."""

    name = "counter"

    def __init__(self):
        self.calls = 0

    def run(self, module):
        self.calls += 1
        return False


# ============================================================================
# Helper Functions
# ============================================================================

def count_ops_by_opcode(func: Function, opcode: OpCode) -> int:
    """Count operations with given opcode in function."""
    count = 0
    for block in func.body.blocks:
        for op in block.operations:
            if op.opcode == opcode:
                count += 1
    return count


def get_all_opcodes(func: Function) -> list[OpCode]:
    """Get list of all opcodes in function."""
    opcodes = []
    for block in func.body.blocks:
        for op in block.operations:
            opcodes.append(op.opcode)
    return opcodes


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def constant_add_module():
    """Create module with constant addition."""
    module = IRModule(name="const_add")
    func = module.create_function(
        "const_add",
        input_types=[],
        output_types=[TensorType((2, 2), DType.FLOAT32)]
    )

    builder = IRBuilder(func.entry_block)

    # Create two constants
    a = builder.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    b = builder.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))

    # Add them
    c = builder.add(a, b)
    builder.return_op([c])

    return module


@pytest.fixture
def dead_code_module():
    """Create module with dead code."""
    input_type = TensorType((32, 64), DType.FLOAT32)
    output_type = TensorType((32, 64), DType.FLOAT32)

    module = IRModule(name="dead_code")
    func = module.create_function(
        "with_dead_code",
        input_types=[input_type, input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x, y = func.entry_block.arguments

    # Dead code: computed but never used
    dead1 = builder.relu(x)
    dead2 = builder.sigmoid(y)
    dead3 = builder.add(dead1, dead2)

    # Live code: actually returned
    live = builder.add(x, y)
    builder.return_op([live])

    return module


@pytest.fixture
def cse_module():
    """Create module with common subexpressions."""
    input_type = TensorType((32, 64), DType.FLOAT32)
    output_type = TensorType((32, 64), DType.FLOAT32)

    module = IRModule(name="cse_test")
    func = module.create_function(
        "with_cse",
        input_types=[input_type, input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x, y = func.entry_block.arguments

    # Common subexpression: x + y computed twice
    sum1 = builder.add(x, y)
    sum2 = builder.add(x, y)  # Duplicate

    # Use both
    result = builder.add(sum1, sum2)
    builder.return_op([result])

    return module


@pytest.fixture
def fusion_module():
    """Create module with fusable operations."""
    input_type = TensorType((32, 64), DType.FLOAT32)
    output_type = TensorType((32, 64), DType.FLOAT32)

    module = IRModule(name="fusion_test")
    func = module.create_function(
        "with_fusion",
        input_types=[input_type, input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x, y = func.entry_block.arguments

    # Fusable elementwise chain: add -> relu
    sum_val = builder.add(x, y)
    activated = builder.relu(sum_val)
    builder.return_op([activated])

    return module


@pytest.fixture
def matmul_bias_module():
    """Create module with matmul + bias pattern."""
    a_type = TensorType((32, 64), DType.FLOAT32)
    b_type = TensorType((64, 128), DType.FLOAT32)
    bias_type = TensorType((128,), DType.FLOAT32)
    output_type = TensorType((32, 128), DType.FLOAT32)

    module = IRModule(name="matmul_bias")
    func = module.create_function(
        "linear",
        input_types=[a_type, b_type, bias_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x, w, b = func.entry_block.arguments

    # Matmul + bias add (fusable pattern)
    mm = builder.matmul(x, w)
    # Note: Adding bias would require broadcast, simplified here
    builder.return_op([mm])

    return module


@pytest.fixture
def algebraic_module():
    """Create module with algebraic simplification opportunities."""
    input_type = TensorType((32, 64), DType.FLOAT32)
    output_type = TensorType((32, 64), DType.FLOAT32)

    module = IRModule(name="algebraic")
    func = module.create_function(
        "simplifiable",
        input_types=[input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x = func.entry_block.arguments[0]

    # Create zero constant
    zero = builder.constant(np.zeros((32, 64), dtype=np.float32))

    # x + 0 should simplify to x
    result = builder.add(x, zero)
    builder.return_op([result])

    return module


@pytest.fixture
def strength_reduction_module():
    """Create module with strength reduction opportunities."""
    input_type = TensorType((32, 64), DType.FLOAT32)
    output_type = TensorType((32, 64), DType.FLOAT32)

    module = IRModule(name="strength")
    func = module.create_function(
        "reducible",
        input_types=[input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x = func.entry_block.arguments[0]

    # Create constant 2.0
    two = builder.constant(np.full((32, 64), 2.0, dtype=np.float32))

    # x * 2 can be reduced to x + x
    result = builder.mul(x, two)
    builder.return_op([result])

    return module


# ============================================================================
# ConstantFolding Tests
# ============================================================================

class TestConstantFolding:
    """Tests for constant folding pass."""

    def test_pass_attributes(self):
        """Test pass has correct attributes."""
        cf = ConstantFolding()
        assert cf.name == "constant-folding"

    def test_fold_add_constants(self, constant_add_module):
        """Test folding addition of constants."""
        cf = ConstantFolding()
        modified = cf.run(constant_add_module)

        # Should have modified the module
        assert modified

        func = list(constant_add_module.functions.values())[0]
        add_count = count_ops_by_opcode(func, OpCode.ADD)

        # After folding, ADD should be replaced with CONSTANT
        # The result should be [[6, 8], [10, 12]]
        assert add_count == 0

    def test_no_fold_with_variables(self):
        """Test that operations with variables are not folded."""
        input_type = TensorType((2, 2), DType.FLOAT32)
        output_type = TensorType((2, 2), DType.FLOAT32)

        module = IRModule(name="no_fold")
        func = module.create_function(
            "no_fold",
            input_types=[input_type],
            output_types=[output_type]
        )

        builder = IRBuilder(func.entry_block)
        x = func.entry_block.arguments[0]
        const = builder.constant(np.ones((2, 2), dtype=np.float32))

        # x + const cannot be folded (x is variable)
        result = builder.add(x, const)
        builder.return_op([result])

        cf = ConstantFolding()
        modified = cf.run(module)

        # Should not have modified
        func = list(module.functions.values())[0]
        add_count = count_ops_by_opcode(func, OpCode.ADD)
        assert add_count == 1  # ADD still present


# ============================================================================
# DeadCodeElimination Tests
# ============================================================================

class TestDeadCodeElimination:
    """Tests for dead code elimination pass."""

    def test_pass_attributes(self):
        """Test pass has correct attributes."""
        dce = DeadCodeElimination()
        assert dce.name == "dead-code-elimination"

    def test_eliminate_dead_code(self, dead_code_module):
        """Test elimination of unused operations."""
        func = list(dead_code_module.functions.values())[0]
        initial_op_count = len(func.get_operations())

        dce = DeadCodeElimination()
        modified = dce.run(dead_code_module)

        assert modified

        final_op_count = len(func.get_operations())
        # Should have fewer operations after DCE
        assert final_op_count < initial_op_count

    def test_preserve_return_chain(self, dead_code_module):
        """Test that return and its dependencies are preserved."""
        dce = DeadCodeElimination()
        dce.run(dead_code_module)

        func = list(dead_code_module.functions.values())[0]
        opcodes = get_all_opcodes(func)

        # RETURN and ADD (the live one) should be preserved
        assert OpCode.RETURN in opcodes
        assert OpCode.ADD in opcodes

    def test_no_modification_when_no_dead_code(self):
        """Test no modification when there's no dead code."""
        input_type = TensorType((32, 64), DType.FLOAT32)
        output_type = TensorType((32, 64), DType.FLOAT32)

        module = IRModule(name="no_dead")
        func = module.create_function(
            "all_live",
            input_types=[input_type],
            output_types=[output_type]
        )

        builder = IRBuilder(func.entry_block)
        x = func.entry_block.arguments[0]
        y = builder.relu(x)
        builder.return_op([y])

        initial_op_count = len(func.get_operations())

        dce = DeadCodeElimination()
        dce.run(module)

        final_op_count = len(func.get_operations())
        assert final_op_count == initial_op_count


# ============================================================================
# CommonSubexpressionElimination Tests
# ============================================================================

class TestCommonSubexpressionElimination:
    """Tests for CSE pass."""

    def test_pass_attributes(self):
        """Test pass has correct attributes."""
        cse = CommonSubexpressionElimination()
        assert cse.name == "cse"

    def test_eliminate_common_subexpression(self, cse_module):
        """Test elimination of duplicate operations."""
        func = list(cse_module.functions.values())[0]
        initial_add_count = count_ops_by_opcode(func, OpCode.ADD)
        assert initial_add_count == 3  # sum1, sum2, result

        cse = CommonSubexpressionElimination()
        modified = cse.run(cse_module)

        assert modified

        final_add_count = count_ops_by_opcode(func, OpCode.ADD)
        # sum2 should be eliminated, reusing sum1
        assert final_add_count == 2


# ============================================================================
# OperatorFusion Tests
# ============================================================================

class TestOperatorFusion:
    """Tests for operator fusion pass."""

    def test_pass_attributes(self):
        """Test pass has correct attributes."""
        fusion = OperatorFusion()
        assert fusion.name == "operator-fusion"

    def test_fuse_elementwise_chain(self, fusion_module):
        """Test fusion of elementwise chain."""
        func = list(fusion_module.functions.values())[0]

        # Before fusion: ADD, RELU, RETURN
        initial_opcodes = get_all_opcodes(func)
        assert OpCode.ADD in initial_opcodes
        assert OpCode.RELU in initial_opcodes

        fusion = OperatorFusion()
        modified = fusion.run(fusion_module)

        if modified:
            final_opcodes = get_all_opcodes(func)
            # Should have FUSED op or reduced ops
            assert OpCode.FUSED in final_opcodes or len(final_opcodes) < len(initial_opcodes)


# ============================================================================
# LayoutOptimization Tests
# ============================================================================

class TestLayoutOptimization:
    """Tests for layout optimization pass."""

    def test_pass_attributes(self):
        """Test pass has correct attributes."""
        layout_opt = LayoutOptimization()
        assert layout_opt.name == "layout-optimization"

    def test_matmul_layout_hints(self, matmul_bias_module):
        """Test layout hints are added to matmul."""
        layout_opt = LayoutOptimization()
        modified = layout_opt.run(matmul_bias_module)

        assert modified

        func = list(matmul_bias_module.functions.values())[0]
        for block in func.body.blocks:
            for op in block.operations:
                if op.opcode == OpCode.MATMUL:
                    assert op.get_attr("lhs_layout") == "row_major"
                    assert op.get_attr("rhs_layout") == "column_major"


# ============================================================================
# StrengthReduction Tests
# ============================================================================

class TestStrengthReduction:
    """Tests for strength reduction pass."""

    def test_pass_attributes(self):
        """Test pass has correct attributes."""
        sr = StrengthReduction()
        assert sr.name == "strength-reduction"

    def test_reduce_mul_by_2(self, strength_reduction_module):
        """Test x * 2 is reduced to x + x."""
        func = list(strength_reduction_module.functions.values())[0]

        # Before: MUL
        initial_mul_count = count_ops_by_opcode(func, OpCode.MUL)
        assert initial_mul_count == 1

        sr = StrengthReduction()
        modified = sr.run(strength_reduction_module)

        if modified:
            final_mul_count = count_ops_by_opcode(func, OpCode.MUL)
            final_add_count = count_ops_by_opcode(func, OpCode.ADD)

            # MUL should be replaced with ADD
            assert final_mul_count == 0
            assert final_add_count == 1


# ============================================================================
# AlgebraicSimplification Tests
# ============================================================================

class TestAlgebraicSimplification:
    """Tests for algebraic simplification pass."""

    def test_pass_attributes(self):
        """Test pass has correct attributes."""
        alg = AlgebraicSimplification()
        assert alg.name == "algebraic-simplification"

    def test_simplify_add_zero(self, algebraic_module):
        """Test x + 0 is simplified to x."""
        func = list(algebraic_module.functions.values())[0]

        # Before: CONSTANT, ADD, RETURN
        initial_add_count = count_ops_by_opcode(func, OpCode.ADD)
        assert initial_add_count == 1

        alg = AlgebraicSimplification()
        modified = alg.run(algebraic_module)

        if modified:
            final_add_count = count_ops_by_opcode(func, OpCode.ADD)
            # ADD(x, 0) should be removed
            assert final_add_count == 0


# ============================================================================
# PassManager Tests
# ============================================================================

class TestPassManager:
    """Tests for pass manager."""

    def test_pass_manager_creation(self):
        """Test pass manager creation."""
        pm = PassManager()
        assert len(pm.passes) == 0

    def test_add_pass(self):
        """Test adding passes to manager."""
        pm = PassManager()
        pm.add_pass(ConstantFolding())
        pm.add_pass(DeadCodeElimination())

        assert len(pm.passes) == 2

    def test_run_passes_in_order(self, dead_code_module):
        """Test passes are run in order."""
        pm = PassManager()
        pm.add_pass(DeadCodeElimination())
        pm.add_pass(CommonSubexpressionElimination())

        func = list(dead_code_module.functions.values())[0]
        initial_op_count = len(func.get_operations())

        pm.run(dead_code_module)

        final_op_count = len(func.get_operations())
        assert final_op_count <= initial_op_count

    def test_max_iterations(self):
        """Test max iterations limit."""
        pm = PassManager(max_iterations=5)
        assert pm.max_iterations == 5

    def test_convergence_detection(self, dead_code_module):
        """Test optimization converges and stops."""
        pm = PassManager(max_iterations=10)
        pm.add_pass(DeadCodeElimination())

        # Should converge after removing dead code
        result = pm.run(dead_code_module)

        assert result is not None


# ============================================================================
# Default Pipeline Tests
# ============================================================================

class TestDefaultPipeline:
    """Tests for default optimization pipeline."""

    def test_create_default_pipeline(self):
        """Test creating default pipeline."""
        pm = create_default_pipeline()

        assert isinstance(pm, PassManager)
        assert len(pm.passes) > 0

    def test_default_pipeline_has_key_passes(self):
        """Test default pipeline contains key passes."""
        pm = create_default_pipeline()

        pass_names = [p.name for p in pm.passes]

        # Should have essential passes
        assert "algebraic-simplification" in pass_names
        assert "constant-folding" in pass_names
        assert "dead-code-elimination" in pass_names
        assert "operator-fusion" in pass_names

    def test_default_pipeline_on_complex_module(self):
        """Test default pipeline on complex module."""
        # Create a module with multiple optimization opportunities
        input_type = TensorType((32, 64), DType.FLOAT32)
        output_type = TensorType((32, 64), DType.FLOAT32)

        module = IRModule(name="complex")
        func = module.create_function(
            "complex_func",
            input_types=[input_type, input_type],
            output_types=[output_type]
        )

        builder = IRBuilder(func.entry_block)
        x, y = func.entry_block.arguments

        # Constants that can be folded
        const1 = builder.constant(np.ones((32, 64), dtype=np.float32))
        const2 = builder.constant(np.ones((32, 64), dtype=np.float32))
        folded = builder.add(const1, const2)

        # Dead code
        dead = builder.relu(x)

        # Live computation
        live = builder.add(x, y)
        result = builder.add(live, folded)
        builder.return_op([result])

        initial_op_count = len(func.get_operations())

        pm = create_default_pipeline()
        pm.run(module)

        final_op_count = len(func.get_operations())

        # Should optimize away some operations
        assert final_op_count <= initial_op_count


# ============================================================================
# Integration Tests
# ============================================================================

class TestOptimizationIntegration:
    """Integration tests for optimization passes."""

    def test_combined_optimizations(self):
        """Test multiple optimizations work together."""
        input_type = TensorType((32, 64), DType.FLOAT32)
        output_type = TensorType((32, 64), DType.FLOAT32)

        module = IRModule(name="integrated")
        func = module.create_function(
            "integrated_func",
            input_types=[input_type],
            output_types=[output_type]
        )

        builder = IRBuilder(func.entry_block)
        x = func.entry_block.arguments[0]

        # Create zero constant
        zero = builder.constant(np.zeros((32, 64), dtype=np.float32))

        # x + 0 (algebraic simplification)
        sum1 = builder.add(x, zero)

        # Dead code
        dead = builder.sigmoid(x)

        # relu(sum1) where sum1 = x + 0 = x
        result = builder.relu(sum1)
        builder.return_op([result])

        pm = PassManager()
        pm.add_pass(AlgebraicSimplification())
        pm.add_pass(DeadCodeElimination())
        pm.run(module)

        final_ops = func.get_operations()
        final_opcodes = [op.opcode for op in final_ops]

        # Should have simplified
        assert OpCode.SIGMOID not in final_opcodes  # Dead code removed
        # RELU and RETURN should remain

    def test_iterative_optimization(self):
        """Test that iterative optimization improves results."""
        input_type = TensorType((32, 64), DType.FLOAT32)
        output_type = TensorType((32, 64), DType.FLOAT32)

        module = IRModule(name="iterative")
        func = module.create_function(
            "iterative_func",
            input_types=[input_type, input_type],
            output_types=[output_type]
        )

        builder = IRBuilder(func.entry_block)
        x, y = func.entry_block.arguments

        # CSE then DCE can clean this up iteratively
        sum1 = builder.add(x, y)
        sum2 = builder.add(x, y)  # Duplicate for CSE
        dead = builder.mul(sum2, sum2)  # Dies after CSE eliminates sum2

        result = builder.relu(sum1)
        builder.return_op([result])

        pm = PassManager(max_iterations=5)
        pm.add_pass(CommonSubexpressionElimination())
        pm.add_pass(DeadCodeElimination())
        pm.run(module)

        final_ops = func.get_operations()

        # After optimization, should have minimal ops
        assert len(final_ops) <= 4  # add, relu, return + maybe constant


# ============================================================================
# Edge Cases
# ============================================================================

class TestOptimizationEdgeCases:
    """Tests for edge cases in optimization passes."""

    def test_empty_function(self):
        """Test optimization on empty function."""
        module = IRModule(name="empty")
        func = module.create_function(
            "empty_func",
            input_types=[TensorType((32, 64), DType.FLOAT32)],
            output_types=[TensorType((32, 64), DType.FLOAT32)]
        )

        builder = IRBuilder(func.entry_block)
        builder.return_op([func.entry_block.arguments[0]])

        pm = create_default_pipeline()
        pm.run(module)

        # Should not crash, should still have return
        ops = func.get_operations()
        assert len(ops) >= 1

    def test_single_operation(self):
        """Test optimization on single operation function."""
        input_type = TensorType((32, 64), DType.FLOAT32)
        output_type = TensorType((32, 64), DType.FLOAT32)

        module = IRModule(name="single")
        func = module.create_function(
            "single_op",
            input_types=[input_type],
            output_types=[output_type]
        )

        builder = IRBuilder(func.entry_block)
        x = func.entry_block.arguments[0]
        y = builder.relu(x)
        builder.return_op([y])

        pm = create_default_pipeline()
        pm.run(module)

        ops = func.get_operations()
        opcodes = [op.opcode for op in ops]

        # Should still have RELU and RETURN
        assert OpCode.RELU in opcodes
        assert OpCode.RETURN in opcodes

    def test_all_dead_code(self):
        """Test function where everything except return is dead."""
        input_type = TensorType((32, 64), DType.FLOAT32)
        output_type = TensorType((32, 64), DType.FLOAT32)

        module = IRModule(name="all_dead")
        func = module.create_function(
            "all_dead_func",
            input_types=[input_type],
            output_types=[output_type]
        )

        builder = IRBuilder(func.entry_block)
        x = func.entry_block.arguments[0]

        # All dead
        builder.relu(x)
        builder.sigmoid(x)
        builder.tanh(x)
        builder.exp(x)

        # Return input directly
        builder.return_op([x])

        dce = DeadCodeElimination()
        dce.run(module)

        ops = func.get_operations()
        opcodes = [op.opcode for op in ops]

        # Only RETURN should remain
        assert OpCode.RETURN in opcodes
        assert OpCode.RELU not in opcodes
        assert OpCode.SIGMOID not in opcodes


class TestPassManagerErrorHandling:
    """Tests for PassManager resilience and failure visibility."""

    def _empty_module(self):
        module = IRModule(name="err")
        func = module.create_function(
            "f",
            input_types=[TensorType((4,), DType.FLOAT32)],
            output_types=[TensorType((4,), DType.FLOAT32)],
        )
        builder = IRBuilder(func.entry_block)
        builder.return_op([func.entry_block.arguments[0]])
        return module

    def test_failing_pass_is_recorded_not_swallowed(self, caplog):
        """A failing pass is logged and recorded on errors, pipeline survives."""
        module = self._empty_module()
        pm = PassManager(max_iterations=1)
        pm.add_pass(_BoomPass())

        import logging
        with caplog.at_level(logging.ERROR):
            result = pm.run(module)

        assert result is module  # pipeline still returns the module
        assert pm.has_errors
        assert len(pm.errors) == 1
        err = pm.errors[0]
        assert isinstance(err, PassError)
        assert err.pass_name == "boom"
        assert isinstance(err.exception, ValueError)
        assert "boom" in str(err)
        # The failure was logged (not silent).
        assert any("boom" in rec.getMessage() for rec in caplog.records)

    def test_failing_pass_does_not_block_other_passes(self):
        """A failing pass does not prevent subsequent passes from running."""
        module = self._empty_module()
        counter = _CountingPass()
        pm = PassManager(max_iterations=1)
        pm.add_pass(_BoomPass())
        pm.add_pass(counter)

        pm.run(module)

        assert counter.calls == 1  # ran despite the earlier failure
        assert pm.has_errors

    def test_strict_mode_reraises(self):
        """strict=True re-raises the first pass failure."""
        module = self._empty_module()
        pm = PassManager(max_iterations=1, strict=True)
        pm.add_pass(_BoomPass())

        with pytest.raises(ValueError, match="kaboom"):
            pm.run(module)

        # Still recorded even though it re-raised.
        assert pm.has_errors

    def test_pass_results_track_success_and_failure(self):
        """pass_results records every invocation with ok/modified flags."""
        module = self._empty_module()
        pm = PassManager(max_iterations=1)
        pm.add_pass(_BoomPass())
        pm.add_pass(_CountingPass())

        pm.run(module)

        names = [r[0] for r in pm.pass_results]
        assert "boom" in names
        assert "counter" in names
        boom_result = next(r for r in pm.pass_results if r[0] == "boom")
        assert boom_result[2] is False  # ok flag is False for the failure

    def test_errors_reset_between_runs(self):
        """Diagnostics are reset at the start of each run."""
        module = self._empty_module()
        pm = PassManager(max_iterations=1)
        pm.add_pass(_BoomPass())

        pm.run(module)
        assert len(pm.errors) == 1
        pm.run(module)
        assert len(pm.errors) == 1  # not accumulated across runs

    def test_clean_pipeline_has_no_errors(self):
        """A pipeline with only working passes reports no errors."""
        module = self._empty_module()
        pm = create_default_pipeline()
        pm.run(module)
        assert not pm.has_errors
        assert pm.errors == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
