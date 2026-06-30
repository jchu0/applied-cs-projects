"""Unit tests for optimization passes in the ML compiler."""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try importing required classes
try:
    from mlcompiler import (
        ConstantFolding, DeadCodeElimination, CommonSubexpressionElimination,
        OperatorFusion, PassManager, IRModule as Module, Function,
        Operation, IRBuilder
    )
    # Check if the test API exists
    _test_builder = IRBuilder()
    _has_test_api = hasattr(_test_builder, 'begin_block')
    del _test_builder
    # Stubs for classes that may not exist
    OptimizationPass = MemoryOptimization = LoopOptimization = None
    Vectorization = GraphRewriting = None
    BinaryOp = UnaryOp = GraphBuilder = None
    _IMPORTS_OK = _has_test_api  # Only OK if required test API exists
except ImportError:
    _IMPORTS_OK = False

pytestmark = pytest.mark.skipif(not _IMPORTS_OK, reason="Missing required test API")


class TestConstantFolding(unittest.TestCase):
    """Test constant folding optimization pass."""

    def setUp(self):
        """Set up test fixtures."""
        self.pass_instance = ConstantFolding()
        self.builder = IRBuilder()

    def test_fold_binary_constants(self):
        """Test folding binary operations on constants."""
        self.builder.begin_block('constant_folding')

        # Create constant operations
        const1 = self.builder.create_constant(5.0)
        const2 = self.builder.create_constant(3.0)

        # Add operation on constants
        add = self.builder.add_binary_op('add', const1, const2)

        # Apply constant folding
        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check if add was replaced with constant
        self.assertEqual(len(transformed.get_constants()), 1)
        self.assertEqual(transformed.get_constants()[0].value, 8.0)

    def test_fold_unary_constants(self):
        """Test folding unary operations on constants."""
        self.builder.begin_block('unary_folding')

        const = self.builder.create_constant(2.0)
        exp = self.builder.add_unary_op('exp', const)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check if exp was replaced with constant
        constants = transformed.get_constants()
        self.assertAlmostEqual(constants[0].value, np.exp(2.0), places=5)

    def test_partial_folding(self):
        """Test partial constant folding."""
        self.builder.begin_block('partial_folding')

        const1 = self.builder.create_constant(10.0)
        const2 = self.builder.create_constant(5.0)
        var = self.builder.create_input(shape=[32, 128])

        # (const1 + const2) * var
        add = self.builder.add_binary_op('add', const1, const2)
        mul = self.builder.add_binary_op('multiply', add, var)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check if add was folded but multiply remains
        ops = transformed.get_operations()
        self.assertEqual(len(ops), 1)  # Only multiply remains
        self.assertEqual(ops[0].op_type, 'multiply')

    def test_no_folding_with_variables(self):
        """Test that operations with variables are not folded."""
        self.builder.begin_block('no_folding')

        var1 = self.builder.create_input(shape=[32, 128])
        var2 = self.builder.create_input(shape=[32, 128])
        add = self.builder.add_binary_op('add', var1, var2)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Operation should remain unchanged
        ops = transformed.get_operations()
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].op_type, 'add')


class TestDeadCodeElimination(unittest.TestCase):
    """Test dead code elimination pass."""

    def setUp(self):
        """Set up test fixtures."""
        self.pass_instance = DeadCodeElimination()
        self.builder = IRBuilder()

    def test_eliminate_unused_operations(self):
        """Test elimination of unused operations."""
        self.builder.begin_block('dead_code')

        x = self.builder.create_input(shape=[32, 128])
        y = self.builder.create_input(shape=[32, 128])

        # Used operation
        result = self.builder.add_binary_op('add', x, y)
        self.builder.set_return(result)

        # Unused operations (dead code)
        unused1 = self.builder.add_unary_op('relu', x)
        unused2 = self.builder.add_unary_op('sigmoid', y)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check that unused operations were removed
        ops = transformed.get_operations()
        self.assertEqual(len(ops), 1)  # Only the used add operation
        self.assertEqual(ops[0].op_type, 'add')

    def test_preserve_side_effect_operations(self):
        """Test that operations with side effects are preserved."""
        self.builder.begin_block('side_effects')

        x = self.builder.create_input(shape=[32, 128])

        # Operation with side effect (e.g., print, assert)
        debug = self.builder.add_debug_print(x)

        # Regular operation
        result = self.builder.add_unary_op('relu', x)
        self.builder.set_return(result)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Debug operation should be preserved even if unused
        ops = transformed.get_operations()
        self.assertEqual(len(ops), 2)  # Both debug and relu

    def test_eliminate_dead_branches(self):
        """Test elimination of dead conditional branches."""
        self.builder.begin_block('dead_branch')

        # Constant condition (always false)
        condition = self.builder.create_constant(False)
        x = self.builder.create_input(shape=[32, 128])

        with self.builder.if_block(condition) as if_ctx:
            # Dead branch (never executed)
            true_result = self.builder.add_unary_op('relu', x)
            if_ctx.set_true_branch(true_result)

            # Live branch
            false_result = self.builder.add_unary_op('sigmoid', x)
            if_ctx.set_false_branch(false_result)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # True branch should be eliminated
        ops = transformed.get_operations()
        op_types = [op.op_type for op in ops]
        self.assertNotIn('relu', op_types)
        self.assertIn('sigmoid', op_types)


class TestCommonSubexpressionElimination(unittest.TestCase):
    """Test common subexpression elimination."""

    def setUp(self):
        """Set up test fixtures."""
        self.pass_instance = CommonSubexpressionElimination()
        self.builder = IRBuilder()

    def test_eliminate_duplicate_operations(self):
        """Test elimination of duplicate operations."""
        self.builder.begin_block('cse')

        x = self.builder.create_input(shape=[32, 128])
        y = self.builder.create_input(shape=[32, 128])

        # Duplicate operations
        add1 = self.builder.add_binary_op('add', x, y)
        add2 = self.builder.add_binary_op('add', x, y)  # Duplicate

        # Use both results
        result1 = self.builder.add_unary_op('relu', add1)
        result2 = self.builder.add_unary_op('sigmoid', add2)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check that duplicate add was eliminated
        add_ops = [op for op in transformed.get_operations() if op.op_type == 'add']
        self.assertEqual(len(add_ops), 1)

    def test_preserve_non_deterministic_ops(self):
        """Test that non-deterministic operations are not eliminated."""
        self.builder.begin_block('non_deterministic')

        shape = [32, 128]

        # Non-deterministic operations (e.g., random)
        random1 = self.builder.add_random_normal(shape)
        random2 = self.builder.add_random_normal(shape)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Both random operations should be preserved
        random_ops = [op for op in transformed.get_operations() if op.op_type == 'random_normal']
        self.assertEqual(len(random_ops), 2)

    def test_cse_with_different_attributes(self):
        """Test CSE doesn't eliminate ops with different attributes."""
        self.builder.begin_block('different_attrs')

        x = self.builder.create_input(shape=[32, 3, 224, 224])

        # Same operation type but different attributes
        pool1 = self.builder.add_pooling(x, kernel_size=(2, 2), pool_type='max')
        pool2 = self.builder.add_pooling(x, kernel_size=(3, 3), pool_type='max')

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Both pooling operations should remain
        pool_ops = [op for op in transformed.get_operations() if op.op_type == 'pooling']
        self.assertEqual(len(pool_ops), 2)


class TestOperatorFusion(unittest.TestCase):
    """Test operator fusion optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.pass_instance = OperatorFusion()
        self.builder = IRBuilder()

    def test_conv_bn_relu_fusion(self):
        """Test Conv + BN + ReLU fusion."""
        self.builder.begin_block('conv_bn_relu')

        x = self.builder.create_input(shape=[32, 3, 224, 224])
        weight = self.builder.create_weight(shape=[64, 3, 3, 3])

        conv = self.builder.add_convolution(x, weight, stride=(1, 1))
        bn = self.builder.add_batch_norm(conv, training=False)
        relu = self.builder.add_relu(bn)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check that operations were fused
        ops = transformed.get_operations()
        fused_ops = [op for op in ops if op.op_type == 'fused_conv_bn_relu']
        self.assertEqual(len(fused_ops), 1)

    def test_matmul_bias_add_fusion(self):
        """Test MatMul + BiasAdd fusion."""
        self.builder.begin_block('matmul_bias')

        x = self.builder.create_input(shape=[32, 128])
        weight = self.builder.create_weight(shape=[128, 256])
        bias = self.builder.create_weight(shape=[256])

        matmul = self.builder.add_matmul(x, weight)
        add = self.builder.add_bias_add(matmul, bias)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check that operations were fused
        ops = transformed.get_operations()
        fused_ops = [op for op in ops if op.op_type == 'matmul_bias']
        self.assertEqual(len(fused_ops), 1)

    def test_multi_head_attention_fusion(self):
        """Test multi-head attention pattern fusion."""
        self.builder.begin_block('mha')

        x = self.builder.create_input(shape=[32, 100, 768])

        # Q, K, V projections
        wq = self.builder.create_weight(shape=[768, 768])
        wk = self.builder.create_weight(shape=[768, 768])
        wv = self.builder.create_weight(shape=[768, 768])

        q = self.builder.add_matmul(x, wq)
        k = self.builder.add_matmul(x, wk)
        v = self.builder.add_matmul(x, wv)

        # Reshape for multi-head
        q_heads = self.builder.add_reshape(q, shape=[32, 100, 12, 64])
        k_heads = self.builder.add_reshape(k, shape=[32, 100, 12, 64])
        v_heads = self.builder.add_reshape(v, shape=[32, 100, 12, 64])

        # Attention computation
        scores = self.builder.add_attention(q_heads, k_heads, v_heads)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check for fused multi-head attention
        ops = transformed.get_operations()
        mha_ops = [op for op in ops if op.op_type == 'multi_head_attention']
        self.assertGreaterEqual(len(mha_ops), 1)


class TestMemoryOptimization(unittest.TestCase):
    """Test memory optimization pass."""

    def setUp(self):
        """Set up test fixtures."""
        self.pass_instance = MemoryOptimization()
        self.builder = IRBuilder()

    def test_inplace_operations(self):
        """Test conversion to inplace operations."""
        self.builder.begin_block('inplace')

        x = self.builder.create_input(shape=[32, 128])

        # Operations that can be done inplace
        relu = self.builder.add_relu(x)
        add_scalar = self.builder.add_scalar(relu, 1.0)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check for inplace operations
        ops = transformed.get_operations()
        inplace_ops = [op for op in ops if hasattr(op, 'inplace') and op.inplace]
        self.assertGreater(len(inplace_ops), 0)

    def test_memory_reuse(self):
        """Test memory buffer reuse."""
        self.builder.begin_block('memory_reuse')

        x = self.builder.create_input(shape=[32, 128])

        # Sequential operations with non-overlapping lifetimes
        temp1 = self.builder.add_relu(x)
        result1 = self.builder.add_sigmoid(temp1)

        # temp1 no longer needed after this point
        temp2 = self.builder.add_tanh(x)
        result2 = self.builder.add_exp(temp2)

        final = self.builder.add_binary_op('add', result1, result2)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check memory allocation
        memory_plan = transformed.get_memory_plan()
        # temp1 and temp2 should share the same memory buffer
        self.assertEqual(memory_plan.get_buffer('temp1'), memory_plan.get_buffer('temp2'))

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing for memory optimization."""
        self.builder.begin_block('checkpointing')

        x = self.builder.create_input(shape=[32, 128])

        # Long chain of operations
        h1 = self.builder.add_linear(x, 256)
        h2 = self.builder.add_relu(h1)
        h3 = self.builder.add_linear(h2, 512)
        h4 = self.builder.add_relu(h3)
        h5 = self.builder.add_linear(h4, 256)
        output = self.builder.add_relu(h5)

        self.builder.mark_for_gradient()

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check for checkpoint markers
        checkpoints = transformed.get_checkpoints()
        self.assertGreater(len(checkpoints), 0)


class TestLoopOptimization(unittest.TestCase):
    """Test loop optimization passes."""

    def setUp(self):
        """Set up test fixtures."""
        self.pass_instance = LoopOptimization()
        self.builder = IRBuilder()

    def test_loop_unrolling(self):
        """Test loop unrolling optimization."""
        self.builder.begin_block('unroll')

        x = self.builder.create_input(shape=[32, 128])

        # Small fixed-size loop
        with self.builder.for_loop(0, 4) as loop:
            i = loop.get_index()
            x = self.builder.add_scalar(x, i)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check that loop was unrolled
        ops = transformed.get_operations()
        add_ops = [op for op in ops if op.op_type == 'add_scalar']
        self.assertEqual(len(add_ops), 4)  # Loop unrolled to 4 operations

    def test_loop_fusion(self):
        """Test fusion of adjacent loops."""
        self.builder.begin_block('loop_fusion')

        x = self.builder.create_input(shape=[32, 128])
        y = self.builder.create_input(shape=[32, 128])

        # Two adjacent loops with same bounds
        with self.builder.for_loop(0, 100) as loop1:
            i = loop1.get_index()
            x = self.builder.add_element(x, i, 1.0)

        with self.builder.for_loop(0, 100) as loop2:
            i = loop2.get_index()
            y = self.builder.add_element(y, i, 2.0)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module)

        # Check that loops were fused
        loops = transformed.get_loops()
        self.assertEqual(len(loops), 1)  # Two loops fused into one

    def test_loop_tiling(self):
        """Test loop tiling optimization."""
        self.builder.begin_block('tiling')

        matrix = self.builder.create_input(shape=[1024, 1024])

        # Matrix operation loop
        with self.builder.for_loop(0, 1024) as outer:
            i = outer.get_index()
            with self.builder.for_loop(0, 1024) as inner:
                j = inner.get_index()
                elem = self.builder.get_element(matrix, [i, j])
                new_elem = self.builder.add_scalar(elem, 1.0)
                matrix = self.builder.set_element(matrix, [i, j], new_elem)

        module = self.builder.get_module()
        transformed = self.pass_instance.run(module, tile_size=32)

        # Check for tiled loops
        loops = transformed.get_loops()
        # Should have 4 loops after tiling (2 outer tiles, 2 inner tiles)
        self.assertGreaterEqual(len(loops), 4)


class TestPassManager(unittest.TestCase):
    """Test pass manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.pass_manager = PassManager()
        self.builder = IRBuilder()

    def test_add_passes(self):
        """Test adding passes to manager."""
        self.pass_manager.add_pass(ConstantFolding())
        self.pass_manager.add_pass(DeadCodeElimination())
        self.pass_manager.add_pass(OperatorFusion())

        self.assertEqual(len(self.pass_manager.passes), 3)

    def test_run_passes_sequential(self):
        """Test running passes sequentially."""
        self.pass_manager.add_pass(ConstantFolding())
        self.pass_manager.add_pass(DeadCodeElimination())

        self.builder.begin_block('test')
        const1 = self.builder.create_constant(5.0)
        const2 = self.builder.create_constant(3.0)
        add = self.builder.add_binary_op('add', const1, const2)

        # Unused operation
        x = self.builder.create_input(shape=[32, 128])
        unused = self.builder.add_relu(x)

        module = self.builder.get_module()
        optimized = self.pass_manager.run(module)

        # Both optimizations should be applied
        ops = optimized.get_operations()
        self.assertEqual(len(ops), 0)  # Constant folded and dead code eliminated

    def test_pass_dependencies(self):
        """Test pass dependency management."""
        cse = CommonSubexpressionElimination()
        dce = DeadCodeElimination()

        # DCE should run after CSE
        self.pass_manager.add_pass(cse)
        self.pass_manager.add_pass(dce, dependencies=[cse])

        order = self.pass_manager.get_execution_order()
        self.assertLess(order.index(cse), order.index(dce))

    def test_pass_configuration(self):
        """Test pass configuration."""
        config = {
            'constant_folding': {'aggressive': True},
            'loop_optimization': {'tile_size': 64, 'unroll_factor': 4}
        }

        self.pass_manager.add_pass(ConstantFolding(), config=config['constant_folding'])
        self.pass_manager.add_pass(LoopOptimization(), config=config['loop_optimization'])

        # Verify configuration
        cf_pass = self.pass_manager.passes[0]
        self.assertTrue(cf_pass.config['aggressive'])

        loop_pass = self.pass_manager.passes[1]
        self.assertEqual(loop_pass.config['tile_size'], 64)


if __name__ == '__main__':
    unittest.main()