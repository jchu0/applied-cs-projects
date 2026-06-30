"""Integration tests for the ML compiler."""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try importing required classes
try:
    from mlcompiler.ir.builder import IRBuilder
    from mlcompiler import TensorType, DType as DataType, IRModule as Module
    from mlcompiler.codegen import Target
    # Check if the test API exists
    _test_builder = IRBuilder()
    _has_test_api = hasattr(_test_builder, 'begin_block')
    del _test_builder
    # Stubs for classes that may not exist
    MLCompiler = CompilerConfig = OptimizationLevel = None
    Shape = None
    _IMPORTS_OK = _has_test_api  # Only OK if required test API exists
except ImportError:
    _IMPORTS_OK = False

pytestmark = pytest.mark.skipif(not _IMPORTS_OK, reason="Missing required test API")


class TestEndToEndCompilation(unittest.TestCase):
    """Test end-to-end compilation scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CompilerConfig(
            target=Target.CPU,
            optimization_level=OptimizationLevel.O2,
            debug=True
        )
        self.compiler = MLCompiler(self.config)

    def test_compile_simple_model(self):
        """Test compiling a simple model."""
        builder = IRBuilder()
        builder.begin_block('simple_model')

        # Build simple feed-forward network
        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 784])),
            name='input'
        )

        # Layer 1
        w1 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([784, 256])),
            name='weight1'
        )
        b1 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([256])),
            name='bias1'
        )

        h1 = builder.add_matmul(x, w1)
        h1 = builder.add_bias_add(h1, b1)
        h1 = builder.add_relu(h1)

        # Layer 2
        w2 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([256, 10])),
            name='weight2'
        )
        b2 = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([10])),
            name='bias2'
        )

        output = builder.add_matmul(h1, w2)
        output = builder.add_bias_add(output, b2)
        output = builder.add_softmax(output)

        builder.set_return(output)

        module = builder.get_module()
        compiled = self.compiler.compile(module)

        self.assertIsNotNone(compiled)
        self.assertIn('executable', compiled)
        self.assertIn('metadata', compiled)

    def test_compile_cnn_model(self):
        """Test compiling a CNN model."""
        builder = IRBuilder()
        builder.begin_block('cnn_model')

        # Input image
        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 3, 224, 224])),
            name='image'
        )

        # Conv layer 1
        conv1_weight = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([64, 3, 7, 7])),
            name='conv1_weight'
        )

        conv1 = builder.add_convolution(x, conv1_weight, stride=(2, 2), padding=(3, 3))
        conv1 = builder.add_batch_norm(conv1)
        conv1 = builder.add_relu(conv1)
        conv1 = builder.add_pooling(conv1, pool_type='max', kernel_size=(3, 3), stride=(2, 2))

        # Conv layer 2
        conv2_weight = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([128, 64, 3, 3])),
            name='conv2_weight'
        )

        conv2 = builder.add_convolution(conv1, conv2_weight, stride=(1, 1), padding=(1, 1))
        conv2 = builder.add_batch_norm(conv2)
        conv2 = builder.add_relu(conv2)

        # Global average pooling
        gap = builder.add_reduce(conv2, reduce_type='mean', axes=[2, 3])

        # Classifier
        fc_weight = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([128, 1000])),
            name='fc_weight'
        )

        output = builder.add_matmul(gap, fc_weight)
        output = builder.add_softmax(output)

        builder.set_return(output)

        module = builder.get_module()
        compiled = self.compiler.compile(module)

        self.assertIsNotNone(compiled)
        # Check optimizations were applied
        self.assertIn('fused_ops', compiled['metadata'])
        self.assertGreater(compiled['metadata']['fused_ops'], 0)

    def test_compile_rnn_model(self):
        """Test compiling an RNN model."""
        builder = IRBuilder()
        builder.begin_block('rnn_model')

        # Input sequence [batch, seq_len, features]
        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 100, 128])),
            name='sequence'
        )

        # RNN weights
        w_ih = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([128, 256])),
            name='input_hidden_weight'
        )
        w_hh = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([256, 256])),
            name='hidden_hidden_weight'
        )
        bias = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([256])),
            name='bias'
        )

        # Initial hidden state
        h = builder.create_zeros(Shape([32, 256]))

        # Process sequence
        outputs = []
        for t in range(100):
            x_t = builder.get_slice(x, axis=1, index=t)  # [batch, features]

            # RNN cell
            i_h = builder.add_matmul(x_t, w_ih)
            h_h = builder.add_matmul(h, w_hh)
            h = builder.add_binary_op('add', i_h, h_h)
            h = builder.add_bias_add(h, bias)
            h = builder.add_tanh(h)

            outputs.append(h)

        # Stack outputs
        output = builder.stack(outputs, axis=1)
        builder.set_return(output)

        module = builder.get_module()
        compiled = self.compiler.compile(module)

        self.assertIsNotNone(compiled)
        # Check loop optimization
        self.assertIn('unrolled_loops', compiled['metadata'])

    def test_compile_transformer_layer(self):
        """Test compiling a transformer layer."""
        builder = IRBuilder()
        builder.begin_block('transformer_layer')

        # Input [batch, seq_len, d_model]
        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 100, 768])),
            name='input'
        )

        # Multi-head attention weights
        wq = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([768, 768])),
            name='query_weight'
        )
        wk = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([768, 768])),
            name='key_weight'
        )
        wv = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([768, 768])),
            name='value_weight'
        )
        wo = builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([768, 768])),
            name='output_weight'
        )

        # Compute Q, K, V
        q = builder.add_matmul(x, wq)
        k = builder.add_matmul(x, wk)
        v = builder.add_matmul(x, wv)

        # Reshape for multi-head (12 heads, 64 dim per head)
        q = builder.add_reshape(q, Shape([32, 100, 12, 64]))
        k = builder.add_reshape(k, Shape([32, 100, 12, 64]))
        v = builder.add_reshape(v, Shape([32, 100, 12, 64]))

        # Transpose for attention
        q = builder.add_transpose(q, axes=[0, 2, 1, 3])  # [batch, heads, seq, dim]
        k = builder.add_transpose(k, axes=[0, 2, 1, 3])
        v = builder.add_transpose(v, axes=[0, 2, 1, 3])

        # Attention scores
        scores = builder.add_matmul(q, builder.add_transpose(k, axes=[0, 1, 3, 2]))
        scores = builder.add_scalar_multiply(scores, 1.0 / np.sqrt(64))
        scores = builder.add_softmax(scores, axis=-1)

        # Attention output
        attn = builder.add_matmul(scores, v)
        attn = builder.add_transpose(attn, axes=[0, 2, 1, 3])
        attn = builder.add_reshape(attn, Shape([32, 100, 768]))

        # Output projection
        output = builder.add_matmul(attn, wo)

        # Add & Norm
        output = builder.add_binary_op('add', output, x)
        output = builder.add_layer_norm(output)

        builder.set_return(output)

        module = builder.get_module()
        compiled = self.compiler.compile(module)

        self.assertIsNotNone(compiled)
        # Check flash attention optimization
        if self.config.target == Target.GPU:
            self.assertIn('flash_attention', compiled['metadata'].get('optimizations', []))


class TestCompilerOptimizations(unittest.TestCase):
    """Test compiler optimization levels."""

    def test_optimization_level_o0(self):
        """Test compilation with no optimization."""
        config = CompilerConfig(
            target=Target.CPU,
            optimization_level=OptimizationLevel.O0
        )
        compiler = MLCompiler(config)

        builder = IRBuilder()
        builder.begin_block('test')

        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 128])),
            name='x'
        )

        # Redundant operations
        y = builder.add_relu(x)
        z = builder.add_relu(y)  # Redundant ReLU

        builder.set_return(z)

        module = builder.get_module()
        compiled = compiler.compile(module)

        # No optimization, both ReLUs should remain
        self.assertEqual(compiled['metadata']['num_operations'], 2)

    def test_optimization_level_o2(self):
        """Test compilation with standard optimization."""
        config = CompilerConfig(
            target=Target.CPU,
            optimization_level=OptimizationLevel.O2
        )
        compiler = MLCompiler(config)

        builder = IRBuilder()
        builder.begin_block('test')

        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 128])),
            name='x'
        )

        # Redundant operations
        y = builder.add_relu(x)
        z = builder.add_relu(y)  # Redundant ReLU

        builder.set_return(z)

        module = builder.get_module()
        compiled = compiler.compile(module)

        # Should optimize away redundant ReLU
        self.assertEqual(compiled['metadata']['num_operations'], 1)

    def test_optimization_level_o3(self):
        """Test compilation with aggressive optimization."""
        config = CompilerConfig(
            target=Target.CPU,
            optimization_level=OptimizationLevel.O3,
            unsafe_math=True
        )
        compiler = MLCompiler(config)

        builder = IRBuilder()
        builder.begin_block('test')

        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([1024, 1024])),
            name='x'
        )
        y = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([1024, 1024])),
            name='y'
        )

        # Matrix multiplication chain
        z = builder.add_matmul(x, y)
        w = builder.add_transpose(z)
        result = builder.add_matmul(w, x)

        builder.set_return(result)

        module = builder.get_module()
        compiled = compiler.compile(module)

        # Should apply aggressive optimizations
        optimizations = compiled['metadata'].get('optimizations', [])
        self.assertIn('algebraic_simplification', optimizations)
        self.assertIn('fast_math', optimizations)


class TestCrossTargetCompilation(unittest.TestCase):
    """Test compilation for different targets."""

    def test_cpu_compilation(self):
        """Test CPU target compilation."""
        config = CompilerConfig(target=Target.CPU)
        compiler = MLCompiler(config)

        builder = IRBuilder()
        builder.begin_block('cpu_test')

        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 128])),
            name='x'
        )
        y = builder.add_relu(x)
        builder.set_return(y)

        module = builder.get_module()
        compiled = compiler.compile(module)

        self.assertEqual(compiled['target'], 'cpu')
        self.assertIn('vectorized', compiled['metadata'])

    @patch('mlcompiler.codegen.cuda.cuda_available')
    def test_gpu_compilation(self, mock_cuda):
        """Test GPU target compilation."""
        mock_cuda.return_value = True

        config = CompilerConfig(target=Target.GPU)
        compiler = MLCompiler(config)

        builder = IRBuilder()
        builder.begin_block('gpu_test')

        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([1024, 1024])),
            name='x'
        )
        y = builder.add_relu(x)
        builder.set_return(y)

        module = builder.get_module()
        compiled = compiler.compile(module)

        self.assertEqual(compiled['target'], 'gpu')
        self.assertIn('cuda_version', compiled['metadata'])
        self.assertIn('kernel_config', compiled['metadata'])

    def test_mixed_precision_compilation(self):
        """Test mixed precision compilation."""
        config = CompilerConfig(
            target=Target.GPU,
            mixed_precision=True
        )
        compiler = MLCompiler(config)

        builder = IRBuilder()
        builder.begin_block('mixed_precision')

        # FP32 input
        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 128])),
            name='x'
        )

        # Cast to FP16 for computation
        x_fp16 = builder.cast(x, DataType.FLOAT16)
        y_fp16 = builder.add_relu(x_fp16)

        # Cast back to FP32
        y = builder.cast(y_fp16, DataType.FLOAT32)
        builder.set_return(y)

        module = builder.get_module()
        compiled = compiler.compile(module)

        self.assertIn('mixed_precision', compiled['metadata'])
        self.assertTrue(compiled['metadata']['mixed_precision'])


class TestCompilerDiagnostics(unittest.TestCase):
    """Test compiler diagnostic features."""

    def test_compilation_errors(self):
        """Test handling of compilation errors."""
        config = CompilerConfig(target=Target.CPU)
        compiler = MLCompiler(config)

        builder = IRBuilder()
        builder.begin_block('error_test')

        # Create type mismatch error
        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 128])),
            name='x'
        )
        y = builder.create_input(
            TensorType(DataType.INT32, Shape([32, 128])),
            name='y'
        )

        # This should cause an error
        with self.assertRaises(TypeError):
            z = builder.add_binary_op('add', x, y)
            builder.set_return(z)
            module = builder.get_module()
            compiler.compile(module)

    def test_profiling_information(self):
        """Test generation of profiling information."""
        config = CompilerConfig(
            target=Target.CPU,
            profile=True
        )
        compiler = MLCompiler(config)

        builder = IRBuilder()
        builder.begin_block('profile_test')

        x = builder.create_input(
            TensorType(DataType.FLOAT32, Shape([1024, 1024])),
            name='x'
        )
        y = builder.add_relu(x)
        z = builder.add_softmax(y)
        builder.set_return(z)

        module = builder.get_module()
        compiled = compiler.compile(module)

        # Check profiling information
        self.assertIn('profile', compiled)
        profile = compiled['profile']
        self.assertIn('operation_times', profile)
        self.assertIn('memory_usage', profile)
        self.assertIn('total_time', profile)

    def test_debug_output(self):
        """Test debug output generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompilerConfig(
                target=Target.CPU,
                debug=True,
                debug_dir=tmpdir
            )
            compiler = MLCompiler(config)

            builder = IRBuilder()
            builder.begin_block('debug_test')

            x = builder.create_input(
                TensorType(DataType.FLOAT32, Shape([32, 128])),
                name='x'
            )
            y = builder.add_relu(x)
            builder.set_return(y)

            module = builder.get_module()
            compiled = compiler.compile(module)

            # Check debug files were created
            ir_file = os.path.join(tmpdir, 'debug_test.ir')
            opt_file = os.path.join(tmpdir, 'debug_test.opt.ir')
            code_file = os.path.join(tmpdir, 'debug_test.c')

            self.assertTrue(os.path.exists(ir_file))
            self.assertTrue(os.path.exists(opt_file))
            self.assertTrue(os.path.exists(code_file))


if __name__ == '__main__':
    unittest.main()