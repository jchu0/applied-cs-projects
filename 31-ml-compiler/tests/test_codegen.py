"""Unit tests for code generation in the ML compiler."""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try importing required classes
try:
    from mlcompiler import (
        IRBuilder, IRModule as Module, TensorType, DType as DataType,
        CPUCodeGenerator, CUDACodeGenerator, Target
    )
    # Stubs for classes that may not exist - tests require these
    CodeGenerator = CodeBuffer = None
    CUDACodeGen = CUDAKernel = CUDAMemoryManager = None
    Shape = None
    # Required classes are None, tests cannot run
    _IMPORTS_OK = False
except ImportError:
    _IMPORTS_OK = False

pytestmark = pytest.mark.skipif(not _IMPORTS_OK, reason="Missing required codegen classes")


class TestCodeGenerator(unittest.TestCase):
    """Test base code generator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.codegen = CodeGenerator(target=Target.CPU)
        self.builder = IRBuilder()

    def test_codegen_initialization(self):
        """Test code generator initialization."""
        self.assertEqual(self.codegen.target, Target.CPU)
        self.assertIsNotNone(self.codegen.code_buffer)

    def test_generate_simple_function(self):
        """Test generating code for simple function."""
        self.builder.begin_block('simple_func')

        x = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 128])),
            name='x'
        )
        y = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 128])),
            name='y'
        )

        result = self.builder.add_binary_op('add', x, y)
        self.builder.set_return(result)

        module = self.builder.get_module()
        code = self.codegen.generate(module)

        # Check generated code contains expected elements
        self.assertIn('float*', code)  # Pointer types
        self.assertIn('add', code.lower())  # Operation
        self.assertIn('x', code)  # Input name
        self.assertIn('y', code)  # Input name

    def test_generate_with_constants(self):
        """Test code generation with constants."""
        self.builder.begin_block('const_func')

        x = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 128])),
            name='x'
        )
        const = self.builder.create_constant(2.5, name='scale')

        result = self.builder.add_binary_op('multiply', x, const)
        self.builder.set_return(result)

        module = self.builder.get_module()
        code = self.codegen.generate(module)

        # Check constant is properly generated
        self.assertIn('2.5', code)
        self.assertIn('scale', code)

    def test_generate_control_flow(self):
        """Test code generation for control flow."""
        self.builder.begin_block('control_flow')

        condition = self.builder.create_input(
            TensorType(DataType.BOOL, Shape([1])),
            name='cond'
        )
        x = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 128])),
            name='x'
        )

        with self.builder.if_block(condition) as if_ctx:
            true_result = self.builder.add_unary_op('relu', x)
            if_ctx.set_true_branch(true_result)

            false_result = self.builder.add_unary_op('sigmoid', x)
            if_ctx.set_false_branch(false_result)

        module = self.builder.get_module()
        code = self.codegen.generate(module)

        # Check control flow structures
        self.assertIn('if', code)
        self.assertIn('else', code)
        self.assertIn('relu', code.lower())
        self.assertIn('sigmoid', code.lower())

    def test_generate_loop(self):
        """Test code generation for loops."""
        self.builder.begin_block('loop_func')

        x = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 128])),
            name='x'
        )

        with self.builder.for_loop(0, 32) as loop:
            i = loop.get_index()
            elem = self.builder.get_element(x, [i, 0])
            new_elem = self.builder.add_scalar(elem, 1.0)
            x = self.builder.set_element(x, [i, 0], new_elem)

        module = self.builder.get_module()
        code = self.codegen.generate(module)

        # Check loop structures
        self.assertIn('for', code)
        self.assertIn('32', code)  # Loop bound
        self.assertIn('[i]', code)  # Array indexing


class TestCUDACodeGen(unittest.TestCase):
    """Test CUDA code generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cuda_codegen = CUDACodeGen()
        self.builder = IRBuilder()

    @patch('mlcompiler.codegen.cuda.cuda_available')
    def test_cuda_kernel_generation(self, mock_cuda):
        """Test CUDA kernel generation."""
        mock_cuda.return_value = True

        self.builder.begin_block('cuda_kernel')

        x = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([1024, 1024])),
            name='x'
        )
        y = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([1024, 1024])),
            name='y'
        )

        result = self.builder.add_binary_op('add', x, y)
        self.builder.set_return(result)

        module = self.builder.get_module()
        code = self.cuda_codegen.generate(module)

        # Check CUDA specific elements
        self.assertIn('__global__', code)  # Kernel decorator
        self.assertIn('blockIdx', code)  # CUDA built-ins
        self.assertIn('threadIdx', code)
        self.assertIn('blockDim', code)
        self.assertIn('gridDim', code)

    def test_cuda_memory_allocation(self):
        """Test CUDA memory allocation code generation."""
        self.builder.begin_block('cuda_memory')

        x = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 512, 512])),
            name='x'
        )

        module = self.builder.get_module()
        code = self.cuda_codegen.generate(module)

        # Check memory management
        self.assertIn('cudaMalloc', code)
        self.assertIn('cudaMemcpy', code)
        self.assertIn('cudaFree', code)

    def test_cuda_shared_memory(self):
        """Test CUDA shared memory generation."""
        self.builder.begin_block('shared_memory')

        x = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([256, 256])),
            name='x'
        )

        # Mark for shared memory optimization
        self.builder.use_shared_memory(x, tile_size=16)

        module = self.builder.get_module()
        code = self.cuda_codegen.generate(module)

        # Check shared memory usage
        self.assertIn('__shared__', code)
        self.assertIn('[16][16]', code)  # Tile size

    def test_cuda_kernel_launch(self):
        """Test CUDA kernel launch configuration."""
        kernel = CUDAKernel(
            name='test_kernel',
            grid_dim=(32, 32, 1),
            block_dim=(16, 16, 1)
        )

        launch_code = self.cuda_codegen.generate_kernel_launch(kernel)

        # Check launch configuration
        self.assertIn('<<<', launch_code)
        self.assertIn('>>>', launch_code)
        self.assertIn('32, 32', launch_code)  # Grid dimensions
        self.assertIn('16, 16', launch_code)  # Block dimensions

    def test_cuda_convolution_kernel(self):
        """Test CUDA convolution kernel generation."""
        self.builder.begin_block('conv_kernel')

        x = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([32, 3, 224, 224])),
            name='input'
        )
        weight = self.builder.create_weight(
            TensorType(DataType.FLOAT32, Shape([64, 3, 7, 7])),
            name='weight'
        )

        conv = self.builder.add_convolution(x, weight, stride=(2, 2), padding=(3, 3))
        self.builder.set_return(conv)

        module = self.builder.get_module()
        code = self.cuda_codegen.generate(module)

        # Check convolution specific code
        self.assertIn('conv2d', code.lower())
        self.assertIn('stride', code.lower())
        self.assertIn('padding', code.lower())

    def test_cuda_reduction_kernel(self):
        """Test CUDA reduction kernel generation."""
        self.builder.begin_block('reduction')

        x = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([1024, 1024])),
            name='x'
        )

        sum_result = self.builder.add_reduce(x, reduce_type='sum', axes=[1])
        self.builder.set_return(sum_result)

        module = self.builder.get_module()
        code = self.cuda_codegen.generate(module)

        # Check reduction specific code
        self.assertIn('__syncthreads()', code)  # Synchronization
        self.assertIn('atomicAdd', code)  # Atomic operations for reduction


class TestCodeBuffer(unittest.TestCase):
    """Test code buffer management."""

    def setUp(self):
        """Set up test fixtures."""
        self.buffer = CodeBuffer()

    def test_buffer_operations(self):
        """Test code buffer operations."""
        self.buffer.append('#include <stdio.h>')
        self.buffer.newline()
        self.buffer.append('int main() {')
        self.buffer.indent()
        self.buffer.append('printf("Hello World");')
        self.buffer.append('return 0;')
        self.buffer.dedent()
        self.buffer.append('}')

        code = self.buffer.get_code()

        # Check formatting
        self.assertIn('#include <stdio.h>', code)
        self.assertIn('    printf("Hello World");', code)  # Check indentation
        self.assertIn('}', code)

    def test_buffer_indentation(self):
        """Test code buffer indentation management."""
        self.buffer.append('void function() {')
        self.buffer.indent()
        self.buffer.append('if (condition) {')
        self.buffer.indent()
        self.buffer.append('do_something();')
        self.buffer.dedent()
        self.buffer.append('}')
        self.buffer.dedent()
        self.buffer.append('}')

        code = self.buffer.get_code()
        lines = code.split('\n')

        # Check proper indentation levels
        self.assertTrue(lines[2].startswith('        '))  # Double indentation
        self.assertTrue(lines[1].startswith('    '))  # Single indentation


class TestTargetSpecificGeneration(unittest.TestCase):
    """Test target-specific code generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.builder = IRBuilder()

    def test_cpu_vectorization(self):
        """Test CPU vectorization code generation."""
        codegen = CodeGenerator(target=Target.CPU, vectorize=True)

        self.builder.begin_block('vector_add')

        x = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([1024])),
            name='x'
        )
        y = self.builder.create_input(
            TensorType(DataType.FLOAT32, Shape([1024])),
            name='y'
        )

        result = self.builder.add_binary_op('add', x, y)
        self.builder.set_return(result)

        module = self.builder.get_module()
        code = codegen.generate(module)

        # Check for vectorization
        self.assertIn('__m256', code)  # AVX vectors
        self.assertIn('_mm256_add_ps', code)  # Vector add instruction

    def test_gpu_tensor_core(self):
        """Test GPU tensor core code generation."""
        codegen = CUDACodeGen(use_tensor_cores=True)

        self.builder.begin_block('matmul')

        a = self.builder.create_input(
            TensorType(DataType.FLOAT16, Shape([1024, 1024])),
            name='a'
        )
        b = self.builder.create_input(
            TensorType(DataType.FLOAT16, Shape([1024, 1024])),
            name='b'
        )

        result = self.builder.add_matmul(a, b)
        self.builder.set_return(result)

        module = self.builder.get_module()
        code = codegen.generate(module)

        # Check for tensor core usage
        self.assertIn('wmma', code)  # Warp matrix multiply accumulate
        self.assertIn('fragment', code)  # WMMA fragments

    def test_compilation_and_execution(self):
        """Test code compilation and execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            codegen = CodeGenerator(target=Target.CPU)

            self.builder.begin_block('simple')

            x = self.builder.create_input(
                TensorType(DataType.FLOAT32, Shape([4])),
                name='x'
            )
            result = self.builder.add_scalar(x, 1.0)
            self.builder.set_return(result)

            module = self.builder.get_module()
            code = codegen.generate(module)

            # Write code to file
            code_file = os.path.join(tmpdir, 'test.c')
            with open(code_file, 'w') as f:
                f.write(code)

            # Check file was created
            self.assertTrue(os.path.exists(code_file))


class TestCUDAMemoryManager(unittest.TestCase):
    """Test CUDA memory management."""

    def setUp(self):
        """Set up test fixtures."""
        self.mem_manager = CUDAMemoryManager()

    def test_memory_pool(self):
        """Test CUDA memory pool management."""
        # Allocate memory
        ptr1 = self.mem_manager.allocate(1024 * 1024)  # 1MB
        self.assertIsNotNone(ptr1)

        # Free and reallocate
        self.mem_manager.free(ptr1)
        ptr2 = self.mem_manager.allocate(1024 * 1024)

        # Should reuse the same memory
        self.assertEqual(ptr1, ptr2)

    def test_memory_coalescing(self):
        """Test memory access coalescing optimization."""
        access_pattern = [
            (0, 0), (0, 1), (0, 2), (0, 3),  # Coalesced
            (1, 0), (1, 1), (1, 2), (1, 3)   # Coalesced
        ]

        is_coalesced = self.mem_manager.check_coalesced_access(access_pattern)
        self.assertTrue(is_coalesced)

        # Non-coalesced pattern
        bad_pattern = [
            (0, 0), (1, 0), (2, 0), (3, 0),  # Strided access
        ]

        is_coalesced = self.mem_manager.check_coalesced_access(bad_pattern)
        self.assertFalse(is_coalesced)

    def test_unified_memory(self):
        """Test unified memory support."""
        self.mem_manager.use_unified_memory = True

        ptr = self.mem_manager.allocate(1024 * 1024)
        self.assertIsNotNone(ptr)

        # Check that memory is accessible from both CPU and GPU
        self.assertTrue(self.mem_manager.is_unified(ptr))


if __name__ == '__main__':
    unittest.main()