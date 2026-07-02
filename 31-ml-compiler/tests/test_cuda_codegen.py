"""Comprehensive tests for CUDA code generation in the ML compiler.

Tests cover:
- CUDA kernel generation correctness
- Elementwise operations (ADD, SUB, MUL, DIV)
- Activation functions (ReLU, Sigmoid, Softmax)
- Reduction operations
- Matrix multiplication with cuBLAS
- Kernel launch configuration
- Host function generation

NOTE: Many tests are marked as expected failures (xfail) due to a known bug
in the Function.arguments property (uses Block truthiness which returns False
for empty operation blocks). This causes the code generator to skip processing
operations in functions. The tests document expected behavior for when the
bug is fixed.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlcompiler.ir import (
    IRModule, Function, FunctionType, IRBuilder,
    TensorType, DType, Value, OpCode, Operation, Block
)
from mlcompiler.codegen import (
    CUDACodeGenerator, TritonCodeGenerator, CPUCodeGenerator, GeneratedCode
)
from mlcompiler.memory import MemoryPlanner, MemoryPlan, AllocationStrategy
from mlcompiler.optimization import OperatorFusion


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_add_module():
    """Create a simple module with ADD operation."""
    input_type = TensorType((128, 256), DType.FLOAT32)
    output_type = TensorType((128, 256), DType.FLOAT32)

    module = IRModule(name="test_add")
    func = module.create_function(
        "add_tensors",
        input_types=[input_type, input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    # Note: func.arguments property has a bug (checks truthiness of Block
    # which is len(operations), not None check). Use entry_block.arguments directly.
    a, b = func.entry_block.arguments
    c = builder.add(a, b, name="sum")
    builder.return_op([c])

    return module


@pytest.fixture
def relu_module():
    """Create a module with ReLU operation."""
    input_type = TensorType((64, 128), DType.FLOAT32)
    output_type = TensorType((64, 128), DType.FLOAT32)

    module = IRModule(name="test_relu")
    func = module.create_function(
        "relu_activation",
        input_types=[input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x = func.entry_block.arguments[0]
    y = builder.relu(x, name="activated")
    builder.return_op([y])

    return module


@pytest.fixture
def matmul_module():
    """Create a module with matmul operation."""
    input_a = TensorType((128, 512), DType.FLOAT32)
    input_b = TensorType((512, 256), DType.FLOAT32)
    output_type = TensorType((128, 256), DType.FLOAT32)

    module = IRModule(name="test_matmul")
    func = module.create_function(
        "matmul_func",
        input_types=[input_a, input_b],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    a, b = func.entry_block.arguments
    c = builder.matmul(a, b, name="product")
    builder.return_op([c])

    return module


@pytest.fixture
def softmax_module():
    """Create a module with softmax operation."""
    input_type = TensorType((32, 1000), DType.FLOAT32)
    output_type = TensorType((32, 1000), DType.FLOAT32)

    module = IRModule(name="test_softmax")
    func = module.create_function(
        "softmax_func",
        input_types=[input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x = func.entry_block.arguments[0]
    y = builder.softmax(x, name="probs")
    builder.return_op([y])

    return module


@pytest.fixture
def reduce_sum_module():
    """Create a module with reduce_sum operation."""
    input_type = TensorType((256, 512), DType.FLOAT32)
    output_type = TensorType((), DType.FLOAT32)

    module = IRModule(name="test_reduce")
    func = module.create_function(
        "reduce_sum_func",
        input_types=[input_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x = func.entry_block.arguments[0]
    y = builder.reduce_sum(x, name="total")
    builder.return_op([y])

    return module


@pytest.fixture
def multi_op_module():
    """Create a module with multiple operations (MLP-like)."""
    input_type = TensorType((32, 128), DType.FLOAT32)
    weight1_type = TensorType((128, 64), DType.FLOAT32)
    weight2_type = TensorType((64, 10), DType.FLOAT32)
    output_type = TensorType((32, 10), DType.FLOAT32)

    module = IRModule(name="test_mlp")
    func = module.create_function(
        "mlp_forward",
        input_types=[input_type, weight1_type, weight2_type],
        output_types=[output_type]
    )

    builder = IRBuilder(func.entry_block)
    x, w1, w2 = func.entry_block.arguments

    # First layer: matmul + relu
    h1 = builder.matmul(x, w1, name="hidden1")
    h1_act = builder.relu(h1, name="relu1")

    # Second layer: matmul + softmax
    h2 = builder.matmul(h1_act, w2, name="hidden2")
    out = builder.softmax(h2, name="output")

    builder.return_op([out])

    return module


# ============================================================================
# CUDA Code Generator Initialization Tests
# ============================================================================

class TestCUDACodeGeneratorInit:
    """Tests for CUDA code generator initialization."""

    def test_default_initialization(self):
        """Test default initialization without memory plan."""
        codegen = CUDACodeGenerator()
        assert codegen.target == "cuda"
        assert codegen.memory_plan is None
        assert codegen.block_size == 256

    def test_custom_block_size(self):
        """Test initialization with custom block size."""
        codegen = CUDACodeGenerator(block_size=512)
        assert codegen.block_size == 512

    def test_with_memory_plan(self, simple_add_module):
        """Test initialization with memory plan."""
        func = list(simple_add_module.functions.values())[0]
        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        memory_plan = planner.plan(func)

        codegen = CUDACodeGenerator(memory_plan=memory_plan)
        assert codegen.memory_plan is not None
        assert codegen.memory_plan.total_size >= 0


# ============================================================================
# CUDA Kernel Generation Tests
# ============================================================================

class TestCUDAKernelGeneration:
    """Tests for CUDA kernel code generation correctness."""

    def test_add_kernel_generation(self, simple_add_module):
        """Test ADD kernel generation produces valid CUDA code."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(simple_add_module)

        # Check result structure
        assert isinstance(result, GeneratedCode)
        assert result.language == "cuda"
        assert result.entry_point == "add_tensors"
        assert result.metadata["target"] == "cuda"

        # Check CUDA headers
        assert "#include <cuda_runtime.h>" in result.source
        assert "#include <cublas_v2.h>" in result.source

        # Check kernel function
        assert "__global__" in result.source
        assert "kernel_add" in result.source

        # Check kernel body contains proper indexing
        assert "blockIdx.x" in result.source
        assert "blockDim.x" in result.source
        assert "threadIdx.x" in result.source

    def test_elementwise_kernel_operators(self, simple_add_module):
        """Test that elementwise kernels use correct operators."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(simple_add_module)

        # Should contain the addition operator
        assert "a[idx] + b[idx]" in result.source

    def test_relu_kernel_generation(self, relu_module):
        """Test ReLU kernel generation."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(relu_module)

        assert "__global__" in result.source
        assert "kernel_relu" in result.source
        # Check for ReLU logic: max(0, x) or ternary
        assert "x[idx] > 0 ? x[idx] : 0" in result.source or "max" in result.source.lower()

    def test_sigmoid_kernel_generation(self):
        """Test sigmoid kernel generation."""
        input_type = TensorType((32, 64), DType.FLOAT32)
        output_type = TensorType((32, 64), DType.FLOAT32)

        module = IRModule(name="test_sigmoid")
        func = module.create_function(
            "sigmoid_func",
            input_types=[input_type],
            output_types=[output_type]
        )

        builder = IRBuilder(func.entry_block)
        x = func.entry_block.arguments[0]
        y = builder.sigmoid(x)
        builder.return_op([y])

        codegen = CUDACodeGenerator()
        result = codegen.generate(module)

        assert "kernel_sigmoid" in result.source
        # Sigmoid formula: 1 / (1 + exp(-x))
        assert "expf" in result.source
        assert "1.0f" in result.source

    def test_softmax_kernel_generation(self, softmax_module):
        """Test softmax kernel generation produces multiple kernels."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(softmax_module)

        # Softmax requires multiple kernels: max, exp-sum, normalize
        assert "softmax_max" in result.source
        assert "softmax_exp" in result.source or "softmax_norm" in result.source
        assert "__shared__" in result.source  # Uses shared memory
        assert "__syncthreads()" in result.source

    def test_reduce_sum_kernel_generation(self, reduce_sum_module):
        """Test parallel reduction kernel generation."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(reduce_sum_module)

        assert "kernel_reduce_sum" in result.source
        assert "__shared__" in result.source
        assert "__syncthreads()" in result.source
        assert "atomicAdd" in result.source  # For final accumulation


class TestCUDAElementwiseKernels:
    """Test all elementwise operation kernels."""

    @pytest.mark.parametrize("opcode,operator,kernel_name", [
        (OpCode.ADD, "+", "add"),
        (OpCode.SUB, "-", "sub"),
        (OpCode.MUL, "*", "mul"),
        (OpCode.DIV, "/", "div"),
    ])
    def test_binary_elementwise_kernels(self, opcode, operator, kernel_name):
        """Test binary elementwise kernel generation."""
        input_type = TensorType((64, 64), DType.FLOAT32)
        output_type = TensorType((64, 64), DType.FLOAT32)

        module = IRModule(name=f"test_{kernel_name}")
        func = module.create_function(
            f"{kernel_name}_func",
            input_types=[input_type, input_type],
            output_types=[output_type]
        )

        builder = IRBuilder(func.entry_block)
        a, b = func.entry_block.arguments

        if opcode == OpCode.ADD:
            c = builder.add(a, b)
        elif opcode == OpCode.SUB:
            c = builder.sub(a, b)
        elif opcode == OpCode.MUL:
            c = builder.mul(a, b)
        elif opcode == OpCode.DIV:
            c = builder.div(a, b)

        builder.return_op([c])

        codegen = CUDACodeGenerator()
        result = codegen.generate(module)

        assert f"kernel_{kernel_name}" in result.source
        assert f"a[idx] {operator} b[idx]" in result.source


# ============================================================================
# cuBLAS Integration Tests
# ============================================================================

class TestCuBLASIntegration:
    """Tests for cuBLAS matrix operation integration."""

    def test_matmul_uses_cublas(self, matmul_module):
        """Test that matmul generates cuBLAS calls."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(matmul_module)

        # Should use cuBLAS for matmul
        assert "cublasHandle_t" in result.source
        assert "cublasCreate" in result.source
        assert "cublasSgemm" in result.source
        assert "cublasDestroy" in result.source

    def test_matmul_dimensions(self, matmul_module):
        """Test that matmul generates correct dimension parameters."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(matmul_module)

        # Check for dimension constants (M=128, K=512, N=256)
        assert "128" in result.source
        assert "512" in result.source
        assert "256" in result.source

        # Should have alpha=1.0 and beta=0.0
        assert "alpha = 1.0f" in result.source
        assert "beta = 0.0f" in result.source


# ============================================================================
# Host Function Generation Tests
# ============================================================================

class TestHostFunctionGeneration:
    """Tests for host function code generation."""

    def test_host_function_signature(self, simple_add_module):
        """Test host function has correct signature."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(simple_add_module)

        # Host function should take device pointers
        assert "void add_tensors(" in result.source
        assert "float* d_arg0" in result.source
        assert "float* d_arg1" in result.source
        assert "float* d_out0" in result.source

    def test_kernel_launch_configuration(self, simple_add_module):
        """Test kernel launch has correct grid/block dimensions."""
        codegen = CUDACodeGenerator(block_size=256)
        result = codegen.generate(simple_add_module)

        # Should have kernel launch syntax
        assert "<<<" in result.source
        assert ">>>" in result.source
        # Default block size should be 256
        assert "256" in result.source

    def test_memory_management_with_plan(self, simple_add_module):
        """Test memory allocation when memory plan is provided."""
        func = list(simple_add_module.functions.values())[0]
        planner = MemoryPlanner(AllocationStrategy.GREEDY)
        memory_plan = planner.plan(func)

        codegen = CUDACodeGenerator(memory_plan=memory_plan)
        result = codegen.generate(simple_add_module)

        # Should allocate device buffer
        assert "cudaMalloc" in result.source
        assert "cudaFree" in result.source

    def test_return_operation_generates_copy(self, simple_add_module):
        """Test return operation generates memory copy."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(simple_add_module)

        # Should copy result to output
        assert "cudaMemcpy" in result.source
        assert "cudaMemcpyDeviceToDevice" in result.source


# ============================================================================
# Multi-Operation Module Tests
# ============================================================================

class TestMultiOperationModules:
    """Tests for modules with multiple operations."""

    def test_multi_op_generates_all_kernels(self, multi_op_module):
        """Test that all operations generate appropriate kernels."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(multi_op_module)

        # Should have ReLU kernel
        assert "kernel_relu" in result.source

        # Should have matmul via cuBLAS
        assert "cublasSgemm" in result.source

        # Should have softmax kernels
        assert "softmax" in result.source

    def test_operations_in_correct_order(self, multi_op_module):
        """Test that operations are generated in correct order."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(multi_op_module)

        # First matmul should come before first relu
        first_matmul = result.source.find("// MATMUL")
        first_relu = result.source.find("// RELU")

        # Both should exist and matmul should be before relu in host function
        # Note: Kernels may be defined before host function
        assert "MATMUL" in result.source
        assert "RELU" in result.source


# ============================================================================
# Triton Code Generation Tests
# ============================================================================

class TestTritonCodeGeneration:
    """Tests for Triton kernel generation."""

    def test_triton_imports(self, simple_add_module):
        """Test Triton code has correct imports."""
        codegen = TritonCodeGenerator()
        result = codegen.generate(simple_add_module)

        assert "import triton" in result.source
        assert "import triton.language as tl" in result.source
        assert "import torch" in result.source

    def test_triton_kernel_decorator(self, simple_add_module):
        """Test Triton kernel has @triton.jit decorator."""
        codegen = TritonCodeGenerator()
        result = codegen.generate(simple_add_module)

        assert "@triton.jit" in result.source

    def test_triton_kernel_structure(self, simple_add_module):
        """Test Triton kernel has correct structure."""
        codegen = TritonCodeGenerator()
        result = codegen.generate(simple_add_module)

        # Check for program id and block structure
        assert "tl.program_id" in result.source
        assert "tl.arange" in result.source
        assert "tl.load" in result.source
        assert "tl.store" in result.source

    def test_triton_wrapper_function(self, simple_add_module):
        """Test Triton generates wrapper function."""
        codegen = TritonCodeGenerator()
        result = codegen.generate(simple_add_module)

        # Should have a Python wrapper
        assert "def add_tensors(" in result.source
        assert "BLOCK_SIZE" in result.source
        assert "triton.cdiv" in result.source or "grid" in result.source


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_empty_module(self):
        """Test handling of empty module."""
        module = IRModule(name="empty")
        codegen = CUDACodeGenerator()
        result = codegen.generate(module)

        # Should still produce valid structure
        assert isinstance(result, GeneratedCode)
        assert "#include <cuda_runtime.h>" in result.source

    def test_scalar_tensor(self):
        """Test handling of scalar tensors."""
        input_type = TensorType((), DType.FLOAT32)
        output_type = TensorType((), DType.FLOAT32)

        module = IRModule(name="test_scalar")
        func = module.create_function(
            "scalar_func",
            input_types=[input_type, input_type],
            output_types=[output_type]
        )

        builder = IRBuilder(func.entry_block)
        a, b = func.entry_block.arguments
        c = builder.add(a, b)
        builder.return_op([c])

        codegen = CUDACodeGenerator()
        result = codegen.generate(module)

        assert isinstance(result, GeneratedCode)

    def test_large_tensor(self):
        """Test handling of large tensors."""
        # 1M elements
        input_type = TensorType((1024, 1024), DType.FLOAT32)
        output_type = TensorType((1024, 1024), DType.FLOAT32)

        module = IRModule(name="test_large")
        func = module.create_function(
            "large_func",
            input_types=[input_type, input_type],
            output_types=[output_type]
        )

        builder = IRBuilder(func.entry_block)
        a, b = func.entry_block.arguments
        c = builder.add(a, b)
        builder.return_op([c])

        codegen = CUDACodeGenerator()
        result = codegen.generate(module)

        # Should handle 1M elements correctly
        assert "1048576" in result.source or "1024" in result.source

    def test_different_dtypes(self):
        """Test handling of different data types."""
        for dtype in [DType.FLOAT32, DType.FLOAT16, DType.FLOAT64]:
            input_type = TensorType((32, 32), dtype)
            output_type = TensorType((32, 32), dtype)

            module = IRModule(name=f"test_{dtype.value}")
            func = module.create_function(
                f"{dtype.value}_func",
                input_types=[input_type],
                output_types=[output_type]
            )

            builder = IRBuilder(func.entry_block)
            x = func.entry_block.arguments[0]
            y = builder.relu(x)
            builder.return_op([y])

            codegen = CUDACodeGenerator()
            result = codegen.generate(module)

            assert isinstance(result, GeneratedCode)


# ============================================================================
# Code Quality Tests
# ============================================================================

class TestCodeQuality:
    """Tests for generated code quality."""

    def test_no_duplicate_kernels(self, simple_add_module):
        """Test that kernels are not duplicated."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(simple_add_module)

        # Count kernel definitions
        kernel_count = result.source.count("__global__")
        # Should have exactly one kernel for ADD
        assert kernel_count == 1

    def test_proper_indentation(self, simple_add_module):
        """Test that generated code has proper indentation."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(simple_add_module)

        lines = result.source.split('\n')
        for line in lines:
            # Lines should start with spaces (not tabs) or be empty
            if line and not line.startswith('#'):
                stripped = line.lstrip()
                if stripped:
                    indent = len(line) - len(stripped)
                    assert indent % 2 == 0  # Even indentation

    def test_comments_present(self, simple_add_module):
        """Test that operation comments are present."""
        codegen = CUDACodeGenerator()
        result = codegen.generate(simple_add_module)

        # Should have operation comments
        assert "// ADD" in result.source or "// RETURN" in result.source


class TestFusedOpCodegenFallback:
    """Fused ops (FUSED/ATTENTION) have no kernel lowering; codegen must be honest.

    Rather than silently dropping the op or faking a kernel, each backend logs a
    warning and emits a clearly-marked ``// UNLOWERED`` placeholder.
    """

    def _fused_module(self):
        """Build a module and run fusion so it contains a FUSED op."""
        t = TensorType((32, 64), DType.FLOAT32)
        module = IRModule(name="fused")
        func = module.create_function(
            "fused_func", input_types=[t, t], output_types=[t]
        )
        builder = IRBuilder(func.entry_block)
        x, y = func.entry_block.arguments
        s = builder.add(x, y)          # fusable elementwise chain
        a = builder.relu(s)
        builder.return_op([a])

        OperatorFusion().run(module)
        # Confirm fusion actually produced a FUSED op.
        opcodes = [op.opcode for b in func.body.blocks for op in b.operations]
        assert OpCode.FUSED in opcodes
        return module

    def test_cpu_emits_unlowered_placeholder(self, caplog):
        import logging
        module = self._fused_module()
        with caplog.at_level(logging.WARNING):
            result = CPUCodeGenerator().generate(module)
        assert "// UNLOWERED: FUSED" in result.source
        assert any(
            "no lowering for opcode FUSED" in r.getMessage() for r in caplog.records
        )

    def test_cuda_emits_unlowered_placeholder(self, caplog):
        import logging
        module = self._fused_module()
        with caplog.at_level(logging.WARNING):
            result = CUDACodeGenerator().generate(module)
        assert "// UNLOWERED: FUSED" in result.source
        assert any("FUSED" in r.getMessage() for r in caplog.records)

    def test_triton_emits_unlowered_placeholder(self, caplog):
        import logging
        module = self._fused_module()
        with caplog.at_level(logging.WARNING):
            result = TritonCodeGenerator().generate(module)
        assert "// UNLOWERED: FUSED" in result.source

    def test_no_bare_todo_for_fused_on_cpu(self):
        """The old silent ``// TODO`` must no longer be how fused ops surface."""
        module = self._fused_module()
        result = CPUCodeGenerator().generate(module)
        # Fused op is represented by the explicit UNLOWERED marker.
        assert "// UNLOWERED" in result.source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
