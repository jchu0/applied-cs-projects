"""Main ML compiler implementation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import logging
import time
import numpy as np

from .ir import (
    IRModule, Function, IRBuilder, TensorType, DType, Value, OpCode
)
from .optimization import PassManager, create_default_pipeline
from .memory import MemoryPlanner, AllocationStrategy, MemoryPlan, analyze_memory_usage
from .codegen import (
    GeneratedCode, CPUCodeGenerator, CUDACodeGenerator, TritonCodeGenerator
)

logger = logging.getLogger(__name__)


class Target(Enum):
    """Compilation targets."""
    CPU = "cpu"
    CUDA = "cuda"
    TRITON = "triton"


class OptLevel(Enum):
    """Optimization levels."""
    O0 = 0  # No optimization
    O1 = 1  # Basic optimizations
    O2 = 2  # Full optimizations
    O3 = 3  # Aggressive optimizations


@dataclass
class CompilerConfig:
    """Compiler configuration."""
    target: Target = Target.CPU
    opt_level: OptLevel = OptLevel.O2
    memory_strategy: AllocationStrategy = AllocationStrategy.GREEDY
    enable_profiling: bool = False
    debug_ir: bool = False
    block_size: int = 256  # For GPU targets


@dataclass
class CompilationResult:
    """Result of compilation."""
    code: GeneratedCode
    memory_plan: MemoryPlan
    stats: dict[str, Any]


@dataclass
class CompilerStats:
    """Compiler statistics."""
    parse_time_ms: float = 0.0
    optimize_time_ms: float = 0.0
    memory_plan_time_ms: float = 0.0
    codegen_time_ms: float = 0.0
    total_time_ms: float = 0.0
    num_ops_before: int = 0
    num_ops_after: int = 0
    memory_reduction_pct: float = 0.0


class MLCompiler:
    """Main ML compiler class."""

    def __init__(self, config: CompilerConfig = None):
        """Initialize compiler.

        Args:
            config: Compiler configuration
        """
        self.config = config or CompilerConfig()
        self._pass_manager = None
        self._memory_planner = None

        # Initialize based on config
        self._setup_passes()
        self._setup_memory_planner()

    def _setup_passes(self):
        """Setup optimization passes based on opt level."""
        if self.config.opt_level == OptLevel.O0:
            self._pass_manager = PassManager()
        else:
            self._pass_manager = create_default_pipeline()

            if self.config.opt_level == OptLevel.O3:
                # Add more aggressive passes
                self._pass_manager.max_iterations = 20

    def _setup_memory_planner(self):
        """Setup memory planner."""
        self._memory_planner = MemoryPlanner(self.config.memory_strategy)

    def compile(self, module: IRModule) -> CompilationResult:
        """Compile IR module.

        Args:
            module: IR module to compile

        Returns:
            Compilation result
        """
        stats = CompilerStats()
        start_total = time.time()

        # Count ops before
        stats.num_ops_before = self._count_ops(module)

        if self.config.debug_ir:
            logger.info(f"Input IR:\n{module}")

        # Optimize
        start = time.time()
        optimized = self._pass_manager.run(module)
        stats.optimize_time_ms = (time.time() - start) * 1000

        stats.num_ops_after = self._count_ops(optimized)

        if self.config.debug_ir:
            logger.info(f"Optimized IR:\n{optimized}")

        # Memory planning
        start = time.time()
        memory_plans = {}
        for func_name, func in optimized.functions.items():
            memory_plans[func_name] = self._memory_planner.plan(func)
        stats.memory_plan_time_ms = (time.time() - start) * 1000

        # Code generation
        start = time.time()
        # Use first function's memory plan
        first_plan = list(memory_plans.values())[0] if memory_plans else None
        code = self._generate_code(optimized, first_plan)
        stats.codegen_time_ms = (time.time() - start) * 1000

        stats.total_time_ms = (time.time() - start_total) * 1000

        # Calculate memory reduction
        if first_plan:
            mem_stats = analyze_memory_usage(first_plan)
            stats.memory_reduction_pct = mem_stats.buffer_reuse_rate * 100

        logger.info(
            f"Compiled {stats.num_ops_before} -> {stats.num_ops_after} ops "
            f"in {stats.total_time_ms:.1f}ms"
        )

        return CompilationResult(
            code=code,
            memory_plan=first_plan,
            stats={
                "parse_time_ms": stats.parse_time_ms,
                "optimize_time_ms": stats.optimize_time_ms,
                "memory_plan_time_ms": stats.memory_plan_time_ms,
                "codegen_time_ms": stats.codegen_time_ms,
                "total_time_ms": stats.total_time_ms,
                "num_ops_before": stats.num_ops_before,
                "num_ops_after": stats.num_ops_after,
                "memory_reduction_pct": stats.memory_reduction_pct,
            }
        )

    def _count_ops(self, module: IRModule) -> int:
        """Count operations in module."""
        count = 0
        for func in module.functions.values():
            for block in func.body.blocks:
                count += len(block.operations)
        return count

    def _generate_code(self, module: IRModule, memory_plan: MemoryPlan) -> GeneratedCode:
        """Generate code for target.

        Args:
            module: Optimized IR module
            memory_plan: Memory allocation plan

        Returns:
            Generated code
        """
        if self.config.target == Target.CPU:
            generator = CPUCodeGenerator(memory_plan)
        elif self.config.target == Target.CUDA:
            generator = CUDACodeGenerator(memory_plan, self.config.block_size)
        elif self.config.target == Target.TRITON:
            generator = TritonCodeGenerator(memory_plan)
        else:
            generator = CPUCodeGenerator(memory_plan)

        return generator.generate(module)

    def compile_function(
        self,
        name: str,
        input_types: list[TensorType],
        output_types: list[TensorType],
        builder_fn
    ) -> CompilationResult:
        """Compile a function using builder pattern.

        Args:
            name: Function name
            input_types: Input tensor types
            output_types: Output tensor types
            builder_fn: Function that builds IR using IRBuilder

        Returns:
            Compilation result
        """
        # Create module and function
        module = IRModule()
        func = module.create_function(name, input_types, output_types)

        # Create builder and let user build the function
        builder = IRBuilder(func.entry_block)
        builder_fn(builder, func.arguments)

        return self.compile(module)


def create_compiler(
    target: str = "cpu",
    opt_level: int = 2,
    **kwargs
) -> MLCompiler:
    """Create a compiler with specified configuration.

    Args:
        target: Target platform ("cpu", "cuda", "triton")
        opt_level: Optimization level (0-3)
        **kwargs: Additional config options

    Returns:
        Configured compiler
    """
    target_map = {
        "cpu": Target.CPU,
        "cuda": Target.CUDA,
        "triton": Target.TRITON,
    }

    opt_map = {
        0: OptLevel.O0,
        1: OptLevel.O1,
        2: OptLevel.O2,
        3: OptLevel.O3,
    }

    config = CompilerConfig(
        target=target_map.get(target, Target.CPU),
        opt_level=opt_map.get(opt_level, OptLevel.O2),
        **kwargs
    )

    return MLCompiler(config)


# Convenience functions for building common patterns

def build_mlp(
    builder: IRBuilder,
    inputs: list[Value],
    hidden_sizes: list[int],
    activation: str = "relu"
) -> Value:
    """Build MLP using builder.

    Args:
        builder: IR builder
        inputs: Input values
        hidden_sizes: Hidden layer sizes
        activation: Activation function

    Returns:
        Output value
    """
    x = inputs[0]

    for i, hidden_size in enumerate(hidden_sizes):
        # Would need weight values - simplified here
        # In practice, weights would be additional inputs
        pass

    return x


def build_attention(
    builder: IRBuilder,
    query: Value,
    key: Value,
    value: Value,
    scale: float = None
) -> Value:
    """Build attention using builder.

    Args:
        builder: IR builder
        query: Query tensor
        key: Key tensor
        value: Value tensor
        scale: Attention scale

    Returns:
        Attention output
    """
    return builder.attention(query, key, value, scale=scale)


def build_transformer_block(
    builder: IRBuilder,
    x: Value,
    num_heads: int,
    hidden_size: int,
    mlp_ratio: float = 4.0
) -> Value:
    """Build transformer block.

    Args:
        builder: IR builder
        x: Input tensor
        num_heads: Number of attention heads
        hidden_size: Hidden dimension
        mlp_ratio: MLP expansion ratio

    Returns:
        Output tensor
    """
    # Simplified - would need full implementation with weights
    return x


# Example usage
def example_compilation():
    """Example of using the compiler."""
    # Create compiler
    compiler = create_compiler(target="cpu", opt_level=2)

    # Define function types
    input_types = [
        TensorType((128, 512), DType.FLOAT32),
        TensorType((512, 256), DType.FLOAT32),
    ]
    output_types = [TensorType((128, 256), DType.FLOAT32)]

    # Build function
    def build_fn(builder: IRBuilder, args: list[Value]):
        a, b = args
        c = builder.matmul(a, b)
        c = builder.relu(c)
        builder.return_op([c])

    # Compile
    result = compiler.compile_function(
        "matmul_relu",
        input_types,
        output_types,
        build_fn
    )

    print(f"Generated {len(result.code.source)} bytes of code")
    print(f"Peak memory: {result.memory_plan.peak_memory} bytes")

    return result
