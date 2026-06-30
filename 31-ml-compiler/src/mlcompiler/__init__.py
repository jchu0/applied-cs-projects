"""ML Compiler - XLA/TVM-lite compiler for ML models."""

from .ir import (
    DType,
    TensorType,
    Value,
    Constant,
    OpCode,
    Operation,
    Block,
    Function,
    IRModule,
    IRBuilder,
)
from .optimization import (
    Pass,
    PassManager,
    ConstantFolding,
    DeadCodeElimination,
    CommonSubexpressionElimination,
    OperatorFusion,
    create_default_pipeline,
)
from .memory import (
    MemoryPlanner,
    MemoryPlan,
    AllocationStrategy,
    analyze_memory_usage,
)
from .codegen import (
    GeneratedCode,
    CPUCodeGenerator,
    CUDACodeGenerator,
    TritonCodeGenerator,
)
from .compiler import (
    MLCompiler,
    CompilerConfig,
    CompilationResult,
    Target,
    OptLevel,
    create_compiler,
)

__version__ = "0.1.0"

__all__ = [
    # IR
    "DType",
    "TensorType",
    "Value",
    "Constant",
    "OpCode",
    "Operation",
    "Block",
    "Function",
    "IRModule",
    "IRBuilder",
    # Optimization
    "Pass",
    "PassManager",
    "ConstantFolding",
    "DeadCodeElimination",
    "CommonSubexpressionElimination",
    "OperatorFusion",
    "create_default_pipeline",
    # Memory
    "MemoryPlanner",
    "MemoryPlan",
    "AllocationStrategy",
    "analyze_memory_usage",
    # Codegen
    "GeneratedCode",
    "CPUCodeGenerator",
    "CUDACodeGenerator",
    "TritonCodeGenerator",
    # Compiler
    "MLCompiler",
    "CompilerConfig",
    "CompilationResult",
    "Target",
    "OptLevel",
    "create_compiler",
]
