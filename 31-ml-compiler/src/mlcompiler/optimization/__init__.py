"""Optimization passes for ML compiler."""

from .passes import (
    Pass,
    FunctionPass,
    ConstantFolding,
    DeadCodeElimination,
    CommonSubexpressionElimination,
    OperatorFusion,
    LayoutOptimization,
    StrengthReduction,
    AlgebraicSimplification,
    PassManager,
    PassError,
    create_default_pipeline,
)

__all__ = [
    "Pass",
    "FunctionPass",
    "ConstantFolding",
    "DeadCodeElimination",
    "CommonSubexpressionElimination",
    "OperatorFusion",
    "LayoutOptimization",
    "StrengthReduction",
    "AlgebraicSimplification",
    "PassManager",
    "PassError",
    "create_default_pipeline",
]
