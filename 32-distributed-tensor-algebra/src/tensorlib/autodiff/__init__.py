"""Automatic differentiation for tensor operations."""

from .tape import (
    GradientTape,
    AutoDiffContext,
    grad,
    value_and_grad,
    vjp,
    jacobian,
    hessian,
)

__all__ = [
    "GradientTape",
    "AutoDiffContext",
    "grad",
    "value_and_grad",
    "vjp",
    "jacobian",
    "hessian",
]
