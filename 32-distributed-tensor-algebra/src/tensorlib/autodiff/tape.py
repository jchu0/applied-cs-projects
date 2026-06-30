"""Gradient tape for automatic differentiation."""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..core.tensor import LazyTensor, array, ones


class GradientTape:
    """Records operations for reverse-mode automatic differentiation."""

    def __init__(self):
        self.trace: List[Tuple[Any, List[LazyTensor], List[LazyTensor]]] = []
        self._watching: Set[int] = set()

    def watch(self, tensor: LazyTensor):
        """Mark a tensor to be watched for gradients."""
        self._watching.add(id(tensor))

    def record(self, prim: Any, inputs: List[LazyTensor], outputs: List[LazyTensor]):
        """Record an operation on the tape."""
        self.trace.append((prim, inputs, outputs))

    def gradient(self, target: LazyTensor, sources: List[LazyTensor]) -> List[LazyTensor]:
        """Compute gradients of target with respect to sources."""
        # Initialize output gradient
        grad_map: Dict[int, LazyTensor] = {
            id(target): ones(target.shape, target.dtype)
        }

        # Reverse topological traversal
        for prim, inputs, outputs in reversed(self.trace):
            out_grads = [grad_map.get(id(out)) for out in outputs]

            if all(g is None for g in out_grads):
                continue

            # Compute input gradients via VJP
            if len(outputs) == 1 and out_grads[0] is not None:
                try:
                    input_grads = prim.vjp(inputs, None, out_grads[0], **prim.params)

                    # Accumulate gradients
                    for inp, grad in zip(inputs, input_grads):
                        if grad is not None:
                            if id(inp) in grad_map:
                                grad_map[id(inp)] = grad_map[id(inp)] + grad
                            else:
                                grad_map[id(inp)] = grad
                except NotImplementedError:
                    pass

        # Extract gradients for sources
        return [grad_map.get(id(src)) for src in sources]


class AutoDiffContext:
    """Context manager for automatic differentiation."""

    _tape_stack: List[GradientTape] = []

    @classmethod
    def get_current_tape(cls) -> Optional[GradientTape]:
        return cls._tape_stack[-1] if cls._tape_stack else None

    def __init__(self):
        self.tape = GradientTape()

    def __enter__(self):
        AutoDiffContext._tape_stack.append(self.tape)
        return self.tape

    def __exit__(self, *args):
        AutoDiffContext._tape_stack.pop()


def grad(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """Create a function that computes gradients.

    Args:
        fun: Function to differentiate
        argnums: Which arguments to differentiate with respect to

    Returns:
        Function that computes gradients
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def grad_fun(*args, **kwargs):
        with AutoDiffContext() as tape:
            for i in argnums:
                tape.watch(args[i])

            result = fun(*args, **kwargs)
            sources = [args[i] for i in argnums]
            grads = tape.gradient(result, sources)

            if len(grads) == 1:
                return grads[0]
            return grads

    return grad_fun


def value_and_grad(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """Return both function value and gradients.

    Args:
        fun: Function to differentiate
        argnums: Which arguments to differentiate with respect to

    Returns:
        Function that returns (value, gradients)
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def value_and_grad_fun(*args, **kwargs):
        with AutoDiffContext() as tape:
            for i in argnums:
                tape.watch(args[i])

            result = fun(*args, **kwargs)
            sources = [args[i] for i in argnums]
            grads = tape.gradient(result, sources)

            if len(grads) == 1:
                return result, grads[0]
            return result, grads

    return value_and_grad_fun


def vjp(fun: Callable, *primals):
    """Compute vector-Jacobian product.

    Args:
        fun: Function to compute VJP for
        *primals: Input values

    Returns:
        (output, vjp_fun) where vjp_fun takes output tangent
    """
    with AutoDiffContext() as tape:
        for p in primals:
            tape.watch(p)

        output = fun(*primals)

    def vjp_fun(g):
        # Create new tape with same trace but g as initial gradient
        grad_map: Dict[int, LazyTensor] = {id(output): g}

        for prim, inputs, outputs in reversed(tape.trace):
            out_grads = [grad_map.get(id(out)) for out in outputs]

            if all(og is None for og in out_grads):
                continue

            if len(outputs) == 1 and out_grads[0] is not None:
                try:
                    input_grads = prim.vjp(inputs, None, out_grads[0], **prim.params)

                    for inp, grad in zip(inputs, input_grads):
                        if grad is not None:
                            if id(inp) in grad_map:
                                grad_map[id(inp)] = grad_map[id(inp)] + grad
                            else:
                                grad_map[id(inp)] = grad
                except NotImplementedError:
                    pass

        return tuple(grad_map.get(id(p)) for p in primals)

    return output, vjp_fun


def jacobian(fun: Callable, argnums: int = 0) -> Callable:
    """Compute full Jacobian matrix.

    Args:
        fun: Function to compute Jacobian for
        argnums: Which argument to differentiate

    Returns:
        Function that computes Jacobian
    """
    def jacobian_fun(*args, **kwargs):
        x = args[argnums]
        x_flat = x.flatten()
        n = x_flat.size

        # Compute Jacobian column by column
        cols = []
        for i in range(n):
            # Create one-hot vector
            v = array([1.0 if j == i else 0.0 for j in range(n)]).reshape(x.shape)

            # Compute directional derivative (would need JVP)
            # For now, use finite differences as fallback
            eps = 1e-5
            x_plus = array(x.numpy() + eps * v.numpy())

            args_plus = list(args)
            args_plus[argnums] = x_plus

            f_plus = fun(*args_plus, **kwargs)
            f_val = fun(*args, **kwargs)

            col = (f_plus - f_val) / array(eps)
            cols.append(col.flatten())

        # Stack columns
        from ..core.primitives import reshape
        import numpy as np
        jac = np.stack([c.numpy() for c in cols], axis=-1)
        return array(jac)

    return jacobian_fun


def hessian(fun: Callable, argnums: int = 0) -> Callable:
    """Compute Hessian matrix (second derivatives).

    Args:
        fun: Function to compute Hessian for
        argnums: Which argument to differentiate

    Returns:
        Function that computes Hessian
    """
    return jacobian(grad(fun, argnums), argnums)
