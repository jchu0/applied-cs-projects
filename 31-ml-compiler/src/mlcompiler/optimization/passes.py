"""Optimization passes for ML compiler."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import logging
import numpy as np

from ..ir import (
    IRModule, Function, Operation, OpCode, Value, Block,
    TensorType, DType, Constant
)

logger = logging.getLogger(__name__)


class Pass(ABC):
    """Base class for optimization passes."""

    name: str = "pass"

    @abstractmethod
    def run(self, module: IRModule) -> bool:
        """Run the pass on module.

        Args:
            module: IR module to optimize

        Returns:
            True if module was modified
        """
        pass


class FunctionPass(Pass):
    """Pass that operates on individual functions."""

    def run(self, module: IRModule) -> bool:
        modified = False
        for func in module.functions.values():
            if self.run_on_function(func):
                modified = True
        return modified

    @abstractmethod
    def run_on_function(self, func: Function) -> bool:
        """Run pass on a function."""
        pass


class ConstantFolding(FunctionPass):
    """Fold constant operations at compile time."""

    name = "constant-folding"

    def run_on_function(self, func: Function) -> bool:
        modified = False

        for block in func.body.blocks:
            ops_to_remove = []
            ops_to_add = []  # (index, new_op) pairs

            # Iterate over a copy to avoid mutation during iteration
            for op in list(block.operations):
                # Check if all inputs are constants
                if not self._all_inputs_constant(op):
                    continue

                # Try to evaluate
                result = self._evaluate(op)
                if result is not None:
                    # Replace with constant
                    const_op = Operation(
                        OpCode.CONSTANT, [], op.outputs,
                        {"value": result, "type": op.outputs[0].type}
                    )
                    # Update defining op
                    for output in op.outputs:
                        output.defining_op = const_op

                    ops_to_remove.append(op)
                    ops_to_add.append((block.operations.index(op), const_op))
                    modified = True

            # Apply additions (in reverse to preserve indices)
            for idx, new_op in reversed(ops_to_add):
                block.operations.insert(idx, new_op)

            for op in ops_to_remove:
                block.remove_operation(op)

        return modified

    def _all_inputs_constant(self, op: Operation) -> bool:
        """Check if all inputs are constants."""
        for inp in op.inputs:
            if inp.defining_op is None:
                return False
            if inp.defining_op.opcode != OpCode.CONSTANT:
                return False
        return True

    def _evaluate(self, op: Operation) -> Constant:
        """Evaluate operation on constants."""
        values = []
        for inp in op.inputs:
            const = inp.defining_op.get_attr("value")
            values.append(const.value)

        try:
            if op.opcode == OpCode.ADD:
                result = values[0] + values[1]
            elif op.opcode == OpCode.SUB:
                result = values[0] - values[1]
            elif op.opcode == OpCode.MUL:
                result = values[0] * values[1]
            elif op.opcode == OpCode.DIV:
                result = values[0] / values[1]
            elif op.opcode == OpCode.NEG:
                result = -values[0]
            elif op.opcode == OpCode.SQRT:
                result = np.sqrt(values[0])
            elif op.opcode == OpCode.EXP:
                result = np.exp(values[0])
            elif op.opcode == OpCode.LOG:
                result = np.log(values[0])
            elif op.opcode == OpCode.MATMUL:
                result = np.matmul(values[0], values[1])
            elif op.opcode == OpCode.TRANSPOSE:
                perm = op.get_attr("perm")
                result = np.transpose(values[0], perm)
            else:
                return None

            return Constant.from_array(result.astype(np.float32))

        except Exception as e:
            logger.debug(f"Could not fold {op.opcode}: {e}")
            return None


class DeadCodeElimination(FunctionPass):
    """Remove unused operations."""

    name = "dead-code-elimination"

    def run_on_function(self, func: Function) -> bool:
        modified = False

        # Find live values (outputs and their transitive inputs)
        live_ops = set()
        worklist = []

        # Start with return operations
        for block in func.body.blocks:
            for op in block.operations:
                if op.opcode == OpCode.RETURN:
                    worklist.append(op)
                    live_ops.add(op.id)

        # Propagate liveness
        while worklist:
            op = worklist.pop()
            for inp in op.inputs:
                if inp.defining_op and inp.defining_op.id not in live_ops:
                    live_ops.add(inp.defining_op.id)
                    worklist.append(inp.defining_op)

        # Remove dead ops
        for block in func.body.blocks:
            dead_ops = [op for op in block.operations if op.id not in live_ops]
            for op in dead_ops:
                block.remove_operation(op)
                modified = True

        return modified


class CommonSubexpressionElimination(FunctionPass):
    """Eliminate common subexpressions."""

    name = "cse"

    def run_on_function(self, func: Function) -> bool:
        modified = False

        # Hash-based CSE
        seen = {}  # hash -> Operation

        for block in func.body.blocks:
            ops_to_remove = []

            for op in block.operations:
                # Skip non-pure ops
                if op.opcode in {OpCode.LOAD, OpCode.STORE, OpCode.CALL}:
                    continue

                # Compute hash
                op_hash = self._hash_op(op)
                if op_hash in seen:
                    # Replace uses with existing value
                    existing = seen[op_hash]
                    for i, output in enumerate(op.outputs):
                        for use in list(output.uses):
                            # Replace input
                            for j, inp in enumerate(use.inputs):
                                if inp == output:
                                    use.inputs[j] = existing.outputs[i]
                    ops_to_remove.append(op)
                    modified = True
                else:
                    seen[op_hash] = op

            for op in ops_to_remove:
                block.remove_operation(op)

        return modified

    def _hash_op(self, op: Operation) -> tuple:
        """Create hash for operation."""
        input_ids = tuple(inp.id for inp in op.inputs)
        attrs = tuple(sorted(op.attributes.items()))
        return (op.opcode, input_ids, attrs)


class OperatorFusion(FunctionPass):
    """Fuse consecutive operations."""

    name = "operator-fusion"

    def run_on_function(self, func: Function) -> bool:
        modified = False

        for block in func.body.blocks:
            # Fuse elementwise chains
            modified |= self._fuse_elementwise_chain(block)

            # Fuse matmul + bias
            modified |= self._fuse_matmul_bias(block)

            # Fuse attention patterns
            modified |= self._fuse_attention(block)

        return modified

    def _fuse_elementwise_chain(self, block: Block) -> bool:
        """Fuse chains of elementwise operations."""
        modified = False
        i = 0

        while i < len(block.operations) - 1:
            op = block.operations[i]
            if not op.is_elementwise:
                i += 1
                continue

            # Look for elementwise consumer
            output = op.outputs[0]
            consumer = None
            for use in output.uses:
                if use._parent_block == block and use.is_elementwise:
                    # Only fuse if this is the only use
                    if len(output.uses) == 1:
                        consumer = use
                        break

            if consumer:
                # Create fused operation
                fused = Operation(
                    OpCode.FUSED,
                    op.inputs,
                    consumer.outputs,
                    {"fused_ops": [op.opcode, consumer.opcode]}
                )
                fused._fused_ops = [op, consumer]

                # Insert fused op
                idx = block.operations.index(op)
                block.operations[idx] = fused

                # Remove consumer
                block.operations.remove(consumer)
                modified = True
            else:
                i += 1

        return modified

    def _fuse_matmul_bias(self, block: Block) -> bool:
        """Fuse matmul followed by bias add."""
        modified = False
        i = 0

        while i < len(block.operations) - 1:
            op = block.operations[i]
            if op.opcode != OpCode.MATMUL:
                i += 1
                continue

            output = op.outputs[0]
            # Look for add as consumer
            for use in output.uses:
                if use._parent_block == block and use.opcode == OpCode.ADD:
                    if len(output.uses) == 1:
                        # Create fused matmul+bias
                        fused = Operation(
                            OpCode.FUSED,
                            op.inputs + [use.inputs[1] if use.inputs[0] == output else use.inputs[0]],
                            use.outputs,
                            {"fused_ops": [OpCode.MATMUL, OpCode.ADD], "has_bias": True}
                        )
                        block.operations[i] = fused
                        block.operations.remove(use)
                        modified = True
                        break
            i += 1

        return modified

    def _fuse_attention(self, block: Block) -> bool:
        """Detect and fuse attention pattern."""
        # Simplified attention fusion
        # Pattern: matmul(Q, K^T) -> scale -> softmax -> matmul(_, V)
        modified = False

        # Find softmax operations
        for i, op in enumerate(block.operations):
            if op.opcode != OpCode.SOFTMAX:
                continue

            # Check if input is matmul
            if op.inputs[0].defining_op and op.inputs[0].defining_op.opcode == OpCode.MATMUL:
                qk_matmul = op.inputs[0].defining_op

                # Check if output goes to matmul
                for use in op.outputs[0].uses:
                    if use.opcode == OpCode.MATMUL and use._parent_block == block:
                        # Found attention pattern - create fused op
                        q = qk_matmul.inputs[0]
                        k = qk_matmul.inputs[1]
                        v = use.inputs[1] if use.inputs[0] == op.outputs[0] else use.inputs[0]

                        fused = Operation(
                            OpCode.ATTENTION,
                            [q, k, v],
                            use.outputs,
                            {"fused": True}
                        )

                        # Replace ops
                        idx = block.operations.index(qk_matmul)
                        block.operations[idx] = fused
                        block.operations.remove(op)
                        block.operations.remove(use)
                        modified = True
                        break

        return modified


class LayoutOptimization(FunctionPass):
    """Optimize memory layouts for operations."""

    name = "layout-optimization"

    def run_on_function(self, func: Function) -> bool:
        modified = False

        for block in func.body.blocks:
            for op in block.operations:
                if op.opcode == OpCode.MATMUL:
                    # Add layout hints for matmul
                    op.set_attr("lhs_layout", "row_major")
                    op.set_attr("rhs_layout", "column_major")
                    modified = True

                elif op.opcode == OpCode.CONV2D:
                    # NCHW -> NHWC for better memory access
                    op.set_attr("data_format", "NHWC")
                    modified = True

        return modified


class StrengthReduction(FunctionPass):
    """Reduce operation strength where possible."""

    name = "strength-reduction"

    def run_on_function(self, func: Function) -> bool:
        modified = False

        for block in func.body.blocks:
            for i, op in enumerate(block.operations):
                # x * 2 -> x + x
                if op.opcode == OpCode.MUL:
                    if self._is_const_value(op.inputs[1], 2.0):
                        new_op = Operation(
                            OpCode.ADD,
                            [op.inputs[0], op.inputs[0]],
                            op.outputs
                        )
                        block.operations[i] = new_op
                        modified = True

                # x / 2 -> x * 0.5
                elif op.opcode == OpCode.DIV:
                    if self._is_const_value(op.inputs[1], 2.0):
                        # Would need to create constant 0.5
                        pass

        return modified

    def _is_const_value(self, value: Value, target: float) -> bool:
        """Check if value is constant with target value."""
        if value.defining_op and value.defining_op.opcode == OpCode.CONSTANT:
            const = value.defining_op.get_attr("value")
            if const and hasattr(const, 'value'):
                arr = const.value
                if arr.size == 1:
                    return float(arr.flat[0]) == target
        return False


class AlgebraicSimplification(FunctionPass):
    """Apply algebraic simplifications."""

    name = "algebraic-simplification"

    def run_on_function(self, func: Function) -> bool:
        modified = False

        for block in func.body.blocks:
            ops_to_remove = []

            for op in block.operations:
                # x + 0 -> x
                if op.opcode == OpCode.ADD:
                    if self._is_zero(op.inputs[1]):
                        self._replace_with_input(op, op.inputs[0])
                        ops_to_remove.append(op)
                        modified = True
                    elif self._is_zero(op.inputs[0]):
                        self._replace_with_input(op, op.inputs[1])
                        ops_to_remove.append(op)
                        modified = True

                # x * 1 -> x
                elif op.opcode == OpCode.MUL:
                    if self._is_one(op.inputs[1]):
                        self._replace_with_input(op, op.inputs[0])
                        ops_to_remove.append(op)
                        modified = True
                    elif self._is_one(op.inputs[0]):
                        self._replace_with_input(op, op.inputs[1])
                        ops_to_remove.append(op)
                        modified = True

                # x * 0 -> 0
                elif op.opcode == OpCode.MUL:
                    if self._is_zero(op.inputs[0]) or self._is_zero(op.inputs[1]):
                        # Replace with zero constant
                        zero_op = Operation(
                            OpCode.CONSTANT, [], op.outputs,
                            {"value": Constant.from_scalar(0.0), "type": op.outputs[0].type}
                        )
                        block.operations[block.operations.index(op)] = zero_op
                        modified = True

            for op in ops_to_remove:
                if op in block.operations:
                    block.remove_operation(op)

        return modified

    def _is_zero(self, value: Value) -> bool:
        """Check if value is zero constant."""
        if value.defining_op and value.defining_op.opcode == OpCode.CONSTANT:
            const = value.defining_op.get_attr("value")
            if const:
                return np.all(const.value == 0)
        return False

    def _is_one(self, value: Value) -> bool:
        """Check if value is one constant."""
        if value.defining_op and value.defining_op.opcode == OpCode.CONSTANT:
            const = value.defining_op.get_attr("value")
            if const:
                return np.all(const.value == 1)
        return False

    def _replace_with_input(self, op: Operation, replacement: Value):
        """Replace operation output with input."""
        for output in op.outputs:
            for use in list(output.uses):
                for i, inp in enumerate(use.inputs):
                    if inp == output:
                        use.inputs[i] = replacement


@dataclass
class PassError:
    """Record of a single optimization pass that raised an exception.

    The pipeline is resilient by default: a failing pass is logged and
    recorded here, then skipped, so the compile continues with the IR
    as it stood before the failure rather than silently no-op'ing.
    """

    pass_name: str
    iteration: int
    exception: Exception

    def __str__(self) -> str:
        return (
            f"pass '{self.pass_name}' failed on iteration {self.iteration}: "
            f"{type(self.exception).__name__}: {self.exception}"
        )


@dataclass
class PassManager:
    """Manages and runs optimization passes.

    By default the manager is *resilient*: if a pass raises, the exception is
    logged with the pass name and recorded on ``errors`` (and in
    ``pass_results``), and the pipeline continues with the un-optimized IR. Set
    ``strict=True`` to re-raise the first failure instead, which is useful when
    developing a new pass.
    """

    passes: list[Pass] = field(default_factory=list)
    max_iterations: int = 10
    strict: bool = False
    # Populated by ``run``: one PassError per failed pass invocation.
    errors: list[PassError] = field(default_factory=list)
    # Populated by ``run``: (pass_name, iteration, ok, modified) per invocation.
    pass_results: list[tuple[str, int, bool, bool]] = field(default_factory=list)

    def add_pass(self, pass_: Pass):
        """Add pass to pipeline."""
        self.passes.append(pass_)

    @property
    def has_errors(self) -> bool:
        """True if any pass failed during the last ``run``."""
        return bool(self.errors)

    def run(self, module: IRModule) -> IRModule:
        """Run all passes on module.

        Args:
            module: IR module

        Returns:
            Optimized module

        Raises:
            Exception: If ``strict`` is True and a pass raises, the original
                exception is re-raised after being logged and recorded.
        """
        # Reset per-run diagnostics so repeated runs do not accumulate.
        self.errors = []
        self.pass_results = []

        for iteration in range(self.max_iterations):
            modified = False

            for pass_ in self.passes:
                try:
                    pass_modified = bool(pass_.run(module))
                    self.pass_results.append(
                        (pass_.name, iteration, True, pass_modified)
                    )
                    if pass_modified:
                        modified = True
                        logger.debug(f"Pass {pass_.name} modified module")
                except Exception as e:
                    # Record the failure so it is visible rather than silent.
                    error = PassError(pass_.name, iteration, e)
                    self.errors.append(error)
                    self.pass_results.append((pass_.name, iteration, False, False))
                    logger.error(
                        "Optimization pass '%s' failed on iteration %d; "
                        "continuing with un-optimized IR: %s: %s",
                        pass_.name,
                        iteration,
                        type(e).__name__,
                        e,
                        exc_info=True,
                    )
                    if self.strict:
                        raise

            if not modified:
                logger.info(f"Optimization converged after {iteration + 1} iterations")
                break

        return module


def create_default_pipeline() -> PassManager:
    """Create default optimization pipeline."""
    pm = PassManager()

    # Canonicalization
    pm.add_pass(AlgebraicSimplification())

    # High-level optimizations
    pm.add_pass(ConstantFolding())
    pm.add_pass(CommonSubexpressionElimination())
    pm.add_pass(DeadCodeElimination())

    # Operator fusion
    pm.add_pass(OperatorFusion())

    # Low-level optimizations
    pm.add_pass(StrengthReduction())
    pm.add_pass(LayoutOptimization())

    # Cleanup
    pm.add_pass(DeadCodeElimination())

    return pm
