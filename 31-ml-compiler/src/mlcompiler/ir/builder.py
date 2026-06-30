"""IR builder for constructing operations."""

from typing import Any
import uuid
import numpy as np

from .types import Value, TensorType, DType, Constant
from .operations import Operation, OpCode, Block
from .module import Function, IRModule


class IRBuilder:
    """Builder for constructing IR operations."""

    def __init__(self, block: Block = None):
        """Initialize builder.

        Args:
            block: Block to insert operations into
        """
        self._block = block
        self._value_counter = 0

    def set_insertion_point(self, block: Block):
        """Set insertion point to end of block."""
        self._block = block

    def _create_value(self, tensor_type: TensorType, name: str = "") -> Value:
        """Create a new SSA value."""
        value_id = f"v{self._value_counter}"
        self._value_counter += 1
        return Value(id=value_id, type=tensor_type, name=name)

    def _add_op(self, op: Operation) -> Operation:
        """Add operation to current block."""
        if self._block is not None:
            self._block.add_operation(op)
        return op

    # Arithmetic operations

    def add(self, a: Value, b: Value, name: str = "") -> Value:
        """Create add operation."""
        output = self._create_value(a.type, name)
        op = Operation(OpCode.ADD, [a, b], [output])
        self._add_op(op)
        return output

    def sub(self, a: Value, b: Value, name: str = "") -> Value:
        """Create subtraction operation."""
        output = self._create_value(a.type, name)
        op = Operation(OpCode.SUB, [a, b], [output])
        self._add_op(op)
        return output

    def mul(self, a: Value, b: Value, name: str = "") -> Value:
        """Create multiplication operation."""
        output = self._create_value(a.type, name)
        op = Operation(OpCode.MUL, [a, b], [output])
        self._add_op(op)
        return output

    def div(self, a: Value, b: Value, name: str = "") -> Value:
        """Create division operation."""
        output = self._create_value(a.type, name)
        op = Operation(OpCode.DIV, [a, b], [output])
        self._add_op(op)
        return output

    def neg(self, x: Value, name: str = "") -> Value:
        """Create negation operation."""
        output = self._create_value(x.type, name)
        op = Operation(OpCode.NEG, [x], [output])
        self._add_op(op)
        return output

    def sqrt(self, x: Value, name: str = "") -> Value:
        """Create square root operation."""
        output = self._create_value(x.type, name)
        op = Operation(OpCode.SQRT, [x], [output])
        self._add_op(op)
        return output

    def exp(self, x: Value, name: str = "") -> Value:
        """Create exponential operation."""
        output = self._create_value(x.type, name)
        op = Operation(OpCode.EXP, [x], [output])
        self._add_op(op)
        return output

    def log(self, x: Value, name: str = "") -> Value:
        """Create logarithm operation."""
        output = self._create_value(x.type, name)
        op = Operation(OpCode.LOG, [x], [output])
        self._add_op(op)
        return output

    # Matrix operations

    def matmul(self, a: Value, b: Value, name: str = "") -> Value:
        """Create matrix multiplication operation."""
        # Compute output shape
        a_shape = a.type.shape
        b_shape = b.type.shape
        if len(a_shape) == 2 and len(b_shape) == 2:
            out_shape = (a_shape[0], b_shape[1])
        else:
            batch = a_shape[:-2]
            out_shape = batch + (a_shape[-2], b_shape[-1])

        output_type = TensorType(out_shape, a.type.dtype)
        output = self._create_value(output_type, name)
        op = Operation(OpCode.MATMUL, [a, b], [output])
        self._add_op(op)
        return output

    def transpose(self, x: Value, perm: list[int] = None, name: str = "") -> Value:
        """Create transpose operation."""
        if perm is None:
            perm = list(range(len(x.type.shape) - 1, -1, -1))

        new_shape = tuple(x.type.shape[i] for i in perm)
        output_type = TensorType(new_shape, x.type.dtype)
        output = self._create_value(output_type, name)
        op = Operation(OpCode.TRANSPOSE, [x], [output], {"perm": perm})
        self._add_op(op)
        return output

    # Reduction operations

    def reduce_sum(
        self,
        x: Value,
        axis: int | list[int] = None,
        keepdims: bool = False,
        name: str = ""
    ) -> Value:
        """Create sum reduction."""
        output_type = self._compute_reduction_type(x.type, axis, keepdims)
        output = self._create_value(output_type, name)
        op = Operation(
            OpCode.REDUCE_SUM, [x], [output],
            {"axis": axis, "keepdims": keepdims}
        )
        self._add_op(op)
        return output

    def reduce_max(
        self,
        x: Value,
        axis: int | list[int] = None,
        keepdims: bool = False,
        name: str = ""
    ) -> Value:
        """Create max reduction."""
        output_type = self._compute_reduction_type(x.type, axis, keepdims)
        output = self._create_value(output_type, name)
        op = Operation(
            OpCode.REDUCE_MAX, [x], [output],
            {"axis": axis, "keepdims": keepdims}
        )
        self._add_op(op)
        return output

    def reduce_mean(
        self,
        x: Value,
        axis: int | list[int] = None,
        keepdims: bool = False,
        name: str = ""
    ) -> Value:
        """Create mean reduction."""
        output_type = self._compute_reduction_type(x.type, axis, keepdims)
        output = self._create_value(output_type, name)
        op = Operation(
            OpCode.REDUCE_MEAN, [x], [output],
            {"axis": axis, "keepdims": keepdims}
        )
        self._add_op(op)
        return output

    def _compute_reduction_type(
        self,
        input_type: TensorType,
        axis: int | list[int],
        keepdims: bool
    ) -> TensorType:
        """Compute output type for reduction."""
        if axis is None:
            return TensorType((), input_type.dtype)

        new_shape = list(input_type.shape)
        if isinstance(axis, int):
            axis = [axis]

        for ax in sorted(axis, reverse=True):
            if keepdims:
                new_shape[ax] = 1
            else:
                del new_shape[ax]

        return TensorType(tuple(new_shape), input_type.dtype)

    # Neural network operations

    def relu(self, x: Value, name: str = "") -> Value:
        """Create ReLU activation."""
        output = self._create_value(x.type, name)
        op = Operation(OpCode.RELU, [x], [output])
        self._add_op(op)
        return output

    def gelu(self, x: Value, name: str = "") -> Value:
        """Create GELU activation."""
        output = self._create_value(x.type, name)
        op = Operation(OpCode.GELU, [x], [output])
        self._add_op(op)
        return output

    def sigmoid(self, x: Value, name: str = "") -> Value:
        """Create sigmoid activation."""
        output = self._create_value(x.type, name)
        op = Operation(OpCode.SIGMOID, [x], [output])
        self._add_op(op)
        return output

    def tanh(self, x: Value, name: str = "") -> Value:
        """Create tanh activation."""
        output = self._create_value(x.type, name)
        op = Operation(OpCode.TANH, [x], [output])
        self._add_op(op)
        return output

    def softmax(self, x: Value, axis: int = -1, name: str = "") -> Value:
        """Create softmax operation."""
        output = self._create_value(x.type, name)
        op = Operation(OpCode.SOFTMAX, [x], [output], {"axis": axis})
        self._add_op(op)
        return output

    def layernorm(
        self,
        x: Value,
        gamma: Value,
        beta: Value,
        eps: float = 1e-5,
        name: str = ""
    ) -> Value:
        """Create layer normalization."""
        output = self._create_value(x.type, name)
        op = Operation(
            OpCode.LAYERNORM, [x, gamma, beta], [output],
            {"eps": eps}
        )
        self._add_op(op)
        return output

    def conv2d(
        self,
        x: Value,
        weight: Value,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        name: str = ""
    ) -> Value:
        """Create 2D convolution."""
        n, c, h, w = x.type.shape
        out_channels = weight.type.shape[0]
        kh, kw = weight.type.shape[2:]
        out_h = (h + 2 * padding[0] - kh) // stride[0] + 1
        out_w = (w + 2 * padding[1] - kw) // stride[1] + 1

        output_type = TensorType((n, out_channels, out_h, out_w), x.type.dtype)
        output = self._create_value(output_type, name)
        op = Operation(
            OpCode.CONV2D, [x, weight], [output],
            {
                "stride": stride,
                "padding": padding,
                "out_channels": out_channels,
                "kernel_size": (kh, kw)
            }
        )
        self._add_op(op)
        return output

    # Attention

    def attention(
        self,
        query: Value,
        key: Value,
        value: Value,
        mask: Value = None,
        scale: float = None,
        name: str = ""
    ) -> Value:
        """Create attention operation."""
        # Output same shape as value for standard attention
        output = self._create_value(query.type, name)
        inputs = [query, key, value]
        if mask:
            inputs.append(mask)
        op = Operation(
            OpCode.ATTENTION, inputs, [output],
            {"scale": scale}
        )
        self._add_op(op)
        return output

    def flash_attention(
        self,
        query: Value,
        key: Value,
        value: Value,
        name: str = ""
    ) -> Value:
        """Create flash attention operation."""
        output = self._create_value(query.type, name)
        op = Operation(OpCode.FLASH_ATTENTION, [query, key, value], [output])
        self._add_op(op)
        return output

    # Memory operations

    def constant(self, value: np.ndarray, name: str = "") -> Value:
        """Create constant value."""
        const = Constant.from_array(value)
        output = self._create_value(const.type, name)
        op = Operation(
            OpCode.CONSTANT, [], [output],
            {"value": const, "type": const.type}
        )
        self._add_op(op)
        return output

    def reshape(self, x: Value, shape: tuple[int, ...], name: str = "") -> Value:
        """Create reshape operation."""
        output_type = TensorType(shape, x.type.dtype)
        output = self._create_value(output_type, name)
        op = Operation(OpCode.RESHAPE, [x], [output], {"shape": shape})
        self._add_op(op)
        return output

    def concat(self, values: list[Value], axis: int = 0, name: str = "") -> Value:
        """Create concatenation operation."""
        # Compute output shape
        shapes = [v.type.shape for v in values]
        out_shape = list(shapes[0])
        out_shape[axis] = sum(s[axis] for s in shapes)

        output_type = TensorType(tuple(out_shape), values[0].type.dtype)
        output = self._create_value(output_type, name)
        op = Operation(OpCode.CONCAT, values, [output], {"axis": axis})
        self._add_op(op)
        return output

    def slice(
        self,
        x: Value,
        starts: list[int],
        ends: list[int],
        name: str = ""
    ) -> Value:
        """Create slice operation."""
        out_shape = tuple(e - s for s, e in zip(starts, ends))
        output_type = TensorType(out_shape, x.type.dtype)
        output = self._create_value(output_type, name)
        op = Operation(
            OpCode.SLICE, [x], [output],
            {"starts": starts, "ends": ends}
        )
        self._add_op(op)
        return output

    # Control flow

    def return_op(self, values: list[Value]):
        """Create return operation."""
        op = Operation(OpCode.RETURN, values, [])
        self._add_op(op)
        return op

    def call(self, func_name: str, args: list[Value], result_types: list[TensorType], name: str = "") -> list[Value]:
        """Create function call."""
        outputs = [self._create_value(t, f"{name}_{i}") for i, t in enumerate(result_types)]
        op = Operation(OpCode.CALL, args, outputs, {"callee": func_name})
        self._add_op(op)
        return outputs
