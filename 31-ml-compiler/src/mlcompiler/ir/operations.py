"""IR operations for ML compiler."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
import uuid

from .types import Value, TensorType, Attribute, DType


class OpCode(Enum):
    """Operation codes for IR."""
    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    SQRT = auto()
    EXP = auto()
    LOG = auto()
    POW = auto()
    ABS = auto()

    # Comparison
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()

    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()

    # Matrix operations
    MATMUL = auto()
    DOT = auto()
    TRANSPOSE = auto()
    BROADCAST = auto()

    # Reductions
    REDUCE_SUM = auto()
    REDUCE_MAX = auto()
    REDUCE_MIN = auto()
    REDUCE_MEAN = auto()

    # Neural network operations
    CONV2D = auto()
    POOL2D = auto()
    BATCHNORM = auto()
    LAYERNORM = auto()
    SOFTMAX = auto()
    RELU = auto()
    GELU = auto()
    SIGMOID = auto()
    TANH = auto()
    DROPOUT = auto()

    # Attention
    ATTENTION = auto()
    FLASH_ATTENTION = auto()

    # Memory operations
    LOAD = auto()
    STORE = auto()
    CONSTANT = auto()
    RESHAPE = auto()
    SLICE = auto()
    CONCAT = auto()
    PAD = auto()
    GATHER = auto()
    SCATTER = auto()

    # Control flow
    IF = auto()
    WHILE = auto()
    CALL = auto()
    RETURN = auto()

    # Special
    FUSED = auto()
    CUSTOM = auto()


@dataclass
class Operation:
    """SSA operation in the IR."""
    opcode: OpCode
    inputs: list[Value]
    outputs: list[Value]
    attributes: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Metadata for optimization
    _parent_block: Any = None
    _fused_ops: list = field(default_factory=list)

    def __post_init__(self):
        # Set defining op for outputs
        for output in self.outputs:
            output.defining_op = self

        # Register uses
        for inp in self.inputs:
            if inp and hasattr(inp, 'uses'):
                inp.uses.append(self)

    def get_attr(self, name: str, default: Any = None) -> Any:
        """Get attribute value."""
        return self.attributes.get(name, default)

    def set_attr(self, name: str, value: Any):
        """Set attribute value."""
        self.attributes[name] = value

    @property
    def is_elementwise(self) -> bool:
        """Check if operation is elementwise."""
        elementwise_ops = {
            OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV,
            OpCode.NEG, OpCode.SQRT, OpCode.EXP, OpCode.LOG,
            OpCode.ABS, OpCode.RELU, OpCode.SIGMOID, OpCode.TANH,
            OpCode.GELU, OpCode.POW,
        }
        return self.opcode in elementwise_ops

    @property
    def is_reduction(self) -> bool:
        """Check if operation is a reduction."""
        return self.opcode in {
            OpCode.REDUCE_SUM, OpCode.REDUCE_MAX,
            OpCode.REDUCE_MIN, OpCode.REDUCE_MEAN
        }

    @property
    def is_compute_intensive(self) -> bool:
        """Check if operation is compute intensive."""
        return self.opcode in {
            OpCode.MATMUL, OpCode.CONV2D, OpCode.ATTENTION,
            OpCode.FLASH_ATTENTION
        }

    @property
    def is_memory_bound(self) -> bool:
        """Check if operation is memory bound."""
        return self.opcode in {
            OpCode.LOAD, OpCode.STORE, OpCode.GATHER,
            OpCode.SCATTER, OpCode.RESHAPE, OpCode.TRANSPOSE
        }

    def compute_output_type(self) -> TensorType:
        """Compute output type based on inputs and operation."""
        if not self.inputs:
            if self.opcode == OpCode.CONSTANT:
                return self.get_attr("type", TensorType((), DType.FLOAT32))
            return TensorType((), DType.FLOAT32)

        input_type = self.inputs[0].type

        if self.is_elementwise:
            # Elementwise ops preserve shape
            return input_type

        if self.opcode == OpCode.MATMUL:
            # Matrix multiply
            a_shape = self.inputs[0].type.shape
            b_shape = self.inputs[1].type.shape
            if len(a_shape) == 2 and len(b_shape) == 2:
                out_shape = (a_shape[0], b_shape[1])
            else:
                # Batched matmul
                batch = a_shape[:-2]
                out_shape = batch + (a_shape[-2], b_shape[-1])
            return TensorType(out_shape, input_type.dtype)

        if self.opcode == OpCode.TRANSPOSE:
            perm = self.get_attr("perm", list(range(len(input_type.shape) - 1, -1, -1)))
            new_shape = tuple(input_type.shape[i] for i in perm)
            return TensorType(new_shape, input_type.dtype)

        if self.is_reduction:
            axis = self.get_attr("axis", None)
            keepdims = self.get_attr("keepdims", False)
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

        if self.opcode == OpCode.RESHAPE:
            new_shape = self.get_attr("shape", input_type.shape)
            return TensorType(tuple(new_shape), input_type.dtype)

        if self.opcode == OpCode.SOFTMAX:
            return input_type

        if self.opcode == OpCode.CONV2D:
            # Simplified conv2d output shape
            n, c, h, w = input_type.shape
            out_channels = self.get_attr("out_channels", c)
            kernel = self.get_attr("kernel_size", (3, 3))
            stride = self.get_attr("stride", (1, 1))
            padding = self.get_attr("padding", (0, 0))
            out_h = (h + 2 * padding[0] - kernel[0]) // stride[0] + 1
            out_w = (w + 2 * padding[1] - kernel[1]) // stride[1] + 1
            return TensorType((n, out_channels, out_h, out_w), input_type.dtype)

        return input_type

    def __str__(self) -> str:
        inputs_str = ", ".join(str(inp) for inp in self.inputs)
        outputs_str = ", ".join(str(out) for out in self.outputs)
        attrs_str = ", ".join(f"{k}={v}" for k, v in self.attributes.items())

        result = f"{outputs_str} = {self.opcode.name.lower()}({inputs_str})"
        if attrs_str:
            result += f" {{{attrs_str}}}"
        return result


@dataclass
class Block:
    """Basic block containing operations."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operations: list[Operation] = field(default_factory=list)
    arguments: list[Value] = field(default_factory=list)
    _parent: Any = None

    def add_operation(self, op: Operation):
        """Add operation to block."""
        op._parent_block = self
        self.operations.append(op)

    def insert_operation(self, index: int, op: Operation):
        """Insert operation at index."""
        op._parent_block = self
        self.operations.insert(index, op)

    def remove_operation(self, op: Operation):
        """Remove operation from block."""
        self.operations.remove(op)
        op._parent_block = None

    def __iter__(self):
        return iter(self.operations)

    def __len__(self):
        return len(self.operations)

    def __bool__(self) -> bool:
        """Block is always truthy - use 'is None' to check for absence."""
        return True


@dataclass
class Region:
    """Region containing blocks."""
    blocks: list[Block] = field(default_factory=list)

    @property
    def entry_block(self) -> Block:
        """Get entry block."""
        return self.blocks[0] if self.blocks else None

    def add_block(self, block: Block = None) -> Block:
        """Add block to region."""
        if block is None:
            block = Block()
        block._parent = self
        self.blocks.append(block)
        return block
