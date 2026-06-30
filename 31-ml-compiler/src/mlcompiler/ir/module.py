"""IR module and function definitions."""

from dataclasses import dataclass, field
from typing import Any
import uuid

from .types import Value, TensorType, DType
from .operations import Block, Region, Operation, OpCode


@dataclass
class FunctionType:
    """Function type with input and output types."""
    input_types: list[TensorType]
    output_types: list[TensorType]

    def __str__(self) -> str:
        inputs = ", ".join(str(t) for t in self.input_types)
        outputs = ", ".join(str(t) for t in self.output_types)
        return f"({inputs}) -> ({outputs})"


@dataclass
class Function:
    """IR function definition."""
    name: str
    func_type: FunctionType
    body: Region = field(default_factory=Region)
    attributes: dict[str, Any] = field(default_factory=dict)
    _module: Any = None

    def __post_init__(self):
        # Create entry block with arguments
        if not self.body.blocks:
            entry = self.body.add_block()
            for i, input_type in enumerate(self.func_type.input_types):
                arg = Value(
                    id=f"arg{i}",
                    type=input_type,
                    name=f"arg{i}"
                )
                entry.arguments.append(arg)

    @property
    def entry_block(self) -> Block:
        """Get entry block."""
        return self.body.entry_block

    @property
    def arguments(self) -> list[Value]:
        """Get function arguments."""
        return self.entry_block.arguments if self.entry_block is not None else []

    def add_block(self) -> Block:
        """Add new block to function."""
        return self.body.add_block()

    def get_operations(self) -> list[Operation]:
        """Get all operations in function."""
        ops = []
        for block in self.body.blocks:
            ops.extend(block.operations)
        return ops

    def __str__(self) -> str:
        lines = [f"func @{self.name}{self.func_type} {{"]
        for block in self.body.blocks:
            lines.append(f"  ^{block.id}:")
            for op in block.operations:
                lines.append(f"    {op}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class IRModule:
    """Container for IR functions and globals."""
    name: str = "module"
    functions: dict[str, Function] = field(default_factory=dict)
    globals: dict[str, Value] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_function(self, func: Function):
        """Add function to module."""
        func._module = self
        self.functions[func.name] = func

    def get_function(self, name: str) -> Function:
        """Get function by name."""
        return self.functions.get(name)

    def create_function(
        self,
        name: str,
        input_types: list[TensorType],
        output_types: list[TensorType]
    ) -> Function:
        """Create and add a new function."""
        func_type = FunctionType(input_types, output_types)
        func = Function(name, func_type)
        self.add_function(func)
        return func

    def add_global(self, name: str, value: Value):
        """Add global value."""
        self.globals[name] = value

    def get_global(self, name: str) -> Value:
        """Get global value."""
        return self.globals.get(name)

    def __str__(self) -> str:
        lines = [f"module @{self.name} {{"]

        for name, value in self.globals.items():
            lines.append(f"  global @{name}: {value.type}")

        for func in self.functions.values():
            for line in str(func).split("\n"):
                lines.append(f"  {line}")

        lines.append("}")
        return "\n".join(lines)

    def clone(self) -> "IRModule":
        """Create a deep copy of the module."""
        # Simplified clone - in production would need deep copy
        import copy
        return copy.deepcopy(self)

    def verify(self) -> list[str]:
        """Verify module correctness."""
        errors = []

        for func_name, func in self.functions.items():
            # Check function has entry block
            if not func.entry_block:
                errors.append(f"Function {func_name} has no entry block")

            # Check operations
            for op in func.get_operations():
                # Verify inputs are defined
                for inp in op.inputs:
                    if inp.defining_op is None and inp not in func.arguments:
                        errors.append(
                            f"Input {inp} not defined in function {func_name}"
                        )

                # Verify output types match
                for output in op.outputs:
                    expected = op.compute_output_type()
                    if output.type.shape != expected.shape:
                        errors.append(
                            f"Output shape mismatch in {op.opcode}: "
                            f"expected {expected.shape}, got {output.type.shape}"
                        )

        return errors
