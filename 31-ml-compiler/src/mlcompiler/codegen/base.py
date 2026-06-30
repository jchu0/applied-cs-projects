"""Base code generator for ML compiler."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import logging

from ..ir import IRModule, Function, Operation, OpCode, Value, TensorType, DType
from ..memory import MemoryPlan, BufferAllocation

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Container for generated code."""
    source: str
    language: str
    entry_point: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CodeGenerator(ABC):
    """Base class for code generators."""

    target: str = "unknown"

    def __init__(self, memory_plan: MemoryPlan = None):
        """Initialize code generator.

        Args:
            memory_plan: Memory allocation plan
        """
        self.memory_plan = memory_plan
        self._indent = 0
        self._code_lines = []

    @abstractmethod
    def generate(self, module: IRModule) -> GeneratedCode:
        """Generate code for module.

        Args:
            module: IR module

        Returns:
            Generated code
        """
        pass

    def _emit(self, line: str = ""):
        """Emit a line of code."""
        if line:
            self._code_lines.append("  " * self._indent + line)
        else:
            self._code_lines.append("")

    def _indent_inc(self):
        """Increase indentation."""
        self._indent += 1

    def _indent_dec(self):
        """Decrease indentation."""
        self._indent = max(0, self._indent - 1)

    def _get_code(self) -> str:
        """Get accumulated code."""
        return "\n".join(self._code_lines)

    def _clear_code(self):
        """Clear accumulated code."""
        self._code_lines = []
        self._indent = 0

    def _dtype_to_ctype(self, dtype: DType) -> str:
        """Convert dtype to C type."""
        mapping = {
            DType.FLOAT16: "half",
            DType.FLOAT32: "float",
            DType.FLOAT64: "double",
            DType.INT8: "int8_t",
            DType.INT16: "int16_t",
            DType.INT32: "int32_t",
            DType.INT64: "int64_t",
            DType.UINT8: "uint8_t",
            DType.BOOL: "bool",
        }
        return mapping.get(dtype, "float")

    def _get_buffer_ptr(self, value: Value) -> str:
        """Get buffer pointer for value."""
        if self.memory_plan and value.id in self.memory_plan.allocations:
            alloc = self.memory_plan.allocations[value.id]
            return f"(buffer + {alloc.offset})"
        return f"buf_{value.id}"


class CPUCodeGenerator(CodeGenerator):
    """Code generator for CPU execution."""

    target = "cpu"

    def generate(self, module: IRModule) -> GeneratedCode:
        """Generate C code for CPU.

        Args:
            module: IR module

        Returns:
            Generated C code
        """
        self._clear_code()

        # Includes
        self._emit("#include <math.h>")
        self._emit("#include <string.h>")
        self._emit("#include <stdint.h>")
        self._emit("#include <stdbool.h>")
        self._emit("#include <cblas.h>")
        self._emit()

        # Generate functions
        for func in module.functions.values():
            self._generate_function(func)

        return GeneratedCode(
            source=self._get_code(),
            language="c",
            entry_point=list(module.functions.keys())[0] if module.functions else "main",
            metadata={"target": "cpu"}
        )

    def _generate_function(self, func: Function):
        """Generate code for function."""
        # Function signature
        args = []
        for i, arg in enumerate(func.arguments):
            ctype = self._dtype_to_ctype(arg.type.dtype)
            args.append(f"{ctype}* arg{i}")

        # Output arguments
        for i, out_type in enumerate(func.func_type.output_types):
            ctype = self._dtype_to_ctype(out_type.dtype)
            args.append(f"{ctype}* out{i}")

        self._emit(f"void {func.name}({', '.join(args)}) {{")
        self._indent_inc()

        # Allocate temporaries
        if self.memory_plan:
            self._emit(f"char buffer[{self.memory_plan.total_size}];")
            self._emit()

        # Generate operations
        for block in func.body.blocks:
            for op in block.operations:
                self._generate_operation(op)

        self._indent_dec()
        self._emit("}")
        self._emit()

    def _generate_operation(self, op: Operation):
        """Generate code for operation."""
        if op.opcode == OpCode.ADD:
            self._generate_elementwise(op, "+")
        elif op.opcode == OpCode.SUB:
            self._generate_elementwise(op, "-")
        elif op.opcode == OpCode.MUL:
            self._generate_elementwise(op, "*")
        elif op.opcode == OpCode.DIV:
            self._generate_elementwise(op, "/")
        elif op.opcode == OpCode.NEG:
            self._generate_unary(op, "-")
        elif op.opcode == OpCode.SQRT:
            self._generate_unary_func(op, "sqrtf")
        elif op.opcode == OpCode.EXP:
            self._generate_unary_func(op, "expf")
        elif op.opcode == OpCode.LOG:
            self._generate_unary_func(op, "logf")
        elif op.opcode == OpCode.MATMUL:
            self._generate_matmul(op)
        elif op.opcode == OpCode.RELU:
            self._generate_relu(op)
        elif op.opcode == OpCode.SIGMOID:
            self._generate_sigmoid(op)
        elif op.opcode == OpCode.SOFTMAX:
            self._generate_softmax(op)
        elif op.opcode == OpCode.REDUCE_SUM:
            self._generate_reduce_sum(op)
        elif op.opcode == OpCode.TRANSPOSE:
            self._generate_transpose(op)
        elif op.opcode == OpCode.CONSTANT:
            self._generate_constant(op)
        elif op.opcode == OpCode.RETURN:
            self._generate_return(op)
        else:
            self._emit(f"// TODO: {op.opcode.name}")

    def _generate_elementwise(self, op: Operation, operator: str):
        """Generate elementwise binary operation."""
        out = op.outputs[0]
        a = op.inputs[0]
        b = op.inputs[1]
        n = out.type.num_elements

        out_ptr = self._get_buffer_ptr(out)
        a_ptr = self._get_buffer_ptr(a)
        b_ptr = self._get_buffer_ptr(b)

        self._emit(f"// {op.opcode.name}")
        self._emit(f"for (int i = 0; i < {n}; i++) {{")
        self._indent_inc()
        self._emit(f"{out_ptr}[i] = {a_ptr}[i] {operator} {b_ptr}[i];")
        self._indent_dec()
        self._emit("}")
        self._emit()

    def _generate_unary(self, op: Operation, operator: str):
        """Generate unary operation."""
        out = op.outputs[0]
        x = op.inputs[0]
        n = out.type.num_elements

        out_ptr = self._get_buffer_ptr(out)
        x_ptr = self._get_buffer_ptr(x)

        self._emit(f"// {op.opcode.name}")
        self._emit(f"for (int i = 0; i < {n}; i++) {{")
        self._indent_inc()
        self._emit(f"{out_ptr}[i] = {operator}{x_ptr}[i];")
        self._indent_dec()
        self._emit("}")
        self._emit()

    def _generate_unary_func(self, op: Operation, func_name: str):
        """Generate unary function operation."""
        out = op.outputs[0]
        x = op.inputs[0]
        n = out.type.num_elements

        out_ptr = self._get_buffer_ptr(out)
        x_ptr = self._get_buffer_ptr(x)

        self._emit(f"// {op.opcode.name}")
        self._emit(f"for (int i = 0; i < {n}; i++) {{")
        self._indent_inc()
        self._emit(f"{out_ptr}[i] = {func_name}({x_ptr}[i]);")
        self._indent_dec()
        self._emit("}")
        self._emit()

    def _generate_matmul(self, op: Operation):
        """Generate matrix multiplication using BLAS."""
        out = op.outputs[0]
        a = op.inputs[0]
        b = op.inputs[1]

        # Get dimensions
        m = a.type.shape[-2]
        k = a.type.shape[-1]
        n = b.type.shape[-1]

        out_ptr = self._get_buffer_ptr(out)
        a_ptr = self._get_buffer_ptr(a)
        b_ptr = self._get_buffer_ptr(b)

        self._emit("// MATMUL")
        self._emit(f"cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,")
        self._emit(f"            {m}, {n}, {k}, 1.0f,")
        self._emit(f"            {a_ptr}, {k},")
        self._emit(f"            {b_ptr}, {n},")
        self._emit(f"            0.0f, {out_ptr}, {n});")
        self._emit()

    def _generate_relu(self, op: Operation):
        """Generate ReLU activation."""
        out = op.outputs[0]
        x = op.inputs[0]
        n = out.type.num_elements

        out_ptr = self._get_buffer_ptr(out)
        x_ptr = self._get_buffer_ptr(x)

        self._emit("// RELU")
        self._emit(f"for (int i = 0; i < {n}; i++) {{")
        self._indent_inc()
        self._emit(f"{out_ptr}[i] = {x_ptr}[i] > 0 ? {x_ptr}[i] : 0;")
        self._indent_dec()
        self._emit("}")
        self._emit()

    def _generate_sigmoid(self, op: Operation):
        """Generate sigmoid activation."""
        out = op.outputs[0]
        x = op.inputs[0]
        n = out.type.num_elements

        out_ptr = self._get_buffer_ptr(out)
        x_ptr = self._get_buffer_ptr(x)

        self._emit("// SIGMOID")
        self._emit(f"for (int i = 0; i < {n}; i++) {{")
        self._indent_inc()
        self._emit(f"{out_ptr}[i] = 1.0f / (1.0f + expf(-{x_ptr}[i]));")
        self._indent_dec()
        self._emit("}")
        self._emit()

    def _generate_softmax(self, op: Operation):
        """Generate softmax operation."""
        out = op.outputs[0]
        x = op.inputs[0]
        n = out.type.num_elements

        out_ptr = self._get_buffer_ptr(out)
        x_ptr = self._get_buffer_ptr(x)

        self._emit("// SOFTMAX")
        self._emit("{")
        self._indent_inc()
        self._emit("float max_val = -INFINITY;")
        self._emit(f"for (int i = 0; i < {n}; i++) {{")
        self._indent_inc()
        self._emit(f"if ({x_ptr}[i] > max_val) max_val = {x_ptr}[i];")
        self._indent_dec()
        self._emit("}")
        self._emit("float sum = 0;")
        self._emit(f"for (int i = 0; i < {n}; i++) {{")
        self._indent_inc()
        self._emit(f"{out_ptr}[i] = expf({x_ptr}[i] - max_val);")
        self._emit(f"sum += {out_ptr}[i];")
        self._indent_dec()
        self._emit("}")
        self._emit(f"for (int i = 0; i < {n}; i++) {{")
        self._indent_inc()
        self._emit(f"{out_ptr}[i] /= sum;")
        self._indent_dec()
        self._emit("}")
        self._indent_dec()
        self._emit("}")
        self._emit()

    def _generate_reduce_sum(self, op: Operation):
        """Generate sum reduction."""
        out = op.outputs[0]
        x = op.inputs[0]
        n = x.type.num_elements

        out_ptr = self._get_buffer_ptr(out)
        x_ptr = self._get_buffer_ptr(x)

        self._emit("// REDUCE_SUM")
        self._emit(f"{out_ptr}[0] = 0;")
        self._emit(f"for (int i = 0; i < {n}; i++) {{")
        self._indent_inc()
        self._emit(f"{out_ptr}[0] += {x_ptr}[i];")
        self._indent_dec()
        self._emit("}")
        self._emit()

    def _generate_transpose(self, op: Operation):
        """Generate transpose operation."""
        out = op.outputs[0]
        x = op.inputs[0]

        out_ptr = self._get_buffer_ptr(out)
        x_ptr = self._get_buffer_ptr(x)

        # Simple 2D transpose
        if len(x.type.shape) == 2:
            rows, cols = x.type.shape
            self._emit("// TRANSPOSE")
            self._emit(f"for (int i = 0; i < {rows}; i++) {{")
            self._indent_inc()
            self._emit(f"for (int j = 0; j < {cols}; j++) {{")
            self._indent_inc()
            self._emit(f"{out_ptr}[j * {rows} + i] = {x_ptr}[i * {cols} + j];")
            self._indent_dec()
            self._emit("}")
            self._indent_dec()
            self._emit("}")
        else:
            self._emit(f"// TODO: Transpose for rank {len(x.type.shape)}")
        self._emit()

    def _generate_constant(self, op: Operation):
        """Generate constant initialization."""
        out = op.outputs[0]
        const = op.get_attr("value")

        out_ptr = self._get_buffer_ptr(out)

        self._emit(f"// CONSTANT {out.id}")
        if const and hasattr(const, 'value'):
            values = const.value.flatten()
            if len(values) <= 8:
                vals_str = ", ".join(f"{v}f" for v in values)
                self._emit(f"float _const_{out.id}[] = {{{vals_str}}};")
                self._emit(f"memcpy({out_ptr}, _const_{out.id}, {len(values) * 4});")
            else:
                self._emit(f"// Large constant: {len(values)} elements")
        self._emit()

    def _generate_return(self, op: Operation):
        """Generate return operation."""
        self._emit("// RETURN")
        for i, val in enumerate(op.inputs):
            val_ptr = self._get_buffer_ptr(val)
            n = val.type.num_elements
            self._emit(f"memcpy(out{i}, {val_ptr}, {n * 4});")
        self._emit()
