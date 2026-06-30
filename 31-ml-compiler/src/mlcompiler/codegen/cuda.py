"""CUDA code generator for ML compiler."""

from ..ir import IRModule, Function, Operation, OpCode, Value, TensorType, DType
from ..memory import MemoryPlan
from .base import CodeGenerator, GeneratedCode


class CUDACodeGenerator(CodeGenerator):
    """Code generator for CUDA GPUs."""

    target = "cuda"

    def __init__(self, memory_plan: MemoryPlan = None, block_size: int = 256):
        """Initialize CUDA code generator.

        Args:
            memory_plan: Memory allocation plan
            block_size: CUDA thread block size
        """
        super().__init__(memory_plan)
        self.block_size = block_size

    def generate(self, module: IRModule) -> GeneratedCode:
        """Generate CUDA code.

        Args:
            module: IR module

        Returns:
            Generated CUDA code
        """
        self._clear_code()

        # CUDA includes
        self._emit("#include <cuda_runtime.h>")
        self._emit("#include <cublas_v2.h>")
        self._emit("#include <math.h>")
        self._emit()

        # Generate kernels and host functions
        for func in module.functions.values():
            self._generate_function(func)

        return GeneratedCode(
            source=self._get_code(),
            language="cuda",
            entry_point=list(module.functions.keys())[0] if module.functions else "main",
            metadata={
                "target": "cuda",
                "block_size": self.block_size
            }
        )

    def _generate_function(self, func: Function):
        """Generate CUDA code for function."""
        # Generate kernels for operations
        kernels = self._generate_kernels(func)

        # Generate host function
        self._generate_host_function(func, kernels)

    def _generate_kernels(self, func: Function) -> list[str]:
        """Generate CUDA kernels for operations."""
        kernels = []

        for block in func.body.blocks:
            for op in block.operations:
                kernel_name = self._generate_kernel(op)
                if kernel_name:
                    kernels.append(kernel_name)

        return kernels

    def _generate_kernel(self, op: Operation) -> str:
        """Generate kernel for operation."""
        if op.opcode == OpCode.ADD:
            return self._generate_elementwise_kernel(op, "add", "+")
        elif op.opcode == OpCode.SUB:
            return self._generate_elementwise_kernel(op, "sub", "-")
        elif op.opcode == OpCode.MUL:
            return self._generate_elementwise_kernel(op, "mul", "*")
        elif op.opcode == OpCode.DIV:
            return self._generate_elementwise_kernel(op, "div", "/")
        elif op.opcode == OpCode.RELU:
            return self._generate_relu_kernel(op)
        elif op.opcode == OpCode.SIGMOID:
            return self._generate_sigmoid_kernel(op)
        elif op.opcode == OpCode.SOFTMAX:
            return self._generate_softmax_kernel(op)
        elif op.opcode == OpCode.REDUCE_SUM:
            return self._generate_reduce_sum_kernel(op)

        return None

    def _generate_elementwise_kernel(self, op: Operation, name: str, operator: str) -> str:
        """Generate elementwise binary kernel."""
        kernel_name = f"kernel_{name}_{op.id}"
        n = op.outputs[0].type.num_elements

        self._emit(f"__global__ void {kernel_name}(float* out, float* a, float* b, int n) {{")
        self._indent_inc()
        self._emit("int idx = blockIdx.x * blockDim.x + threadIdx.x;")
        self._emit("if (idx < n) {")
        self._indent_inc()
        self._emit(f"out[idx] = a[idx] {operator} b[idx];")
        self._indent_dec()
        self._emit("}")
        self._indent_dec()
        self._emit("}")
        self._emit()

        return kernel_name

    def _generate_relu_kernel(self, op: Operation) -> str:
        """Generate ReLU kernel."""
        kernel_name = f"kernel_relu_{op.id}"

        self._emit(f"__global__ void {kernel_name}(float* out, float* x, int n) {{")
        self._indent_inc()
        self._emit("int idx = blockIdx.x * blockDim.x + threadIdx.x;")
        self._emit("if (idx < n) {")
        self._indent_inc()
        self._emit("out[idx] = x[idx] > 0 ? x[idx] : 0;")
        self._indent_dec()
        self._emit("}")
        self._indent_dec()
        self._emit("}")
        self._emit()

        return kernel_name

    def _generate_sigmoid_kernel(self, op: Operation) -> str:
        """Generate sigmoid kernel."""
        kernel_name = f"kernel_sigmoid_{op.id}"

        self._emit(f"__global__ void {kernel_name}(float* out, float* x, int n) {{")
        self._indent_inc()
        self._emit("int idx = blockIdx.x * blockDim.x + threadIdx.x;")
        self._emit("if (idx < n) {")
        self._indent_inc()
        self._emit("out[idx] = 1.0f / (1.0f + expf(-x[idx]));")
        self._indent_dec()
        self._emit("}")
        self._indent_dec()
        self._emit("}")
        self._emit()

        return kernel_name

    def _generate_softmax_kernel(self, op: Operation) -> str:
        """Generate softmax kernels (max, exp-sum, normalize)."""
        n = op.outputs[0].type.num_elements

        # Max kernel
        max_kernel = f"kernel_softmax_max_{op.id}"
        self._emit(f"__global__ void {max_kernel}(float* max_val, float* x, int n) {{")
        self._indent_inc()
        self._emit("__shared__ float smax[256];")
        self._emit("int tid = threadIdx.x;")
        self._emit("int idx = blockIdx.x * blockDim.x + threadIdx.x;")
        self._emit("smax[tid] = (idx < n) ? x[idx] : -INFINITY;")
        self._emit("__syncthreads();")
        self._emit("for (int s = blockDim.x / 2; s > 0; s >>= 1) {")
        self._indent_inc()
        self._emit("if (tid < s && smax[tid] < smax[tid + s]) {")
        self._indent_inc()
        self._emit("smax[tid] = smax[tid + s];")
        self._indent_dec()
        self._emit("}")
        self._emit("__syncthreads();")
        self._indent_dec()
        self._emit("}")
        self._emit("if (tid == 0) atomicMax((int*)max_val, __float_as_int(smax[0]));")
        self._indent_dec()
        self._emit("}")
        self._emit()

        # Exp and sum kernel
        exp_kernel = f"kernel_softmax_exp_{op.id}"
        self._emit(f"__global__ void {exp_kernel}(float* out, float* sum, float* x, float max_val, int n) {{")
        self._indent_inc()
        self._emit("int idx = blockIdx.x * blockDim.x + threadIdx.x;")
        self._emit("if (idx < n) {")
        self._indent_inc()
        self._emit("float val = expf(x[idx] - max_val);")
        self._emit("out[idx] = val;")
        self._emit("atomicAdd(sum, val);")
        self._indent_dec()
        self._emit("}")
        self._indent_dec()
        self._emit("}")
        self._emit()

        # Normalize kernel
        norm_kernel = f"kernel_softmax_norm_{op.id}"
        self._emit(f"__global__ void {norm_kernel}(float* out, float sum, int n) {{")
        self._indent_inc()
        self._emit("int idx = blockIdx.x * blockDim.x + threadIdx.x;")
        self._emit("if (idx < n) {")
        self._indent_inc()
        self._emit("out[idx] /= sum;")
        self._indent_dec()
        self._emit("}")
        self._indent_dec()
        self._emit("}")
        self._emit()

        return max_kernel

    def _generate_reduce_sum_kernel(self, op: Operation) -> str:
        """Generate parallel reduction kernel."""
        kernel_name = f"kernel_reduce_sum_{op.id}"

        self._emit(f"__global__ void {kernel_name}(float* out, float* x, int n) {{")
        self._indent_inc()
        self._emit("__shared__ float sdata[256];")
        self._emit("int tid = threadIdx.x;")
        self._emit("int idx = blockIdx.x * blockDim.x + threadIdx.x;")
        self._emit("sdata[tid] = (idx < n) ? x[idx] : 0;")
        self._emit("__syncthreads();")
        self._emit()
        self._emit("// Parallel reduction")
        self._emit("for (int s = blockDim.x / 2; s > 0; s >>= 1) {")
        self._indent_inc()
        self._emit("if (tid < s) {")
        self._indent_inc()
        self._emit("sdata[tid] += sdata[tid + s];")
        self._indent_dec()
        self._emit("}")
        self._emit("__syncthreads();")
        self._indent_dec()
        self._emit("}")
        self._emit("if (tid == 0) atomicAdd(out, sdata[0]);")
        self._indent_dec()
        self._emit("}")
        self._emit()

        return kernel_name

    def _generate_host_function(self, func: Function, kernels: list[str]):
        """Generate host function that launches kernels."""
        # Function signature
        args = []
        for i, arg in enumerate(func.arguments):
            args.append(f"float* d_arg{i}")

        for i, out_type in enumerate(func.func_type.output_types):
            args.append(f"float* d_out{i}")

        self._emit(f"void {func.name}({', '.join(args)}) {{")
        self._indent_inc()

        # Allocate device temporaries
        if self.memory_plan:
            self._emit(f"float* d_buffer;")
            self._emit(f"cudaMalloc(&d_buffer, {self.memory_plan.total_size});")
            self._emit()

        # Generate kernel launches
        for block in func.body.blocks:
            for op in block.operations:
                self._generate_kernel_launch(op)

        # Free temporaries
        if self.memory_plan:
            self._emit("cudaFree(d_buffer);")

        self._indent_dec()
        self._emit("}")
        self._emit()

    def _generate_kernel_launch(self, op: Operation):
        """Generate kernel launch for operation."""
        if op.opcode in {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV}:
            n = op.outputs[0].type.num_elements
            grid = (n + self.block_size - 1) // self.block_size

            out_ptr = self._get_buffer_ptr(op.outputs[0])
            a_ptr = self._get_buffer_ptr(op.inputs[0])
            b_ptr = self._get_buffer_ptr(op.inputs[1])

            op_name = op.opcode.name.lower()
            kernel_name = f"kernel_{op_name}_{op.id}"

            self._emit(f"// {op.opcode.name}")
            self._emit(f"{kernel_name}<<<{grid}, {self.block_size}>>>({out_ptr}, {a_ptr}, {b_ptr}, {n});")
            self._emit()

        elif op.opcode == OpCode.RELU:
            n = op.outputs[0].type.num_elements
            grid = (n + self.block_size - 1) // self.block_size

            out_ptr = self._get_buffer_ptr(op.outputs[0])
            x_ptr = self._get_buffer_ptr(op.inputs[0])

            self._emit("// RELU")
            self._emit(f"kernel_relu_{op.id}<<<{grid}, {self.block_size}>>>({out_ptr}, {x_ptr}, {n});")
            self._emit()

        elif op.opcode == OpCode.MATMUL:
            # Use cuBLAS for matmul
            self._generate_cublas_matmul(op)

        elif op.opcode == OpCode.RETURN:
            # Copy outputs
            self._emit("// RETURN")
            for i, val in enumerate(op.inputs):
                val_ptr = self._get_buffer_ptr(val)
                n = val.type.num_elements
                self._emit(f"cudaMemcpy(d_out{i}, {val_ptr}, {n * 4}, cudaMemcpyDeviceToDevice);")
            self._emit()

    def _generate_cublas_matmul(self, op: Operation):
        """Generate cuBLAS matrix multiplication."""
        out = op.outputs[0]
        a = op.inputs[0]
        b = op.inputs[1]

        m = a.type.shape[-2]
        k = a.type.shape[-1]
        n = b.type.shape[-1]

        out_ptr = self._get_buffer_ptr(out)
        a_ptr = self._get_buffer_ptr(a)
        b_ptr = self._get_buffer_ptr(b)

        self._emit("// MATMUL (cuBLAS)")
        self._emit("{")
        self._indent_inc()
        self._emit("cublasHandle_t handle;")
        self._emit("cublasCreate(&handle);")
        self._emit("float alpha = 1.0f, beta = 0.0f;")
        self._emit(f"cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,")
        self._emit(f"            {n}, {m}, {k}, &alpha,")
        self._emit(f"            {b_ptr}, {n},")
        self._emit(f"            {a_ptr}, {k},")
        self._emit(f"            &beta, {out_ptr}, {n});")
        self._emit("cublasDestroy(handle);")
        self._indent_dec()
        self._emit("}")
        self._emit()


class TritonCodeGenerator(CodeGenerator):
    """Code generator for Triton kernels."""

    target = "triton"

    def generate(self, module: IRModule) -> GeneratedCode:
        """Generate Triton Python code.

        Args:
            module: IR module

        Returns:
            Generated Triton code
        """
        self._clear_code()

        # Imports
        self._emit("import triton")
        self._emit("import triton.language as tl")
        self._emit("import torch")
        self._emit()

        # Generate functions
        for func in module.functions.values():
            self._generate_function(func)

        return GeneratedCode(
            source=self._get_code(),
            language="python",
            entry_point=list(module.functions.keys())[0] if module.functions else "main",
            metadata={"target": "triton"}
        )

    def _generate_function(self, func: Function):
        """Generate Triton kernel for function."""
        # For simple functions, generate a fused kernel
        self._emit("@triton.jit")
        self._emit(f"def {func.name}_kernel(")
        self._indent_inc()

        # Arguments
        for i, arg in enumerate(func.arguments):
            self._emit(f"arg{i}_ptr,")

        for i in range(len(func.func_type.output_types)):
            self._emit(f"out{i}_ptr,")

        self._emit("n_elements,")
        self._emit("BLOCK_SIZE: tl.constexpr,")
        self._indent_dec()
        self._emit("):")
        self._indent_inc()

        # Kernel body
        self._emit("pid = tl.program_id(0)")
        self._emit("block_start = pid * BLOCK_SIZE")
        self._emit("offsets = block_start + tl.arange(0, BLOCK_SIZE)")
        self._emit("mask = offsets < n_elements")
        self._emit()

        # Load inputs
        for i in range(len(func.arguments)):
            self._emit(f"x{i} = tl.load(arg{i}_ptr + offsets, mask=mask)")

        # Generate operations
        self._emit()
        result_var = f"x{len(func.arguments) - 1}"

        for block in func.body.blocks:
            for op in block.operations:
                result_var = self._generate_triton_op(op, result_var)

        # Store output
        self._emit()
        self._emit(f"tl.store(out0_ptr + offsets, {result_var}, mask=mask)")

        self._indent_dec()
        self._emit()

        # Generate wrapper
        self._generate_triton_wrapper(func)

    def _generate_triton_op(self, op: Operation, input_var: str) -> str:
        """Generate Triton operation."""
        if op.opcode == OpCode.ADD:
            self._emit(f"result = x0 + x1")
            return "result"
        elif op.opcode == OpCode.MUL:
            self._emit(f"result = x0 * x1")
            return "result"
        elif op.opcode == OpCode.RELU:
            self._emit(f"result = tl.maximum({input_var}, 0)")
            return "result"
        elif op.opcode == OpCode.SIGMOID:
            self._emit(f"result = tl.sigmoid({input_var})")
            return "result"
        elif op.opcode == OpCode.EXP:
            self._emit(f"result = tl.exp({input_var})")
            return "result"
        elif op.opcode == OpCode.SQRT:
            self._emit(f"result = tl.sqrt({input_var})")
            return "result"

        return input_var

    def _generate_triton_wrapper(self, func: Function):
        """Generate Python wrapper for Triton kernel."""
        self._emit(f"def {func.name}(")
        self._indent_inc()

        args = [f"arg{i}" for i in range(len(func.arguments))]
        self._emit(", ".join(args) + ",")
        self._indent_dec()
        self._emit("):")
        self._indent_inc()

        self._emit("n_elements = arg0.numel()")
        self._emit("out = torch.empty_like(arg0)")
        self._emit("grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)")
        self._emit()

        kernel_args = ", ".join([f"arg{i}" for i in range(len(func.arguments))])
        self._emit(f"{func.name}_kernel[grid]({kernel_args}, out, n_elements, BLOCK_SIZE=1024)")
        self._emit("return out")

        self._indent_dec()
        self._emit()
