# ML Compiler API Documentation

## Table of Contents
1. [Compiler API](#compiler-api)
2. [IR Builder API](#ir-builder-api)
3. [Optimization API](#optimization-api)
4. [Code Generation API](#code-generation-api)
5. [Runtime API](#runtime-api)

## Compiler API

### MLCompiler

The main compiler interface for compiling ML models.

```python
from mlcompiler import MLCompiler, CompilerConfig, OptimizationLevel, Target

# Create compiler with configuration
config = CompilerConfig(
    target=Target.CUDA,
    optimization_level=OptimizationLevel.O2,
    debug=False
)
compiler = MLCompiler(config)
```

#### Methods

##### compile(module: Module) -> CompiledModel
Compile an IR module to executable code.

**Parameters:**
- `module`: IR module to compile

**Returns:**
- `CompiledModel`: Compiled executable model

**Example:**
```python
module = builder.get_module()
compiled = compiler.compile(module)

# Run the compiled model
output = compiled.run(input_data)
```

##### compile_from_onnx(path: str) -> CompiledModel
Compile directly from ONNX model.

**Example:**
```python
compiled = compiler.compile_from_onnx("model.onnx")
```

### CompilerConfig

Configuration for the compiler.

```python
config = CompilerConfig(
    target=Target.CPU,              # Target hardware
    optimization_level=OptimizationLevel.O2,  # Optimization level
    debug=True,                      # Enable debug output
    profile=True,                    # Enable profiling
    mixed_precision=False,           # Use mixed precision
    vectorize=True,                  # Enable vectorization
    parallel=True,                   # Enable parallelization
    num_threads=8,                   # Number of threads
    memory_limit=4*1024**3,          # Memory limit (4GB)
    cache_dir="/tmp/mlcompiler",    # Cache directory
)
```

### OptimizationLevel

Optimization levels for compilation.

```python
class OptimizationLevel(Enum):
    O0 = 0  # No optimization
    O1 = 1  # Basic optimizations
    O2 = 2  # Standard optimizations (recommended)
    O3 = 3  # Aggressive optimizations
    Os = 4  # Optimize for size
```

### Target

Hardware targets for code generation.

```python
class Target(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    OPENCL = "opencl"
    METAL = "metal"
    WEBGPU = "webgpu"
    VULKAN = "vulkan"
```

## IR Builder API

### IRBuilder

Builder for constructing IR modules.

```python
from mlcompiler.ir import IRBuilder

builder = IRBuilder()
builder.begin_block("my_model")
```

#### Input/Output Operations

##### create_input(type: TensorType, name: str) -> Value
Create an input tensor.

**Example:**
```python
x = builder.create_input(
    TensorType(DataType.FLOAT32, Shape([32, 784])),
    name="input"
)
```

##### create_weight(type: TensorType, name: str, initializer=None) -> Value
Create a weight tensor.

**Example:**
```python
w = builder.create_weight(
    TensorType(DataType.FLOAT32, Shape([784, 256])),
    name="weight",
    initializer="xavier_uniform"
)
```

##### create_constant(value, dtype=None, name=None) -> Value
Create a constant value.

**Example:**
```python
const = builder.create_constant(2.5, dtype=DataType.FLOAT32)
```

##### set_return(value: Value)
Set the return value of the current function.

**Example:**
```python
output = builder.add_softmax(logits)
builder.set_return(output)
```

#### Neural Network Operations

##### add_convolution(input, weight, stride, padding, dilation=1, groups=1)
Add 2D convolution operation.

**Example:**
```python
conv = builder.add_convolution(
    input, weight,
    stride=(2, 2),
    padding=(1, 1),
    dilation=(1, 1),
    groups=1
)
```

##### add_batch_norm(input, epsilon=1e-5, momentum=0.1, training=False)
Add batch normalization.

**Example:**
```python
bn = builder.add_batch_norm(conv, training=False)
```

##### add_matmul(a, b, transpose_a=False, transpose_b=False)
Add matrix multiplication.

**Example:**
```python
output = builder.add_matmul(input, weight)
```

##### add_pooling(input, pool_type, kernel_size, stride, padding=0)
Add pooling operation.

**Example:**
```python
pool = builder.add_pooling(
    input,
    pool_type="max",
    kernel_size=(2, 2),
    stride=(2, 2)
)
```

#### Activation Functions

##### add_relu(input, negative_slope=0)
Add ReLU activation.

**Example:**
```python
activated = builder.add_relu(input)
# Leaky ReLU
activated = builder.add_relu(input, negative_slope=0.01)
```

##### add_sigmoid(input)
Add sigmoid activation.

##### add_tanh(input)
Add tanh activation.

##### add_softmax(input, axis=-1)
Add softmax activation.

**Example:**
```python
probs = builder.add_softmax(logits, axis=-1)
```

##### add_gelu(input)
Add GELU activation.

#### Tensor Operations

##### add_reshape(input, shape: Shape)
Reshape tensor.

**Example:**
```python
reshaped = builder.add_reshape(input, Shape([32, -1]))
```

##### add_transpose(input, axes: List[int])
Transpose tensor dimensions.

**Example:**
```python
transposed = builder.add_transpose(input, axes=[0, 2, 1])
```

##### add_concat(inputs: List[Value], axis: int)
Concatenate tensors.

**Example:**
```python
concat = builder.add_concat([tensor1, tensor2, tensor3], axis=1)
```

##### add_split(input, num_splits: int, axis: int)
Split tensor.

**Example:**
```python
splits = builder.add_split(input, num_splits=4, axis=1)
```

##### add_reduce(input, reduce_type: str, axes: List[int], keepdims=False)
Reduce operation.

**Example:**
```python
# Sum along axis 1
sum_result = builder.add_reduce(input, "sum", axes=[1], keepdims=True)

# Global mean
mean = builder.add_reduce(input, "mean", axes=[1, 2, 3])
```

#### Control Flow

##### if_block(condition) -> IfContext
Create conditional execution block.

**Example:**
```python
with builder.if_block(condition) as ctx:
    true_value = builder.add_relu(x)
    ctx.set_true_branch(true_value)

    false_value = builder.add_sigmoid(x)
    ctx.set_false_branch(false_value)

result = ctx.get_result()
```

##### for_loop(start, stop, step=1) -> LoopContext
Create for loop.

**Example:**
```python
with builder.for_loop(0, 100) as loop:
    i = loop.get_index()
    # Loop body operations
```

##### while_loop() -> WhileLoopContext
Create while loop.

**Example:**
```python
counter = builder.create_variable(0)
with builder.while_loop() as loop:
    cond = builder.less_than(counter.get(), 100)
    loop.set_condition(cond)

    # Loop body
    counter.set(builder.add(counter.get(), 1))
```

### TensorType

Type descriptor for tensors.

```python
from mlcompiler.ir.types import TensorType, DataType, Shape

tensor_type = TensorType(
    dtype=DataType.FLOAT32,
    shape=Shape([32, 3, 224, 224])
)
```

### DataType

Supported data types.

```python
class DataType(Enum):
    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"
```

### Shape

Shape descriptor for tensors.

```python
# Static shape
shape = Shape([32, 128, 256])

# Dynamic shape (batch dimension)
shape = Shape([None, 128, 256])

# Symbolic dimensions
shape = Shape(["batch", 128, "seq_len"])
```

## Optimization API

### PassManager

Manages optimization passes.

```python
from mlcompiler.optimization import PassManager

pass_manager = PassManager()
pass_manager.add_pass(ConstantFolding())
pass_manager.add_pass(DeadCodeElimination())
pass_manager.add_pass(OperatorFusion())

optimized = pass_manager.run(module)
```

### OptimizationPass

Base class for optimization passes.

```python
class CustomPass(OptimizationPass):
    def run(self, module: Module) -> Module:
        # Transform module
        return transformed_module
```

### Built-in Passes

#### ConstantFolding
Evaluate operations on constants at compile time.

```python
pass_manager.add_pass(ConstantFolding(aggressive=True))
```

#### DeadCodeElimination
Remove unreachable code.

```python
pass_manager.add_pass(DeadCodeElimination())
```

#### CommonSubexpressionElimination
Eliminate redundant computations.

```python
pass_manager.add_pass(CommonSubexpressionElimination())
```

#### OperatorFusion
Fuse compatible operations.

```python
pass_manager.add_pass(OperatorFusion(
    patterns=["conv_bn_relu", "matmul_bias", "multi_head_attention"]
))
```

#### MemoryOptimization
Optimize memory usage.

```python
pass_manager.add_pass(MemoryOptimization(
    enable_inplace=True,
    enable_reuse=True,
    enable_checkpointing=True
))
```

#### LoopOptimization
Optimize loop structures.

```python
pass_manager.add_pass(LoopOptimization(
    unroll_factor=4,
    tile_size=32,
    enable_fusion=True
))
```

#### Vectorization
Enable SIMD vectorization.

```python
pass_manager.add_pass(Vectorization(
    vector_width=8,  # AVX-256
    alignment=32
))
```

## Code Generation API

### CodeGenerator

Base class for code generators.

```python
from mlcompiler.codegen import CodeGenerator, Target

codegen = CodeGenerator(target=Target.CPU)
code = codegen.generate(module)
```

### CUDACodeGen

CUDA-specific code generator.

```python
from mlcompiler.codegen.cuda import CUDACodeGen

cuda_gen = CUDACodeGen(
    compute_capability="8.6",  # Ampere
    use_tensor_cores=True,
    use_cudnn=True
)
```

#### Kernel Configuration

```python
kernel_config = {
    "block_size": (256, 1, 1),
    "grid_size": "auto",
    "shared_memory": 48*1024,  # 48KB
    "registers_per_thread": 64
}

cuda_gen.set_kernel_config(kernel_config)
```

### Code Generation Options

```python
gen_options = {
    "optimization_level": 2,
    "fast_math": True,
    "fuse_multiply_add": True,
    "unroll_loops": True,
    "inline_functions": True,
    "vectorize": True
}

codegen = CodeGenerator(target=Target.CPU, **gen_options)
```

## Runtime API

### CompiledModel

Compiled model ready for execution.

```python
# Load compiled model
model = CompiledModel.load("model.mlc")

# Run inference
output = model.run(input_data)

# Batch inference
outputs = model.run_batch([input1, input2, input3])
```

#### Methods

##### run(inputs: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray
Execute the model with given inputs.

**Example:**
```python
# Single input
output = model.run(input_tensor)

# Multiple inputs
outputs = model.run({
    "input1": tensor1,
    "input2": tensor2
})
```

##### benchmark(inputs, num_runs=100, warmup=10) -> BenchmarkResult
Benchmark model performance.

**Example:**
```python
result = model.benchmark(input_data, num_runs=1000)
print(f"Mean latency: {result.mean_time:.2f}ms")
print(f"Throughput: {result.throughput:.0f} samples/sec")
```

##### profile(inputs) -> ProfileResult
Profile model execution.

**Example:**
```python
profile = model.profile(input_data)
print(profile.summary())

# Detailed operation timing
for op, time in profile.operation_times.items():
    print(f"{op}: {time:.3f}ms")
```

### Memory Management

```python
# Pre-allocate memory pools
model.allocate_memory(batch_size=32)

# Get memory statistics
stats = model.get_memory_stats()
print(f"Peak memory: {stats.peak_memory / 1024**2:.1f}MB")

# Clear memory
model.clear_memory()
```

### Debugging

```python
# Enable debug mode
model.set_debug_mode(True)

# Set breakpoint at operation
model.set_breakpoint("conv2d_1")

# Step through execution
for step in model.step_through(input_data):
    print(f"Operation: {step.operation}")
    print(f"Output shape: {step.output.shape}")
```

## Advanced Usage

### Custom Patterns

Define custom fusion patterns:

```python
from mlcompiler.optimization import Pattern, PatternBuilder

builder = PatternBuilder()
pattern = builder.create_pattern("custom_fusion")

# Define pattern: Conv -> Add -> ReLU
conv = pattern.add_op("Convolution")
add = pattern.add_op("Add")
relu = pattern.add_op("ReLU")

pattern.add_edge(conv, add)
pattern.add_edge(add, relu)

# Register pattern
compiler.register_pattern(pattern, fused_op="CustomFusedOp")
```

### Auto-tuning

Enable auto-tuning for optimal performance:

```python
from mlcompiler import AutoTuner

tuner = AutoTuner(
    model=model,
    inputs=sample_inputs,
    metric="latency",  # or "throughput"
    timeout=3600,       # 1 hour
    num_trials=100
)

best_config = tuner.tune()
model.apply_config(best_config)
```

### Distributed Compilation

Compile for distributed execution:

```python
from mlcompiler.distributed import DistributedCompiler

dist_compiler = DistributedCompiler(
    devices=["gpu:0", "gpu:1", "gpu:2", "gpu:3"],
    strategy="data_parallel"  # or "model_parallel"
)

dist_model = dist_compiler.compile(module)
```

### Quantization

Apply quantization for inference:

```python
from mlcompiler.quantization import quantize

quantized_model = quantize(
    model,
    calibration_data=calib_data,
    method="symmetric",  # or "asymmetric"
    bits=8               # INT8 quantization
)
```

## Error Handling

```python
from mlcompiler.errors import CompilationError, RuntimeError

try:
    compiled = compiler.compile(module)
except CompilationError as e:
    print(f"Compilation failed: {e.message}")
    print(f"Location: {e.location}")
    print(f"Suggestion: {e.suggestion}")

try:
    output = model.run(input_data)
except RuntimeError as e:
    print(f"Runtime error: {e.message}")
    print(f"Operation: {e.operation}")
    print(f"Traceback: {e.traceback}")
```

## Examples

### Complete Example: CNN Model

```python
from mlcompiler import MLCompiler, CompilerConfig, Target
from mlcompiler.ir import IRBuilder
from mlcompiler.ir.types import TensorType, DataType, Shape

# Build model
builder = IRBuilder()
builder.begin_block("cnn_classifier")

# Input image [batch, channels, height, width]
image = builder.create_input(
    TensorType(DataType.FLOAT32, Shape([1, 3, 224, 224])),
    name="image"
)

# First convolutional layer
conv1_w = builder.create_weight(
    TensorType(DataType.FLOAT32, Shape([64, 3, 7, 7])),
    name="conv1_weight"
)
conv1 = builder.add_convolution(image, conv1_w, stride=(2, 2), padding=(3, 3))
conv1 = builder.add_batch_norm(conv1)
conv1 = builder.add_relu(conv1)
pool1 = builder.add_pooling(conv1, "max", kernel_size=(3, 3), stride=(2, 2))

# Second convolutional layer
conv2_w = builder.create_weight(
    TensorType(DataType.FLOAT32, Shape([128, 64, 3, 3])),
    name="conv2_weight"
)
conv2 = builder.add_convolution(pool1, conv2_w, stride=(1, 1), padding=(1, 1))
conv2 = builder.add_batch_norm(conv2)
conv2 = builder.add_relu(conv2)

# Global average pooling
gap = builder.add_reduce(conv2, "mean", axes=[2, 3])

# Classifier
fc_w = builder.create_weight(
    TensorType(DataType.FLOAT32, Shape([128, 1000])),
    name="fc_weight"
)
fc_b = builder.create_weight(
    TensorType(DataType.FLOAT32, Shape([1000])),
    name="fc_bias"
)

logits = builder.add_matmul(gap, fc_w)
logits = builder.add_bias_add(logits, fc_b)
output = builder.add_softmax(logits)

builder.set_return(output)

# Compile model
config = CompilerConfig(
    target=Target.CUDA,
    optimization_level=OptimizationLevel.O2
)
compiler = MLCompiler(config)
module = builder.get_module()
compiled = compiler.compile(module)

# Run inference
import numpy as np
input_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
predictions = compiled.run(input_image)

print(f"Output shape: {predictions.shape}")
print(f"Top prediction: {np.argmax(predictions)}")
```