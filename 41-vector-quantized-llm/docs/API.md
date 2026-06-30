# Vector Quantized LLM - API Documentation

## Table of Contents

1. [Quantizers API](#quantizers-api)
2. [Calibration API](#calibration-api)
3. [Inference API](#inference-api)
4. [Core Types](#core-types)
5. [Utilities](#utilities)

## Quantizers API

### INT8Quantizer

8-bit integer quantization for balanced performance and accuracy.

```python
from vqllm.quantize import INT8Quantizer
from vqllm.core.types import QuantConfig, ScaleType

# Initialize quantizer
config = QuantConfig(
    quant_type=QuantType.INT8,
    scale_type=ScaleType.PER_CHANNEL,
    symmetric=True
)
quantizer = INT8Quantizer(config)

# Quantize a weight tensor
weight = np.random.randn(768, 768).astype(np.float32)
qtensor = quantizer.quantize_weight(weight, name="layer.weight")

# Quantize entire model
quantized_model = quantizer.quantize_model(model)
```

#### Methods

**`__init__(config: QuantConfig = None)`**
- Initialize the INT8 quantizer
- Parameters:
  - `config`: Quantization configuration (optional)

**`quantize_weight(weight: np.ndarray, name: str = "") -> QuantizedTensor`**
- Quantize a single weight tensor
- Parameters:
  - `weight`: Weight tensor to quantize
  - `name`: Layer name for logging
- Returns: QuantizedTensor object

**`quantize_model(model: Any) -> Any`**
- Quantize all weights in a model
- Parameters:
  - `model`: Model to quantize
- Returns: Quantized model

### INT4Quantizer

4-bit integer quantization for maximum compression.

```python
from vqllm.quantize import INT4Quantizer

# Initialize with custom block size
config = QuantConfig(
    quant_type=QuantType.INT4,
    block_size=128,
    group_size=64
)
quantizer = INT4Quantizer(config)

# Quantize weight with packing
weight = np.random.randn(1024, 1024).astype(np.float32)
qtensor = quantizer.quantize_weight(weight)

# Access packed data
packed_data = qtensor.packed_data  # 2 INT4 values per byte
print(f"Compression ratio: {weight.nbytes / packed_data.nbytes:.2f}x")
```

#### Methods

**`pack_int4(values: np.ndarray) -> np.ndarray`**
- Pack INT4 values into bytes
- Parameters:
  - `values`: INT4 values to pack
- Returns: Packed byte array

**`unpack_int4(packed: np.ndarray, shape: tuple) -> np.ndarray`**
- Unpack INT4 values from bytes
- Parameters:
  - `packed`: Packed byte array
  - `shape`: Original tensor shape
- Returns: Unpacked INT4 values

### GPTQQuantizer

Gradient-based Post-Training Quantization for optimal quality.

```python
from vqllm.quantize import GPTQQuantizer
from vqllm.calibration import compute_hessian

# Configure GPTQ
config = QuantConfig(
    quant_type=QuantType.GPTQ,
    bits=4,
    block_size=128,
    dampening=0.01,
    percdamp=0.01
)
quantizer = GPTQQuantizer(config)

# Compute Hessian from calibration data
calibration_inputs = [np.random.randn(32, 768) for _ in range(100)]
hessian = compute_hessian(calibration_inputs, dampening=0.01)

# Quantize with Hessian information
weight = np.random.randn(768, 3072).astype(np.float32)
qtensor = quantizer.quantize_weight(weight, "mlp.fc1", hessian=hessian)
```

#### Methods

**`quantize_weight(weight: np.ndarray, name: str = "", hessian: np.ndarray = None) -> QuantizedTensor`**
- Quantize weight using GPTQ algorithm
- Parameters:
  - `weight`: Weight tensor
  - `name`: Layer name
  - `hessian`: Hessian matrix for optimization
- Returns: Optimally quantized tensor

**`_compute_optimal_order(hessian: np.ndarray) -> np.ndarray`**
- Compute optimal quantization order
- Parameters:
  - `hessian`: Hessian matrix
- Returns: Optimal column ordering

### AWQQuantizer

Activation-aware Weight Quantization for improved accuracy.

```python
from vqllm.quantize import AWQQuantizer
from vqllm.calibration import collect_activations

# Configure AWQ
config = QuantConfig(
    quant_type=QuantType.AWQ,
    bits=4,
    group_size=128,
    zero_point=True
)
quantizer = AWQQuantizer(config)

# Collect activation scales
activations = [np.random.randn(32, 768) for _ in range(100)]
activation_scale = collect_activations(activations)["scale"]

# Quantize with activation awareness
weight = np.random.randn(768, 3072).astype(np.float32)
qtensor = quantizer.quantize_weight(
    weight,
    "mlp.fc1",
    activation_scale=activation_scale
)
```

#### Methods

**`quantize_weight(weight: np.ndarray, name: str = "", activation_scale: np.ndarray = None) -> QuantizedTensor`**
- Quantize weight with activation awareness
- Parameters:
  - `weight`: Weight tensor
  - `name`: Layer name
  - `activation_scale`: Per-channel activation scales
- Returns: Activation-aware quantized tensor

**`_search_optimal_scale(weight: np.ndarray, activation_scale: np.ndarray) -> np.ndarray`**
- Search for optimal scaling factors
- Parameters:
  - `weight`: Weight tensor
  - `activation_scale`: Activation scales
- Returns: Optimal scales

## Calibration API

### CalibrationDataset

Manages calibration data for quantization.

```python
from vqllm.calibration import CalibrationDataset

# From token data
data = [
    {"input_ids": np.random.randint(0, 1000, (128,))},
    {"input_ids": np.random.randint(0, 1000, (128,))},
]
dataset = CalibrationDataset(data, max_samples=100)

# From text
texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
dataset = CalibrationDataset.from_texts(texts, max_length=128)

# Iterate through dataset
for sample in dataset:
    process(sample["input_ids"])

# Get batch
batch = dataset.get_batch(batch_size=4, indices=[0, 1, 2, 3])
```

#### Methods

**`__init__(data: List[Dict], max_samples: int = None, shuffle: bool = False)`**
- Initialize calibration dataset
- Parameters:
  - `data`: List of samples
  - `max_samples`: Maximum number of samples
  - `shuffle`: Whether to shuffle data

**`from_texts(texts: List[str], max_length: int = 512) -> CalibrationDataset`**
- Create dataset from text strings
- Parameters:
  - `texts`: List of text strings
  - `max_length`: Maximum sequence length
- Returns: CalibrationDataset instance

### HessianCalibrator

Calibrator for GPTQ quantization.

```python
from vqllm.calibration import HessianCalibrator

# Initialize calibrator
config = QuantConfig(quant_type=QuantType.GPTQ, dampening=0.01)
calibrator = HessianCalibrator(config)

# Collect Hessians for model layers
model = load_model()
calib_data = CalibrationDataset(...)

hessians = calibrator.collect_layer_hessians(model, calib_data)

# Use for quantization
for layer_name, hessian in hessians.items():
    quantized = quantizer.quantize_weight(
        model.get_layer(layer_name).weight,
        layer_name,
        hessian=hessian
    )
```

#### Methods

**`collect_layer_hessians(model: Any, data: List) -> Dict[str, np.ndarray]`**
- Collect Hessian matrices for all layers
- Parameters:
  - `model`: Model to calibrate
  - `data`: Calibration data
- Returns: Dictionary of layer names to Hessian matrices

**`update_hessian(H: np.ndarray, inp: np.ndarray) -> np.ndarray`**
- Update Hessian matrix incrementally
- Parameters:
  - `H`: Current Hessian
  - `inp`: New input batch
- Returns: Updated Hessian

### ActivationCalibrator

Calibrator for AWQ quantization.

```python
from vqllm.calibration import ActivationCalibrator

# Initialize calibrator
config = QuantConfig(quant_type=QuantType.AWQ, group_size=128)
calibrator = ActivationCalibrator(config)

# Collect activation statistics
activations = []
for batch in dataloader:
    act = model.get_activations(batch)
    activations.append(act)

# Compute scales
scales = calibrator.compute_activation_scales(activations)

# Smooth scales for stability
smoothed = calibrator.smooth_scales(scales, alpha=0.5)
```

#### Methods

**`compute_activation_scales(activations: List[np.ndarray]) -> np.ndarray`**
- Compute per-channel activation scales
- Parameters:
  - `activations`: List of activation tensors
- Returns: Per-channel scales

**`smooth_scales(scales: np.ndarray, alpha: float = 0.5) -> np.ndarray`**
- Apply smoothing to scales
- Parameters:
  - `scales`: Raw scales
  - `alpha`: Smoothing factor
- Returns: Smoothed scales

## Inference API

### InferenceEngine

Main engine for running quantized model inference.

```python
from vqllm.inference import InferenceEngine

# Initialize engine
config = QuantConfig(quant_type=QuantType.INT8)
engine = InferenceEngine(
    config,
    device="cuda",
    batch_size=8,
    use_kv_cache=True
)

# Load quantized model
engine.load_model(quantized_model)

# Single forward pass
input_ids = np.array([[1, 2, 3, 4, 5]])
output = engine.forward(input_ids)

# Streaming generation
for token in engine.stream_generate(input_ids, max_length=100):
    print(token)

# Batch inference
batch_input = np.random.randint(0, 1000, (4, 128))
batch_output = engine.forward(batch_input)
```

#### Methods

**`__init__(config: QuantConfig, device: str = "cpu", batch_size: int = 1)`**
- Initialize inference engine
- Parameters:
  - `config`: Quantization configuration
  - `device`: Device to run on
  - `batch_size`: Default batch size

**`load_model(model: Any)`**
- Load a quantized model
- Parameters:
  - `model`: Quantized model to load

**`forward(input_ids: np.ndarray, attention_mask: np.ndarray = None) -> np.ndarray`**
- Run forward pass
- Parameters:
  - `input_ids`: Input token IDs
  - `attention_mask`: Attention mask (optional)
- Returns: Model output

**`stream_generate(input_ids: np.ndarray, max_length: int, **kwargs) -> Iterator`**
- Stream token generation
- Parameters:
  - `input_ids`: Initial input tokens
  - `max_length`: Maximum generation length
- Yields: Generated tokens

### BatchedInference

Optimized batched inference with dynamic batching.

```python
from vqllm.inference import BatchedInference

# Initialize with configuration
batch_engine = BatchedInference(
    config=config,
    max_batch_size=16,
    max_seq_length=2048,
    continuous_batching=True
)

# Add requests
batch_engine.add_request("req_1", input_ids_1, priority=10)
batch_engine.add_request("req_2", input_ids_2, priority=5)
batch_engine.add_request("req_3", input_ids_3, priority=1)

# Process batch (automatic padding and masking)
if batch_engine.should_process_batch():
    batch, mask = batch_engine.create_padded_batch()
    outputs = model(batch, attention_mask=mask)
    batch_engine.return_results(outputs)

# Get results
result = batch_engine.get_result("req_1")
```

#### Methods

**`add_request(request_id: str, input_ids: np.ndarray, priority: int = 5)`**
- Add inference request to queue
- Parameters:
  - `request_id`: Unique request identifier
  - `input_ids`: Input tokens
  - `priority`: Request priority (higher = more urgent)

**`create_padded_batch() -> Tuple[np.ndarray, np.ndarray]`**
- Create padded batch with attention masks
- Returns: (padded_inputs, attention_mask)

**`should_process_batch() -> bool`**
- Check if batch should be processed
- Returns: True if batch is ready

### KVCache

Key-Value cache management for efficient autoregressive generation.

```python
from vqllm.inference import KVCache

# Initialize cache
cache = KVCache(
    num_layers=12,
    max_batch_size=8,
    max_seq_length=2048,
    hidden_size=768,
    num_heads=12
)

# Update cache during generation
for layer_idx in range(12):
    # Get current keys/values
    keys = compute_keys(hidden_states, layer_idx)
    values = compute_values(hidden_states, layer_idx)

    # Update cache
    cache.update(layer_idx, keys, values, seq_position=current_pos)

    # Retrieve full cache for attention
    all_keys = cache.get_keys(layer_idx, batch_size, seq_len)
    all_values = cache.get_values(layer_idx, batch_size, seq_len)

# Clear cache for new sequence
cache.clear()
```

#### Methods

**`update(layer_idx: int, keys: np.ndarray, values: np.ndarray, seq_position: int)`**
- Update cache with new keys/values
- Parameters:
  - `layer_idx`: Layer index
  - `keys`: New key tensors
  - `values`: New value tensors
  - `seq_position`: Sequence position

**`get_keys(layer_idx: int, batch_size: int, seq_len: int) -> np.ndarray`**
- Retrieve cached keys
- Parameters:
  - `layer_idx`: Layer index
  - `batch_size`: Current batch size
  - `seq_len`: Current sequence length
- Returns: Cached keys

## Core Types

### QuantConfig

Configuration for quantization.

```python
from vqllm.core.types import QuantConfig, QuantType, ScaleType

config = QuantConfig(
    quant_type=QuantType.GPTQ,      # Quantization method
    bits=4,                          # Number of bits
    group_size=128,                  # Group size for group-wise quantization
    block_size=128,                  # Block size for block-wise processing
    scale_type=ScaleType.PER_GROUP,  # Scale granularity
    symmetric=False,                 # Symmetric vs asymmetric
    zero_point=True,                 # Use zero point
    dampening=0.01,                  # Dampening factor (GPTQ)
    percdamp=0.01,                   # Percentage dampening
    calibration_samples=256          # Number of calibration samples
)

# Validate configuration
config.validate()
```

### QuantizedTensor

Tensor with quantization metadata.

```python
from vqllm.core.types import QuantizedTensor

# Create quantized tensor
qtensor = QuantizedTensor(
    data=quantized_values,           # Quantized data (INT8/INT4)
    scale=scale_factors,             # Scale factors
    zero_point=zero_points,          # Zero points
    shape=original_shape,            # Original tensor shape
    config=quant_config              # Quantization config
)

# Dequantize
dequantized = qtensor.dequantize()

# Get memory usage
memory_bytes = qtensor.memory_usage()

# Save/load
qtensor.save("quantized.npz")
loaded = QuantizedTensor.load("quantized.npz")

# Properties
print(f"Dtype: {qtensor.dtype}")
print(f"Shape: {qtensor.shape}")
print(f"Compressed: {qtensor.is_quantized}")
```

### QuantType

Enumeration of quantization types.

```python
from vqllm.core.types import QuantType

# Available types
QuantType.INT8    # 8-bit integer
QuantType.INT4    # 4-bit integer
QuantType.GPTQ    # GPTQ method
QuantType.AWQ     # AWQ method
QuantType.FP16    # Half precision (no quantization)
QuantType.DYNAMIC # Dynamic quantization
```

### ScaleType

Enumeration of scale granularities.

```python
from vqllm.core.types import ScaleType

# Available types
ScaleType.PER_TENSOR   # Single scale for entire tensor
ScaleType.PER_CHANNEL  # Scale per output channel
ScaleType.PER_GROUP    # Scale per group of elements
ScaleType.PER_BLOCK    # Scale per block
```

## Utilities

### Benchmarking

```python
from vqllm.utils import benchmark_latency, benchmark_throughput

# Benchmark latency
latency_stats = benchmark_latency(
    model,
    input_data,
    num_runs=100,
    warmup=10
)
print(f"Mean latency: {latency_stats['mean']:.2f}ms")
print(f"P95 latency: {latency_stats['p95']:.2f}ms")

# Benchmark throughput
throughput = benchmark_throughput(
    engine,
    dataset,
    batch_sizes=[1, 2, 4, 8, 16]
)
```

### Profiling

```python
from vqllm.utils import profile_model, memory_profile

# Profile model execution
with profile_model() as prof:
    output = model(input)
prof.print_stats()

# Memory profiling
mem_stats = memory_profile(
    lambda: quantizer.quantize_weight(weight)
)
print(f"Peak memory: {mem_stats['peak_mb']:.2f} MB")
```

### Optimization

```python
from vqllm.utils import optimize_graph, fuse_operations

# Optimize computation graph
optimized = optimize_graph(model.graph)

# Fuse operations
fused = fuse_operations(
    model,
    patterns=[
        ("quantize", "matmul", "dequantize"),
        ("add", "layernorm")
    ]
)
```

## Examples

### Complete Quantization Pipeline

```python
import numpy as np
from vqllm import (
    GPTQQuantizer,
    HessianCalibrator,
    CalibrationDataset,
    InferenceEngine,
    QuantConfig,
    QuantType
)

# 1. Load model
model = load_pretrained_model("model_name")

# 2. Prepare calibration data
texts = load_calibration_texts("calibration_data.txt")
dataset = CalibrationDataset.from_texts(texts, max_length=512)

# 3. Configure quantization
config = QuantConfig(
    quant_type=QuantType.GPTQ,
    bits=4,
    block_size=128,
    dampening=0.01
)

# 4. Calibrate
calibrator = HessianCalibrator(config)
hessians = calibrator.collect_layer_hessians(model, dataset)

# 5. Quantize
quantizer = GPTQQuantizer(config)
quantized_model = quantizer.quantize_model(model, hessians)

# 6. Save quantized model
quantized_model.save("model_quantized.npz")

# 7. Load for inference
engine = InferenceEngine(config)
engine.load_model(quantized_model)

# 8. Run inference
input_ids = tokenize("Hello, world!")
output = engine.forward(input_ids)
print(decode(output))
```

### Mixed Precision Quantization

```python
from vqllm import INT8Quantizer, INT4Quantizer, QuantConfig

# Different configs for different layers
attention_config = QuantConfig(quant_type=QuantType.INT8)
mlp_config = QuantConfig(quant_type=QuantType.INT4)

# Apply different quantization
for name, layer in model.named_modules():
    if "attention" in name:
        quantizer = INT8Quantizer(attention_config)
    elif "mlp" in name:
        quantizer = INT4Quantizer(mlp_config)
    else:
        continue  # Skip other layers

    layer.weight = quantizer.quantize_weight(layer.weight, name)
```

### Custom Calibration

```python
class CustomCalibrator(Calibrator):
    def collect_statistics(self, model, data):
        stats = {}

        for batch in data:
            # Custom statistics collection
            activations = model.get_activations(batch)

            # Compute custom metrics
            stats["custom_metric"] = compute_custom_metric(activations)

        return stats

# Use custom calibrator
calibrator = CustomCalibrator(config)
stats = calibrator.collect_statistics(model, dataset)
```

## Error Handling

All API methods may raise the following exceptions:

- `ValueError`: Invalid configuration or parameters
- `RuntimeError`: Runtime errors during quantization/inference
- `MemoryError`: Insufficient memory for operation
- `NotImplementedError`: Feature not yet implemented

Example error handling:

```python
try:
    qtensor = quantizer.quantize_weight(weight)
except ValueError as e:
    print(f"Configuration error: {e}")
except MemoryError as e:
    print(f"Out of memory: {e}")
    # Try with smaller batch size
except Exception as e:
    print(f"Unexpected error: {e}")
```