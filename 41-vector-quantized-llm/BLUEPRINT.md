# Fully Vector-Quantized LLM Pipeline - Technical Blueprint

## Executive Summary

A comprehensive quantization pipeline for Large Language Models supporting INT4, FP8, and GGUF formats, designed to minimize model size while preserving accuracy through sophisticated calibration, error-minimizing scale computation, and optimized dequantization kernels. This system enables efficient deployment of LLMs on resource-constrained hardware while maintaining competitive perplexity and zero-shot performance.

> **Concepts covered:** [§04 Quantization (INT8/INT4/FP8/GPTQ/AWQ/SmoothQuant)](../../04-ai-engineering/04-llm-inference/quantization/quantization.md) — this project *implements* the full quantizer family the tutorial describes · [§04 vLLM](../../04-ai-engineering/04-llm-inference/vllm/vllm.md). Pairs with [Project 47 (on-device LLM)](../47-on-device-llm/) for the quantized-model runtime and [Project 45 (neural compression)](../45-neural-compression/) for the broader compression toolkit. Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## System Architecture

### High-Level Architecture

```
+----------------------------------------------------------+
|                    Model Input (FP16/FP32)                 |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                   Calibration Pipeline                    |
|  +-------------+  +-------------+  +------------------+  |
|  | Activation  |  | Weight      |  | Cross-Layer      |  |
|  | Profiling   |  | Analysis    |  | Correlation      |  |
|  +-------------+  +-------------+  +------------------+  |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                   Scale/Zero-Point Optimizer              |
|  +-------------+  +-------------+  +------------------+  |
|  | MSE         |  | Cross       |  | Hessian-Aware    |  |
|  | Minimizer   |  | Entropy     |  | Optimization     |  |
|  +-------------+  +-------------+  +------------------+  |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    Quantization Engine                    |
|  +-------------+  +-------------+  +------------------+  |
|  | INT4        |  | FP8         |  | GGUF             |  |
|  | (Group)     |  | (E4M3/E5M2) |  | (K-Quant)        |  |
|  +-------------+  +-------------+  +------------------+  |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                   Quantized Model Output                  |
|  (weights, scales, zero-points, metadata)                |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    Runtime Engine                         |
|  +-------------+  +-------------+  +------------------+  |
|  | Dequant     |  | INT4        |  | KV Cache         |  |
|  | Kernels     |  | Matmul      |  | Quantization     |  |
|  +-------------+  +-------------+  +------------------+  |
+----------------------------------------------------------+
```

### Core Design Principles

1. **Accuracy Preservation**: Minimize quantization error through optimal calibration
2. **Hardware Awareness**: Design for efficient execution on target devices
3. **Format Flexibility**: Support multiple quantization formats
4. **Mixed Precision**: Apply different precisions to different layers
5. **Evaluation-Driven**: Continuous perplexity and accuracy monitoring

## Component Design

### 1. Calibration Pipeline

```python
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class CalibrationStats:
    """Statistics collected during calibration"""
    min_val: float
    max_val: float
    mean: float
    std: float
    histogram: np.ndarray
    bin_edges: np.ndarray
    num_samples: int

@dataclass
class LayerCalibration:
    """Calibration data for a single layer"""
    name: str
    weight_stats: CalibrationStats
    input_stats: CalibrationStats
    output_stats: CalibrationStats
    hessian_diag: Optional[torch.Tensor] = None

class CalibrationCollector:
    """
    Collects activation and weight statistics for quantization calibration.
    Uses representative data to determine optimal quantization parameters.
    """

    def __init__(self, model: nn.Module, num_bins: int = 2048):
        self.model = model
        self.num_bins = num_bins
        self.layer_stats: Dict[str, LayerCalibration] = {}
        self.hooks = []

    def register_hooks(self, layers: Optional[List[str]] = None):
        """Register forward hooks for calibration"""
        for name, module in self.model.named_modules():
            if layers and name not in layers:
                continue

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Register hooks for input/output capture
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

                # Initialize stats
                self.layer_stats[name] = LayerCalibration(
                    name=name,
                    weight_stats=self._compute_weight_stats(module.weight),
                    input_stats=None,
                    output_stats=None
                )

    def _make_hook(self, name: str):
        """Create forward hook for a layer"""
        def hook(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input

            # Update input statistics
            if self.layer_stats[name].input_stats is None:
                self.layer_stats[name].input_stats = self._compute_activation_stats(inp)
            else:
                self._update_stats(self.layer_stats[name].input_stats, inp)

            # Update output statistics
            if self.layer_stats[name].output_stats is None:
                self.layer_stats[name].output_stats = self._compute_activation_stats(output)
            else:
                self._update_stats(self.layer_stats[name].output_stats, output)

        return hook

    def _compute_weight_stats(self, weight: torch.Tensor) -> CalibrationStats:
        """Compute statistics for weight tensor"""
        w = weight.detach().float().cpu()

        min_val = w.min().item()
        max_val = w.max().item()

        histogram, bin_edges = np.histogram(
            w.numpy().flatten(),
            bins=self.num_bins,
            range=(min_val, max_val)
        )

        return CalibrationStats(
            min_val=min_val,
            max_val=max_val,
            mean=w.mean().item(),
            std=w.std().item(),
            histogram=histogram,
            bin_edges=bin_edges,
            num_samples=w.numel()
        )

    def _compute_activation_stats(self, activation: torch.Tensor) -> CalibrationStats:
        """Compute statistics for activation tensor"""
        act = activation.detach().float().cpu()

        min_val = act.min().item()
        max_val = act.max().item()

        histogram, bin_edges = np.histogram(
            act.numpy().flatten(),
            bins=self.num_bins,
            range=(min_val, max_val)
        )

        return CalibrationStats(
            min_val=min_val,
            max_val=max_val,
            mean=act.mean().item(),
            std=act.std().item(),
            histogram=histogram,
            bin_edges=bin_edges,
            num_samples=act.numel()
        )

    def _update_stats(self, stats: CalibrationStats, tensor: torch.Tensor):
        """Update running statistics with new tensor"""
        t = tensor.detach().float().cpu()

        # Update min/max
        new_min = t.min().item()
        new_max = t.max().item()
        stats.min_val = min(stats.min_val, new_min)
        stats.max_val = max(stats.max_val, new_max)

        # Update histogram (rebin if needed)
        new_hist, _ = np.histogram(
            t.numpy().flatten(),
            bins=stats.bin_edges
        )
        stats.histogram += new_hist
        stats.num_samples += t.numel()

    @torch.no_grad()
    def calibrate(self, dataloader, num_batches: int = 128):
        """Run calibration on representative data"""
        self.model.eval()

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Move to device
            if isinstance(batch, dict):
                batch = {k: v.to(next(self.model.parameters()).device)
                        for k, v in batch.items() if isinstance(v, torch.Tensor)}
                self.model(**batch)
            else:
                input_ids = batch[0].to(next(self.model.parameters()).device)
                self.model(input_ids)

        # Compute Hessian diagonal for GPTQ-style methods
        self._compute_hessian_diagonal()

    def _compute_hessian_diagonal(self):
        """Compute diagonal of Hessian for weight importance"""
        for name, stats in self.layer_stats.items():
            # Approximate Hessian diagonal as E[x^2] for input x
            if stats.input_stats is not None:
                # Use histogram to approximate E[x^2]
                centers = (stats.input_stats.bin_edges[:-1] +
                          stats.input_stats.bin_edges[1:]) / 2
                probs = stats.input_stats.histogram / stats.input_stats.num_samples
                e_x2 = np.sum(centers ** 2 * probs)

                # This is a simplified approximation
                stats.hessian_diag = torch.full((1,), e_x2)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
```

### 2. Scale and Zero-Point Optimization

```python
class ScaleOptimizer:
    """
    Optimizes quantization scales and zero-points to minimize error.
    Supports multiple optimization objectives.
    """

    def __init__(self, method: str = 'mse'):
        self.method = method
        self.optimizers = {
            'mse': self._optimize_mse,
            'minmax': self._optimize_minmax,
            'percentile': self._optimize_percentile,
            'entropy': self._optimize_entropy,
            'gptq': self._optimize_gptq,
        }

    def optimize(self, tensor: torch.Tensor, stats: CalibrationStats,
                 bits: int, symmetric: bool = False,
                 group_size: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute optimal scale and zero-point for tensor quantization.

        Returns:
            scales: Quantization scales
            zero_points: Zero points (0 for symmetric)
        """
        if group_size > 0:
            return self._optimize_grouped(tensor, stats, bits, symmetric, group_size)

        return self.optimizers[self.method](tensor, stats, bits, symmetric)

    def _optimize_mse(self, tensor: torch.Tensor, stats: CalibrationStats,
                     bits: int, symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find scale that minimizes mean squared error"""
        qmin, qmax = self._get_quant_range(bits, symmetric)

        # Grid search for optimal scale
        best_scale = None
        best_mse = float('inf')

        # Try different percentiles
        for p in [99.0, 99.5, 99.9, 99.99, 100.0]:
            if symmetric:
                abs_max = np.percentile(
                    np.abs(tensor.cpu().numpy().flatten()), p
                )
                scale = abs_max / qmax
            else:
                min_val = np.percentile(tensor.cpu().numpy().flatten(), 100 - p)
                max_val = np.percentile(tensor.cpu().numpy().flatten(), p)
                scale = (max_val - min_val) / (qmax - qmin)
                zero_point = qmin - min_val / scale

            # Compute MSE
            if symmetric:
                zero_point = 0
            quantized = torch.clamp(
                torch.round(tensor / scale) + zero_point,
                qmin, qmax
            )
            dequantized = (quantized - zero_point) * scale
            mse = ((tensor - dequantized) ** 2).mean().item()

            if mse < best_mse:
                best_mse = mse
                best_scale = scale
                best_zp = zero_point if not symmetric else 0

        return torch.tensor([best_scale]), torch.tensor([best_zp])

    def _optimize_entropy(self, tensor: torch.Tensor, stats: CalibrationStats,
                         bits: int, symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find scale that minimizes KL divergence between distributions"""
        qmin, qmax = self._get_quant_range(bits, symmetric)
        num_quantized_bins = qmax - qmin + 1

        # Reference histogram
        ref_hist = stats.histogram.astype(np.float64)
        ref_hist = ref_hist / ref_hist.sum()

        best_scale = None
        best_kl = float('inf')

        # Search over different thresholds
        for threshold_bin in range(128, len(stats.histogram)):
            # Compute scale for this threshold
            threshold = stats.bin_edges[threshold_bin]
            if symmetric:
                scale = threshold / qmax
            else:
                scale = threshold / (qmax - qmin)

            # Simulate quantization
            q_hist = self._quantize_histogram(
                ref_hist[:threshold_bin],
                num_quantized_bins
            )

            # Compute KL divergence
            kl = self._kl_divergence(ref_hist[:threshold_bin], q_hist)

            if kl < best_kl:
                best_kl = kl
                best_scale = scale

        zero_point = 0 if symmetric else qmin
        return torch.tensor([best_scale]), torch.tensor([zero_point])

    def _optimize_gptq(self, tensor: torch.Tensor, stats: CalibrationStats,
                      bits: int, symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPTQ-style optimization using Hessian information.
        Minimizes ||W - Q||_H where H is the Hessian.
        """
        qmin, qmax = self._get_quant_range(bits, symmetric)

        # Use Hessian diagonal for weighted error
        H_diag = stats.hessian_diag if stats.hessian_diag is not None else torch.ones(1)

        # Iterative optimization
        W = tensor.clone()
        scale = self._get_initial_scale(W, qmin, qmax, symmetric)

        for iteration in range(10):
            # Quantize with current scale
            if symmetric:
                Q = torch.clamp(torch.round(W / scale), qmin, qmax)
            else:
                zero_point = qmin - W.min() / scale
                Q = torch.clamp(torch.round(W / scale + zero_point), qmin, qmax)

            # Compute weighted error
            error = (W - Q * scale) ** 2 * H_diag
            weighted_mse = error.mean()

            # Adjust scale
            gradient = self._compute_scale_gradient(W, Q, scale, H_diag)
            scale = scale - 0.01 * gradient

        zero_point = 0 if symmetric else (qmin - W.min() / scale)
        return torch.tensor([scale.item()]), torch.tensor([zero_point.item()])

    def _optimize_grouped(self, tensor: torch.Tensor, stats: CalibrationStats,
                         bits: int, symmetric: bool,
                         group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-group scales for grouped quantization"""
        # Reshape to groups
        orig_shape = tensor.shape
        if tensor.ndim == 2:
            # [out, in] -> [out, num_groups, group_size]
            num_groups = tensor.shape[1] // group_size
            grouped = tensor.view(tensor.shape[0], num_groups, group_size)
        else:
            grouped = tensor.view(-1, group_size)

        # Compute scale for each group
        scales = []
        zero_points = []

        for group in grouped.view(-1, group_size):
            group_stats = CalibrationStats(
                min_val=group.min().item(),
                max_val=group.max().item(),
                mean=group.mean().item(),
                std=group.std().item(),
                histogram=np.array([]),
                bin_edges=np.array([]),
                num_samples=group.numel()
            )

            s, zp = self._optimize_minmax(group, group_stats, bits, symmetric)
            scales.append(s)
            zero_points.append(zp)

        return torch.stack(scales), torch.stack(zero_points)

    @staticmethod
    def _get_quant_range(bits: int, symmetric: bool) -> Tuple[int, int]:
        """Get quantization range for given bit width"""
        if symmetric:
            qmax = 2 ** (bits - 1) - 1
            qmin = -qmax
        else:
            qmax = 2 ** bits - 1
            qmin = 0
        return qmin, qmax
```

### 3. Quantization Engine

```python
class QuantizationEngine:
    """
    Core quantization engine supporting multiple formats.
    """

    def __init__(self, config: 'QuantConfig'):
        self.config = config
        self.scale_optimizer = ScaleOptimizer(method=config.calibration_method)

    def quantize_model(self, model: nn.Module,
                      calibration_data: Dict[str, LayerCalibration]) -> nn.Module:
        """Quantize all layers in a model"""
        quantized_model = copy.deepcopy(model)

        for name, module in quantized_model.named_modules():
            if name not in calibration_data:
                continue

            if isinstance(module, nn.Linear):
                quant_layer = self._quantize_linear(
                    module, calibration_data[name]
                )
                # Replace in model
                parent = self._get_parent(quantized_model, name)
                setattr(parent, name.split('.')[-1], quant_layer)

        return quantized_model

    def _quantize_linear(self, layer: nn.Linear,
                        calib: LayerCalibration) -> 'QuantizedLinear':
        """Quantize a linear layer"""
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None

        # Compute scales
        scales, zero_points = self.scale_optimizer.optimize(
            weight, calib.weight_stats,
            bits=self.config.weight_bits,
            symmetric=self.config.symmetric,
            group_size=self.config.group_size
        )

        # Quantize weights
        qweight = self._quantize_tensor(
            weight, scales, zero_points,
            bits=self.config.weight_bits,
            format=self.config.format
        )

        return QuantizedLinear(
            qweight=qweight,
            scales=scales,
            zero_points=zero_points,
            bias=bias,
            in_features=layer.in_features,
            out_features=layer.out_features,
            group_size=self.config.group_size,
            bits=self.config.weight_bits,
            format=self.config.format
        )

    def _quantize_tensor(self, tensor: torch.Tensor,
                        scales: torch.Tensor, zero_points: torch.Tensor,
                        bits: int, format: str) -> torch.Tensor:
        """Quantize tensor to specified format"""
        if format == 'int4':
            return self._quantize_int4(tensor, scales, zero_points)
        elif format == 'fp8_e4m3':
            return self._quantize_fp8_e4m3(tensor, scales)
        elif format == 'fp8_e5m2':
            return self._quantize_fp8_e5m2(tensor, scales)
        elif format == 'gguf':
            return self._quantize_gguf(tensor, scales, zero_points, bits)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _quantize_int4(self, tensor: torch.Tensor,
                      scales: torch.Tensor, zero_points: torch.Tensor) -> torch.Tensor:
        """Quantize to INT4 with packing"""
        # Scale and round
        quantized = torch.round(tensor / scales.view(-1, 1) + zero_points.view(-1, 1))
        quantized = torch.clamp(quantized, 0, 15).to(torch.uint8)

        # Pack two INT4 values into one byte
        packed = self._pack_int4(quantized)
        return packed

    def _pack_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pack INT4 values into bytes"""
        # Ensure even number of elements
        if tensor.numel() % 2 != 0:
            tensor = torch.nn.functional.pad(tensor.view(-1), (0, 1))

        flat = tensor.view(-1)
        low = flat[0::2] & 0x0F
        high = flat[1::2] << 4
        packed = (high | low).to(torch.uint8)

        return packed

    def _quantize_fp8_e4m3(self, tensor: torch.Tensor,
                          scales: torch.Tensor) -> torch.Tensor:
        """Quantize to FP8 E4M3 format"""
        # E4M3: 4 exponent bits, 3 mantissa bits
        # Range: ~[-448, 448], smallest: 2^-9

        scaled = tensor / scales.view(-1, 1)

        # Clamp to FP8 range
        max_val = 448.0
        clamped = torch.clamp(scaled, -max_val, max_val)

        # Convert to FP8 representation
        # This is simplified - real implementation needs proper bit manipulation
        return self._float_to_fp8_e4m3(clamped)

    def _quantize_gguf(self, tensor: torch.Tensor,
                      scales: torch.Tensor, zero_points: torch.Tensor,
                      bits: int) -> torch.Tensor:
        """Quantize using GGUF K-quant format"""
        # GGUF uses block-based quantization with super-blocks
        block_size = 32
        superblock_size = 256

        # Reshape into blocks
        num_blocks = tensor.numel() // block_size
        blocks = tensor.view(num_blocks, block_size)

        # Quantize each block
        quantized_blocks = []
        block_scales = []
        block_mins = []

        for block in blocks:
            # Compute block-local scale and min
            block_min = block.min()
            block_max = block.max()
            block_scale = (block_max - block_min) / (2**bits - 1)

            # Quantize
            q = torch.round((block - block_min) / block_scale)
            q = torch.clamp(q, 0, 2**bits - 1).to(torch.uint8)

            quantized_blocks.append(q)
            block_scales.append(block_scale)
            block_mins.append(block_min)

        return {
            'quantized': torch.stack(quantized_blocks),
            'scales': torch.tensor(block_scales),
            'mins': torch.tensor(block_mins),
            'format': f'q{bits}_k'
        }


@dataclass
class QuantConfig:
    """Configuration for quantization"""
    weight_bits: int = 4
    activation_bits: int = 8
    format: str = 'int4'  # int4, fp8_e4m3, fp8_e5m2, gguf
    symmetric: bool = True
    group_size: int = 128
    calibration_method: str = 'mse'  # mse, minmax, entropy, gptq
    mixed_precision: bool = False
    sensitive_layers: List[str] = field(default_factory=list)
```

### 4. Quantized Layers and Kernels

```python
class QuantizedLinear(nn.Module):
    """
    Quantized linear layer with efficient inference.
    """

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor,
                 zero_points: torch.Tensor, bias: Optional[torch.Tensor],
                 in_features: int, out_features: int,
                 group_size: int, bits: int, format: str):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = bits
        self.format = format

        # Register quantized parameters
        self.register_buffer('qweight', qweight)
        self.register_buffer('scales', scales)
        self.register_buffer('zero_points', zero_points)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization"""
        # Dequantize weights
        weight = self._dequantize()

        # Matrix multiplication
        output = torch.nn.functional.linear(x, weight, self.bias)

        return output

    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights to FP16/FP32"""
        if self.format == 'int4':
            return self._dequantize_int4()
        elif self.format.startswith('fp8'):
            return self._dequantize_fp8()
        elif self.format == 'gguf':
            return self._dequantize_gguf()

    def _dequantize_int4(self) -> torch.Tensor:
        """Dequantize INT4 weights"""
        # Unpack
        unpacked = self._unpack_int4(self.qweight)

        # Apply scale and zero-point
        if self.group_size > 0:
            # Per-group dequantization
            num_groups = self.in_features // self.group_size
            unpacked = unpacked.view(self.out_features, num_groups, self.group_size)
            scales = self.scales.view(self.out_features, num_groups, 1)
            zps = self.zero_points.view(self.out_features, num_groups, 1)

            dequantized = (unpacked.float() - zps) * scales
            return dequantized.view(self.out_features, self.in_features)
        else:
            return (unpacked.float() - self.zero_points) * self.scales

    def _unpack_int4(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack INT4 values from bytes"""
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F

        # Interleave
        unpacked = torch.zeros(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
        unpacked[0::2] = low
        unpacked[1::2] = high

        return unpacked.view(self.out_features, self.in_features)


class Int4MatmulKernel:
    """
    Optimized INT4 matrix multiplication kernel.
    For production, this would be implemented in CUDA.
    """

    @staticmethod
    def matmul(input: torch.Tensor, qweight: torch.Tensor,
               scales: torch.Tensor, zero_points: torch.Tensor,
               group_size: int) -> torch.Tensor:
        """
        INT4 matrix multiplication with on-the-fly dequantization.

        Args:
            input: [batch, in_features] FP16 input
            qweight: [out_features, in_features//2] packed INT4 weights
            scales: [out_features, num_groups] or [out_features]
            zero_points: [out_features, num_groups] or [out_features]
            group_size: Number of elements per group

        Returns:
            output: [batch, out_features] FP16 output
        """
        # This is a reference implementation
        # Production would use CUDA kernels

        batch, in_features = input.shape
        out_features = scales.shape[0]

        # Unpack weights
        unpacked = Int4MatmulKernel._unpack_int4_cuda(qweight)

        # Grouped dequantization during matmul
        if group_size > 0:
            num_groups = in_features // group_size
            output = torch.zeros(batch, out_features, device=input.device, dtype=input.dtype)

            for g in range(num_groups):
                start = g * group_size
                end = start + group_size

                # Dequantize this group
                w_group = unpacked[:, start:end].float()
                s = scales[:, g:g+1] if scales.ndim > 1 else scales.unsqueeze(1)
                zp = zero_points[:, g:g+1] if zero_points.ndim > 1 else zero_points.unsqueeze(1)

                dq_weight = (w_group - zp) * s

                # Partial matmul
                output += input[:, start:end].float() @ dq_weight.t()

            return output.to(input.dtype)
        else:
            # Per-tensor dequantization
            dq_weight = (unpacked.float() - zero_points) * scales
            return input @ dq_weight.t()
```

### 5. KV Cache Quantization

```python
class QuantizedKVCache:
    """
    Quantized KV cache for memory-efficient inference.
    Reduces memory usage by quantizing keys and values.
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 max_seq_len: int, dtype: torch.dtype,
                 kv_bits: int = 8, group_size: int = 64):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.kv_bits = kv_bits
        self.group_size = group_size

        # Quantized cache storage
        # [num_layers, batch, num_heads, max_seq_len, head_dim]
        bytes_per_elem = kv_bits // 8
        self.k_cache = torch.zeros(
            num_layers, 1, num_heads, max_seq_len, head_dim * bytes_per_elem,
            dtype=torch.uint8
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        # Scales per group
        num_groups = head_dim // group_size
        self.k_scales = torch.zeros(num_layers, 1, num_heads, max_seq_len, num_groups)
        self.v_scales = torch.zeros_like(self.k_scales)

        self.seq_len = 0

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """Update cache with new key/value"""
        batch_size, num_heads, seq_len, head_dim = key.shape

        # Quantize
        qk, k_scale = self._quantize(key)
        qv, v_scale = self._quantize(value)

        # Store
        start = self.seq_len
        end = self.seq_len + seq_len

        self.k_cache[layer_idx, :batch_size, :, start:end] = qk
        self.v_cache[layer_idx, :batch_size, :, start:end] = qv
        self.k_scales[layer_idx, :batch_size, :, start:end] = k_scale
        self.v_scales[layer_idx, :batch_size, :, start:end] = v_scale

        self.seq_len = end

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dequantized key/value for attention"""
        # Retrieve and dequantize
        qk = self.k_cache[layer_idx, :, :, :self.seq_len]
        qv = self.v_cache[layer_idx, :, :, :self.seq_len]
        k_scale = self.k_scales[layer_idx, :, :, :self.seq_len]
        v_scale = self.v_scales[layer_idx, :, :, :self.seq_len]

        key = self._dequantize(qk, k_scale)
        value = self._dequantize(qv, v_scale)

        return key, value

    def _quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor for KV cache"""
        batch, heads, seq, dim = tensor.shape

        # Reshape for grouped quantization
        num_groups = dim // self.group_size
        grouped = tensor.view(batch, heads, seq, num_groups, self.group_size)

        # Compute per-group scales
        max_abs = grouped.abs().amax(dim=-1, keepdim=True)
        qmax = 2 ** (self.kv_bits - 1) - 1
        scales = max_abs / qmax

        # Quantize
        quantized = torch.round(grouped / (scales + 1e-8))
        quantized = torch.clamp(quantized, -qmax, qmax)

        # Pack if INT4
        if self.kv_bits == 4:
            quantized = self._pack_int4(quantized + qmax)  # Make unsigned

        return quantized.view(batch, heads, seq, -1), scales.squeeze(-1)

    def _dequantize(self, quantized: torch.Tensor,
                   scales: torch.Tensor) -> torch.Tensor:
        """Dequantize KV cache values"""
        batch, heads, seq, _ = quantized.shape
        dim = self.head_dim

        # Unpack if INT4
        if self.kv_bits == 4:
            qmax = 2 ** (self.kv_bits - 1) - 1
            quantized = self._unpack_int4(quantized) - qmax  # Make signed

        # Reshape for grouped dequantization
        num_groups = dim // self.group_size
        grouped = quantized.view(batch, heads, seq, num_groups, self.group_size)
        scales = scales.unsqueeze(-1)

        # Dequantize
        dequantized = grouped.float() * scales

        return dequantized.view(batch, heads, seq, dim)

    def memory_usage(self) -> dict:
        """Get memory usage statistics"""
        qkv_bytes = self.k_cache.numel() + self.v_cache.numel()
        scale_bytes = (self.k_scales.numel() + self.v_scales.numel()) * 4  # FP32

        return {
            'quantized_kv_mb': qkv_bytes / 1024 / 1024,
            'scales_mb': scale_bytes / 1024 / 1024,
            'total_mb': (qkv_bytes + scale_bytes) / 1024 / 1024,
            'compression_ratio': (2 * self.num_layers * self.num_heads *
                                 self.max_seq_len * self.head_dim * 2) /
                                (qkv_bytes + scale_bytes)  # vs FP16
        }
```

### 6. Evaluation Pipeline

```python
class QuantizationEvaluator:
    """
    Evaluates quantized model quality.
    """

    def __init__(self, tokenizer, device: str = 'cuda'):
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def compute_perplexity(self, model: nn.Module, dataset,
                          max_samples: int = 100) -> float:
        """Compute perplexity on evaluation dataset"""
        model.eval()
        total_loss = 0
        total_tokens = 0

        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            input_ids = sample['input_ids'].to(self.device)
            labels = input_ids.clone()

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity

    @torch.no_grad()
    def compute_accuracy(self, model: nn.Module, dataset,
                        task: str = 'classification') -> float:
        """Compute accuracy on downstream task"""
        model.eval()
        correct = 0
        total = 0

        for sample in dataset:
            input_ids = sample['input_ids'].to(self.device)
            labels = sample['labels'].to(self.device)

            outputs = model(input_ids)
            predictions = outputs.logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.numel()

        return correct / total

    def compare_outputs(self, original: nn.Module, quantized: nn.Module,
                       test_input: torch.Tensor) -> dict:
        """Compare outputs between original and quantized model"""
        original.eval()
        quantized.eval()

        with torch.no_grad():
            orig_output = original(test_input).logits
            quant_output = quantized(test_input).logits

        # Compute metrics
        mse = ((orig_output - quant_output) ** 2).mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            orig_output.view(-1), quant_output.view(-1), dim=0
        ).item()
        max_diff = (orig_output - quant_output).abs().max().item()

        return {
            'mse': mse,
            'cosine_similarity': cosine_sim,
            'max_absolute_diff': max_diff,
            'snr_db': 10 * np.log10(orig_output.pow(2).mean().item() / (mse + 1e-10))
        }

    def benchmark_speed(self, model: nn.Module, input_shape: tuple,
                       num_iterations: int = 100) -> dict:
        """Benchmark inference speed"""
        model.eval()
        test_input = torch.randint(0, 32000, input_shape, device=self.device)

        # Warmup
        for _ in range(10):
            _ = model(test_input)

        # Benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_iterations):
            _ = model(test_input)
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)

        return {
            'avg_latency_ms': elapsed_ms / num_iterations,
            'throughput_tokens_per_sec': input_shape[0] * input_shape[1] * num_iterations / (elapsed_ms / 1000)
        }
```

## Enterprise Features

### Auto-Tuner

```python
class QuantizationAutoTuner:
    """
    Automatically finds best quantization configuration.
    """

    def __init__(self, model: nn.Module, calibration_data,
                 target_perplexity_degradation: float = 0.1):
        self.model = model
        self.calibration_data = calibration_data
        self.target_degradation = target_perplexity_degradation

    def tune(self) -> QuantConfig:
        """Find optimal quantization configuration"""
        # Compute baseline perplexity
        evaluator = QuantizationEvaluator(self.tokenizer)
        baseline_ppl = evaluator.compute_perplexity(self.model, self.calibration_data)

        best_config = None
        best_size = float('inf')

        # Search configurations
        for bits in [4, 8]:
            for group_size in [32, 64, 128]:
                for method in ['mse', 'entropy', 'gptq']:
                    config = QuantConfig(
                        weight_bits=bits,
                        group_size=group_size,
                        calibration_method=method
                    )

                    # Quantize and evaluate
                    engine = QuantizationEngine(config)
                    quantized = engine.quantize_model(self.model, self.calibration_data)

                    ppl = evaluator.compute_perplexity(quantized, self.calibration_data)
                    size = self._compute_model_size(quantized)

                    # Check if within target degradation
                    degradation = (ppl - baseline_ppl) / baseline_ppl
                    if degradation <= self.target_degradation:
                        if size < best_size:
                            best_size = size
                            best_config = config

        return best_config
```

## Development Phases

### Phase 1: Calibration (Weeks 1-3)
- Activation/weight profiling
- Histogram collection
- MinMax/percentile calibration
- Hessian diagonal estimation

### Phase 2: Scale Optimization (Weeks 4-5)
- MSE minimization
- Entropy-based calibration
- GPTQ-style optimization
- Grouped quantization

### Phase 3: Quantization Formats (Weeks 6-8)
- INT4 with packing
- FP8 (E4M3, E5M2)
- GGUF K-quant
- Mixed precision

### Phase 4: Kernels (Weeks 9-10)
- Dequantization kernels
- INT4 matmul
- FP8 matmul
- CUDA optimization

### Phase 5: Evaluation (Week 11)
- Perplexity measurement
- Zero-shot accuracy
- Speed benchmarking
- Memory profiling

### Phase 6: Enterprise (Week 12+)
- Auto-tuner
- KV cache quantization
- QAT support
- Model registry

## Testing Strategy

### Unit Tests
- Scale computation correctness
- Pack/unpack operations
- Dequantization accuracy
- Kernel numerical accuracy

### Integration Tests
- End-to-end quantization
- Model loading/saving
- Inference correctness
- Memory usage

### Accuracy Tests
- Perplexity degradation < 1%
- Zero-shot accuracy drop < 2%
- Output cosine similarity > 0.99

### Performance Tests
- Quantization speed
- Inference latency
- Memory reduction
- Throughput

## Performance Targets

| Metric | Target |
|--------|--------|
| Perplexity degradation | < 1% |
| Model size reduction (INT4) | 4x |
| Inference speedup | > 2x |
| Memory reduction | 3-4x |
| Calibration time (7B model) | < 1 hour |

## Dependencies

- **PyTorch**: Model manipulation
- **transformers**: HuggingFace models
- **bitsandbytes**: Reference implementations
- **CUDA**: GPU kernels

## References

- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- AWQ: Activation-aware Weight Quantization
- SmoothQuant: Accurate and Efficient Post-Training Quantization
- GGML/GGUF format specification
- FP8 Formats for Deep Learning (NVIDIA)
