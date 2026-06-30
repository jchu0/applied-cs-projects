# On-Device LLM Runtime

A lightweight LLM inference runtime optimized for on-device deployment using memory-mapped GGUF model loading, Q4/Q8 dequantization, SIMD-accelerated kernels, a CPU-optimized KV cache with sliding-window support, and a pluggable compute backend (CPU, CUDA, Metal/MPS, NPU).

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §04 `04-llm-inference` (quantization, KV cache, serving at the edge); pairs with [P41 vector-quantized LLM](../41-vector-quantized-llm/) for the model-prep side and [P44 autoregressive inference engine](../44-autoregressive-inference/) for the server-side contrast.

## What's real vs simulated

- **Real:** GGUF header parsing, mmap-based tensor access, Q4_0/Q8_0 dequantization, CPU transformer forward pass (RMSNorm, RoPE, GQA attention, SwiGLU FFN), KV cache with sliding window, top-k/top-p sampler, speculative decoding scaffold.
- **Simulated:** The NPU path (`npu.py`) ships a `SimulatedNPUBackend` that runs NumPy under a simulated latency delay. The NNAPI, CoreML, and Hexagon backends delegate to `onnxruntime` at the interface level but no real ONNX model conversion or hardware execution is wired end-to-end. Several dequantization variants (Q4_K, Q5_x, Q6_K) raise `NotImplementedError`.

## Layout

```
src/on_device_llm/
    loader.py        — GGUF parser, mmap tensor access, dequantization (Q4_0, Q8_0)
    quantization.py  — GGMLType enum, block-size constants, dequantization helpers
    operators.py     — SIMD kernels: matmul, RMSNorm, softmax, RoPE (Numba optional)
    inference.py     — TransformerInference forward pass, KV cache, Sampler
    memory.py        — Memory-budget tracking and allocation helpers
    backend.py       — ComputeBackend abstraction: CPU, CUDA (CuPy), Metal, NPU
    npu.py           — NPU manager: NNAPI, CoreML, Hexagon, SimulatedNPU backends
    speculative.py   — Speculative decoding (draft-model + verification scaffold)

tests/
    test_operators.py     (51 tests)
    test_backends.py      (31 tests)
    test_inference.py     (45 tests)
    test_npu.py           (46 tests)
    test_quantization.py  (39 tests)
    test_speculative.py   (19 tests)

BLUEPRINT.md     — full architecture design and API reference
PROGRESS.md      — implementation status notes
```

## Build & Run

```bash
conda activate dev
cd 06-real-world-projects/47-on-device-llm
pip install -e ".[dev]"
pytest tests/ -v
```

Optional extras: `pip install -e ".[simd]"` for Numba SIMD kernels; `pip install -e ".[cuda]"` for CuPy GPU backend.
