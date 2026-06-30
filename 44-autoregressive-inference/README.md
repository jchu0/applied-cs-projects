# Autoregressive Inference Engine

A from-scratch LLM inference engine implementing continuous batching, paged KV cache management, and speculative decoding — the same mechanisms that underpin systems like vLLM and TGI.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §04 LLM inference (vLLM/continuous batching, paged KV cache, TGI, serving at scale) · §01 LLM fundamentals (tokenization, attention, decode loop) · §06 CUDA optimization (KV cache memory layout)

## What's real vs simulated

The scheduling logic, KV cache allocator, batching engine, sampling strategies (top-k, top-p, repetition penalty), and speculative decoding driver are fully implemented in pure Python/NumPy. The `InferenceScheduler` falls back to a **mock decode path** (hardcoded token ID 100) when no model object is supplied — this is the path used by the test suite so tests run without a GPU or real weights. The `DraftModel` / `TargetModel` types are Protocols (interfaces): callers must supply a real or fake model that satisfies the interface. No actual transformer weights are included.

## Layout

```
src/autoregressive_inference/
    requests.py       # InferenceRequest, RequestManager, priority queue
    kv_cache.py       # PagedKVCacheManager, KVCacheBlock, SlidingWindowCache
    batching.py       # ContinuousBatcher, BatchedInputs
    sampling.py       # TokenSampler (top-k, top-p, penalties)
    scheduler.py      # InferenceScheduler (prefill/decode loop, preemption)
    speculative.py    # SpeculativeDecoder, TreeSpeculativeDecoder, DraftModel/TargetModel protocols

tests/
    test_requests.py
    test_kv_cache.py
    test_batching.py
    test_sampling.py
    test_scheduler.py
    test_speculative.py   # 231 tests total

BLUEPRINT.md          # Full architecture, design decisions, performance targets
```

## Build & run

```bash
conda activate dev
cd 06-real-world-projects/44-autoregressive-inference
pip install -e ".[dev]"
pytest tests/ -v
```

To include PyTorch-dependent paths (requires a CUDA-capable environment):

```bash
pip install -e ".[full]"
pytest tests/ -v -m "not gpu"   # skip GPU-only tests
```
