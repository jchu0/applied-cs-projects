# DynaGraph — Dynamic Graph Execution: Technical Blueprint

> **Concepts covered:** §03 ml-engineering — `02-deep-learning/pytorch`, `02-deep-learning/custom-layers`

> **Status:** reference implementation / teaching scaffold. Authored retroactively (2026-06) to document the as-built system; see [`../PROJECTS_STATUS.md`](../PROJECTS_STATUS.md) for production-readiness caveats and [`PROGRESS.md`](PROGRESS.md) for status.

## 1. Executive Summary

DynaGraph is a from-scratch, PyTorch-style framework for **eager (define-by-run) execution with reverse-mode automatic differentiation**, built on NumPy. The computation graph is constructed implicitly as operations run, so native Python control flow (loops, branches) participates in differentiation without a separate graph-definition step. A secondary path lowers a traced graph to pluggable backends for optimization and export.

The goal is pedagogical fidelity to how modern eager autograd engines work — `Tensor`, `Function`, the tape-free graph of `grad_fn` nodes, and a `GradientTape` — not to compete with PyTorch on performance or coverage.

## 2. Goals / Non-Goals

**Goals**
- Eager `Tensor` with `requires_grad` and operator overloading (`+ - * @`, `sum`, `reshape`, indexing, activations).
- Correct reverse-mode autodiff through dynamic control flow.
- A `Function` base class for defining custom differentiable ops (`forward`/`backward` + `FunctionContext`).
- A graph/executor layer and pluggable backends (native, ONNX) for tracing, optimization passes, and export.

**Non-Goals**
- Higher-order automatic differentiation (double backprop). Gradients are returned as NumPy arrays, not re-traced tensors — so second derivatives are obtained by finite-differencing the first-order gradient, not by differentiating the backward graph (see §5).
- GPU execution; storage is NumPy `float32` on CPU.
- Full ONNX operator coverage.

## 3. Architecture

```
                 ┌───────────────────────────────────────────────┐
   user code ───▶│  core/tensor.py    Tensor (data, grad, grad_fn)│
                 │                    operator overloads, backward │
                 └───────────────┬───────────────────────────────┘
                                 │ each op records a grad_fn node
                 ┌───────────────▼───────────────────────────────┐
                 │  autograd/autograd.py                          │
                 │   Function / FunctionContext (custom ops)      │
                 │   backward(), grad(), GradientTape             │
                 │   jacobian()  — reverse-mode, per output elem  │
                 │   hessian()   — finite diff of the gradient    │
                 │   Add/Mul/MatMul/ReLU/Sigmoid/Tanh/Softmax …   │
                 └───────────────┬───────────────────────────────┘
                                 │ optional: trace → graph IR
        ┌────────────────────────▼───────────────┐   ┌───────────────────────┐
        │ graph/graph.py   computation graph IR   │──▶│ executor/executor.py  │
        │ backend/passes.py  optimization passes  │   │  graph execution      │
        └────────────────────────┬───────────────┘   └───────────────────────┘
                                 │ lower
                 ┌───────────────▼───────────────────────────────┐
                 │ backend/lowering.py · native.py · onnx_backend │
                 └───────────────────────────────────────────────┘
```

## 4. Core Abstractions

- **`Tensor`** (`core/tensor.py`): wraps a NumPy `float32` array, an optional `grad`, and a `_grad_fn` link to the op that produced it. `backward(grad_output)` seeds the output gradient, accumulates into `.grad`, then calls `self._grad_fn.backward(grad_output)`, recursively propagating to leaves. Leaves are tensors with `requires_grad=True` and no `grad_fn`.
- **`Function` / `FunctionContext`** (`autograd/autograd.py`): the extension point for custom ops. `forward(ctx, *inputs)` runs the computation and `ctx.save_for_backward(...)` stashes what the backward needs; `backward(ctx, grad_output)` returns input gradients. Built-ins: `Add`, `Mul`, `MatMul`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`.
- **`grad(outputs, inputs, grad_outputs=…)`** and the module-level **`backward(...)`**: PyTorch-style gradient entry points.
- **`GradientTape`**: a TensorFlow-style context manager (`watch`, `gradient`, `jacobian`, `batch_jacobian`) layered over the same engine.

## 5. Autograd Design Notes

- **Reverse mode.** A single backward pass costs roughly one forward pass and yields the gradient of a scalar (or a seeded vector-Jacobian product) w.r.t. all inputs.
- **Jacobian** (`jacobian(output, input)`): computed by seeding a one-hot gradient at each output component and running backward, reading one row of the Jacobian per pass. This reuses the existing graph — no recomputation. Cost is `O(output.size)` backward passes.
- **Hessian** (`hessian(output, input, func=…)`): because gradients are returned as NumPy arrays (no double backprop), the second derivative cannot be read from the static graph. It is formed by **central finite differences of the reverse-mode gradient**, so the caller must pass `func` — a callable that recomputes the scalar output from a fresh input tensor. With `float32` storage the practical accuracy is ≈1e-3–1e-4 (step size tuned to the float32 round-off floor). Calling `hessian` without `func` raises `NotImplementedError` rather than returning a misleading result.

## 6. Graph, Executor & Backends

- **`graph/graph.py`**: an explicit computation-graph IR for the trace/optimize/export path (distinct from the implicit eager graph).
- **`backend/passes.py`**: optimization passes (e.g. fusion candidates) over the IR.
- **`backend/native.py`**: direct NumPy execution of the lowered graph.
- **`backend/onnx_backend.py`**: ONNX export/interop. **Known limitation:** `ONNXBackend.execute()` is not implemented (raises `NotImplementedError`); export scaffolding exists but the inference path is incomplete.
- **`executor/executor.py`**: drives execution of a lowered graph.

## 7. Known Limitations

- No higher-order autodiff; the Hessian is finite-difference based and requires a recompute `func` (§5).
- `float32`-only, CPU-only.
- `MatMul.backward` assumes ≥2-D operands (it transposes the last two axes); matrix–vector products with a 1-D operand are not supported.
- ONNX execution backend is incomplete (§6).

## 8. Testing

171 tests across the engine: `test_autograd.py` (reverse-mode correctness, Jacobian/Hessian), `test_dynamic_shapes.py`, `test_graph_optimization.py`, `test_jit_trace.py`, `test_backend.py`, `test_memory_optimization.py`.

```bash
conda activate dev
cd 06-real-world-projects/38-dynamic-graph-execution
pip install -e ".[dev]"
pytest tests/ -v
```

## 9. Future Work

- True higher-order autodiff by making `backward` produce `Tensor`s (re-traceable) instead of NumPy arrays.
- Complete the ONNX execution backend.
- Generalize `MatMul.backward` to handle 1-D operands (matrix–vector).
- GPU backend.
