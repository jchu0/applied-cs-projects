# Custom Embedding Model

Embedding model training infrastructure in Rust implementing contrastive learning with hard negative mining, bi-encoder architecture, distributed training (DDP/FSDP), comprehensive retrieval evaluation metrics, ONNX export, and MLOps experiment tracking.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §04 Embeddings · §04 Vector Stores · §03 Distributed Training (data parallelism). See also the [concept-to-project map](../CONCEPT_TO_PROJECT_MAP.md).

## What's real vs simulated

**Real, trainable model:** `src/trainable.rs` (`TrainableEmbedder`) is a genuinely learnable embedding model — a token-embedding table with mean pooling, trained by **real analytic gradient descent** on a contrastive (logistic) loss. Its tests show training reduces loss and pulls relevant pairs together while pushing irrelevant pairs apart. It is dependency-free (the gradient math is hand-derived and fully visible); a production system would swap the backprop for `candle`/`tch`/`burn` while keeping the same objective.

**Still simulated (legacy path):** the original `BiEncoder` (`src/model.rs`) applies *random, untrained* linear projections, and `EmbeddingTrainer::update_weights` (`src/trainer.rs`) perturbs weights with random noise rather than gradients — so that path does not learn. Distributed training (`DDP`, `FSDP`) and ONNX export run in CPU-simulated mode: the data structures, configuration, and API surface are complete, but no actual gradient synchronization or ONNX graph serialization occurs. The loss functions, evaluation metrics (Recall@k, MRR, MAP, NDCG), hard negative miner, memory bank, and MLOps tracker are all functionally implemented in pure Rust.

## Layout

```
src/
  model.rs        Bi-encoder with CLS / mean / max pooling strategies
  dataset.rs      EmbeddingExample/Batch types, HardNegativeMiner, MemoryBank
  loss.rs         MultipleNegativesRankingLoss, TripletMarginLoss
  trainer.rs      EmbeddingTrainer with gradient accumulation
  evaluation.rs   Retrieval metrics: Recall@k, MRR, MAP, NDCG
  distributed.rs  DDP and FSDP wrappers (simulated single-process)
  onnx.rs         OnnxExporter, OnnxQuantizer, DynamicBatcher (simulated)
  mlops.rs        ExperimentTracker, ModelRegistry, HyperparameterSearch
  serving.rs      EmbeddingService REST API types and handler stubs
  lib.rs          Crate root, shared error types, constants

benches/
  embedding_benchmarks.rs   Criterion benchmarks for encode / loss / eval

BLUEPRINT.md      Full architecture design and component specifications
PROGRESS.md       Implementation status notes (may be stale)
```

## Build & Test

```bash
cd 06-real-world-projects/21-custom-embedding-model
cargo build
cargo test
cargo bench   # optional — runs Criterion benchmarks
```

The test suite covers ~269 unit tests across all modules. No external services or ML runtimes are required.

## Known gaps (78% complete)

- No real ML backend: replacing random weights with `candle` or `ort` is the primary open item (~5 days effort).
- No live HTTP server: `serving.rs` defines request/response types but does not start a real listener.
- ONNX export writes a placeholder file rather than a valid ONNX graph.
