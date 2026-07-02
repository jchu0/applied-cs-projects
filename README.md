# Applied Computer Science — Projects

52 from-scratch systems that implement the concepts taught in the companion book
**[Applied Computer Science](https://jameshu.io/books/applied-cs)**. Where a
book chapter says **"Build it →"**, it points here.

Each project is built to real engineering standards — trait/interface-driven
design, typed errors, and a substantial test suite — and each is **honest about
its scope**. Some are complete, hardened services; others are deliberately
CPU-only simulations or teaching implementations (the GPU, OS-kernel, and
async-runtime work in particular). Every project's `README` and `BLUEPRINT.md`
state what is real versus simulated, so nothing here overclaims.

## Conventions

- **Python** (34 projects) — FastAPI + `pytest`, packaged with `pip install -e .`.
- **Rust** (17 projects) — trait-based design, `Result` error handling, Criterion
  benchmarks, built with `cargo build` / `cargo test`.
- **Go** (1 project) — gRPC + protobuf.
- Every project ships its own `README.md`, a `docs/BLUEPRINT.md` design document,
  and build/run instructions. Served APIs share an opt-in hardening baseline
  (API-key auth, in-process rate limiting, request timeouts).

## Projects

### Foundation & backend (01–10)

| # | Project | Lang |
|---|---------|------|
| 01 | [Distributed Job Queue](01-distributed-job-queue) | Python |
| 02 | [Microservice Platform](02-microservice-platform) | Go |
| 03 | [High-Performance Cache (redis-lite)](03-high-performance-cache) | Rust |
| 04 | [ML Training Orchestrator](04-ml-training-orchestrator) | Python |
| 05 | [SaaS Web Platform](05-saas-web-platform) | Python |
| 06 | [Async Runtime](06-async-runtime) | Rust |
| 07 | [Data Lakehouse](07-data-lakehouse) | Python |
| 08 | [Streaming Platform](08-streaming-platform) | Python |
| 09 | [Data Observability Platform](09-data-observability) | Python |
| 10 | [Warehouse Semantic Layer](10-warehouse-semantic-layer) | Python |

### Distributed systems (11–20)

| # | Project | Lang |
|---|---------|------|
| 11 | [Distributed Key-Value Store (Raft)](11-distributed-kv-raft) | Rust |
| 12 | [Distributed Log System (Kafka-lite)](12-distributed-log-system) | Rust |
| 13 | [Service Mesh](13-service-mesh) | Rust |
| 14 | [Network Stack (TCP + HTTP)](14-network-stack) | Rust |
| 15 | [Minimal OS Kernel](15-minimal-os-kernel) | Rust |
| 16 | [CRDT Collaboration Engine](16-crdt-collaboration) | Rust |
| 17 | [Columnar Query Engine](17-columnar-query-engine) | Rust |
| 18 | [Python Subset Compiler & Interpreter](18-compiler-interpreter) | Rust |
| 19 | [GPU GEMM Optimization (cuBLAS-lite)](19-gpu-kernel-optimization) | Rust |
| 20 | [SIMD Analytics Engine](20-simd-analytics-engine) | Rust |

### ML / AI core (21–37)

| # | Project | Lang |
|---|---------|------|
| 21 | [Custom Embedding Model](21-custom-embedding-model) | Rust |
| 22 | [Long-Context Attention](22-long-context-attention) | Rust |
| 23 | [LLM Agentic Runtime](23-llm-agentic-runtime) | Rust |
| 24 | [Synthetic Data Generator](24-synthetic-data-generator) | Python |
| 25 | [RAG Baseline](25-rag-baseline) | Python |
| 26 | [Advanced RAG](26-advanced-rag) | Python |
| 27 | [Micro-Model Orchestrated RAG](27-micro-model-orchestrated-rag) | Python |
| 28 | [AI Workflow Engine](28-ai-workflow-engine) | Python |
| 29 | [Model Routing Layer](29-model-routing-layer) | Python |
| 30 | [Large-Scale Parameter Server](30-parameter-server) | Python |
| 31 | [ML Compiler](31-ml-compiler) | Python |
| 32 | [Distributed Tensor Algebra](32-distributed-tensor-algebra) | Python |
| 33 | [RL Physics Engine](33-rl-physics-engine) | Python |
| 34 | [HDFS-Lite Distributed File System](34-distributed-file-system) | Python |
| 35 | [Differentiable Programming](35-differentiable-programming) | Python |
| 36 | [Distributed Streaming Analytics](36-distributed-streaming-analytics) | Python |
| 37 | [Dynamic Graph Execution Runtime](37-dynamic-graph-runtime) | Python |

### Advanced ML (38–49)

| # | Project | Lang |
|---|---------|------|
| 38 | [Dynamic Graph Execution (DynaGraph)](38-dynamic-graph-execution) | Python |
| 39 | [GPU Memory Manager](39-gpu-memory-manager) | Python |
| 40 | [Distributed Autograd System](40-distributed-autograd) | Python |
| 41 | [Vector-Quantized LLM](41-vector-quantized-llm) | Python |
| 42 | [GNN Runtime](42-gnn-runtime) | Python |
| 43 | [Vector Index](43-vector-index) | Python |
| 44 | [Autoregressive Inference Engine](44-autoregressive-inference) | Python |
| 45 | [Neural Compression Engine](45-neural-compression) | Python |
| 46 | [Multi-Tenant GPU Scheduler](46-multi-tenant-gpu-scheduler) | Python |
| 47 | [On-Device LLM Runtime](47-on-device-llm) | Python |
| 48 | [Multi-GPU Kernel Scheduler](48-multi-gpu-kernel-scheduler) | Python |
| 49 | [AI Benchmark Suite](49-ai-benchmark-suite) | Python |

### Data infrastructure (50–52)

| # | Project | Lang |
|---|---------|------|
| 50 | [Feature Engineering Platform](50-feature-engineering-platform) | Python |
| 51 | [Message Queue](51-message-queue) | Rust |
| 52 | [Time-Series Database](52-time-series-database) | Rust |

## Running a project

Each project is self-contained. From its directory:

```bash
# Python
pip install -e ".[dev]"
pytest

# Rust
cargo test
cargo bench
```

See the project's own `README.md` for endpoints, configuration, and design notes.

## License

[MIT](LICENSE) © 2026 James Hu
