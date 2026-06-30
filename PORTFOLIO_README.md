# ML/AI Systems Engineering Projects

## Staff-Level Design Documentation

This repository contains comprehensive blueprints for 48 advanced projects spanning systems engineering, data engineering, and ML infrastructure. Each project is designed to teach deep technical concepts at a staff/principal engineer level.

---

## Project Categories

### Software Engineering (1-6)
| # | Project | Difficulty | Key Technologies |
|---|---------|------------|------------------|
| 1 | [Distributed Job Queue + Scheduler](./01-distributed-job-queue/) | ⭐⭐⭐⭐ | Redis, RabbitMQ, gRPC |
| 2 | [Microservice Platform](./02-microservice-platform/) | ⭐⭐⭐⭐⭐ | gRPC, Protobuf, K8s |
| 3 | [High-Performance Cache](./03-high-performance-cache/) | ⭐⭐⭐⭐ | RESP Protocol, Epoll |
| 4 | [Full SaaS Platform](./04-saas-web-platform/) | ⭐⭐⭐⭐⭐ | Next.js, Django/Go |
| 5 | [Async Runtime](./05-async-runtime/) | ⭐⭐⭐⭐⭐ | Epoll/Kqueue, Futures |
| 6 | [Container Runtime](./06-container-runtime/) | ⭐⭐⭐⭐⭐ | Linux Namespaces, Cgroups |

### Data Engineering (7-10)
| # | Project | Difficulty | Key Technologies |
|---|---------|------------|------------------|
| 7 | [Data Lakehouse](./07-data-lakehouse/) | ⭐⭐⭐⭐ | Delta Lake, Spark/Flink |
| 8 | [Streaming Platform](./08-streaming-platform/) | ⭐⭐⭐⭐⭐ | Kafka, Flink |
| 9 | [Data Observability](./09-data-observability/) | ⭐⭐⭐⭐ | Great Expectations, Airflow |
| 10 | [Warehouse + Semantic Layer](./10-warehouse-semantic-layer/) | ⭐⭐⭐⭐ | dbt, BigQuery/Snowflake |

### Distributed Systems (11-20)
| # | Project | Difficulty | Key Technologies |
|---|---------|------------|------------------|
| 11 | [Distributed KV Store + Raft](./11-distributed-kv-raft/) | ⭐⭐⭐⭐⭐ | Raft Consensus, WAL |
| 12 | [Distributed Log System](./12-distributed-log-system/) | ⭐⭐⭐⭐⭐ | Segment Files, Replication |
| 13 | [Service Mesh](./13-service-mesh/) | ⭐⭐⭐⭐⭐ | mTLS, Sidecar Proxy |
| 14 | [Network Stack](./14-network-stack/) | ⭐⭐⭐⭐⭐ | TCP, HTTP/1.1 |
| 15 | [Minimal OS Kernel](./15-minimal-os-kernel/) | ⭐⭐⭐⭐⭐ | Paging, Syscalls |
| 16 | [CRDT Collaboration Engine](./16-crdt-collaboration/) | ⭐⭐⭐⭐⭐ | CRDTs, WebSockets |
| 17 | [Columnar Query Engine](./17-columnar-query-engine/) | ⭐⭐⭐⭐⭐ | Vectorized Execution |
| 18 | [Compiler/Interpreter](./18-compiler-interpreter/) | ⭐⭐⭐⭐⭐ | AST, Bytecode VM |
| 19 | [GPU Kernel Optimization](./19-gpu-kernel-optimization/) | ⭐⭐⭐⭐⭐ | CUDA, Tiling |
| 20 | [SIMD Analytics Engine](./20-simd-analytics-engine/) | ⭐⭐⭐⭐⭐ | AVX2/AVX-512, NUMA |

### ML/AI Infrastructure (21-30)
| # | Project | Difficulty | Key Technologies |
|---|---------|------------|------------------|
| 21 | [Custom Embedding Model](./21-custom-embedding-model/) | ⭐⭐⭐⭐ | Contrastive Learning |
| 22 | [Long-Context Attention](./22-long-context-attention/) | ⭐⭐⭐⭐⭐ | FlashAttention, Triton |
| 23 | [LLM Agentic Runtime](./23-llm-agentic-runtime/) | ⭐⭐⭐⭐ | ReAct, Tool Use |
| 24 | [Synthetic Data Generator](./24-synthetic-data-generator/) | ⭐⭐⭐⭐ | LLM Augmentation |
| 25 | [RAG Baseline](./25-rag-baseline/) | ⭐⭐⭐ | Vector DB, Embeddings |
| 26 | [Advanced RAG](./26-advanced-rag/) | ⭐⭐⭐⭐ | Reranking, Hybrid Search |
| 27 | [Micro-Model Orchestrated RAG](./27-micro-model-orchestrated-rag/) | ⭐⭐⭐⭐⭐ | SLM Family, Routing |
| 28 | [AI Workflow Engine](./28-ai-workflow-engine/) | ⭐⭐⭐⭐⭐ | DSL, DAG Execution |
| 29 | [Model Routing Layer](./29-model-routing-layer/) | ⭐⭐⭐⭐ | Load Balancing, Queues |
| 30 | [Parameter Server](./30-parameter-server/) | ⭐⭐⭐⭐⭐ | Sharding, Async Updates |

### Advanced ML Systems (31-40)
| # | Project | Difficulty | Key Technologies |
|---|---------|------------|------------------|
| 31 | [ML Compiler](./31-ml-compiler/) | ⭐⭐⭐⭐⭐ | IR, Codegen, Fusion |
| 32 | [Distributed Tensor Algebra](./32-distributed-tensor-algebra/) | ⭐⭐⭐⭐⭐ | Auto-diff, Sharding |
| 33 | [RL Physics Engine](./33-rl-physics-engine/) | ⭐⭐⭐⭐⭐ | Rigid Body, GPU Physics |
| 34 | [Distributed Filesystem](./34-distributed-filesystem/) | ⭐⭐⭐⭐⭐ | MDS, Chunk Servers |
| 35 | [Differentiable Programming](./35-differentiable-programming/) | ⭐⭐⭐⭐⭐ | Autograd, Backward Graph |
| 36 | [Streaming Analytics Engine](./36-streaming-analytics-engine/) | ⭐⭐⭐⭐⭐ | MPP, Shuffle |
| 37 | [Dynamic Graph Runtime](./37-dynamic-graph-runtime/) | ⭐⭐⭐⭐⭐ | Bytecode Tracing |
| 38 | [GPU Memory Manager](./38-gpu-memory-manager/) | ⭐⭐⭐⭐⭐ | Caching Allocator |
| 39 | [Distributed Autograd](./39-distributed-autograd/) | ⭐⭐⭐⭐⭐ | AllReduce, Bucketization |
| 40 | [Vector-Quantized LLM](./40-vector-quantized-llm/) | ⭐⭐⭐⭐⭐ | INT4/FP8, GGUF |

### Specialized Systems (41-48)
| # | Project | Difficulty | Key Technologies |
|---|---------|------------|------------------|
| 41 | [GNN Runtime](./41-gnn-runtime/) | ⭐⭐⭐⭐⭐ | Sparse Ops, Message Passing |
| 42 | [Vector Index](./42-vector-index/) | ⭐⭐⭐⭐⭐ | HNSW, IVF, PQ |
| 43 | [Autoregressive Inference](./43-autoregressive-inference/) | ⭐⭐⭐⭐⭐ | Continuous Batching, KV Cache |
| 44 | [Neural Compression](./44-neural-compression/) | ⭐⭐⭐⭐⭐ | VAE, Arithmetic Coding |
| 45 | [GPU Cluster Scheduler](./45-gpu-cluster-scheduler/) | ⭐⭐⭐⭐⭐ | Gang Scheduling, DRF |
| 46 | [On-Device LLM](./46-on-device-llm/) | ⭐⭐⭐⭐⭐ | mmap, SIMD Kernels |
| 47 | [Multi-GPU Kernel Scheduler](./47-multi-gpu-kernel-scheduler/) | ⭐⭐⭐⭐⭐ | DAG Scheduling, Fusion |
| 48 | [AI Benchmark Suite](./48-ai-benchmark-suite/) | ⭐⭐⭐⭐ | Workload Generation |

---

## Project Structure

Each project contains:

```
project-name/
├── BLUEPRINT.md        # Comprehensive design document
├── PROGRESS.md         # Progress tracking
├── SESSION_CONTEXT.md  # Session state and context
├── src/                # Source code
├── tests/              # Test suites
├── docs/               # Additional documentation
└── config/             # Configuration files
```

---

## Getting Started

1. **Choose a project** based on your learning goals
2. **Read the BLUEPRINT.md** to understand architecture
3. **Check PROGRESS.md** for implementation phases
4. **Use SESSION_CONTEXT.md** to track your work sessions

See [SETUP_RECOMMENDATIONS.md](./SETUP_RECOMMENDATIONS.md) for detailed setup guidance.

---

## Difficulty Legend

- ⭐⭐⭐ = Intermediate (2-4 weeks)
- ⭐⭐⭐⭐ = Advanced (4-8 weeks)
- ⭐⭐⭐⭐⭐ = Expert (8-16 weeks)

---

## Prerequisites

### General
- Strong programming fundamentals (Python, Go, Rust, or C++)
- Understanding of data structures and algorithms
- Basic distributed systems knowledge

### For Systems Projects (1-20)
- Linux internals, networking basics
- Concurrency and parallelism patterns

### For ML/AI Projects (21-48)
- PyTorch/JAX fundamentals
- GPU programming basics (CUDA)
- Machine learning theory

---

## License

These blueprints are for educational purposes. Implement responsibly.
