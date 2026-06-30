# Concept → Project Map

> **A bridge from the tutorial sections to the 52 from-scratch implementation projects that demonstrate them.**

This document is intentionally a map, not a script. When you finish a tutorial and want to see the idea in a working system, look it up here. When you're working on a project and want background, follow the links back into `01-` … `05-`.

The mapping is curated, not exhaustive — a project usually exercises *many* concepts; the entries below highlight the ones it demonstrates **best**. Use [`PROJECTS_STATUS.md`](PROJECTS_STATUS.md) for full per-project detail.

> **Convention:** Project numbers refer to subdirectories of `06-real-world-projects/`. Tutorial paths are relative to the omnibus root (e.g. `04-ai-engineering/03-vector-databases/`).

---

## §01 Software Engineering

Paths are relative to the omnibus root, under `01-software-engineering/`.

| Tutorial topic | Best-fit projects |
|----------------|-------------------|
| `python/03-concurrency/python-concurrency.md` | **01** (job queue, Celery + asyncio), **08** (streaming platform, asyncio) |
| `python/07-web-development/` (`fastapi-production.md`, `python-web-development.md`) | **05** (SaaS platform), **24, 25, 26, 29** (FastAPI-served ML systems) |
| `python/05-performance/python-performance.md` | **20** (SIMD analytics), **03** (cache), **17** (query engine) |
| `python/08-observability/python-observability.md` | **09** (data observability), **05** (SaaS) |
| `rust/05-async-rust/rust-async.md`, `rust/03-concurrency/rust-concurrency.md` | **06** (async runtime from scratch), **51** (message queue), **52** (TSDB), **11** (Raft) |
| `rust/00-fundamentals/`, `rust/01-ownership-borrowing/`, `rust/02-advanced-types/` | **03** (cache), **06** (async runtime), **11** (Raft), **17** (query engine) |
| `rust/04-unsafe-rust/rust-unsafe.md` | **20** (SIMD analytics), **19** (GPU kernels), **15** (OS kernel) |
| `go/*` (all Go tutorials) | **02** (microservice platform — gRPC + Kong; the repo's sole Go system) |
| `typescript/*` (fundamentals, advanced-types, testing, async-promises, node-ecosystem, web-development) | **05** (SaaS frontend), **16** (CRDT TS client) |
| `cpp/01-modern-cpp/`, `cpp/03-templates-metaprogramming/` | (no current project — see §07 benchmarks) |
| `java/01-fundamentals/java-fundamentals.md` | (no current project — see §07 Docker examples) |

---

## §02 Data Engineering

| Tutorial subsection | Best-fit projects |
|---------------------|-------------------|
| **01 data-pipelines** (Airflow, dbt, orchestration) | **07** (data lakehouse), **10** (warehouse semantic layer) |
| **02 data-processing** (Spark, batch ETL) | **07** (lakehouse), **50** (feature engineering platform) |
| **03 data-warehousing** (columnar, OLAP, semantic layers) | **10** (semantic layer), **17** (columnar query engine), **52** (TSDB) |
| **04 streaming** (Kafka, Flink, exactly-once) | **08** (streaming platform), **12** (distributed log), **36** (streaming analytics), **51** (message queue) |
| **05 data-quality** (validation, lineage, observability) | **09** (data observability), **49** (benchmark suite — for ML data quality) |
| **06 infrastructure** (cloud platforms, Kubernetes, Terraform) | **07** (lakehouse), **34** (distributed file system), **52** (TSDB), **10** (warehouse semantic layer), **50** (feature platform — stateful K8s workload) |

---

## §03 Machine Learning Engineering

| Tutorial subsection | Best-fit projects |
|---------------------|-------------------|
| **01 ml-fundamentals** | **24** (synthetic data), **35** (differentiable programming primer) |
| **02 deep-learning** (autograd, training loops) | **35** (diffprog), **38** (dynamic graph execution), **40** (distributed autograd) |
| **03 ml-systems** (model serving, feature stores) | **29** (model routing), **44** (autoregressive inference), **50** (feature platform) |
| **04 production-ml** (MLOps, monitoring, drift) | **04** (training orchestrator), **09** (observability), **49** (benchmark suite) |
| **05 distributed-training** (DDP, FSDP, parameter servers) | **04** (orchestrator), **30** (parameter server), **32** (distributed tensor algebra), **40** (distributed autograd), **48** (multi-GPU scheduler) |
| **06 cuda-optimization** | **19** (GPU kernel optimization), **39** (GPU memory manager), **44** (autoregressive inference, CUDA path), **48** (multi-GPU scheduler) |

---

## §04 AI Engineering

| Tutorial subsection | Best-fit projects |
|---------------------|-------------------|
| **01 llm-fundamentals** (tokenization, transformers, attention) | **22** (long-context attention), **23** (agentic runtime), **41** (quantized LLM), **44** (autoregressive inference — decode loop) |
| **02 llm-applications** (RAG, agents, workflows) | **25** (RAG baseline), **26** (advanced RAG), **27** (micro-model RAG), **28** (workflow engine), **23** (agentic runtime) |
| **03 vector-databases** | **21** (custom embedding model), **43** (vector index), **25, 26, 27** (RAG end-to-end) |
| **04 llm-inference** (KV cache, batching, quantization) | **22** (long-context), **41** (quantization), **44** (autoregressive inference), **47** (on-device LLM) |
| **05 multimodal-ai** | (no current project — gap) |
| **06 ai-safety** | **49** (benchmark suite — eval harness), **29** (routing — guardrails) |
| **07 custom-models** (training, custom inference, CUDA kernels) | **04** (orchestrator), **27** (micro-model RAG fine-tunes), **45** (neural compression) for *training*; **47** (on-device), **44** (autoregressive inference), **41** (quantized) for *custom inference*; **19** (GPU kernel optimization), **39** (GPU memory manager), **48** (multi-GPU kernel scheduler) for *CUDA kernels* — see also §03 `06-cuda-optimization` |

---

## §05 Cross-Cutting Concerns

| Subsection | Best-fit projects |
|------------|-------------------|
| **security** (mTLS, auth, secrets) | **02** (microservice platform — Kong auth), **05** (SaaS — full auth stack), **13** (service mesh — mTLS + SPIFFE) |
| **observability** (metrics, traces, logs) | **09** (data observability), **49** (benchmark suite), **05** (SaaS) |
| **ci-cd** | **05** (Phase 6 DevOps — partial), **49** (benchmark CI) |
| **cost-optimization** | **46** (multi-tenant GPU scheduler), **48** (multi-GPU kernel scheduler) |

---

## §07 Infrastructure

The §07 tutorials provide language-comparison benchmarks and Docker / Kubernetes deployment templates. Projects that exercise these heavily:

| §07 topic | Best-fit projects |
|-----------|-------------------|
| `benchmarks/` + `benchmarks/languages` (cross-language perf) | **03, 06, 14, 17, 19, 20, 51, 52** (Rust) — direct comparison points |
| `benchmarks/databases` | **17** (columnar query engine), **52** (TSDB), **03** (cache) |
| `benchmarks/frameworks` (web frameworks) | **02** (microservice platform — Go/gRPC), **05** (SaaS — FastAPI/ASGI) |
| `benchmarks/ml` | **49** (AI benchmark suite — direct analog), **44** (autoregressive inference), **19** (GPU kernels), **48** (multi-GPU scheduler) |
| `docker/python/fastapi` | **24, 25, 26, 29, 50** (all FastAPI) |
| `docker/go/microservice` | **02** |
| `docker/rust/actix` | **03, 06, 13, 14, 17, 51, 52** |
| `kubernetes/` | **02** (Kong + services), **05** (SaaS, K8s gap), **13** (service mesh) |

---

## Book: *Applied Computer Science* (chapters → projects)

The narrative book — [*Applied Computer Science*](https://github.com/jchu0/applied-cs-book) — re-presents the §01–§07 tutorials in
house style; every chapter links back to the projects that implement its concepts
via the `> **Build it →**` pattern. This table keeps that **book → project**
direction explicit (the per-section tables above remain the tutorial→project
source of truth; book chapters draw their links from them).

Part I (Cross-Language Foundations) teaches the cross-cutting concepts once,
comparatively across all six languages; Part II (Language Field Guides) is the
slimmed per-language material that builds on it.

| Book chapter | Build-it projects |
|--------------|-------------------|
| Part I · Concurrency and Parallelism Models | **01, 02, 06, 08, 11** |
| Part I · Type Systems and Generics | **17, 18, 31** |
| Part I · Memory and Resource Management | **03, 14, 39** |
| Part I · Error Handling | **01, 11, 13** |
| Part I · Testing and Quality | **01, 09, 49** |
| Part I · Performance and Profiling | **03, 19, 20, 49** |
| Part II · Python: Advanced Language Features | **05, 24, 50** |
| Part II · Python: Design Patterns & Architecture | **05, 28, 29** |
| Part II · Python: Web Development | **05, 24, 50** |
| Part II · Python: Microservices | **02, 29** |
| Part II · Python: Observability | **05, 09, 49** |
| Part II · TypeScript: Fundamentals | **05, 16** |
| Part II · TypeScript: The Node Ecosystem | **05, 16** |
| Part II · TypeScript: Frontend with React | **05, 16** |
| Part II · Go: Fundamentals | **02** |
| Part II · Go: Packages & Modules | **02** |
| Part II · Go: Web Services & gRPC | **02** |
| Part II · Rust: Fundamentals | **03, 06, 11, 17** |
| Part II · Rust: Ownership & Borrowing | **03, 06, 11** |
| Part II · Rust: Unsafe Rust | **15, 19, 20** |
| Part III · Data Orchestration & Pipelines | **07, 10** |
| Part III · Data Processing Engines | **07, 17, 50** |
| Part III · Data Warehousing & Modeling | **10, 17, 52** |
| Part III · Streaming & Real-Time Data | **08, 12, 36** |
| Part III · Data Quality & Testing | **09, 49** |
| Part III · Data Infrastructure | **07, 34, 52** |
| Part IV · Machine Learning Foundations | **24, 35** |
| Part IV · Deep Learning Frameworks | **35, 38, 40** |
| Part IV · ML Systems: Tracking, Features & Serving | **29, 44, 50** |
| Part IV · Production ML | **04, 09, 49** |
| Part IV · Distributed Training | **04, 30, 40** |
| Part IV · GPU Programming & CUDA | **19, 39, 48** |
| Part V · CI/CD and Deployment Automation | **05, 49** |
| Part V · Observability | **05, 09, 49** |
| Part V · Security | **02, 05, 13** |
| Part V · Cost Optimization | **46, 48** |
| Part VI · Containerization with Docker | **02, 03, 05, 07, 08, 13, 50, 51, 52** |
| Part VI · Orchestration with Kubernetes | **02, 05, 13, 50** |
| Part VI · Benchmarking Systems | **02, 03, 05, 06, 17, 19, 20, 44, 49, 52** |

> The folded cross-cutting topics (per-language Concurrency, Testing, Performance,
> Type Systems, Error Handling, Memory) now live in the Part I Foundations rows
> above, which carry their Build-it links.
>
> **Part II · Java** chapters have no rows above: the repo has no Java §06 system
> (a known portfolio gap — see the §01 table). The one artifact link is *Java:
> Spring Boot & Web Services* → the book's [Containerization chapter](https://github.com/jchu0/applied-cs-book/blob/main/book/infra-containerization/index.md).
> **Part II · C++** is likewise project-less (see §01).

---

## Reverse Index — "Which tutorials should I read before this project?"

For projects with the steepest concept ramp:

| Project | Recommended pre-reading |
|---------|-------------------------|
| **06** async-runtime | §01 `rust/05-async-rust/`, `rust/03-concurrency/`, `rust/04-unsafe-rust/` |
| **11** distributed-kv-raft | §01 `rust/*`, §02 `04-streaming` (replication intuition) |
| **15** minimal-os-kernel | §01 `rust/04-unsafe-rust/`, `cpp/01-modern-cpp/` |
| **20** simd-analytics | §01 `rust/04-unsafe-rust/`, §07 `benchmarks/languages/` |
| **22** long-context-attention | §03 `02-deep-learning`, §04 `01-llm-fundamentals`, §04 `04-llm-inference` |
| **26** advanced-rag | §04 `02-llm-applications`, §04 `03-vector-databases` |
| **30** parameter-server | §03 `05-distributed-training` |
| **40** distributed-autograd | §03 `02-deep-learning`, §03 `05-distributed-training` |
| **48** multi-gpu-kernel-scheduler | §03 `06-cuda-optimization`, §03 `05-distributed-training` |
| **51** message-queue | §02 `04-streaming`, §01 `rust/05-async-rust/` |

---

## Maintaining This Map

- **Adding a project:** add a row in the relevant §01–§05 table and (if non-trivial) a reverse-index entry.
- **Adding a tutorial:** scan the project list for existing systems that already demonstrate it; cross-link both directions.
- **Adding a book chapter:** add a row to the *Book → projects* table above, drawn from the matching §01–§07 tutorial→project table, so the bidirectional link stays accurate.
- **Discovered gaps:** the `05-multimodal-ai` and `cpp/*` cells above are intentionally empty — they mark where the portfolio doesn't yet have a real-world example. Future projects targeting those gaps would slot in here.

Update this file in the same commit that introduces the cross-link on the tutorial / project side, so the map and the linked content stay synchronized.
