# Real-World Projects

> **52 production-grade implementations demonstrating the concepts taught in sections 01–05 in real, working systems.**

---

## What This Section Is

The other sections of this repo teach concepts. This section *applies* them. Every project here is a working system — not a snippet, not a toy — written in the language and style appropriate to its domain, with tests, documentation, and a `BLUEPRINT.md` describing the design.

| Metric | Value |
|--------|-------|
| **Projects** | 52 |
| **Languages** | Python (35), Rust (16), Go (1) |
| **Lines of code** | ~300,000 |
| **Tests** | ~9,500 |
| **Average completion** | 92% (per 2026-05-12 audit) |

See [`PROJECTS_STATUS.md`](PROJECTS_STATUS.md) for the master tracker, [`audit/AUDIT_REPORT.md`](audit/AUDIT_REPORT.md) for the latest blueprint-vs-implementation audit, and [`CONCEPT_TO_PROJECT_MAP.md`](CONCEPT_TO_PROJECT_MAP.md) for the bridge from tutorial concepts to the projects that demonstrate them.

---

## Layout

Projects live as flat numbered subdirectories: `01-distributed-job-queue/`, `02-microservice-platform/`, … `52-time-series-database/`. Numbering is preserved from the source repo so the audit reports and status tracker stay valid.

Each project contains:

- `BLUEPRINT.md` — technical design and architecture (read this first)
- `PROGRESS.md` — implementation status (verify against actual code; some are stale)
- `src/` — source
- `tests/` — test suite
- `SESSION_CONTEXT.md` — Claude session notes (where present)

Archived / build-blocked projects live under [`archive/`](archive/).

---

## Categories

| Range | Category | Count | Avg completion |
|-------|----------|-------|----------------|
| 01–10 | Foundation & Backend Infrastructure | 10 | 89% |
| 11–20 | Distributed Systems & Infrastructure | 10 | 90% |
| 21–37 | ML/AI Core Systems | 17 | 89% |
| 38–49 | Advanced ML Infrastructure | 12 | 95% |
| 50–52 | Data Infrastructure | 3 | 95% |

---

## Working on a Project

```bash
# Python projects
conda activate dev
cd 06-real-world-projects/XX-project-name
pip install -e ".[dev]"
pytest tests/ -v

# Rust projects
cd 06-real-world-projects/XX-project-name
cargo build --release
cargo test

# Go (project 02 only)
cd 06-real-world-projects/02-microservice-platform
make deps && make build && make test
```

The repo's top-level [`CLAUDE.md`](../CLAUDE.md) has the full command reference and per-language conventions.

---

## How This Section Connects to the Tutorials

When you finish a tutorial in §01–05 and want to see the idea in a complete system, follow the cross-links in [`CONCEPT_TO_PROJECT_MAP.md`](CONCEPT_TO_PROJECT_MAP.md). For example:

- §04 *RAG fundamentals* → projects 25 (baseline RAG), 26 (advanced RAG), 27 (micro-model orchestrated RAG)
- §01 *Rust async runtimes* → project 06 (async runtime from scratch)
- §03 *Distributed training* → projects 04 (training orchestrator), 30 (parameter server), 40 (distributed autograd)
- §02 *Streaming systems* → projects 08 (streaming platform), 36 (distributed streaming analytics), 51 (message queue)

The reverse direction is also supported: each project's `BLUEPRINT.md` references the tutorials covering its constituent concepts (added incrementally as projects are touched).
