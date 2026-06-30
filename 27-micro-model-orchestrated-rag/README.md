# Micro-Model Orchestrated RAG

Efficient RAG system using small language models (SLMs) for cost-effective, low-latency retrieval and generation.

## Features

- **SLM Orchestration**: Coordinates multiple small models for different tasks
- **Query Routing**: Lightweight classifier routes queries to appropriate models
- **Efficient Retrieval**: Optimized embedding models for fast vector search
- **Fallback to LLMs**: Automatic fallback to larger models when needed
- **Guardrails**: Input/output validation and safety checks
- **Tracing**: LangSmith/Arize Phoenix integration for debugging

## Installation

```bash
# Basic installation
pip install -e .

# With all dependencies
pip install -e ".[full]"
```

## Quick Start

```python
from microrag import Orchestrator

# Initialize with small models
orchestrator = Orchestrator(
    query_router="sentence-transformers/all-MiniLM-L6-v2",
    retriever="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    generator="microsoft/phi-2"
)

# Process query with SLM pipeline
response = orchestrator.query(
    "Explain quantum computing",
    context_docs=documents
)
```

## Model Configuration

The system uses three main SLM components:

| Component | Default Model | Size | Purpose |
|-----------|--------------|------|---------|
| Query Router | all-MiniLM-L6-v2 | 22M | Classify query intent |
| Retriever | multi-qa-MiniLM-L6 | 22M | Semantic search |
| Generator | phi-2 | 2.7B | Response generation |

## Infrastructure

```bash
docker-compose up -d
# ChromaDB: http://localhost:8000
```

## Configuration

Copy `.env.example` to `.env` and configure model paths and API keys.

## Testing

```bash
pytest tests/ -v
```
