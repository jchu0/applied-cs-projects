# Advanced RAG

Advanced Retrieval-Augmented Generation system with reranking, query rewriting, and hybrid retrieval strategies.

## Features

- **Hybrid Retrieval**: Combines dense (embedding) and sparse (BM25) retrieval
- **Query Rewriting**: LLM-powered query expansion and reformulation
- **Reranking**: Cross-encoder reranking for improved relevance
- **Multi-Vector Store Support**: ChromaDB, Qdrant, Pinecone, Weaviate
- **Caching**: Redis-based result caching
- **Observability**: OpenTelemetry tracing and Prometheus metrics

## Installation

```bash
# Basic installation
pip install -e .

# With all optional dependencies
pip install -e ".[full]"

# Specific extras
pip install -e ".[ml,vectordb,llm]"
```

## Quick Start

```python
from advancedrag import AdvancedRAGPipeline

# Initialize pipeline
pipeline = AdvancedRAGPipeline(
    vector_store="chromadb",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Index documents
pipeline.index(documents)

# Query with hybrid retrieval + reranking
results = pipeline.query(
    "What is the capital of France?",
    top_k=10,
    use_reranking=True
)
```

## Infrastructure

```bash
# Start required services
docker-compose up -d

# ChromaDB: http://localhost:8000
# Redis: localhost:6379
```

## Configuration

Copy `.env.example` to `.env` and configure:
- LLM API keys (OpenAI, Anthropic, Cohere)
- Vector store settings
- Embedding and reranker models

## Testing

```bash
pytest tests/ -v
```

## Architecture

See `BLUEPRINT.md` for detailed architecture documentation.
