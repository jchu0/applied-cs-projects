# Project 26: Advanced RAG - Setup Guide

## Overview
Production-grade RAG system with neural reranking, query rewriting, hybrid search, and advanced evaluation.

## Prerequisites
- Python 3.9+
- 16GB+ RAM (32GB recommended)
- GPU recommended (for reranking)
- Redis (for caching)
- LLM API access

## Installation

### 1. System Dependencies

**Ubuntu/Debian**
```bash
sudo apt-get update
sudo apt-get install -y redis-server
```

**macOS**
```bash
brew install redis
```

**Windows**
Download Redis from: https://github.com/microsoftarchive/redis/releases

### 2. Python Environment
```bash
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### 3. Download Models
```bash
# Download cross-encoder model
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Download embedding model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Configuration

### 1. Environment Variables
```bash
# .env file
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
COHERE_API_KEY=your_key

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Reranker Model
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Vector Store
VECTOR_STORE_TYPE=chroma
CHROMA_PATH=./data/chroma_db

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Monitoring
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

### 2. Redis Setup
```bash
# Start Redis
redis-server

# Verify connection
redis-cli ping  # Should return PONG
```

### 3. Vector Store Setup
```python
from advancedrag import create_pipeline

# Initialize with default settings
pipeline = create_pipeline()

# Or with custom configuration
from advancedrag import AdvancedRAGConfig

config = AdvancedRAGConfig(
    retrieval_k=20,
    rerank_k=5,
    use_hybrid_search=True,
    use_query_rewriting=True,
    cache_enabled=True
)

pipeline = create_pipeline(config=config)
```

## Usage

### Complete Pipeline
```python
from advancedrag import create_pipeline
from advancedrag import Document

# Create pipeline
pipeline = await create_pipeline()

# Add documents
docs = [
    Document(
        id="doc1",
        content="Machine learning is...",
        metadata={"topic": "ML"}
    ),
    Document(
        id="doc2",
        content="Deep learning uses...",
        metadata={"topic": "DL"}
    )
]

await pipeline.add_documents(docs)

# Query with all features
result = await pipeline.execute(
    query="What is deep learning?",
    top_k=5,
    rewrite_query=True,
    use_hybrid=True
)

print(f"Answer: {result.answer.answer}")
print(f"Confidence: {result.answer.confidence}")
for citation in result.answer.citations:
    print(f"  - {citation.source_title}: {citation.quoted_text}")
```

### Running API Server
```bash
# Start server
uvicorn advancedrag.api.main:app --host 0.0.0.0 --port 8000

# With auto-reload for development
uvicorn advancedrag.api.main:app --reload
```

### API Examples

**Query with Reranking**
```bash
curl -X POST "http://localhost:8000/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5,
    "rewrite_query": true,
    "include_citations": true
  }'
```

**Index Documents**
```bash
curl -X POST "http://localhost:8000/v1/index" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "Machine learning is a subset of AI...",
        "metadata": {"source": "ml_basics"}
      }
    ]
  }'
```

**Search Only (No Generation)**
```bash
curl -X POST "http://localhost:8000/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deep learning",
    "top_k": 10
  }'
```

## Advanced Features

### 1. Query Rewriting
```python
from advancedrag import QueryRewriter

rewriter = QueryRewriter(llm_client=your_llm)

# Generate multiple query variations
rewrites = await rewriter.rewrite(
    "What is ML?",
    num_rewrites=3
)
# Returns: ["What is machine learning?",
#           "Explain ML concepts",
#           "Machine learning definition"]
```

### 2. Hybrid Search (Vector + BM25)
```python
from advancedrag import HybridRetriever

retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    alpha=0.7  # Weight for vector search
)

results = await retriever.retrieve(query, top_k=10)
```

### 3. Multi-Stage Reranking
```python
from advancedrag import MultiStageReranker, CrossEncoderReranker

reranker = MultiStageReranker(
    stage1_reranker=CrossEncoderReranker(),
    stage2_reranker=SLMReranker(llm_client),
    stage1_top_k=50,
    diversity_lambda=0.7
)

reranked = await reranker.rerank(query, candidates, top_k=5)
```

### 4. Context Construction
```python
from advancedrag import ContextConstructor

constructor = ContextConstructor(
    max_tokens=2000,
    prioritize_by="relevance"
)

context = constructor.construct(
    documents=retrieved_docs,
    query=query
)
```

## Evaluation

### Running Evaluations
```python
from advancedrag import RAGEvaluator

evaluator = RAGEvaluator()

# Evaluate on test set
results = await evaluator.evaluate(
    pipeline=pipeline,
    test_cases=test_data,
    metrics=["relevance", "faithfulness", "answer_correctness"]
)

print(f"Average Relevance: {results['relevance']}")
print(f"Average Faithfulness: {results['faithfulness']}")
```

### A/B Testing
```python
from advancedrag import ABTestManager

manager = ABTestManager()

# Define variants
manager.add_variant("baseline", config_baseline)
manager.add_variant("reranked", config_with_reranking)

# Run test
results = await manager.run_test(
    test_queries=queries,
    num_samples_per_variant=100
)
```

## Monitoring & Observability

### 1. Enable OpenTelemetry
```python
from advancedrag import setup_tracing

setup_tracing(
    service_name="advanced-rag",
    endpoint="http://localhost:4318"
)
```

### 2. Prometheus Metrics
```bash
# Metrics available at /metrics endpoint
curl http://localhost:8000/metrics
```

### 3. Custom Monitoring
```python
from advancedrag import get_monitor

monitor = get_monitor()

# Track custom metrics
monitor.track_latency("retrieval", latency_ms)
monitor.track_quality("reranking", score)
```

## Performance Tuning

### 1. Caching Strategy
```python
from advancedrag import configure_cache

configure_cache(
    cache_type="redis",
    ttl=3600,
    max_size=10000
)
```

### 2. Batch Processing
```python
# Process multiple queries in batch
results = await pipeline.execute_batch(
    queries=["query1", "query2", "query3"],
    batch_size=10
)
```

### 3. GPU Acceleration
```bash
# Install GPU-enabled sentence-transformers
pip install sentence-transformers[gpu]

# Use GPU for reranking
export CUDA_VISIBLE_DEVICES=0
```

## Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=advancedrag --cov-report=html

# Run benchmarks
pytest tests/test_reranking.py --benchmark-only
```

## Common Issues

### Issue: Slow reranking
**Solution**: Use smaller cross-encoder model or reduce candidates
```python
config = AdvancedRAGConfig(retrieval_k=10)  # Reduce from 20
```

### Issue: Redis connection errors
**Solution**: Verify Redis is running
```bash
redis-cli ping
```

### Issue: OOM during embedding
**Solution**: Reduce batch size
```python
embedder.batch_size = 8  # Reduce from default 32
```

## Project Structure
```
26-advanced-rag/
├── src/advancedrag/
│   ├── pipeline.py           # Main pipeline
│   ├── reranking/           # Rerankers
│   ├── query/               # Query processing
│   ├── retrieval/           # Hybrid retrieval
│   ├── evaluation/          # Evaluation tools
│   ├── api/                 # FastAPI app
│   └── ...
├── tests/
├── requirements.txt
└── SETUP.md
```

## Next Steps
1. Configure reranking models
2. Set up hybrid search
3. Enable caching
4. Run evaluations
5. Deploy with monitoring

## Resources
- [Sentence Transformers - Cross Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Redis Documentation](https://redis.io/docs/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
