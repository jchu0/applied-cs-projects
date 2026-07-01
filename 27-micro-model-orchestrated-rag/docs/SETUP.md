# Project 27: Micro-Model Orchestrated RAG - Setup Guide

## Overview
RAG pipeline orchestrated by specialized Small Language Models (SLMs) for efficient, composable retrieval and generation.

## Prerequisites
- Python 3.9+
- 16GB+ RAM
- GPU recommended (for SLM inference)
- 20GB+ disk space (for models)

## Installation

### 1. Create Environment
```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Download SLM Models
```bash
# Download specialized small models
python scripts/download_models.py

# Or manually:
from transformers import AutoModel, AutoTokenizer

# Chunking SLM (lightweight)
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Retrieval SLM
model = AutoModel.from_pretrained("facebook/contriever")

# Reranking SLM
model = AutoModel.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

### 3. Install Guardrails (Optional)
```bash
# For NeMo Guardrails
pip install nemo-guardrails

# For Guardrails AI
pip install guardrails-ai
```

## Configuration

### 1. Environment Variables
```bash
# .env file
# Model Configuration
CHUNKER_MODEL=all-MiniLM-L6-v2
EMBEDDER_MODEL=all-MiniLM-L6-v2
RETRIEVER_MODEL=facebook/contriever
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
SUMMARIZER_MODEL=facebook/bart-base

# Inference Settings
USE_ONNX=false
USE_GPU=true
BATCH_SIZE=32

# Tracing
ENABLE_TRACING=true
LANGSMITH_API_KEY=your_key
ARIZE_API_KEY=your_key

# Guardrails
ENABLE_GUARDRAILS=true
MIN_RELEVANCE=0.3
MIN_CONFIDENCE=0.4
```

### 2. Component Registry Setup
```python
from microrag import ComponentRegistry

registry = ComponentRegistry()

# Register custom SLMs
from microrag.slm import ChunkerSLM, EmbedderSLM

registry.register("chunker_slm", ChunkerSLM(model_name="..."))
registry.register("embedder_slm", EmbedderSLM(model_name="..."))
```

## Usage

### Basic Pipeline
```python
from microrag import create_pipeline

# Create pipeline with default SLMs
pipeline = create_pipeline(
    use_mock=False,  # Use real SLMs
    use_guardrails=True
)

# Index document
result = await pipeline.index_document(
    document="Your document content here...",
    doc_id="doc1",
    metadata={"source": "manual"}
)

# Query
answer = await pipeline.query(
    query="What is the main topic?",
    top_k=10
)

print(f"Answer: {answer.answer}")
print(f"Confidence: {answer.confidence}")
print(f"Reasoning: {answer.reasoning}")
```

### Custom Graph Construction
```python
from microrag import GraphBuilder, ComponentRegistry

registry = ComponentRegistry()
# ... register components ...

# Build custom computation graph
graph = (
    GraphBuilder(registry)
    .slm("embed_query", "embedder_slm", dependencies=["query"])
    .slm("retrieve", "retriever_slm", dependencies=["embed_query"])
    .slm("rerank", "reranker_slm", dependencies=["query", "retrieve"])
    .slm("compress", "cot_compressor_slm", dependencies=["rerank"])
    .slm("answer", "answer_stabilizer_slm", dependencies=["query", "compress"])
    .build()
)

# Execute graph
results = await graph.execute({"query": "What is X?"})
```

### Specialized SLMs

**1. Chunking SLM**
```python
from microrag.slm import ChunkerSLM

chunker = ChunkerSLM(
    model_name="semantic-chunker-v1",
    max_chunk_size=512
)

chunks = await chunker.process(
    document="Long document content..."
)
```

**2. Retrieval SLM**
```python
from microrag.slm import RetrieverSLM

retriever = RetrieverSLM(
    model_name="facebook/contriever",
    top_k=20
)

results = await retriever.process(
    query="search query",
    chunk=chunks
)
```

**3. Reranking SLM**
```python
from microrag.slm import RerankerSLM

reranker = RerankerSLM(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

reranked = await reranker.process(
    query="query",
    retrieve=retrieval_results
)
```

**4. Chain-of-Thought Compressor**
```python
from microrag.slm import CoTCompressorSLM

compressor = CoTCompressorSLM(
    compression_ratio=0.3
)

compressed = await compressor.process(
    query="query",
    summarize=summarized_docs
)
```

## Guardrails

### Configure Guardrails
```python
from microrag.enterprise import (
    GuardrailEngine,
    RelevanceGuardrail,
    ConfidenceGuardrail,
    SafetyGuardrail
)

guardrails = GuardrailEngine([
    RelevanceGuardrail(min_relevance=0.3),
    ConfidenceGuardrail(min_confidence=0.4),
    SafetyGuardrail(check_toxicity=True)
])

# Check output
result = await guardrails.check(
    step_name="answer",
    query="What is X?",
    output=answer
)

if not result.passed:
    print(f"Guardrail failed: {result.action}")
```

### Custom Guardrails
```python
from microrag.enterprise import BaseGuardrail

class CustomGuardrail(BaseGuardrail):
    async def check(self, step_name, query, output):
        # Your custom logic
        if some_condition:
            return GuardrailResult(
                passed=False,
                action="reject",
                reason="Custom check failed"
            )
        return GuardrailResult(passed=True)

guardrails.add(CustomGuardrail())
```

## Tracing & Monitoring

### LangSmith Integration
```python
from microrag.orchestrator import get_tracer

tracer = get_tracer(
    provider="langsmith",
    api_key="your_key"
)

# Tracing is automatic in pipeline
answer = await pipeline.query(
    query="What is X?",
    trace_id="custom-trace-id"
)
```

### Arize Phoenix
```python
import phoenix as px

# Start Phoenix server
session = px.launch_app()

# Pipeline will auto-trace
pipeline = create_pipeline(use_guardrails=True)
```

### Metrics Collection
```python
from microrag.enterprise import get_metrics

metrics = get_metrics()

# Get summary
summary = metrics.get_summary()
print(f"Total requests: {summary['total_requests']}")
print(f"Success rate: {summary['success_rate']}")
print(f"Avg latency: {summary['avg_latency_ms']}ms")
```

## Model Optimization

### ONNX Conversion
```python
from microrag.slm import optimize_model_to_onnx

# Convert model to ONNX for faster inference
optimized_model = optimize_model_to_onnx(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    output_path="./models/reranker.onnx"
)
```

### Quantization
```python
from optimum.onnxruntime import ORTQuantizer

quantizer = ORTQuantizer.from_pretrained(
    "models/reranker.onnx"
)

quantizer.quantize(
    save_dir="./models/reranker-quantized"
)
```

## Testing
```bash
# Run all tests
pytest

# Test specific component
pytest tests/test_slm.py

# Test graph execution
pytest tests/test_orchestrator.py -v

# Benchmark SLMs
pytest tests/test_slm.py --benchmark-only
```

## Performance Tuning

### 1. Batch Processing
```python
# Process multiple queries in batch
queries = ["query1", "query2", "query3"]
answers = await pipeline.batch_query(queries, batch_size=3)
```

### 2. Model Caching
```python
from microrag.slm import enable_model_cache

enable_model_cache(
    cache_dir="./model_cache",
    max_size_gb=10
)
```

### 3. GPU Optimization
```bash
# Use mixed precision
export TORCH_DTYPE=float16

# Pin memory for faster transfers
export PIN_MEMORY=true
```

## Common Issues

### Issue: Out of memory with SLMs
**Solution**: Reduce batch size or use quantized models
```python
config.batch_size = 8  # Reduce from default
```

### Issue: Slow model loading
**Solution**: Use model caching
```python
enable_model_cache(cache_dir="./models")
```

### Issue: Low answer quality
**Solution**: Tune SLM parameters or use larger models
```python
reranker = RerankerSLM(
    model_name="cross-encoder/ms-marco-TinyBERT-L-6",  # Try different model
    top_k=10
)
```

## Project Structure
```
27-micro-model-orchestrated-rag/
├── src/microrag/
│   ├── pipeline.py          # Main pipeline
│   ├── orchestrator/        # Graph orchestration
│   ├── slm/                # Specialized SLMs
│   ├── enterprise/         # Guardrails & metrics
│   └── ...
├── tests/
├── models/                 # Downloaded models
├── requirements.txt
└── SETUP.md
```

## Next Steps
1. Download and optimize SLMs
2. Configure custom graph
3. Set up guardrails
4. Enable tracing
5. Benchmark performance

## Resources
- [Optimum Documentation](https://huggingface.co/docs/optimum)
- [LangSmith](https://docs.smith.langchain.com/)
- [Arize Phoenix](https://docs.arize.com/phoenix)
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
