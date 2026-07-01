# Project 25: RAG Baseline - Setup Guide

## Overview
Production-ready Retrieval-Augmented Generation (RAG) pipeline with vector stores, embeddings, and multi-tenant support.

## Prerequisites
- Python 3.9+
- 16GB+ RAM recommended
- GPU optional (for faster embeddings)
- OpenAI API key or other LLM provider

## Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### 3. Download NLP Models
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 4. Initialize Vector Store
```bash
# ChromaDB will auto-initialize on first run
# Or manually create directory
mkdir -p data/chroma_db
```

## Configuration

### 1. Environment Variables
Create `.env` file:

```bash
# LLM Provider
OPENAI_API_KEY=your_key_here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Store
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma_db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

### 2. Vector Store Options

**ChromaDB (Recommended for development)**
```python
from ragbaseline import get_vector_store

store = get_vector_store(
    store_type="chroma",
    collection_name="documents",
    persist_directory="./chroma_db"
)
```

**Qdrant (For production)**
```python
store = get_vector_store(
    store_type="qdrant",
    url="http://localhost:6333",
    collection_name="documents"
)
```

## Usage

### Basic RAG Pipeline
```python
from ragbaseline import RAGPipeline, RAGConfig, RAGIndex
from ragbaseline import get_embedding_model, get_llm_provider

# Initialize components
embedding_model = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
llm_provider = get_llm_provider("openai", model="gpt-4")

# Create index
index = RAGIndex(
    embedding_model=embedding_model,
    collection_name="my_docs"
)

# Configure pipeline
config = RAGConfig(
    top_k=5,
    chunk_size=512,
    chunk_overlap=50
)

# Create pipeline
pipeline = RAGPipeline(index, llm_provider, config)

# Index documents
from ragbaseline import Document

doc = Document(
    id="doc1",
    content="Your document content here...",
    metadata={"source": "manual"}
)
index.index_document(doc)

# Query
result = await pipeline.query("What is RAG?")
print(result.answer)
```

### Running the API Server
```bash
# Start server
uvicorn ragbaseline.api:app --host 0.0.0.0 --port 8000 --reload

# Or using the CLI
python -m ragbaseline.api
```

### API Examples

**Ingest Documents**
```bash
curl -X POST "http://localhost:8000/ingest/text" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "default",
    "content": "RAG combines retrieval with generation...",
    "metadata": {"source": "docs"}
  }'
```

**Query**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "tenant_id": "default",
    "top_k": 5
  }'
```

**Search Only**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "tenant_id": "default",
    "top_k": 10
  }'
```

## Document Ingestion

### From Files
```python
from ragbaseline import DocumentIngestion
from pathlib import Path

ingestion = DocumentIngestion()

# PDF
doc = ingestion.ingest(Path("document.pdf"))

# HTML
doc = ingestion.ingest(Path("webpage.html"))

# Markdown
doc = ingestion.ingest(Path("readme.md"))
```

### Batch Ingestion
```bash
# Ingest directory of files
for file in docs/*.pdf; do
  curl -X POST "http://localhost:8000/ingest" \
    -H "Content-Type: application/json" \
    -d "{\"file_path\": \"$file\", \"tenant_id\": \"default\"}"
done
```

## Multi-Tenant Setup

```python
from ragbaseline import TenantManager

manager = TenantManager(base_directory="./data/tenants")

# Each tenant gets isolated index
index_tenant_a = manager.get_tenant_index("tenant_a")
index_tenant_b = manager.get_tenant_index("tenant_b")

# Index documents separately
index_tenant_a.index_document(doc_a)
index_tenant_b.index_document(doc_b)
```

## Testing
```bash
# Run all tests
pytest

# Test specific component
pytest tests/test_retrieval.py

# With coverage
pytest --cov=ragbaseline --cov-report=html
```

## Performance Tuning

### 1. Chunk Size Optimization
```python
# Smaller chunks for precise retrieval
config = RAGConfig(chunk_size=256, chunk_overlap=25)

# Larger chunks for context
config = RAGConfig(chunk_size=1024, chunk_overlap=100)
```

### 2. Embedding Model Selection
```python
# Fast but less accurate
model = get_embedding_model("all-MiniLM-L6-v2")

# Slower but more accurate
model = get_embedding_model("all-mpnet-base-v2")
```

### 3. Caching
Enable caching for repeated queries:
```python
from ragbaseline import CachedEmbeddingModel

model = CachedEmbeddingModel(
    base_model=embedding_model,
    cache_size=1000
)
```

## Common Issues

### Issue: ChromaDB persistence errors
**Solution**: Ensure write permissions on persist directory
```bash
chmod -R 755 ./data/chroma_db
```

### Issue: Out of memory with large documents
**Solution**: Reduce batch size in chunking
```python
chunker = SentenceChunker(max_chunk_size=256)
```

### Issue: Slow embedding generation
**Solution**: Use GPU or smaller model
```bash
pip install sentence-transformers[gpu]
```

## Project Structure
```
25-rag-baseline/
├── src/ragbaseline/
│   ├── pipeline.py        # Main RAG pipeline
│   ├── embeddings.py      # Embedding models
│   ├── vectorstore.py     # Vector stores
│   ├── retrieval.py       # Retrieval logic
│   ├── api.py            # FastAPI app
│   └── ...
├── tests/
├── data/                 # Data directory
├── requirements.txt
└── SETUP.md
```

## Next Steps
1. Ingest your documents
2. Tune chunking strategy
3. Evaluate retrieval quality
4. Set up monitoring
5. Deploy to production

## Resources
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
