# RAG Baseline System

## Executive Summary

Production-ready Retrieval-Augmented Generation baseline system with document ingestion, embedding-based retrieval, and LLM generation. Features multi-format support (PDF, HTML, Markdown), configurable chunking strategies, vector search with metadata filtering, and streaming API responses. Designed for enterprise deployment with per-tenant isolation and comprehensive logging.

> **Concepts covered:** [§04 RAG systems](../../04-ai-engineering/02-llm-applications/rag/rag-systems.md) · [§04 Vector stores](../../04-ai-engineering/03-vector-databases/vector-stores/vector-stores.md) · [§04 Embeddings](../../04-ai-engineering/03-vector-databases/embeddings/embeddings.md). See also the [concept-to-project map](../CONCEPT_TO_PROJECT_MAP.md).

## System Architecture

```
+------------------------------------------------------------------+
|                       RAG Baseline System                         |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +-------------------+    +-------------+ |
|  | Ingestion        |    | Retrieval         |    | Generation  | |
|  |------------------|    |-------------------|    |-------------| |
|  | - PDF Parser     |    | - Query Embed     |    | - Prompt    | |
|  | - HTML Parser    |    | - Vector Search   |    | - LLM Call  | |
|  | - Markdown       |    | - BM25 (hybrid)   |    | - Stream    | |
|  | - Chunking       |    | - Reranking       |    | - Format    | |
|  +------------------+    +-------------------+    +-------------+ |
|           |                       |                      |        |
|           v                       v                      v        |
|  +------------------------------------------------------------------+
|  |                      Embedding & Index                          |
|  |----------------------------------------------------------------|
|  | Models: BGE / MiniLM / OpenAI | Vector DB: Chroma / Qdrant     |
|  +------------------------------------------------------------------+
|           |                                                        |
|           v                                                        |
|  +------------------------------------------------------------------+
|  |                    Enterprise Infrastructure                    |
|  |----------------------------------------------------------------|
|  | Per-Tenant Collections | Retrieval Logs | API Rate Limiting    |
|  +------------------------------------------------------------------+
|                                                                   |
+------------------------------------------------------------------+
```

## Core Components

### 1. Document Ingestion Pipeline

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
import hashlib

@dataclass
class Document:
    """Ingested document."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""

@dataclass
class Chunk:
    """Document chunk for embedding."""
    id: str
    content: str
    document_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class DocumentParser(ABC):
    """Base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """Parse file into document."""
        pass

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """Check if parser supports file type."""
        pass


class PDFParser(DocumentParser):
    """Parse PDF documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> Document:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        text_parts = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            text_parts.append(text)

        content = "\n\n".join(text_parts)

        return Document(
            id=self._generate_id(file_path),
            content=content,
            metadata={
                "filename": file_path.name,
                "num_pages": len(doc),
                "file_type": "pdf",
            },
            source=str(file_path),
        )

    def _generate_id(self, file_path: Path) -> str:
        return hashlib.sha256(str(file_path).encode()).hexdigest()[:16]


class HTMLParser(DocumentParser):
    """Parse HTML documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".html", ".htm"]

    def parse(self, file_path: Path) -> Document:
        from bs4 import BeautifulSoup

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text
        text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        content = "\n".join(line for line in lines if line)

        # Extract title
        title = soup.title.string if soup.title else file_path.stem

        return Document(
            id=self._generate_id(file_path),
            content=content,
            metadata={
                "filename": file_path.name,
                "title": title,
                "file_type": "html",
            },
            source=str(file_path),
        )

    def _generate_id(self, file_path: Path) -> str:
        return hashlib.sha256(str(file_path).encode()).hexdigest()[:16]


class MarkdownParser(DocumentParser):
    """Parse Markdown documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".md", ".markdown"]

    def parse(self, file_path: Path) -> Document:
        import markdown
        from bs4 import BeautifulSoup

        with open(file_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert to HTML then extract text
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, "html.parser")
        content = soup.get_text(separator="\n")

        # Extract title from first heading
        title = file_path.stem
        lines = md_content.split("\n")
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return Document(
            id=self._generate_id(file_path),
            content=content,
            metadata={
                "filename": file_path.name,
                "title": title,
                "file_type": "markdown",
            },
            source=str(file_path),
        )

    def _generate_id(self, file_path: Path) -> str:
        return hashlib.sha256(str(file_path).encode()).hexdigest()[:16]


class DocumentIngestion:
    """Main ingestion pipeline."""

    def __init__(self):
        self.parsers = [
            PDFParser(),
            HTMLParser(),
            MarkdownParser(),
        ]

    def ingest(self, file_path: Path) -> Document:
        """Ingest single file."""
        for parser in self.parsers:
            if parser.supports(file_path):
                return parser.parse(file_path)

        raise ValueError(f"No parser found for {file_path}")

    def ingest_directory(self, dir_path: Path) -> Iterator[Document]:
        """Ingest all supported files in directory."""
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                try:
                    yield self.ingest(file_path)
                except ValueError:
                    continue  # Skip unsupported files
```

### 2. Chunking Strategies

```python
from typing import Callable

class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into chunks."""
        pass


class FixedSizeChunker(ChunkingStrategy):
    """Fixed size chunks with overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append(Chunk(
                id=f"{document.id}_{chunk_index}",
                content=chunk_text,
                document_id=document.id,
                chunk_index=chunk_index,
                metadata={
                    **document.metadata,
                    "start_char": start,
                    "end_char": end,
                },
            ))

            start = end - self.chunk_overlap
            chunk_index += 1

        return chunks


class SentenceChunker(ChunkingStrategy):
    """Chunk by sentences with size limit."""

    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        import nltk
        nltk.download('punkt', quiet=True)

        sentences = nltk.sent_tokenize(document.content)
        chunks = []

        current_chunk = []
        current_size = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    id=f"{document.id}_{chunk_index}",
                    content=chunk_text,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata=document.metadata,
                ))

                current_chunk = []
                current_size = 0
                chunk_index += 1

            current_chunk.append(sentence)
            current_size += sentence_size

        # Don't forget last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    id=f"{document.id}_{chunk_index}",
                    content=chunk_text,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata=document.metadata,
                ))

        return chunks


class SemanticChunker(ChunkingStrategy):
    """Chunk by semantic similarity (paragraph boundaries)."""

    def __init__(
        self,
        embedding_model: "EmbeddingModel",
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 1000,
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        # Split by paragraphs
        paragraphs = document.content.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        # Embed paragraphs
        embeddings = self.embedding_model.encode(paragraphs)

        # Group by similarity
        chunks = []
        current_group = [paragraphs[0]]
        current_size = len(paragraphs[0])
        chunk_index = 0

        for i in range(1, len(paragraphs)):
            # Compute similarity with previous
            similarity = self._cosine_similarity(
                embeddings[i-1],
                embeddings[i]
            )

            # Check if should merge
            if (similarity > self.similarity_threshold and
                current_size + len(paragraphs[i]) < self.max_chunk_size):
                current_group.append(paragraphs[i])
                current_size += len(paragraphs[i])
            else:
                # Save current chunk
                chunks.append(Chunk(
                    id=f"{document.id}_{chunk_index}",
                    content="\n\n".join(current_group),
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata=document.metadata,
                ))

                current_group = [paragraphs[i]]
                current_size = len(paragraphs[i])
                chunk_index += 1

        # Last chunk
        if current_group:
            chunks.append(Chunk(
                id=f"{document.id}_{chunk_index}",
                content="\n\n".join(current_group),
                document_id=document.id,
                chunk_index=chunk_index,
                metadata=document.metadata,
            ))

        return chunks

    def _cosine_similarity(self, a, b):
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### 3. Embedding and Indexing

```python
import numpy as np
from typing import Protocol

class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        ...


class SentenceTransformerEmbedding:
    """Embedding using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class OpenAIEmbedding:
    """Embedding using OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small"):
        import openai
        self.client = openai.OpenAI()
        self.model = model
        self.dimension = 1536 if "small" in model else 3072

    def encode(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)


class VectorStore(Protocol):
    """Protocol for vector stores."""

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
    ): ...

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter: dict = None,
    ) -> list[dict]: ...


class ChromaVectorStore:
    """Vector store using Chroma."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
    ):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
        ))

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
    ):
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter: dict = None,
    ) -> list[dict]:
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            where=filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to score
            })

        return formatted

    def delete(self, ids: list[str]):
        self.collection.delete(ids=ids)


class RAGIndex:
    """Main index combining embedding and vector store."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunker: ChunkingStrategy,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunker = chunker

    def index_document(self, document: Document):
        """Index a single document."""
        chunks = self.chunker.chunk(document)

        if not chunks:
            return

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        # Add to vector store
        self.vector_store.add(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def index_documents(self, documents: list[Document]):
        """Index multiple documents."""
        for doc in documents:
            self.index_document(doc)

    def search(
        self,
        query: str,
        k: int = 5,
        filter: dict = None,
    ) -> list[dict]:
        """Search for relevant chunks."""
        query_embedding = self.embedding_model.encode([query])
        return self.vector_store.search(query_embedding, k, filter)
```

### 4. Query Pipeline

```python
from dataclasses import dataclass
from typing import AsyncIterator

@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    sources: list[dict]
    query: str

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    # Retrieval
    top_k: int = 5
    rerank: bool = False
    rerank_top_k: int = 3

    # Generation
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False

    # Prompt
    system_prompt: str = """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain enough information to answer, say so.
Always cite your sources by referencing the document titles or filenames."""


class RAGPipeline:
    """Main RAG query pipeline."""

    def __init__(
        self,
        index: RAGIndex,
        llm_provider: "LLMProvider",
        config: RAGConfig = None,
    ):
        self.index = index
        self.llm = llm_provider
        self.config = config or RAGConfig()

    async def query(
        self,
        question: str,
        filter: dict = None,
    ) -> RAGResponse:
        """Execute RAG query."""
        # 1. Retrieve relevant chunks
        results = self.index.search(
            question,
            k=self.config.top_k,
            filter=filter,
        )

        # 2. Rerank if enabled
        if self.config.rerank and len(results) > self.config.rerank_top_k:
            results = await self._rerank(question, results)
            results = results[:self.config.rerank_top_k]

        # 3. Build prompt with context
        context = self._build_context(results)
        messages = self._build_messages(question, context)

        # 4. Generate answer
        answer = await self.llm.generate(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return RAGResponse(
            answer=answer,
            sources=results,
            query=question,
        )

    async def query_stream(
        self,
        question: str,
        filter: dict = None,
    ) -> AsyncIterator[str]:
        """Execute RAG query with streaming response."""
        # Retrieve
        results = self.index.search(
            question,
            k=self.config.top_k,
            filter=filter,
        )

        # Build prompt
        context = self._build_context(results)
        messages = self._build_messages(question, context)

        # Stream generation
        async for chunk in self.llm.generate_stream(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ):
            yield chunk

    def _build_context(self, results: list[dict]) -> str:
        """Build context string from search results."""
        context_parts = []

        for i, result in enumerate(results, 1):
            source = result["metadata"].get("filename", "Unknown")
            text = result["document"]
            context_parts.append(f"[{i}] {source}:\n{text}")

        return "\n\n".join(context_parts)

    def _build_messages(self, question: str, context: str) -> list[dict]:
        """Build messages for LLM."""
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": f"""Context:
{context}

Question: {question}

Please answer the question based on the context provided above."""},
        ]

    async def _rerank(
        self,
        query: str,
        results: list[dict],
    ) -> list[dict]:
        """Rerank results using cross-encoder."""
        from sentence_transformers import CrossEncoder

        # Load reranker
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Score pairs
        pairs = [[query, r["document"]] for r in results]
        scores = reranker.predict(pairs)

        # Sort by score
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [r for r, s in scored_results]


class LLMProvider:
    """LLM provider for generation."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        import openai
        self.client = openai.AsyncOpenAI()
        self.model = model

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

## Enterprise Features

### Per-Tenant Collections

```python
class TenantManager:
    """Manage per-tenant document collections."""

    def __init__(self, base_directory: str = "./tenants"):
        self.base_directory = Path(base_directory)
        self.tenants: dict[str, RAGIndex] = {}

    def get_tenant_index(
        self,
        tenant_id: str,
        embedding_model: EmbeddingModel = None,
    ) -> RAGIndex:
        """Get or create tenant-specific index."""
        if tenant_id not in self.tenants:
            # Create tenant directory
            tenant_dir = self.base_directory / tenant_id
            tenant_dir.mkdir(parents=True, exist_ok=True)

            # Create index
            vector_store = ChromaVectorStore(
                collection_name=f"tenant_{tenant_id}",
                persist_directory=str(tenant_dir / "chroma"),
            )

            embedding_model = embedding_model or SentenceTransformerEmbedding()
            chunker = SentenceChunker()

            self.tenants[tenant_id] = RAGIndex(
                embedding_model=embedding_model,
                vector_store=vector_store,
                chunker=chunker,
            )

        return self.tenants[tenant_id]

    def delete_tenant(self, tenant_id: str):
        """Delete tenant and all data."""
        if tenant_id in self.tenants:
            del self.tenants[tenant_id]

        tenant_dir = self.base_directory / tenant_id
        if tenant_dir.exists():
            import shutil
            shutil.rmtree(tenant_dir)
```

### Retrieval Logging

```python
import json
from datetime import datetime

class RetrievalLogger:
    """Log retrieval queries and results for analysis."""

    def __init__(self, log_file: str = "retrieval_logs.jsonl"):
        self.log_file = log_file

    def log_query(
        self,
        query: str,
        results: list[dict],
        response: str,
        tenant_id: str = None,
        user_id: str = None,
        latency_ms: float = None,
    ):
        """Log a retrieval query."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "num_results": len(results),
            "result_ids": [r["id"] for r in results],
            "result_scores": [r.get("score", 0) for r in results],
            "response_length": len(response),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "latency_ms": latency_ms,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def analyze_logs(self) -> dict:
        """Analyze retrieval logs."""
        queries = []
        latencies = []

        with open(self.log_file) as f:
            for line in f:
                entry = json.loads(line)
                queries.append(entry["query"])
                if entry.get("latency_ms"):
                    latencies.append(entry["latency_ms"])

        return {
            "total_queries": len(queries),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
        }
```

### Streaming API

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="RAG Baseline API")

class QueryRequest(BaseModel):
    question: str
    tenant_id: str = "default"
    top_k: int = 5
    filter: dict | None = None
    stream: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute RAG query."""
    # Get tenant index
    index = tenant_manager.get_tenant_index(request.tenant_id)

    # Execute query
    pipeline = RAGPipeline(index, llm_provider, config)

    start_time = time.time()
    result = await pipeline.query(
        request.question,
        filter=request.filter,
    )
    latency_ms = (time.time() - start_time) * 1000

    # Log query
    retrieval_logger.log_query(
        query=request.question,
        results=result.sources,
        response=result.answer,
        tenant_id=request.tenant_id,
        latency_ms=latency_ms,
    )

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
    )

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Execute RAG query with streaming response."""
    index = tenant_manager.get_tenant_index(request.tenant_id)
    pipeline = RAGPipeline(index, llm_provider, config)

    async def generate():
        async for chunk in pipeline.query_stream(
            request.question,
            filter=request.filter,
        ):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )

class IngestRequest(BaseModel):
    tenant_id: str = "default"
    file_path: str

@app.post("/ingest")
async def ingest_document(request: IngestRequest):
    """Ingest document into index."""
    index = tenant_manager.get_tenant_index(request.tenant_id)

    ingestion = DocumentIngestion()
    document = ingestion.ingest(Path(request.file_path))
    index.index_document(document)

    return {"status": "success", "document_id": document.id}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
```

## Implementation Phases

### Phase 1: Core Pipeline (Weeks 1-2)
- [ ] Document parsers (PDF, HTML, Markdown)
- [ ] Basic chunking strategies
- [ ] Embedding models (BGE, MiniLM)
- [ ] Vector store (Chroma)
- [ ] Basic search

### Phase 2: Query Pipeline (Weeks 3-4)
- [ ] RAG pipeline
- [ ] Prompt templates
- [ ] LLM generation
- [ ] Response formatting
- [ ] Source citation

### Phase 3: Advanced Features (Weeks 5-6)
- [ ] Semantic chunking
- [ ] Reranking
- [ ] Hybrid search (BM25)
- [ ] Metadata filtering

### Phase 4: Enterprise (Weeks 7-8)
- [ ] Per-tenant collections
- [ ] Streaming responses
- [ ] Retrieval logging
- [ ] API rate limiting

### Phase 5: Production (Weeks 9-10)
- [ ] FastAPI service
- [ ] Docker deployment
- [ ] Documentation
- [ ] Examples

## Testing Strategy

```python
import pytest

class TestRAGPipeline:
    @pytest.fixture
    def rag_system(self):
        embedding = SentenceTransformerEmbedding()
        vector_store = ChromaVectorStore(collection_name="test")
        chunker = SentenceChunker()
        index = RAGIndex(embedding, vector_store, chunker)
        llm = LLMProvider()
        return RAGPipeline(index, llm)

    def test_ingestion(self, rag_system):
        doc = Document(
            id="test",
            content="Python is a programming language.",
            metadata={"filename": "test.txt"},
        )
        rag_system.index.index_document(doc)

        results = rag_system.index.search("What is Python?")
        assert len(results) > 0
        assert "programming" in results[0]["document"].lower()

    @pytest.mark.asyncio
    async def test_query(self, rag_system):
        # Index document
        doc = Document(
            id="test",
            content="The capital of France is Paris.",
            metadata={"filename": "geo.txt"},
        )
        rag_system.index.index_document(doc)

        # Query
        result = await rag_system.query("What is the capital of France?")

        assert "Paris" in result.answer
        assert len(result.sources) > 0
```

## Stretch Goals

### Hybrid Retrieval with BM25

```python
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """Combine vector search with BM25."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        alpha: float = 0.5,  # Weight for vector vs BM25
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.bm25 = None
        self.documents = []

    def add_documents(self, documents: list[str], ids: list[str]):
        """Add documents to both indices."""
        # BM25
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.documents = list(zip(ids, documents))

        # Vector
        embeddings = self.embedding_model.encode(documents)
        self.vector_store.add(ids, embeddings, documents, [{}] * len(documents))

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Hybrid search combining vector and BM25."""
        # Vector search
        query_emb = self.embedding_model.encode([query])
        vector_results = self.vector_store.search(query_emb, k * 2)

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_results = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:k * 2]

        # Combine scores
        combined = {}
        for result in vector_results:
            combined[result["id"]] = {
                "vector_score": result["score"],
                "bm25_score": 0,
                "document": result["document"],
            }

        for idx, score in bm25_results:
            doc_id = self.documents[idx][0]
            if doc_id in combined:
                combined[doc_id]["bm25_score"] = score
            else:
                combined[doc_id] = {
                    "vector_score": 0,
                    "bm25_score": score,
                    "document": self.documents[idx][1],
                }

        # Compute final scores
        results = []
        for doc_id, scores in combined.items():
            final_score = (
                self.alpha * scores["vector_score"] +
                (1 - self.alpha) * scores["bm25_score"]
            )
            results.append({
                "id": doc_id,
                "document": scores["document"],
                "score": final_score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
```

### Metadata Filters

```python
# Example filter usage
results = await pipeline.query(
    "What are the pricing options?",
    filter={
        "file_type": "pdf",
        "date": {"$gte": "2024-01-01"},
        "category": {"$in": ["pricing", "sales"]},
    }
)
```

## References

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models](https://arxiv.org/abs/2104.08663)
- [Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Sentence-Transformers](https://www.sbert.net/)
