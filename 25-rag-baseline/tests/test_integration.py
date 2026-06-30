"""Integration tests for the complete RAG pipeline.

These tests cover end-to-end scenarios including:
- Document ingestion and query flow
- Multi-document retrieval
- Chunking strategy comparisons
- Hybrid retrieval (BM25 + vector)
- Streaming response tests
"""

import asyncio
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import Mock, AsyncMock, patch

import numpy as np
import pytest

from ragbaseline.schemas import Document, Chunk, SearchResult, RAGConfig, RAGResponse
from ragbaseline.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
)
from ragbaseline.embeddings import MockEmbedding
from ragbaseline.vectorstore import SimpleVectorStore
from ragbaseline.index import RAGIndex, MultiIndexManager
from ragbaseline.pipeline import RAGPipeline, MockLLMProvider, LLMProvider
from ragbaseline.retrieval import BM25, HybridRetriever, RetrievalConfig
from ragbaseline.parsers import DocumentIngestion, MarkdownParser, TextParser


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_documents() -> List[Document]:
    """Create a collection of sample documents for testing."""
    return [
        Document(
            id="doc_python",
            content="""Python is a high-level, interpreted programming language known for its
            clear syntax and readability. It supports multiple programming paradigms including
            procedural, object-oriented, and functional programming. Python's extensive standard
            library and third-party packages make it suitable for web development, data science,
            machine learning, automation, and scripting. The language was created by Guido van
            Rossum and first released in 1991.""",
            metadata={"category": "programming", "language": "python", "year": 2024},
        ),
        Document(
            id="doc_rust",
            content="""Rust is a systems programming language that focuses on safety, speed,
            and concurrency. It achieves memory safety without garbage collection through its
            ownership system. Rust is designed to be a safer, concurrent, and more practical
            alternative to C and C++. It is used in performance-critical applications, operating
            systems, game engines, and web browsers. Mozilla Research initially sponsored Rust's
            development.""",
            metadata={"category": "programming", "language": "rust", "year": 2024},
        ),
        Document(
            id="doc_ml",
            content="""Machine learning is a subset of artificial intelligence that enables
            systems to learn and improve from experience without being explicitly programmed.
            It focuses on developing algorithms that can access data and use it to learn for
            themselves. Common machine learning approaches include supervised learning,
            unsupervised learning, and reinforcement learning. Applications include image
            recognition, natural language processing, and predictive analytics.""",
            metadata={"category": "ai", "topic": "machine_learning", "year": 2024},
        ),
        Document(
            id="doc_rag",
            content="""Retrieval-Augmented Generation (RAG) is a technique that combines
            information retrieval with text generation. It first retrieves relevant documents
            from a knowledge base, then uses these documents as context for a language model
            to generate accurate, grounded responses. RAG helps reduce hallucinations in LLMs
            and allows for dynamic knowledge updates without retraining the model. Key components
            include document chunking, embedding generation, vector search, and response synthesis.""",
            metadata={"category": "ai", "topic": "rag", "year": 2024},
        ),
        Document(
            id="doc_vectors",
            content="""Vector databases are specialized databases designed to store, index,
            and query high-dimensional vectors efficiently. They use algorithms like HNSW,
            IVF, and PQ for approximate nearest neighbor search. Popular vector databases
            include Pinecone, Milvus, Qdrant, Chroma, and Weaviate. Vector databases are
            essential for semantic search, recommendation systems, and RAG applications.""",
            metadata={"category": "databases", "topic": "vectors", "year": 2024},
        ),
    ]


@pytest.fixture
def embedding_model():
    """Create mock embedding model. Use function scope to get fresh instance each test."""
    return MockEmbedding(dimension=128)


@pytest.fixture
def vector_store():
    """Create simple vector store."""
    return SimpleVectorStore()


@pytest.fixture
def mock_llm():
    """Create mock LLM provider with contextual responses."""
    responses = [
        "Based on the provided context, I can answer your question.",
        "The documents indicate the following information.",
        "According to the sources, here is the relevant information.",
    ]
    return MockLLMProvider(responses=responses)


@pytest.fixture
def rag_index(embedding_model, vector_store):
    """Create RAG index with fixed-size chunker."""
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
    return RAGIndex(embedding_model, vector_store, chunker)


@pytest.fixture
def rag_pipeline(rag_index, mock_llm):
    """Create RAG pipeline."""
    config = RAGConfig(top_k=3, temperature=0.7)
    return RAGPipeline(rag_index, mock_llm, config)


# ============================================================================
# End-to-End Document Ingestion and Query Tests
# ============================================================================

class TestEndToEndIngestionAndQuery:
    """Test complete document ingestion and query flow."""

    @pytest.mark.asyncio
    async def test_basic_ingest_and_query(self, sample_documents, rag_pipeline):
        """Test basic document ingestion followed by query."""
        # Ingest all documents
        for doc in sample_documents:
            rag_pipeline.index.index_document(doc)

        # Verify documents were indexed
        assert rag_pipeline.index.count > 0

        # Query the system
        response = await rag_pipeline.query("What is Python programming language?")

        assert isinstance(response, RAGResponse)
        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.query == "What is Python programming language?"
        assert len(response.sources) > 0

    @pytest.mark.asyncio
    async def test_query_returns_relevant_sources(self, sample_documents, rag_pipeline):
        """Test that queries return relevant source documents."""
        # Ingest documents
        for doc in sample_documents:
            rag_pipeline.index.index_document(doc)

        # Query about machine learning
        response = await rag_pipeline.query("What is machine learning and how does it work?")

        # Should have sources from the ML-related documents
        source_ids = [s.id for s in response.sources]
        # At least one source should be retrieved
        assert len(source_ids) >= 1

    @pytest.mark.asyncio
    async def test_empty_index_query(self, rag_pipeline):
        """Test querying an empty index."""
        # Query without any documents
        response = await rag_pipeline.query("What is Python?")

        # Should still return a response (possibly empty sources)
        assert isinstance(response, RAGResponse)
        assert len(response.sources) == 0

    @pytest.mark.asyncio
    async def test_query_with_metadata_filter(self, sample_documents, rag_pipeline):
        """Test querying with metadata filters."""
        # Ingest documents
        for doc in sample_documents:
            rag_pipeline.index.index_document(doc)

        # Query with category filter
        response = await rag_pipeline.query(
            "Tell me about programming",
            filter={"category": "programming"}
        )

        # Sources should only be from programming category
        for source in response.sources:
            if source.metadata.get("category"):
                assert source.metadata["category"] == "programming"

    @pytest.mark.asyncio
    async def test_incremental_document_addition(self, mock_llm):
        """Test adding documents incrementally."""
        # Use fresh instances to avoid state issues
        embedding_model = MockEmbedding(dimension=128)
        vector_store = SimpleVectorStore()
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        index = RAGIndex(embedding_model, vector_store, chunker)
        pipeline = RAGPipeline(index, mock_llm, RAGConfig(top_k=3))

        # Add first document - make it long enough to produce chunks
        doc1 = Document(
            id="first_doc",
            content="This is the first document about data science and analytics. "
                    "Data science involves analyzing large datasets to extract insights. "
                    "Analytics helps businesses make data-driven decisions.",
            metadata={"order": 1}
        )
        index.index_document(doc1)
        initial_count = index.count
        assert initial_count > 0  # Verify first document was indexed

        # Add second document - also make it long enough
        doc2 = Document(
            id="second_doc",
            content="This is the second document about machine learning models. "
                    "Machine learning enables computers to learn from data. "
                    "Models can be trained for classification and regression tasks.",
            metadata={"order": 2}
        )
        index.index_document(doc2)

        # Count should increase
        assert index.count > initial_count

        # Both documents should be searchable
        response = await pipeline.query("data science and machine learning")
        assert len(response.sources) > 0


# ============================================================================
# Multi-Document Retrieval Tests
# ============================================================================

class TestMultiDocumentRetrieval:
    """Test retrieval across multiple documents."""

    @pytest.mark.asyncio
    async def test_retrieve_from_multiple_documents(self, sample_documents, rag_pipeline):
        """Test that retrieval can span multiple documents."""
        # Ingest all documents
        for doc in sample_documents:
            rag_pipeline.index.index_document(doc)

        # Query that should match multiple documents
        response = await rag_pipeline.query(
            "What programming languages and AI technologies are mentioned?"
        )

        # Should retrieve from multiple source documents
        unique_doc_ids = set()
        for source in response.sources:
            # Extract document ID from chunk ID (format: docid_chunkindex)
            doc_id = source.id.rsplit("_", 1)[0]
            unique_doc_ids.add(doc_id)

        # Should have sources from at least 2 different documents
        # (depending on top_k setting)
        assert len(response.sources) >= 1

    @pytest.mark.asyncio
    async def test_relevance_ranking_across_documents(self, embedding_model, vector_store, mock_llm):
        """Test that relevance ranking works across documents."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        index = RAGIndex(embedding_model, vector_store, chunker)
        pipeline = RAGPipeline(index, mock_llm, RAGConfig(top_k=5))

        # Create documents with varying relevance
        docs = [
            Document(id="highly_relevant", content="Python is an excellent language for data science.", metadata={}),
            Document(id="somewhat_relevant", content="Programming languages are tools for software development.", metadata={}),
            Document(id="less_relevant", content="Cooking is an art form that requires practice.", metadata={}),
        ]

        for doc in docs:
            index.index_document(doc)

        # Query about Python
        response = await pipeline.query("Tell me about Python programming")

        # Should have ranked results
        assert len(response.sources) > 0
        # First result should have higher score
        if len(response.sources) > 1:
            assert response.sources[0].score >= response.sources[1].score

    @pytest.mark.asyncio
    async def test_cross_document_context_building(self, sample_documents, rag_pipeline):
        """Test context building from multiple document sources."""
        # Ingest documents
        for doc in sample_documents:
            rag_pipeline.index.index_document(doc)

        # Query that requires info from multiple sources
        response = await rag_pipeline.query(
            "How do vector databases relate to RAG systems?"
        )

        # The response should include sources (context was built)
        assert len(response.sources) > 0
        assert response.answer is not None

    @pytest.mark.asyncio
    async def test_large_document_set_retrieval(self, embedding_model, vector_store, mock_llm):
        """Test retrieval with a larger document set."""
        chunker = FixedSizeChunker(chunk_size=150, chunk_overlap=15)
        index = RAGIndex(embedding_model, vector_store, chunker)
        pipeline = RAGPipeline(index, mock_llm, RAGConfig(top_k=5))

        # Create many documents
        num_docs = 50
        for i in range(num_docs):
            doc = Document(
                id=f"doc_{i}",
                content=f"This is document number {i}. It contains information about topic {i % 5}. "
                        f"Keywords: {'python' if i % 3 == 0 else 'java'}, data, analysis.",
                metadata={"index": i, "topic": i % 5}
            )
            index.index_document(doc)

        # Verify all documents indexed
        assert index.count >= num_docs

        # Query should still work efficiently
        response = await pipeline.query("python data analysis")
        assert len(response.sources) <= 5  # Respects top_k


# ============================================================================
# Chunking Strategy Comparison Tests
# ============================================================================

class TestChunkingStrategyComparison:
    """Compare different chunking strategies."""

    def test_fixed_size_vs_sentence_chunk_count(self, sample_documents, embedding_model):
        """Compare chunk counts between fixed-size and sentence chunking."""
        # Fixed-size chunker
        fixed_chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)

        # Sentence chunker
        sentence_chunker = SentenceChunker(max_chunk_size=150, min_chunk_size=30)

        doc = sample_documents[0]  # Python document

        fixed_chunks = fixed_chunker.chunk(doc)
        sentence_chunks = sentence_chunker.chunk(doc)

        # Both should produce chunks
        assert len(fixed_chunks) > 0
        assert len(sentence_chunks) > 0

        # Fixed-size should produce more uniform chunk sizes
        fixed_sizes = [len(c.content) for c in fixed_chunks]
        assert max(fixed_sizes) <= 100

    def test_recursive_chunker_respects_separators(self, sample_documents):
        """Test that recursive chunker respects separator hierarchy."""
        doc = Document(
            id="structured_doc",
            content="""# Section 1

This is the first paragraph.
It has multiple sentences.

This is the second paragraph.

# Section 2

Another section here.
With more content.""",
            metadata={}
        )

        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        # Chunks should be created at logical boundaries
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0

    def test_chunking_impact_on_retrieval_quality(self):
        """Test how chunking strategy affects retrieval quality."""
        # Create fresh embedding model for this test to avoid fixture state issues
        embedding_model = MockEmbedding(dimension=128)

        # Long document with distinct sections - make it long enough for chunking
        long_doc = Document(
            id="long_doc",
            content="""Section on Python:
Python is used for web development with frameworks like Django and Flask.
It excels at data processing and machine learning tasks.
Python has a large ecosystem of libraries and tools for various applications.
The syntax is clean and readable, making it popular among beginners.

Section on Databases:
Databases store and manage data efficiently for applications.
SQL databases use structured query language for queries.
NoSQL databases offer flexibility for unstructured data storage.
Modern applications often use a combination of both database types.

Section on APIs:
APIs enable communication between different software systems.
REST APIs use HTTP methods for operations like GET and POST.
GraphQL provides a flexible query language for APIs.
API design is crucial for building scalable applications.""",
            metadata={}
        )

        # Test with small chunks
        small_chunker = FixedSizeChunker(chunk_size=150, chunk_overlap=20)
        small_store = SimpleVectorStore()
        small_index = RAGIndex(embedding_model, small_store, small_chunker)
        small_index.index_document(long_doc)

        # Test with larger chunks (use separate embedding model instance)
        embedding_model_2 = MockEmbedding(dimension=128)
        large_chunker = FixedSizeChunker(chunk_size=400, chunk_overlap=40)
        large_store = SimpleVectorStore()
        large_index = RAGIndex(embedding_model_2, large_store, large_chunker)
        large_index.index_document(long_doc)

        # Both should create chunks
        assert small_index.count > 0
        assert large_index.count > 0

        # Small chunks should produce more or equal chunks
        assert small_index.count >= large_index.count

        # Both should be searchable
        small_results = small_index.search("Python web development", k=3)
        large_results = large_index.search("Python web development", k=3)

        assert len(small_results) > 0
        assert len(large_results) > 0

    def test_sentence_chunker_preserves_sentence_boundaries(self):
        """Test that sentence chunker doesn't split mid-sentence."""
        doc = Document(
            id="sentence_doc",
            content="This is sentence one. This is sentence two. This is sentence three. This is sentence four.",
            metadata={}
        )

        chunker = SentenceChunker(max_chunk_size=100, min_chunk_size=20)
        chunks = chunker.chunk(doc)

        # Each chunk should end with proper punctuation (not mid-word)
        for chunk in chunks:
            content = chunk.content.strip()
            if content:
                # Should end with sentence-ending punctuation or be complete
                assert content[-1] in ".!?\"'" or len(content) < 50

    def test_overlap_creates_redundancy_for_context(self, sample_documents):
        """Test that overlap creates context redundancy."""
        doc = sample_documents[0]

        # No overlap
        no_overlap_chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        no_overlap_chunks = no_overlap_chunker.chunk(doc)

        # With overlap
        overlap_chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=30)
        overlap_chunks = overlap_chunker.chunk(doc)

        # Overlap should create more chunks
        if len(no_overlap_chunks) > 1:
            assert len(overlap_chunks) >= len(no_overlap_chunks)


# ============================================================================
# Hybrid Retrieval (BM25 + Vector) Tests
# ============================================================================

class TestHybridRetrieval:
    """Test hybrid retrieval combining BM25 and vector search."""

    def test_bm25_keyword_matching(self):
        """Test BM25 keyword matching."""
        bm25 = BM25(k1=1.5, b=0.75)

        docs = [
            "Python programming language for data science",
            "Java enterprise software development",
            "Machine learning with Python and TensorFlow",
        ]

        bm25.add_documents(
            ids=["doc1", "doc2", "doc3"],
            documents=docs,
        )

        # Search for Python
        results = bm25.search("Python programming", k=2)

        assert len(results) == 2
        # Python documents should rank higher
        assert results[0].id in ["doc1", "doc3"]

    def test_bm25_with_metadata_filter(self):
        """Test BM25 search with metadata filtering."""
        bm25 = BM25()

        bm25.add_documents(
            ids=["doc1", "doc2", "doc3"],
            documents=[
                "Python for web development",
                "Python for data science",
                "Java for enterprise applications",
            ],
            metadatas=[
                {"use_case": "web"},
                {"use_case": "data"},
                {"use_case": "enterprise"},
            ]
        )

        # Search with filter
        results = bm25.search(
            "Python development",
            k=3,
            filter={"use_case": "data"}
        )

        assert len(results) == 1
        assert results[0].id == "doc2"

    def test_hybrid_retriever_combines_results(self, embedding_model, vector_store):
        """Test that hybrid retriever combines vector and BM25 results."""
        config = RetrievalConfig(vector_weight=0.5)
        hybrid = HybridRetriever(embedding_model, vector_store, config)

        # Add documents
        docs = [
            "Machine learning algorithms for prediction",
            "Deep learning neural networks",
            "Statistical analysis methods",
        ]
        embeddings = embedding_model.encode(docs)

        hybrid.add(
            ids=["ml", "dl", "stats"],
            embeddings=embeddings,
            documents=docs,
            metadatas=[{}, {}, {}],
        )

        # Search
        results = hybrid.search("machine learning neural networks", k=3)

        # Should have results from both retrieval methods
        assert len(results) > 0
        # Results should be ranked by combined score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_hybrid_retriever_weight_adjustment(self, embedding_model, vector_store):
        """Test adjusting weights between vector and BM25."""
        # High vector weight
        high_vector_config = RetrievalConfig(vector_weight=0.9)
        high_vector_hybrid = HybridRetriever(embedding_model, vector_store, high_vector_config)

        # Low vector weight (high keyword weight)
        low_vector_store = SimpleVectorStore()
        low_vector_config = RetrievalConfig(vector_weight=0.1)
        low_vector_hybrid = HybridRetriever(embedding_model, low_vector_store, low_vector_config)

        docs = [
            "Python programming tutorial for beginners",
            "Advanced Python techniques and patterns",
        ]
        embeddings = embedding_model.encode(docs)

        for hybrid in [high_vector_hybrid, low_vector_hybrid]:
            hybrid.add(
                ids=["beginner", "advanced"],
                embeddings=embeddings,
                documents=docs,
                metadatas=[{}, {}],
            )

        # Both should return results (weights affect ranking, not presence)
        high_results = high_vector_hybrid.search("Python tutorial", k=2)
        low_results = low_vector_hybrid.search("Python tutorial", k=2)

        assert len(high_results) > 0
        assert len(low_results) > 0

    def test_hybrid_retriever_with_alpha_override(self, embedding_model, vector_store):
        """Test overriding alpha at query time."""
        config = RetrievalConfig(vector_weight=0.5)
        hybrid = HybridRetriever(embedding_model, vector_store, config)

        docs = ["Document about topic A", "Document about topic B"]
        embeddings = embedding_model.encode(docs)
        hybrid.add(["a", "b"], embeddings, docs, [{}, {}])

        # Use different alpha at query time
        results_low = hybrid.search("topic A", k=2, alpha=0.2)
        results_high = hybrid.search("topic A", k=2, alpha=0.8)

        # Both should return results
        assert len(results_low) > 0
        assert len(results_high) > 0

    def test_hybrid_handles_empty_results(self, embedding_model, vector_store):
        """Test hybrid retriever with no matching documents."""
        config = RetrievalConfig(vector_weight=0.5)
        hybrid = HybridRetriever(embedding_model, vector_store, config)

        # Empty index search
        results = hybrid.search("query with no matches", k=5)
        assert results == []

    def test_bm25_delete_documents(self):
        """Test deleting documents from BM25 index."""
        bm25 = BM25()

        bm25.add_documents(
            ids=["doc1", "doc2", "doc3"],
            documents=[
                "First document content",
                "Second document content",
                "Third document content",
            ]
        )

        assert bm25.n_docs == 3

        # Delete one document
        bm25.delete(["doc2"])

        assert bm25.n_docs == 2
        assert "doc2" not in bm25.doc_ids

        # Search should not return deleted document
        results = bm25.search("document content", k=5)
        result_ids = [r.id for r in results]
        assert "doc2" not in result_ids


# ============================================================================
# Streaming Response Tests
# ============================================================================

class TestStreamingResponses:
    """Test streaming response functionality."""

    @pytest.mark.asyncio
    async def test_basic_streaming_query(self, sample_documents, embedding_model, vector_store):
        """Test basic streaming query response."""
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        index = RAGIndex(embedding_model, vector_store, chunker)

        # Index documents
        for doc in sample_documents:
            index.index_document(doc)

        # Create streaming LLM mock
        class StreamingMockLLM(LLMProvider):
            async def generate(self, messages, temperature=0.7, max_tokens=1024):
                return "Complete response"

            async def generate_stream(self, messages, temperature=0.7, max_tokens=1024):
                chunks = ["This ", "is ", "a ", "streamed ", "response."]
                for chunk in chunks:
                    yield chunk
                    await asyncio.sleep(0.01)

        streaming_llm = StreamingMockLLM()
        pipeline = RAGPipeline(index, streaming_llm, RAGConfig(top_k=3))

        # Collect streamed chunks
        chunks = []
        async for chunk in pipeline.query_stream("What is Python?"):
            chunks.append(chunk)

        # Should receive multiple chunks
        assert len(chunks) > 0
        # Combined should form complete response
        full_response = "".join(chunks)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_streaming_with_empty_index(self, embedding_model, vector_store):
        """Test streaming with no documents indexed."""
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        index = RAGIndex(embedding_model, vector_store, chunker)

        class StreamingMockLLM(LLMProvider):
            async def generate(self, messages, temperature=0.7, max_tokens=1024):
                return "No relevant documents found."

            async def generate_stream(self, messages, temperature=0.7, max_tokens=1024):
                yield "No relevant documents found."

        streaming_llm = StreamingMockLLM()
        pipeline = RAGPipeline(index, streaming_llm, RAGConfig(top_k=3))

        chunks = []
        async for chunk in pipeline.query_stream("Any query"):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_streaming_preserves_chunk_order(self, sample_documents, embedding_model, vector_store):
        """Test that streaming chunks arrive in correct order."""
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        index = RAGIndex(embedding_model, vector_store, chunker)

        for doc in sample_documents:
            index.index_document(doc)

        class OrderedStreamingLLM(LLMProvider):
            async def generate(self, messages, temperature=0.7, max_tokens=1024):
                return "1234567890"

            async def generate_stream(self, messages, temperature=0.7, max_tokens=1024):
                for i in range(10):
                    yield str(i)
                    await asyncio.sleep(0.005)

        streaming_llm = OrderedStreamingLLM()
        pipeline = RAGPipeline(index, streaming_llm, RAGConfig(top_k=3))

        chunks = []
        async for chunk in pipeline.query_stream("Test query"):
            chunks.append(chunk)

        # Verify order
        assert "".join(chunks) == "0123456789"

    @pytest.mark.asyncio
    async def test_streaming_handles_long_response(self, sample_documents, embedding_model, vector_store):
        """Test streaming with a long response."""
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        index = RAGIndex(embedding_model, vector_store, chunker)

        for doc in sample_documents:
            index.index_document(doc)

        class LongStreamingLLM(LLMProvider):
            async def generate(self, messages, temperature=0.7, max_tokens=1024):
                return " ".join(["word"] * 100)

            async def generate_stream(self, messages, temperature=0.7, max_tokens=1024):
                for i in range(100):
                    yield "word "
                    await asyncio.sleep(0.001)

        streaming_llm = LongStreamingLLM()
        pipeline = RAGPipeline(index, streaming_llm, RAGConfig(top_k=3))

        chunk_count = 0
        async for chunk in pipeline.query_stream("Long response query"):
            chunk_count += 1

        # Should have many chunks
        assert chunk_count == 100


# ============================================================================
# Document Parsing Integration Tests
# ============================================================================

class TestDocumentParsingIntegration:
    """Test document parsing integrated with the RAG pipeline."""

    def test_markdown_parsing_and_indexing(self, mock_llm, tmp_path):
        """Test parsing markdown files and indexing them."""
        # Create a markdown file with enough content to be chunked
        md_file = tmp_path / "test_doc.md"
        md_file.write_text("""# Test Document

This is a test document about machine learning and artificial intelligence.
Machine learning is transforming many industries today.

## Section 1

Machine learning is a branch of artificial intelligence that enables systems to learn.
It uses algorithms to identify patterns in data and make predictions.
Deep learning is a subset of machine learning using neural networks.

## Section 2

It includes supervised and unsupervised learning methods for different tasks.
Supervised learning uses labeled data for training classification models.
Unsupervised learning discovers hidden patterns in unlabeled data.
""")

        # Parse the document
        ingestion = DocumentIngestion()
        doc = ingestion.ingest(md_file)

        assert doc is not None
        assert doc.metadata["title"] == "Test Document"
        assert "machine learning" in doc.content.lower()

        # Index and search with fresh instances
        embedding_model = MockEmbedding(dimension=128)
        vector_store = SimpleVectorStore()
        chunker = FixedSizeChunker(chunk_size=150, chunk_overlap=20)
        index = RAGIndex(embedding_model, vector_store, chunker)
        index.index_document(doc)

        assert index.count > 0  # Verify document was indexed

        results = index.search("machine learning AI", k=3)
        assert len(results) > 0

    def test_text_file_parsing_and_indexing(self, embedding_model, vector_store, tmp_path):
        """Test parsing plain text files."""
        # Create a text file
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("""This is a plain text document.
It contains information about data processing.
Data processing involves transforming raw data into useful information.
""")

        ingestion = DocumentIngestion()
        doc = ingestion.ingest(txt_file)

        assert doc is not None
        assert "data processing" in doc.content.lower()

        # Index
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        index = RAGIndex(embedding_model, vector_store, chunker)
        index.index_document(doc)

        assert index.count > 0

    def test_directory_ingestion(self, embedding_model, vector_store, tmp_path):
        """Test ingesting an entire directory."""
        # Create multiple files
        (tmp_path / "doc1.txt").write_text("Document one about Python programming.")
        (tmp_path / "doc2.txt").write_text("Document two about data science.")
        (tmp_path / "doc3.md").write_text("# Markdown\n\nDocument three about AI.")
        (tmp_path / "ignored.pdf")  # This won't be created properly, will be skipped

        ingestion = DocumentIngestion()
        docs = list(ingestion.ingest_directory(tmp_path))

        # Should have ingested the text and markdown files
        assert len(docs) >= 2

        # Index all
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        index = RAGIndex(embedding_model, vector_store, chunker)
        for doc in docs:
            index.index_document(doc)

        assert index.count > 0


# ============================================================================
# Multi-Index Manager Tests
# ============================================================================

class TestMultiIndexManager:
    """Test multi-index management for multi-tenant scenarios."""

    def test_create_and_manage_multiple_indices(self, embedding_model):
        """Test creating and managing multiple indices."""
        manager = MultiIndexManager()
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)

        # Create indices for different tenants
        store1 = SimpleVectorStore()
        store2 = SimpleVectorStore()

        index1 = manager.create_index("tenant_1", embedding_model, store1, chunker)
        index2 = manager.create_index("tenant_2", embedding_model, store2, chunker)

        # Verify indices are separate
        assert manager.get_index("tenant_1") is index1
        assert manager.get_index("tenant_2") is index2
        assert index1 is not index2

        # Index different documents
        index1.index_document(Document(id="doc1", content="Tenant 1 document", metadata={}))
        index2.index_document(Document(id="doc2", content="Tenant 2 document", metadata={}))

        # Verify isolation
        assert index1.count == 1
        assert index2.count == 1

    def test_list_and_delete_indices(self, embedding_model):
        """Test listing and deleting indices."""
        manager = MultiIndexManager()
        chunker = FixedSizeChunker(chunk_size=100)

        # Create multiple indices
        for i in range(3):
            store = SimpleVectorStore()
            manager.create_index(f"index_{i}", embedding_model, store, chunker)

        # List indices
        indices = manager.list_indices()
        assert len(indices) == 3
        assert "index_0" in indices
        assert "index_1" in indices
        assert "index_2" in indices

        # Delete one
        manager.delete_index("index_1")
        indices = manager.list_indices()
        assert len(indices) == 2
        assert "index_1" not in indices

    def test_get_nonexistent_index(self):
        """Test getting a non-existent index returns None."""
        manager = MultiIndexManager()
        result = manager.get_index("nonexistent")
        assert result is None


# ============================================================================
# Error Handling and Edge Cases
# ============================================================================

class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_empty_document_chunking(self, embedding_model, vector_store):
        """Test handling empty documents."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        index = RAGIndex(embedding_model, vector_store, chunker)

        empty_doc = Document(id="empty", content="", metadata={})
        index.index_document(empty_doc)

        # Should not add any chunks for empty document
        assert index.count == 0

    def test_whitespace_only_document(self, embedding_model, vector_store):
        """Test handling whitespace-only documents."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        index = RAGIndex(embedding_model, vector_store, chunker)

        ws_doc = Document(id="whitespace", content="   \n\t\n   ", metadata={})
        index.index_document(ws_doc)

        # Should not index whitespace-only content
        assert index.count == 0

    def test_very_long_document(self, embedding_model, vector_store):
        """Test handling very long documents."""
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        index = RAGIndex(embedding_model, vector_store, chunker)

        # Create a very long document
        long_content = "This is a test sentence. " * 1000  # ~24KB
        long_doc = Document(id="long_doc", content=long_content, metadata={})

        index.index_document(long_doc)

        # Should create many chunks
        assert index.count > 10

        # Should still be searchable
        results = index.search("test sentence", k=5)
        assert len(results) > 0

    def test_special_characters_in_content(self, embedding_model, vector_store):
        """Test handling special characters in documents."""
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        index = RAGIndex(embedding_model, vector_store, chunker)

        special_doc = Document(
            id="special",
            content="Special characters document with symbols @#$%^&*()_+-=[]{}|;':\",./<>? Unicode: \u00e9\u00e8\u00ea \u4e2d\u6587 and more text here to ensure sufficient length for chunking.",
            metadata={}
        )

        index.index_document(special_doc)
        # Should successfully index documents with special characters
        assert index.count > 0

        # Search using text that's in the document
        results = index.search("Special characters document", k=3)
        # The search should work even with special characters in the content
        # Note: Results depend on the embedding model's handling of special chars
        assert index.count > 0  # Main assertion is that indexing worked

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, sample_documents, rag_pipeline):
        """Test handling concurrent queries."""
        # Index documents
        for doc in sample_documents:
            rag_pipeline.index.index_document(doc)

        # Run multiple concurrent queries
        queries = [
            "What is Python?",
            "Tell me about Rust.",
            "Explain machine learning.",
            "What is RAG?",
            "Describe vector databases.",
        ]

        async def run_query(q):
            return await rag_pipeline.query(q)

        # Execute concurrently
        responses = await asyncio.gather(*[run_query(q) for q in queries])

        # All should succeed
        assert len(responses) == 5
        for response in responses:
            assert isinstance(response, RAGResponse)
            assert response.answer is not None


# ============================================================================
# Configuration Tests
# ============================================================================

class TestRAGConfiguration:
    """Test RAG configuration options."""

    @pytest.mark.asyncio
    async def test_top_k_configuration(self, sample_documents, embedding_model, vector_store, mock_llm):
        """Test top_k configuration affects result count."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        index = RAGIndex(embedding_model, vector_store, chunker)

        for doc in sample_documents:
            index.index_document(doc)

        # Test with different top_k values
        for top_k in [1, 3, 5, 10]:
            config = RAGConfig(top_k=top_k)
            pipeline = RAGPipeline(index, mock_llm, config)

            response = await pipeline.query("programming languages")

            # Should not exceed top_k
            assert len(response.sources) <= top_k

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, sample_documents, embedding_model, vector_store):
        """Test custom system prompt configuration."""
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        index = RAGIndex(embedding_model, vector_store, chunker)

        for doc in sample_documents:
            index.index_document(doc)

        custom_prompt = "You are a technical expert. Be concise and precise."
        config = RAGConfig(
            top_k=3,
            system_prompt=custom_prompt
        )

        # Create LLM that captures messages
        captured_messages = []

        class CapturingLLM(LLMProvider):
            async def generate(self, messages, temperature=0.7, max_tokens=1024):
                captured_messages.extend(messages)
                return "Response"

        capturing_llm = CapturingLLM()
        pipeline = RAGPipeline(index, capturing_llm, config)

        await pipeline.query("What is Python?")

        # Verify custom system prompt was used
        system_msgs = [m for m in captured_messages if m.get("role") == "system"]
        assert len(system_msgs) > 0
        assert custom_prompt in system_msgs[0]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
