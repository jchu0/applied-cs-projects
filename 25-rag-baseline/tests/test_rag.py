"""Tests for RAG baseline system."""

import pytest
import tempfile
from pathlib import Path

from ragbaseline.schemas import Document, Chunk, RAGConfig
from ragbaseline.parsers import DocumentIngestion, MarkdownParser
from ragbaseline.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker
from ragbaseline.embeddings import MockEmbedding
from ragbaseline.vectorstore import SimpleVectorStore
from ragbaseline.index import RAGIndex
from ragbaseline.pipeline import RAGPipeline, MockLLMProvider


class TestDocument:
    """Tests for Document schema."""

    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            id="test123",
            content="Test content",
            metadata={"key": "value"},
            source="/path/to/file",
        )

        assert doc.id == "test123"
        assert doc.content == "Test content"
        assert doc.metadata["key"] == "value"

    def test_document_to_dict(self):
        """Test converting document to dict."""
        doc = Document(id="test", content="content")
        data = doc.to_dict()

        assert data["id"] == "test"
        assert data["content"] == "content"


class TestChunking:
    """Tests for chunking strategies."""

    @pytest.fixture
    def document(self):
        """Create test document."""
        return Document(
            id="testdoc",
            content="This is sentence one. This is sentence two. This is sentence three. " * 10,
            metadata={"filename": "test.txt"},
        )

    def test_fixed_size_chunker(self, document):
        """Test fixed size chunking."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        assert all(len(chunk.content) <= 100 for chunk in chunks)
        assert all(chunk.document_id == "testdoc" for chunk in chunks)

    def test_sentence_chunker(self, document):
        """Test sentence-based chunking."""
        chunker = SentenceChunker(max_chunk_size=200, min_chunk_size=50)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == "testdoc"

    def test_recursive_chunker(self, document):
        """Test recursive chunking."""
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.chunk(document)

        assert len(chunks) > 0

    def test_empty_document(self):
        """Test chunking empty document."""
        doc = Document(id="empty", content="")
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(doc)

        assert len(chunks) == 0


class TestEmbeddings:
    """Tests for embedding models."""

    def test_mock_embedding(self):
        """Test mock embedding model."""
        model = MockEmbedding(dimension=128)

        texts = ["Hello world", "Test text"]
        embeddings = model.encode(texts)

        assert embeddings.shape == (2, 128)

    def test_mock_embedding_deterministic(self):
        """Test mock embedding is deterministic."""
        model = MockEmbedding(dimension=64)

        text = "Same text"
        emb1 = model.encode([text])
        emb2 = model.encode([text])

        assert (emb1 == emb2).all()


class TestVectorStore:
    """Tests for vector store."""

    @pytest.fixture
    def store(self):
        """Create vector store."""
        return SimpleVectorStore()

    @pytest.fixture
    def embedding(self):
        """Create embedding model."""
        return MockEmbedding(dimension=64)

    def test_add_and_search(self, store, embedding):
        """Test adding and searching documents."""
        texts = ["Python programming", "Java development", "Web design"]
        embeddings = embedding.encode(texts)

        store.add(
            ids=["1", "2", "3"],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{} for _ in texts],
        )

        # Search
        query_emb = embedding.encode(["Python code"])
        results = store.search(query_emb, k=2)

        assert len(results) == 2
        assert all(hasattr(r, 'score') for r in results)

    def test_delete(self, store, embedding):
        """Test deleting documents."""
        texts = ["doc1", "doc2"]
        embeddings = embedding.encode(texts)

        store.add(
            ids=["1", "2"],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{}, {}],
        )

        store.delete(["1"])

        assert store.count() == 1

    def test_filter(self, store, embedding):
        """Test metadata filtering."""
        texts = ["Python code", "Java code"]
        embeddings = embedding.encode(texts)

        store.add(
            ids=["1", "2"],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"lang": "python"}, {"lang": "java"}],
        )

        query_emb = embedding.encode(["code"])
        results = store.search(query_emb, k=5, filter={"lang": "python"})

        assert len(results) == 1
        assert results[0].metadata["lang"] == "python"


class TestRAGIndex:
    """Tests for RAG index."""

    @pytest.fixture
    def index(self):
        """Create RAG index."""
        embedding = MockEmbedding(dimension=64)
        store = SimpleVectorStore()
        chunker = FixedSizeChunker(chunk_size=100)
        return RAGIndex(embedding, store, chunker)

    def test_index_document(self, index):
        """Test indexing a document."""
        doc = Document(
            id="test",
            content="Python is a programming language used for many purposes.",
            metadata={"filename": "test.txt"},
        )

        index.index_document(doc)

        assert index.count > 0

    def test_search(self, index):
        """Test searching indexed documents."""
        doc = Document(
            id="test",
            content="The capital of France is Paris. Paris is known for the Eiffel Tower.",
            metadata={"filename": "geo.txt"},
        )

        index.index_document(doc)

        results = index.search("What is the capital of France?", k=2)

        assert len(results) > 0


class TestRAGPipeline:
    """Tests for RAG pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create RAG pipeline."""
        embedding = MockEmbedding(dimension=64)
        store = SimpleVectorStore()
        chunker = SentenceChunker(max_chunk_size=200)
        index = RAGIndex(embedding, store, chunker)

        llm = MockLLMProvider(responses=["The capital of France is Paris."])
        config = RAGConfig(top_k=3)

        return RAGPipeline(index, llm, config)

    @pytest.mark.asyncio
    async def test_query(self, pipeline):
        """Test RAG query."""
        # Index document
        doc = Document(
            id="test",
            content="France is a country in Europe. Its capital is Paris.",
            metadata={"filename": "geo.txt"},
        )
        pipeline.index.index_document(doc)

        # Query
        result = await pipeline.query("What is the capital of France?")

        assert "Paris" in result.answer
        assert result.query == "What is the capital of France?"
        assert len(result.sources) > 0

    @pytest.mark.asyncio
    async def test_query_with_filter(self, pipeline):
        """Test RAG query with filter."""
        doc = Document(
            id="test",
            content="Test content",
            metadata={"category": "test"},
        )
        pipeline.index.index_document(doc)

        result = await pipeline.query(
            "test query",
            filter={"category": "test"}
        )

        assert result is not None


class TestDocumentParsing:
    """Tests for document parsing."""

    def test_markdown_parser(self):
        """Test Markdown parser."""
        # Create temp markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Document\n\nThis is test content.")
            temp_path = f.name

        try:
            parser = MarkdownParser()
            assert parser.supports(Path(temp_path))

            doc = parser.parse(Path(temp_path))
            assert doc.content
            assert doc.metadata["title"] == "Test Document"
        finally:
            Path(temp_path).unlink()

    def test_ingestion_pipeline(self):
        """Test document ingestion pipeline."""
        ingestion = DocumentIngestion()

        # Check supported extensions
        extensions = ingestion.supported_extensions()
        assert ".md" in extensions
        assert ".txt" in extensions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
