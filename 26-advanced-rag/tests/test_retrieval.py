"""Tests for retrieval modules (BM25, vector, and hybrid)."""

import pytest
import numpy as np

from advancedrag.schemas import Document, RetrievalResult
from advancedrag.retrieval.bm25 import BM25Index
from advancedrag.retrieval.vector import (
    SimpleVectorStore,
    VectorRetriever,
    MockEmbedding,
)
from advancedrag.retrieval.hybrid import HybridRetriever


class TestBM25Index:
    """Test BM25 indexing and search."""

    @pytest.fixture
    def bm25_index(self):
        """Create BM25 index with default parameters."""
        return BM25Index()

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="doc1",
                content="Machine learning is a subset of artificial intelligence.",
                metadata={"category": "tech", "year": 2023}
            ),
            Document(
                id="doc2",
                content="Deep learning uses neural networks with multiple layers.",
                metadata={"category": "tech", "year": 2022}
            ),
            Document(
                id="doc3",
                content="Natural language processing handles text data.",
                metadata={"category": "nlp", "year": 2023}
            ),
            Document(
                id="doc4",
                content="Cooking recipes for healthy pasta dishes.",
                metadata={"category": "food", "year": 2021}
            ),
        ]

    def test_add_single_document(self, bm25_index):
        """Test adding a single document."""
        doc = Document(id="doc1", content="Test document", metadata={})
        bm25_index.add_documents([doc])

        assert bm25_index.count == 1

    def test_add_multiple_documents(self, bm25_index, sample_documents):
        """Test adding multiple documents."""
        bm25_index.add_documents(sample_documents)

        assert bm25_index.count == 4

    def test_basic_search(self, bm25_index, sample_documents):
        """Test basic BM25 search."""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search("machine learning", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        # First result should be about machine learning
        assert results[0].document.id == "doc1"

    def test_search_returns_scores(self, bm25_index, sample_documents):
        """Test that search returns BM25 scores."""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search("machine learning")

        assert all(r.score > 0 for r in results if r.document.id == "doc1")

    def test_search_with_no_matches(self, bm25_index, sample_documents):
        """Test search with query that has no matches."""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search("xyz123nonexistent")

        assert len(results) == 0

    def test_search_ranking_order(self, bm25_index, sample_documents):
        """Test results are ranked by relevance."""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search("machine learning")

        # Scores should be in descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_top_k_limit(self, bm25_index, sample_documents):
        """Test top_k limits results."""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search("learning", top_k=2)

        assert len(results) <= 2

    def test_retrieval_type_is_bm25(self, bm25_index, sample_documents):
        """Test retriever_type is set to bm25."""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search("machine learning")

        assert all(r.retriever_type == "bm25" for r in results)

    def test_ranks_are_assigned(self, bm25_index, sample_documents):
        """Test ranks are correctly assigned."""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search("learning")

        for i, result in enumerate(results):
            assert result.rank == i

    def test_search_with_filter_exact_match(self, bm25_index, sample_documents):
        """Test search with exact metadata filter."""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search("learning", filter_dict={"category": "tech"})

        assert all(r.document.metadata["category"] == "tech" for r in results)

    def test_search_with_filter_in_operator(self, bm25_index, sample_documents):
        """Test search with $in filter operator."""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search(
            "learning",
            filter_dict={"category": {"$in": ["tech", "nlp"]}}
        )

        assert all(
            r.document.metadata["category"] in ["tech", "nlp"]
            for r in results
        )

    def test_search_with_filter_gte_operator(self, bm25_index, sample_documents):
        """Test search with $gte filter operator."""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search(
            "learning",
            filter_dict={"year": {"$gte": 2023}}
        )

        assert all(r.document.metadata["year"] >= 2023 for r in results)

    def test_search_with_filter_ne_operator(self, bm25_index, sample_documents):
        """Test search with $ne filter operator."""
        bm25_index.add_documents(sample_documents)

        results = bm25_index.search(
            "learning",
            filter_dict={"category": {"$ne": "food"}}
        )

        assert all(r.document.metadata["category"] != "food" for r in results)

    def test_delete_document(self, bm25_index, sample_documents):
        """Test deleting a document."""
        bm25_index.add_documents(sample_documents)

        bm25_index.delete(["doc1"])

        assert bm25_index.count == 3
        results = bm25_index.search("machine learning")
        assert not any(r.document.id == "doc1" for r in results)

    def test_delete_multiple_documents(self, bm25_index, sample_documents):
        """Test deleting multiple documents."""
        bm25_index.add_documents(sample_documents)

        bm25_index.delete(["doc1", "doc2"])

        assert bm25_index.count == 2

    def test_empty_index_search(self, bm25_index):
        """Test searching empty index."""
        results = bm25_index.search("test query")

        assert len(results) == 0

    def test_tokenization_removes_stopwords(self, bm25_index):
        """Test that stopwords are removed during tokenization."""
        tokens = bm25_index._tokenize("the machine learning is a technology")

        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "machine" in tokens
        assert "learning" in tokens

    def test_tokenization_lowercases(self, bm25_index):
        """Test that tokenization lowercases text."""
        tokens = bm25_index._tokenize("Machine Learning")

        assert "machine" in tokens
        assert "Machine" not in tokens

    def test_bm25_parameters(self):
        """Test custom BM25 parameters."""
        index = BM25Index(k1=2.0, b=0.5)

        assert index.k1 == 2.0
        assert index.b == 0.5


class TestSimpleVectorStore:
    """Test SimpleVectorStore implementation."""

    @pytest.fixture
    def vector_store(self):
        """Create simple vector store."""
        return SimpleVectorStore()

    def test_add_documents(self, vector_store):
        """Test adding documents to vector store."""
        embeddings = np.random.randn(3, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        vector_store.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=embeddings,
            documents=["Content 1", "Content 2", "Content 3"],
            metadatas=[{"k": "v1"}, {"k": "v2"}, {"k": "v3"}]
        )

        assert vector_store.count == 3

    def test_search_returns_similar(self, vector_store):
        """Test search returns similar documents."""
        # Create embeddings with known similarities
        base_emb = np.random.randn(384)
        base_emb = base_emb / np.linalg.norm(base_emb)

        # Similar to base
        similar_emb = base_emb + np.random.randn(384) * 0.1
        similar_emb = similar_emb / np.linalg.norm(similar_emb)

        # Different from base
        different_emb = np.random.randn(384)
        different_emb = different_emb / np.linalg.norm(different_emb)

        embeddings = np.vstack([base_emb, similar_emb, different_emb])

        vector_store.add(
            ids=["base", "similar", "different"],
            embeddings=embeddings,
            documents=["Base doc", "Similar doc", "Different doc"],
            metadatas=[{}, {}, {}]
        )

        results = vector_store.search(base_emb.reshape(1, -1), k=2)

        # Base should be most similar to itself
        assert results[0][0] == "base"

    def test_search_with_filter(self, vector_store):
        """Test search with metadata filter."""
        embeddings = np.random.randn(3, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        vector_store.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=embeddings,
            documents=["Content 1", "Content 2", "Content 3"],
            metadatas=[{"type": "a"}, {"type": "b"}, {"type": "a"}]
        )

        query_emb = np.random.randn(1, 384)

        results = vector_store.search(query_emb, k=10, filter_dict={"type": "a"})

        assert all(r[2]["type"] == "a" for r in results if r[3] > 0)

    def test_delete_documents(self, vector_store):
        """Test deleting documents from vector store."""
        embeddings = np.random.randn(3, 384)

        vector_store.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=embeddings,
            documents=["Content 1", "Content 2", "Content 3"],
            metadatas=[{}, {}, {}]
        )

        vector_store.delete(["doc2"])

        assert vector_store.count == 2
        assert "doc2" not in vector_store.ids

    def test_empty_store_search(self, vector_store):
        """Test searching empty store."""
        query_emb = np.random.randn(1, 384)

        results = vector_store.search(query_emb, k=10)

        assert len(results) == 0


class TestMockEmbedding:
    """Test MockEmbedding for testing."""

    @pytest.fixture
    def mock_embedding(self):
        """Create mock embedding model."""
        return MockEmbedding(dimension=384)

    def test_encode_single_text(self, mock_embedding):
        """Test encoding a single text."""
        embeddings = mock_embedding.encode(["Test text"])

        assert embeddings.shape == (1, 384)

    def test_encode_multiple_texts(self, mock_embedding):
        """Test encoding multiple texts."""
        embeddings = mock_embedding.encode(["Text 1", "Text 2", "Text 3"])

        assert embeddings.shape == (3, 384)

    def test_embeddings_are_normalized(self, mock_embedding):
        """Test embeddings are normalized."""
        embeddings = mock_embedding.encode(["Test text"])
        norm = np.linalg.norm(embeddings[0])

        assert abs(norm - 1.0) < 1e-6

    def test_embeddings_are_deterministic(self, mock_embedding):
        """Test same text produces same embedding."""
        emb1 = mock_embedding.encode(["Test text"])
        emb2 = mock_embedding.encode(["Test text"])

        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_different_texts_different_embeddings(self, mock_embedding):
        """Test different texts produce different embeddings."""
        embeddings = mock_embedding.encode(["Text A", "Text B"])

        assert not np.allclose(embeddings[0], embeddings[1])

    def test_custom_dimension(self):
        """Test custom embedding dimension."""
        mock_embedding = MockEmbedding(dimension=768)
        embeddings = mock_embedding.encode(["Test"])

        assert embeddings.shape == (1, 768)


class TestVectorRetriever:
    """Test VectorRetriever."""

    @pytest.fixture
    def retriever(self):
        """Create vector retriever with mock components."""
        embedding_model = MockEmbedding(dimension=384)
        vector_store = SimpleVectorStore()
        return VectorRetriever(embedding_model, vector_store)

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""
        return [
            Document(id="doc1", content="Machine learning algorithms", metadata={}),
            Document(id="doc2", content="Deep learning neural networks", metadata={}),
            Document(id="doc3", content="Cooking pasta recipes", metadata={}),
        ]

    def test_add_documents(self, retriever, sample_documents):
        """Test adding documents to retriever."""
        retriever.add_documents(sample_documents)

        assert retriever.vector_store.count == 3

    def test_search_returns_results(self, retriever, sample_documents):
        """Test search returns retrieval results."""
        retriever.add_documents(sample_documents)

        results = retriever.search("machine learning", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_search_retriever_type(self, retriever, sample_documents):
        """Test retriever_type is set to vector."""
        retriever.add_documents(sample_documents)

        results = retriever.search("machine learning")

        assert all(r.retriever_type == "vector" for r in results)

    def test_search_ranks(self, retriever, sample_documents):
        """Test ranks are assigned correctly."""
        retriever.add_documents(sample_documents)

        results = retriever.search("learning")

        for i, result in enumerate(results):
            assert result.rank == i

    def test_delete_documents(self, retriever, sample_documents):
        """Test deleting documents."""
        retriever.add_documents(sample_documents)
        retriever.delete(["doc1"])

        results = retriever.search("machine learning")

        assert not any(r.document.id == "doc1" for r in results)


class TestHybridRetriever:
    """Test HybridRetriever combining BM25 and vector search."""

    @pytest.fixture
    def hybrid_retriever(self):
        """Create hybrid retriever."""
        embedding_model = MockEmbedding(dimension=384)
        vector_store = SimpleVectorStore()
        return HybridRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            bm25_weight=0.4,
            vector_weight=0.6,
            fusion_strategy="rrf"
        )

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""
        return [
            Document(
                id="doc1",
                content="Machine learning is a type of artificial intelligence.",
                metadata={"category": "tech"}
            ),
            Document(
                id="doc2",
                content="Deep learning uses multi-layer neural networks.",
                metadata={"category": "tech"}
            ),
            Document(
                id="doc3",
                content="Natural language processing analyzes text.",
                metadata={"category": "nlp"}
            ),
            Document(
                id="doc4",
                content="Cooking healthy meals with vegetables.",
                metadata={"category": "food"}
            ),
        ]

    def test_add_documents(self, hybrid_retriever, sample_documents):
        """Test adding documents to both indices."""
        hybrid_retriever.add_documents(sample_documents)

        assert hybrid_retriever.count == 4
        assert hybrid_retriever.bm25_index.count == 4
        assert hybrid_retriever.vector_retriever.vector_store.count == 4

    def test_hybrid_search_returns_results(self, hybrid_retriever, sample_documents):
        """Test hybrid search returns fused results."""
        hybrid_retriever.add_documents(sample_documents)

        results = hybrid_retriever.search("machine learning", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_rrf_fusion_type(self, hybrid_retriever, sample_documents):
        """Test RRF fusion sets correct retriever type."""
        hybrid_retriever.add_documents(sample_documents)

        results = hybrid_retriever.search("machine learning")

        assert all(r.retriever_type == "hybrid_rrf" for r in results)

    def test_linear_fusion(self, sample_documents):
        """Test linear fusion strategy."""
        embedding_model = MockEmbedding(dimension=384)
        vector_store = SimpleVectorStore()
        retriever = HybridRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            fusion_strategy="linear"
        )

        retriever.add_documents(sample_documents)
        results = retriever.search("machine learning")

        assert all(r.retriever_type == "hybrid_linear" for r in results)

    def test_alpha_override(self, hybrid_retriever, sample_documents):
        """Test alpha parameter overrides weights."""
        hybrid_retriever.add_documents(sample_documents)

        # Alpha = 1.0 should use only vector search
        results_vector_only = hybrid_retriever.search("machine learning", alpha=1.0)
        # Alpha = 0.0 should use only BM25
        results_bm25_only = hybrid_retriever.search("machine learning", alpha=0.0)

        assert len(results_vector_only) > 0
        assert len(results_bm25_only) > 0

    def test_search_with_filter(self, hybrid_retriever, sample_documents):
        """Test hybrid search with metadata filter."""
        hybrid_retriever.add_documents(sample_documents)

        results = hybrid_retriever.search(
            "learning",
            filter_dict={"category": "tech"}
        )

        assert all(r.document.metadata["category"] == "tech" for r in results)

    def test_delete_from_both_indices(self, hybrid_retriever, sample_documents):
        """Test delete removes from both indices."""
        hybrid_retriever.add_documents(sample_documents)
        hybrid_retriever.delete(["doc1"])

        assert hybrid_retriever.count == 3

        results = hybrid_retriever.search("machine learning")
        assert not any(r.document.id == "doc1" for r in results)

    def test_score_normalization(self):
        """Test score normalization for linear combination."""
        embedding_model = MockEmbedding(dimension=384)
        vector_store = SimpleVectorStore()
        retriever = HybridRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            fusion_strategy="linear"
        )

        # Test normalization function directly
        results = [
            RetrievalResult(
                document=Document(id="d1", content="", metadata={}),
                score=10.0, retriever_type="test", rank=0
            ),
            RetrievalResult(
                document=Document(id="d2", content="", metadata={}),
                score=5.0, retriever_type="test", rank=1
            ),
            RetrievalResult(
                document=Document(id="d3", content="", metadata={}),
                score=0.0, retriever_type="test", rank=2
            ),
        ]

        normalized = retriever._normalize_scores(results)

        assert normalized[0] == 1.0  # Max score
        assert normalized[2] == 0.0  # Min score
        assert 0 < normalized[1] < 1  # Middle score

    def test_normalize_empty_list(self, hybrid_retriever):
        """Test score normalization with empty list."""
        normalized = hybrid_retriever._normalize_scores([])

        assert normalized == []

    def test_normalize_single_item(self, hybrid_retriever):
        """Test score normalization with single item."""
        results = [
            RetrievalResult(
                document=Document(id="d1", content="", metadata={}),
                score=5.0, retriever_type="test", rank=0
            )
        ]

        normalized = hybrid_retriever._normalize_scores(results)

        assert normalized == [1.0]

    def test_rrf_formula(self, hybrid_retriever, sample_documents):
        """Test RRF fusion formula produces valid scores."""
        hybrid_retriever.add_documents(sample_documents)

        results = hybrid_retriever.search("learning", top_k=10)

        # All scores should be positive
        assert all(r.score > 0 for r in results)
        # Scores should be in descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score


class TestRetrievalResultStructure:
    """Test RetrievalResult data structure."""

    def test_retrieval_result_creation(self):
        """Test creating a RetrievalResult."""
        doc = Document(id="doc1", content="Test content", metadata={"key": "value"})
        result = RetrievalResult(
            document=doc,
            score=0.85,
            retriever_type="bm25",
            rank=0
        )

        assert result.document.id == "doc1"
        assert result.score == 0.85
        assert result.retriever_type == "bm25"
        assert result.rank == 0

    def test_retrieval_result_to_dict(self):
        """Test RetrievalResult serialization."""
        doc = Document(id="doc1", content="Test content", metadata={"key": "value"})
        result = RetrievalResult(
            document=doc,
            score=0.85,
            retriever_type="vector",
            rank=1
        )

        result_dict = result.to_dict()

        assert result_dict["score"] == 0.85
        assert result_dict["retriever_type"] == "vector"
        assert result_dict["rank"] == 1
        assert result_dict["document"]["id"] == "doc1"


class TestDocumentStructure:
    """Test Document data structure."""

    def test_document_creation(self):
        """Test creating a Document."""
        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"key": "value"}
        )

        assert doc.id == "doc1"
        assert doc.content == "Test content"
        assert doc.metadata == {"key": "value"}

    def test_document_default_metadata(self):
        """Test Document with default metadata."""
        doc = Document(id="doc1", content="Test")

        assert doc.metadata == {}

    def test_document_to_dict(self):
        """Test Document serialization."""
        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"key": "value"}
        )

        doc_dict = doc.to_dict()

        assert doc_dict["id"] == "doc1"
        assert doc_dict["content"] == "Test content"
        assert doc_dict["metadata"] == {"key": "value"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
