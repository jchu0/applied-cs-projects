"""
RAG Baseline System

A production-ready Retrieval-Augmented Generation system with document
ingestion, embedding-based retrieval, and LLM generation.
"""

from .schemas import (
    Document,
    Chunk,
    SearchResult,
    RAGResponse,
    RAGConfig,
    generate_id,
)
from .parsers import (
    DocumentParser,
    PDFParser,
    HTMLParser,
    MarkdownParser,
    TextParser,
    DocumentIngestion,
)
from .chunking import (
    ChunkingStrategy,
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
)
from .embeddings import (
    EmbeddingModel,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
    HuggingFaceEmbedding,
    MockEmbedding,
    get_embedding_model,
)
from .vectorstore import (
    VectorStore,
    ChromaVectorStore,
    SimpleVectorStore,
    get_vector_store,
)
from .index import (
    RAGIndex,
    MultiIndexManager,
)
from .pipeline import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    MockLLMProvider,
    RAGPipeline,
    get_llm_provider,
)
from .retrieval import (
    BM25,
    HybridRetriever,
    AsyncHybridRetriever,
    RetrievalConfig,
    Reranker,
    CrossEncoderReranker,
    CohereReranker,
    MetadataFilter,
    get_reranker,
    VectorRetriever,
    FusionRetriever,
    MMRRetriever,
)
from .enterprise import (
    TenantManager,
    RetrievalLogger,
    UsageTracker,
)

__version__ = "1.0.0"
__all__ = [
    # Schemas
    "Document",
    "Chunk",
    "SearchResult",
    "RAGResponse",
    "RAGConfig",
    "generate_id",
    # Parsers
    "DocumentParser",
    "PDFParser",
    "HTMLParser",
    "MarkdownParser",
    "TextParser",
    "DocumentIngestion",
    # Chunking
    "ChunkingStrategy",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "SemanticChunker",
    # Embeddings
    "EmbeddingModel",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    "HuggingFaceEmbedding",
    "MockEmbedding",
    "get_embedding_model",
    # Vector Store
    "VectorStore",
    "ChromaVectorStore",
    "SimpleVectorStore",
    "get_vector_store",
    # Index
    "RAGIndex",
    "MultiIndexManager",
    # Pipeline
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "MockLLMProvider",
    "RAGPipeline",
    "get_llm_provider",
    # Retrieval
    "BM25",
    "HybridRetriever",
    "AsyncHybridRetriever",
    "VectorRetriever",
    "FusionRetriever",
    "MMRRetriever",
    "RetrievalConfig",
    "Reranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "MetadataFilter",
    "get_reranker",
    # Enterprise
    "TenantManager",
    "RetrievalLogger",
    "UsageTracker",
]
