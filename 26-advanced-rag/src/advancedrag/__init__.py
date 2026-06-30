"""
Advanced RAG System

Production-grade Retrieval-Augmented Generation with multi-stage retrieval,
query rewriting, neural reranking, and hallucination detection.
"""

from .schemas import (
    Document,
    QueryIntent,
    RewrittenQuery,
    RetrievalResult,
    RerankResult,
    Citation,
    ConstructedContext,
    GeneratedAnswer,
    RAGResult,
    generate_id,
)
from .pipeline import RAGPipeline, create_pipeline
from .retrieval.bm25 import BM25Index
from .retrieval.vector import (
    VectorRetriever,
    SimpleVectorStore,
    MockEmbedding,
    SentenceTransformerEmbedding,
)
from .retrieval.hybrid import HybridRetriever
from .retrieval.multi_hop import MultiHopRetriever
from .query.rewriter import (
    QueryRewriter,
    LLMQueryRewriter,
    RuleBasedRewriter,
    HybridQueryRewriter,
)
from .reranking.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    SLMReranker,
    MultiStageReranker,
    MockReranker,
)
from .context.constructor import (
    ContextConstructor,
    LLMCompressor,
    SemanticDeduplicator,
)
from .generation.answerer import (
    LLMAnswerer,
    HallucinationDetector,
    CitationExtractor,
)
from .enterprise.tenant import (
    TenantConfig,
    TenantConfigManager,
)
from .enterprise.registry import (
    RerankerRegistry,
    reranker_registry,
    setup_default_registries,
)
# Evaluation imports (optional - require scipy)
try:
    from .evaluation.evaluator import (
        RAGEvaluator,
        EvaluationResult,
        EvaluationReport,
        answer_faithfulness_metric,
        answer_relevance_metric,
        context_precision_metric,
        context_recall_metric,
        answer_correctness_metric,
    )
    from .evaluation.ab_testing import (
        ABTestManager,
        ABTest,
        ABTestAnalysis,
    )
    _HAS_EVALUATION = True
except ImportError:
    RAGEvaluator = None
    EvaluationResult = None
    EvaluationReport = None
    ABTestManager = None
    ABTest = None
    ABTestAnalysis = None
    answer_faithfulness_metric = None
    answer_relevance_metric = None
    context_precision_metric = None
    context_recall_metric = None
    answer_correctness_metric = None
    _HAS_EVALUATION = False
from .utils.cache import (
    LRUCache,
    TTLCache,
    RAGCacheManager,
)
from .utils.monitoring import (
    MetricsCollector,
    get_collector,
)
from .utils.batch import (
    BatchProcessor,
    parallel_map,
)

__version__ = "1.0.0"
__all__ = [
    # Schemas
    "Document",
    "QueryIntent",
    "RewrittenQuery",
    "RetrievalResult",
    "RerankResult",
    "Citation",
    "ConstructedContext",
    "GeneratedAnswer",
    "RAGResult",
    "generate_id",
    # Pipeline
    "RAGPipeline",
    "create_pipeline",
    # Retrieval
    "BM25Index",
    "VectorRetriever",
    "SimpleVectorStore",
    "MockEmbedding",
    "SentenceTransformerEmbedding",
    "HybridRetriever",
    "MultiHopRetriever",
    # Query
    "QueryRewriter",
    "LLMQueryRewriter",
    "RuleBasedRewriter",
    "HybridQueryRewriter",
    # Reranking
    "BaseReranker",
    "CrossEncoderReranker",
    "SLMReranker",
    "MultiStageReranker",
    "MockReranker",
    # Context
    "ContextConstructor",
    "LLMCompressor",
    "SemanticDeduplicator",
    # Generation
    "LLMAnswerer",
    "HallucinationDetector",
    "CitationExtractor",
    # Enterprise
    "TenantConfig",
    "TenantConfigManager",
    "RerankerRegistry",
    "reranker_registry",
    "setup_default_registries",
    # Evaluation
    "RAGEvaluator",
    "EvaluationResult",
    "EvaluationReport",
    "ABTestManager",
    "ABTest",
    "ABTestAnalysis",
    # Evaluation metrics
    "answer_faithfulness_metric",
    "answer_relevance_metric",
    "context_precision_metric",
    "context_recall_metric",
    "answer_correctness_metric",
    # Utils
    "LRUCache",
    "TTLCache",
    "RAGCacheManager",
    "MetricsCollector",
    "get_collector",
    "BatchProcessor",
    "parallel_map",
]
