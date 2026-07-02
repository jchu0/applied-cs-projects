"""Main orchestrated RAG pipeline."""

from typing import Optional
import time
import logging

from .schemas import StabilizedAnswer, generate_id
from .orchestrator import (
    ComponentRegistry,
    DynamicComputationGraph,
    GraphBuilder,
    PipelineTracer,
    get_tracer,
)
from .slm import (
    # Real implementations
    ChunkerSLM,
    EmbedderSLM,
    RetrieverSLM,
    RerankerSLM,
    SummarizerSLM,
    CoTCompressorSLM,
    AnswerStabilizerSLM,
    # Mock implementations (for fallback/testing)
    MockChunkerSLM,
    MockEmbedderSLM,
    MockRetrieverSLM,
    MockRerankerSLM,
    MockSummarizerSLM,
    MockCoTCompressorSLM,
    MockAnswerStabilizerSLM,
)
from .enterprise import (
    GuardrailEngine,
    RelevanceGuardrail,
    ConfidenceGuardrail,
    get_metrics,
)

logger = logging.getLogger(__name__)


class MicroRAGPipeline:
    """Main orchestrated RAG pipeline with specialized SLMs."""

    def __init__(
        self,
        registry: ComponentRegistry = None,
        use_guardrails: bool = True,
        use_mock: bool = False,
        tracer: PipelineTracer = None
    ):
        """Initialize pipeline.

        Args:
            registry: Component registry (creates default if None)
            use_guardrails: Whether to enable guardrails
            use_mock: Whether to use mock implementations (for testing)
            tracer: Pipeline tracer
        """
        self.use_mock = use_mock
        self.registry = registry or self._create_default_registry()
        self.tracer = tracer or get_tracer()
        self.metrics = get_metrics()

        # Build computation graph
        self.graph = self._build_graph()

        # Set up guardrails
        self.guardrails = None
        if use_guardrails:
            self.guardrails = GuardrailEngine([
                RelevanceGuardrail(min_relevance=0.2),
                ConfidenceGuardrail(min_confidence=0.3),
            ])

    def _create_default_registry(self) -> ComponentRegistry:
        """Create registry with either real or mock components."""
        registry = ComponentRegistry()

        if self.use_mock:
            # Register mock SLMs for testing
            logger.info("Using mock SLM implementations for testing")
            registry.register("chunker_slm", MockChunkerSLM())
            registry.register("embedder_slm", MockEmbedderSLM())
            registry.register("retriever_slm", MockRetrieverSLM())
            registry.register("reranker_slm", MockRerankerSLM())
            registry.register("summarizer_slm", MockSummarizerSLM())
            registry.register("cot_compressor_slm", MockCoTCompressorSLM())
            registry.register("answer_stabilizer_slm", MockAnswerStabilizerSLM())
        else:
            # Register real SLM implementations
            logger.info("Using real SLM implementations")

            # Chunker with semantic chunking
            registry.register(
                "chunker_slm",
                ChunkerSLM(
                    model_name="Qwen/Qwen2-0.5B-Instruct",
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                    max_chunk_tokens=512,
                    similarity_threshold=0.75
                )
            )

            # Embedder with BGE model
            registry.register(
                "embedder_slm",
                EmbedderSLM(
                    model_name="BAAI/bge-small-en-v1.5",
                    embedding_dim=384,
                    normalize_embeddings=True,
                    batch_size=32
                )
            )

            # Retriever with query routing
            registry.register(
                "retriever_slm",
                RetrieverSLM(
                    model_name="Qwen/Qwen2-0.5B-Instruct",
                    embedding_model="BAAI/bge-small-en-v1.5",
                    default_top_k=100,
                    use_faiss=True
                )
            )

            # Reranker with cross-encoder
            registry.register(
                "reranker_slm",
                RerankerSLM(
                    model_name="BAAI/bge-reranker-base",
                    llm_model="Qwen/Qwen2-1.5B-Instruct",
                    default_top_k=10,
                    batch_size=32
                )
            )

            # Summarizer with Qwen2
            registry.register(
                "summarizer_slm",
                SummarizerSLM(
                    model_name="Qwen/Qwen2-1.5B-Instruct",
                    max_summary_length=200,
                    temperature=0.3
                )
            )

            # CoT Compressor
            registry.register(
                "cot_compressor_slm",
                CoTCompressorSLM(
                    model_name="Qwen/Qwen2-1.5B-Instruct",
                    compression_ratio=0.3,
                    preserve_key_steps=True,
                    temperature=0.2
                )
            )

            # Answer Stabilizer
            registry.register(
                "answer_stabilizer_slm",
                AnswerStabilizerSLM(
                    model_name="Qwen/Qwen2-1.5B-Instruct",
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                    num_samples=3,
                    temperature_range=(0.3, 0.7),
                    consistency_threshold=0.8
                )
            )

        return registry

    def _build_graph(self) -> DynamicComputationGraph:
        """Build RAG computation graph."""
        return (
            GraphBuilder(self.registry)
            .slm("retrieve", "retriever_slm", dependencies=["query"])
            .slm("rerank", "reranker_slm", dependencies=["query", "retrieve"])
            .slm("summarize", "summarizer_slm", dependencies=["query", "rerank"])
            .slm("compress_cot", "cot_compressor_slm", dependencies=["query", "summarize"])
            .slm("stabilize", "answer_stabilizer_slm", dependencies=["query", "compress_cot"])
            .build()
        )

    async def query(
        self,
        query: str,
        top_k: int = 10,
        trace_id: str = None
    ) -> StabilizedAnswer:
        """Execute RAG query.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            trace_id: Optional trace ID

        Returns:
            Stabilized answer
        """
        trace_id = trace_id or generate_id()
        start_time = time.time()

        try:
            # Execute graph
            with self.tracer.trace_step("rag_query", trace_id) as span:
                self.tracer.add_attribute(span, "query", query)
                self.tracer.add_attribute(span, "top_k", top_k)
                self.tracer.add_attribute(span, "use_mock", self.use_mock)

                results = await self.graph.execute(
                    {"query": query},
                    trace_id
                )

            # Get final answer
            answer = results.get("stabilize")

            # Record metrics
            latency = (time.time() - start_time) * 1000
            self.metrics.record_latency("pipeline", latency)
            if hasattr(answer, 'confidence'):
                self.metrics.record_quality("pipeline", answer.confidence)
            self.metrics.record_request(success=True)

            # Check guardrails
            if self.guardrails:
                guard_result = await self.guardrails.check(
                    "stabilize",
                    query,
                    answer
                )
                if not guard_result.passed:
                    answer.reasoning += f" [Guardrail warning: {guard_result.action}]"

            logger.info(
                f"Query completed in {latency:.2f}ms with confidence {answer.confidence:.2f}"
            )

            return answer

        except Exception as e:
            # Graceful degradation: return a zero-confidence error answer, but
            # log the full traceback so the failure is never silent.
            logger.exception(f"Pipeline error while processing query: {str(e)}")
            self.metrics.record_request(success=False)

            # Return error response
            return StabilizedAnswer(
                answer=f"Error processing query: {str(e)}",
                confidence=0.0,
                consistency=0.0,
                num_samples=0,
                reasoning="Pipeline execution failed"
            )

    async def index_document(
        self,
        document: str,
        doc_id: str = None,
        metadata: dict = None
    ) -> dict:
        """Index a document.

        Args:
            document: Document content
            doc_id: Optional document ID
            metadata: Optional metadata

        Returns:
            Indexing result
        """
        doc_id = doc_id or generate_id()

        try:
            # Chunk document
            chunker = self.registry.get("chunker_slm")
            chunks = await chunker.process(document=document)

            # Embed chunks
            embedder = self.registry.get("embedder_slm")
            embeddings = await embedder.process(chunk=chunks)

            logger.info(
                f"Indexed document {doc_id}: {len(chunks)} chunks, "
                f"embedding dim {embeddings.shape[1] if len(embeddings.shape) > 1 else 0}"
            )

            return {
                "doc_id": doc_id,
                "num_chunks": len(chunks),
                "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Indexing error for document {doc_id}: {str(e)}")
            return {
                "doc_id": doc_id,
                "num_chunks": 0,
                "embedding_dim": 0,
                "status": "error",
                "error": str(e)
            }

    def get_execution_summary(self) -> dict:
        """Get execution summary."""
        return {
            "graph_summary": self.graph.get_execution_summary(),
            "metrics": self.metrics.get_summary(),
            "mode": "mock" if self.use_mock else "production",
        }

    async def warmup(self):
        """Warm up models by loading them into memory."""
        logger.info("Warming up SLM models...")

        components = [
            "chunker_slm",
            "embedder_slm",
            "retriever_slm",
            "reranker_slm",
            "summarizer_slm",
            "cot_compressor_slm",
            "answer_stabilizer_slm"
        ]

        for component_name in components:
            try:
                component = self.registry.get(component_name)
                if hasattr(component, 'load') and not component.is_loaded:
                    component.load()
                    logger.info(f"Loaded {component_name}")
            except Exception as e:
                logger.warning(f"Failed to warm up {component_name}: {str(e)}")

        logger.info("Model warmup complete")

    async def cleanup(self):
        """Clean up resources and unload models."""
        logger.info("Cleaning up SLM models...")

        components = [
            "chunker_slm",
            "embedder_slm",
            "retriever_slm",
            "reranker_slm",
            "summarizer_slm",
            "cot_compressor_slm",
            "answer_stabilizer_slm"
        ]

        for component_name in components:
            try:
                component = self.registry.get(component_name)
                if hasattr(component, 'unload'):
                    component.unload()
                    logger.info(f"Unloaded {component_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {component_name}: {str(e)}")

        logger.info("Cleanup complete")


def create_pipeline(
    use_mock: bool = False,
    use_guardrails: bool = True,
    warmup: bool = False
) -> MicroRAGPipeline:
    """Factory function to create pipeline.

    Args:
        use_mock: Use mock SLMs (for testing/development)
        use_guardrails: Enable guardrails
        warmup: Whether to warm up models on creation

    Returns:
        Configured pipeline
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create pipeline
    pipeline = MicroRAGPipeline(
        use_guardrails=use_guardrails,
        use_mock=use_mock
    )

    # Optionally warm up models
    if warmup and not use_mock:
        import asyncio
        asyncio.create_task(pipeline.warmup())

    return pipeline