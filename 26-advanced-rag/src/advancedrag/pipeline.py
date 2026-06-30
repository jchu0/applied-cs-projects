"""Main RAG pipeline orchestration."""

import time
from typing import Optional

from .schemas import (
    Document, RAGResult, RewrittenQuery, ConstructedContext, GeneratedAnswer, Citation
)
from .retrieval.hybrid import HybridRetriever
from .retrieval.vector import EmbeddingModel, VectorStore, SimpleVectorStore, MockEmbedding
from .query.rewriter import QueryRewriter, RuleBasedRewriter
from .reranking.reranker import BaseReranker, MockReranker


class RAGPipeline:
    """Main Advanced RAG pipeline."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: BaseReranker,
        query_rewriter: Optional[QueryRewriter] = None,
        llm_client = None,
        config: dict = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.query_rewriter = query_rewriter or RuleBasedRewriter()
        self.llm = llm_client
        self.config = config or {
            "top_k_retrieval": 100,
            "top_k_rerank": 10,
            "max_context_tokens": 4000,
        }

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict = None,
        rewrite_query: bool = True,
    ) -> RAGResult:
        """Execute full RAG pipeline.

        Args:
            query: User query
            top_k: Number of final results
            filter_dict: Metadata filter
            rewrite_query: Whether to rewrite query

        Returns:
            Complete RAG result
        """
        start_time = time.time()

        # Step 1: Query rewriting
        if rewrite_query:
            rewritten = await self.query_rewriter.rewrite(query)
            search_query = rewritten.rewritten
        else:
            rewritten = None
            search_query = query

        # Step 2: Hybrid retrieval
        retrieval_results = self.retriever.search(
            search_query,
            top_k=self.config["top_k_retrieval"],
            filter_dict=filter_dict,
        )

        # Step 3: Reranking
        reranked = await self.reranker.rerank(
            query,  # Use original query for reranking
            retrieval_results,
            top_k=self.config["top_k_rerank"],
        )

        # Step 4: Context construction
        context = self._construct_context(reranked[:top_k])

        # Step 5: Answer generation
        if self.llm:
            answer = await self._generate_answer(query, context)
        else:
            # Mock answer for testing
            answer = GeneratedAnswer(
                answer=f"Based on the {len(reranked)} retrieved documents, here is the answer to: {query}",
                citations=[
                    Citation(
                        source_id=r.document.id,
                        source_title=r.document.metadata.get("title", f"Source {i+1}"),
                        quoted_text=r.document.content[:100],
                        relevance_score=r.relevance_score,
                    )
                    for i, r in enumerate(reranked[:3])
                ],
                confidence=0.8,
                hallucination_flags=[],
                metadata={},
            )

        latency_ms = (time.time() - start_time) * 1000

        return RAGResult(
            query=query,
            rewritten_query=rewritten,
            retrieval_results=retrieval_results[:top_k],
            reranked_results=reranked[:top_k],
            context=context,
            answer=answer,
            latency_ms=latency_ms,
            metadata={
                "total_retrieved": len(retrieval_results),
                "total_reranked": len(reranked),
            },
        )

    def _construct_context(self, results) -> ConstructedContext:
        """Construct context from reranked results."""
        if not results:
            return ConstructedContext(
                content="",
                source_documents=[],
                compression_ratio=1.0,
                token_count=0,
            )

        # Simple context construction
        context_parts = []
        source_docs = []

        for i, result in enumerate(results):
            context_parts.append(f"[{i+1}] {result.document.content}")
            source_docs.append(result.document)

        content = "\n\n".join(context_parts)

        return ConstructedContext(
            content=content,
            source_documents=source_docs,
            compression_ratio=1.0,
            token_count=len(content.split()),
        )

    async def _generate_answer(self, query: str, context: ConstructedContext) -> GeneratedAnswer:
        """Generate answer using LLM."""
        prompt = f"""Answer the question based on the provided context.
Use [1], [2], etc. to cite sources.

Context:
{context.content}

Question: {query}

Answer:"""

        response = await self.llm.generate(prompt)

        return GeneratedAnswer(
            answer=response,
            citations=[
                Citation(
                    source_id=doc.id,
                    source_title=doc.metadata.get("title", f"Source {i+1}"),
                    quoted_text=doc.content[:100],
                    relevance_score=0.8,
                )
                for i, doc in enumerate(context.source_documents)
            ],
            confidence=0.8,
            hallucination_flags=[],
            metadata={},
        )

    def add_documents(self, documents: list[Document]):
        """Add documents to the retriever."""
        self.retriever.add_documents(documents)

    def delete_documents(self, doc_ids: list[str]):
        """Delete documents from the retriever."""
        self.retriever.delete(doc_ids)


def create_pipeline(
    embedding_model: EmbeddingModel = None,
    vector_store: VectorStore = None,
    reranker: BaseReranker = None,
    query_rewriter: QueryRewriter = None,
    llm_client = None,
    config: dict = None,
) -> RAGPipeline:
    """Factory function to create RAG pipeline.

    Args:
        embedding_model: Embedding model (default: MockEmbedding)
        vector_store: Vector store (default: SimpleVectorStore)
        reranker: Reranker (default: MockReranker)
        query_rewriter: Query rewriter (default: RuleBasedRewriter)
        llm_client: LLM client for generation
        config: Pipeline configuration

    Returns:
        Configured RAG pipeline
    """
    # Defaults
    embedding_model = embedding_model or MockEmbedding()
    vector_store = vector_store or SimpleVectorStore()
    reranker = reranker or MockReranker()
    query_rewriter = query_rewriter or RuleBasedRewriter()

    # Create hybrid retriever
    retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    return RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        query_rewriter=query_rewriter,
        llm_client=llm_client,
        config=config,
    )
