"""Context construction for RAG generation."""

from typing import Optional
import numpy as np

from ..schemas import Document, RerankResult, ConstructedContext


class ContextConstructor:
    """Constructs optimized context from retrieved documents."""

    def __init__(
        self,
        max_tokens: int = 4000,
        compressor = None,
        deduplicator = None,
    ):
        self.max_tokens = max_tokens
        self.compressor = compressor or LLMCompressor()
        self.deduplicator = deduplicator or SemanticDeduplicator()

    async def construct(
        self,
        query: str,
        reranked_results: list[RerankResult],
    ) -> ConstructedContext:
        """Construct context from reranked results.

        Args:
            query: User query
            reranked_results: Reranked documents

        Returns:
            Constructed context
        """
        # Extract documents
        documents = [r.document for r in reranked_results]

        if not documents:
            return ConstructedContext(
                content="",
                source_documents=[],
                compression_ratio=1.0,
                token_count=0,
            )

        # Step 1: Deduplicate
        deduplicated = await self.deduplicator.deduplicate(documents)

        # Step 2: Compress each document
        compressed = await self._compress_documents(query, deduplicated)

        # Step 3: Assemble within token budget
        context, used_docs = self._assemble_context(compressed)

        # Calculate stats
        original_tokens = sum(self._count_tokens(d.content) for d in documents)
        final_tokens = self._count_tokens(context)

        return ConstructedContext(
            content=context,
            source_documents=used_docs,
            compression_ratio=final_tokens / original_tokens if original_tokens > 0 else 1.0,
            token_count=final_tokens,
            metadata={
                "original_count": len(documents),
                "used_count": len(used_docs),
                "dedup_removed": len(documents) - len(deduplicated),
            },
        )

    async def _compress_documents(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        """Compress documents."""
        compressed = []
        for doc in documents:
            compressed_content = await self.compressor.compress(query, doc.content)
            compressed.append(Document(
                id=doc.id,
                content=compressed_content,
                metadata={**doc.metadata, "compressed": True},
            ))
        return compressed

    def _assemble_context(
        self,
        documents: list[Document],
    ) -> tuple[str, list[Document]]:
        """Assemble context within token budget."""
        context_parts = []
        used_docs = []
        current_tokens = 0

        for i, doc in enumerate(documents):
            doc_tokens = self._count_tokens(doc.content)

            if current_tokens + doc_tokens > self.max_tokens:
                # Try to fit partial content
                remaining = self.max_tokens - current_tokens
                if remaining > 100:
                    truncated = self._truncate_to_tokens(doc.content, remaining)
                    context_parts.append(f"[Source {i+1}]\n{truncated}\n")
                    used_docs.append(doc)
                break

            context_parts.append(f"[Source {i+1}]\n{doc.content}\n")
            used_docs.append(doc)
            current_tokens += doc_tokens

        return "\n".join(context_parts), used_docs

    def _count_tokens(self, text: str) -> int:
        """Estimate token count (simple word-based)."""
        return len(text.split())

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        words = text.split()
        return " ".join(words[:max_tokens]) + "..."


class LLMCompressor:
    """Compress documents using LLM while preserving relevant info."""

    def __init__(self, llm_client=None, target_ratio: float = 0.5):
        self.llm = llm_client
        self.target_ratio = target_ratio

    async def compress(self, query: str, content: str) -> str:
        """Compress content.

        If no LLM client, uses extractive compression.
        """
        if self.llm:
            return await self._llm_compress(query, content)
        else:
            return self._extractive_compress(query, content)

    async def _llm_compress(self, query: str, content: str) -> str:
        """Compress using LLM."""
        prompt = f"""Compress the following text while preserving information relevant to the query.

Query: {query}

Text:
{content}

Compressed text (keep ~{int(self.target_ratio * 100)}% of original):"""

        return await self.llm.generate(prompt)

    def _extractive_compress(self, query: str, content: str) -> str:
        """Simple extractive compression by selecting relevant sentences."""
        # Split into sentences
        sentences = content.replace('\n', ' ').split('. ')

        # Score sentences by query term overlap
        query_terms = set(query.lower().split())
        scored = []

        for sentence in sentences:
            if not sentence.strip():
                continue
            sentence_terms = set(sentence.lower().split())
            overlap = len(query_terms & sentence_terms)
            scored.append((overlap, sentence))

        # Sort by relevance and take top portion
        scored.sort(reverse=True)
        target_count = max(1, int(len(scored) * self.target_ratio))
        selected = [s for _, s in scored[:target_count]]

        return '. '.join(selected)


class SemanticDeduplicator:
    """Remove semantically duplicate content."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold

    async def deduplicate(self, documents: list[Document]) -> list[Document]:
        """Remove duplicate documents based on content similarity.

        Uses simple text overlap for efficiency.
        """
        if len(documents) <= 1:
            return documents

        unique = [documents[0]]

        for doc in documents[1:]:
            is_duplicate = False

            for unique_doc in unique:
                similarity = self._text_similarity(doc.content, unique_doc.content)
                if similarity > self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(doc)

        return unique

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using word overlap (Jaccard)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
