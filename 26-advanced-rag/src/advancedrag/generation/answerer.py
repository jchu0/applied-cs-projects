"""LLM answer generation with citations and hallucination detection."""

import re
from typing import Optional

from ..schemas import (
    Document, Citation, ConstructedContext, GeneratedAnswer
)


class LLMAnswerer:
    """Generate answers with citations and quality checks."""

    def __init__(
        self,
        llm_client,
        citation_extractor=None,
        hallucination_detector=None,
    ):
        self.llm = llm_client
        self.citation_extractor = citation_extractor or CitationExtractor()
        self.hallucination_detector = hallucination_detector or HallucinationDetector(llm_client)

    async def generate(
        self,
        query: str,
        context: ConstructedContext,
    ) -> GeneratedAnswer:
        """Generate answer with citations and hallucination check.

        Args:
            query: User query
            context: Constructed context

        Returns:
            Generated answer with metadata
        """
        # Generate answer with citations
        raw_answer = await self._generate_with_citations(query, context)

        # Extract citations
        answer_text, citations = self.citation_extractor.extract(
            raw_answer,
            context.source_documents,
        )

        # Detect hallucinations
        hallucination_flags = await self.hallucination_detector.detect(
            query, answer_text, context.content
        )

        # Compute confidence
        confidence = self._compute_confidence(
            citations, hallucination_flags, context
        )

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            hallucination_flags=hallucination_flags,
            metadata={
                "sources_used": len(context.source_documents),
                "context_tokens": context.token_count,
            },
        )

    async def _generate_with_citations(
        self,
        query: str,
        context: ConstructedContext,
    ) -> str:
        """Generate answer with inline citations."""
        prompt = f"""Answer the question based ONLY on the provided sources.

Question: {query}

Sources:
{context.content}

Instructions:
1. Answer directly and comprehensively
2. Use [1], [2], etc. to cite sources inline
3. If information is not in sources, say "I don't have information about..."
4. Never make up information not in the sources
5. If sources conflict, mention the discrepancy

Answer:"""

        return await self.llm.generate(prompt)

    def _compute_confidence(
        self,
        citations: list[Citation],
        hallucination_flags: list[str],
        context: ConstructedContext,
    ) -> float:
        """Compute answer confidence score."""
        # Base confidence from citations
        citation_score = min(len(citations) / 3, 1.0) * 0.4

        # Penalty for hallucinations
        hallucination_penalty = len(hallucination_flags) * 0.15

        # Bonus for relevance
        relevance_bonus = sum(
            c.relevance_score for c in citations
        ) / max(len(citations), 1) * 0.3

        # Compression bonus
        compression_bonus = (1 - context.compression_ratio) * 0.1

        confidence = citation_score + relevance_bonus + compression_bonus - hallucination_penalty
        return max(0.0, min(1.0, confidence))


class HallucinationDetector:
    """Detect unsupported claims in generated answers."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    async def detect(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> list[str]:
        """Detect hallucinated claims.

        Args:
            query: Original query
            answer: Generated answer
            context: Source context

        Returns:
            List of unsupported claims
        """
        if not self.llm:
            return self._rule_based_detect(answer, context)

        # Extract claims
        claims = await self._extract_claims(answer)

        # Verify each claim
        unsupported = []
        for claim in claims:
            is_supported = await self._verify_claim(claim, context)
            if not is_supported:
                unsupported.append(claim)

        return unsupported

    async def _extract_claims(self, answer: str) -> list[str]:
        """Extract factual claims from answer."""
        prompt = f"""Extract all factual claims from this answer.

Answer: {answer}

List each distinct factual claim on a separate line.
Only include verifiable claims (not opinions).

Claims:"""

        response = await self.llm.generate(prompt)
        return [c.strip() for c in response.split('\n') if c.strip()]

    async def _verify_claim(self, claim: str, context: str) -> bool:
        """Verify if claim is supported by context."""
        prompt = f"""Determine if this claim is supported by the context.

Claim: {claim}

Context:
{context[:2000]}

Answer with only "SUPPORTED" or "NOT SUPPORTED"."""

        response = await self.llm.generate(prompt)
        return "SUPPORTED" in response.upper()

    def _rule_based_detect(self, answer: str, context: str) -> list[str]:
        """Simple rule-based hallucination detection."""
        issues = []

        # Check for numbers not in context
        answer_numbers = set(re.findall(r'\b\d+\.?\d*\b', answer))
        context_numbers = set(re.findall(r'\b\d+\.?\d*\b', context))

        unknown_numbers = answer_numbers - context_numbers
        for num in list(unknown_numbers)[:3]:
            issues.append(f"Number '{num}' not found in sources")

        # Check for quotes not in context
        quotes = re.findall(r'"([^"]+)"', answer)
        for quote in quotes:
            if quote.lower() not in context.lower() and len(quote) > 10:
                issues.append(f"Quote not found in sources: {quote[:50]}...")

        return issues


class CitationExtractor:
    """Extract and validate citations from generated text."""

    def __init__(self):
        pass

    def extract(
        self,
        answer: str,
        source_documents: list[Document],
    ) -> tuple[str, list[Citation]]:
        """Extract citations from answer.

        Args:
            answer: Generated answer with citation markers
            source_documents: Source documents

        Returns:
            Tuple of (answer_text, citations)
        """
        # Find citation markers [1], [2], etc.
        pattern = r'\[(\d+)\]'
        markers = re.findall(pattern, answer)

        citations = []
        seen_markers = set()

        for marker in markers:
            if marker in seen_markers:
                continue
            seen_markers.add(marker)

            idx = int(marker) - 1
            if 0 <= idx < len(source_documents):
                doc = source_documents[idx]

                # Find quoted text near citation
                quoted = self._find_relevant_text(answer, marker, doc.content)

                citations.append(Citation(
                    source_id=doc.id,
                    source_title=doc.metadata.get("title", f"Source {marker}"),
                    quoted_text=quoted,
                    relevance_score=self._compute_relevance(quoted, doc.content),
                ))

        return answer, citations

    def _find_relevant_text(
        self,
        answer: str,
        marker: str,
        doc_content: str,
    ) -> str:
        """Find text in answer that relates to the citation."""
        # Find sentence containing the citation marker
        pattern = rf'[^.]*\[{marker}\][^.]*\.'
        matches = re.findall(pattern, answer)

        if matches:
            # Clean up the sentence
            sentence = matches[0].strip()
            sentence = re.sub(r'\[\d+\]', '', sentence).strip()
            return sentence[:200]

        return doc_content[:100]

    def _compute_relevance(self, quoted: str, doc_content: str) -> float:
        """Compute relevance score between quote and document."""
        if not quoted or not doc_content:
            return 0.5

        quoted_words = set(quoted.lower().split())
        doc_words = set(doc_content.lower().split())

        if not quoted_words:
            return 0.5

        overlap = len(quoted_words & doc_words) / len(quoted_words)
        return min(1.0, overlap + 0.3)  # Boost since it's from the doc
