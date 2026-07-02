"""Summarizer SLM for faithful summarization."""

from typing import List, Optional
import re

from .base import BaseSLM, GenerativeModelMixin, logger, require_torch
from ..schemas import RerankResult


def __getattr__(name):
    """Lazily resolve ``torch`` so mock/offline imports don't require it."""
    if name == "torch":
        return require_torch()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class SummarizerSLM(BaseSLM, GenerativeModelMixin):
    """Faithful summarization with extraction + abstraction using SLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        max_summary_length: int = 200,
        temperature: float = 0.3,
        top_p: float = 0.9
    ):
        super().__init__(model_name)
        self.max_length = max_summary_length
        self.temperature = temperature
        self.top_p = top_p

    def _load_model(self):
        """Load the generative model for summarization."""
        self._load_generative_model(self.model_name, load_in_8bit=True)

    async def process(
        self,
        query: str = None,
        rerank: List[RerankResult] = None,
        method: str = "hybrid",
        max_documents: int = 5,
        **kwargs
    ) -> str:
        """Summarize documents for query using actual language model.

        Args:
            query: Original query
            rerank: Reranked results to summarize
            method: extractive, abstractive, or hybrid
            max_documents: Maximum number of documents to summarize

        Returns:
            Summary text
        """
        if not query or not rerank:
            return ""

        if not self._loaded:
            self.load()

        # Get top documents
        documents = [r.document for r in rerank[:max_documents]]

        if method == "extractive":
            return await self._extractive_summarize(documents, query)
        elif method == "abstractive":
            return await self._abstractive_summarize(documents, query)
        else:
            return await self._hybrid_summarize(documents, query)

    async def _extractive_summarize(self, documents, query: str) -> str:
        """Extract key sentences relevant to query."""
        if not documents:
            return ""

        # Extract relevant sentences from each document
        all_sentences = []
        for doc in documents:
            sentences = self._extract_sentences(doc.content)
            # Score sentences based on query relevance
            scored = []
            for sent in sentences:
                score = self._score_sentence_relevance(sent, query)
                scored.append((sent, score))

            # Sort by relevance and take top sentences
            scored.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s for s, _ in scored[:3]]  # Top 3 per document
            all_sentences.extend(top_sentences)

        # Deduplicate while preserving order
        seen = set()
        unique_sentences = []
        for sent in all_sentences:
            if sent not in seen:
                seen.add(sent)
                unique_sentences.append(sent)

        # Combine into summary
        summary = ' '.join(unique_sentences[:5])  # Top 5 overall

        # Truncate if too long
        if len(summary) > self.max_length * 5:  # Rough character count
            summary = summary[:self.max_length * 5] + "..."

        return summary

    async def _abstractive_summarize(self, documents, query: str) -> str:
        """Generate abstractive summary using language model."""
        if not documents:
            return ""

        # Prepare context from documents
        context = self._prepare_context(documents, max_length=1500)

        # Create prompt for summarization
        prompt = f"""Summarize the following documents to answer the query.

Query: {query}

Documents:
{context}

Provide a concise summary (max {self.max_length} words) that directly answers the query:
Summary:"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.device)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                summary = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            # Clean up the summary
            summary = self._clean_summary(summary)
            return summary

        except Exception as e:
            logger.error(f"Failed to generate abstractive summary: {str(e)}")
            # Fallback to extractive
            return await self._extractive_summarize(documents, query)

    async def _hybrid_summarize(self, documents, query: str) -> str:
        """Extract then abstract - combine both approaches."""
        if not documents:
            return ""

        # First extract key information
        extracted = await self._extractive_summarize(documents, query)

        # Then create coherent summary using the language model
        prompt = f"""Rewrite the following extracted information into a coherent summary that answers the query.

Query: {query}

Extracted information:
{extracted}

Coherent summary (max {self.max_length // 2} words):"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                ).to(self.device)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=self.max_length // 2,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                refined = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            return self._clean_summary(refined)

        except Exception as e:
            logger.warning(f"Failed to refine summary: {str(e)}")
            return extracted

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Use regex for sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out very short sentences
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _score_sentence_relevance(self, sentence: str, query: str) -> float:
        """Score sentence relevance to query."""
        sentence_lower = sentence.lower()
        query_terms = set(query.lower().split())

        # Count query term matches
        matches = sum(1 for term in query_terms if term in sentence_lower)

        # Normalize by query length
        score = matches / (len(query_terms) + 1)

        # Boost for exact phrase match
        if query.lower() in sentence_lower:
            score *= 2

        return min(1.0, score)

    def _prepare_context(self, documents, max_length: int = 1500) -> str:
        """Prepare document context for summarization."""
        context_parts = []
        current_length = 0

        for i, doc in enumerate(documents, 1):
            # Truncate individual documents if needed
            content = doc.content
            if len(content) > max_length // len(documents):
                content = content[:max_length // len(documents)] + "..."

            context_parts.append(f"Document {i}: {content}")
            current_length += len(content)

            if current_length > max_length:
                break

        return "\n\n".join(context_parts)

    def _clean_summary(self, summary: str) -> str:
        """Clean up generated summary."""
        # Remove any repeated phrases
        lines = summary.split('.')
        unique_lines = []
        seen = set()

        for line in lines:
            line = line.strip()
            if line and line.lower() not in seen:
                seen.add(line.lower())
                unique_lines.append(line)

        summary = '. '.join(unique_lines)

        # Ensure proper ending
        if summary and not summary[-1] in '.!?':
            summary += '.'

        return summary

    async def generate_query_focused_summary(
        self,
        query: str,
        documents: List,
        focus: str = "relevance"
    ) -> str:
        """Generate summary with specific focus.

        Args:
            query: User query
            documents: Documents to summarize
            focus: Focus type (relevance, completeness, brevity)

        Returns:
            Focused summary
        """
        if not self._loaded:
            self.load()

        context = self._prepare_context(documents, max_length=1000)

        # Adjust prompt based on focus
        if focus == "relevance":
            instruction = "Focus only on information directly relevant to the query"
        elif focus == "completeness":
            instruction = "Provide a comprehensive summary covering all aspects"
        elif focus == "brevity":
            instruction = "Provide the briefest possible answer"
        else:
            instruction = "Provide a balanced summary"

        prompt = f"""Create a summary following this instruction: {instruction}

Query: {query}

Documents:
{context}

Summary:"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1500
                ).to(self.device)

                max_tokens = 50 if focus == "brevity" else self.max_length

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=max_tokens,
                    temperature=0.2 if focus == "brevity" else self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                summary = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            return self._clean_summary(summary)

        except Exception as e:
            logger.error(f"Failed to generate focused summary: {str(e)}")
            return f"Summary generation failed: {str(e)}"


class MockSummarizerSLM(BaseSLM):
    """Mock summarizer for testing."""

    def __init__(self):
        super().__init__("mock")
        self._loaded = True

    async def process(
        self,
        query: str = None,
        rerank: List[RerankResult] = None,
        **kwargs
    ) -> str:
        if not rerank:
            return "No documents to summarize."

        # Simple mock summary
        num_docs = len(rerank)
        return f"Mock summary of {num_docs} documents for query: {query}"


class ChainOfDensitySummarizer(SummarizerSLM):
    """Implements Chain of Density summarization for progressively detailed summaries."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        max_summary_length: int = 200,
        density_levels: int = 3
    ):
        super().__init__(model_name, max_summary_length)
        self.density_levels = density_levels

    async def process(
        self,
        query: str = None,
        rerank: List[RerankResult] = None,
        **kwargs
    ) -> str:
        """Generate progressively denser summaries.

        Args:
            query: User query
            rerank: Documents to summarize

        Returns:
            Final dense summary
        """
        if not query or not rerank:
            return ""

        if not self._loaded:
            self.load()

        documents = [r.document for r in rerank[:3]]  # Use top 3 documents
        context = self._prepare_context(documents, max_length=1000)

        current_summary = ""

        # Generate progressively denser summaries
        for level in range(1, self.density_levels + 1):
            density_instruction = self._get_density_instruction(level, self.density_levels)

            if level == 1:
                prompt = f"""Summarize these documents to answer the query.
{density_instruction}

Query: {query}

Documents:
{context}

Summary (exactly {self.max_length // self.density_levels} words):"""
            else:
                prompt = f"""Make this summary denser by adding more key information.
{density_instruction}

Query: {query}

Current summary: {current_summary}

Original documents:
{context}

Denser summary (exactly {self.max_length * level // self.density_levels} words):"""

            try:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1500
                    ).to(self.device)

                    outputs = self.model.generate(
                        inputs['input_ids'],
                        max_new_tokens=self.max_length * level // self.density_levels,
                        temperature=0.3,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                    current_summary = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()

            except Exception as e:
                logger.warning(f"Failed at density level {level}: {str(e)}")
                break

        return self._clean_summary(current_summary)

    def _get_density_instruction(self, level: int, max_level: int) -> str:
        """Get instruction for current density level."""
        if level == 1:
            return "Start with a high-level overview, focusing on the main topic."
        elif level == max_level:
            return "Include all critical details, key facts, and specific information."
        else:
            return "Add more specific details and key information while maintaining coherence."