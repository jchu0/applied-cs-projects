"""Query rewriting and enhancement."""

from abc import ABC, abstractmethod
from typing import Optional

from ..schemas import QueryIntent, RewrittenQuery


class QueryRewriter(ABC):
    """Base class for query rewriters."""

    @abstractmethod
    async def rewrite(
        self,
        query: str,
        context: Optional[dict] = None
    ) -> RewrittenQuery:
        """Rewrite query for improved retrieval."""
        pass


class LLMQueryRewriter(QueryRewriter):
    """LLM-based query rewriting."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def rewrite(
        self,
        query: str,
        context: Optional[dict] = None
    ) -> RewrittenQuery:
        """Rewrite query using LLM."""
        # Detect intent
        intent = await self._detect_intent(query)

        # Generate expansions
        expansions = await self._expand_query(query, intent)

        # Decompose if complex
        sub_queries = await self._decompose_query(query, intent)

        # Main rewrite
        rewritten = await self._rewrite_for_retrieval(query, intent, context)

        return RewrittenQuery(
            original=query,
            rewritten=rewritten,
            intent=intent,
            expansions=expansions,
            sub_queries=sub_queries,
            confidence=0.8,
            metadata={"context_used": context is not None},
        )

    async def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent."""
        prompt = f"""Classify the intent of this query into one of these categories:
- FACTUAL: Seeking specific facts or data
- COMPARISON: Comparing entities or concepts
- PROCEDURAL: How-to or step-by-step instructions
- EXPLORATORY: Open-ended exploration
- CLARIFICATION: Seeking explanation

Query: {query}

Return only the category name in uppercase."""

        response = await self.llm.generate(prompt)
        intent_str = response.strip().upper()

        try:
            return QueryIntent(intent_str.lower())
        except ValueError:
            return QueryIntent.FACTUAL

    async def _expand_query(self, query: str, intent: QueryIntent) -> list[str]:
        """Generate query expansions."""
        prompt = f"""Generate 3 semantically related search queries for:
Original: {query}
Intent: {intent.value}

Focus on synonyms, related concepts, and alternative phrasings.
Return one query per line, no numbering."""

        response = await self.llm.generate(prompt)
        return [q.strip() for q in response.split('\n') if q.strip()][:3]

    async def _decompose_query(self, query: str, intent: QueryIntent) -> list[str]:
        """Decompose complex queries into sub-queries."""
        if intent not in [QueryIntent.COMPARISON, QueryIntent.PROCEDURAL]:
            return []

        prompt = f"""Break down this complex query into simple, atomic sub-queries:
Query: {query}

Each sub-query should be independently answerable.
Return one sub-query per line, no numbering."""

        response = await self.llm.generate(prompt)
        return [q.strip() for q in response.split('\n') if q.strip()][:5]

    async def _rewrite_for_retrieval(
        self,
        query: str,
        intent: QueryIntent,
        context: Optional[dict]
    ) -> str:
        """Rewrite query for better retrieval."""
        prompt = f"""Rewrite this query to improve search retrieval:
Original: {query}
Intent: {intent.value}

Guidelines:
- Expand abbreviations and acronyms
- Add relevant keywords
- Make implicit context explicit
- Keep it concise but comprehensive

Rewritten query:"""

        response = await self.llm.generate(prompt)
        return response.strip()


class RuleBasedRewriter(QueryRewriter):
    """Rule-based query rewriting."""

    def __init__(self):
        # Acronym expansions
        self.acronyms = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "nlp": "natural language processing",
            "llm": "large language model",
            "rag": "retrieval augmented generation",
            "api": "application programming interface",
            "db": "database",
            "sql": "structured query language",
        }

        # Query patterns
        self.patterns = [
            (r"what is (\w+)", r"definition of \1, explanation of \1"),
            (r"how to (\w+)", r"steps to \1, guide to \1, tutorial \1"),
            (r"difference between (\w+) and (\w+)", r"\1 vs \2, compare \1 \2"),
        ]

    async def rewrite(
        self,
        query: str,
        context: Optional[dict] = None
    ) -> RewrittenQuery:
        """Apply rule-based rewriting."""
        import re

        rewritten = query.lower()

        # Expand acronyms
        for acronym, expansion in self.acronyms.items():
            pattern = r'\b' + acronym + r'\b'
            rewritten = re.sub(pattern, f"{acronym} ({expansion})", rewritten, flags=re.IGNORECASE)

        # Detect intent from patterns
        intent = QueryIntent.FACTUAL
        if "how to" in query.lower():
            intent = QueryIntent.PROCEDURAL
        elif "difference" in query.lower() or " vs " in query.lower():
            intent = QueryIntent.COMPARISON
        elif "what is" in query.lower() or "explain" in query.lower():
            intent = QueryIntent.CLARIFICATION

        return RewrittenQuery(
            original=query,
            rewritten=rewritten,
            intent=intent,
            expansions=[],
            sub_queries=[],
            confidence=0.6,
            metadata={"method": "rule_based"},
        )


class HybridQueryRewriter(QueryRewriter):
    """Combines rule-based and LLM-based rewriting."""

    def __init__(self, llm_client):
        self.rule_rewriter = RuleBasedRewriter()
        self.llm_rewriter = LLMQueryRewriter(llm_client)

    async def rewrite(
        self,
        query: str,
        context: Optional[dict] = None
    ) -> RewrittenQuery:
        """Apply hybrid rewriting."""
        # First apply rules
        rule_result = await self.rule_rewriter.rewrite(query, context)

        # Then enhance with LLM
        llm_result = await self.llm_rewriter.rewrite(rule_result.rewritten, context)

        # Combine results
        return RewrittenQuery(
            original=query,
            rewritten=llm_result.rewritten,
            intent=llm_result.intent,
            expansions=llm_result.expansions,
            sub_queries=llm_result.sub_queries,
            confidence=max(rule_result.confidence, llm_result.confidence),
            metadata={
                "rule_applied": True,
                "llm_enhanced": True,
            },
        )


class MockLLMClient:
    """Mock LLM client for testing."""

    async def generate(self, prompt: str) -> str:
        """Generate mock response."""
        prompt_lower = prompt.lower()
        # Check for intent classification first (specific keyword "classify")
        if "classify" in prompt_lower:
            return "FACTUAL"
        # Check for query expansion (before checking "intent" which appears in many prompts)
        elif "generate" in prompt_lower and "related" in prompt_lower:
            return "related query 1\nrelated query 2\nrelated query 3"
        elif "decompose" in prompt_lower or "break down" in prompt_lower:
            return "sub query 1\nsub query 2"
        elif "rewrite" in prompt_lower:
            return "enhanced search query with better keywords"
        else:
            return "mock response"
