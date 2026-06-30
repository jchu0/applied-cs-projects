"""Tests for query rewriting module."""

import pytest
from unittest.mock import Mock, AsyncMock

from advancedrag.schemas import QueryIntent, RewrittenQuery
from advancedrag.query.rewriter import (
    QueryRewriter,
    LLMQueryRewriter,
    RuleBasedRewriter,
    HybridQueryRewriter,
    MockLLMClient,
)


class TestRuleBasedRewriter:
    """Test rule-based query rewriting."""

    @pytest.fixture
    def rewriter(self):
        """Create rule-based rewriter instance."""
        return RuleBasedRewriter()

    @pytest.mark.asyncio
    async def test_acronym_expansion_ml(self, rewriter):
        """Test ML acronym is expanded."""
        result = await rewriter.rewrite("What is ML?")

        assert "machine learning" in result.rewritten.lower()
        assert result.original == "What is ML?"

    @pytest.mark.asyncio
    async def test_acronym_expansion_ai(self, rewriter):
        """Test AI acronym is expanded."""
        result = await rewriter.rewrite("How does AI work?")

        assert "artificial intelligence" in result.rewritten.lower()

    @pytest.mark.asyncio
    async def test_acronym_expansion_nlp(self, rewriter):
        """Test NLP acronym is expanded."""
        result = await rewriter.rewrite("Explain NLP techniques")

        assert "natural language processing" in result.rewritten.lower()

    @pytest.mark.asyncio
    async def test_acronym_expansion_llm(self, rewriter):
        """Test LLM acronym is expanded."""
        result = await rewriter.rewrite("How do LLM models work?")

        assert "large language model" in result.rewritten.lower()

    @pytest.mark.asyncio
    async def test_acronym_expansion_rag(self, rewriter):
        """Test RAG acronym is expanded."""
        result = await rewriter.rewrite("What is RAG?")

        assert "retrieval augmented generation" in result.rewritten.lower()

    @pytest.mark.asyncio
    async def test_multiple_acronyms(self, rewriter):
        """Test multiple acronyms in one query."""
        result = await rewriter.rewrite("Compare ML and AI approaches")

        assert "machine learning" in result.rewritten.lower()
        assert "artificial intelligence" in result.rewritten.lower()

    @pytest.mark.asyncio
    async def test_intent_detection_procedural(self, rewriter):
        """Test procedural intent detection."""
        result = await rewriter.rewrite("How to build a neural network?")

        assert result.intent == QueryIntent.PROCEDURAL

    @pytest.mark.asyncio
    async def test_intent_detection_comparison(self, rewriter):
        """Test comparison intent detection."""
        result = await rewriter.rewrite("What is the difference between CNN and RNN?")

        assert result.intent == QueryIntent.COMPARISON

    @pytest.mark.asyncio
    async def test_intent_detection_clarification(self, rewriter):
        """Test clarification intent detection."""
        result = await rewriter.rewrite("What is deep learning?")

        assert result.intent == QueryIntent.CLARIFICATION

    @pytest.mark.asyncio
    async def test_intent_detection_comparison_vs(self, rewriter):
        """Test comparison intent with 'vs' keyword."""
        result = await rewriter.rewrite("Python vs Java for ML")

        assert result.intent == QueryIntent.COMPARISON

    @pytest.mark.asyncio
    async def test_default_intent_factual(self, rewriter):
        """Test default intent is factual."""
        result = await rewriter.rewrite("TensorFlow documentation")

        assert result.intent == QueryIntent.FACTUAL

    @pytest.mark.asyncio
    async def test_rewritten_query_structure(self, rewriter):
        """Test RewrittenQuery has correct structure."""
        result = await rewriter.rewrite("Test query")

        assert isinstance(result, RewrittenQuery)
        assert hasattr(result, 'original')
        assert hasattr(result, 'rewritten')
        assert hasattr(result, 'intent')
        assert hasattr(result, 'expansions')
        assert hasattr(result, 'sub_queries')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'metadata')

    @pytest.mark.asyncio
    async def test_rule_based_confidence(self, rewriter):
        """Test rule-based rewriter returns appropriate confidence."""
        result = await rewriter.rewrite("Test query")

        assert result.confidence == 0.6
        assert result.metadata.get("method") == "rule_based"

    @pytest.mark.asyncio
    async def test_case_insensitive_acronym_matching(self, rewriter):
        """Test acronym matching is case insensitive."""
        result1 = await rewriter.rewrite("What is ML?")
        result2 = await rewriter.rewrite("What is ml?")

        assert "machine learning" in result1.rewritten.lower()
        assert "machine learning" in result2.rewritten.lower()


class TestMockLLMClient:
    """Test mock LLM client for testing."""

    @pytest.fixture
    def client(self):
        """Create mock LLM client."""
        return MockLLMClient()

    @pytest.mark.asyncio
    async def test_intent_classification_response(self, client):
        """Test mock response for intent classification."""
        response = await client.generate("Please classify the intent of this query")

        assert response == "FACTUAL"

    @pytest.mark.asyncio
    async def test_query_expansion_response(self, client):
        """Test mock response for query expansion."""
        response = await client.generate("Generate related queries for expansion")

        assert "related query" in response.lower()

    @pytest.mark.asyncio
    async def test_query_decomposition_response(self, client):
        """Test mock response for query decomposition."""
        response = await client.generate("Break down this complex query into sub-queries")

        assert "sub query" in response.lower()

    @pytest.mark.asyncio
    async def test_query_rewrite_response(self, client):
        """Test mock response for query rewriting."""
        response = await client.generate("Rewrite this query for better retrieval")

        assert "enhanced" in response.lower() or "keywords" in response.lower()

    @pytest.mark.asyncio
    async def test_default_response(self, client):
        """Test default mock response."""
        response = await client.generate("Some other prompt without keywords")

        assert response == "mock response"


class TestLLMQueryRewriter:
    """Test LLM-based query rewriting."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client with controlled responses."""
        return MockLLMClient()

    @pytest.fixture
    def rewriter(self, mock_llm):
        """Create LLM query rewriter with mock."""
        return LLMQueryRewriter(llm_client=mock_llm)

    @pytest.mark.asyncio
    async def test_basic_rewrite(self, rewriter):
        """Test basic query rewriting."""
        result = await rewriter.rewrite("What is machine learning?")

        assert isinstance(result, RewrittenQuery)
        assert result.original == "What is machine learning?"
        assert len(result.rewritten) > 0

    @pytest.mark.asyncio
    async def test_intent_detection(self, rewriter):
        """Test intent is detected."""
        result = await rewriter.rewrite("What is machine learning?")

        assert result.intent == QueryIntent.FACTUAL

    @pytest.mark.asyncio
    async def test_query_expansion_generated(self, rewriter):
        """Test query expansions are generated."""
        result = await rewriter.rewrite("What is machine learning?")

        assert isinstance(result.expansions, list)
        # Mock returns 3 related queries
        assert len(result.expansions) == 3

    @pytest.mark.asyncio
    async def test_confidence_score(self, rewriter):
        """Test confidence score is set."""
        result = await rewriter.rewrite("What is machine learning?")

        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_metadata_context_tracking(self, rewriter):
        """Test metadata tracks context usage."""
        result_no_ctx = await rewriter.rewrite("Query without context")
        result_with_ctx = await rewriter.rewrite("Query with context", context={"user": "test"})

        assert result_no_ctx.metadata.get("context_used") is False
        assert result_with_ctx.metadata.get("context_used") is True

    @pytest.mark.asyncio
    async def test_custom_llm_responses(self):
        """Test with custom LLM mock responses."""
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(side_effect=[
            "COMPARISON",  # Intent detection
            "expansion 1\nexpansion 2\nexpansion 3",  # Query expansion
            "",  # Sub-query decomposition (factual doesn't decompose)
            "improved query with better keywords"  # Rewrite
        ])

        rewriter = LLMQueryRewriter(llm_client=mock_llm)
        result = await rewriter.rewrite("Compare A and B")

        assert result.intent == QueryIntent.COMPARISON
        assert len(result.expansions) == 3
        assert result.rewritten == "improved query with better keywords"

    @pytest.mark.asyncio
    async def test_sub_query_decomposition_for_comparison(self):
        """Test sub-queries are generated for comparison queries."""
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(side_effect=[
            "COMPARISON",
            "expansion 1\nexpansion 2\nexpansion 3",
            "What is A?\nWhat is B?\nHow are A and B different?",
            "improved comparison query"
        ])

        rewriter = LLMQueryRewriter(llm_client=mock_llm)
        result = await rewriter.rewrite("What is the difference between A and B?")

        assert len(result.sub_queries) == 3
        assert "What is A?" in result.sub_queries

    @pytest.mark.asyncio
    async def test_sub_query_decomposition_for_procedural(self):
        """Test sub-queries are generated for procedural queries."""
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(side_effect=[
            "PROCEDURAL",
            "expansion 1\nexpansion 2",
            "Step 1 question\nStep 2 question",
            "improved procedural query"
        ])

        rewriter = LLMQueryRewriter(llm_client=mock_llm)
        result = await rewriter.rewrite("How to build a model?")

        assert result.intent == QueryIntent.PROCEDURAL
        assert len(result.sub_queries) >= 1

    @pytest.mark.asyncio
    async def test_invalid_intent_fallback(self):
        """Test fallback to FACTUAL for invalid intent."""
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(side_effect=[
            "INVALID_INTENT",
            "exp 1\nexp 2\nexp 3",
            "",
            "rewritten"
        ])

        rewriter = LLMQueryRewriter(llm_client=mock_llm)
        result = await rewriter.rewrite("Some query")

        assert result.intent == QueryIntent.FACTUAL


class TestHybridQueryRewriter:
    """Test hybrid query rewriting combining rules and LLM."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        return MockLLMClient()

    @pytest.fixture
    def rewriter(self, mock_llm):
        """Create hybrid query rewriter."""
        return HybridQueryRewriter(llm_client=mock_llm)

    @pytest.mark.asyncio
    async def test_combines_rule_and_llm(self, rewriter):
        """Test hybrid combines rule-based and LLM rewriting."""
        result = await rewriter.rewrite("What is ML?")

        # Should have expanded ML (from rules) and then enhanced (from LLM)
        assert result.metadata.get("rule_applied") is True
        assert result.metadata.get("llm_enhanced") is True

    @pytest.mark.asyncio
    async def test_uses_llm_intent(self, rewriter):
        """Test hybrid uses LLM-detected intent."""
        result = await rewriter.rewrite("Simple query")

        # MockLLMClient returns FACTUAL for intent classification
        assert result.intent == QueryIntent.FACTUAL

    @pytest.mark.asyncio
    async def test_uses_llm_expansions(self, rewriter):
        """Test hybrid uses LLM-generated expansions."""
        result = await rewriter.rewrite("Test query")

        # MockLLMClient returns 3 related queries
        assert len(result.expansions) == 3

    @pytest.mark.asyncio
    async def test_confidence_is_max(self, rewriter):
        """Test confidence is max of both methods."""
        result = await rewriter.rewrite("Test query")

        # Rule-based: 0.6, LLM: 0.8
        assert result.confidence == 0.8


class TestQueryIntent:
    """Test QueryIntent enum."""

    def test_query_intent_values(self):
        """Test all intent values exist."""
        assert QueryIntent.FACTUAL.value == "factual"
        assert QueryIntent.COMPARISON.value == "comparison"
        assert QueryIntent.PROCEDURAL.value == "procedural"
        assert QueryIntent.EXPLORATORY.value == "exploratory"
        assert QueryIntent.CLARIFICATION.value == "clarification"

    def test_intent_from_string(self):
        """Test creating intent from string."""
        intent = QueryIntent("factual")
        assert intent == QueryIntent.FACTUAL

    def test_invalid_intent_raises(self):
        """Test invalid intent raises ValueError."""
        with pytest.raises(ValueError):
            QueryIntent("invalid")


class TestRewrittenQuery:
    """Test RewrittenQuery data structure."""

    def test_rewritten_query_creation(self):
        """Test creating a RewrittenQuery."""
        query = RewrittenQuery(
            original="What is ML?",
            rewritten="What is machine learning?",
            intent=QueryIntent.CLARIFICATION,
            expansions=["machine learning basics", "ML fundamentals"],
            sub_queries=[],
            confidence=0.9,
            metadata={"source": "test"}
        )

        assert query.original == "What is ML?"
        assert query.rewritten == "What is machine learning?"
        assert query.intent == QueryIntent.CLARIFICATION
        assert len(query.expansions) == 2
        assert query.confidence == 0.9

    def test_rewritten_query_default_metadata(self):
        """Test RewrittenQuery with default metadata."""
        query = RewrittenQuery(
            original="test",
            rewritten="test",
            intent=QueryIntent.FACTUAL,
            expansions=[],
            sub_queries=[],
            confidence=0.5
        )

        assert query.metadata == {}


class TestQueryRewriterInterface:
    """Test QueryRewriter abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test QueryRewriter cannot be directly instantiated."""
        with pytest.raises(TypeError):
            QueryRewriter()


class TestEdgeCases:
    """Test edge cases in query rewriting."""

    @pytest.fixture
    def rule_rewriter(self):
        """Create rule-based rewriter."""
        return RuleBasedRewriter()

    @pytest.mark.asyncio
    async def test_empty_query(self, rule_rewriter):
        """Test handling of empty query."""
        result = await rule_rewriter.rewrite("")

        assert result.original == ""
        assert result.rewritten == ""

    @pytest.mark.asyncio
    async def test_whitespace_only_query(self, rule_rewriter):
        """Test handling of whitespace-only query."""
        result = await rule_rewriter.rewrite("   ")

        assert result.original == "   "

    @pytest.mark.asyncio
    async def test_very_long_query(self, rule_rewriter):
        """Test handling of very long query."""
        long_query = "What is " + "machine learning " * 100

        result = await rule_rewriter.rewrite(long_query)

        assert len(result.rewritten) > 0

    @pytest.mark.asyncio
    async def test_special_characters(self, rule_rewriter):
        """Test handling of special characters."""
        result = await rule_rewriter.rewrite("What is @#$% ML?")

        assert "machine learning" in result.rewritten.lower()

    @pytest.mark.asyncio
    async def test_unicode_characters(self, rule_rewriter):
        """Test handling of unicode characters."""
        result = await rule_rewriter.rewrite("What is ML in Chinese: \u673a\u5668\u5b66\u4e60?")

        assert "machine learning" in result.rewritten.lower()

    @pytest.mark.asyncio
    async def test_multiple_question_marks(self, rule_rewriter):
        """Test handling of multiple question marks."""
        result = await rule_rewriter.rewrite("What is AI???")

        assert "artificial intelligence" in result.rewritten.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
