"""Tests for multi-hop retrieval module."""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock

from advancedrag.schemas import Document, RetrievalResult
from advancedrag.retrieval.multi_hop import MultiHopRetriever


def _make_retrieval_result(doc_id, content="content", score=0.9, rank=0):
    return RetrievalResult(
        document=Document(id=doc_id, content=content, metadata={}),
        score=score,
        retriever_type="test",
        rank=rank,
    )


def _make_mock_retriever(results_per_call):
    """Create a mock retriever that returns different results per call."""
    retriever = Mock()
    retriever.search = Mock(side_effect=results_per_call)
    return retriever


class TestMultiHopRetriever:
    """Test MultiHopRetriever."""

    @pytest.mark.asyncio
    async def test_single_hop_sufficient(self):
        """If first hop returns enough results, no further hops needed."""
        results = [_make_retrieval_result(f"d{i}", score=0.9 - i * 0.1, rank=i) for i in range(10)]
        retriever = _make_mock_retriever([results])
        mhr = MultiHopRetriever(retriever, max_hops=3)
        output = await mhr.retrieve("test query", top_k=5)
        assert len(output) == 5
        assert retriever.search.call_count == 1

    @pytest.mark.asyncio
    async def test_multi_hop_triggered(self):
        """When first hop returns too few, additional hops should be triggered."""
        hop1_results = [_make_retrieval_result("d1", score=0.9, rank=0),
                        _make_retrieval_result("d2", score=0.8, rank=1)]
        hop2_results = [_make_retrieval_result("d3", score=0.85, rank=0),
                        _make_retrieval_result("d4", score=0.75, rank=1),
                        _make_retrieval_result("d5", score=0.7, rank=2)]
        hop3_results = [_make_retrieval_result("d6", score=0.6, rank=0),
                        _make_retrieval_result("d7", score=0.55, rank=1),
                        _make_retrieval_result("d8", score=0.5, rank=2)]
        retriever = _make_mock_retriever([hop1_results, hop2_results, hop3_results])
        mhr = MultiHopRetriever(retriever, max_hops=3)
        output = await mhr.retrieve("explain quantum computing", top_k=5)
        assert len(output) == 5
        assert retriever.search.call_count >= 2

    @pytest.mark.asyncio
    async def test_max_hops_limit(self):
        """Should not exceed max_hops even if results are insufficient."""
        small_results = [_make_retrieval_result(f"d{i}", score=0.5, rank=0) for i in range(1)]
        retriever = _make_mock_retriever([
            [_make_retrieval_result("d1", score=0.5)],
            [_make_retrieval_result("d2", score=0.5)],
        ])
        mhr = MultiHopRetriever(retriever, max_hops=2)
        output = await mhr.retrieve("complex query about many topics", top_k=10)
        assert retriever.search.call_count <= 2

    @pytest.mark.asyncio
    async def test_dedup_across_hops(self):
        """Results with same doc ID across hops should be deduplicated."""
        hop1 = [_make_retrieval_result("d1", score=0.9), _make_retrieval_result("d2", score=0.8)]
        hop2 = [_make_retrieval_result("d1", score=0.7), _make_retrieval_result("d3", score=0.85)]  # d1 repeated
        retriever = _make_mock_retriever([hop1, hop2])
        mhr = MultiHopRetriever(retriever, max_hops=2)
        output = await mhr.retrieve("query about dedup", top_k=10)
        doc_ids = [r.document.id for r in output]
        assert len(doc_ids) == len(set(doc_ids))  # No duplicates
        assert len(output) == 3  # d1, d2, d3

    @pytest.mark.asyncio
    async def test_llm_client_followup(self):
        """With LLM client, should generate follow-up queries via LLM."""
        hop1 = [_make_retrieval_result("d1", score=0.9)]
        hop2 = [_make_retrieval_result("d2", score=0.85), _make_retrieval_result("d3", score=0.8)]
        retriever = _make_mock_retriever([hop1, hop2])

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="follow-up about quantum entanglement")
        mhr = MultiHopRetriever(retriever, llm_client=mock_llm, max_hops=2)
        output = await mhr.retrieve("quantum computing basics", top_k=3)
        assert len(output) == 3
        mock_llm.generate.assert_called()

    @pytest.mark.asyncio
    async def test_rule_based_fallback(self):
        """Without LLM, should generate follow-ups via rule-based heuristics."""
        hop1 = [_make_retrieval_result("d1", score=0.9)]
        hop2 = [_make_retrieval_result("d2", score=0.85), _make_retrieval_result("d3", score=0.8)]
        retriever = _make_mock_retriever([hop1, hop2])
        mhr = MultiHopRetriever(retriever, llm_client=None, max_hops=2)
        output = await mhr.retrieve("explain quantum computing principles", top_k=3)
        assert len(output) == 3
        # Second call should use a rule-based follow-up query
        assert retriever.search.call_count == 2
        second_call_query = retriever.search.call_args_list[1][0][0]
        assert "more details" in second_call_query

    @pytest.mark.asyncio
    async def test_empty_retrieval(self):
        """Should handle empty retrieval results gracefully."""
        retriever = _make_mock_retriever([[], []])  # Two empty hops
        mhr = MultiHopRetriever(retriever, max_hops=2)
        output = await mhr.retrieve("query with no results", top_k=5)
        assert len(output) == 0

    @pytest.mark.asyncio
    async def test_merge_keeps_highest_score(self):
        """Merge should keep the highest score when duplicates are found."""
        hop1 = [_make_retrieval_result("d1", score=0.5)]
        hop2 = [_make_retrieval_result("d1", score=0.9)]  # Higher score for same doc
        retriever = _make_mock_retriever([hop1, hop2])
        mhr = MultiHopRetriever(retriever, max_hops=2)
        output = await mhr.retrieve("query about scoring", top_k=10)
        assert len(output) == 1
        assert output[0].score == 0.9  # Should keep the higher score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
