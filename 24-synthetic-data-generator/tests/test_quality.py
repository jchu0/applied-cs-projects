"""Tests for quality scoring."""

import pytest
import json

from syntheticdata.schemas import RAGExample, InstructionExample, DifficultyLevel
from syntheticdata.quality import QualityScorer, HallucinationDetector
from syntheticdata.provider import MockProvider


@pytest.fixture
def quality_scorer():
    """Create quality scorer with mock provider."""
    provider = MockProvider([
        json.dumps({
            "accuracy": 9,
            "relevance": 8,
            "completeness": 9,
            "clarity": 8,
            "naturalness": 9,
            "overall_score": 0.85,
            "issues": [],
        })
    ])
    return QualityScorer(provider)


@pytest.fixture
def hallucination_detector():
    """Create hallucination detector with mock provider."""
    provider = MockProvider([
        json.dumps({
            "claims": [
                {"claim": "Paris is capital", "supported": True, "evidence": "stated in context"}
            ],
            "hallucination_score": 0.1,
            "hallucinated_parts": [],
        })
    ])
    return HallucinationDetector(provider)


class TestQualityScorer:
    """Tests for QualityScorer."""

    @pytest.mark.asyncio
    async def test_score_rag_example(self, quality_scorer):
        """Test scoring RAG QA example."""
        example = RAGExample(
            id="test-1",
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
            context="France is a country in Europe. Its capital is Paris.",
        )

        score = await quality_scorer.score(example)

        assert 0 <= score <= 1
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_score_instruction_example(self):
        """Test scoring instruction example."""
        provider = MockProvider([
            json.dumps({
                "accuracy": 8,
                "instruction_clarity": 9,
                "output_quality": 8,
                "format": 9,
                "completeness": 8,
                "overall_score": 0.80,
                "issues": [],
            })
        ])
        scorer = QualityScorer(provider)

        example = InstructionExample(
            id="test-2",
            instruction="Summarize the following text",
            input="Long text about machine learning...",
            output="Machine learning is a subset of AI...",
        )

        score = await scorer.score(example)

        assert score == 0.80

    @pytest.mark.asyncio
    async def test_compare_pair(self, quality_scorer):
        """Test comparing two responses."""
        quality_scorer.model = MockProvider([
            json.dumps({
                "score_a": 8,
                "score_b": 6,
                "winner": "A",
                "reasoning": "Response A is more comprehensive",
            })
        ])

        score_a, score_b = await quality_scorer.compare_pair(
            example_a="Detailed response about topic...",
            example_b="Brief response.",
            prompt="Explain the topic",
        )

        assert score_a == 0.8
        assert score_b == 0.6

    @pytest.mark.asyncio
    async def test_score_fallback_on_error(self):
        """Test fallback score on parse error."""
        provider = MockProvider(["invalid json"])
        scorer = QualityScorer(provider)

        example = RAGExample(
            id="test",
            question="Test?",
            answer="Test.",
            context="Context",
        )

        score = await scorer.score(example)

        # Should return default 0.5 on error
        assert score == 0.5


class TestHallucinationDetector:
    """Tests for HallucinationDetector."""

    @pytest.mark.asyncio
    async def test_detect_no_hallucination(self, hallucination_detector):
        """Test detecting no hallucination."""
        score = await hallucination_detector.detect(
            answer="Paris is the capital of France.",
            context="France is a country in Europe. Its capital is Paris.",
        )

        assert score == 0.1  # Low hallucination score

    @pytest.mark.asyncio
    async def test_detect_hallucination(self):
        """Test detecting hallucination."""
        provider = MockProvider([
            json.dumps({
                "claims": [
                    {"claim": "Berlin is capital", "supported": False, "evidence": "not in context"}
                ],
                "hallucination_score": 0.9,
                "hallucinated_parts": ["Berlin is the capital"],
            })
        ])
        detector = HallucinationDetector(provider)

        score = await detector.detect(
            answer="Berlin is the capital of France.",
            context="France is a country in Europe. Its capital is Paris.",
        )

        assert score == 0.9  # High hallucination score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
