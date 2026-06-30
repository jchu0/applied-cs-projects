"""Tests for generation module (answerer, hallucination detection, citation extraction)."""

import pytest
from unittest.mock import AsyncMock, Mock

from advancedrag.schemas import Document, Citation, ConstructedContext, GeneratedAnswer
from advancedrag.generation.answerer import (
    LLMAnswerer,
    HallucinationDetector,
    CitationExtractor,
)


class TestCitationExtractor:
    """Test CitationExtractor."""

    @pytest.fixture
    def extractor(self):
        return CitationExtractor()

    def test_extract_markers(self, extractor):
        """Should extract citation markers like [1], [2]."""
        docs = [
            Document(id="d1", content="AI is transforming industries", metadata={"title": "AI paper"}),
            Document(id="d2", content="Machine learning powers modern apps", metadata={"title": "ML paper"}),
        ]
        answer = "AI is transforming everything [1]. ML powers apps [2]."
        text, citations = extractor.extract(answer, docs)
        assert len(citations) == 2
        assert citations[0].source_id == "d1"
        assert citations[1].source_id == "d2"

    def test_no_citations(self, extractor):
        """Answer without citation markers should return empty citations."""
        docs = [Document(id="d1", content="Some content", metadata={})]
        answer = "This answer has no citations at all."
        text, citations = extractor.extract(answer, docs)
        assert len(citations) == 0

    def test_duplicate_markers(self, extractor):
        """Duplicate markers should only produce one citation each."""
        docs = [Document(id="d1", content="AI content", metadata={"title": "Paper"})]
        answer = "First point [1]. Another point [1]. More [1]."
        text, citations = extractor.extract(answer, docs)
        assert len(citations) == 1

    def test_relevance_score(self, extractor):
        """Citations should have relevance scores."""
        docs = [Document(id="d1", content="Machine learning algorithms process data efficiently", metadata={"title": "ML"})]
        answer = "Machine learning algorithms are efficient [1]."
        text, citations = extractor.extract(answer, docs)
        assert len(citations) == 1
        assert 0.0 <= citations[0].relevance_score <= 1.0


class TestHallucinationDetector:
    """Test HallucinationDetector."""

    @pytest.mark.asyncio
    async def test_rule_based_numbers(self):
        """Rule-based detection should flag numbers not in context."""
        detector = HallucinationDetector(llm_client=None)
        answer = "The model achieved 99.5% accuracy on 1000 samples."
        context = "The model was tested on samples and showed good results."
        issues = await detector.detect("accuracy", answer, context)
        assert len(issues) > 0
        assert any("not found" in issue.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_rule_based_quotes(self):
        """Rule-based detection should flag fabricated quotes."""
        detector = HallucinationDetector(llm_client=None)
        answer = 'The paper states "this is a completely fabricated quotation from nowhere".'
        context = "The paper discusses machine learning approaches."
        issues = await detector.detect("paper content", answer, context)
        assert len(issues) > 0
        assert any("quote" in issue.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_llm_detection_with_mock(self):
        """LLM-based detection should extract and verify claims."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=[
            "AI is transforming healthcare.\nAI reduces costs by 50%.",  # claims
            "SUPPORTED",  # first claim verified
            "UNVERIFIABLE",  # second claim not supported (no "SUPPORTED" substring)
        ])
        detector = HallucinationDetector(llm_client=mock_llm)
        issues = await detector.detect("AI in healthcare", "AI is transforming healthcare. AI reduces costs by 50%.", "AI is being used in healthcare.")
        assert len(issues) == 1
        assert "50%" in issues[0]

    @pytest.mark.asyncio
    async def test_clean_input(self):
        """Clean answer matching context should have no issues."""
        detector = HallucinationDetector(llm_client=None)
        answer = "Machine learning processes data."
        context = "Machine learning processes data efficiently."
        issues = await detector.detect("ML", answer, context)
        assert len(issues) == 0


class TestLLMAnswerer:
    """Test LLMAnswerer."""

    @pytest.mark.asyncio
    async def test_generate_with_mock_llm(self):
        """Should generate answer using LLM client."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="Based on source [1], AI is transformative.")
        answerer = LLMAnswerer(llm_client=mock_llm)

        context = ConstructedContext(
            content="[Source 1]\nAI is transforming industries.",
            source_documents=[Document(id="d1", content="AI is transforming industries", metadata={"title": "AI Paper"})],
            compression_ratio=0.8,
            token_count=10,
        )
        result = await answerer.generate("What is AI?", context)
        assert isinstance(result, GeneratedAnswer)
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_confidence_computation(self):
        """Confidence should be computed based on citations and hallucinations."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="AI is great [1]. Very powerful [2].")
        answerer = LLMAnswerer(llm_client=mock_llm)

        context = ConstructedContext(
            content="[Source 1]\nAI is great\n[Source 2]\nAI is powerful",
            source_documents=[
                Document(id="d1", content="AI is great", metadata={"title": "S1"}),
                Document(id="d2", content="AI is powerful", metadata={"title": "S2"}),
            ],
            compression_ratio=0.5,
            token_count=20,
        )
        result = await answerer.generate("Tell me about AI", context)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_citations_extracted(self):
        """Citations should be extracted from generated answer."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="According to research [1], AI is advancing.")
        answerer = LLMAnswerer(llm_client=mock_llm)

        context = ConstructedContext(
            content="[Source 1]\nAI research is advancing rapidly",
            source_documents=[Document(id="d1", content="AI research is advancing rapidly", metadata={"title": "Research"})],
            compression_ratio=1.0,
            token_count=10,
        )
        result = await answerer.generate("AI advancement", context)
        assert len(result.citations) >= 1

    @pytest.mark.asyncio
    async def test_hallucination_flags(self):
        """Hallucination flags should be populated."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="The model achieves 99.9% accuracy on 5000 samples [1].")
        answerer = LLMAnswerer(
            llm_client=mock_llm,
            hallucination_detector=HallucinationDetector(llm_client=None),
        )

        context = ConstructedContext(
            content="[Source 1]\nThe model was evaluated on a dataset.",
            source_documents=[Document(id="d1", content="The model was evaluated on a dataset.", metadata={"title": "Eval"})],
            compression_ratio=1.0,
            token_count=10,
        )
        result = await answerer.generate("model accuracy", context)
        assert isinstance(result.hallucination_flags, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
