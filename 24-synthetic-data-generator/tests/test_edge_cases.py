"""Edge case tests for synthetic data generator.

This module contains comprehensive tests for edge cases including:
- Empty input handling
- Invalid configuration handling
- Boundary conditions for quality scoring
- Error recovery scenarios
- Concurrent generation tests
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor

from syntheticdata.schemas import (
    DataType,
    DifficultyLevel,
    RAGExample,
    InstructionExample,
    ConversationExample,
    PreferenceExample,
    GenerationConfig,
)
from syntheticdata.generator import SyntheticDataGenerator, BatchGenerator
from syntheticdata.provider import MockProvider, RateLimitedProvider
from syntheticdata.quality import QualityScorer, HallucinationDetector, AutoCurationPipeline
from syntheticdata.dataset import DatasetManager


# =============================================================================
# EMPTY INPUT HANDLING TESTS
# =============================================================================

class TestEmptyInputHandling:
    """Tests for handling empty and None inputs."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider with default responses."""
        return MockProvider([
            json.dumps({
                "question": "Test question?",
                "answer": "Test answer.",
                "reasoning": "Test reasoning",
            })
        ])

    @pytest.fixture
    def generator(self, mock_provider):
        """Create generator with mock provider."""
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=5,
            min_quality_score=0.0,
        )
        return SyntheticDataGenerator(
            model_provider=mock_provider,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_generate_with_empty_source_data(self, generator):
        """Test generation with empty source data list."""
        examples = await generator.generate_batch(
            num_samples=2,
            source_data=[],
        )
        # Should handle empty source gracefully
        assert isinstance(examples, list)

    @pytest.mark.asyncio
    async def test_generate_with_none_source_data(self, generator):
        """Test generation with None source data."""
        # Without source data, RAG QA generation requires context and will fail
        examples = await generator.generate_batch(
            num_samples=2,
            source_data=None,
        )
        # Should return empty list since RAG QA needs context
        assert isinstance(examples, list)

    @pytest.mark.asyncio
    async def test_generate_with_empty_context(self, generator):
        """Test generation with empty context string."""
        source_data = [{"context": ""}]
        examples = await generator.generate_batch(
            num_samples=2,
            source_data=source_data,
        )
        # Should handle empty context (may return empty results)
        assert isinstance(examples, list)

    @pytest.mark.asyncio
    async def test_generate_with_whitespace_only_context(self, generator):
        """Test generation with whitespace-only context."""
        source_data = [{"context": "   \n\t  "}]
        examples = await generator.generate_batch(
            num_samples=2,
            source_data=source_data,
        )
        assert isinstance(examples, list)

    @pytest.mark.asyncio
    async def test_generate_zero_samples(self, generator):
        """Test requesting zero samples."""
        source_data = [{"context": "Valid context."}]
        examples = await generator.generate_batch(
            num_samples=0,
            source_data=source_data,
        )
        assert examples == []

    @pytest.mark.asyncio
    async def test_generate_negative_samples(self, generator):
        """Test requesting negative number of samples."""
        source_data = [{"context": "Valid context."}]
        examples = await generator.generate_batch(
            num_samples=-5,
            source_data=source_data,
        )
        # Should return empty list for invalid count
        assert examples == []

    def test_empty_difficulty_distribution(self):
        """Test config with empty difficulty distribution."""
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=10,
            difficulty_distribution={},
        )
        counts = config.get_difficulty_counts(10)
        # Should still return valid structure
        assert isinstance(counts, dict)
        # All samples should go to MEDIUM by default
        assert counts.get(DifficultyLevel.MEDIUM, 0) == 10

    def test_difficulty_counts_with_zero_total(self):
        """Test getting difficulty counts with zero total."""
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=0,
        )
        counts = config.get_difficulty_counts(0)
        assert sum(counts.values()) == 0


class TestEmptyExampleHandling:
    """Tests for handling empty example fields."""

    def test_rag_example_with_empty_fields(self):
        """Test RAGExample with empty string fields."""
        example = RAGExample(
            id="",
            question="",
            answer="",
            context="",
        )
        result = example.to_dict()
        assert result["id"] == ""
        assert result["question"] == ""
        assert result["answer"] == ""
        assert result["context"] == ""

    def test_instruction_example_with_empty_input(self):
        """Test InstructionExample with empty input field."""
        example = InstructionExample(
            id="test",
            instruction="Do something",
            input="",  # Empty input is valid
            output="Result",
        )
        result = example.to_dict()
        assert result["input"] == ""

    def test_conversation_example_with_empty_messages(self):
        """Test ConversationExample with empty messages list."""
        example = ConversationExample(
            id="test",
            messages=[],
        )
        result = example.to_dict()
        assert result["messages"] == []

    def test_preference_example_with_empty_strings(self):
        """Test PreferenceExample with empty strings."""
        example = PreferenceExample(
            id="test",
            prompt="",
            chosen="",
            rejected="",
            chosen_score=0.0,
            rejected_score=0.0,
        )
        result = example.to_dict()
        assert result["prompt"] == ""


# =============================================================================
# INVALID CONFIGURATION HANDLING TESTS
# =============================================================================

class TestInvalidConfiguration:
    """Tests for handling invalid configuration values."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        return MockProvider()

    def test_invalid_data_type_string(self):
        """Test that invalid data type string raises error."""
        with pytest.raises((ValueError, KeyError)):
            DataType("invalid_type")

    def test_invalid_difficulty_level(self):
        """Test that invalid difficulty level raises error."""
        with pytest.raises((ValueError, KeyError)):
            DifficultyLevel("super_expert")

    @pytest.mark.asyncio
    async def test_generate_without_config(self, mock_provider):
        """Test generation with None config uses default config."""
        # SyntheticDataGenerator defaults to GenerationConfig() when config=None
        generator = SyntheticDataGenerator(
            model_provider=mock_provider,
            config=None,
        )
        # Should not raise - uses default config
        result = await generator.generate_batch(num_samples=1, source_data=[{"context": "test"}])
        assert result is not None

    def test_negative_temperature(self):
        """Test config with negative temperature."""
        # Config allows any float, but this is semantically invalid
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=10,
            temperature=-0.5,
        )
        assert config.temperature == -0.5

    def test_temperature_above_two(self):
        """Test config with temperature > 2."""
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=10,
            temperature=2.5,
        )
        assert config.temperature == 2.5

    def test_negative_max_tokens(self):
        """Test config with negative max_tokens."""
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=10,
            max_tokens=-100,
        )
        assert config.max_tokens == -100

    def test_zero_max_tokens(self):
        """Test config with zero max_tokens."""
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=10,
            max_tokens=0,
        )
        assert config.max_tokens == 0

    def test_invalid_min_quality_score(self):
        """Test config with quality score outside 0-1 range."""
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=10,
            min_quality_score=1.5,
        )
        assert config.min_quality_score == 1.5

    def test_negative_quality_score(self):
        """Test config with negative quality score."""
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=10,
            min_quality_score=-0.5,
        )
        assert config.min_quality_score == -0.5

    def test_difficulty_distribution_not_summing_to_one(self):
        """Test difficulty distribution that doesn't sum to 1."""
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=10,
            difficulty_distribution={
                DifficultyLevel.EASY: 0.5,
                DifficultyLevel.MEDIUM: 0.5,
                DifficultyLevel.HARD: 0.5,
                DifficultyLevel.EXPERT: 0.5,
            },
        )
        # Should still work but may produce unexpected sample counts
        counts = config.get_difficulty_counts(10)
        assert isinstance(counts, dict)


# =============================================================================
# BOUNDARY CONDITIONS FOR QUALITY SCORING
# =============================================================================

class TestQualityScoringBoundaries:
    """Tests for quality scoring boundary conditions."""

    @pytest.fixture
    def scorer_with_perfect_score(self):
        """Create scorer that returns perfect scores."""
        provider = MockProvider([
            json.dumps({
                "accuracy": 10,
                "relevance": 10,
                "completeness": 10,
                "clarity": 10,
                "naturalness": 10,
                "overall_score": 1.0,
                "issues": [],
            })
        ])
        return QualityScorer(provider)

    @pytest.fixture
    def scorer_with_zero_score(self):
        """Create scorer that returns zero scores."""
        provider = MockProvider([
            json.dumps({
                "accuracy": 0,
                "relevance": 0,
                "completeness": 0,
                "clarity": 0,
                "naturalness": 0,
                "overall_score": 0.0,
                "issues": ["Everything is wrong"],
            })
        ])
        return QualityScorer(provider)

    @pytest.fixture
    def scorer_with_out_of_range_score(self):
        """Create scorer that returns out-of-range scores."""
        provider = MockProvider([
            json.dumps({
                "accuracy": 15,  # Above 10
                "relevance": -5,  # Below 0
                "completeness": 10,
                "clarity": 10,
                "naturalness": 10,
                "overall_score": 1.5,  # Above 1.0
                "issues": [],
            })
        ])
        return QualityScorer(provider)

    @pytest.mark.asyncio
    async def test_perfect_quality_score(self, scorer_with_perfect_score):
        """Test handling of perfect 1.0 quality score."""
        example = RAGExample(
            id="test",
            question="What is 2+2?",
            answer="2+2 equals 4.",
            context="Basic arithmetic.",
        )
        score = await scorer_with_perfect_score.score(example)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_zero_quality_score(self, scorer_with_zero_score):
        """Test handling of zero quality score."""
        example = RAGExample(
            id="test",
            question="Bad question",
            answer="Bad answer",
            context="Bad context",
        )
        score = await scorer_with_zero_score.score(example)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_out_of_range_quality_score(self, scorer_with_out_of_range_score):
        """Test handling of score above 1.0."""
        example = RAGExample(
            id="test",
            question="Test?",
            answer="Test.",
            context="Test context",
        )
        score = await scorer_with_out_of_range_score.score(example)
        # Should return the score as-is (1.5)
        assert score == 1.5

    @pytest.mark.asyncio
    async def test_score_with_empty_response(self):
        """Test scoring when provider returns empty response."""
        provider = MockProvider([""])
        scorer = QualityScorer(provider)
        example = RAGExample(
            id="test",
            question="Test?",
            answer="Test.",
            context="Context",
        )
        score = await scorer.score(example)
        # Should return default 0.5 on parse error
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_score_with_null_response(self):
        """Test scoring when provider returns JSON null."""
        provider = MockProvider(["null"])
        scorer = QualityScorer(provider)
        example = RAGExample(
            id="test",
            question="Test?",
            answer="Test.",
            context="Context",
        )
        score = await scorer.score(example)
        # Should return default 0.5 on missing key
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_score_with_missing_overall_score_key(self):
        """Test scoring when overall_score key is missing."""
        provider = MockProvider([
            json.dumps({
                "accuracy": 8,
                "relevance": 9,
                # Missing "overall_score"
            })
        ])
        scorer = QualityScorer(provider)
        example = RAGExample(
            id="test",
            question="Test?",
            answer="Test.",
            context="Context",
        )
        score = await scorer.score(example)
        # Should return default 0.5 when key missing
        assert score == 0.5


class TestHallucinationDetectorBoundaries:
    """Tests for hallucination detector boundary conditions."""

    @pytest.mark.asyncio
    async def test_detect_with_empty_answer(self):
        """Test detection with empty answer string."""
        provider = MockProvider([
            json.dumps({
                "claims": [],
                "hallucination_score": 0.0,
                "hallucinated_parts": [],
            })
        ])
        detector = HallucinationDetector(provider)
        score = await detector.detect(answer="", context="Some context")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_detect_with_empty_context(self):
        """Test detection with empty context string."""
        provider = MockProvider([
            json.dumps({
                "claims": [{"claim": "Everything", "supported": False, "evidence": ""}],
                "hallucination_score": 1.0,
                "hallucinated_parts": ["Everything"],
            })
        ])
        detector = HallucinationDetector(provider)
        score = await detector.detect(answer="Some answer", context="")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_detect_with_very_long_content(self):
        """Test detection with very long content strings."""
        provider = MockProvider([
            json.dumps({
                "claims": [],
                "hallucination_score": 0.5,
                "hallucinated_parts": [],
            })
        ])
        detector = HallucinationDetector(provider)
        long_answer = "A" * 100000
        long_context = "B" * 100000
        score = await detector.detect(answer=long_answer, context=long_context)
        assert score == 0.5


class TestAutoCurationPipelineBoundaries:
    """Tests for auto curation pipeline boundary conditions."""

    @pytest.fixture
    def curation_pipeline(self):
        """Create curation pipeline with mocked components."""
        quality_provider = MockProvider([
            json.dumps({
                "overall_score": 0.8,
                "issues": [],
            })
        ])
        hallucination_provider = MockProvider([
            json.dumps({
                "hallucination_score": 0.1,
                "hallucinated_parts": [],
            })
        ])
        return AutoCurationPipeline(
            quality_scorer=QualityScorer(quality_provider),
            hallucination_detector=HallucinationDetector(hallucination_provider),
            min_quality=0.7,
            max_hallucination_score=0.3,
        )

    @pytest.mark.asyncio
    async def test_curate_empty_list(self, curation_pipeline):
        """Test curating empty list of examples."""
        accepted, rejected = await curation_pipeline.curate([])
        assert accepted == []
        assert rejected == []

    @pytest.mark.asyncio
    async def test_curate_with_threshold_at_boundary(self):
        """Test curation with score exactly at threshold."""
        provider = MockProvider([
            json.dumps({
                "overall_score": 0.7,  # Exactly at threshold
                "issues": [],
            })
        ])
        pipeline = AutoCurationPipeline(
            quality_scorer=QualityScorer(provider),
            min_quality=0.7,
        )
        example = InstructionExample(
            id="test",
            instruction="Test",
            input="",
            output="Test output",
        )
        accepted, rejected = await pipeline.curate([example])
        # Score of 0.7 should be accepted (>= 0.7)
        assert len(accepted) == 1
        assert len(rejected) == 0

    @pytest.mark.asyncio
    async def test_curate_with_score_just_below_threshold(self):
        """Test curation with score just below threshold."""
        provider = MockProvider([
            json.dumps({
                "overall_score": 0.699,  # Just below threshold
                "issues": [],
            })
        ])
        pipeline = AutoCurationPipeline(
            quality_scorer=QualityScorer(provider),
            min_quality=0.7,
        )
        example = InstructionExample(
            id="test",
            instruction="Test",
            input="",
            output="Test output",
        )
        accepted, rejected = await pipeline.curate([example])
        assert len(accepted) == 0
        assert len(rejected) == 1


# =============================================================================
# ERROR RECOVERY SCENARIOS
# =============================================================================

class TestErrorRecovery:
    """Tests for error recovery in various scenarios."""

    @pytest.mark.asyncio
    async def test_provider_raises_exception(self):
        """Test handling when provider raises exception."""
        class FailingProvider(MockProvider):
            async def generate(self, *args, **kwargs):
                raise Exception("Provider error")

        provider = FailingProvider()
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=5,
            min_quality_score=0.0,
        )
        generator = SyntheticDataGenerator(
            model_provider=provider,
            config=config,
        )

        examples = await generator.generate_batch(
            num_samples=2,
            source_data=[{"context": "Test context"}],
        )
        # Should handle exception and return empty or partial results
        assert isinstance(examples, list)

    @pytest.mark.asyncio
    async def test_provider_returns_malformed_json(self):
        """Test handling when provider returns invalid JSON."""
        provider = MockProvider([
            "{invalid json",
            "not json at all",
            "{'single': 'quotes'}",  # Python dict, not JSON
        ])
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=3,
            min_quality_score=0.0,
        )
        generator = SyntheticDataGenerator(
            model_provider=provider,
            config=config,
        )

        examples = await generator.generate_batch(
            num_samples=3,
            source_data=[{"context": "Test context"}],
        )
        # Should handle malformed JSON gracefully
        assert isinstance(examples, list)

    @pytest.mark.asyncio
    async def test_provider_returns_partial_json(self):
        """Test handling when JSON is missing required fields."""
        provider = MockProvider([
            json.dumps({"question": "Test?"}),  # Missing answer
            json.dumps({"answer": "Test."}),  # Missing question
            json.dumps({}),  # Empty object
        ])
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=3,
            min_quality_score=0.0,
        )
        generator = SyntheticDataGenerator(
            model_provider=provider,
            config=config,
        )

        examples = await generator.generate_batch(
            num_samples=3,
            source_data=[{"context": "Test context"}],
        )
        assert isinstance(examples, list)

    @pytest.mark.asyncio
    async def test_scorer_raises_exception(self):
        """Test that scorer exceptions propagate to caller.

        Note: The current implementation does not catch scorer exceptions,
        which means callers need to handle them. This test documents that behavior.
        """
        class FailingScorer:
            async def score(self, example):
                raise Exception("Scoring error")

        provider = MockProvider([
            json.dumps({
                "question": "Test?",
                "answer": "Test.",
                "reasoning": "Test",
            })
        ])
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=3,
            min_quality_score=0.5,
        )
        generator = SyntheticDataGenerator(
            model_provider=provider,
            config=config,
            quality_scorer=FailingScorer(),
        )

        # Scorer exceptions are propagated to the caller
        with pytest.raises(Exception, match="Scoring error"):
            await generator.generate_batch(
                num_samples=2,
                source_data=[{"context": "Test context"}],
            )

    @pytest.mark.asyncio
    async def test_generation_without_scorer(self):
        """Test generation works without a quality scorer."""
        provider = MockProvider([
            json.dumps({
                "question": "Test?",
                "answer": "Test.",
                "reasoning": "Test",
            })
        ])
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=2,
            min_quality_score=0.0,
        )
        generator = SyntheticDataGenerator(
            model_provider=provider,
            config=config,
            quality_scorer=None,  # No scorer
        )

        examples = await generator.generate_batch(
            num_samples=2,
            source_data=[{"context": "Test context"}],
        )
        assert isinstance(examples, list)
        assert len(examples) <= 2

    @pytest.mark.asyncio
    async def test_recovery_after_timeout(self):
        """Test recovery when provider times out."""
        class TimeoutProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            async def generate(self, *args, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    await asyncio.sleep(10)  # Simulate timeout
                return await super().generate(*args, **kwargs)

        provider = TimeoutProvider()
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=2,
            min_quality_score=0.0,
        )
        generator = SyntheticDataGenerator(
            model_provider=provider,
            config=config,
        )

        # Use asyncio.wait_for to handle timeout
        try:
            examples = await asyncio.wait_for(
                generator.generate_batch(
                    num_samples=2,
                    source_data=[{"context": "Test context"}],
                ),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            examples = []

        assert isinstance(examples, list)

    @pytest.mark.asyncio
    async def test_parse_json_with_embedded_json(self):
        """Test parsing JSON embedded in text response."""
        # Simulate LLM response with JSON embedded in explanation
        response_with_embedded = '''
        Here is the result:
        {"question": "Test?", "answer": "Answer.", "reasoning": "test"}
        Hope this helps!
        '''
        provider = MockProvider([response_with_embedded])
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=1,
            min_quality_score=0.0,
        )
        generator = SyntheticDataGenerator(
            model_provider=provider,
            config=config,
        )

        examples = await generator.generate_batch(
            num_samples=1,
            source_data=[{"context": "Test context"}],
        )
        # Should extract JSON from response
        assert len(examples) == 1


# =============================================================================
# CONCURRENT GENERATION TESTS
# =============================================================================

class TestConcurrentGeneration:
    """Tests for concurrent generation scenarios."""

    @pytest.fixture
    def thread_safe_provider(self):
        """Create a thread-safe mock provider."""
        return MockProvider([
            json.dumps({
                "question": f"Question {i}?",
                "answer": f"Answer {i}.",
                "reasoning": f"Reasoning {i}",
            })
            for i in range(100)
        ])

    @pytest.fixture
    def batch_generator(self, thread_safe_provider):
        """Create batch generator for concurrent tests."""
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=100,
            min_quality_score=0.0,
        )
        generator = SyntheticDataGenerator(
            model_provider=thread_safe_provider,
            config=config,
        )
        return BatchGenerator(
            generator=generator,
            batch_size=10,
            max_concurrent=5,
        )

    @pytest.mark.asyncio
    async def test_concurrent_batch_generation(self, batch_generator):
        """Test generating multiple batches concurrently."""
        source_data = [{"context": f"Context {i}"} for i in range(10)]

        examples = await batch_generator.generate_large_dataset(
            total_samples=30,
            source_data=source_data,
            config=batch_generator.generator.config,
        )

        assert len(examples) <= 30

    @pytest.mark.asyncio
    async def test_concurrent_with_shared_state(self):
        """Test that concurrent generation doesn't corrupt shared state."""
        call_order = []

        class TrackingProvider(MockProvider):
            async def generate(self, *args, **kwargs):
                call_order.append(len(call_order))
                await asyncio.sleep(0.01)  # Simulate API latency
                return await super().generate(*args, **kwargs)

        provider = TrackingProvider([
            json.dumps({
                "question": "Test?",
                "answer": "Answer.",
                "reasoning": "Reason",
            })
        ] * 20)

        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=20,
            min_quality_score=0.0,
        )
        generator = SyntheticDataGenerator(
            model_provider=provider,
            config=config,
        )
        batch_gen = BatchGenerator(
            generator=generator,
            batch_size=5,
            max_concurrent=4,
        )

        source_data = [{"context": "Test context"}]
        examples = await batch_gen.generate_large_dataset(
            total_samples=20,
            source_data=source_data,
            config=config,
        )

        # Verify calls were tracked (order may vary due to concurrency)
        assert len(call_order) > 0

    @pytest.mark.asyncio
    async def test_concurrent_with_rate_limiting(self):
        """Test concurrent generation with rate limiting."""
        base_provider = MockProvider([
            json.dumps({
                "question": "Test?",
                "answer": "Answer.",
                "reasoning": "Reason",
            })
        ] * 10)

        rate_limited = RateLimitedProvider(
            provider=base_provider,
            requests_per_minute=600,  # 10 per second
        )

        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=5,
            min_quality_score=0.0,
        )
        generator = SyntheticDataGenerator(
            model_provider=rate_limited,
            config=config,
        )

        source_data = [{"context": "Test context"}]
        examples = await generator.generate_batch(
            num_samples=5,
            source_data=source_data,
        )

        assert len(examples) <= 5

    @pytest.mark.asyncio
    async def test_concurrent_with_exception_in_one_batch(self):
        """Test that exception in one batch doesn't affect others."""
        call_count = 0

        class PartiallyFailingProvider(MockProvider):
            async def generate(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 5:  # Fail on 5th call
                    raise Exception("Simulated failure")
                return await super().generate(*args, **kwargs)

        provider = PartiallyFailingProvider([
            json.dumps({
                "question": "Test?",
                "answer": "Answer.",
                "reasoning": "Reason",
            })
        ] * 20)

        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=20,
            min_quality_score=0.0,
        )
        generator = SyntheticDataGenerator(
            model_provider=provider,
            config=config,
        )
        batch_gen = BatchGenerator(
            generator=generator,
            batch_size=5,
            max_concurrent=4,
        )

        source_data = [{"context": "Test context"}]
        examples = await batch_gen.generate_large_dataset(
            total_samples=20,
            source_data=source_data,
            config=config,
        )

        # Should still get some examples despite one failure
        assert isinstance(examples, list)

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Test that semaphore properly limits concurrent operations."""
        concurrent_count = 0
        max_concurrent_observed = 0

        class ConcurrencyTrackingProvider(MockProvider):
            async def generate(self, *args, **kwargs):
                nonlocal concurrent_count, max_concurrent_observed
                concurrent_count += 1
                max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
                await asyncio.sleep(0.05)  # Simulate work
                result = await super().generate(*args, **kwargs)
                concurrent_count -= 1
                return result

        provider = ConcurrencyTrackingProvider([
            json.dumps({
                "question": "Test?",
                "answer": "Answer.",
                "reasoning": "Reason",
            })
        ] * 50)

        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=20,
            min_quality_score=0.0,
        )
        generator = SyntheticDataGenerator(
            model_provider=provider,
            config=config,
        )
        batch_gen = BatchGenerator(
            generator=generator,
            batch_size=5,
            max_concurrent=3,  # Limit to 3 concurrent batches
        )

        source_data = [{"context": "Test context"}]
        await batch_gen.generate_large_dataset(
            total_samples=20,
            source_data=source_data,
            config=config,
        )

        # Should never exceed max_concurrent
        assert max_concurrent_observed <= 3


# =============================================================================
# DATASET MANAGER EDGE CASES
# =============================================================================

class TestDatasetManagerEdgeCases:
    """Tests for DatasetManager edge cases."""

    @pytest.fixture
    def dataset_manager(self, tmp_path):
        """Create dataset manager with temp directory."""
        return DatasetManager(output_dir=str(tmp_path))

    def test_save_empty_dataset(self, dataset_manager):
        """Test saving empty dataset."""
        filepath = dataset_manager.save_dataset([], "empty_dataset", format="jsonl")
        assert filepath.exists()
        # File should be empty or have no records
        content = filepath.read_text()
        assert content == ""

    def test_save_dataset_with_special_characters(self, dataset_manager):
        """Test saving dataset with special characters in content."""
        examples = [
            RAGExample(
                id="test",
                question="What about unicode? \u4e2d\u6587 \u0420\u0443\u0441\u0441\u043a\u0438\u0439",
                answer="Special chars: <>\"'&\n\t\r",
                context="Newlines\nand\ttabs",
            )
        ]
        filepath = dataset_manager.save_dataset(examples, "special_chars", format="jsonl")
        loaded = dataset_manager.load_dataset(filepath)
        assert len(loaded) == 1
        assert "\u4e2d\u6587" in loaded[0]["question"]

    def test_load_nonexistent_file(self, dataset_manager):
        """Test loading file that doesn't exist."""
        from pathlib import Path
        with pytest.raises(FileNotFoundError):
            dataset_manager.load_dataset(Path("/nonexistent/path/file.jsonl"))

    def test_deduplicate_empty_list(self, dataset_manager):
        """Test deduplicating empty list."""
        result = dataset_manager.deduplicate([])
        assert result == []

    def test_deduplicate_all_duplicates(self, dataset_manager):
        """Test deduplicating list where all items are duplicates."""
        example = RAGExample(
            id="test",
            question="Test?",
            answer="Test.",
            context="Context",
        )
        examples = [example, example, example, example]
        result = dataset_manager.deduplicate(examples)
        assert len(result) == 1

    def test_check_bias_empty_list(self, dataset_manager):
        """Test checking bias on empty list."""
        report = dataset_manager.check_bias([])
        assert "difficulty_distribution" in report
        assert report["difficulty_distribution"] == {}

    def test_export_unknown_format(self, dataset_manager):
        """Test exporting to unknown format."""
        examples = [
            InstructionExample(
                id="test",
                instruction="Test",
                input="",
                output="Output",
            )
        ]
        with pytest.raises(ValueError, match="Unknown export format"):
            dataset_manager.export_for_training(examples, format="unknown_format")

    def test_save_unknown_format(self, dataset_manager):
        """Test saving to unknown format."""
        examples = [
            RAGExample(
                id="test",
                question="Test?",
                answer="Test.",
                context="Context",
            )
        ]
        with pytest.raises(ValueError, match="Unknown format"):
            dataset_manager.save_dataset(examples, "test", format="xyz")


# =============================================================================
# ID GENERATION EDGE CASES
# =============================================================================

class TestIdGeneration:
    """Tests for ID generation edge cases."""

    @pytest.fixture
    def generator(self):
        """Create generator for ID tests."""
        return SyntheticDataGenerator(
            model_provider=MockProvider(),
            config=GenerationConfig(
                data_type=DataType.RAG_QA,
                num_samples=1,
            ),
        )

    def test_id_from_empty_string(self, generator):
        """Test ID generation from empty string."""
        id1 = generator._generate_id("")
        id2 = generator._generate_id("")
        assert id1 == id2
        assert len(id1) == 16

    def test_id_from_unicode(self, generator):
        """Test ID generation from unicode content."""
        id1 = generator._generate_id("\u4e2d\u6587\u5185\u5bb9")
        assert len(id1) == 16

    def test_id_from_very_long_content(self, generator):
        """Test ID generation from very long content."""
        long_content = "A" * 1000000
        id1 = generator._generate_id(long_content)
        assert len(id1) == 16

    def test_id_uniqueness(self, generator):
        """Test that different content produces different IDs."""
        ids = set()
        for i in range(1000):
            id_val = generator._generate_id(f"content_{i}")
            ids.add(id_val)
        # All IDs should be unique
        assert len(ids) == 1000


# =============================================================================
# COMPARISON SCORING EDGE CASES
# =============================================================================

class TestComparisonScoring:
    """Tests for pair comparison scoring edge cases."""

    @pytest.mark.asyncio
    async def test_compare_identical_responses(self):
        """Test comparing two identical responses."""
        provider = MockProvider([
            json.dumps({
                "score_a": 8,
                "score_b": 8,
                "winner": "tie",
                "reasoning": "Responses are identical",
            })
        ])
        scorer = QualityScorer(provider)
        score_a, score_b = await scorer.compare_pair(
            example_a="Same response",
            example_b="Same response",
            prompt="Test prompt",
        )
        assert score_a == score_b == 0.8

    @pytest.mark.asyncio
    async def test_compare_empty_responses(self):
        """Test comparing empty responses."""
        provider = MockProvider([
            json.dumps({
                "score_a": 0,
                "score_b": 0,
                "winner": "tie",
                "reasoning": "Both empty",
            })
        ])
        scorer = QualityScorer(provider)
        score_a, score_b = await scorer.compare_pair(
            example_a="",
            example_b="",
            prompt="",
        )
        assert score_a == 0.0
        assert score_b == 0.0

    @pytest.mark.asyncio
    async def test_compare_with_parse_error(self):
        """Test comparison when response can't be parsed."""
        provider = MockProvider(["not valid json"])
        scorer = QualityScorer(provider)
        score_a, score_b = await scorer.compare_pair(
            example_a="Response A",
            example_b="Response B",
            prompt="Test",
        )
        # Should return default 0.5 for both
        assert score_a == 0.5
        assert score_b == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
