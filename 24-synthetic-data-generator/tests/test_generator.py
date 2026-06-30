"""Tests for synthetic data generator."""

import pytest
import json
import asyncio

from syntheticdata.schemas import (
    DataType,
    DifficultyLevel,
    RAGExample,
    InstructionExample,
    GenerationConfig,
)
from syntheticdata.generator import SyntheticDataGenerator, BatchGenerator
from syntheticdata.provider import MockProvider
from syntheticdata.templates import PromptTemplateLibrary


@pytest.fixture
def mock_provider():
    """Create mock provider with sample responses."""
    responses = [
        json.dumps({
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "reasoning": "Simple factual question",
        }),
        json.dumps({
            "question": "How does photosynthesis work?",
            "answer": "Photosynthesis converts light energy into chemical energy.",
            "reasoning": "Requires understanding of process",
        }),
    ]
    return MockProvider(responses)


@pytest.fixture
def generator(mock_provider):
    """Create generator with mock provider."""
    config = GenerationConfig(
        data_type=DataType.RAG_QA,
        num_samples=10,
        min_quality_score=0.0,  # Disable quality filtering for tests
    )
    return SyntheticDataGenerator(
        model_provider=mock_provider,
        config=config,
    )


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator."""

    @pytest.mark.asyncio
    async def test_generate_rag_qa_batch(self, generator):
        """Test generating batch of RAG QA examples."""
        source_data = [
            {"context": "France is a country in Western Europe. Its capital is Paris."},
            {"context": "Photosynthesis is the process by which plants convert light."},
        ]

        examples = await generator.generate_batch(
            num_samples=2,
            source_data=source_data,
        )

        assert len(examples) > 0
        for ex in examples:
            assert isinstance(ex, RAGExample)
            assert ex.question
            assert ex.answer
            assert ex.context

    @pytest.mark.asyncio
    async def test_difficulty_distribution(self, generator):
        """Test that difficulty distribution is respected."""
        generator.config.difficulty_distribution = {
            DifficultyLevel.EASY: 0.5,
            DifficultyLevel.MEDIUM: 0.3,
            DifficultyLevel.HARD: 0.2,
            DifficultyLevel.EXPERT: 0.0,
        }

        counts = generator.config.get_difficulty_counts(100)

        assert counts[DifficultyLevel.EASY] == 50
        assert counts[DifficultyLevel.MEDIUM] == 30
        assert counts[DifficultyLevel.HARD] == 20
        assert counts[DifficultyLevel.EXPERT] == 0

    @pytest.mark.asyncio
    async def test_generate_with_domain(self, generator, mock_provider):
        """Test generation with domain configuration."""
        generator.config.domain = "medical"

        source_data = [{"context": "Diabetes is a metabolic disorder."}]

        examples = await generator.generate_batch(
            num_samples=1,
            source_data=source_data,
        )

        # Verify domain was set
        if examples:
            assert examples[0].domain == "medical"

    def test_generate_id(self, generator):
        """Test ID generation is deterministic."""
        id1 = generator._generate_id("test content")
        id2 = generator._generate_id("test content")
        id3 = generator._generate_id("different content")

        assert id1 == id2
        assert id1 != id3
        assert len(id1) == 16


class TestBatchGenerator:
    """Tests for BatchGenerator."""

    @pytest.mark.asyncio
    async def test_generate_large_dataset(self, generator):
        """Test generating large dataset in batches."""
        batch_gen = BatchGenerator(
            generator=generator,
            batch_size=5,
            max_concurrent=2,
        )

        source_data = [{"context": "Test context for batch generation."}]

        # This will generate with mock provider
        examples = await batch_gen.generate_large_dataset(
            total_samples=10,
            source_data=source_data,
            config=generator.config,
        )

        # Should get up to 10 examples
        assert len(examples) <= 10


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_default_difficulty_distribution(self):
        """Test default difficulty distribution."""
        config = GenerationConfig(
            data_type=DataType.RAG_QA,
            num_samples=100,
        )

        counts = config.get_difficulty_counts(100)
        total = sum(counts.values())

        assert total == 100
        assert counts[DifficultyLevel.EASY] == 20
        assert counts[DifficultyLevel.MEDIUM] == 40
        assert counts[DifficultyLevel.HARD] == 30
        assert counts[DifficultyLevel.EXPERT] == 10

    def test_custom_difficulty_distribution(self):
        """Test custom difficulty distribution."""
        config = GenerationConfig(
            data_type=DataType.INSTRUCTION,
            num_samples=50,
            difficulty_distribution={
                DifficultyLevel.EASY: 1.0,
                DifficultyLevel.MEDIUM: 0.0,
                DifficultyLevel.HARD: 0.0,
                DifficultyLevel.EXPERT: 0.0,
            },
        )

        counts = config.get_difficulty_counts(50)

        assert counts[DifficultyLevel.EASY] == 50


class TestMockProvider:
    """Tests for MockProvider."""

    @pytest.mark.asyncio
    async def test_mock_provider_cycles_responses(self):
        """Test that mock provider cycles through responses."""
        responses = ["response1", "response2"]
        provider = MockProvider(responses)

        r1 = await provider.generate([{"role": "user", "content": "test"}])
        r2 = await provider.generate([{"role": "user", "content": "test"}])
        r3 = await provider.generate([{"role": "user", "content": "test"}])

        assert r1 == "response1"
        assert r2 == "response2"
        assert r3 == "response1"  # Cycles back

    @pytest.mark.asyncio
    async def test_mock_provider_tracks_calls(self):
        """Test that mock provider tracks calls."""
        provider = MockProvider()

        await provider.generate([{"role": "user", "content": "test1"}])
        await provider.generate([{"role": "user", "content": "test2"}])

        assert provider.call_count == 2
        assert len(provider.calls) == 2
        assert provider.calls[0]["messages"][0]["content"] == "test1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
