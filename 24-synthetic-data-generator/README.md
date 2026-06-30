# Synthetic Data Generator

A production-grade synthetic data generation pipeline for creating high-quality training data for RAG systems and LLM fine-tuning.

## Features

- **Multiple Data Types**: RAG Q&A pairs, instruction-following, conversations, preference data
- **Quality Scoring**: LLM-as-judge for automated quality assessment
- **Curriculum Learning**: Difficulty-based progressive sampling
- **Domain Support**: Legal, medical, technical, financial domains
- **Dataset Management**: Versioning, deduplication, export formats

## Installation

```bash
cd projects/24-synthetic-data-generator
pip install -e .
```

## Quick Start

```python
import asyncio
from syntheticdata import (
    SyntheticDataGenerator,
    GenerationConfig,
    DataType,
    DifficultyLevel,
)
from syntheticdata.provider import OpenAIProvider

async def main():
    # Initialize provider
    provider = OpenAIProvider(api_key="your-api-key", model="gpt-4")

    # Configure generation
    config = GenerationConfig(
        data_type=DataType.RAG_QA,
        num_samples=100,
        domain="technical",
        min_quality_score=0.7,
        difficulty_distribution={
            DifficultyLevel.EASY: 0.2,
            DifficultyLevel.MEDIUM: 0.4,
            DifficultyLevel.HARD: 0.3,
            DifficultyLevel.EXPERT: 0.1,
        },
    )

    # Create generator
    generator = SyntheticDataGenerator(
        model_provider=provider,
        config=config,
    )

    # Generate data
    source_data = [
        {"context": "Your source document text here..."},
    ]

    examples = await generator.generate_batch(
        num_samples=100,
        source_data=source_data,
    )

    print(f"Generated {len(examples)} examples")

asyncio.run(main())
```

## Data Types

### RAG Q&A
Question-answer pairs with context for retrieval-augmented generation training.

```python
from syntheticdata import RAGExample

example = RAGExample(
    id="qa-001",
    question="What is machine learning?",
    answer="Machine learning is a subset of AI...",
    context="Source document about ML...",
    difficulty=DifficultyLevel.MEDIUM,
)
```

### Instruction Following
Instruction-input-output triplets for fine-tuning.

```python
from syntheticdata import InstructionExample

example = InstructionExample(
    id="inst-001",
    instruction="Summarize the following text",
    input="Long article text...",
    output="Concise summary...",
    task_type="summarization",
)
```

### Conversations
Multi-turn dialogues for conversational AI.

```python
from syntheticdata import ConversationExample

example = ConversationExample(
    id="conv-001",
    messages=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ],
    system_prompt="You are a helpful assistant.",
)
```

## Quality Scoring

Use LLM-as-judge for automated quality assessment:

```python
from syntheticdata import QualityScorer

scorer = QualityScorer(provider)
score = await scorer.score(example)  # Returns 0-1

# Compare two responses
score_a, score_b = await scorer.compare_pair(
    example_a="Response A",
    example_b="Response B",
    prompt="Original prompt",
)
```

## Curriculum Learning

Progressive difficulty sampling for better training:

```python
from syntheticdata import CurriculumSampler

sampler = CurriculumSampler(
    initial_difficulty=0.2,
    difficulty_increase_rate=0.1,
    warmup_steps=100,
)

batch = sampler.sample_batch(examples, batch_size=32)
```

## Dataset Management

Save, load, and export datasets:

```python
from syntheticdata import DatasetManager

manager = DatasetManager("./output")

# Save dataset
manager.save_dataset(examples, "train", format="jsonl")

# Export for training
manager.export_for_training(examples, format="sharegpt")
manager.export_for_training(examples, format="alpaca")

# Check for biases
report = manager.check_bias(examples)
print(report["difficulty_distribution"])
```

## Domain Support

Generate domain-specific data:

```python
config = GenerationConfig(
    data_type=DataType.INSTRUCTION,
    num_samples=100,
    domain="medical",  # or "legal", "technical", "financial"
)
```

## Project Structure

```
24-synthetic-data-generator/
├── src/syntheticdata/
│   ├── __init__.py
│   ├── schemas.py      # Data models
│   ├── templates.py    # Prompt templates
│   ├── generator.py    # Generation engine
│   ├── quality.py      # Quality scoring
│   ├── curriculum.py   # Curriculum sampling
│   ├── dataset.py      # Dataset management
│   └── provider.py     # LLM providers
├── tests/
│   ├── test_generator.py
│   └── test_quality.py
├── examples/
├── configs/
└── README.md
```

## Implementation Status

### Phase 1: Core Generation ✅
- [x] Data schemas
- [x] Prompt templates
- [x] Basic generation engine
- [x] Single-type generation

### Phase 2: Quality Scoring ✅
- [x] LLM-as-judge implementation
- [x] Multi-criteria scoring
- [x] Preference comparison
- [x] Hallucination detection

### Phase 3: Curriculum Learning ✅
- [x] Difficulty scoring
- [x] Curriculum sampler
- [x] Balanced batching

### Phase 4: Dataset Management ✅
- [x] Save/load datasets
- [x] Deduplication
- [x] Bias checking
- [x] Export formats

### Phase 5-6: Remaining
- [ ] Domain validation
- [ ] API service
- [ ] Dataset versioning (DVC)
- [ ] Advanced curation

## Testing

```bash
pytest tests/ -v
```

## References

- [Self-Instruct Paper](https://arxiv.org/abs/2212.10560)
- [WizardLM Paper](https://arxiv.org/abs/2304.12244)
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)
