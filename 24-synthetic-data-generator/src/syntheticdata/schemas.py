"""Data schemas for synthetic data generation."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DataType(Enum):
    """Types of synthetic data that can be generated."""
    RAG_QA = "rag_qa"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    PREFERENCE = "preference"


class DifficultyLevel(Enum):
    """Difficulty levels for generated examples."""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class RAGExample:
    """Single RAG training example with question, answer, and context."""
    id: str
    question: str
    answer: str
    context: str
    context_relevance: float = 1.0
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    domain: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "context_relevance": self.context_relevance,
            "difficulty": self.difficulty.name,
            "domain": self.domain,
            "metadata": self.metadata,
        }


@dataclass
class InstructionExample:
    """Instruction-following example for fine-tuning."""
    id: str
    instruction: str
    output: str
    input: str = ""
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    task_type: str = "general"
    domain: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "difficulty": self.difficulty.name,
            "task_type": self.task_type,
            "domain": self.domain,
            "metadata": self.metadata,
        }


@dataclass
class ConversationExample:
    """Multi-turn conversation example."""
    id: str
    messages: list  # [{"role": "user/assistant", "content": "..."}]
    system_prompt: Optional[str] = None
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "messages": self.messages,
            "system_prompt": self.system_prompt,
            "difficulty": self.difficulty.name,
            "metadata": self.metadata,
        }


@dataclass
class PreferenceExample:
    """Preference/ranking example for RLHF training."""
    id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            "metadata": self.metadata,
        }


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    data_type: DataType = DataType.INSTRUCTION
    num_samples: int = 10
    domain: Optional[str] = None

    # Generation settings
    model: str = "gpt-4"
    temperature: float = 0.8
    max_tokens: int = 2048

    # Quality settings
    min_quality_score: float = 0.7
    quality_threshold: float = 0.7  # Alias for min_quality_score
    require_human_review: bool = False

    # Batch/parallel settings
    dataset_size: int = 100
    batch_size: int = 10
    max_parallel: int = 5

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Difficulty distribution
    difficulty_distribution: dict = field(default_factory=lambda: {
        DifficultyLevel.EASY: 0.2,
        DifficultyLevel.MEDIUM: 0.4,
        DifficultyLevel.HARD: 0.3,
        DifficultyLevel.EXPERT: 0.1,
    })

    # Domain-specific settings
    domain_config: dict = field(default_factory=dict)

    def __post_init__(self):
        """Sync quality threshold aliases."""
        if self.quality_threshold != 0.7:
            self.min_quality_score = self.quality_threshold

    def get_difficulty_counts(self, total: int) -> dict:
        """Compute counts per difficulty level based on distribution."""
        counts = {}
        remaining = total

        for diff, ratio in self.difficulty_distribution.items():
            count = int(total * ratio)
            counts[diff] = count
            remaining -= count

        # Distribute remaining to medium
        counts[DifficultyLevel.MEDIUM] = counts.get(
            DifficultyLevel.MEDIUM, 0
        ) + remaining

        return counts
