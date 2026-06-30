# Synthetic Data Generator for RAG + Fine-Tuning

> **Concepts covered:** §03 ml-engineering — `01-ml-fundamentals/feature-engineering`

## Executive Summary

Production-grade synthetic data generation pipeline for creating high-quality training data for RAG systems and LLM fine-tuning. Features LLM-powered generation, quality scoring with LLM-as-judge, difficulty-based curriculum sampling, automatic curation, and domain-specific generation support.

## System Architecture

```
+------------------------------------------------------------------+
|                   Synthetic Data Generator                        |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +-------------------+    +-------------+ |
|  | Data Schema      |    | Generation Engine |    | Quality     | |
|  |------------------|    |-------------------|    | Scoring     | |
|  | - RAG QA Pairs   |    | - Prompt Templates|    |-------------| |
|  | - Instructions   |    | - LLM Generator   |    | - LLM Judge | |
|  | - Conversations  |    | - Batch Pipeline  |    | - Metrics   | |
|  | - Domain Config  |    | - Augmentation    |    | - Filtering | |
|  +------------------+    +-------------------+    +-------------+ |
|           |                       |                      |        |
|           v                       v                      v        |
|  +------------------------------------------------------------------+
|  |                    Curriculum Sampler                           |
|  |----------------------------------------------------------------|
|  | Difficulty Scoring | Progressive Sampling | Balanced Batches   |
|  +------------------------------------------------------------------+
|           |                                                        |
|           v                                                        |
|  +------------------------------------------------------------------+
|  |                   Dataset Management                            |
|  |----------------------------------------------------------------|
|  | Versioning (DVC) | Deduplication | Bias Checks | Export        |
|  +------------------------------------------------------------------+
|                                                                   |
+------------------------------------------------------------------+
```

## Core Components

### 1. Data Schema

```python
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
import json

class DataType(Enum):
    RAG_QA = "rag_qa"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    PREFERENCE = "preference"

class DifficultyLevel(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4

@dataclass
class RAGExample:
    """Single RAG training example."""
    id: str
    question: str
    answer: str
    context: str
    context_relevance: float = 1.0  # How relevant context is to question
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    domain: Optional[str] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class InstructionExample:
    """Instruction-following example."""
    id: str
    instruction: str
    input: str
    output: str
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    task_type: str = "general"
    metadata: dict = field(default_factory=dict)

@dataclass
class ConversationExample:
    """Multi-turn conversation example."""
    id: str
    messages: list[dict]  # [{"role": "user/assistant", "content": "..."}]
    system_prompt: Optional[str] = None
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    metadata: dict = field(default_factory=dict)

@dataclass
class PreferenceExample:
    """Preference/ranking example for RLHF."""
    id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    metadata: dict = field(default_factory=dict)

@dataclass
class GenerationConfig:
    """Configuration for data generation."""
    data_type: DataType
    num_samples: int
    domain: Optional[str] = None

    # Generation settings
    model: str = "gpt-4"
    temperature: float = 0.8
    max_tokens: int = 2048

    # Quality settings
    min_quality_score: float = 0.7
    require_human_review: bool = False

    # Difficulty distribution
    difficulty_distribution: dict = field(default_factory=lambda: {
        DifficultyLevel.EASY: 0.2,
        DifficultyLevel.MEDIUM: 0.4,
        DifficultyLevel.HARD: 0.3,
        DifficultyLevel.EXPERT: 0.1,
    })

    # Domain-specific settings
    domain_config: dict = field(default_factory=dict)
```

### 2. Prompt Templates

```python
from string import Template

class PromptTemplateLibrary:
    """Library of prompt templates for different generation tasks."""

    # RAG QA Generation
    RAG_QA_SYSTEM = """You are an expert at creating question-answer pairs for training retrieval-augmented generation systems.

Given a context passage, generate a question that can be answered using the information in the context, along with a comprehensive answer.

Requirements:
- Question should be natural and specific
- Answer should be accurate and based on the context
- Answer should be self-contained (understandable without the context)
- Vary question types: factual, analytical, comparative, etc.
"""

    RAG_QA_USER = Template("""Context:
$context

Difficulty level: $difficulty

Generate a question-answer pair at this difficulty level.
- Easy: Simple factual questions with direct answers
- Medium: Questions requiring understanding and synthesis
- Hard: Questions requiring inference or combining multiple facts
- Expert: Complex analytical questions

Output format (JSON):
{
    "question": "...",
    "answer": "...",
    "reasoning": "Why this question is at the specified difficulty"
}""")

    # Instruction Generation
    INSTRUCTION_SYSTEM = """You are an expert at creating instruction-following examples for training language models.

Generate diverse, realistic instructions with corresponding inputs and outputs. The instructions should be clear and the outputs should be high-quality.

Focus on task type: $task_type
"""

    INSTRUCTION_USER = Template("""Generate an instruction-following example.

Task type: $task_type
Difficulty: $difficulty
Domain: $domain

Requirements:
- Instruction should be clear and actionable
- Input should be realistic
- Output should be comprehensive and correct
- Follow the specified difficulty level

Output format (JSON):
{
    "instruction": "...",
    "input": "...",
    "output": "...",
    "explanation": "Brief explanation of why output is correct"
}""")

    # Conversation Generation
    CONVERSATION_SYSTEM = """You are an expert at creating multi-turn conversations for training conversational AI.

Generate natural, coherent conversations that demonstrate helpful assistant behavior.
"""

    CONVERSATION_USER = Template("""Generate a multi-turn conversation.

Topic: $topic
Number of turns: $num_turns
Difficulty: $difficulty
Domain: $domain

Requirements:
- Conversation should be natural and coherent
- Each turn should build on previous context
- Assistant responses should be helpful and informative
- Include appropriate follow-up questions

Output format (JSON):
{
    "system_prompt": "Optional system prompt for the assistant",
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
}""")

    # Question from Answer (Reverse)
    REVERSE_QA_SYSTEM = """You are an expert at creating questions that would lead to a given answer.

Given an answer and context, generate a natural question that would elicit this answer.
"""

    REVERSE_QA_USER = Template("""Answer: $answer

Context: $context

Generate a question that would naturally lead to this answer.

Output format (JSON):
{
    "question": "...",
    "question_type": "factual/analytical/comparative/etc."
}""")

    # Paraphrase for Augmentation
    PARAPHRASE_SYSTEM = """You are an expert at paraphrasing text while preserving meaning.

Generate paraphrases that maintain semantic equivalence but vary in structure and vocabulary.
"""

    PARAPHRASE_USER = Template("""Original text: $text

Generate $num_paraphrases paraphrases with varying styles:
- Formal
- Casual
- Concise
- Detailed

Output format (JSON):
{
    "paraphrases": [
        {"text": "...", "style": "..."},
        ...
    ]
}""")


class DomainPromptTemplates:
    """Domain-specific prompt templates."""

    LEGAL = {
        "system_context": """You are a legal expert. Generate examples using proper legal terminology and considering legal principles. Be precise with citations and legal concepts.""",
        "requirements": [
            "Use appropriate legal terminology",
            "Reference relevant laws or precedents when applicable",
            "Maintain formal tone",
            "Consider jurisdictional differences",
        ],
    }

    MEDICAL = {
        "system_context": """You are a medical expert. Generate examples using proper medical terminology and following evidence-based medicine principles. Always include appropriate disclaimers.""",
        "requirements": [
            "Use correct medical terminology",
            "Reference clinical guidelines when applicable",
            "Include appropriate safety considerations",
            "Note when professional consultation is advised",
        ],
    }

    TECHNICAL = {
        "system_context": """You are a technical documentation expert. Generate examples that are precise, well-structured, and follow technical writing best practices.""",
        "requirements": [
            "Use precise technical terminology",
            "Include code examples when relevant",
            "Follow consistent formatting",
            "Provide clear explanations",
        ],
    }

    FINANCIAL = {
        "system_context": """You are a financial expert. Generate examples using proper financial terminology and considering regulatory requirements. Include appropriate disclaimers.""",
        "requirements": [
            "Use accurate financial terminology",
            "Consider regulatory context",
            "Include risk disclosures when appropriate",
            "Note that this is not financial advice",
        ],
    }
```

### 3. Generation Engine

```python
import asyncio
from typing import AsyncIterator
import hashlib

class SyntheticDataGenerator:
    """Main generation engine for synthetic data."""

    def __init__(
        self,
        model_provider: "ModelProvider",
        template_library: PromptTemplateLibrary,
        quality_scorer: "QualityScorer",
        config: GenerationConfig,
    ):
        self.model = model_provider
        self.templates = template_library
        self.scorer = quality_scorer
        self.config = config

    async def generate_batch(
        self,
        num_samples: int,
        source_data: list[dict] = None,
    ) -> list:
        """Generate a batch of synthetic examples."""
        examples = []
        attempts = 0
        max_attempts = num_samples * 3  # Allow retries for quality filtering

        # Determine difficulty distribution
        difficulty_counts = self._compute_difficulty_counts(num_samples)

        async for example in self._generate_stream(source_data, difficulty_counts):
            # Score quality
            score = await self.scorer.score(example)

            if score >= self.config.min_quality_score:
                example.metadata["quality_score"] = score
                examples.append(example)

                if len(examples) >= num_samples:
                    break

            attempts += 1
            if attempts >= max_attempts:
                break

        return examples

    async def _generate_stream(
        self,
        source_data: list[dict],
        difficulty_counts: dict,
    ) -> AsyncIterator:
        """Stream generated examples."""
        # Flatten difficulty requirements
        difficulties = []
        for diff, count in difficulty_counts.items():
            difficulties.extend([diff] * count)

        # Shuffle for variety
        import random
        random.shuffle(difficulties)

        for i, difficulty in enumerate(difficulties):
            # Select source data if provided
            source = source_data[i % len(source_data)] if source_data else None

            # Generate based on data type
            if self.config.data_type == DataType.RAG_QA:
                example = await self._generate_rag_qa(source, difficulty)
            elif self.config.data_type == DataType.INSTRUCTION:
                example = await self._generate_instruction(source, difficulty)
            elif self.config.data_type == DataType.CONVERSATION:
                example = await self._generate_conversation(source, difficulty)
            else:
                raise ValueError(f"Unknown data type: {self.config.data_type}")

            if example:
                yield example

    async def _generate_rag_qa(
        self,
        source: dict,
        difficulty: DifficultyLevel,
    ) -> RAGExample:
        """Generate RAG QA pair."""
        context = source.get("context", "") if source else ""

        if not context:
            raise ValueError("Context required for RAG QA generation")

        # Build prompt
        user_prompt = self.templates.RAG_QA_USER.substitute(
            context=context,
            difficulty=difficulty.name.lower(),
        )

        # Generate
        response = await self.model.generate(
            messages=[
                {"role": "system", "content": self.templates.RAG_QA_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Parse response
        try:
            data = json.loads(response)
            return RAGExample(
                id=self._generate_id(data["question"]),
                question=data["question"],
                answer=data["answer"],
                context=context,
                difficulty=difficulty,
                domain=self.config.domain,
                metadata={"reasoning": data.get("reasoning", "")},
            )
        except (json.JSONDecodeError, KeyError) as e:
            return None

    async def _generate_instruction(
        self,
        source: dict,
        difficulty: DifficultyLevel,
    ) -> InstructionExample:
        """Generate instruction-following example."""
        task_type = self.config.domain_config.get("task_type", "general")

        # Get domain-specific context
        domain_context = ""
        if self.config.domain:
            domain_templates = getattr(
                DomainPromptTemplates,
                self.config.domain.upper(),
                None
            )
            if domain_templates:
                domain_context = domain_templates["system_context"]

        system_prompt = self.templates.INSTRUCTION_SYSTEM.replace(
            "$task_type", task_type
        )
        if domain_context:
            system_prompt = f"{domain_context}\n\n{system_prompt}"

        user_prompt = self.templates.INSTRUCTION_USER.substitute(
            task_type=task_type,
            difficulty=difficulty.name.lower(),
            domain=self.config.domain or "general",
        )

        response = await self.model.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        try:
            data = json.loads(response)
            return InstructionExample(
                id=self._generate_id(data["instruction"]),
                instruction=data["instruction"],
                input=data["input"],
                output=data["output"],
                difficulty=difficulty,
                task_type=task_type,
                metadata={"explanation": data.get("explanation", "")},
            )
        except (json.JSONDecodeError, KeyError):
            return None

    async def _generate_conversation(
        self,
        source: dict,
        difficulty: DifficultyLevel,
    ) -> ConversationExample:
        """Generate multi-turn conversation."""
        topic = source.get("topic", "general") if source else "general"
        num_turns = source.get("num_turns", 4) if source else 4

        user_prompt = self.templates.CONVERSATION_USER.substitute(
            topic=topic,
            num_turns=num_turns,
            difficulty=difficulty.name.lower(),
            domain=self.config.domain or "general",
        )

        response = await self.model.generate(
            messages=[
                {"role": "system", "content": self.templates.CONVERSATION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        try:
            data = json.loads(response)
            return ConversationExample(
                id=self._generate_id(str(data["messages"])),
                messages=data["messages"],
                system_prompt=data.get("system_prompt"),
                difficulty=difficulty,
                metadata={"topic": topic},
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _compute_difficulty_counts(self, total: int) -> dict:
        """Compute counts per difficulty level."""
        counts = {}
        remaining = total

        for diff, ratio in self.config.difficulty_distribution.items():
            count = int(total * ratio)
            counts[diff] = count
            remaining -= count

        # Distribute remaining to medium
        counts[DifficultyLevel.MEDIUM] = counts.get(
            DifficultyLevel.MEDIUM, 0
        ) + remaining

        return counts

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for example."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### 4. Quality Scoring (LLM-as-Judge)

```python
from typing import Union

class QualityScorer:
    """Score generated examples using LLM-as-judge."""

    def __init__(
        self,
        model_provider: "ModelProvider",
        criteria: list[str] = None,
    ):
        self.model = model_provider
        self.criteria = criteria or [
            "accuracy",
            "relevance",
            "completeness",
            "clarity",
            "naturalness",
        ]

    async def score(
        self,
        example: Union[RAGExample, InstructionExample, ConversationExample],
    ) -> float:
        """Score example quality (0-1)."""
        if isinstance(example, RAGExample):
            return await self._score_rag_qa(example)
        elif isinstance(example, InstructionExample):
            return await self._score_instruction(example)
        elif isinstance(example, ConversationExample):
            return await self._score_conversation(example)
        else:
            raise ValueError(f"Unknown example type: {type(example)}")

    async def _score_rag_qa(self, example: RAGExample) -> float:
        """Score RAG QA example."""
        prompt = f"""Evaluate this question-answer pair for quality.

Context:
{example.context}

Question: {example.question}
Answer: {example.answer}

Score each criterion from 0 to 10:
1. Accuracy: Is the answer factually correct based on the context?
2. Relevance: Is the question relevant to the context?
3. Completeness: Does the answer fully address the question?
4. Clarity: Is the question clear and unambiguous?
5. Naturalness: Does the question sound natural?

Output format (JSON):
{{
    "accuracy": <score>,
    "relevance": <score>,
    "completeness": <score>,
    "clarity": <score>,
    "naturalness": <score>,
    "overall_score": <0-1>,
    "issues": ["list of issues if any"]
}}"""

        response = await self.model.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent scoring
            max_tokens=500,
        )

        try:
            scores = json.loads(response)
            return scores["overall_score"]
        except (json.JSONDecodeError, KeyError):
            return 0.5  # Default score on parse failure

    async def _score_instruction(self, example: InstructionExample) -> float:
        """Score instruction example."""
        prompt = f"""Evaluate this instruction-following example for quality.

Instruction: {example.instruction}
Input: {example.input}
Output: {example.output}

Score each criterion from 0 to 10:
1. Accuracy: Is the output correct for the instruction and input?
2. Instruction clarity: Is the instruction clear and actionable?
3. Output quality: Is the output well-written and comprehensive?
4. Format: Does the output follow appropriate formatting?
5. Completeness: Does the output fully address the instruction?

Output format (JSON):
{{
    "accuracy": <score>,
    "instruction_clarity": <score>,
    "output_quality": <score>,
    "format": <score>,
    "completeness": <score>,
    "overall_score": <0-1>,
    "issues": ["list of issues if any"]
}}"""

        response = await self.model.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )

        try:
            scores = json.loads(response)
            return scores["overall_score"]
        except (json.JSONDecodeError, KeyError):
            return 0.5

    async def _score_conversation(self, example: ConversationExample) -> float:
        """Score conversation example."""
        messages_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in example.messages
        ])

        prompt = f"""Evaluate this conversation for quality.

{messages_text}

Score each criterion from 0 to 10:
1. Coherence: Does the conversation flow naturally?
2. Helpfulness: Are the assistant responses helpful?
3. Accuracy: Are the responses factually correct?
4. Engagement: Is the conversation engaging?
5. Completeness: Are queries fully addressed?

Output format (JSON):
{{
    "coherence": <score>,
    "helpfulness": <score>,
    "accuracy": <score>,
    "engagement": <score>,
    "completeness": <score>,
    "overall_score": <0-1>,
    "issues": ["list of issues if any"]
}}"""

        response = await self.model.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )

        try:
            scores = json.loads(response)
            return scores["overall_score"]
        except (json.JSONDecodeError, KeyError):
            return 0.5

    async def compare_pair(
        self,
        example_a: str,
        example_b: str,
        prompt: str,
    ) -> tuple[float, float]:
        """Compare two responses and return scores (for preference data)."""
        comparison_prompt = f"""Compare these two responses to the same prompt.

Prompt: {prompt}

Response A:
{example_a}

Response B:
{example_b}

Evaluate which response is better overall considering:
- Accuracy
- Helpfulness
- Clarity
- Completeness

Output format (JSON):
{{
    "score_a": <0-10>,
    "score_b": <0-10>,
    "winner": "A" or "B" or "tie",
    "reasoning": "..."
}}"""

        response = await self.model.generate(
            messages=[{"role": "user", "content": comparison_prompt}],
            temperature=0.1,
            max_tokens=500,
        )

        try:
            result = json.loads(response)
            return (result["score_a"] / 10, result["score_b"] / 10)
        except (json.JSONDecodeError, KeyError):
            return (0.5, 0.5)
```

### 5. Curriculum Sampler

```python
import numpy as np
from collections import defaultdict

class CurriculumSampler:
    """Sample examples with curriculum learning (easy to hard)."""

    def __init__(
        self,
        initial_difficulty: float = 0.2,  # Start easy
        difficulty_increase_rate: float = 0.1,
        warmup_steps: int = 100,
    ):
        self.current_difficulty = initial_difficulty
        self.increase_rate = difficulty_increase_rate
        self.warmup_steps = warmup_steps
        self.step = 0

    def sample_batch(
        self,
        examples: list,
        batch_size: int,
    ) -> list:
        """Sample batch based on current curriculum stage."""
        # Group by difficulty
        by_difficulty = defaultdict(list)
        for ex in examples:
            by_difficulty[ex.difficulty.value].append(ex)

        # Compute sampling weights
        weights = self._compute_weights()

        # Sample from each difficulty level
        batch = []
        for diff_value, weight in weights.items():
            count = int(batch_size * weight)
            available = by_difficulty.get(diff_value, [])

            if available:
                sampled = np.random.choice(
                    available,
                    size=min(count, len(available)),
                    replace=False
                ).tolist()
                batch.extend(sampled)

        # Fill remaining with medium difficulty
        while len(batch) < batch_size:
            medium = by_difficulty.get(DifficultyLevel.MEDIUM.value, [])
            if medium:
                batch.append(np.random.choice(medium))
            else:
                break

        # Update curriculum
        self._step()

        return batch[:batch_size]

    def _compute_weights(self) -> dict:
        """Compute sampling weights based on current difficulty."""
        # Sigmoid curve for difficulty progression
        weights = {}

        for diff in DifficultyLevel:
            # Higher difficulty -> need higher current_difficulty
            threshold = diff.value / 4.0
            weight = 1 / (1 + np.exp(-10 * (self.current_difficulty - threshold)))
            weights[diff.value] = weight

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _step(self):
        """Update curriculum difficulty."""
        self.step += 1

        if self.step > self.warmup_steps:
            self.current_difficulty = min(
                1.0,
                self.current_difficulty + self.increase_rate
            )

    def get_difficulty_for_step(self, step: int) -> float:
        """Get target difficulty for a given step."""
        if step <= self.warmup_steps:
            return self.current_difficulty
        else:
            return min(
                1.0,
                self.current_difficulty + (step - self.warmup_steps) * self.increase_rate
            )


class DifficultyScorer:
    """Score difficulty of examples."""

    def __init__(self, model_provider: "ModelProvider"):
        self.model = model_provider

    async def score_difficulty(
        self,
        example: Union[RAGExample, InstructionExample],
    ) -> DifficultyLevel:
        """Estimate difficulty level of example."""
        if isinstance(example, RAGExample):
            content = f"Question: {example.question}\nAnswer: {example.answer}"
        else:
            content = f"Instruction: {example.instruction}\nOutput: {example.output}"

        prompt = f"""Rate the difficulty level of this example.

{content}

Consider:
- Complexity of language
- Amount of reasoning required
- Domain expertise needed
- Length and detail of response

Output only one of: EASY, MEDIUM, HARD, EXPERT"""

        response = await self.model.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
        )

        difficulty_map = {
            "EASY": DifficultyLevel.EASY,
            "MEDIUM": DifficultyLevel.MEDIUM,
            "HARD": DifficultyLevel.HARD,
            "EXPERT": DifficultyLevel.EXPERT,
        }

        return difficulty_map.get(
            response.strip().upper(),
            DifficultyLevel.MEDIUM
        )
```

### 6. Dataset Management

```python
import pandas as pd
from pathlib import Path
import hashlib

class DatasetManager:
    """Manage generated datasets with versioning and curation."""

    def __init__(
        self,
        output_dir: str,
        use_dvc: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_dvc = use_dvc

    def save_dataset(
        self,
        examples: list,
        name: str,
        format: str = "jsonl",
    ) -> Path:
        """Save dataset to file."""
        filepath = self.output_dir / f"{name}.{format}"

        if format == "jsonl":
            with open(filepath, "w") as f:
                for ex in examples:
                    f.write(json.dumps(self._to_dict(ex)) + "\n")
        elif format == "parquet":
            df = pd.DataFrame([self._to_dict(ex) for ex in examples])
            df.to_parquet(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Version with DVC
        if self.use_dvc:
            import subprocess
            subprocess.run(["dvc", "add", str(filepath)])

        return filepath

    def load_dataset(self, filepath: Path) -> list:
        """Load dataset from file."""
        examples = []

        if filepath.suffix == ".jsonl":
            with open(filepath) as f:
                for line in f:
                    examples.append(json.loads(line))
        elif filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
            examples = df.to_dict("records")

        return examples

    def deduplicate(
        self,
        examples: list,
        similarity_threshold: float = 0.9,
    ) -> list:
        """Remove duplicate or near-duplicate examples."""
        # Simple hash-based dedup
        seen_hashes = set()
        unique = []

        for ex in examples:
            # Hash based on content
            if isinstance(ex, RAGExample):
                content = f"{ex.question}|{ex.answer}"
            elif isinstance(ex, InstructionExample):
                content = f"{ex.instruction}|{ex.output}"
            else:
                content = str(ex)

            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique.append(ex)

        return unique

    def check_bias(
        self,
        examples: list,
        dimensions: list[str] = None,
    ) -> dict:
        """Check for biases in dataset."""
        dimensions = dimensions or ["difficulty", "length", "domain"]
        report = {}

        # Difficulty distribution
        if "difficulty" in dimensions:
            diff_counts = defaultdict(int)
            for ex in examples:
                diff_counts[ex.difficulty.name] += 1
            report["difficulty_distribution"] = dict(diff_counts)

        # Length distribution
        if "length" in dimensions:
            lengths = []
            for ex in examples:
                if isinstance(ex, RAGExample):
                    lengths.append(len(ex.answer))
                elif isinstance(ex, InstructionExample):
                    lengths.append(len(ex.output))

            report["length_stats"] = {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": np.min(lengths),
                "max": np.max(lengths),
            }

        # Domain distribution
        if "domain" in dimensions:
            domain_counts = defaultdict(int)
            for ex in examples:
                domain = ex.domain or "general"
                domain_counts[domain] += 1
            report["domain_distribution"] = dict(domain_counts)

        return report

    def export_for_training(
        self,
        examples: list,
        format: str = "sharegpt",
        output_path: Path = None,
    ) -> Path:
        """Export in format suitable for training."""
        output_path = output_path or self.output_dir / "train.json"

        if format == "sharegpt":
            # ShareGPT format for many fine-tuning tools
            converted = []
            for ex in examples:
                if isinstance(ex, InstructionExample):
                    converted.append({
                        "conversations": [
                            {"from": "human", "value": f"{ex.instruction}\n\n{ex.input}"},
                            {"from": "gpt", "value": ex.output},
                        ]
                    })
                elif isinstance(ex, ConversationExample):
                    convs = []
                    for msg in ex.messages:
                        role = "human" if msg["role"] == "user" else "gpt"
                        convs.append({"from": role, "value": msg["content"]})
                    converted.append({"conversations": convs})

            with open(output_path, "w") as f:
                json.dump(converted, f, indent=2)

        elif format == "alpaca":
            # Alpaca format
            converted = []
            for ex in examples:
                if isinstance(ex, InstructionExample):
                    converted.append({
                        "instruction": ex.instruction,
                        "input": ex.input,
                        "output": ex.output,
                    })

            with open(output_path, "w") as f:
                json.dump(converted, f, indent=2)

        return output_path

    def _to_dict(self, example) -> dict:
        """Convert example to dictionary."""
        if hasattr(example, "__dataclass_fields__"):
            from dataclasses import asdict
            d = asdict(example)
            # Convert enums
            if "difficulty" in d:
                d["difficulty"] = d["difficulty"].name
            return d
        return example
```

## Enterprise Features

### Auto-Curation Pipeline

```python
class AutoCurationPipeline:
    """Automated curation with quality gates."""

    def __init__(
        self,
        quality_scorer: QualityScorer,
        hallucination_detector: "HallucinationDetector",
        min_quality: float = 0.7,
        max_hallucination_score: float = 0.3,
    ):
        self.quality_scorer = quality_scorer
        self.hallucination_detector = hallucination_detector
        self.min_quality = min_quality
        self.max_hallucination = max_hallucination_score

    async def curate(self, examples: list) -> tuple[list, list]:
        """Curate examples, returning (accepted, rejected)."""
        accepted = []
        rejected = []

        for ex in examples:
            # Quality check
            quality_score = await self.quality_scorer.score(ex)

            if quality_score < self.min_quality:
                rejected.append((ex, f"Low quality: {quality_score:.2f}"))
                continue

            # Hallucination check for RAG examples
            if isinstance(ex, RAGExample):
                hallucination_score = await self.hallucination_detector.detect(
                    ex.answer, ex.context
                )

                if hallucination_score > self.max_hallucination:
                    rejected.append((
                        ex,
                        f"Hallucination detected: {hallucination_score:.2f}"
                    ))
                    continue

            accepted.append(ex)

        return accepted, rejected


class HallucinationDetector:
    """Detect hallucinations in generated content."""

    def __init__(self, model_provider: "ModelProvider"):
        self.model = model_provider

    async def detect(self, answer: str, context: str) -> float:
        """Detect hallucination score (0=no hallucination, 1=full hallucination)."""
        prompt = f"""Analyze whether this answer contains hallucinations (information not supported by the context).

Context:
{context}

Answer:
{answer}

For each claim in the answer, check if it's supported by the context.

Output format (JSON):
{{
    "claims": [
        {{"claim": "...", "supported": true/false, "evidence": "..."}}
    ],
    "hallucination_score": <0-1>,
    "hallucinated_parts": ["list of hallucinated claims"]
}}"""

        response = await self.model.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
        )

        try:
            result = json.loads(response)
            return result["hallucination_score"]
        except (json.JSONDecodeError, KeyError):
            return 0.5
```

### Domain-Conditioned Generation

```python
class DomainConditionedGenerator:
    """Generate domain-specific synthetic data."""

    def __init__(
        self,
        base_generator: SyntheticDataGenerator,
        domain_configs: dict,
    ):
        self.base_generator = base_generator
        self.domain_configs = domain_configs

    async def generate_for_domain(
        self,
        domain: str,
        num_samples: int,
        source_data: list[dict] = None,
    ) -> list:
        """Generate domain-specific examples."""
        # Get domain config
        config = self.domain_configs.get(domain, {})

        # Update generator config
        self.base_generator.config.domain = domain
        self.base_generator.config.domain_config = config

        # Use domain-specific source data if not provided
        if source_data is None and "source_data_path" in config:
            source_data = load_domain_data(config["source_data_path"])

        # Generate with domain-specific settings
        examples = await self.base_generator.generate_batch(
            num_samples,
            source_data,
        )

        # Post-process for domain
        if "post_processor" in config:
            examples = [config["post_processor"](ex) for ex in examples]

        return examples


# Domain configurations
DOMAIN_CONFIGS = {
    "legal": {
        "source_data_path": "data/legal/contracts.jsonl",
        "task_type": "legal_analysis",
        "requirements": [
            "Use proper legal terminology",
            "Reference relevant statutes",
            "Include appropriate disclaimers",
        ],
        "post_processor": lambda ex: add_legal_disclaimer(ex),
    },
    "medical": {
        "source_data_path": "data/medical/clinical_notes.jsonl",
        "task_type": "medical_qa",
        "requirements": [
            "Use medical terminology correctly",
            "Note uncertainty appropriately",
            "Include safety warnings",
        ],
        "post_processor": lambda ex: add_medical_disclaimer(ex),
    },
}
```

## API Design

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Synthetic Data Generator")

class GenerationRequest(BaseModel):
    data_type: str  # rag_qa, instruction, conversation
    num_samples: int
    domain: str | None = None
    difficulty_distribution: dict | None = None
    source_data: list[dict] | None = None
    min_quality_score: float = 0.7

class GenerationResponse(BaseModel):
    job_id: str
    status: str
    num_generated: int = 0
    num_accepted: int = 0

@app.post("/generate", response_model=GenerationResponse)
async def generate_data(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
):
    """Start synthetic data generation job."""
    job_id = generate_job_id()

    config = GenerationConfig(
        data_type=DataType[request.data_type.upper()],
        num_samples=request.num_samples,
        domain=request.domain,
        min_quality_score=request.min_quality_score,
    )

    background_tasks.add_task(
        run_generation_job,
        job_id=job_id,
        config=config,
        source_data=request.source_data,
    )

    return GenerationResponse(
        job_id=job_id,
        status="running",
    )

@app.get("/generate/{job_id}", response_model=GenerationResponse)
async def get_generation_status(job_id: str):
    """Get generation job status."""
    status = get_job_status(job_id)
    return status

@app.get("/datasets")
async def list_datasets():
    """List available datasets."""
    return dataset_manager.list_datasets()

@app.get("/datasets/{name}")
async def get_dataset_info(name: str):
    """Get dataset information."""
    return dataset_manager.get_dataset_info(name)
```

## Implementation Phases

### Phase 1: Core Generation (Weeks 1-2)
- [ ] Data schemas
- [ ] Prompt templates
- [ ] Basic generation engine
- [ ] Single-type generation

### Phase 2: Quality Scoring (Weeks 3-4)
- [ ] LLM-as-judge implementation
- [ ] Multi-criteria scoring
- [ ] Preference comparison
- [ ] Score calibration

### Phase 3: Curriculum Learning (Weeks 5-6)
- [ ] Difficulty scoring
- [ ] Curriculum sampler
- [ ] Progressive training
- [ ] Balanced batching

### Phase 4: Curation Pipeline (Weeks 7-8)
- [ ] Auto-curation
- [ ] Hallucination detection
- [ ] Deduplication
- [ ] Bias checking

### Phase 5: Domain Support (Weeks 9-10)
- [ ] Domain configurations
- [ ] Domain-specific templates
- [ ] Post-processing
- [ ] Domain validation

### Phase 6: Production (Weeks 11-12)
- [ ] API service
- [ ] Dataset versioning
- [ ] Export formats
- [ ] Documentation

## Testing Strategy

```python
import pytest

class TestSyntheticDataGenerator:
    @pytest.mark.asyncio
    async def test_rag_qa_generation(self, generator):
        source = [{"context": "Python is a programming language."}]
        examples = await generator.generate_batch(10, source)

        assert len(examples) > 0
        for ex in examples:
            assert ex.question
            assert ex.answer
            assert "python" in ex.answer.lower() or "python" in ex.question.lower()

    @pytest.mark.asyncio
    async def test_quality_filtering(self, generator):
        generator.config.min_quality_score = 0.9
        examples = await generator.generate_batch(10)

        for ex in examples:
            assert ex.metadata.get("quality_score", 0) >= 0.9

    @pytest.mark.asyncio
    async def test_difficulty_distribution(self, generator):
        generator.config.difficulty_distribution = {
            DifficultyLevel.EASY: 0.5,
            DifficultyLevel.HARD: 0.5,
        }
        examples = await generator.generate_batch(100)

        easy_count = sum(1 for ex in examples if ex.difficulty == DifficultyLevel.EASY)
        hard_count = sum(1 for ex in examples if ex.difficulty == DifficultyLevel.HARD)

        # Allow some variance
        assert 30 <= easy_count <= 70
        assert 30 <= hard_count <= 70
```

## Stretch Goals

### Closed-Loop Learner

```python
class ClosedLoopLearner:
    """Iteratively improve generation based on model feedback."""

    async def improve_generation(
        self,
        initial_examples: list,
        num_iterations: int = 3,
    ) -> list:
        """Improve examples through iterative refinement."""
        current_examples = initial_examples

        for iteration in range(num_iterations):
            # Score current examples
            scores = await self.score_batch(current_examples)

            # Identify weak examples
            weak_indices = [
                i for i, score in enumerate(scores)
                if score < self.improvement_threshold
            ]

            # Regenerate weak examples with feedback
            improved = await self.regenerate_with_feedback(
                [current_examples[i] for i in weak_indices],
                [scores[i] for i in weak_indices],
            )

            # Replace weak examples
            for idx, new_ex in zip(weak_indices, improved):
                current_examples[idx] = new_ex

        return current_examples
```

## References

- [Self-Instruct: Aligning LM with Self Generated Instructions](https://arxiv.org/abs/2212.10560)
- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)
- [RLHF: Training language models to follow instructions](https://arxiv.org/abs/2203.02155)
