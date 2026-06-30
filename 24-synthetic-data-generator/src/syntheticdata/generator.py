"""Main generation engine for synthetic data."""

import asyncio
import hashlib
import json
import random
from typing import AsyncIterator, Optional

from .schemas import (
    DataType,
    DifficultyLevel,
    RAGExample,
    InstructionExample,
    ConversationExample,
    GenerationConfig,
)
from .templates import PromptTemplateLibrary, DomainPromptTemplates
from .provider import ModelProvider, MockProvider as MockModelProvider
from .dataset import Dataset


class SyntheticDataGenerator:
    """Main generation engine for synthetic data."""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        model_provider: Optional[ModelProvider] = None,
        template_library: Optional[PromptTemplateLibrary] = None,
        quality_scorer: Optional["QualityScorer"] = None,
    ):
        # Support both new style (config first) and old style (model_provider first)
        if isinstance(config, ModelProvider):
            # Old-style call: SyntheticDataGenerator(model_provider, ...)
            model_provider = config
            config = None

        self.model = model_provider or MockModelProvider()
        self.templates = template_library or PromptTemplateLibrary()
        self.scorer = quality_scorer
        self.config = config or GenerationConfig()

    async def generate_dataset(
        self,
        domain: str = "general",
        num_samples: int = 10,
        diversity_score: float = 0.5,
        **kwargs,
    ) -> Dataset:
        """Generate a complete dataset for a domain."""
        # Update config with parameters
        config = GenerationConfig(
            data_type=DataType.INSTRUCTION,
            num_samples=num_samples,
            domain=domain,
            **{k: v for k, v in kwargs.items() if hasattr(GenerationConfig, k)}
        )

        # Generate samples
        samples = await self.generate_batch(num_samples, config=config)

        # Create dataset
        dataset = Dataset(
            samples=samples,
            metadata={
                "domain": domain,
                "diversity_score": diversity_score,
                "num_samples": len(samples),
            }
        )

        return dataset

    async def generate_batch(
        self,
        num_samples: int,
        source_data: Optional[list[dict]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> list:
        """Generate a batch of synthetic examples."""
        config = config or self.config
        if not config:
            raise ValueError("GenerationConfig required")

        examples = []
        attempts = 0
        max_attempts = num_samples * 3  # Allow retries for quality filtering

        # Determine difficulty distribution
        difficulty_counts = config.get_difficulty_counts(num_samples)

        async for example in self._generate_stream(source_data, difficulty_counts, config):
            if example is None:
                attempts += 1
                if attempts >= max_attempts:
                    break
                continue

            # Score quality if scorer available
            if self.scorer:
                score = await self.scorer.score(example)
                if score < config.min_quality_score:
                    attempts += 1
                    if attempts >= max_attempts:
                        break
                    continue
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
        source_data: Optional[list[dict]],
        difficulty_counts: dict,
        config: GenerationConfig,
    ) -> AsyncIterator:
        """Stream generated examples."""
        # Flatten difficulty requirements
        difficulties = []
        for diff, count in difficulty_counts.items():
            difficulties.extend([diff] * count)

        # Shuffle for variety
        random.shuffle(difficulties)

        for i, difficulty in enumerate(difficulties):
            # Select source data if provided
            source = None
            if source_data:
                source = source_data[i % len(source_data)]

            # Generate based on data type
            try:
                if config.data_type == DataType.RAG_QA:
                    example = await self._generate_rag_qa(source, difficulty, config)
                elif config.data_type == DataType.INSTRUCTION:
                    example = await self._generate_instruction(source, difficulty, config)
                elif config.data_type == DataType.CONVERSATION:
                    example = await self._generate_conversation(source, difficulty, config)
                else:
                    raise ValueError(f"Unknown data type: {config.data_type}")
            except Exception as e:
                example = None

            yield example

    async def _generate_rag_qa(
        self,
        source: Optional[dict],
        difficulty: DifficultyLevel,
        config: GenerationConfig,
    ) -> Optional[RAGExample]:
        """Generate RAG QA pair."""
        context = source.get("context", "") if source else ""

        if not context:
            raise ValueError("Context required for RAG QA generation")

        # Build prompt
        user_prompt = self.templates.RAG_QA_USER.substitute(
            context=context,
            difficulty=difficulty.name.lower(),
        )

        # Add domain context if specified
        system_prompt = self.templates.RAG_QA_SYSTEM
        if config.domain:
            domain_config = DomainPromptTemplates.get_domain_config(config.domain)
            system_prompt = f"{domain_config['system_context']}\n\n{system_prompt}"

        # Generate
        try:
            response = await self.model.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            # Parse response
            data = self._parse_json_response(response)
            if not data:
                return None

            return RAGExample(
                id=self._generate_id(data.get("question", "")),
                question=data["question"],
                answer=data["answer"],
                context=context,
                difficulty=difficulty,
                domain=config.domain,
                metadata={"reasoning": data.get("reasoning", "")},
            )
        except Exception:
            return None

    async def _generate_instruction(
        self,
        source: Optional[dict],
        difficulty: DifficultyLevel,
        config: GenerationConfig,
    ) -> Optional[InstructionExample]:
        """Generate instruction-following example."""
        task_type = config.domain_config.get("task_type", "general")

        # Get domain-specific context
        system_prompt = self.templates.INSTRUCTION_SYSTEM.substitute(
            task_type=task_type
        )

        if config.domain:
            domain_config = DomainPromptTemplates.get_domain_config(config.domain)
            system_prompt = f"{domain_config['system_context']}\n\n{system_prompt}"

        user_prompt = self.templates.INSTRUCTION_USER.substitute(
            task_type=task_type,
            difficulty=difficulty.name.lower(),
            domain=config.domain or "general",
        )

        try:
            response = await self.model.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            data = self._parse_json_response(response)
            if not data:
                return None

            return InstructionExample(
                id=self._generate_id(data.get("instruction", "")),
                instruction=data["instruction"],
                input=data.get("input", ""),
                output=data["output"],
                difficulty=difficulty,
                task_type=task_type,
                metadata={"explanation": data.get("explanation", "")},
            )
        except Exception:
            return None

    async def _generate_conversation(
        self,
        source: Optional[dict],
        difficulty: DifficultyLevel,
        config: GenerationConfig,
    ) -> Optional[ConversationExample]:
        """Generate multi-turn conversation."""
        topic = source.get("topic", "general") if source else "general"
        num_turns = source.get("num_turns", 4) if source else 4

        user_prompt = self.templates.CONVERSATION_USER.substitute(
            topic=topic,
            num_turns=num_turns,
            difficulty=difficulty.name.lower(),
            domain=config.domain or "general",
        )

        try:
            response = await self.model.generate(
                messages=[
                    {"role": "system", "content": self.templates.CONVERSATION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            data = self._parse_json_response(response)
            if not data:
                return None

            return ConversationExample(
                id=self._generate_id(str(data.get("messages", []))),
                messages=data["messages"],
                system_prompt=data.get("system_prompt"),
                difficulty=difficulty,
                metadata={"topic": topic},
            )
        except Exception:
            return None

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Parse JSON from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        return None

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for example."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def set_domain_registry(self, registry) -> None:
        """Set the domain registry for validation."""
        self.domain_registry = registry

    def set_curriculum(self, curriculum) -> None:
        """Set the curriculum for guided generation."""
        self.curriculum = curriculum

    async def augment_dataset(
        self,
        dataset: "Dataset",
        augmentation_factor: int = 2,
        **kwargs,
    ) -> "Dataset":
        """Augment an existing dataset by generating variations."""
        augmented_samples = []
        for sample in dataset.samples:
            augmented_samples.append(sample)
            # Generate variations
            for _ in range(augmentation_factor - 1):
                # Re-generate with similar parameters
                config = GenerationConfig(
                    data_type=DataType.INSTRUCTION,
                    num_samples=1,
                    **kwargs,
                )
                new_samples = await self.generate_batch(1, config=config)
                augmented_samples.extend(new_samples)

        return Dataset(
            samples=augmented_samples,
            metadata={
                **dataset.metadata,
                "augmented": True,
                "augmentation_factor": augmentation_factor,
            }
        )

    async def generate_stream(
        self,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> AsyncIterator:
        """Stream generated samples one at a time."""
        config = config or GenerationConfig(**kwargs) if kwargs else self.config
        difficulty_counts = config.get_difficulty_counts(config.num_samples)

        async for example in self._generate_stream(None, difficulty_counts, config):
            if example is not None:
                yield example

    async def generate_conditional(
        self,
        conditions: dict,
        num_samples: int = 10,
        **kwargs,
    ) -> "Dataset":
        """Generate samples matching specific conditions."""
        config = GenerationConfig(
            num_samples=num_samples,
            **{k: v for k, v in conditions.items() if hasattr(GenerationConfig, k)},
            **{k: v for k, v in kwargs.items() if hasattr(GenerationConfig, k)},
        )
        samples = await self.generate_batch(num_samples, config=config)
        return Dataset(
            samples=samples,
            metadata={"conditions": conditions}
        )


class BatchGenerator:
    """High-throughput batch generation with parallelism."""

    def __init__(
        self,
        generator: SyntheticDataGenerator,
        batch_size: int = 10,
        max_concurrent: int = 5,
    ):
        self.generator = generator
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent

    async def generate_large_dataset(
        self,
        total_samples: int,
        source_data: Optional[list[dict]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> list:
        """Generate large dataset in parallel batches."""
        all_examples = []
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def generate_batch(batch_idx: int) -> list:
            async with semaphore:
                samples_needed = min(
                    self.batch_size,
                    total_samples - batch_idx * self.batch_size
                )
                return await self.generator.generate_batch(
                    samples_needed,
                    source_data,
                    config,
                )

        # Generate all batches
        tasks = [generate_batch(i) for i in range(num_batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in results:
            if isinstance(result, list):
                all_examples.extend(result)

        return all_examples[:total_samples]
