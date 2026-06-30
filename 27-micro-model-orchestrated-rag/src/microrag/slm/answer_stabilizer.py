"""Answer Stabilizer SLM for consistent answer generation."""

import numpy as np
from typing import List, Optional
import torch
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseSLM, GenerativeModelMixin, EmbeddingModelMixin, logger
from ..schemas import StabilizedAnswer


class AnswerStabilizerSLM(BaseSLM, GenerativeModelMixin, EmbeddingModelMixin):
    """Stabilizes answers through consistency checking and voting using actual models."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_samples: int = 3,
        temperature_range: tuple = (0.3, 0.7),
        consistency_threshold: float = 0.8
    ):
        super().__init__(model_name)
        self.embedding_model_name = embedding_model
        self.num_samples = num_samples
        self.temperature_range = temperature_range
        self.consistency_threshold = consistency_threshold
        self.embedder = None

    def _load_model(self):
        """Load both generative model and embedding model."""
        # Load generative model for answer generation
        self._load_generative_model(self.model_name, load_in_8bit=True)

        # Load embedding model for consistency checking
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.embedder.to(self.device)

    async def process(
        self,
        query: str = None,
        compress_cot: str = None,
        **kwargs
    ) -> StabilizedAnswer:
        """Stabilize answer through multi-sample generation and voting.

        Args:
            query: Original query
            compress_cot: Compressed context/reasoning

        Returns:
            Stabilized answer
        """
        if not query:
            return StabilizedAnswer(
                answer="",
                confidence=0.0,
                consistency=0.0,
                num_samples=0
            )

        if not self._loaded:
            self.load()

        context = compress_cot or ""

        # Generate multiple samples with different temperatures
        samples = await self._generate_diverse_samples(query, context)

        if not samples:
            return StabilizedAnswer(
                answer="Unable to generate answer",
                confidence=0.0,
                consistency=0.0,
                num_samples=0,
                reasoning="Failed to generate answer samples"
            )

        # Compute semantic consistency
        consistency_score = self._compute_semantic_consistency(samples)

        # Select final answer based on consistency
        if consistency_score >= self.consistency_threshold:
            # High consistency - use majority vote
            final_answer = await self._semantic_majority_vote(samples)
        else:
            # Low consistency - refine the answer
            final_answer = await self._refine_divergent_answers(query, context, samples)

        # Compute confidence based on multiple factors
        confidence = self._compute_confidence(
            final_answer,
            samples,
            consistency_score
        )

        # Generate reasoning for the answer selection
        reasoning = self._generate_selection_reasoning(
            final_answer,
            samples,
            consistency_score
        )

        return StabilizedAnswer(
            answer=final_answer,
            confidence=confidence,
            consistency=consistency_score,
            num_samples=len(samples),
            reasoning=reasoning
        )

    async def _generate_diverse_samples(
        self,
        query: str,
        context: str
    ) -> List[str]:
        """Generate multiple answer samples with varying temperatures.

        Args:
            query: Query string
            context: Context for answering

        Returns:
            List of answer samples
        """
        samples = []

        # Create base prompt
        if context:
            prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Answer the following question concisely:

Question: {query}

Answer:"""

        # Generate samples with different temperatures for diversity
        for i in range(self.num_samples):
            # Vary temperature for each sample
            temp_range = self.temperature_range[1] - self.temperature_range[0]
            temperature = self.temperature_range[0] + (temp_range * i / max(self.num_samples - 1, 1))

            try:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    outputs = self.model.generate(
                        inputs['input_ids'],
                        max_new_tokens=150,
                        temperature=temperature,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )

                    answer = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()

                if answer:
                    samples.append(answer)

            except Exception as e:
                logger.warning(f"Failed to generate sample {i+1}: {str(e)}")

        return samples

    def _compute_semantic_consistency(self, samples: List[str]) -> float:
        """Compute semantic consistency across answer samples.

        Args:
            samples: List of answer samples

        Returns:
            Consistency score (0-1)
        """
        if len(samples) <= 1:
            return 1.0

        # Get embeddings for all samples
        embeddings = self.embedder.encode(
            samples,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)

        # Return mean similarity as consistency score
        return float(np.mean(similarities)) if similarities else 1.0

    async def _semantic_majority_vote(self, samples: List[str]) -> str:
        """Select answer with highest semantic similarity to others.

        Args:
            samples: List of answer samples

        Returns:
            Selected answer
        """
        if not samples:
            return ""
        if len(samples) == 1:
            return samples[0]

        # Get embeddings
        embeddings = self.embedder.encode(
            samples,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Score each sample by average similarity to others
        scores = []
        for i, emb_i in enumerate(embeddings):
            similarities = []
            for j, emb_j in enumerate(embeddings):
                if i != j:
                    sim = cosine_similarity([emb_i], [emb_j])[0][0]
                    similarities.append(sim)
            avg_similarity = np.mean(similarities) if similarities else 0
            scores.append(avg_similarity)

        # Return sample with highest average similarity
        best_idx = int(np.argmax(scores))
        return samples[best_idx]

    async def _refine_divergent_answers(
        self,
        query: str,
        context: str,
        samples: List[str]
    ) -> str:
        """Refine answer when samples show low consistency.

        Args:
            query: Query string
            context: Context
            samples: Divergent samples

        Returns:
            Refined answer
        """
        # Create a synthesis prompt
        samples_text = "\n".join([f"- {s}" for s in samples[:3]])  # Use top 3 samples

        prompt = f"""Multiple answers were generated for this question, but they differ.
Please synthesize them into a single, coherent answer.

Question: {query}

Different answers:
{samples_text}

Synthesized answer:"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=700
                ).to(self.device)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=150,
                    temperature=0.3,  # Lower temperature for synthesis
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                refined = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            return refined if refined else samples[0]

        except Exception as e:
            logger.warning(f"Failed to refine answer: {str(e)}")
            # Fallback to longest answer
            return max(samples, key=len) if samples else ""

    def _compute_confidence(
        self,
        final_answer: str,
        samples: List[str],
        consistency: float
    ) -> float:
        """Compute confidence score based on multiple factors.

        Args:
            final_answer: Selected answer
            samples: All samples
            consistency: Consistency score

        Returns:
            Confidence score (0-1)
        """
        # Start with consistency as base confidence
        confidence = consistency * 0.6

        # Factor in sample agreement with final answer
        if samples and self.embedder:
            final_embedding = self.embedder.encode(
                [final_answer],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]

            sample_embeddings = self.embedder.encode(
                samples,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Calculate how similar final answer is to all samples
            similarities = cosine_similarity([final_embedding], sample_embeddings)[0]
            avg_similarity = np.mean(similarities)
            confidence += avg_similarity * 0.3

        # Factor in answer length (very short answers might be less confident)
        answer_length = len(final_answer.split())
        if answer_length > 5:
            confidence += 0.1
        elif answer_length < 3:
            confidence -= 0.1

        # Ensure confidence is in [0, 1]
        return max(0.0, min(1.0, confidence))

    def _generate_selection_reasoning(
        self,
        final_answer: str,
        samples: List[str],
        consistency: float
    ) -> str:
        """Generate reasoning for answer selection.

        Args:
            final_answer: Selected answer
            samples: All samples
            consistency: Consistency score

        Returns:
            Reasoning explanation
        """
        if len(samples) <= 1:
            return "Single answer generated with high confidence."

        if consistency >= self.consistency_threshold:
            return f"Selected from {len(samples)} consistent samples (consistency: {consistency:.2f}) using semantic voting."
        else:
            return f"Synthesized from {len(samples)} divergent samples (consistency: {consistency:.2f}) to create coherent answer."


class MockAnswerStabilizerSLM(BaseSLM):
    """Mock answer stabilizer for testing."""

    def __init__(self):
        super().__init__("mock")
        self._loaded = True

    async def process(
        self,
        query: str = None,
        compress_cot: str = None,
        **kwargs
    ) -> StabilizedAnswer:
        answer = f"Mock answer for: {query}" if query else "No query provided"

        return StabilizedAnswer(
            answer=answer,
            confidence=0.85,
            consistency=0.9,
            num_samples=3,
            reasoning="Mock stabilization"
        )


class ConsensusAnswerGenerator(AnswerStabilizerSLM):
    """Generates answers through consensus among multiple reasoning paths."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        num_paths: int = 5,
        consensus_threshold: float = 0.7
    ):
        super().__init__(model_name, num_samples=num_paths)
        self.consensus_threshold = consensus_threshold

    async def generate_consensus_answer(
        self,
        query: str,
        context: str,
        reasoning_paths: Optional[List[str]] = None
    ) -> StabilizedAnswer:
        """Generate answer through consensus of multiple reasoning paths.

        Args:
            query: User query
            context: Context information
            reasoning_paths: Optional pre-generated reasoning paths

        Returns:
            Consensus answer
        """
        if not self._loaded:
            self.load()

        # Generate reasoning paths if not provided
        if not reasoning_paths:
            reasoning_paths = await self._generate_reasoning_paths(query, context)

        # Generate answers from each reasoning path
        path_answers = []
        for path in reasoning_paths:
            answer = await self._answer_from_reasoning(query, path)
            if answer:
                path_answers.append(answer)

        if not path_answers:
            return StabilizedAnswer(
                answer="Unable to generate consensus answer",
                confidence=0.0,
                consistency=0.0,
                num_samples=0,
                reasoning="No valid reasoning paths generated"
            )

        # Find consensus
        consistency = self._compute_semantic_consistency(path_answers)

        if consistency >= self.consensus_threshold:
            final_answer = await self._semantic_majority_vote(path_answers)
            reasoning = f"Strong consensus ({consistency:.2f}) among {len(path_answers)} reasoning paths"
        else:
            # Low consensus - explain disagreement
            final_answer = await self._explain_disagreement(query, path_answers)
            reasoning = f"Low consensus ({consistency:.2f}) - provided balanced perspective"

        return StabilizedAnswer(
            answer=final_answer,
            confidence=consistency,
            consistency=consistency,
            num_samples=len(path_answers),
            reasoning=reasoning
        )

    async def _generate_reasoning_paths(
        self,
        query: str,
        context: str
    ) -> List[str]:
        """Generate different reasoning paths for the query."""
        paths = []
        approaches = [
            "Step-by-step logical reasoning",
            "Consider pros and cons",
            "Analyze from different perspectives",
            "Use analogical reasoning",
            "Apply first principles"
        ]

        for i, approach in enumerate(approaches[:self.num_samples]):
            prompt = f"""Use {approach} to answer this question.

Context: {context}

Question: {query}

Reasoning:"""

            try:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    outputs = self.model.generate(
                        inputs['input_ids'],
                        max_new_tokens=200,
                        temperature=0.5,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                    reasoning = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()

                if reasoning:
                    paths.append(reasoning)

            except Exception as e:
                logger.warning(f"Failed to generate reasoning path {i+1}: {str(e)}")

        return paths

    async def _answer_from_reasoning(self, query: str, reasoning: str) -> str:
        """Extract answer from reasoning path."""
        prompt = f"""Based on this reasoning, provide a concise answer to the question.

Reasoning: {reasoning}

Question: {query}

Concise answer:"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=600
                ).to(self.device)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=100,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                answer = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            return answer

        except Exception as e:
            logger.warning(f"Failed to extract answer: {str(e)}")
            return ""

    async def _explain_disagreement(
        self,
        query: str,
        answers: List[str]
    ) -> str:
        """Explain when answers disagree."""
        answers_text = "\n".join([f"- {a}" for a in answers[:3]])

        prompt = f"""These different answers were generated for the same question.
Provide a balanced response that acknowledges the different perspectives.

Question: {query}

Different answers:
{answers_text}

Balanced response:"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=600
                ).to(self.device)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=150,
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            return response

        except Exception as e:
            logger.warning(f"Failed to explain disagreement: {str(e)}")
            return answers[0] if answers else ""