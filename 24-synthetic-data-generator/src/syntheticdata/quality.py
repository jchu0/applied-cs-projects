"""Quality scoring using LLM-as-judge."""

import json
from typing import Optional, Union

from .schemas import RAGExample, InstructionExample, ConversationExample
from .provider import ModelProvider, MockProvider


class QualityScorer:
    """Score generated examples using LLM-as-judge."""

    def __init__(
        self,
        model_provider: Optional[ModelProvider] = None,
        criteria: list[str] = None,
        min_score: float = 0.7,
        check_diversity: bool = False,
    ):
        self.model = model_provider or MockProvider()
        self.min_score = min_score
        self.check_diversity = check_diversity
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

        try:
            response = await self.model.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            scores = self._parse_json(response)
            return scores.get("overall_score", 0.5)
        except Exception:
            return 0.5

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

        try:
            response = await self.model.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            scores = self._parse_json(response)
            return scores.get("overall_score", 0.5)
        except Exception:
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

        try:
            response = await self.model.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            scores = self._parse_json(response)
            return scores.get("overall_score", 0.5)
        except Exception:
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

        try:
            response = await self.model.generate(
                messages=[{"role": "user", "content": comparison_prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            result = self._parse_json(response)
            return (result.get("score_a", 5) / 10, result.get("score_b", 5) / 10)
        except Exception:
            return (0.5, 0.5)

    def _parse_json(self, response: str) -> dict:
        """Parse JSON from response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    async def evaluate(self, dataset) -> dict:
        """Evaluate overall quality of a dataset."""
        total_score = 0.0
        scores = []
        for sample in dataset.samples:
            score = await self.score(sample)
            scores.append(score)
            total_score += score

        avg_score = total_score / len(dataset.samples) if dataset.samples else 0.0
        return {
            "average_score": avg_score,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "num_samples": len(dataset.samples),
            "samples_above_threshold": sum(1 for s in scores if s >= self.min_score),
        }

    def generate_feedback(self, dataset) -> dict:
        """Generate feedback for improving the dataset."""
        return {
            "num_samples": len(dataset.samples),
            "suggestions": [
                "Consider increasing diversity in generated samples",
                "Review low-quality samples for patterns",
            ],
            "metrics": {
                "estimated_coverage": 0.8,
                "diversity_score": 0.7,
            }
        }


class HallucinationDetector:
    """Detect hallucinations in generated content."""

    def __init__(self, model_provider: ModelProvider):
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

        try:
            response = await self.model.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            result = json.loads(response)
            return result.get("hallucination_score", 0.5)
        except Exception:
            return 0.5


class AutoCurationPipeline:
    """Automated curation with quality gates."""

    def __init__(
        self,
        quality_scorer: QualityScorer,
        hallucination_detector: HallucinationDetector = None,
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
            if isinstance(ex, RAGExample) and self.hallucination_detector:
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
