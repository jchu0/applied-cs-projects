"""Chain-of-Thought Compressor SLM for reasoning distillation."""

from typing import List, Optional
import torch
import re

from .base import BaseSLM, GenerativeModelMixin, logger


class CoTCompressorSLM(BaseSLM, GenerativeModelMixin):
    """Compresses chain-of-thought reasoning while preserving logic using SLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        compression_ratio: float = 0.3,
        preserve_key_steps: bool = True,
        temperature: float = 0.2
    ):
        super().__init__(model_name)
        self.compression_ratio = compression_ratio
        self.preserve_key_steps = preserve_key_steps
        self.temperature = temperature

    def _load_model(self):
        """Load the generative model for reasoning compression."""
        self._load_generative_model(self.model_name, load_in_8bit=True)

    async def process(
        self,
        query: str = None,
        summarize: str = None,
        **kwargs
    ) -> str:
        """Compress reasoning chain using language model.

        Args:
            query: Original query
            summarize: Summary/reasoning to compress

        Returns:
            Compressed reasoning
        """
        if not summarize:
            return ""

        if not self._loaded:
            self.load()

        # Calculate target length
        words = summarize.split()
        target_words = max(10, int(len(words) * self.compression_ratio))

        # Use LLM to intelligently compress the reasoning
        compressed = await self._llm_compress(summarize, query, target_words)

        return compressed

    async def _llm_compress(
        self,
        reasoning: str,
        query: str,
        target_words: int
    ) -> str:
        """Compress reasoning using language model.

        Args:
            reasoning: Full reasoning text
            query: Original query
            target_words: Target number of words

        Returns:
            Compressed text
        """
        # Create compression prompt
        prompt = f"""Compress the following reasoning while preserving key logical steps.
Keep the most important information that answers the query.

Query: {query}

Original reasoning:
{reasoning}

Compressed version (approximately {target_words} words, preserve logical flow):"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1500
                ).to(self.device)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=target_words * 2,  # Allow some flexibility
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                compressed = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            # Clean and validate
            compressed = self._clean_compressed_text(compressed)

            # If too long, truncate intelligently
            if len(compressed.split()) > target_words * 1.5:
                compressed = self._smart_truncate(compressed, target_words)

            return compressed

        except Exception as e:
            logger.error(f"Failed to compress with LLM: {str(e)}")
            # Fallback to heuristic compression
            return self._heuristic_compress(reasoning, target_words)

    def _heuristic_compress(self, text: str, target_words: int) -> str:
        """Fallback heuristic compression."""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Score sentences by importance
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence_importance(sentence, i, len(sentences))
            scored_sentences.append((sentence, score))

        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        selected = []
        word_count = 0
        for sentence, _ in scored_sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= target_words:
                selected.append(sentence)
                word_count += sentence_words
            elif word_count == 0:  # Ensure at least one sentence
                selected.append(sentence)
                break

        # Reconstruct in original order
        final_sentences = []
        for sentence in sentences:
            if sentence in selected:
                final_sentences.append(sentence)

        return ' '.join(final_sentences)

    def _score_sentence_importance(
        self,
        sentence: str,
        position: int,
        total_sentences: int
    ) -> float:
        """Score sentence importance for compression."""
        score = 0.0

        # Position-based scoring (first and last sentences are important)
        if position == 0:
            score += 0.3
        elif position == total_sentences - 1:
            score += 0.2
        else:
            # Middle sentences get moderate score
            score += 0.1

        # Keyword indicators of importance
        important_keywords = [
            "therefore", "thus", "hence", "because", "result",
            "conclusion", "summary", "key", "important", "main"
        ]
        for keyword in important_keywords:
            if keyword in sentence.lower():
                score += 0.2

        # Length penalty (prefer concise sentences)
        words = len(sentence.split())
        if words < 15:
            score += 0.1
        elif words > 30:
            score -= 0.1

        # Presence of numbers or specific data
        if any(char.isdigit() for char in sentence):
            score += 0.15

        return score

    def _clean_compressed_text(self, text: str) -> str:
        """Clean up compressed text."""
        # Remove redundant spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Ensure proper sentence endings
        if text and text[-1] not in '.!?':
            text += '.'

        # Remove any duplicate phrases
        sentences = text.split('.')
        unique = []
        seen = set()
        for sent in sentences:
            sent = sent.strip()
            if sent and sent.lower() not in seen:
                seen.add(sent.lower())
                unique.append(sent)

        return '. '.join(unique) + '.' if unique else text

    def _smart_truncate(self, text: str, target_words: int) -> str:
        """Intelligently truncate text to target length."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= target_words:
                result.append(sentence)
                word_count += sentence_words
            else:
                break

        # If we haven't added anything, add at least the first sentence
        if not result and sentences:
            result.append(sentences[0])

        return ' '.join(result)

    async def distill_reasoning(
        self,
        full_reasoning: str,
        target_steps: int = 3
    ) -> List[str]:
        """Distill reasoning into key steps using language model.

        Args:
            full_reasoning: Full reasoning text
            target_steps: Number of steps to extract

        Returns:
            List of key reasoning steps
        """
        if not self._loaded:
            self.load()

        prompt = f"""Extract exactly {target_steps} key logical steps from this reasoning.
Each step should be a complete, self-contained statement.

Reasoning:
{full_reasoning}

List the {target_steps} most important steps:
1."""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1000
                ).to(self.device)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=150,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            # Parse the steps
            steps = self._parse_numbered_list(response, target_steps)

            if len(steps) < target_steps:
                # Fallback to heuristic extraction
                steps = self._extract_steps_heuristic(full_reasoning, target_steps)

            return steps

        except Exception as e:
            logger.warning(f"Failed to distill reasoning with LLM: {str(e)}")
            return self._extract_steps_heuristic(full_reasoning, target_steps)

    def _parse_numbered_list(self, text: str, expected_count: int) -> List[str]:
        """Parse a numbered list from text."""
        steps = []

        # Try to find numbered items
        patterns = [
            r'\d+\.\s*(.+?)(?=\d+\.|$)',  # 1. Step one 2. Step two
            r'Step \d+:\s*(.+?)(?=Step \d+|$)',  # Step 1: ... Step 2:
            r'-\s*(.+?)(?=-|$)',  # - Step one - Step two
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                steps = [match.strip() for match in matches]
                break

        # If no pattern matched, split by newlines
        if not steps:
            lines = text.split('\n')
            steps = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]

        return steps[:expected_count]

    def _extract_steps_heuristic(
        self,
        reasoning: str,
        target_steps: int
    ) -> List[str]:
        """Extract steps using heuristic method."""
        sentences = re.split(r'(?<=[.!?])\s+', reasoning)

        if len(sentences) <= target_steps:
            return [f"Step {i+1}: {sent}" for i, sent in enumerate(sentences)]

        # Select evenly spaced sentences
        step_size = len(sentences) // target_steps
        steps = []

        for i in range(target_steps):
            idx = i * step_size
            if idx < len(sentences):
                steps.append(f"Step {i+1}: {sentences[idx]}")

        return steps


class MockCoTCompressorSLM(BaseSLM):
    """Mock CoT compressor for testing."""

    def __init__(self):
        super().__init__("mock")
        self._loaded = True

    async def process(
        self,
        query: str = None,
        summarize: str = None,
        **kwargs
    ) -> str:
        if not summarize:
            return ""

        # Simple truncation as mock compression
        words = summarize.split()
        compressed_words = words[:max(10, len(words) // 3)]
        return ' '.join(compressed_words)


class ReasoningChainOptimizer(CoTCompressorSLM):
    """Optimizes and refines reasoning chains for clarity and coherence."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        compression_ratio: float = 0.5
    ):
        super().__init__(model_name, compression_ratio)

    async def optimize_chain(
        self,
        reasoning: str,
        query: str,
        focus: str = "clarity"
    ) -> str:
        """Optimize reasoning chain for specific qualities.

        Args:
            reasoning: Original reasoning
            query: User query
            focus: Optimization focus (clarity, brevity, completeness)

        Returns:
            Optimized reasoning
        """
        if not self._loaded:
            self.load()

        # Create optimization prompt based on focus
        if focus == "clarity":
            instruction = "Rewrite for maximum clarity and logical flow"
        elif focus == "brevity":
            instruction = "Make as concise as possible while preserving key points"
        elif focus == "completeness":
            instruction = "Ensure all logical steps are explicit and well-explained"
        else:
            instruction = "Optimize for clarity and conciseness"

        prompt = f"""Optimize this reasoning chain.
{instruction}

Query: {query}

Original reasoning:
{reasoning}

Optimized reasoning:"""

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1500
                ).to(self.device)

                target_length = len(reasoning.split())
                if focus == "brevity":
                    target_length = int(target_length * 0.5)

                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=target_length,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                optimized = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            return self._clean_compressed_text(optimized)

        except Exception as e:
            logger.error(f"Failed to optimize reasoning: {str(e)}")
            return reasoning