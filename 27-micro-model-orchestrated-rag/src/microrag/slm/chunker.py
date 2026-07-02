"""Chunker SLM for semantic-aware document chunking."""

from typing import Optional, List
import re
import numpy as np

from .base import (
    BaseSLM,
    EmbeddingModelMixin,
    GenerativeModelMixin,
    logger,
    require_torch,
)
from ..schemas import Chunk


def __getattr__(name):
    """Lazily resolve ``torch`` so mock/offline imports don't require it."""
    if name == "torch":
        return require_torch()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ChunkerSLM(BaseSLM, EmbeddingModelMixin, GenerativeModelMixin):
    """SLM for semantic-aware document chunking using embeddings and language models."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 50,
        similarity_threshold: float = 0.75
    ):
        super().__init__(model_name)
        self.embedding_model_name = embedding_model
        self.max_tokens = max_chunk_tokens
        self.min_tokens = min_chunk_tokens
        self.similarity_threshold = similarity_threshold
        self.sentence_model = None

    def _load_model(self):
        """Load both the generative model for chunk type classification and embedding model."""
        # Load embedding model for semantic chunking
        self._load_embedding_model(self.embedding_model_name)
        self.sentence_model = self.model

        # Load generative model for chunk type classification
        self._load_generative_model(self.model_name, load_in_8bit=True)

    async def process(
        self,
        document: str = None,
        doc_type: Optional[str] = None,
        **kwargs
    ) -> List[Chunk]:
        """Chunk a document using semantic similarity.

        Args:
            document: Document text to chunk
            doc_type: Optional document type hint

        Returns:
            List of chunks
        """
        if not document:
            return []

        if not self._loaded:
            self.load()

        # Split into sentences
        sentences = self._split_sentences(document)
        if not sentences:
            return []

        # Get sentence embeddings
        sentence_embeddings = self._get_sentence_embeddings(sentences)

        # Create semantic chunks
        chunks = self._create_semantic_chunks(
            sentences,
            sentence_embeddings,
            document
        )

        # Classify chunk types using the language model
        for chunk in chunks:
            chunk.chunk_type = await self._classify_chunk_type_llm(chunk.content)
            chunk.semantic_score = await self._compute_semantic_coherence(chunk.content)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling."""
        # Use regex for better sentence splitting
        sentence_endings = re.compile(r'([.!?])\s+')
        sentences = sentence_endings.split(text)

        # Reconstruct sentences
        result = []
        current = ""
        for i, part in enumerate(sentences):
            if i % 2 == 0:
                current = part
            else:
                current += part
                if current.strip():
                    result.append(current.strip())
                current = ""

        # Add any remaining text
        if current.strip():
            result.append(current.strip())

        return result

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings for sentences using the sentence transformer."""
        if not sentences:
            return np.array([])

        with torch.no_grad():
            embeddings = self.sentence_model.encode(
                sentences,
                convert_to_numpy=True,
                show_progress_bar=False
            )

        return embeddings

    def _create_semantic_chunks(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        original_text: str
    ) -> List[Chunk]:
        """Create chunks based on semantic similarity."""
        from sklearn.metrics.pairwise import cosine_similarity
        chunks = []
        current_chunk = []
        current_chunk_embedding = None
        current_start = 0
        current_pos = 0

        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            # Check if we should start a new chunk
            should_split = False

            if current_chunk:
                # Calculate similarity with current chunk
                if current_chunk_embedding is not None:
                    similarity = cosine_similarity(
                        [current_chunk_embedding],
                        [embedding]
                    )[0][0]

                    # Split if similarity is below threshold
                    if similarity < self.similarity_threshold:
                        should_split = True

                # Also check token count
                current_tokens = sum(len(s.split()) for s in current_chunk)
                sentence_tokens = len(sentence.split())

                if current_tokens + sentence_tokens > self.max_tokens:
                    should_split = True

            # Create chunk if needed
            if should_split and len(current_chunk) > 0:
                chunk_content = " ".join(current_chunk)
                chunk_end = current_pos

                chunks.append(Chunk(
                    content=chunk_content,
                    start_idx=current_start,
                    end_idx=chunk_end,
                    chunk_type="pending",  # Will be classified later
                    semantic_score=0.0,  # Will be computed later
                    metadata={
                        "chunk_idx": len(chunks),
                        "sentence_count": len(current_chunk)
                    }
                ))

                # Start new chunk
                current_chunk = [sentence]
                current_chunk_embedding = embedding
                current_start = chunk_end + 1
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                if current_chunk_embedding is None:
                    current_chunk_embedding = embedding
                else:
                    # Update chunk embedding as average
                    current_chunk_embedding = np.mean(
                        [current_chunk_embedding, embedding],
                        axis=0
                    )

            # Update position
            current_pos += len(sentence) + 1

        # Add final chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunks.append(Chunk(
                content=chunk_content,
                start_idx=current_start,
                end_idx=len(original_text),
                chunk_type="pending",
                semantic_score=0.0,
                metadata={
                    "chunk_idx": len(chunks),
                    "sentence_count": len(current_chunk)
                }
            ))

        return chunks

    async def _classify_chunk_type_llm(self, content: str) -> str:
        """Classify chunk type using the language model."""
        # For efficiency, use simple heuristics first
        if "```" in content or re.search(r'^\s*(def |class |function |import )', content, re.MULTILINE):
            return "code_block"
        elif "|" in content and "-" in content and content.count("|") > 2:
            return "table"
        elif re.match(r'^#+\s', content.strip()):
            return "section"

        # For more complex cases, use the LLM if needed
        # This is expensive, so we'll mostly rely on heuristics
        if len(content) > 500 and self.model is not None:
            prompt = f"""Classify the following text chunk into one category:
- code_block: Contains code
- table: Contains tabular data
- section: Section header or title
- paragraph: Regular text paragraph
- list: Contains a list of items

Text: {content[:200]}...

Category:"""

            try:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256
                    ).to(self.device)

                    outputs = self.model.generate(
                        inputs['input_ids'],
                        max_new_tokens=10,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip().lower()

                    # Extract category from response
                    for category in ["code_block", "table", "section", "paragraph", "list"]:
                        if category in response:
                            return category
            except Exception as e:
                logger.warning(f"Failed to classify chunk with LLM: {str(e)}")

        return "paragraph"

    async def _compute_semantic_coherence(self, content: str) -> float:
        """Compute semantic coherence score for a chunk."""
        from sklearn.metrics.pairwise import cosine_similarity
        if not content:
            return 0.0

        # Split into sentences
        sentences = self._split_sentences(content)
        if len(sentences) <= 1:
            return 1.0  # Single sentence is perfectly coherent

        # Get embeddings
        embeddings = self._get_sentence_embeddings(sentences)
        if len(embeddings) == 0:
            return 0.5

        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[i + 1]]
            )[0][0]
            similarities.append(sim)

        # Average similarity as coherence score
        if similarities:
            coherence = float(np.mean(similarities))
            # Normalize to 0-1 range
            coherence = (coherence + 1) / 2
            return min(1.0, max(0.0, coherence))

        return 0.5


class MockChunkerSLM(BaseSLM):
    """Mock chunker for testing."""

    def __init__(self):
        super().__init__("mock")
        self._loaded = True

    async def process(
        self,
        document: str = None,
        **kwargs
    ) -> List[Chunk]:
        if not document:
            return []

        # Simple paragraph splitting
        paragraphs = document.split('\n\n')
        chunks = []
        pos = 0

        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append(Chunk(
                    content=para.strip(),
                    start_idx=pos,
                    end_idx=pos + len(para),
                    chunk_type="paragraph",
                    semantic_score=0.8,
                    metadata={"chunk_idx": i}
                ))
            pos += len(para) + 2

        return chunks


class AdaptiveChunker:
    """Adapts chunking strategy based on document type."""

    def __init__(self, chunker_slm: ChunkerSLM):
        self.slm = chunker_slm
        self.strategies = {
            "technical": {
                "max_tokens": 400,
                "min_tokens": 100,
                "similarity_threshold": 0.7
            },
            "narrative": {
                "max_tokens": 600,
                "min_tokens": 150,
                "similarity_threshold": 0.8
            },
            "code": {
                "max_tokens": 300,
                "min_tokens": 50,
                "similarity_threshold": 0.6
            },
            "legal": {
                "max_tokens": 500,
                "min_tokens": 200,
                "similarity_threshold": 0.85
            }
        }

    async def chunk(
        self,
        document: str,
        doc_type: Optional[str] = None
    ) -> List[Chunk]:
        """Chunk document with adaptive strategy.

        Args:
            document: Document text
            doc_type: Optional document type

        Returns:
            List of chunks
        """
        # Auto-detect document type if not provided
        if not doc_type:
            doc_type = self._detect_document_type(document)

        # Apply strategy
        strategy = self.strategies.get(doc_type, self.strategies["technical"])
        self.slm.max_tokens = strategy["max_tokens"]
        self.slm.min_tokens = strategy["min_tokens"]
        self.slm.similarity_threshold = strategy["similarity_threshold"]

        return await self.slm.process(document=document, doc_type=doc_type)

    def _detect_document_type(self, document: str) -> str:
        """Detect document type from content."""
        # Count code indicators
        code_indicators = document.count("```") + document.count("def ") + document.count("function ")

        # Count legal indicators
        legal_indicators = sum(1 for word in ["whereas", "hereby", "pursuant", "thereof"]
                               if word in document.lower())

        # Decide based on indicators
        if code_indicators > 5:
            return "code"
        elif legal_indicators > 3:
            return "legal"
        elif len(document.split('.')) / len(document.split()) > 0.1:  # Many sentences
            return "narrative"
        else:
            return "technical"