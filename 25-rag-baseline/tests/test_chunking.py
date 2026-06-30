"""Tests for chunking module."""

import pytest
from unittest.mock import Mock, patch
from typing import List

from ragbaseline.chunking import (
    ChunkingStrategy,
    FixedSizeChunker,
    SentenceChunker,
    SemanticChunker,
    HierarchicalChunker,
    Document,
    Chunk,
)


class TestChunkingStrategy:
    """Test base chunking strategy interface."""

    def test_abstract_methods(self):
        """Test that abstract methods are enforced."""
        with pytest.raises(TypeError):
            ChunkingStrategy()


class TestFixedSizeChunker:
    """Test fixed-size chunking strategy."""

    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        return Document(
            id="doc1",
            content="This is the first sentence. This is the second sentence. "
                   "This is the third sentence. This is the fourth sentence. "
                   "This is the fifth sentence. This is the sixth sentence.",
            metadata={"source": "test.txt", "author": "Test Author"}
        )

    def test_basic_chunking(self, sample_document):
        """Test basic fixed-size chunking."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(len(c.content) <= 50 for c in chunks)
        assert all(c.document_id == "doc1" for c in chunks)

    def test_chunking_with_overlap(self, sample_document):
        """Test chunking with overlap."""
        chunker = FixedSizeChunker(chunk_size=60, overlap=20)
        chunks = chunker.chunk(sample_document)

        # Check that consecutive chunks have overlapping content
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i].content[-20:]
            chunk2_start = chunks[i + 1].content[:20]

            # There should be some overlap
            if len(chunk1_end) == 20 and len(chunk2_start) == 20:
                assert chunk1_end in chunks[i + 1].content

    def test_metadata_preservation(self, sample_document):
        """Test that metadata is preserved in chunks."""
        chunker = FixedSizeChunker(chunk_size=50)
        chunks = chunker.chunk(sample_document)

        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["author"] == "Test Author"
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata

    def test_empty_document(self):
        """Test handling of empty document."""
        doc = Document(id="empty", content="", metadata={})
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 0

    def test_small_document(self):
        """Test document smaller than chunk size."""
        doc = Document(id="small", content="Small text", metadata={})
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Small text"

    def test_word_boundary_splitting(self):
        """Test that chunking respects word boundaries."""
        doc = Document(
            id="test",
            content="This is a longer sentence that should be split carefully at word boundaries.",
            metadata={}
        )
        chunker = FixedSizeChunker(chunk_size=30, overlap=5, split_on="word")
        chunks = chunker.chunk(doc)

        # Check no words are split
        for chunk in chunks:
            assert not chunk.content.startswith(" ")
            assert not chunk.content.endswith(" ")


class TestSentenceChunker:
    """Test sentence-based chunking strategy."""

    @pytest.fixture
    def multi_sentence_document(self):
        """Create document with multiple sentences."""
        return Document(
            id="doc1",
            content=(
                "This is sentence one. This is sentence two! "
                "Is this sentence three? This is sentence four; "
                "it has a semicolon. This is sentence five... "
                "This is the final sentence."
            ),
            metadata={"type": "test"}
        )

    def test_sentence_detection(self, multi_sentence_document):
        """Test correct sentence detection."""
        chunker = SentenceChunker(sentences_per_chunk=1)
        chunks = chunker.chunk(multi_sentence_document)

        assert len(chunks) == 6  # 6 sentences
        assert chunks[0].content.strip() == "This is sentence one."
        assert chunks[1].content.strip() == "This is sentence two!"

    def test_multiple_sentences_per_chunk(self, multi_sentence_document):
        """Test grouping multiple sentences per chunk."""
        chunker = SentenceChunker(sentences_per_chunk=2, overlap_sentences=0)
        chunks = chunker.chunk(multi_sentence_document)

        assert len(chunks) == 3  # 6 sentences / 2 per chunk
        assert "This is sentence one" in chunks[0].content
        assert "This is sentence two" in chunks[0].content

    def test_sentence_overlap(self, multi_sentence_document):
        """Test overlapping sentences between chunks."""
        chunker = SentenceChunker(sentences_per_chunk=2, overlap_sentences=1)
        chunks = chunker.chunk(multi_sentence_document)

        # Check overlap
        for i in range(len(chunks) - 1):
            # Last sentence of chunk i should be first sentence of chunk i+1
            sentences_i = chunker._split_sentences(chunks[i].content)
            sentences_next = chunker._split_sentences(chunks[i + 1].content)

            if sentences_i and sentences_next:
                assert sentences_i[-1].strip() == sentences_next[0].strip()

    def test_handling_abbreviations(self):
        """Test handling of abbreviations and edge cases."""
        doc = Document(
            id="test",
            content="Dr. Smith works at U.S.A. Inc. He is very professional.",
            metadata={}
        )
        chunker = SentenceChunker(sentences_per_chunk=1)
        chunks = chunker.chunk(doc)

        # Should correctly identify 2 sentences, not split on abbreviations
        assert len(chunks) == 2
        assert "Dr. Smith" in chunks[0].content
        assert "U.S.A. Inc" in chunks[0].content

    def test_handling_special_punctuation(self):
        """Test handling of various punctuation marks."""
        doc = Document(
            id="test",
            content='He said "Hello!" Then he left... Really? Yes! (Amazing.)',
            metadata={}
        )
        chunker = SentenceChunker(sentences_per_chunk=1)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 3  # At least 3 clear sentences


class TestSemanticChunker:
    """Test semantic chunking strategy."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder for semantic chunking."""
        embedder = Mock()
        # Return incrementing embeddings for each sentence
        embedder.embed = Mock(side_effect=lambda text: [len(text) / 100.0])
        return embedder

    @pytest.mark.asyncio
    async def test_semantic_grouping(self, mock_embedder):
        """Test grouping by semantic similarity."""
        doc = Document(
            id="test",
            content=(
                "The cat sat on the mat. The dog played in the yard. "
                "Machine learning is fascinating. Neural networks are complex. "
                "The weather is nice today."
            ),
            metadata={}
        )

        chunker = SemanticChunker(
            embedder=mock_embedder,
            similarity_threshold=0.7
        )
        chunks = await chunker.chunk_async(doc)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_max_chunk_size_limit(self, mock_embedder):
        """Test that semantic chunks respect max size limit."""
        doc = Document(
            id="test",
            content=" ".join(["This is a test sentence." for _ in range(20)]),
            metadata={}
        )

        chunker = SemanticChunker(
            embedder=mock_embedder,
            similarity_threshold=0.9,
            max_chunk_size=100
        )
        chunks = await chunker.chunk_async(doc)

        # All chunks should be under max size
        assert all(len(c.content) <= 100 for c in chunks)

    @pytest.mark.asyncio
    async def test_semantic_metadata(self, mock_embedder):
        """Test that semantic chunking adds appropriate metadata."""
        doc = Document(
            id="test",
            content="Test content for semantic chunking.",
            metadata={"original": "data"}
        )

        chunker = SemanticChunker(embedder=mock_embedder)
        chunks = await chunker.chunk_async(doc)

        for chunk in chunks:
            assert "original" in chunk.metadata
            assert "semantic_chunk" in chunk.metadata
            assert chunk.metadata["semantic_chunk"] is True
            assert "similarity_threshold" in chunk.metadata


class TestHierarchicalChunker:
    """Test hierarchical chunking strategy."""

    @pytest.fixture
    def markdown_document(self):
        """Create a markdown document with hierarchy."""
        return Document(
            id="doc1",
            content="""
# Main Title

## Section 1
This is the content of section 1.
It has multiple paragraphs.

### Subsection 1.1
Details about subsection 1.1.

### Subsection 1.2
Details about subsection 1.2.

## Section 2
This is section 2 content.

### Subsection 2.1
More detailed content here.

## Section 3
Final section content.
""",
            metadata={"format": "markdown"}
        )

    def test_hierarchy_detection(self, markdown_document):
        """Test detection of document hierarchy."""
        chunker = HierarchicalChunker(min_chunk_size=10)
        chunks = chunker.chunk(markdown_document)

        assert len(chunks) > 0

        # Check that chunks maintain hierarchy information
        for chunk in chunks:
            assert "hierarchy_level" in chunk.metadata
            assert "section_path" in chunk.metadata

    def test_section_based_chunking(self, markdown_document):
        """Test chunking based on sections."""
        chunker = HierarchicalChunker(
            split_level="section",
            min_chunk_size=10
        )
        chunks = chunker.chunk(markdown_document)

        # Should have chunks for each major section
        section_titles = [c.metadata.get("section_title") for c in chunks]
        assert any("Section 1" in str(t) for t in section_titles)
        assert any("Section 2" in str(t) for t in section_titles)

    def test_nested_hierarchy_preservation(self, markdown_document):
        """Test that nested hierarchy is preserved."""
        chunker = HierarchicalChunker(
            split_level="subsection",
            preserve_hierarchy=True
        )
        chunks = chunker.chunk(markdown_document)

        # Check subsections maintain parent section info
        subsection_chunks = [
            c for c in chunks
            if "Subsection" in c.metadata.get("section_title", "")
        ]

        assert len(subsection_chunks) > 0
        for chunk in subsection_chunks:
            assert "parent_section" in chunk.metadata

    def test_min_chunk_size_enforcement(self):
        """Test that minimum chunk size is enforced."""
        doc = Document(
            id="test",
            content="""
# Title
Short.

## Section
Also short.

### Subsection
This is a longer piece of content that should not be merged.
""",
            metadata={}
        )

        chunker = HierarchicalChunker(min_chunk_size=50)
        chunks = chunker.chunk(doc)

        # Short sections should be merged
        assert not any(c.content.strip() == "Short." for c in chunks)

    def test_code_block_handling(self):
        """Test handling of code blocks in hierarchical documents."""
        doc = Document(
            id="test",
            content="""
# Code Examples

## Python Example
```python
def hello_world():
    print("Hello, World!")
```

## JavaScript Example
```javascript
function helloWorld() {
    console.log("Hello, World!");
}
```
""",
            metadata={}
        )

        chunker = HierarchicalChunker(preserve_code_blocks=True)
        chunks = chunker.chunk(doc)

        # Code blocks should be preserved intact
        python_chunks = [c for c in chunks if "python" in c.content.lower()]
        assert len(python_chunks) > 0
        assert "def hello_world():" in python_chunks[0].content


class TestChunkUtilities:
    """Test utility functions and chunk operations."""

    def test_chunk_id_generation(self):
        """Test that chunk IDs are unique and consistent."""
        doc = Document(id="doc1", content="Test content", metadata={})
        chunker = FixedSizeChunker(chunk_size=5)
        chunks = chunker.chunk(doc)

        # All chunk IDs should be unique
        chunk_ids = [c.id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

        # IDs should follow pattern
        for i, chunk in enumerate(chunks):
            assert f"doc1_chunk_{i}" in chunk.id

    def test_chunk_serialization(self):
        """Test chunk serialization and deserialization."""
        chunk = Chunk(
            id="test_chunk",
            content="Test content",
            document_id="doc1",
            metadata={"key": "value", "number": 42},
            start_char=0,
            end_char=12
        )

        # Test to_dict
        chunk_dict = chunk.to_dict()
        assert chunk_dict["id"] == "test_chunk"
        assert chunk_dict["content"] == "Test content"
        assert chunk_dict["metadata"]["key"] == "value"

        # Test from_dict
        restored_chunk = Chunk.from_dict(chunk_dict)
        assert restored_chunk.id == chunk.id
        assert restored_chunk.content == chunk.content
        assert restored_chunk.metadata == chunk.metadata

    def test_chunk_overlap_calculation(self):
        """Test overlap calculation between chunks."""
        doc = Document(
            id="test",
            content="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            metadata={}
        )

        chunker = FixedSizeChunker(chunk_size=10, overlap=3)
        chunks = chunker.chunk(doc)

        # Calculate actual overlap
        for i in range(len(chunks) - 1):
            overlap_content = set(chunks[i].content) & set(chunks[i + 1].content)
            # Should have at least some overlapping characters
            if len(chunks[i].content) == 10 and len(chunks[i + 1].content) >= 3:
                assert len(overlap_content) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])