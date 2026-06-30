"""Chunking strategies for document processing."""

from abc import ABC, abstractmethod
import re
import numpy as np

from .schemas import Document, Chunk


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into chunks."""
        pass


class FixedSizeChunker(ChunkingStrategy):
    """Fixed size chunks with overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = None,
        overlap: int = None,
        split_on: str = None,  # "word" for word boundary splitting
    ):
        self.chunk_size = chunk_size
        # Support both parameter names, default to 0 if not specified
        if overlap is not None:
            self.chunk_overlap = overlap
        elif chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        else:
            self.chunk_overlap = 0
        self.split_on = split_on

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        chunks = []
        start = 0
        chunk_index = 0

        # Calculate step size (must be positive)
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < len(text):
            end = start + self.chunk_size

            # Handle word boundary splitting
            if self.split_on == "word" and end < len(text):
                # Find last space before end
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end]

            # Strip leading/trailing spaces for word mode
            if self.split_on == "word":
                chunk_text = chunk_text.strip()

            # Skip empty chunks
            if chunk_text.strip():
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    content=chunk_text,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata={
                        **document.metadata,
                        "start_char": start,
                        "end_char": min(end, len(text)),
                        "chunk_index": chunk_index,
                        "total_chunks": 0,  # Will be updated below
                    },
                    start_char=start,
                    end_char=min(end, len(text)),
                ))
                chunk_index += 1

            start += step

        # Update total_chunks in all chunk metadata
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_chunks"] = total

        return chunks


class SentenceChunker(ChunkingStrategy):
    """Chunk by sentences with configurable grouping."""

    def __init__(
        self,
        sentences_per_chunk: int = 3,
        overlap_sentences: int = 0,
        max_chunk_size: int = None,
        min_chunk_size: int = None,
    ):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []

        chunks = []
        chunk_index = 0
        i = 0

        while i < len(sentences):
            # Get sentences for this chunk
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences)

            chunks.append(Chunk(
                id=f"{document.id}_chunk_{chunk_index}",
                content=chunk_text,
                document_id=document.id,
                chunk_index=chunk_index,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "sentence_count": len(chunk_sentences),
                },
            ))
            chunk_index += 1

            # Move forward by sentences_per_chunk minus overlap
            step = max(1, self.sentences_per_chunk - self.overlap_sentences)
            i += step

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences with abbreviation handling."""
        if not text:
            return []

        # Honorific titles - these are ALWAYS followed by a name, never end sentence
        honorifics = {'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Jr', 'Sr'}

        # Other abbreviations - these may or may not end a sentence
        # If followed by common sentence starters, they likely end a sentence
        other_abbreviations = {
            'Inc', 'Ltd', 'Corp', 'Co', 'vs', 'etc',
            'U.S.A', 'U.S', 'U.K', 'E.U',
            'i.e', 'e.g', 'al', 'ca', 'cf',
        }

        # Common words that typically start new sentences
        sentence_starters = {
            'He', 'She', 'It', 'They', 'We', 'I', 'You',
            'The', 'A', 'An', 'This', 'That', 'These', 'Those',
            'There', 'Here', 'What', 'When', 'Where', 'Why', 'How',
            'If', 'But', 'And', 'So', 'Yet', 'Or', 'For',
            'However', 'Moreover', 'Furthermore', 'Therefore',
            'Meanwhile', 'Finally', 'First', 'Second', 'Next',
        }

        sentences = []
        current_start = 0

        i = 0
        while i < len(text):
            # Look for sentence-ending punctuation
            if text[i] in '.!?':
                # Check if followed by space and uppercase (potential sentence boundary)
                if i + 2 < len(text) and text[i + 1] == ' ' and text[i + 2].isupper():
                    # Get the word before punctuation
                    word_start = i - 1
                    while word_start >= current_start and text[word_start] not in ' \t\n':
                        word_start -= 1
                    word_start += 1
                    word_before = text[word_start:i]

                    # Handle multi-period abbreviations like U.S.A
                    check_abbrev = word_before.replace('.', '') if '.' in word_before else word_before

                    # Get the word after the punctuation
                    word_after_start = i + 2
                    word_after_end = word_after_start
                    while word_after_end < len(text) and text[word_after_end].isalpha():
                        word_after_end += 1
                    word_after = text[word_after_start:word_after_end]

                    # Determine if this is a sentence boundary
                    is_honorific = check_abbrev in honorifics or word_before in honorifics
                    is_other_abbrev = check_abbrev in other_abbreviations or word_before in other_abbreviations
                    next_is_starter = word_after in sentence_starters

                    # Split if:
                    # 1. Not a honorific AND
                    # 2. Either not an abbreviation OR next word is a sentence starter
                    should_split = (not is_honorific and
                                    (not is_other_abbrev or next_is_starter))

                    if should_split:
                        sentence = text[current_start:i + 1].strip()
                        if sentence:
                            sentences.append(sentence)
                        current_start = i + 2
                        i = current_start
                        continue
            i += 1

        # Add remaining text
        remaining = text[current_start:].strip()
        if remaining:
            sentences.append(remaining)

        return sentences


class RecursiveChunker(ChunkingStrategy):
    """Recursively split by separators."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, document: Document) -> list[Chunk]:
        chunks = self._split_text(document.content, self.separators)

        return [
            Chunk(
                id=f"{document.id}_chunk_{i}",
                content=chunk,
                document_id=document.id,
                chunk_index=i,
                metadata=document.metadata,
            )
            for i, chunk in enumerate(chunks)
        ]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text."""
        if not text:
            return []

        # Get current separator
        separator = separators[0]
        remaining_separators = separators[1:] if len(separators) > 1 else [""]

        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits into chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for split in splits:
            split_size = len(split)

            if current_size + split_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                if len(chunk_text) > self.chunk_size and remaining_separators:
                    # Recursively split if still too large
                    chunks.extend(self._split_text(chunk_text, remaining_separators))
                else:
                    chunks.append(chunk_text)

                # Handle overlap
                overlap_text = separator.join(current_chunk[-2:]) if len(current_chunk) > 1 else ""
                if len(overlap_text) < self.chunk_overlap:
                    current_chunk = current_chunk[-2:] if len(current_chunk) > 1 else []
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(split)
            current_size += split_size + len(separator)

        # Handle last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if chunk_text.strip():
                if len(chunk_text) > self.chunk_size and remaining_separators:
                    chunks.extend(self._split_text(chunk_text, remaining_separators))
                else:
                    chunks.append(chunk_text)

        return [c for c in chunks if c.strip()]


class SemanticChunker(ChunkingStrategy):
    """Chunk by semantic similarity (paragraph boundaries)."""

    def __init__(
        self,
        embedder=None,
        embedding_model=None,  # Alias for backwards compatibility
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 1000,
    ):
        # Support both parameter names
        self.embedder = embedder or embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        """Synchronous chunk method (uses numpy for embeddings)."""
        # Split by paragraphs
        paragraphs = document.content.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        if len(paragraphs) == 1:
            return [Chunk(
                id=f"{document.id}_chunk_0",
                content=paragraphs[0],
                document_id=document.id,
                chunk_index=0,
                metadata={
                    **document.metadata,
                    "semantic_chunk": True,
                    "similarity_threshold": self.similarity_threshold,
                },
            )]

        # Get embeddings if embedder available
        if self.embedder:
            embeddings = [self.embedder.embed(p) for p in paragraphs]
        else:
            # Simple fallback: use paragraph length as "embedding"
            embeddings = [[len(p) / 100.0] for p in paragraphs]

        # Group by similarity
        chunks = []
        current_group = [paragraphs[0]]
        current_size = len(paragraphs[0])
        chunk_index = 0

        for i in range(1, len(paragraphs)):
            # Compute similarity with previous
            similarity = self._cosine_similarity(
                np.array(embeddings[i - 1]),
                np.array(embeddings[i])
            )

            # Check if should merge
            if (similarity > self.similarity_threshold and
                    current_size + len(paragraphs[i]) < self.max_chunk_size):
                current_group.append(paragraphs[i])
                current_size += len(paragraphs[i])
            else:
                # Save current chunk
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    content="\n\n".join(current_group),
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata={
                        **document.metadata,
                        "semantic_chunk": True,
                        "similarity_threshold": self.similarity_threshold,
                    },
                ))

                current_group = [paragraphs[i]]
                current_size = len(paragraphs[i])
                chunk_index += 1

        # Last chunk
        if current_group:
            chunks.append(Chunk(
                id=f"{document.id}_chunk_{chunk_index}",
                content="\n\n".join(current_group),
                document_id=document.id,
                chunk_index=chunk_index,
                metadata={
                    **document.metadata,
                    "semantic_chunk": True,
                    "similarity_threshold": self.similarity_threshold,
                },
            ))

        return chunks

    async def chunk_async(self, document: Document) -> list[Chunk]:
        """Async chunk method for use with async embedders."""
        # Split by paragraphs
        paragraphs = document.content.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        # Split paragraphs that are too large
        split_paragraphs = []
        for p in paragraphs:
            if len(p) > self.max_chunk_size:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', p)
                current = []
                current_len = 0
                for s in sentences:
                    if current_len + len(s) > self.max_chunk_size and current:
                        split_paragraphs.append(" ".join(current))
                        current = [s]
                        current_len = len(s)
                    else:
                        current.append(s)
                        current_len += len(s) + 1
                if current:
                    split_paragraphs.append(" ".join(current))
            else:
                split_paragraphs.append(p)
        paragraphs = split_paragraphs

        if len(paragraphs) == 1:
            return [Chunk(
                id=f"{document.id}_chunk_0",
                content=paragraphs[0],
                document_id=document.id,
                chunk_index=0,
                metadata={
                    **document.metadata,
                    "semantic_chunk": True,
                    "similarity_threshold": self.similarity_threshold,
                },
            )]

        # Get embeddings
        if self.embedder:
            if hasattr(self.embedder, 'embed') and callable(self.embedder.embed):
                # Mock or sync embedder
                embeddings = [self.embedder.embed(p) for p in paragraphs]
            elif hasattr(self.embedder, 'embed_batch'):
                embeddings = await self.embedder.embed_batch(paragraphs)
            else:
                embeddings = [[len(p) / 100.0] for p in paragraphs]
        else:
            embeddings = [[len(p) / 100.0] for p in paragraphs]

        # Group by similarity
        chunks = []
        current_group = [paragraphs[0]]
        current_size = len(paragraphs[0])
        chunk_index = 0

        for i in range(1, len(paragraphs)):
            emb_prev = np.array(embeddings[i - 1])
            emb_curr = np.array(embeddings[i])
            similarity = self._cosine_similarity(emb_prev, emb_curr)

            # Check if should merge
            if (similarity > self.similarity_threshold and
                    current_size + len(paragraphs[i]) < self.max_chunk_size):
                current_group.append(paragraphs[i])
                current_size += len(paragraphs[i])
            else:
                # Save current chunk
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    content="\n\n".join(current_group),
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata={
                        **document.metadata,
                        "semantic_chunk": True,
                        "similarity_threshold": self.similarity_threshold,
                    },
                ))

                current_group = [paragraphs[i]]
                current_size = len(paragraphs[i])
                chunk_index += 1

        # Last chunk
        if current_group:
            chunks.append(Chunk(
                id=f"{document.id}_chunk_{chunk_index}",
                content="\n\n".join(current_group),
                document_id=document.id,
                chunk_index=chunk_index,
                metadata={
                    **document.metadata,
                    "semantic_chunk": True,
                    "similarity_threshold": self.similarity_threshold,
                },
            ))

        return chunks

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        a = np.atleast_1d(a)
        b = np.atleast_1d(b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class HierarchicalChunker(ChunkingStrategy):
    """Hierarchical chunking that preserves document structure."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
        split_level: str = "paragraph",  # "section", "subsection", "paragraph"
        preserve_hierarchy: bool = True,
        preserve_code_blocks: bool = False,
        levels: list[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.split_level = split_level
        self.preserve_hierarchy = preserve_hierarchy
        self.preserve_code_blocks = preserve_code_blocks
        self.levels = levels or ["section", "paragraph", "sentence"]

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document hierarchically."""
        content = document.content

        # Parse markdown headers
        sections = self._parse_markdown_hierarchy(content)

        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._process_section(
                document, section, chunk_index
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # Handle min_chunk_size by merging small chunks
        merged_chunks = self._merge_small_chunks(chunks, document)

        # Re-index after merging
        for i, chunk in enumerate(merged_chunks):
            chunk.chunk_index = i
            chunk.metadata["chunk_index"] = i

        return merged_chunks

    def _parse_markdown_hierarchy(self, content: str) -> list[dict]:
        """Parse markdown content into hierarchical sections."""
        lines = content.split("\n")
        sections = []
        current_section = {
            "level": 0,
            "title": "",
            "content": [],
            "parent_section": None,
            "section_path": [],
            "code_blocks": [],
        }
        in_code_block = False
        code_block_content = []

        section_stack = []

        for line in lines:
            # Handle code blocks
            if line.strip().startswith("```"):
                if in_code_block:
                    # End code block
                    code_block_content.append(line)
                    current_section["code_blocks"].append("\n".join(code_block_content))
                    current_section["content"].append("\n".join(code_block_content))
                    code_block_content = []
                    in_code_block = False
                else:
                    # Start code block
                    in_code_block = True
                    code_block_content = [line]
                continue

            if in_code_block:
                code_block_content.append(line)
                continue

            # Check for markdown headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save current section if it has content
                if current_section["content"] or current_section["title"]:
                    sections.append(current_section)

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Update section stack
                while section_stack and section_stack[-1]["level"] >= level:
                    section_stack.pop()

                parent = section_stack[-1]["title"] if section_stack else None
                section_path = [s["title"] for s in section_stack] + [title]

                current_section = {
                    "level": level,
                    "title": title,
                    "content": [],
                    "parent_section": parent,
                    "section_path": section_path,
                    "code_blocks": [],
                }

                section_stack.append({"level": level, "title": title})
            else:
                current_section["content"].append(line)

        # Don't forget last section
        if current_section["content"] or current_section["title"]:
            sections.append(current_section)

        return sections

    def _process_section(
        self, document: Document, section: dict, start_index: int
    ) -> list[Chunk]:
        """Process a single section into chunks."""
        content = "\n".join(section["content"]).strip()
        if not content and not section["title"]:
            return []

        # Include title in content if present
        if section["title"]:
            full_content = f"# {section['title']}\n\n{content}" if content else f"# {section['title']}"
        else:
            full_content = content

        # Check if content fits in one chunk
        if len(full_content) <= self.chunk_size:
            return [Chunk(
                id=f"{document.id}_chunk_{start_index}",
                content=full_content,
                document_id=document.id,
                chunk_index=start_index,
                metadata={
                    **document.metadata,
                    "hierarchy_level": section["level"],
                    "section_path": section["section_path"],
                    "section_title": section["title"],
                    "parent_section": section["parent_section"],
                },
            )]

        # Split into smaller chunks
        chunks = []
        chunk_index = start_index

        # If preserving code blocks, extract them first
        if self.preserve_code_blocks and section["code_blocks"]:
            # Create separate chunks for code blocks
            for code_block in section["code_blocks"]:
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    content=code_block,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata={
                        **document.metadata,
                        "hierarchy_level": section["level"],
                        "section_path": section["section_path"],
                        "section_title": section["title"],
                        "parent_section": section["parent_section"],
                        "is_code_block": True,
                    },
                ))
                chunk_index += 1

            # Remove code blocks from content for remaining processing
            remaining_content = content
            for code_block in section["code_blocks"]:
                remaining_content = remaining_content.replace(code_block, "")
            full_content = remaining_content.strip()

        # Split remaining content
        start = 0
        while start < len(full_content):
            end = min(start + self.chunk_size, len(full_content))

            # Try to break at sentence/paragraph boundary
            if end < len(full_content):
                for delim in ["\n\n", "\n", ". ", "! ", "? "]:
                    last_delim = full_content[start:end].rfind(delim)
                    if last_delim > self.chunk_size // 2:
                        end = start + last_delim + len(delim)
                        break

            chunk_text = full_content[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    content=chunk_text,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata={
                        **document.metadata,
                        "hierarchy_level": section["level"],
                        "section_path": section["section_path"],
                        "section_title": section["title"],
                        "parent_section": section["parent_section"],
                    },
                ))
                chunk_index += 1

            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def _merge_small_chunks(
        self, chunks: list[Chunk], document: Document
    ) -> list[Chunk]:
        """Merge chunks smaller than min_chunk_size."""
        if not chunks:
            return chunks

        merged = []
        current_content = []
        current_metadata = None

        for chunk in chunks:
            if len(chunk.content) >= self.min_chunk_size:
                # Flush any accumulated small chunks
                if current_content:
                    merged.append(Chunk(
                        id=f"{document.id}_chunk_{len(merged)}",
                        content="\n\n".join(current_content),
                        document_id=document.id,
                        chunk_index=len(merged),
                        metadata=current_metadata or {},
                    ))
                    current_content = []
                    current_metadata = None

                merged.append(chunk)
            else:
                # Accumulate small chunk
                current_content.append(chunk.content)
                if current_metadata is None:
                    current_metadata = chunk.metadata.copy()

                # Check if accumulated is now large enough
                combined = "\n\n".join(current_content)
                if len(combined) >= self.min_chunk_size:
                    merged.append(Chunk(
                        id=f"{document.id}_chunk_{len(merged)}",
                        content=combined,
                        document_id=document.id,
                        chunk_index=len(merged),
                        metadata=current_metadata,
                    ))
                    current_content = []
                    current_metadata = None

        # Flush remaining
        if current_content:
            # Merge with previous chunk if possible
            if merged and len("\n\n".join(current_content)) < self.min_chunk_size:
                last_chunk = merged[-1]
                last_chunk.content = last_chunk.content + "\n\n" + "\n\n".join(current_content)
            else:
                merged.append(Chunk(
                    id=f"{document.id}_chunk_{len(merged)}",
                    content="\n\n".join(current_content),
                    document_id=document.id,
                    chunk_index=len(merged),
                    metadata=current_metadata or {},
                ))

        return merged

    def get_hierarchy(self, document: Document) -> dict:
        """Get document hierarchy structure."""
        sections = document.content.split("\n\n")
        hierarchy = {
            "document_id": document.id,
            "sections": [],
        }

        for idx, section in enumerate(sections):
            if section.strip():
                hierarchy["sections"].append({
                    "index": idx,
                    "length": len(section),
                    "preview": section[:100] + "..." if len(section) > 100 else section,
                })

        return hierarchy


# Re-export for convenience
__all__ = [
    "ChunkingStrategy",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "Document",
    "Chunk",
]
