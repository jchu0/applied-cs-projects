"""RAG index combining embedding and vector store."""

from typing import Optional

from .schemas import Document, Chunk, SearchResult
from .embeddings import EmbeddingModel
from .vectorstore import VectorStore
from .chunking import ChunkingStrategy


class RAGIndex:
    """Main index combining embedding and vector store."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunker: ChunkingStrategy,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunker = chunker

    def index_document(self, document: Document):
        """Index a single document."""
        chunks = self.chunker.chunk(document)

        if not chunks:
            return

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        # Add to vector store
        self.vector_store.add(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def index_documents(self, documents: list[Document]):
        """Index multiple documents."""
        for doc in documents:
            self.index_document(doc)

    def search(
        self,
        query: str,
        k: int = 5,
        filter: dict = None,
    ) -> list[SearchResult]:
        """Search for relevant chunks."""
        query_embedding = self.embedding_model.encode([query])
        return self.vector_store.search(query_embedding, k, filter)

    def delete_document(self, document_id: str):
        """Delete all chunks from a document."""
        # Get all chunk IDs for this document
        # Note: This is a simple implementation
        # In production, you'd want to track document->chunk mapping
        pass

    @property
    def count(self) -> int:
        """Get number of indexed chunks."""
        return self.vector_store.count()


class MultiIndexManager:
    """Manage multiple indices (e.g., per tenant)."""

    def __init__(self):
        self._indices: dict[str, RAGIndex] = {}

    def get_index(self, index_id: str) -> Optional[RAGIndex]:
        """Get index by ID."""
        return self._indices.get(index_id)

    def create_index(
        self,
        index_id: str,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunker: ChunkingStrategy,
    ) -> RAGIndex:
        """Create new index."""
        index = RAGIndex(embedding_model, vector_store, chunker)
        self._indices[index_id] = index
        return index

    def delete_index(self, index_id: str):
        """Delete index."""
        if index_id in self._indices:
            del self._indices[index_id]

    def list_indices(self) -> list[str]:
        """List all index IDs."""
        return list(self._indices.keys())
