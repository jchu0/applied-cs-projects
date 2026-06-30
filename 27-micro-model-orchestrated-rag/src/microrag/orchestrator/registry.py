"""Component registry for managing SLM components."""

from typing import Any


class ComponentRegistry:
    """Registry for managing SLM components."""

    def __init__(self):
        self._components: dict[str, Any] = {}
        self._metadata: dict[str, dict] = {}

    def register(
        self,
        name: str,
        component: Any,
        metadata: dict = None
    ):
        """Register a component.

        Args:
            name: Unique component name
            component: Component instance or class
            metadata: Optional metadata about component
        """
        self._components[name] = component
        self._metadata[name] = metadata or {}

    def get(self, name: str) -> Any:
        """Get a component by name.

        Args:
            name: Component name

        Returns:
            Registered component

        Raises:
            KeyError: If component not found
        """
        if name not in self._components:
            available = ", ".join(self._components.keys())
            raise KeyError(f"Component '{name}' not found. Available: {available}")
        return self._components[name]

    def has(self, name: str) -> bool:
        """Check if component exists."""
        return name in self._components

    def list_components(self) -> list[str]:
        """List all registered component names."""
        return list(self._components.keys())

    def get_metadata(self, name: str) -> dict:
        """Get component metadata."""
        return self._metadata.get(name, {})

    def unregister(self, name: str):
        """Unregister a component."""
        self._components.pop(name, None)
        self._metadata.pop(name, None)

    def clear(self):
        """Clear all registrations."""
        self._components.clear()
        self._metadata.clear()


class SLMRegistry(ComponentRegistry):
    """Registry specialized for SLM components."""

    def register_slm(
        self,
        name: str,
        slm: Any,
        task: str,
        base_model: str,
        **kwargs
    ):
        """Register an SLM with additional metadata.

        Args:
            name: SLM name
            slm: SLM instance
            task: Task type (chunking, embedding, etc.)
            base_model: Base model name
            **kwargs: Additional metadata
        """
        metadata = {
            "task": task,
            "base_model": base_model,
            **kwargs
        }
        self.register(name, slm, metadata)

    def get_by_task(self, task: str) -> list[str]:
        """Get all SLMs for a specific task.

        Args:
            task: Task type

        Returns:
            List of SLM names
        """
        return [
            name for name, meta in self._metadata.items()
            if meta.get("task") == task
        ]


# Global registry instances
component_registry = ComponentRegistry()
slm_registry = SLMRegistry()


def setup_default_components():
    """Register default components."""
    from .slm.chunker import ChunkerSLM, MockChunkerSLM
    from .slm.embedder import EmbedderSLM, MockEmbedderSLM
    from .slm.retriever import RetrieverSLM, MockRetrieverSLM
    from .slm.reranker import RerankerSLM, MockRerankerSLM
    from .slm.summarizer import SummarizerSLM, MockSummarizerSLM
    from .slm.cot_compressor import CoTCompressorSLM, MockCoTCompressorSLM
    from .slm.answer_stabilizer import AnswerStabilizerSLM, MockAnswerStabilizerSLM

    # Register mock SLMs for testing
    slm_registry.register_slm(
        "chunker_slm", MockChunkerSLM(),
        task="chunking", base_model="mock"
    )
    slm_registry.register_slm(
        "embedder_slm", MockEmbedderSLM(),
        task="embedding", base_model="mock"
    )
    slm_registry.register_slm(
        "retriever_slm", MockRetrieverSLM(),
        task="retrieval", base_model="mock"
    )
    slm_registry.register_slm(
        "reranker_slm", MockRerankerSLM(),
        task="reranking", base_model="mock"
    )
    slm_registry.register_slm(
        "summarizer_slm", MockSummarizerSLM(),
        task="summarization", base_model="mock"
    )
    slm_registry.register_slm(
        "cot_compressor_slm", MockCoTCompressorSLM(),
        task="compression", base_model="mock"
    )
    slm_registry.register_slm(
        "answer_stabilizer_slm", MockAnswerStabilizerSLM(),
        task="stabilization", base_model="mock"
    )
