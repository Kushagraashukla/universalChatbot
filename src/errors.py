"""Error types shared by the RAG engine, so callers (CLI, API) can tell failures apart.

Both subclass RuntimeError, so existing `except RuntimeError` code keeps working.
"""


class OllamaError(RuntimeError):
    """Ollama is unreachable, a model is missing, or a request to it failed."""


class VectorStoreError(RuntimeError):
    """The ChromaDB index is missing, empty, or was built with another embedding model."""
