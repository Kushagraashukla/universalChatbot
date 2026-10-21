"""Save chunks + their vectors in a local ChromaDB database on disk.

ChromaDB keeps everything in the `chroma_db/` folder, so the data survives
restarts. Set the CHROMA_DIR environment variable to store it elsewhere.
Each knowledge base lives in its own collection (default: `knowledge_base`).
"""

import os
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_core.documents import Document

from src.errors import VectorStoreError
from src.ingestion.embedder import EMBEDDING_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = "knowledge_base"


def get_collection(name: str = COLLECTION_NAME) -> chromadb.Collection:
    """Open (or create on first run) a knowledge-base collection. Used by ingestion."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name,
        embedding_function=None,  # we always pass our own Ollama vectors; never let Chroma embed
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": EMBEDDING_MODEL},
    )
    _check_model(collection)
    return collection


def open_collection(name: str = COLLECTION_NAME) -> chromadb.Collection:
    """Open an existing collection for reading. Never creates a new, empty one."""
    if not CHROMA_DIR.is_dir():
        raise VectorStoreError(f"No vector database at {CHROMA_DIR}. Run ingestion first: python -m src.ingestion.pipeline")
    try:
        collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(name)
    except NotFoundError as exc:
        raise VectorStoreError(f"Collection '{name}' not found in {CHROMA_DIR}. Run ingestion first.") from exc
    _check_model(collection)
    return collection


def _check_model(collection: chromadb.Collection) -> None:
    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model != EMBEDDING_MODEL:
        raise VectorStoreError(
            f"Collection '{collection.name}' was built with '{stored_model}', not '{EMBEDDING_MODEL}'. "
            f"Vectors from different models cannot be mixed; delete that collection and re-run ingestion."
        )


def index_chunks(
    chunks: list[Document], vectors: list[list[float]], collection_name: str = COLLECTION_NAME
) -> chromadb.Collection:
    """Make the collection match `chunks` exactly. Safe to run again and again.

    - new or changed chunks are written (upsert = insert or overwrite by ID)
    - chunks that no longer exist (edited or deleted files) are removed
    """
    if not chunks:
        raise ValueError("No chunks to index; refusing to touch the existing index.")
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors.")

    collection = get_collection(collection_name)
    ids = [c.metadata["chunk_id"] for c in chunks]

    # shortcut: one upsert call; split into batches of client.get_max_batch_size() if the KB grows past ~5000 chunks.
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    stale = set(collection.get(include=[])["ids"]) - set(ids)
    if stale:
        collection.delete(ids=list(stale))

    print(f"ChromaDB '{collection_name}': {len(ids)} chunks upserted, {len(stale)} stale removed, {collection.count()} total")
    return collection
