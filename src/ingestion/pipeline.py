"""Run ingestion end to end: load -> chunk -> embed -> store in ChromaDB.

Usage: .venv\\Scripts\\python.exe -m src.ingestion.pipeline [app_id]
app_id comes from config/apps.json (default: "default").
"""

import logging
import sys

from src.apps import load_apps
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_chunks
from src.ingestion.loader import load_documents
from src.ingestion.vector_store import index_chunks


def run(app_id: str = "default"):
    app = load_apps().get(app_id)
    if app is None:
        raise ValueError(f"Unknown app_id '{app_id}'. Add it to config/apps.json first.")
    docs = load_documents(app.kb_path)
    chunks = chunk_documents(docs)
    vectors = embed_chunks(chunks)
    print(f"{len(docs)} documents -> {len(chunks)} chunks -> {len(vectors)} vectors of size {len(vectors[0])}")
    index_chunks(chunks, vectors, app.collection)
    return chunks, vectors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run(sys.argv[1] if len(sys.argv) > 1 else "default")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}")
