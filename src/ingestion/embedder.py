"""Turn text into vectors (lists of numbers) with a local Ollama model.

`get_embeddings()` is the single place the embedding model is chosen. Use it
for document chunks now and for user questions later: vectors from two
different models cannot be compared.
"""

import hashlib
import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from ollama import ResponseError

from src.errors import OllamaError

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / ".cache" / "embeddings.json"


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def call_ollama(fn, *args, model: str = EMBEDDING_MODEL):
    """Call Ollama and turn its low-level errors into clear messages."""
    try:
        return fn(*args)
    except ConnectionError as exc:
        raise OllamaError("Cannot reach Ollama. Start the Ollama app (or run `ollama serve`) and try again.") from exc
    except ResponseError as exc:
        if exc.status_code == 404:
            raise OllamaError(f"Model '{model}' is not installed. Run: ollama pull {model}") from exc
        raise OllamaError(f"Ollama request to '{model}' failed: {exc.error}") from exc


# Qwen3-Embedding is instruction-aware: queries get a task instruction, documents are embedded as-is (as in the
# index). Measured on 12 probe questions (hybrid search): right section in the top 3 for 12/12, vs 11/12 without
# it ("What API endpoint creates a student?" moved from rank 15 to 1-2).
QUERY_INSTRUCTION = "Instruct: Given a question, retrieve documentation passages that answer it\nQuery: "


def embed_query(question: str) -> list[float]:
    return call_ollama(get_embeddings().embed_query, QUERY_INSTRUCTION + question)


def _cache_key(text: str) -> str:
    # The model name is part of the key, so switching models never reuses old vectors.
    return hashlib.sha256(f"{EMBEDDING_MODEL}\n{text}".encode("utf-8")).hexdigest()


def embed_chunks(chunks: list[Document], cache_path: Path = CACHE_PATH) -> list[list[float]]:
    """Return one vector per chunk, only calling Ollama for text not seen before.

    shortcut: JSON cache grows forever (old chunks are never removed); ChromaDB replaces it next phase.
    """
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    keys = [_cache_key(c.page_content) for c in chunks]

    new = {k: c.page_content for k, c in zip(keys, chunks) if k not in cache}
    if new:
        vectors = call_ollama(get_embeddings().embed_documents, list(new.values()))
        cache.update(zip(new.keys(), vectors))
        cache_path.parent.mkdir(exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    print(f"Embeddings: {len(new)} new, {len(chunks) - len(new)} reused from cache")
    return [cache[k] for k in keys]
