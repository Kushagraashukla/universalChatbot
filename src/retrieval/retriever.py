"""Find the knowledge-base chunks most relevant to a question.

Hybrid search: vector search (meaning) and BM25 keyword search (exact words) each
rank candidates, and Reciprocal Rank Fusion merges the two lists. Only the question
is embedded here; document vectors are read from ChromaDB.
"""

from src.errors import VectorStoreError
from src.ingestion.embedder import embed_query
from src.ingestion.vector_store import COLLECTION_NAME, open_collection
from src.retrieval.lexical import get_index

TOP_K = 4
CANDIDATES = 20  # per ranking, before fusion
RRF_K = 60  # standard RRF constant; results were the same for 10, 20 and 60 on our probe set


# Words that say *how* to answer, not *what* about. Left in, "What API endpoint creates a student?" matches every
# chunk of the API reference and misses "Admin creates a student" (fused rank 15 -> 1 without them).
SEARCH_FILLER = frozenset({"api", "apis", "endpoint", "endpoints", "route", "routes", "url", "urls", "http", "method"})


def search_text(question: str) -> str:
    """The question without SEARCH_FILLER words; used for searching only (the model still sees the full question)."""
    kept = [w for w in question.split() if w.strip("?.,!:;\"'()`").lower() not in SEARCH_FILLER]
    return " ".join(kept) or question


def _fuse(*rankings: list[str]) -> list[str]:
    """Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank) over the rankings that contain the chunk."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)


def retrieve(
    question: str,
    top_k: int = TOP_K,
    collection_name: str = COLLECTION_NAME,
    max_distance: float | None = None,
    exclude_sources: tuple[str, ...] | list[str] = (),
) -> list[dict]:
    """Return the `top_k` best chunks (best first), each followed by its same-section neighbours.

    `distance` is the cosine distance between question and chunk (lower = closer; 1 - cosine
    similarity, checked in src/evaluation/run_eval.py). It is not a percentage. Hits are ordered by
    the fused rank, so distance does not have to increase down the list.
    `max_distance` (optional) drops hits farther than that value; only allowed for cosine.
    `exclude_sources` lists knowledge-base files never to search (e.g. design notes about the chatbot).
    """
    question = question.strip()
    if not question:
        raise ValueError("Question is empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    collection = open_collection(collection_name)
    total = collection.count()
    if total == 0:
        raise VectorStoreError(f"Collection '{collection_name}' is empty. Run ingestion first.")
    metric = (collection.configuration.get("hnsw") or {}).get("space", "unknown")  # read from the DB, not assumed
    if max_distance is not None and metric != "cosine":
        raise ValueError(f"max_distance is only supported for the cosine metric, not '{metric}'.")

    excluded = frozenset(exclude_sources)
    query = search_text(question)
    query_vector = embed_query(query)
    dense = collection.query(
        query_embeddings=[query_vector],
        n_results=min(max(CANDIDATES, top_k), total),
        where={"source": {"$nin": sorted(excluded)}} if excluded else None,
        include=["distances"],
    )
    dense_distance = dict(zip(dense["ids"][0], dense["distances"][0]))
    index = get_index(collection)
    lexical_ids = index.search(query, max(CANDIDATES, top_k), excluded)

    chunks = []
    for chunk_id in _fuse(dense["ids"][0], lexical_ids)[:top_k]:
        text, meta = index.chunk(chunk_id)
        if chunk_id in dense_distance:
            distance = dense_distance[chunk_id]
        else:  # found by keywords only: measure it the same way Chroma would
            distance = index.cosine_distance(chunk_id, query_vector) if metric == "cosine" else None
        chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "source": meta["source"],
            "title": meta["title"],
            "section": meta["section"],
            "distance": None if distance is None else round(distance, 4),
            "metric": metric,
        })
    if max_distance is not None:
        chunks = [c for c in chunks if c["distance"] <= max_distance]
    return _add_neighbours(collection, chunks, metric)


def _add_neighbours(collection, hits: list[dict], metric: str) -> list[dict]:
    """Around each hit, add the chunk just before/after it if it is in the same section.

    Long tables and lists get split across chunks (e.g. a roles table whose last row
    lands in the next chunk); this keeps them whole and in document order. Neighbours
    were not matched by the search, so their distance is None. Hits keep their order, best first.
    """
    seen = {h["chunk_id"] for h in hits}
    wanted = {}  # hit chunk_id -> neighbour ids
    for h in hits:
        source, index = h["chunk_id"].rsplit("#", 1)
        ids = [f"{source}#{j}" for j in (int(index) - 1, int(index) + 1) if j >= 0 and f"{source}#{j}" not in seen]
        seen.update(ids)
        wanted[h["chunk_id"]] = ids

    all_ids = [i for ids in wanted.values() for i in ids]
    if not all_ids:
        return hits
    found = collection.get(ids=all_ids, include=["documents", "metadatas"])
    by_id = dict(zip(found["ids"], zip(found["documents"], found["metadatas"])))

    def neighbour(nid: str) -> list[dict]:
        if nid not in by_id or by_id[nid][1]["section"] != h["section"]:
            return []
        text, meta = by_id[nid]
        return [{"chunk_id": nid, "text": text, "source": meta["source"], "title": meta["title"],
                 "section": meta["section"], "distance": None, "metric": metric}]

    result = []
    for h in hits:
        index = int(h["chunk_id"].rsplit("#", 1)[1])
        before = [n for nid in wanted[h["chunk_id"]] if int(nid.rsplit("#", 1)[1]) < index for n in neighbour(nid)]
        after = [n for nid in wanted[h["chunk_id"]] if int(nid.rsplit("#", 1)[1]) > index for n in neighbour(nid)]
        # Document order, so a table's header (often in the previous chunk) comes before its rows.
        result += before + [h] + after
    return result
