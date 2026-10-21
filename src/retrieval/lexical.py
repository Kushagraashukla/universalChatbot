"""Keyword (BM25) search over a collection's chunks, used next to vector search.

Vector search finds chunks with a similar meaning but can miss exact words: for
"What API endpoint creates a student?" the chunk "Admin creates a student" ranked
34th. BM25 scores the words a question shares with a chunk, weighting rare words
higher. retriever.py merges both rankings.
"""

import math
import re
from collections import Counter

import numpy as np

K1, B = 1.2, 0.75  # standard BM25 constants
STOPWORDS = frozenset(
    "a an the how do does did i my me we you your can could may might what which who whom where when why "
    "is are was were be to of in on for with by from at and or it this that as into using use via".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case words without stopwords, with a light plural/verb -s strip.

    students -> student, creates -> create, classes -> class, entries -> entry; "class" stays "class".
    """
    words = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOPWORDS:
            continue
        if w.endswith("sses"):
            w = w[:-2]
        elif len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    return words


class LexicalIndex:
    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict], embeddings):
        self.ids, self.documents, self.metadatas = ids, documents, metadatas
        self.position = {cid: i for i, cid in enumerate(ids)}
        self.vectors = np.asarray(embeddings, dtype=np.float32)
        tokens = [tokenize(d) for d in documents]
        self.term_freqs = [Counter(t) for t in tokens]
        self.lengths = [len(t) for t in tokens]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        doc_freq = Counter(w for t in tokens for w in set(t))
        n = len(ids)
        self.idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5)) for w, df in doc_freq.items()}

    def search(self, query: str, limit: int, exclude_sources: frozenset[str] = frozenset()) -> list[str]:
        """Chunk ids ordered by BM25 score (only chunks sharing at least one word)."""
        terms = tokenize(query)
        scored = []
        for i, (tf, length) in enumerate(zip(self.term_freqs, self.lengths)):
            if self.metadatas[i]["source"] in exclude_sources:
                continue
            score = sum(
                self.idf[w] * tf[w] * (K1 + 1) / (tf[w] + K1 * (1 - B + B * length / self.avg_length))
                for w in terms if w in tf
            )
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.ids[i] for _, i in scored[:limit]]

    def chunk(self, chunk_id: str) -> tuple[str, dict]:
        i = self.position[chunk_id]
        return self.documents[i], self.metadatas[i]

    def cosine_distance(self, chunk_id: str, query_vector) -> float:
        v = self.vectors[self.position[chunk_id]]
        q = np.asarray(query_vector, dtype=np.float32)
        return float(1 - v @ q / (np.linalg.norm(v) * np.linalg.norm(q)))


_cache: dict[str, tuple[int, LexicalIndex]] = {}


def get_index(collection) -> LexicalIndex:
    """Build once per collection and process; rebuilt when the chunk count changes.

    shortcut: an edit that keeps the same chunk count is not detected; restart the API after ingestion.
    """
    count = collection.count()
    cached = _cache.get(collection.name)
    if cached is None or cached[0] != count:
        data = collection.get(include=["documents", "metadatas", "embeddings"])
        cached = (count, LexicalIndex(data["ids"], data["documents"], data["metadatas"], data["embeddings"]))
        _cache[collection.name] = cached
    return cached[1]
