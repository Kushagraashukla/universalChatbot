"""Offline checks for the keyword-search helpers (no Ollama, no DB).

Run: .venv\\Scripts\\python.exe tests\\test_retrieval_text.py   (or with pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.lexical import LexicalIndex, tokenize  # noqa: E402
from src.retrieval.retriever import _fuse, search_text  # noqa: E402


def test_search_text_drops_only_filler_words():
    assert search_text("What API endpoint creates a student?") == "What creates a student?"
    assert search_text("How may an admin add students?") == "How may an admin add students?"
    assert search_text("API endpoints?") == "API endpoints?"  # never search for an empty string


def test_tokenize_matches_word_forms():
    assert tokenize("creates") == tokenize("create") == ["create"]
    assert tokenize("Students classes entries") == ["student", "class", "entry"]
    assert tokenize("How do I add a class?") == ["add", "class"]  # stopwords dropped, "class" kept whole
    assert tokenize("POST /api/admin/create-student") == ["post", "api", "admin", "create", "student"]


def test_bm25_prefers_rare_shared_words_and_respects_exclusions():
    index = LexicalIndex(
        ids=["a#0", "b#0", "c#0"],
        documents=["Admin creates a student", "Teacher assigns a course", "Admin creates a student (design note)"],
        metadatas=[{"source": "a.md"}, {"source": "b.md"}, {"source": "design.md"}],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    )
    assert index.search("create student", 5) == ["a#0", "c#0"]
    assert index.search("create student", 5, frozenset({"design.md"})) == ["a#0"]
    assert index.search("unrelated words", 5) == []
    assert abs(index.cosine_distance("a#0", [1.0, 0.0])) < 1e-6


def test_fuse_rewards_agreement():
    # "x" is 2nd in both lists, so it beats items that top only one list.
    assert _fuse(["a", "x", "b"], ["c", "x", "d"])[0] == "x"


if __name__ == "__main__":
    test_tokenize_matches_word_forms()
    test_bm25_prefers_rare_shared_words_and_respects_exclusions()
    test_fuse_rewards_agreement()
    test_search_text_drops_only_filler_words()
    print("retrieval text tests passed")
