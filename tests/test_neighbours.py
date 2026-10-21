"""Offline check for retriever._add_neighbours (no Ollama, no real DB).

Run: .venv\\Scripts\\python.exe tests\\test_neighbours.py   (or with pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.retriever import _add_neighbours  # noqa: E402

STORE = {  # chunk_id -> (text, section)
    "a.md#0": ("intro", "A"),
    "a.md#1": ("table header", "A > Roles"),
    "a.md#2": ("table last row", "A > Roles"),
    "a.md#3": ("other section", "A > Fees"),
}


class FakeCollection:
    def get(self, ids, include):
        ids = [i for i in ids if i in STORE]
        return {"ids": ids, "documents": [STORE[i][0] for i in ids],
                "metadatas": [{"source": "a.md", "title": "A", "section": STORE[i][1]} for i in ids]}


def hit(chunk_id, distance):
    return {"chunk_id": chunk_id, "text": STORE[chunk_id][0], "source": "a.md", "title": "A",
            "section": STORE[chunk_id][1], "distance": distance, "metric": "cosine"}


def test_adds_same_section_neighbours_only():
    out = _add_neighbours(FakeCollection(), [hit("a.md#1", 0.3)], "cosine")
    # #2 is the same section -> added after the hit; #0 is another section -> skipped
    assert [c["chunk_id"] for c in out] == ["a.md#1", "a.md#2"]
    assert out[0]["distance"] == 0.3 and out[1]["distance"] is None


def test_previous_neighbour_comes_before_the_hit():
    out = _add_neighbours(FakeCollection(), [hit("a.md#2", 0.3)], "cosine")
    assert [c["chunk_id"] for c in out] == ["a.md#1", "a.md#2"]  # table header chunk first


def test_no_duplicates_and_best_hit_stays_first():
    out = _add_neighbours(FakeCollection(), [hit("a.md#2", 0.2), hit("a.md#1", 0.4)], "cosine")
    assert [c["chunk_id"] for c in out] == ["a.md#2", "a.md#1"]


def test_budget_keeps_hits_before_neighbours():
    from src.chat import CONTEXT_CHAR_BUDGET, _fit_budget

    big = "x" * (CONTEXT_CHAR_BUDGET // 2)
    chunks = [
        {"chunk_id": "h1", "text": big, "distance": 0.2},
        {"chunk_id": "n1", "text": big, "distance": None},  # neighbour of h1
        {"chunk_id": "h2", "text": big, "distance": 0.3},
    ]
    # Room for two: both hits win over the neighbour; original order kept.
    assert [c["chunk_id"] for c in _fit_budget(chunks)] == ["h1", "h2"]
    # A single oversized chunk is still kept (never send an empty context).
    assert len(_fit_budget([{"chunk_id": "h", "text": "y" * (CONTEXT_CHAR_BUDGET * 2), "distance": 0.1}])) == 1


def test_reply_language():
    from src.generation.prompt import reply_language

    assert reply_language("How does a user reset a forgotten password?") == "English"
    assert reply_language("password kaise reset kare?").startswith("Hinglish")
    assert reply_language("ye app kya karta hai").startswith("Hinglish")
    assert reply_language("पासवर्ड कैसे रीसेट करें?").startswith("Hindi")
    assert reply_language("Is the login API rate limited?") == "English"  # one-off words like "is" don't count


if __name__ == "__main__":
    test_reply_language()
    test_adds_same_section_neighbours_only()
    test_previous_neighbour_comes_before_the_hit()
    test_no_duplicates_and_best_hit_stays_first()
    test_budget_keeps_hits_before_neighbours()
    print("neighbour tests passed")
