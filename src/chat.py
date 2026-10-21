"""Answer one question: retrieve -> build prompt -> generate -> attach citations.

Usage: .venv\\Scripts\\python.exe -m src.chat "How do I reset my password?"
Run ingestion first (python -m src.ingestion.pipeline); this never re-embeds documents.
"""

import logging
import re
import sys

from src.generation.llm import generate
from src.generation.prompt import (AUDIENCE_USER, NOT_FOUND_ANSWER, REFUSAL_ANSWER, REWRITE_FOR_USER, TECHNICAL_LEAK,
                                   audience, build_messages, leaks_system_prompt, role_note)
from src.ingestion.vector_store import COLLECTION_NAME
from src.retrieval.retriever import TOP_K, retrieve

logger = logging.getLogger("rag.chat")

SOURCE_FIELDS = ("chunk_id", "source", "title", "section", "distance", "metric")

# ~2.3k tokens of documents; leaves room in the 4096-token window (llm.NUM_CTX) for rules, a
# 2000-char question and the answer. Ollama silently cuts prompts that are too long.
CONTEXT_CHAR_BUDGET = 8000


def _fit_budget(chunks: list[dict]) -> list[dict]:
    """Keep search hits first (best first), then neighbour chunks, until the budget is used; keep original order."""
    priority = sorted(range(len(chunks)), key=lambda i: chunks[i]["distance"] is None)  # stable sort
    keep, used = set(), 0
    for i in priority:
        size = len(chunks[i]["text"])
        if keep and used + size > CONTEXT_CHAR_BUDGET:
            continue
        keep.add(i)
        used += size
    return [c for i, c in enumerate(chunks) if i in keep]


def _citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Map the model's [n] markers back to the real retrieved chunks; ignore numbers that don't exist."""
    used = []
    for n in re.findall(r"\[(\d+)\]", answer):
        i = int(n)
        if 1 <= i <= len(chunks) and i not in used:
            used.append(i)
    return [{"ref": i, **{k: chunks[i - 1][k] for k in SOURCE_FIELDS}} for i in used]


def ask(
    question: str,
    top_k: int = TOP_K,
    collection_name: str = COLLECTION_NAME,
    max_distance: float | None = None,
    instructions: str = "",
    exclude_sources: tuple[str, ...] | list[str] = (),
    roles: dict[str, list[str]] | None = None,
) -> dict:
    """Returns {answer, citations, sources}.

    citations = chunks the answer actually cites (from retrieved metadata, never from the LLM)
    sources   = every chunk that was retrieved and shown to the model
    """
    chunks = retrieve(question, top_k, collection_name, max_distance, exclude_sources)
    # Sections about another role (e.g. an admin-only page when a teacher asks) are dropped: labelling them in the
    # prompt was not enough, the 8B model still credited admin-only actions to teachers. Kept if nothing else is left.
    same_role = [c for c in chunks if not role_note(question, c["section"], roles or {})]
    chunks = _fit_budget(same_role or chunks)
    sources = [{"ref": i, **{k: c[k] for k in SOURCE_FIELDS}} for i, c in enumerate(chunks, start=1)]
    if not chunks:
        return {"answer": NOT_FOUND_ANSWER, "citations": [], "sources": []}
    messages = build_messages(question, chunks, instructions, roles)
    answer = generate(messages)
    if audience(question) == AUDIENCE_USER and TECHNICAL_LEAK.search(answer):
        # The 8B model sometimes copies endpoints or file names from the developer docs; one rewrite pass.
        logger.info("rewriting answer: technical details in an ordinary-user answer")
        answer = generate(messages + [("ai", answer), ("human", REWRITE_FOR_USER)])
    if leaks_system_prompt(answer):
        logger.warning("answer repeated the system prompt; replaced with a refusal")
        return {"answer": REFUSAL_ANSWER, "citations": [], "sources": sources}
    return {"answer": answer, "citations": _citations(answer, chunks), "sources": sources}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.chat "your question"')
    from src.apps import get_app

    app = get_app("default")  # same settings the API uses for the default app
    if app is None:
        raise SystemExit("ERROR: app 'default' is missing or disabled in config/apps.json")
    try:
        result = ask(" ".join(sys.argv[1:]), app.top_k, app.collection, app.max_distance, app.instructions,
                     app.exclude_sources, app.roles)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}")

    print(result["answer"])
    print("\nCitations:")
    for c in result["citations"] or [{"ref": "-", "source": "(none cited)", "section": "", "chunk_id": ""}]:
        print(f"  [{c['ref']}] {c['source']} | {c['section']} | {c['chunk_id']}")
