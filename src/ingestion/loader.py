"""Load knowledge-base files into LangChain `Document` objects.

A Document is just two things: `page_content` (the text) and `metadata`
(a dict describing where the text came from). Every later step - chunking,
embedding, retrieval - works on Documents, so this is the entry point.
"""

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_DIR = PROJECT_ROOT / "ai-knowledge-base"


def _load_markdown(path: Path) -> str:
    # Raw text keeps headings/tables intact; the chunker will use them later.
    return path.read_text(encoding="utf-8")


# To support a new file type, write a function that returns the file's text
# and register its extension here.
LOADERS = {
    ".md": _load_markdown,
}


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def load_documents(kb_dir: str | Path = DEFAULT_KB_DIR) -> list[Document]:
    """Load every supported file under `kb_dir` (including subfolders).

    Unsupported, empty or unreadable files are skipped with a warning so one
    bad file never stops the whole load.
    """
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base folder not found: {kb_dir}")

    documents = []
    for path in sorted(p for p in kb_dir.rglob("*") if p.is_file()):
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            logger.warning("Skipping unsupported file: %s", path.name)
            continue

        try:
            text = loader(path)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Could not read %s: %s", path.name, exc)
            continue

        if not text.strip():
            logger.warning("Skipping empty file: %s", path.name)
            continue

        documents.append(Document(
            page_content=text,
            metadata={
                "source": path.relative_to(kb_dir).as_posix(),
                "file_name": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "title": _first_heading(text) or path.stem,
            },
        ))

    logger.info("Loaded %d documents from %s", len(documents), kb_dir)
    return documents


if __name__ == "__main__":
    # Quick manual check: python -m src.ingestion.loader
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    docs = load_documents()
    for doc in docs:
        print(f"{doc.metadata['source']:<50} {len(doc.page_content):>6} chars  | {doc.metadata['title']}")
