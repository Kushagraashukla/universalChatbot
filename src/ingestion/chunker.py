"""Split loaded Documents into small chunks for embedding.

Two passes:
1. Split each Markdown file at its #, ## and ### headings, so a chunk never
   mixes two unrelated sections.
2. Split any section still longer than CHUNK_SIZE into overlapping pieces.

Every chunk starts with its heading path (e.g. "Fees > Refunds") so even the
second half of a long section still says what it is about.
"""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 120   # characters repeated between neighbouring chunks

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=True)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    for doc in documents:
        sections = size_splitter.split_documents(header_splitter.split_text(doc.page_content))
        for i, section in enumerate(sections):
            headings = [section.metadata[key] for _, key in HEADERS if key in section.metadata]
            section_path = " > ".join(headings)
            text = f"{section_path}\n\n{section.page_content}" if section_path else section.page_content
            chunks.append(Document(
                page_content=text,
                metadata={
                    **doc.metadata,  # source, file_name, file_type, title from the loader
                    "section": section_path,
                    "chunk_index": i,
                    "chunk_id": f"{doc.metadata['source']}#{i}",
                },
            ))
    return chunks
