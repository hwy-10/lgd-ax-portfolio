"""Extract the datasheet, create chunks, embed them, and persist ChromaDB."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    COLLECTION_NAME,
    DOCUMENT_METADATA,
    PDF_PATH,
    VECTOR_DB_DIR,
)
from src.chunker import Chunk, PageText, create_chunks


def extract_pages(pdf_path: Path) -> list[PageText]:
    """Extract text from each PDF page with 1-based page numbers."""

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Place CH334_CH335_V2.5.pdf in the data directory."
        )
    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError(f"Password-protected PDF is not supported: {pdf_path}")
        return [
            PageText(page=index + 1, text=page.get_text("text"))
            for index, page in enumerate(document)
        ]


def _chunk_id(chunk: Chunk, index: int) -> str:
    identity = f"{chunk.metadata['source']}:{chunk.metadata['page']}:{index}:{chunk.text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def persist_chunks(chunks: list[Chunk], embedder) -> int:
    """Replace the configured collection with the newly embedded chunks."""

    # 선택 경로라 여기서만 가져온다. 기본 검색(BM25)은 chromadb 없이 돈다.
    import chromadb
    from chromadb.errors import NotFoundError

    if not chunks:
        raise ValueError("No text chunks were extracted from the PDF")
    embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except NotFoundError:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[_chunk_id(chunk, index) for index, chunk in enumerate(chunks)],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=embeddings,
    )
    return len(chunks)


def ingest(pdf_path: Path = PDF_PATH) -> int:
    pages = extract_pages(pdf_path)
    metadata = {**DOCUMENT_METADATA, "source": pdf_path.name}
    chunks = create_chunks(
        pages, metadata, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
    )
    from src.embeddings import OpenAIEmbedder

    return persist_chunks(chunks, OpenAIEmbedder())


def main() -> None:
    try:
        count = ingest()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Ingestion complete: {count} chunks stored in '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
