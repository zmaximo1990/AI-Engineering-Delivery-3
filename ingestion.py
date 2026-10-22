"""Load files from /data, chunk them, and persist them in Chroma."""

import hashlib
import logging
from pathlib import Path

from constants import DATA_DIR
from processing import DocumentProcessor
from vectorstore import VectorStoreManager, get_vector_store

logger = logging.getLogger(__name__)

_SOURCE_SUFFIXES = {".md", ".txt"}


def list_source_files(data_dir: Path | None = None) -> list[Path]:
    """Return the .md and .txt files in the data directory, sorted by name."""
    directory = data_dir or DATA_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _SOURCE_SUFFIXES
    )


def _chunk_id(source: str, index: int, text: str) -> str:
    """
    Generate a unique identifier for a chunk of text using the source, index, and text.

    Args:
        source: The name of the source file.
        index: The index of the chunk.
        text: The text of the chunk.

    Returns:
        A unique identifier for the chunk: hash of the source, index, and text.
    """

    payload = f"{source}\n{index}\n{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def ingest_if_needed(
    data_dir: Path | None = None,
) -> int:
    """Chunk and persist the corpus when the collection is still empty.

    A non-empty collection is treated as an existing index, so the files
    are not embedded again. Opening the client creates the folder even
    when it has no documents, so the document count is the signal, not
    the folder itself.

    Returns the number of chunks written on this call.
    """

    manager: VectorStoreManager = get_vector_store()

    existing = manager.count()
    if existing > 0:
        logger.info(
            "Vector store already contains %d chunks; skipping ingestion. "
            "Delete the vectorstore directory to reindex.",
            existing,
        )
        return 0

    files = list_source_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .md or .txt files found in {data_dir or DATA_DIR}")

    processor = DocumentProcessor()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for path in files:
        raw_text = path.read_text(encoding="utf-8")
        chunks = processor.process_document(raw_text)
        logger.info("%s -> %d chunks", path.name, len(chunks))
        for index, chunk in enumerate(chunks):
            ids.append(_chunk_id(path.name, index, chunk))
            documents.append(chunk)
            # TODO: add more tags from business requirements.
            metadatas.append({"source": path.name, "chunk_index": index})

    if not ids:
        raise ValueError("The source files did not produce any chunks")

    manager.upsert_documents(ids, documents, metadatas)
    logger.info("Indexed %d chunks into the vector store", len(ids))
    return len(ids)
