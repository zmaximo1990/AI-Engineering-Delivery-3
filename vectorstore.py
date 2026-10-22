"""Persistent Chroma collection used for indexing and similarity search."""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models import Collection as ChromaDBCollection
from chromadb.config import Settings
from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from constants import COLLECTION_NAME, EMBEDDING_MODEL, TOP_K, VECTORSTORE_DIR

logger = logging.getLogger(__name__)

# Disk failures, the local SQLite store, and Chroma server or internal errors.
_STORE_ERRORS = (OSError, sqlite3.Error, ChromaError)


class VectorMemoryError(Exception):
    """A vector store read or write failed, or the embedding model does not match."""


class VectorMemoryManager:
    """CRUD wrapper around a local Chroma collection.

    Indexing and search share one embedding model. A different model would
    make distances meaningless.
    """

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL,
        embeddings: Embeddings | None = None,
    ):
        persist_path = Path(persist_directory or VECTORSTORE_DIR)
        if not persist_path.is_absolute():
            persist_path = Path(__file__).resolve().parent / persist_path
        persist_path.mkdir(parents=True, exist_ok=True)

        self.embedding_model_name = embedding_model_name
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=embedding_model_name
        )
        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            collection_metadata={"embedding_model": embedding_model_name},
        )
        self.collection: ChromaDBCollection = self.store._collection
        self._require_matching_embedding_model()
        logger.info(
            "Vector store ready at %s (collection=%s, embedding=%s, documents=%d)",
            persist_path,
            collection_name,
            embedding_model_name,
            self.count(),
        )

    def _require_matching_embedding_model(self) -> None:
        recorded = (self.collection.metadata or {}).get("embedding_model")
        if recorded and recorded != self.embedding_model_name:
            raise VectorMemoryError(
                f"This collection was indexed with '{recorded}', "
                f"but the configured embedding model is '{self.embedding_model_name}'. "
                "Use the same model for indexing and search."
            )

    def count(self) -> int:
        return self.collection.count()

    def upsert_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert documents, or replace them when the id already exists."""
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("ids, documents, and metadatas must have the same length")
        if not ids:
            return
        try:
            vectors = self.embeddings.embed_documents(documents)
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=vectors,
            )
        except _STORE_ERRORS as exc:
            raise VectorMemoryError("Could not write documents to the vector store.") from exc

    def semantic_search(
        self,
        query_text: str,
        n_results: int = TOP_K,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the nearest documents. Lower distance means a closer match."""
        try:
            pairs = self.store.similarity_search_with_score(
                query_text,
                k=n_results,
                filter=where,
            )
        except _STORE_ERRORS as exc:
            raise VectorMemoryError("Could not query the vector store.") from exc
        return [
            {
                "document": document.page_content,
                "metadata": document.metadata,
                "distance": float(distance),
            }
            for document, distance in pairs
        ]

    def delete_by_id(self, ids: list[str]) -> None:
        """Remove documents by id."""
        try:
            self.store.delete(ids=ids)
        except _STORE_ERRORS as exc:
            raise VectorMemoryError("Could not delete documents from the vector store.") from exc

    def as_retriever(self, k: int = TOP_K) -> VectorStoreRetriever:
        """Similarity retriever. ``k`` is clamped to the spec range of 3 to 5."""
        top_k = min(5, max(3, k))
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )


_manager: VectorMemoryManager | None = None


def get_vector_memory() -> VectorMemoryManager:
    """Return the process-wide store so the embedding model is loaded once."""
    global _manager
    if _manager is None:
        _manager = VectorMemoryManager()
    return _manager
