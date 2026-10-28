"""Ingestion writes chunks once, and search uses that same index."""

from pathlib import Path

import pytest

from embeddings import HuggingFaceEmbeddings
from ingestion import ingest_if_needed
from vectorstore import ChromaStoreManager, VectorStoreManager


@pytest.fixture(scope="module")
def embeddings():
    return HuggingFaceEmbeddings()


def _manager(tmp_path: Path, embeddings) -> VectorStoreManager:
    return ChromaStoreManager(
        persist_directory=tmp_path / "vectorstore",
        collection_name="test_policies",
        embeddings=embeddings,
    )


def test_ingest_persists_and_second_call_skips(tmp_path, embeddings, monkeypatch):
    data_dir = Path(__file__).resolve().parents[1] / "data"
    manager = _manager(tmp_path, embeddings)
    monkeypatch.setattr("ingestion.get_vector_store", lambda: manager)

    written = ingest_if_needed(data_dir)
    assert written >= 4
    assert manager.count() == written

    skipped = ingest_if_needed(data_dir)
    assert skipped == 0
    assert manager.count() == written

    hits = manager.semantic_search("días de vacaciones por antigüedad", n_results=3)
    assert hits
    assert hits[0]["metadata"]["source"] == "politica_vacaciones.md"


def test_reopen_reads_the_same_collection(tmp_path, embeddings, monkeypatch):
    data_dir = Path(__file__).resolve().parents[1] / "data"
    first = _manager(tmp_path, embeddings)
    active = {"manager": first}
    monkeypatch.setattr("ingestion.get_vector_store", lambda: active["manager"])
    written = ingest_if_needed(data_dir)

    reopened = _manager(tmp_path, embeddings)
    active["manager"] = reopened
    assert reopened.count() == written
    assert ingest_if_needed(data_dir) == 0
