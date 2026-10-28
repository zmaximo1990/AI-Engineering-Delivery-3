"""Chain propagation, source assembly, and structured-output retries."""

import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from chain import (
    _reject_incomplete,
    assemble_response,
    compose_rag_chain,
    format_documents,
    get_rag_response,
)
from constants import RETRY_ATTEMPTS
from schemas import LLMAnswer, LLMConfig, Provider, RAGResponse


class StubRetriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return self.documents


class _StubStore:
    """Stand-in for the store that ``compose_rag_chain`` opens itself."""

    def __init__(self, retriever: StubRetriever):
        self._retriever = retriever

    def as_retriever(self) -> StubRetriever:
        return self._retriever


def _result(*, finish_reason="stop", parsed=None, parsing_error=None):
    return {
        "raw": AIMessage(
            content="",
            response_metadata={"finish_reason": finish_reason},
        ),
        "parsed": parsed,
        "parsing_error": parsing_error,
    }


def _answer_from_context(payload: dict) -> LLMAnswer:
    context = payload["context"]
    assert "[Source:" in context
    if "14 días" in context:
        return LLMAnswer(answer="14 días corridos por año calendario.")
    return LLMAnswer(
        answer="I do not have access to that information in the available documents."
    )


def test_format_documents_labels_each_source():
    documents = [
        Document(page_content="alpha", metadata={"source": "a.md"}),
        Document(page_content="beta", metadata={}),
    ]
    text = format_documents(documents)
    assert "[Source: a.md]\nalpha" in text
    assert "[Source: unknown]\nbeta" in text


def test_assemble_response_reads_sources_from_metadata():
    payload = {
        "documents": [
            Document(page_content="one", metadata={"source": "b.md"}),
            Document(page_content="two", metadata={"source": "a.md"}),
            Document(page_content="three", metadata={"source": "a.md"}),
        ],
        "answer": LLMAnswer(answer="from the documents"),
    }
    result = assemble_response(payload)
    assert result == RAGResponse(
        answer="from the documents",
        sources=["a.md", "b.md"],
        retrieved_fragments=3,
    )


def test_chain_adds_context_and_assembles_sources(monkeypatch):
    documents = [
        Document(
            page_content="Los empleados con menos de 5 años acceden a 14 días corridos.",
            metadata={"source": "politica_vacaciones.md"},
        )
    ]
    retriever = StubRetriever(documents)
    monkeypatch.setattr("chain.get_vector_store", lambda: _StubStore(retriever))
    chain = compose_rag_chain(RunnableLambda(_answer_from_context))

    result = asyncio.run(chain.ainvoke({"question": "¿Cuántos días de vacaciones?"}))

    assert retriever.queries == ["¿Cuántos días de vacaciones?"]
    assert result.answer == "14 días corridos por año calendario."
    assert result.sources == ["politica_vacaciones.md"]
    assert result.retrieved_fragments == 1


def test_chain_keeps_retrieved_sources_when_the_model_refuses(monkeypatch):
    documents = [
        Document(
            page_content="Las contraseñas deben tener un mínimo de 12 caracteres.",
            metadata={"source": "politica_seguridad_informatica.md"},
        )
    ]
    monkeypatch.setattr(
        "chain.get_vector_store",
        lambda: _StubStore(StubRetriever(documents)),
    )
    chain = compose_rag_chain(RunnableLambda(_answer_from_context))

    result = asyncio.run(chain.ainvoke({"question": "¿Cuál es el salario del CEO?"}))

    assert "do not have access" in result.answer
    assert result.sources == ["politica_seguridad_informatica.md"]
    assert result.retrieved_fragments == 1


def test_valid_answer_is_accepted():
    parsed = LLMAnswer(answer="14 días corridos.")
    assert _reject_incomplete(_result(parsed=parsed)) == parsed


def test_empty_answer_is_rejected():
    empty = LLMAnswer.model_construct(answer="  ")
    with pytest.raises(OutputParserException, match="answer is empty"):
        _reject_incomplete(_result(parsed=empty))


def test_invalid_payload_fails_validation():
    with pytest.raises(ValidationError):
        _reject_incomplete(_result(parsed={"answer": ""}))


def test_truncated_output_is_rejected():
    with pytest.raises(OutputParserException, match="finish_reason=max_tokens"):
        _reject_incomplete(
            _result(
                finish_reason="MAX_TOKENS",
                parsed=LLMAnswer(answer="partial"),
            )
        )


def test_truncation_is_retried_then_accepted():
    attempts = {"n": 0}
    accepted = LLMAnswer(answer="14 días corridos.")

    def next_result(_: dict) -> dict:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _result(finish_reason="length")
        return _result(parsed=accepted)

    chain = (
        RunnableLambda(next_result) | RunnableLambda(_reject_incomplete)
    ).with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError),
        stop_after_attempt=RETRY_ATTEMPTS,
    )
    result = asyncio.run(chain.ainvoke({"question": "vacaciones"}))
    assert result == accepted
    assert attempts["n"] == 2


def test_get_rag_response_rejects_blank_input():
    with pytest.raises(ValueError, match="query must not be empty"):
        asyncio.run(get_rag_response("   ", LLMConfig(provider=Provider.OPENAI)))
