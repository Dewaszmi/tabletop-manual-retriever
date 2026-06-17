from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from tabletop_manual_retriever.ingest.service import RetrievedChunk
from tabletop_manual_retriever.main import app
from tabletop_manual_retriever.rag.router import get_rag_service
from tabletop_manual_retriever.rag.service import RagService


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class FakeVectorStore:
    def upsert_chunks(self, **kwargs) -> int:
        raise NotImplementedError

    def search_chunks(self, **kwargs) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="Roll the dice on your turn.",
                page_number=2,
                chunk_index=0,
                filename="rules.pdf",
                score=0.88,
            )
        ]


@pytest.fixture
def rag_client() -> TestClient:
    service = RagService(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )
    app.dependency_overrides[get_rag_service] = lambda: service
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_query_endpoint_returns_rag_response(rag_client: TestClient) -> None:
    response = rag_client.post(
        "/query",
        json={"game_slug": "catan", "question": "What happens on my turn?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["game_slug"] == "catan"
    assert payload["question"] == "What happens on my turn?"
    assert payload["sources"][0]["text"] == "Roll the dice on your turn."
    assert payload["sources"][0]["page_number"] == 2
    assert "Roll the dice on your turn." in payload["answer"]


def test_query_endpoint_rejects_empty_question(rag_client: TestClient) -> None:
    response = rag_client.post(
        "/query",
        json={"game_slug": "catan", "question": "   "},
    )

    assert response.status_code == 400
    assert "Question is required" in response.json()["detail"]
