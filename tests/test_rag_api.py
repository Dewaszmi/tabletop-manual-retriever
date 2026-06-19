from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from tabletop_manual_retriever.ingest.service import RetrievedChunk
from tabletop_manual_retriever.main import app
from tabletop_manual_retriever.rag.llm import (
    ConversationMessage,
    LlmError,
    OpenAICompatibleAnswerGenerator,
)
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


class FakeAnswerGenerator:
    def generate(
        self,
        *,
        question: str,
        game_slug: str,
        sources: Sequence[RetrievedChunk],
        history: Sequence[ConversationMessage] = (),
    ) -> str:
        return "On your turn, roll the dice and move."


@pytest.fixture
def rag_client() -> TestClient:
    service = RagService(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        answer_generator=None,
    )
    app.dependency_overrides[get_rag_service] = lambda: service
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def rag_llm_client() -> TestClient:
    service = RagService(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        answer_generator=FakeAnswerGenerator(),
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
    assert payload["sources"][0]["excerpt"] == "Roll the dice on your turn."
    assert payload["sources"][0]["page_number"] == 2
    assert "Roll the dice on your turn." in payload["answer"]
    assert payload["answer_mode"] == "excerpt"


def test_query_endpoint_uses_llm_answer(rag_llm_client: TestClient) -> None:
    response = rag_llm_client.post(
        "/query",
        json={"game_slug": "catan", "question": "What happens on my turn?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "On your turn, roll the dice and move."
    assert payload["answer_mode"] == "llm"


def test_query_endpoint_rejects_empty_question(rag_client: TestClient) -> None:
    response = rag_client.post(
        "/query",
        json={"game_slug": "catan", "question": "   "},
    )

    assert response.status_code == 400
    assert "Question is required" in response.json()["detail"]


def test_openai_compatible_answer_generator_calls_chat_completions() -> None:
    sources = [
        RetrievedChunk(
            text="Reach 10 victory points to win.",
            page_number=3,
            chunk_index=0,
            filename="rules.pdf",
            score=0.91,
        )
    ]
    generator = OpenAICompatibleAnswerGenerator(
        base_url="http://llm.test/v1",
        model="test-model",
        api_key="secret",
    )
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "You win at 10 points."}}]
    }

    with patch("httpx.post", return_value=mock_response) as post:
        answer = generator.generate(
            question="How do I win?",
            game_slug="catan",
            sources=sources,
        )

    assert answer == "You win at 10 points."
    post.assert_called_once()
    call_kwargs = post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "test-model"
    assert call_kwargs["headers"]["Authorization"] == "Bearer secret"


def test_openai_compatible_answer_generator_raises_llm_error_on_http_failure() -> None:
    generator = OpenAICompatibleAnswerGenerator(
        base_url="http://llm.test/v1",
        model="test-model",
    )
    request = httpx.Request("POST", "http://llm.test/v1/chat/completions")
    response = httpx.Response(status_code=500, request=request)

    with patch(
        "httpx.post",
        side_effect=httpx.HTTPStatusError(
            "server error",
            request=request,
            response=response,
        ),
    ):
        with pytest.raises(LlmError, match="LLM request failed \\(500\\)"):
            generator.generate(
                question="How do I win?",
                game_slug="catan",
                sources=[
                    RetrievedChunk(
                        text="Win at 10 points.",
                        page_number=1,
                        chunk_index=0,
                        filename="rules.pdf",
                        score=0.9,
                    )
                ],
            )
