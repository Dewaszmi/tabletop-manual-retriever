from collections.abc import Sequence

import pytest

from tabletop_manual_retriever.ingest.service import RetrievedChunk
from tabletop_manual_retriever.rag.service import RagService


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


class FakeVectorStore:
    def upsert_chunks(self, **kwargs) -> int:
        raise NotImplementedError

    def search_chunks(
        self,
        *,
        collection_name: str,
        vector: Sequence[float],
        game_slug: str,
        filename: str | None = None,
        limit: int,
    ) -> list[RetrievedChunk]:
        assert collection_name == "manual_chunks"
        assert game_slug == "catan"
        assert filename is None
        assert vector == [13.0]
        assert limit == 2
        return [
            RetrievedChunk(
                text="Reach 10 victory points to win.",
                page_number=3,
                chunk_index=0,
                filename="rules.pdf",
                score=0.91,
            ),
            RetrievedChunk(
                text="Settlements are worth 1 point each.",
                page_number=4,
                chunk_index=1,
                filename="rules.pdf",
                score=0.82,
            ),
        ]


def test_rag_service_returns_sources_and_answer() -> None:
    service = RagService(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        top_k=2,
    )

    result = service.query(
        game_slug="catan",
        question="How do I win?",
    )

    assert result.game_slug == "catan"
    assert result.question == "How do I win?"
    assert len(result.sources) == 2
    assert result.sources[0].page_number == 3
    assert "Reach 10 victory points" in result.context
    assert "From rules.pdf (page 3)" in result.answer


def test_rag_service_rejects_empty_question() -> None:
    service = RagService(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError, match="Question is required"):
        service.query(game_slug="catan", question="   ")


def test_rag_service_filters_by_filename() -> None:
    class FilenameVectorStore(FakeVectorStore):
        def search_chunks(self, **kwargs) -> list[RetrievedChunk]:
            assert kwargs["filename"] == "expansion.pdf"
            return []

    service = RagService(
        embedder=FakeEmbedder(),
        vector_store=FilenameVectorStore(),
    )

    result = service.query(
        game_slug="catan",
        question="How do I win?",
        filename="expansion.pdf",
    )

    assert result.sources == ()
    assert "No relevant manual passages" in result.answer
