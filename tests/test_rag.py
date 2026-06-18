from collections.abc import Sequence

import pytest

from tabletop_manual_retriever.ingest.service import RetrievedChunk
from tabletop_manual_retriever.rag.llm import build_llm_context, build_llm_messages
from tabletop_manual_retriever.rag.service import RagService, build_source_excerpt


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


class FakeAnswerGenerator:
    def generate(
        self,
        *,
        question: str,
        game_slug: str,
        sources: Sequence[RetrievedChunk],
    ) -> str:
        return f"Answer for {game_slug}: {question} ({len(sources)} sources)"


def test_rag_service_returns_sources_and_excerpt_answer() -> None:
    service = RagService(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        answer_generator=None,
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
    assert result.answer_mode == "excerpt"


def test_build_source_excerpt_shortens_long_text_without_cutting_words() -> None:
    excerpt = build_source_excerpt(
        "one two three four five",
        max_chars=15,
    )

    assert excerpt == "one two three..."


def test_rag_service_uses_llm_answer_generator() -> None:
    service = RagService(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        answer_generator=FakeAnswerGenerator(),
    )

    result = service.query(
        game_slug="catan",
        question="How do I win?",
        top_k=2,
    )
    assert result.answer_mode == "llm"


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
    assert result.answer_mode == "none"


def test_build_llm_messages_include_numbered_sources() -> None:
    sources = [
        RetrievedChunk(
            text="Reach 10 victory points to win.",
            page_number=3,
            chunk_index=0,
            filename="rules.pdf",
            score=0.91,
        )
    ]

    messages = build_llm_messages(
        question="How do I win?",
        game_slug="catan",
        sources=sources,
    )

    assert messages[0]["role"] == "system"
    assert "[1] rules.pdf, page 3" in messages[1]["content"]
    assert "How do I win?" in messages[1]["content"]
    assert build_llm_context(sources).startswith("[1] rules.pdf")
