from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tabletop_manual_retriever.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    QDRANT_COLLECTION,
    RAG_TOP_K,
)
from tabletop_manual_retriever.ingest.service import (
    FastEmbedTextEmbedder,
    IngestDependencyError,
    IngestStoreError,
    QdrantVectorStore,
    RetrievedChunk,
    TextEmbedder,
    VectorStore,
)
from tabletop_manual_retriever.rag.llm import (
    AnswerGenerator,
    LlmError,
    OpenAICompatibleAnswerGenerator,
)
from tabletop_manual_retriever.storage.manuals import sanitize_pdf_filename, validate_game_slug


class RagDependencyError(IngestDependencyError):
    """Raised when optional RAG dependencies are not installed."""


class RagStoreError(IngestStoreError):
    """Raised when chunks cannot be read from the vector store."""


@dataclass(frozen=True)
class RagResult:
    game_slug: str
    question: str
    collection_name: str
    sources: tuple[RetrievedChunk, ...]
    context: str
    answer: str
    answer_mode: str


@dataclass
class RagService:
    embedder: TextEmbedder | None = None
    vector_store: VectorStore | None = None
    answer_generator: AnswerGenerator | None = None
    collection_name: str = QDRANT_COLLECTION
    top_k: int = RAG_TOP_K

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = FastEmbedTextEmbedder()
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore()
        if self.answer_generator is None:
            self.answer_generator = _default_answer_generator()

    def query(
        self,
        *,
        game_slug: str,
        question: str,
        filename: str | None = None,
        top_k: int | None = None,
    ) -> RagResult:
        slug = validate_game_slug(game_slug)
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question is required")

        manual_filename = (
            sanitize_pdf_filename(filename) if filename is not None else None
        )
        limit = top_k if top_k is not None else self.top_k

        assert self.embedder is not None
        assert self.vector_store is not None

        query_vector = self.embedder.embed([normalized_question])[0]
        sources = tuple(
            self.vector_store.search_chunks(
                collection_name=self.collection_name,
                vector=query_vector,
                game_slug=slug,
                filename=manual_filename,
                limit=limit,
            )
        )
        context = _build_context(sources)
        answer, answer_mode = _build_answer(
            question=normalized_question,
            game_slug=slug,
            sources=sources,
            answer_generator=self.answer_generator,
        )

        return RagResult(
            game_slug=slug,
            question=normalized_question,
            collection_name=self.collection_name,
            sources=sources,
            context=context,
            answer=answer,
            answer_mode=answer_mode,
        )


def _default_answer_generator() -> AnswerGenerator | None:
    if not LLM_BASE_URL or not LLM_MODEL:
        return None
    return OpenAICompatibleAnswerGenerator(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
    )


def _build_context(sources: Sequence[RetrievedChunk]) -> str:
    if not sources:
        return ""
    return "\n\n---\n\n".join(source.text for source in sources)


def _build_answer(
    *,
    question: str,
    game_slug: str,
    sources: Sequence[RetrievedChunk],
    answer_generator: AnswerGenerator | None,
) -> tuple[str, str]:
    if not sources:
        return "No relevant manual passages were found for that question.", "none"

    if answer_generator is not None:
        answer = answer_generator.generate(
            question=question,
            game_slug=game_slug,
            sources=sources,
        )
        return answer, "llm"

    parts = [
        (
            f"From {source.filename} (page {source.page_number}):\n"
            f"{source.text.strip()}"
        )
        for source in sources
    ]
    return "\n\n".join(parts), "excerpt"
