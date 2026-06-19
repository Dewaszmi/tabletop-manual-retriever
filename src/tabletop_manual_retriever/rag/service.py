from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from tabletop_manual_retriever.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    QDRANT_COLLECTION,
    RAG_CANDIDATE_K,
    RAG_RERANK_ENABLED,
    RAG_RERANK_MODEL,
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
    ConversationMessage,
    LlmError,
    OpenAICompatibleAnswerGenerator,
    build_conversation_context,
)
from tabletop_manual_retriever.storage.manuals import sanitize_pdf_filename, validate_game_slug

SOURCE_EXCERPT_MAX_CHARS = 280
MAX_HISTORY_MESSAGES = 12


class RagDependencyError(IngestDependencyError):
    """Raised when optional RAG dependencies are not installed."""


class RagStoreError(IngestStoreError):
    """Raised when chunks cannot be read from the vector store."""


class ChunkReranker(Protocol):
    def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Return chunks ordered by cross-encoder relevance."""


@dataclass(frozen=True)
class RagResult:
    game_slug: str
    question: str
    collection_name: str
    sources: tuple[RetrievedChunk, ...]
    context: str
    answer: str
    answer_mode: str
    history: tuple[ConversationMessage, ...]


@dataclass
class RagService:
    embedder: TextEmbedder | None = None
    vector_store: VectorStore | None = None
    reranker: ChunkReranker | None = None
    answer_generator: AnswerGenerator | None = None
    collection_name: str = QDRANT_COLLECTION
    top_k: int = RAG_TOP_K
    candidate_k: int = RAG_CANDIDATE_K
    rerank_enabled: bool = RAG_RERANK_ENABLED

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = FastEmbedTextEmbedder()
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore()
        if self.reranker is None and self.rerank_enabled:
            self.reranker = SentenceTransformersCrossEncoderReranker()
        if self.answer_generator is None:
            self.answer_generator = _default_answer_generator()

    def query(
        self,
        *,
        game_slug: str,
        question: str,
        filename: str | None = None,
        top_k: int | None = None,
        history: Sequence[ConversationMessage] | None = None,
    ) -> RagResult:
        slug = validate_game_slug(game_slug)
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question is required")

        manual_filename = (
            sanitize_pdf_filename(filename) if filename is not None else None
        )
        limit = top_k if top_k is not None else self.top_k
        candidate_limit = _candidate_limit(
            final_limit=limit,
            candidate_limit=self.candidate_k,
            reranker=self.reranker,
        )
        conversation_history = _normalize_history(history or ())

        assert self.embedder is not None
        assert self.vector_store is not None

        retrieval_query = _build_retrieval_query(
            question=normalized_question,
            history=conversation_history,
        )
        query_vector = self.embedder.embed([retrieval_query])[0]
        sources = tuple(
            self.vector_store.search_chunks(
                collection_name=self.collection_name,
                vector=query_vector,
                game_slug=slug,
                filename=manual_filename,
                limit=candidate_limit,
            )
        )
        if self.reranker is not None:
            sources = tuple(
                self.reranker.rerank(
                    query=retrieval_query,
                    chunks=sources,
                    limit=limit,
                )
            )
        context = _build_context(sources, history=conversation_history)
        answer, answer_mode = _build_answer(
            question=normalized_question,
            game_slug=slug,
            sources=sources,
            answer_generator=self.answer_generator,
            history=conversation_history,
        )
        updated_history = _append_turn(
            conversation_history,
            question=normalized_question,
            answer=answer,
        )

        return RagResult(
            game_slug=slug,
            question=normalized_question,
            collection_name=self.collection_name,
            sources=sources,
            context=context,
            answer=answer,
            answer_mode=answer_mode,
            history=updated_history,
        )


class SentenceTransformersCrossEncoderReranker:
    def __init__(self, model_name: str = RAG_RERANK_MODEL) -> None:
        if not model_name:
            raise RagDependencyError("RAG_RERANK_MODEL cannot be empty.")
        self.model_name = model_name
        self._model = None

    def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        if limit < 1 or not chunks:
            return []
        if len(chunks) == 1:
            return list(chunks[:limit])

        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RagDependencyError(
                    "Install sentence-transformers to use cross-encoder reranking."
                ) from exc
            self._model = CrossEncoder(self.model_name)

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)
        scored_chunks = [
            replace(chunk, score=float(score))
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        return sorted(
            scored_chunks,
            key=lambda chunk: chunk.score,
            reverse=True,
        )[:limit]


def _default_answer_generator() -> AnswerGenerator | None:
    if not LLM_BASE_URL or not LLM_MODEL:
        return None
    return OpenAICompatibleAnswerGenerator(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
    )


def _candidate_limit(
    *,
    final_limit: int,
    candidate_limit: int,
    reranker: ChunkReranker | None,
) -> int:
    if reranker is None:
        return final_limit
    return max(final_limit, candidate_limit)


def _normalize_history(
    history: Sequence[ConversationMessage],
) -> tuple[ConversationMessage, ...]:
    normalized: list[ConversationMessage] = []
    for message in history:
        content = message.content.strip()
        if not content or message.role not in {"user", "assistant"}:
            continue
        normalized.append(ConversationMessage(role=message.role, content=content))
    return tuple(normalized[-MAX_HISTORY_MESSAGES:])


def _append_turn(
    history: Sequence[ConversationMessage],
    *,
    question: str,
    answer: str,
) -> tuple[ConversationMessage, ...]:
    return (
        *history,
        ConversationMessage(role="user", content=question),
        ConversationMessage(role="assistant", content=answer),
    )[-MAX_HISTORY_MESSAGES:]


def _build_retrieval_query(
    *,
    question: str,
    history: Sequence[ConversationMessage],
) -> str:
    if not history:
        return question
    return (
        f"Conversation history:\n{build_conversation_context(history)}\n\n"
        f"Current question: {question}"
    )


def _build_context(
    sources: Sequence[RetrievedChunk],
    *,
    history: Sequence[ConversationMessage] = (),
) -> str:
    source_context = "\n\n---\n\n".join(source.text for source in sources)
    if not history:
        return source_context
    return (
        f"Conversation history:\n{build_conversation_context(history)}\n\n"
        f"Manual excerpts:\n{source_context}"
    )


def build_source_excerpt(
    text: str,
    max_chars: int = SOURCE_EXCERPT_MAX_CHARS,
) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized

    excerpt = normalized[:max_chars].rsplit(" ", 1)[0]
    return f"{excerpt or normalized[:max_chars]}..."


def _build_answer(
    *,
    question: str,
    game_slug: str,
    sources: Sequence[RetrievedChunk],
    answer_generator: AnswerGenerator | None,
    history: Sequence[ConversationMessage] = (),
) -> tuple[str, str]:
    if not sources:
        return "No relevant manual passages were found for that question.", "none"

    if answer_generator is not None:
        answer = answer_generator.generate(
            question=question,
            game_slug=game_slug,
            sources=sources,
            history=history,
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
