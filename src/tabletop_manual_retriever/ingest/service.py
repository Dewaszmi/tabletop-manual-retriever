from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from tabletop_manual_retriever.config import (
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    EMBEDDING_MODEL,
    PROJECT_ROOT,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from tabletop_manual_retriever.ingest.chunking import chunk_manual
from tabletop_manual_retriever.ingest.models import ManualChunk
from tabletop_manual_retriever.ingest.serialize import save_parsed_manual
from tabletop_manual_retriever.storage.manuals import (
    list_manuals,
    manual_path,
    sanitize_pdf_filename,
    validate_game_slug,
)


class IngestDependencyError(RuntimeError):
    """Raised when optional ingest dependencies are not installed."""


class IngestStoreError(RuntimeError):
    """Raised when chunks cannot be written to the vector store."""


class TextEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per text."""


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page_number: int
    chunk_index: int
    filename: str
    score: float


class VectorStore(Protocol):
    def upsert_chunks(
        self,
        *,
        collection_name: str,
        game_slug: str,
        filename: str,
        source_path: str,
        parsed_path: str,
        chunks: Sequence[ManualChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Store chunk vectors and return the number of upserted points."""

    def delete_chunks(
        self,
        *,
        collection_name: str,
        game_slug: str,
        filename: str | None = None,
    ) -> int:
        """Delete chunk vectors and return the number of removed points."""

    def count_chunks(
        self,
        *,
        collection_name: str,
        game_slug: str,
        filename: str | None = None,
    ) -> int:
        """Return the number of stored chunk vectors matching the filters."""

    def search_chunks(
        self,
        *,
        collection_name: str,
        vector: Sequence[float],
        game_slug: str,
        filename: str | None = None,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Return the closest stored chunks for a query vector."""


@dataclass(frozen=True)
class IngestedManual:
    filename: str
    path: str
    parsed_path: str
    page_count: int
    chunk_count: int
    point_count: int


@dataclass(frozen=True)
class IngestResult:
    game_slug: str
    collection_name: str
    manuals: tuple[IngestedManual, ...]

    @property
    def total_chunks(self) -> int:
        return sum(manual.chunk_count for manual in self.manuals)

    @property
    def total_points(self) -> int:
        return sum(manual.point_count for manual in self.manuals)


@dataclass(frozen=True)
class DeingestedManual:
    filename: str | None
    point_count: int


@dataclass(frozen=True)
class DeingestResult:
    game_slug: str
    collection_name: str
    manuals: tuple[DeingestedManual, ...]

    @property
    def total_points(self) -> int:
        return sum(manual.point_count for manual in self.manuals)


class FastEmbedTextEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise IngestDependencyError(
                    "Install fastembed to generate embeddings."
                ) from exc
            self._model = TextEmbedding(model_name=self.model_name)

        return [
            [float(value) for value in vector]
            for vector in self._model.embed(list(texts))
        ]


class QdrantVectorStore:
    def __init__(self, url: str = QDRANT_URL) -> None:
        self.url = url
        self._client = None

    def upsert_chunks(
        self,
        *,
        collection_name: str,
        game_slug: str,
        filename: str,
        source_path: str,
        parsed_path: str,
        chunks: Sequence[ManualChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise IngestDependencyError(
                "Install qdrant-client to write embeddings to Qdrant."
            ) from exc

        try:
            if self._client is None:
                self._client = QdrantClient(
                    url=self.url,
                    check_compatibility=False,
                )

            vector_size = len(vectors[0])
            if not self._client.collection_exists(collection_name=collection_name):
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )

            points = [
                models.PointStruct(
                    id=_point_id(game_slug, filename, chunk.chunk_index),
                    vector=[float(value) for value in vector],
                    payload={
                        "game_slug": game_slug,
                        "filename": filename,
                        "source_path": source_path,
                        "parsed_path": parsed_path,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]

            self._client.upsert(collection_name=collection_name, points=points)
            return len(points)
        except Exception as exc:
            raise IngestStoreError(f"Could not write chunks to Qdrant: {exc}") from exc

    def delete_chunks(
        self,
        *,
        collection_name: str,
        game_slug: str,
        filename: str | None = None,
    ) -> int:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise IngestDependencyError(
                "Install qdrant-client to delete embeddings from Qdrant."
            ) from exc

        try:
            if self._client is None:
                self._client = QdrantClient(
                    url=self.url,
                    check_compatibility=False,
                )

            if not self._client.collection_exists(collection_name=collection_name):
                return 0

            filter_conditions = [
                models.FieldCondition(
                    key="game_slug",
                    match=models.MatchValue(value=game_slug),
                )
            ]
            if filename is not None:
                filter_conditions.append(
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename),
                    )
                )
            point_filter = models.Filter(must=filter_conditions)
            count = self._client.count(
                collection_name=collection_name,
                count_filter=point_filter,
                exact=True,
            ).count
            self._client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=point_filter),
                wait=True,
            )
            return int(count)
        except Exception as exc:
            raise IngestStoreError(
                f"Could not delete chunks from Qdrant: {exc}"
            ) from exc

    def count_chunks(
        self,
        *,
        collection_name: str,
        game_slug: str,
        filename: str | None = None,
    ) -> int:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise IngestDependencyError(
                "Install qdrant-client to count embeddings in Qdrant."
            ) from exc

        try:
            if self._client is None:
                self._client = QdrantClient(
                    url=self.url,
                    check_compatibility=False,
                )

            if not self._client.collection_exists(collection_name=collection_name):
                return 0

            filter_conditions = [
                models.FieldCondition(
                    key="game_slug",
                    match=models.MatchValue(value=game_slug),
                )
            ]
            if filename is not None:
                filter_conditions.append(
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename),
                    )
                )

            return int(
                self._client.count(
                    collection_name=collection_name,
                    count_filter=models.Filter(must=filter_conditions),
                    exact=True,
                ).count
            )
        except Exception as exc:
            raise IngestStoreError(f"Could not count chunks in Qdrant: {exc}") from exc

    def search_chunks(
        self,
        *,
        collection_name: str,
        vector: Sequence[float],
        game_slug: str,
        filename: str | None = None,
        limit: int,
    ) -> list[RetrievedChunk]:
        if limit < 1:
            return []

        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise IngestDependencyError(
                "Install qdrant-client to search embeddings in Qdrant."
            ) from exc

        try:
            if self._client is None:
                self._client = QdrantClient(
                    url=self.url,
                    check_compatibility=False,
                )

            if not self._client.collection_exists(collection_name=collection_name):
                return []

            filter_conditions = [
                models.FieldCondition(
                    key="game_slug",
                    match=models.MatchValue(value=game_slug),
                )
            ]
            if filename is not None:
                filter_conditions.append(
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename),
                    )
                )

            query_response = self._client.query_points(
                collection_name=collection_name,
                query=[float(value) for value in vector],
                query_filter=models.Filter(must=filter_conditions),
                limit=limit,
                with_payload=True,
            )
            hits = query_response.points
        except Exception as exc:
            raise IngestStoreError(f"Could not search chunks in Qdrant: {exc}") from exc

        results: list[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text")
            page_number = payload.get("page_number")
            chunk_index = payload.get("chunk_index")
            hit_filename = payload.get("filename")
            if (
                not isinstance(text, str)
                or not isinstance(page_number, int)
                or not isinstance(chunk_index, int)
                or not isinstance(hit_filename, str)
            ):
                continue
            results.append(
                RetrievedChunk(
                    text=text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    filename=hit_filename,
                    score=float(hit.score or 0.0),
                )
            )
        return results


@dataclass
class IngestService:
    embedder: TextEmbedder | None = None
    vector_store: VectorStore | None = None
    collection_name: str = QDRANT_COLLECTION
    max_chars: int = CHUNK_MAX_CHARS
    overlap_chars: int = CHUNK_OVERLAP_CHARS

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = FastEmbedTextEmbedder()
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore()

    def ingest_manuals(
        self,
        game_slug: str,
        filename: str | None = None,
    ) -> IngestResult:
        slug = validate_game_slug(game_slug)
        filenames = self._manual_filenames(slug, filename)
        if not filenames:
            raise FileNotFoundError(f"No PDF manuals found for game: {slug}")

        manuals = tuple(self._ingest_manual(slug, name) for name in filenames)
        return IngestResult(
            game_slug=slug,
            collection_name=self.collection_name,
            manuals=manuals,
        )

    def deingest_manuals(
        self,
        game_slug: str,
        filename: str | None = None,
    ) -> DeingestResult:
        slug = validate_game_slug(game_slug)
        manual_filename = (
            sanitize_pdf_filename(filename) if filename is not None else None
        )

        assert self.vector_store is not None
        point_count = self.vector_store.delete_chunks(
            collection_name=self.collection_name,
            game_slug=slug,
            filename=manual_filename,
        )
        return DeingestResult(
            game_slug=slug,
            collection_name=self.collection_name,
            manuals=(
                DeingestedManual(
                    filename=manual_filename,
                    point_count=point_count,
                ),
            ),
        )

    def count_manual_chunks(
        self,
        game_slug: str,
        filename: str | None = None,
    ) -> int:
        slug = validate_game_slug(game_slug)
        manual_filename = (
            sanitize_pdf_filename(filename) if filename is not None else None
        )

        assert self.vector_store is not None
        return self.vector_store.count_chunks(
            collection_name=self.collection_name,
            game_slug=slug,
            filename=manual_filename,
        )

    def _manual_filenames(self, game_slug: str, filename: str | None) -> list[str]:
        if filename is not None:
            return [sanitize_pdf_filename(filename)]
        return list_manuals(game_slug)

    def _ingest_manual(self, game_slug: str, filename: str) -> IngestedManual:
        assert self.embedder is not None
        assert self.vector_store is not None

        pdf_path = manual_path(game_slug, filename)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Manual not found: {filename}")

        manual, parsed_path = save_parsed_manual(pdf_path)
        chunks = chunk_manual(
            manual,
            max_chars=self.max_chars,
            overlap_chars=self.overlap_chars,
        )
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        point_count = self.vector_store.upsert_chunks(
            collection_name=self.collection_name,
            game_slug=game_slug,
            filename=filename,
            source_path=_relative_path(pdf_path),
            parsed_path=_relative_path(parsed_path),
            chunks=chunks,
            vectors=vectors,
        )

        return IngestedManual(
            filename=filename,
            path=_relative_path(pdf_path),
            parsed_path=_relative_path(parsed_path),
            page_count=manual.page_count,
            chunk_count=len(chunks),
            point_count=point_count,
        )


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _point_id(game_slug: str, filename: str, chunk_index: int) -> str:
    raw_id = f"tabletop-manual-retriever:{game_slug}:{filename}:{chunk_index}"
    return str(uuid5(NAMESPACE_URL, raw_id))
