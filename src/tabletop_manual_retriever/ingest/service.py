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
