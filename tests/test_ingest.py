import asyncio
from pathlib import Path

import fitz
import pytest

from tabletop_manual_retriever.ingest.chunking import chunk_manual
from tabletop_manual_retriever.ingest.models import ParsedManual, TextBlock
from tabletop_manual_retriever.ingest.router import ingest_endpoint
from tabletop_manual_retriever.ingest.schemas import IngestRequest
from tabletop_manual_retriever.ingest.service import (
    IngestedManual,
    IngestResult,
    IngestService,
)
from tabletop_manual_retriever.storage.manuals import save_manual


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts = list(texts)
        return [
            [float(index), float(len(text))]
            for index, text in enumerate(texts)
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def upsert_chunks(self, **kwargs) -> int:
        self.calls.append(kwargs)
        return len(kwargs["chunks"])


def _make_sample_pdf(path: Path) -> None:
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Setup\nPlace the board in the center of the table.",
            fontsize=14,
        )
        page.insert_text(
            (72, 120),
            "Victory\nReach 10 points to win.",
            fontsize=12,
        )
        document.save(path)
    finally:
        document.close()


def test_chunk_manual_splits_pages_into_numbered_chunks() -> None:
    manual = ParsedManual(
        source_path=Path("rules.pdf"),
        page_count=1,
        blocks=(
            TextBlock(
                text=" ".join(f"word-{index}" for index in range(40)),
                page_number=1,
                block_index=0,
            ),
        ),
    )

    chunks = chunk_manual(manual, max_chars=80, overlap_chars=10)

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.page_number for chunk in chunks} == {1}
    assert all(chunk.text for chunk in chunks)


def test_ingest_service_parses_pdf_embeds_chunks_and_upserts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tabletop_manual_retriever.storage.manuals.UPLOADS_DIR",
        tmp_path,
    )
    source_pdf = tmp_path / "source.pdf"
    _make_sample_pdf(source_pdf)
    save_manual("catan", "rules.pdf", source_pdf.read_bytes())
    fake_embedder = FakeEmbedder()
    fake_store = FakeVectorStore()
    service = IngestService(
        embedder=fake_embedder,
        vector_store=fake_store,
        collection_name="manuals",
        max_chars=80,
        overlap_chars=10,
    )

    result = service.ingest_manuals("catan")

    assert result.game_slug == "catan"
    assert result.collection_name == "manuals"
    assert result.total_chunks >= 1
    assert result.total_points == result.total_chunks
    assert fake_embedder.texts
    assert fake_store.calls[0]["collection_name"] == "manuals"
    assert fake_store.calls[0]["game_slug"] == "catan"
    assert fake_store.calls[0]["filename"] == "rules.pdf"
    assert (tmp_path / "catan" / "rules.json").is_file()


def test_ingest_endpoint_returns_ingest_response() -> None:
    class FakeService:
        def ingest_manuals(
            self,
            game_slug: str,
            filename: str | None = None,
        ) -> IngestResult:
            assert game_slug == "catan"
            assert filename == "rules.pdf"
            return IngestResult(
                game_slug=game_slug,
                collection_name="manuals",
                manuals=(
                    IngestedManual(
                        filename="rules.pdf",
                        path="data/uploads/catan/rules.pdf",
                        parsed_path="data/uploads/catan/rules.json",
                        page_count=1,
                        chunk_count=2,
                        point_count=2,
                    ),
                ),
            )

    response = asyncio.run(
        ingest_endpoint(
            IngestRequest(game_slug="catan", filename="rules.pdf"),
            FakeService(),
        )
    )

    assert response.model_dump() == {
        "game_slug": "catan",
        "collection_name": "manuals",
        "total_chunks": 2,
        "total_points": 2,
        "manuals": [
            {
                "filename": "rules.pdf",
                "path": "data/uploads/catan/rules.pdf",
                "parsed_path": "data/uploads/catan/rules.json",
                "page_count": 1,
                "chunk_count": 2,
                "point_count": 2,
            }
        ],
    }
