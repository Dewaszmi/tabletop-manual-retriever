from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from tabletop_manual_retriever.main import app


def _make_sample_pdf(path: Path) -> None:
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), "Victory\nReach 10 points.", fontsize=14)
        document.save(path)
    finally:
        document.close()


@pytest.fixture
def uploads_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "tabletop_manual_retriever.storage.manuals.UPLOADS_DIR", tmp_path
    )
    monkeypatch.setattr(
        "tabletop_manual_retriever.config.UPLOADS_DIR", tmp_path
    )
    return tmp_path


def test_upload_manual_endpoint_stores_pdf_and_parsed_json(
    uploads_dir: Path, tmp_path: Path
) -> None:
    pdf_bytes = tmp_path / "source.pdf"
    _make_sample_pdf(pdf_bytes)

    client = TestClient(app)
    response = client.post(
        "/games/catan/manuals",
        files={"file": ("rules.pdf", pdf_bytes.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["game_slug"] == "catan"
    assert payload["filename"] == "rules.pdf"
    assert payload["page_count"] == 1
    assert payload["block_count"] >= 1
    assert payload["overwritten"] is False
    assert (uploads_dir / "catan" / "rules.pdf").is_file()
    assert (uploads_dir / "catan" / "rules.json").is_file()


def test_upload_manual_endpoint_accepts_spaces_in_filename(
    uploads_dir: Path, tmp_path: Path
) -> None:
    pdf_bytes = tmp_path / "source.pdf"
    _make_sample_pdf(pdf_bytes)

    client = TestClient(app)
    response = client.post(
        "/games/catan/manuals",
        files={"file": ("base rules.pdf", pdf_bytes.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "base rules.pdf"
    assert (uploads_dir / "catan" / "base rules.pdf").is_file()
    assert (uploads_dir / "catan" / "base rules.json").is_file()


def test_upload_multiple_manuals_for_one_game(
    uploads_dir: Path, tmp_path: Path
) -> None:
    base_pdf = tmp_path / "base.pdf"
    expansion_pdf = tmp_path / "expansion.pdf"
    _make_sample_pdf(base_pdf)
    _make_sample_pdf(expansion_pdf)

    client = TestClient(app)
    response = client.post(
        "/games/catan/manuals",
        files=[
            ("files", ("base-rules.pdf", base_pdf.read_bytes(), "application/pdf")),
            ("files", ("seafarers.pdf", expansion_pdf.read_bytes(), "application/pdf")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["game_slug"] == "catan"
    assert len(payload["uploads"]) == 2
    assert sorted((uploads_dir / "catan").glob("*.pdf")) == sorted([
        uploads_dir / "catan" / "base-rules.pdf",
        uploads_dir / "catan" / "seafarers.pdf",
    ])
    assert client.get("/games/catan/manuals").json()["manuals"] == [
        "base-rules.pdf",
        "seafarers.pdf",
    ]


def test_upload_manual_endpoint_returns_conflict_without_overwrite(
    uploads_dir: Path, tmp_path: Path
) -> None:
    pdf_bytes = tmp_path / "source.pdf"
    _make_sample_pdf(pdf_bytes)
    client = TestClient(app)

    first = client.post(
        "/games/catan/manuals",
        files={"file": ("rules.pdf", pdf_bytes.read_bytes(), "application/pdf")},
    )
    second = client.post(
        "/games/catan/manuals",
        files={"file": ("rules.pdf", pdf_bytes.read_bytes(), "application/pdf")},
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert "already exists" in second.json()["detail"]


def test_upload_manual_endpoint_can_overwrite_existing_manual(
    uploads_dir: Path, tmp_path: Path
) -> None:
    pdf_bytes = tmp_path / "source.pdf"
    _make_sample_pdf(pdf_bytes)
    client = TestClient(app)

    client.post(
        "/games/catan/manuals",
        files={"file": ("rules.pdf", pdf_bytes.read_bytes(), "application/pdf")},
    )
    response = client.post(
        "/games/catan/manuals?overwrite=true",
        files={"file": ("rules.pdf", pdf_bytes.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["overwritten"] is True


def test_list_games_and_manuals_endpoints(uploads_dir: Path) -> None:
    client = TestClient(app)
    (uploads_dir / "catan").mkdir()
    (uploads_dir / "catan" / "rules.pdf").write_bytes(b"%PDF-1.4")

    games_response = client.get("/games")
    manuals_response = client.get("/games/catan/manuals")

    assert games_response.status_code == 200
    assert games_response.json() == {"games": ["catan"]}
    assert manuals_response.status_code == 200
    assert manuals_response.json() == {
        "game_slug": "catan",
        "manuals": ["rules.pdf"],
    }


def test_library_endpoint_returns_multiple_manuals(uploads_dir: Path) -> None:
    client = TestClient(app)
    game_dir = uploads_dir / "catan"
    game_dir.mkdir()
    (game_dir / "base-rules.pdf").write_bytes(b"%PDF-1.4")
    (game_dir / "seafarers.pdf").write_bytes(b"%PDF-1.4")

    response = client.get("/games/library")

    assert response.status_code == 200
    manuals = response.json()["library"][0]["manuals"]
    assert [manual["filename"] for manual in manuals] == [
        "base-rules.pdf",
        "seafarers.pdf",
    ]


def test_library_endpoint_returns_manual_metadata(uploads_dir: Path) -> None:
    client = TestClient(app)
    (uploads_dir / "catan").mkdir()
    (uploads_dir / "catan" / "rules.pdf").write_bytes(b"%PDF-1.4")
    (uploads_dir / "catan" / "rules.json").write_text(
        '{"page_count": 2, "blocks": [{}, {}]}',
        encoding="utf-8",
    )

    response = client.get("/games/library")

    assert response.status_code == 200
    library = response.json()["library"]
    assert library[0]["game_slug"] == "catan"
    assert library[0]["manuals"][0]["parsed"] is True
    assert library[0]["manuals"][0]["page_count"] == 2
    assert library[0]["manuals"][0]["block_count"] == 2


def test_upload_manual_endpoint_rejects_non_pdf() -> None:
    client = TestClient(app)

    response = client.post(
        "/games/catan/manuals",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "Upload a .pdf file" in response.json()["detail"]


def test_index_page_is_served() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Manual Library" in response.text
