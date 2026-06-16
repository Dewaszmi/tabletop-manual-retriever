from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tabletop_manual_retriever.main import app


@pytest.fixture
def uploads_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "tabletop_manual_retriever.storage.manuals.UPLOADS_DIR", tmp_path
    )
    monkeypatch.setattr(
        "tabletop_manual_retriever.config.UPLOADS_DIR", tmp_path
    )
    return tmp_path


def test_upload_manual_endpoint_stores_pdf(uploads_dir: Path) -> None:
    client = TestClient(app)

    response = client.post(
        "/games/catan/manuals",
        files={"file": ("rules.pdf", b"%PDF-1.4 sample", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["game_slug"] == "catan"
    assert payload["filename"] == "rules.pdf"
    assert payload["size_bytes"] == len(b"%PDF-1.4 sample")
    assert (uploads_dir / "catan" / "rules.pdf").read_bytes() == b"%PDF-1.4 sample"


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


def test_upload_manual_endpoint_rejects_non_pdf() -> None:
    client = TestClient(app)

    response = client.post(
        "/games/catan/manuals",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload a .pdf file"
