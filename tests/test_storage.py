from pathlib import Path

import pytest

from tabletop_manual_retriever.storage.manuals import (
    list_games,
    list_manuals,
    save_manual,
    validate_game_slug,
)


def test_validate_game_slug_accepts_valid_values() -> None:
    assert validate_game_slug("catan") == "catan"
    assert validate_game_slug("Ticket-To-Ride") == "ticket-to-ride"


@pytest.mark.parametrize(
    "slug",
    ["", ".", "..", "catan/rules", "catan rules", "catan_rules"],
)
def test_validate_game_slug_rejects_invalid_values(slug: str) -> None:
    with pytest.raises(ValueError):
        validate_game_slug(slug)


def test_save_manual_creates_game_directory_and_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tabletop_manual_retriever.storage.manuals.UPLOADS_DIR", tmp_path
    )

    saved_path, overwritten = save_manual("catan", "rules.pdf", b"%PDF-1.4")

    assert saved_path == (tmp_path / "catan" / "rules.pdf").resolve()
    assert saved_path.read_bytes() == b"%PDF-1.4"
    assert overwritten is False
    assert list_games() == ["catan"]
    assert list_manuals("catan") == ["rules.pdf"]


def test_save_manual_stores_multiple_pdfs_for_one_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tabletop_manual_retriever.storage.manuals.UPLOADS_DIR", tmp_path
    )

    save_manual("catan", "base-rules.pdf", b"%PDF-1.4")
    save_manual("catan", "seafarers.pdf", b"%PDF-1.5")

    assert list_manuals("catan") == ["base-rules.pdf", "seafarers.pdf"]


def test_save_manual_accepts_spaces_in_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tabletop_manual_retriever.storage.manuals.UPLOADS_DIR", tmp_path
    )

    saved_path, overwritten = save_manual("catan", "base rules v2.pdf", b"%PDF-1.4")

    assert saved_path == (tmp_path / "catan" / "base rules v2.pdf").resolve()
    assert overwritten is False
    assert list_manuals("catan") == ["base rules v2.pdf"]


def test_save_manual_rejects_overwrite_without_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tabletop_manual_retriever.storage.manuals.UPLOADS_DIR", tmp_path
    )

    save_manual("catan", "rules.pdf", b"original")

    with pytest.raises(ValueError, match="already exists"):
        save_manual("catan", "rules.pdf", b"replacement")


def test_save_manual_allows_overwrite_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tabletop_manual_retriever.storage.manuals.UPLOADS_DIR", tmp_path
    )

    save_manual("catan", "rules.pdf", b"original")
    saved_path, overwritten = save_manual(
        "catan",
        "rules.pdf",
        b"replacement",
        overwrite=True,
    )

    assert overwritten is True
    assert saved_path.read_bytes() == b"replacement"


def test_save_manual_rejects_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tabletop_manual_retriever.storage.manuals.UPLOADS_DIR", tmp_path
    )

    with pytest.raises(ValueError, match="empty"):
        save_manual("catan", "rules.pdf", b"")


def test_save_manual_rejects_unsafe_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tabletop_manual_retriever.storage.manuals.UPLOADS_DIR", tmp_path
    )

    with pytest.raises(ValueError, match="Filename"):
        save_manual("catan", "rules.txt", b"%PDF-1.4")
