import re
from pathlib import Path

from tabletop_manual_retriever.config import UPLOADS_DIR

_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+\.pdf$")
_INVALID_SLUGS = frozenset({".", ".."})


def validate_game_slug(slug: str) -> str:
    normalized = slug.strip().lower()
    if not normalized or normalized in _INVALID_SLUGS:
        raise ValueError("Game slug is required")
    if "/" in normalized or "\\" in normalized or not _SLUG_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Game slug may only contain lowercase letters, numbers, and hyphens"
        )
    return normalized


def sanitize_pdf_filename(filename: str) -> str:
    if not filename:
        raise ValueError("Filename is required")

    name = Path(filename).name
    if not _FILENAME_PATTERN.fullmatch(name):
        raise ValueError("Filename must be a .pdf file with only letters, numbers, dots, underscores, or hyphens")

    return name


def game_dir(game_slug: str) -> Path:
    slug = validate_game_slug(game_slug)
    return UPLOADS_DIR / slug


def save_manual(game_slug: str, filename: str, content: bytes) -> Path:
    if not content:
        raise ValueError("Uploaded file is empty")

    slug = validate_game_slug(game_slug)
    safe_name = sanitize_pdf_filename(filename)
    destination = game_dir(slug) / safe_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination.resolve()


def list_games() -> list[str]:
    if not UPLOADS_DIR.exists():
        return []
    return sorted(
        path.name for path in UPLOADS_DIR.iterdir() if path.is_dir()
    )


def list_manuals(game_slug: str) -> list[str]:
    slug = validate_game_slug(game_slug)
    manuals_dir = UPLOADS_DIR / slug
    if not manuals_dir.exists():
        return []
    return sorted(
        path.name
        for path in manuals_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
