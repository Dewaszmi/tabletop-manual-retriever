from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from tabletop_manual_retriever.config import PROJECT_ROOT
from tabletop_manual_retriever.ingest import save_parsed_manual
from tabletop_manual_retriever.storage import (
    list_games,
    list_library,
    list_manuals,
    save_manual,
    validate_game_slug,
)

router = APIRouter(prefix="/games", tags=["uploads"])


def _relative_upload_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@router.get("")
async def list_games_endpoint() -> dict:
    return {"games": list_games()}


@router.get("/library")
async def list_library_endpoint() -> dict:
    return {"library": list_library()}


@router.get("/{game_slug}/manuals")
async def list_manuals_endpoint(game_slug: str) -> dict:
    try:
        slug = validate_game_slug(game_slug)
        manuals = list_manuals(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"game_slug": slug, "manuals": manuals}


@router.post("/{game_slug}/manuals")
async def upload_manual_endpoint(
    game_slug: str,
    file: UploadFile = File(...),
) -> dict:
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Upload a .pdf file")

    content = await file.read()

    try:
        slug = validate_game_slug(game_slug)
        saved_path = save_manual(slug, file.filename, content)
        manual, parsed_path = save_parsed_manual(saved_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "game_slug": slug,
        "filename": saved_path.name,
        "path": _relative_upload_path(saved_path),
        "parsed_path": _relative_upload_path(parsed_path),
        "page_count": manual.page_count,
        "block_count": len(manual.blocks),
        "size_bytes": saved_path.stat().st_size,
    }
