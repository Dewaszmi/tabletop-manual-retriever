from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from tabletop_manual_retriever.config import PROJECT_ROOT
from tabletop_manual_retriever.ingest import save_parsed_manual
from tabletop_manual_retriever.ingest.router import get_ingest_service
from tabletop_manual_retriever.ingest.service import (
    IngestDependencyError,
    IngestService,
    IngestStoreError,
)
from tabletop_manual_retriever.storage import (
    delete_manual,
    list_games,
    list_library,
    list_manuals,
    manual_exists,
    save_manual,
    validate_game_slug,
)
from tabletop_manual_retriever.storage.manuals import sanitize_pdf_filename

router = APIRouter(prefix="/games", tags=["uploads"])


def _relative_upload_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _collect_upload_files(
    file: UploadFile | None,
    files: list[UploadFile] | None,
) -> list[UploadFile]:
    uploads: list[UploadFile] = []
    if file is not None:
        uploads.append(file)
    if files:
        uploads.extend(files)
    return uploads


async def _process_manual_upload(
    game_slug: str,
    upload: UploadFile,
    content: bytes,
    *,
    overwrite: bool,
) -> dict:
    if not upload.filename or Path(upload.filename).suffix.lower() != ".pdf":
        raise ValueError(f"Upload a .pdf file: {upload.filename or 'unnamed file'}")

    saved_path, existed = save_manual(
        game_slug,
        upload.filename,
        content,
        overwrite=overwrite,
    )
    manual, parsed_path = save_parsed_manual(saved_path)

    return {
        "filename": saved_path.name,
        "path": _relative_upload_path(saved_path),
        "parsed_path": _relative_upload_path(parsed_path),
        "page_count": manual.page_count,
        "block_count": len(manual.blocks),
        "size_bytes": saved_path.stat().st_size,
        "overwritten": existed,
    }


def _add_index_metadata(
    library: list[dict],
    service: IngestService,
) -> list[dict]:
    index_lookup_available = True
    for game in library:
        game_slug = game["game_slug"]
        for manual in game["manuals"]:
            if not index_lookup_available:
                manual["indexed"] = None
                manual["indexed_point_count"] = None
                manual["index_status"] = "unknown"
                continue

            try:
                point_count = service.count_manual_chunks(game_slug, manual["filename"])
            except (IngestDependencyError, IngestStoreError):
                index_lookup_available = False
                manual["indexed"] = None
                manual["indexed_point_count"] = None
                manual["index_status"] = "unknown"
                continue

            manual["indexed"] = point_count > 0
            manual["indexed_point_count"] = point_count
            manual["index_status"] = "indexed" if point_count > 0 else "not_indexed"
    return library


@router.get("")
async def list_games_endpoint() -> dict:
    return {"games": list_games()}


@router.get("/library")
async def list_library_endpoint(
    service: Annotated[IngestService, Depends(get_ingest_service)],
) -> dict:
    library = _add_index_metadata(list_library(), service)
    return {"library": library}


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
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    overwrite: bool = False,
) -> dict:
    uploads = _collect_upload_files(file, files)
    if not uploads:
        raise HTTPException(status_code=400, detail="Upload at least one .pdf file")

    try:
        slug = validate_game_slug(game_slug)
        if not overwrite:
            conflicts: list[str] = []
            for upload in uploads:
                if not upload.filename:
                    continue
                try:
                    safe_name = sanitize_pdf_filename(upload.filename)
                except ValueError:
                    continue
                if manual_exists(slug, safe_name):
                    conflicts.append(safe_name)
            if conflicts:
                raise HTTPException(
                    status_code=409,
                    detail=f"Manual already exists: {', '.join(conflicts)}",
                )

        results: list[dict] = []
        for upload in uploads:
            content = await upload.read()
            try:
                results.append(
                    await _process_manual_upload(
                        slug,
                        upload,
                        content,
                        overwrite=overwrite,
                    )
                )
            except ValueError as exc:
                if len(uploads) == 1:
                    raise
                raise HTTPException(
                    status_code=400,
                    detail=f"{upload.filename}: {exc}",
                ) from exc
    except ValueError as exc:
        message = str(exc)
        status_code = 409 if message.startswith("Manual already exists:") else 400
        raise HTTPException(status_code=status_code, detail=message) from exc
    except HTTPException:
        raise

    if len(results) == 1:
        return {"game_slug": slug, **results[0]}

    return {"game_slug": slug, "uploads": results}


@router.delete("/{game_slug}/manuals/{filename}")
async def delete_manual_endpoint(game_slug: str, filename: str) -> dict:
    try:
        slug = validate_game_slug(game_slug)
        pdf_path, parsed_path = delete_manual(slug, filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "game_slug": slug,
        "filename": pdf_path.name,
        "deleted": True,
        "path": _relative_upload_path(pdf_path),
        "parsed_path": _relative_upload_path(parsed_path) if parsed_path else None,
        "parsed_deleted": parsed_path is not None,
    }
