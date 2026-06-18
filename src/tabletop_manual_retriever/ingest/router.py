import asyncio
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from tabletop_manual_retriever.ingest import parse_pdf
from tabletop_manual_retriever.ingest.schemas import (
    DeingestResponse,
    IngestRequest,
    IngestResponse,
)
from tabletop_manual_retriever.ingest.service import (
    IngestDependencyError,
    IngestService,
    IngestStoreError,
)

router = APIRouter(tags=["ingest"])


@lru_cache
def get_ingest_service() -> IngestService:
    return IngestService()


@router.post("/parse-pdf")
async def parse_pdf_endpoint(file: UploadFile = File(...)) -> dict:
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Upload a .pdf file")

    with TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "upload.pdf"
        pdf_path.write_bytes(await file.read())

        try:
            manual = parse_pdf(pdf_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "source_filename": file.filename,
        "page_count": manual.page_count,
        "blocks": [
            {
                "text": block.text,
                "page_number": block.page_number,
                "block_index": block.block_index,
                "font_size": block.font_size,
            }
            for block in manual.blocks
        ],
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    request: IngestRequest,
    service: Annotated[IngestService, Depends(get_ingest_service)],
) -> IngestResponse:
    try:
        result = await asyncio.to_thread(
            service.ingest_manuals,
            game_slug=request.game_slug,
            filename=request.filename,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IngestDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except IngestStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return IngestResponse.from_result(result)


@router.delete("/ingest", response_model=DeingestResponse)
async def deingest_endpoint(
    request: IngestRequest,
    service: Annotated[IngestService, Depends(get_ingest_service)],
) -> DeingestResponse:
    try:
        result = await asyncio.to_thread(
            service.deingest_manuals,
            game_slug=request.game_slug,
            filename=request.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IngestDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except IngestStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return DeingestResponse.from_result(result)
