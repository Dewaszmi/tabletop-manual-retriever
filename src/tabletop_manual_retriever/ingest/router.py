from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, HTTPException, UploadFile

from tabletop_manual_retriever.ingest import parse_pdf

router = APIRouter()


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
