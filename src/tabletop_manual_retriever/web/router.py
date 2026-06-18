from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["web"])

STATIC_DIR = Path(__file__).resolve().parent / "static"


@router.get("/")
@router.get("/upload")
@router.get("/library")
@router.get("/ask")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
