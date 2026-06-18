import asyncio
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from tabletop_manual_retriever.ingest.service import IngestDependencyError, IngestStoreError
from tabletop_manual_retriever.rag.schemas import RagRequest, RagResponse
from tabletop_manual_retriever.rag.service import RagService
from tabletop_manual_retriever.rag.llm import LlmError

router = APIRouter(tags=["rag"])


@lru_cache
def get_rag_service() -> RagService:
    return RagService()


@router.post("/query", response_model=RagResponse)
async def query_endpoint(
    request: RagRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> RagResponse:
    try:
        result = await asyncio.to_thread(
            service.query,
            game_slug=request.game_slug,
            question=request.question,
            filename=request.filename,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IngestDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except IngestStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LlmError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return RagResponse.from_result(result)
