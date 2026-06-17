from pydantic import BaseModel, Field

from tabletop_manual_retriever.ingest.service import IngestResult


class IngestRequest(BaseModel):
    game_slug: str = Field(..., min_length=1)
    filename: str | None = None


class IngestManualResponse(BaseModel):
    filename: str
    path: str
    parsed_path: str
    page_count: int
    chunk_count: int
    point_count: int


class IngestResponse(BaseModel):
    game_slug: str
    collection_name: str
    total_chunks: int
    total_points: int
    manuals: list[IngestManualResponse]

    @classmethod
    def from_result(cls, result: IngestResult) -> "IngestResponse":
        return cls(
            game_slug=result.game_slug,
            collection_name=result.collection_name,
            total_chunks=result.total_chunks,
            total_points=result.total_points,
            manuals=[
                IngestManualResponse(
                    filename=manual.filename,
                    path=manual.path,
                    parsed_path=manual.parsed_path,
                    page_count=manual.page_count,
                    chunk_count=manual.chunk_count,
                    point_count=manual.point_count,
                )
                for manual in result.manuals
            ],
        )
