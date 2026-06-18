from typing import Literal

from pydantic import BaseModel, Field

from tabletop_manual_retriever.rag.llm import ConversationMessage
from tabletop_manual_retriever.rag.service import RagResult, build_source_excerpt


class RagHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)

    def to_conversation_message(self) -> ConversationMessage:
        return ConversationMessage(role=self.role, content=self.content)


class RagRequest(BaseModel):
    game_slug: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    filename: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    history: list[RagHistoryMessage] = Field(default_factory=list, max_length=20)


class RagSourceResponse(BaseModel):
    text: str
    excerpt: str
    page_number: int
    chunk_index: int
    filename: str
    score: float


class RagResponse(BaseModel):
    game_slug: str
    question: str
    collection_name: str
    sources: list[RagSourceResponse]
    context: str
    answer: str
    answer_mode: str
    history: list[RagHistoryMessage]

    @classmethod
    def from_result(cls, result: RagResult) -> "RagResponse":
        return cls(
            game_slug=result.game_slug,
            question=result.question,
            collection_name=result.collection_name,
            sources=[
                RagSourceResponse(
                    text=source.text,
                    excerpt=build_source_excerpt(source.text),
                    page_number=source.page_number,
                    chunk_index=source.chunk_index,
                    filename=source.filename,
                    score=source.score,
                )
                for source in result.sources
            ],
            context=result.context,
            answer=result.answer,
            answer_mode=result.answer_mode,
            history=[
                RagHistoryMessage(role=message.role, content=message.content)
                for message in result.history
            ],
        )
