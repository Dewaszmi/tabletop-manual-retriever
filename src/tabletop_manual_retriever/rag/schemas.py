from typing import Literal

from pydantic import BaseModel, Field

from tabletop_manual_retriever.rag.llm import ConversationMessage
from tabletop_manual_retriever.rag.service import RagResult, build_source_excerpt
from tabletop_manual_retriever.storage.chats import (
    StoredChat,
    StoredChatMessage,
    StoredChatSummary,
)


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
    conversation_id: str | None = None


class RagSourceResponse(BaseModel):
    text: str
    excerpt: str
    page_number: int
    chunk_index: int
    filename: str
    score: float


class RagResponse(BaseModel):
    conversation_id: str | None = None
    game_slug: str
    question: str
    collection_name: str
    sources: list[RagSourceResponse]
    context: str
    answer: str
    answer_mode: str
    history: list[RagHistoryMessage]

    @classmethod
    def from_result(
        cls,
        result: RagResult,
        *,
        conversation_id: str | None = None,
    ) -> "RagResponse":
        return cls(
            conversation_id=conversation_id,
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


class ChatSummaryResponse(BaseModel):
    id: str
    game_slug: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_message: str | None = None

    @classmethod
    def from_summary(cls, summary: StoredChatSummary) -> "ChatSummaryResponse":
        return cls(
            id=summary.id,
            game_slug=summary.game_slug,
            title=summary.title,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            message_count=summary.message_count,
            last_message=summary.last_message,
        )


class ChatListResponse(BaseModel):
    chats: list[ChatSummaryResponse]


class ChatMessageResponse(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[RagSourceResponse] = Field(default_factory=list)
    created_at: str

    @classmethod
    def from_message(cls, message: StoredChatMessage) -> "ChatMessageResponse":
        return cls(
            role=message.role,
            content=message.content,
            sources=[
                RagSourceResponse.model_validate(source)
                for source in message.sources
            ],
            created_at=message.created_at,
        )


class ChatConversationResponse(BaseModel):
    id: str
    game_slug: str
    title: str
    created_at: str
    updated_at: str
    messages: list[ChatMessageResponse]

    @classmethod
    def from_chat(cls, chat: StoredChat) -> "ChatConversationResponse":
        return cls(
            id=chat.id,
            game_slug=chat.game_slug,
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            messages=[
                ChatMessageResponse.from_message(message)
                for message in chat.messages
            ],
        )
