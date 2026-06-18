import asyncio
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from tabletop_manual_retriever.ingest.service import IngestDependencyError, IngestStoreError
from tabletop_manual_retriever.rag.llm import ConversationMessage, LlmError
from tabletop_manual_retriever.rag.schemas import (
    ChatConversationResponse,
    ChatListResponse,
    ChatSummaryResponse,
    RagRequest,
    RagResponse,
)
from tabletop_manual_retriever.rag.service import RagService
from tabletop_manual_retriever.storage.chats import ChatStore, StoredChatMessage

router = APIRouter(tags=["rag"])


@lru_cache
def get_rag_service() -> RagService:
    return RagService()


@lru_cache
def get_chat_store() -> ChatStore:
    return ChatStore()


@router.post("/query", response_model=RagResponse)
async def query_endpoint(
    request: RagRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
    chat_store: Annotated[ChatStore, Depends(get_chat_store)],
) -> RagResponse:
    conversation_id = request.conversation_id
    try:
        history = [
            message.to_conversation_message()
            for message in request.history
        ]
        if conversation_id is not None:
            conversation = chat_store.get_conversation(conversation_id)
            if conversation.game_slug != request.game_slug:
                raise ValueError(
                    "Conversation belongs to a different board game."
                )
            history = _messages_to_history(conversation.messages)

        result = await asyncio.to_thread(
            service.query,
            game_slug=request.game_slug,
            question=request.question,
            filename=request.filename,
            top_k=request.top_k,
            history=history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IngestDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except IngestStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LlmError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if conversation_id is None:
        conversation = chat_store.create_conversation(
            game_slug=result.game_slug,
            title=_conversation_title(result.question),
        )
        conversation_id = conversation.id

    response = RagResponse.from_result(result, conversation_id=conversation_id)
    chat_store.append_turn(
        conversation_id=conversation_id,
        question=result.question,
        answer=result.answer,
        sources=[source.model_dump() for source in response.sources],
    )
    return response


@router.get("/chats", response_model=ChatListResponse)
async def list_chats_endpoint(
    chat_store: Annotated[ChatStore, Depends(get_chat_store)],
) -> ChatListResponse:
    return ChatListResponse(
        chats=[
            ChatSummaryResponse.from_summary(summary)
            for summary in chat_store.list_conversations()
        ]
    )


@router.get("/chats/{conversation_id}", response_model=ChatConversationResponse)
async def get_chat_endpoint(
    conversation_id: str,
    chat_store: Annotated[ChatStore, Depends(get_chat_store)],
) -> ChatConversationResponse:
    try:
        chat = chat_store.get_conversation(conversation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatConversationResponse.from_chat(chat)


@router.delete("/chats/{conversation_id}")
async def delete_chat_endpoint(
    conversation_id: str,
    chat_store: Annotated[ChatStore, Depends(get_chat_store)],
) -> dict:
    deleted = chat_store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation not found: {conversation_id}",
        )
    return {"conversation_id": conversation_id, "deleted": True}


def _messages_to_history(
    messages: tuple[StoredChatMessage, ...],
) -> list[ConversationMessage]:
    return [
        ConversationMessage(role=message.role, content=message.content)
        for message in messages
    ]


def _conversation_title(question: str, max_chars: int = 80) -> str:
    normalized = " ".join(question.split())
    if len(normalized) <= max_chars:
        return normalized or "New conversation"
    return f"{normalized[:max_chars].rsplit(' ', 1)[0]}..."
