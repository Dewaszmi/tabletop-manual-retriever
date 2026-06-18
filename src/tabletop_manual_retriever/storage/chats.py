from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from tabletop_manual_retriever.config import CHAT_DB_PATH
from tabletop_manual_retriever.storage.manuals import validate_game_slug


@dataclass(frozen=True)
class StoredChatMessage:
    role: str
    content: str
    sources: tuple[dict[str, Any], ...]
    created_at: str
    position: int


@dataclass(frozen=True)
class StoredChatSummary:
    id: str
    game_slug: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_message: str | None


@dataclass(frozen=True)
class StoredChat:
    id: str
    game_slug: str
    title: str
    created_at: str
    updated_at: str
    messages: tuple[StoredChatMessage, ...]


class ChatStore:
    def __init__(self, db_path: Path = CHAT_DB_PATH) -> None:
        self.db_path = db_path

    def create_conversation(self, *, game_slug: str, title: str) -> StoredChatSummary:
        slug = validate_game_slug(game_slug)
        chat_id = uuid4().hex
        now = _now()
        safe_title = title.strip() or "New conversation"

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, game_slug, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, slug, safe_title, now, now),
            )

        return StoredChatSummary(
            id=chat_id,
            game_slug=slug,
            title=safe_title,
            created_at=now,
            updated_at=now,
            message_count=0,
            last_message=None,
        )

    def append_turn(
        self,
        *,
        conversation_id: str,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> None:
        now = _now()
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise FileNotFoundError(f"Conversation not found: {conversation_id}")

            row = connection.execute(
                "SELECT COALESCE(MAX(position), -1) AS max_position FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()

            next_position = int(row["max_position"]) + 1
            connection.execute(
                """
                INSERT INTO messages (conversation_id, role, content, sources_json, created_at, position)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, "user", question, "[]", now, next_position),
            )
            connection.execute(
                """
                INSERT INTO messages (conversation_id, role, content, sources_json, created_at, position)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    "assistant",
                    answer,
                    json.dumps(sources),
                    now,
                    next_position + 1,
                ),
            )
            updated = connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            if updated.rowcount == 0:
                raise FileNotFoundError(f"Conversation not found: {conversation_id}")

    def list_conversations(self, *, limit: int = 50) -> list[StoredChatSummary]:
        safe_limit = max(1, min(limit, 200))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.game_slug,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.id) AS message_count,
                    (
                        SELECT content
                        FROM messages
                        WHERE conversation_id = c.id
                        ORDER BY position DESC
                        LIMIT 1
                    ) AS last_message
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        return [_summary_from_row(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> StoredChat:
        with self._connect() as connection:
            conversation = connection.execute(
                """
                SELECT id, game_slug, title, created_at, updated_at
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise FileNotFoundError(f"Conversation not found: {conversation_id}")

            messages = connection.execute(
                """
                SELECT role, content, sources_json, created_at, position
                FROM messages
                WHERE conversation_id = ?
                ORDER BY position ASC
                """,
                (conversation_id,),
            ).fetchall()

        return StoredChat(
            id=str(conversation["id"]),
            game_slug=str(conversation["game_slug"]),
            title=str(conversation["title"]),
            created_at=str(conversation["created_at"]),
            updated_at=str(conversation["updated_at"]),
            messages=tuple(_message_from_row(row) for row in messages),
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
        return result.rowcount > 0

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        _ensure_schema(connection)
        return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            game_slug TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            sources_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            position INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_position
        ON messages (conversation_id, position)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
        ON conversations (updated_at)
        """
    )
    connection.commit()


def _summary_from_row(row: sqlite3.Row) -> StoredChatSummary:
    return StoredChatSummary(
        id=str(row["id"]),
        game_slug=str(row["game_slug"]),
        title=str(row["title"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        message_count=int(row["message_count"]),
        last_message=row["last_message"],
    )


def _message_from_row(row: sqlite3.Row) -> StoredChatMessage:
    return StoredChatMessage(
        role=str(row["role"]),
        content=str(row["content"]),
        sources=tuple(_load_sources(str(row["sources_json"]))),
        created_at=str(row["created_at"]),
        position=int(row["position"]),
    )


def _load_sources(raw_sources: str) -> list[dict[str, Any]]:
    try:
        sources = json.loads(raw_sources)
    except json.JSONDecodeError:
        return []
    if not isinstance(sources, list):
        return []
    return [source for source in sources if isinstance(source, dict)]


def _now() -> str:
    return datetime.now(UTC).isoformat()
