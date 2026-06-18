from pathlib import Path

import pytest

from tabletop_manual_retriever.storage.chats import ChatStore


def test_chat_store_saves_and_loads_conversation(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chats.sqlite3")

    summary = store.create_conversation(
        game_slug="catan",
        title="How do turns work?",
    )
    store.append_turn(
        conversation_id=summary.id,
        question="What happens on my turn?",
        answer="Roll and move.",
        sources=[
            {
                "text": "Roll two dice.",
                "excerpt": "Roll two dice.",
                "page_number": 2,
                "chunk_index": 0,
                "filename": "rules.pdf",
                "score": 0.9,
            }
        ],
    )

    loaded = store.get_conversation(summary.id)
    chats = store.list_conversations()

    assert loaded.game_slug == "catan"
    assert loaded.title == "How do turns work?"
    assert [message.role for message in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[1].sources[0]["filename"] == "rules.pdf"
    assert chats[0].id == summary.id
    assert chats[0].message_count == 2
    assert chats[0].last_message == "Roll and move."


def test_chat_store_deletes_conversation(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chats.sqlite3")
    summary = store.create_conversation(game_slug="catan", title="Rules")

    assert store.delete_conversation(summary.id) is True
    assert store.delete_conversation(summary.id) is False

    with pytest.raises(FileNotFoundError):
        store.get_conversation(summary.id)
