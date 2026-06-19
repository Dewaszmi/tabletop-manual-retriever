from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from tabletop_manual_retriever.ingest.service import RetrievedChunk


class LlmError(RuntimeError):
    """Raised when the LLM request fails."""


@dataclass(frozen=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: str


class AnswerGenerator(Protocol):
    def generate(
        self,
        *,
        question: str,
        game_slug: str,
        sources: Sequence[RetrievedChunk],
        history: Sequence[ConversationMessage] = (),
    ) -> str:
        """Return a natural-language answer grounded in the sources."""


_SYSTEM_PROMPT = (
    "You are a friendly, helpful mentor and tabletop rules referee. "
    "Your goal is to resolve disputes instantly so players can get back to the fun. "
    "1. Base your answer STRICTLY on the provided excerpts. Do not invent rules. "
    "2. Be semantically flexible: players may use generic terms (like 'game', 'turn', or 'board') for game-specific terms (like 'adventure', 'phase', or 'location'). Map these logically. "
    "3. Keep your response extremely short. State the exact rule in the very first sentence. "
    "4. Cite your source naturally in the text (e.g., 'The youngest player starts [3].'). "
    "5. If the rule is completely missing from the text, politely tell the players it is not covered in the manual."
)

def build_llm_context(sources: Sequence[RetrievedChunk]) -> str:
    if not sources:
        return ""
    return "\n\n".join(
        (
            f"[{index}] {source.filename}, page {source.page_number}\n"
            f"{source.text.strip()}"
        )
        for index, source in enumerate(sources, start=1)
    )


def build_conversation_context(history: Sequence[ConversationMessage]) -> str:
    if not history:
        return "No previous conversation."
    return "\n".join(
        f"{message.role}: {message.content.strip()}"
        for message in history
        if message.content.strip()
    )


def build_llm_messages(
    *,
    question: str,
    game_slug: str,
    sources: Sequence[RetrievedChunk],
    history: Sequence[ConversationMessage] = (),
) -> list[dict[str, str]]:
    context = build_llm_context(sources)
    conversation = build_conversation_context(history)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Game: {game_slug}\n\n"
                f"Conversation history:\n{conversation}\n\n"
                f"Manual excerpts:\n{context}\n\n"
                f"Current question: {question}"
            ),
        },
    ]


class OpenAICompatibleAnswerGenerator:
    """Call any OpenAI-compatible chat completions API (OpenAI, Ollama, etc.)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        *,
        question: str,
        game_slug: str,
        sources: Sequence[RetrievedChunk],
        history: Sequence[ConversationMessage] = (),
    ) -> str:
        if not sources:
            raise ValueError("sources are required to generate an answer")

        try:
            import httpx
        except ImportError as exc:
            raise LlmError("Install httpx to call the LLM API.") from exc

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": build_llm_messages(
                question=question,
                game_slug=game_slug,
                sources=sources,
                history=history,
            ),
            "temperature": 0.2,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LlmError(
                f"LLM request failed ({exc.response.status_code}): "
                f"{_format_llm_error_body(exc.response)}"
            ) from exc
        except Exception as exc:
            raise LlmError(f"LLM request failed: {exc}") from exc

        if not isinstance(content, str) or not content.strip():
            raise LlmError("LLM returned an empty answer.")

        return content.strip()


def _format_llm_error_body(response: object) -> str:
    try:
        import httpx

        if isinstance(response, httpx.Response):
            body = response.json()
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict) and error.get("message"):
                    return str(error["message"])
                if isinstance(error, str):
                    return error
            return response.text.strip() or "Unknown error"
    except Exception:
        pass
    return "Unknown error"
