from __future__ import annotations

from tabletop_manual_retriever.ingest.models import ManualChunk, ParsedManual


def chunk_manual(
    manual: ParsedManual,
    *,
    max_chars: int,
    overlap_chars: int,
) -> tuple[ManualChunk, ...]:
    """Split parsed manual pages into embedding-ready chunks."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars")

    chunks: list[ManualChunk] = []
    for page_number, page_text in manual.pages():
        for text in _split_text(page_text, max_chars, overlap_chars):
            chunks.append(
                ManualChunk(
                    text=text,
                    page_number=page_number,
                    chunk_index=len(chunks),
                )
            )

    return tuple(chunks)


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            word_boundary = normalized.rfind(" ", start + 1, end)
            if word_boundary > start:
                end = word_boundary

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break

        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start
        while start < len(normalized) and normalized[start].isspace():
            start += 1

    return chunks
