from __future__ import annotations

from pathlib import Path

import fitz

from tabletop_manual_retriever.ingest.models import ParsedManual, TextBlock


def parse_pdf(path: str | Path) -> ParsedManual:
    """Extract text blocks with page numbers from a PDF manual."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"PDF not found: {source_path}")

    if source_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {source_path.suffix!r}")

    blocks: list[TextBlock] = []

    with fitz.open(source_path) as document:
        for page_index in range(document.page_count):
            page_number = page_index + 1
            page = document[page_index]
            page_dict = page.get_text("dict")

            text_blocks = [
                block
                for block in page_dict.get("blocks", [])
                if block.get("type") == 0
            ]
            text_blocks.sort(key=_block_sort_key)

            for block_index, block in enumerate(text_blocks):
                text = _extract_block_text(block).strip()
                if not text:
                    continue

                blocks.append(
                    TextBlock(
                        text=text,
                        page_number=page_number,
                        block_index=block_index,
                        font_size=_max_font_size(block),
                    )
                )

        return ParsedManual(
            source_path=source_path.resolve(),
            page_count=document.page_count,
            blocks=tuple(blocks),
        )


def _block_sort_key(block: dict) -> tuple[float, float]:
    x0, y0, _, _ = block["bbox"]
    return (round(y0, 1), round(x0, 1))


def _extract_block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        spans = [span.get("text", "") for span in line.get("spans", [])]
        line_text = "".join(spans).strip()
        if line_text:
            lines.append(line_text)
    return "\n".join(lines)


def _max_font_size(block: dict) -> float | None:
    sizes = [
        span["size"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if "size" in span
    ]
    return max(sizes) if sizes else None
