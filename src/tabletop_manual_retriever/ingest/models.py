from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextBlock:
    """A contiguous block of text extracted from a single PDF page."""

    text: str
    page_number: int
    block_index: int
    font_size: float | None = None


@dataclass(frozen=True)
class ParsedManual:
    """Structured text extracted from a rulebook PDF."""

    source_path: Path
    page_count: int
    blocks: tuple[TextBlock, ...]

    @property
    def full_text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text)

    def text_for_page(self, page_number: int) -> str:
        page_blocks = [b.text for b in self.blocks if b.page_number == page_number]
        return "\n\n".join(page_blocks)

    def pages(self) -> list[tuple[int, str]]:
        return [
            (page_number, self.text_for_page(page_number))
            for page_number in range(1, self.page_count + 1)
            if self.text_for_page(page_number).strip()
        ]


@dataclass(frozen=True)
class ManualChunk:
    """A text fragment ready to embed and store in vector search."""

    text: str
    page_number: int
    chunk_index: int
