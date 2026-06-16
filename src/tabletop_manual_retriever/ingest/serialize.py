from __future__ import annotations

import json
from pathlib import Path

from tabletop_manual_retriever.config import PROJECT_ROOT
from tabletop_manual_retriever.ingest.models import ParsedManual
from tabletop_manual_retriever.ingest.parser import parse_pdf


def parsed_manual_path(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(".json")


def manual_to_dict(manual: ParsedManual) -> dict:
    source_path = manual.source_path
    try:
        source_path = source_path.relative_to(PROJECT_ROOT)
    except ValueError:
        pass

    return {
        "source_path": str(source_path),
        "page_count": manual.page_count,
        "blocks": [
            {
                "text": block.text,
                "page_number": block.page_number,
                "block_index": block.block_index,
                "font_size": block.font_size,
            }
            for block in manual.blocks
        ],
    }


def save_parsed_manual(pdf_path: Path) -> tuple[ParsedManual, Path]:
    manual = parse_pdf(pdf_path)
    output_path = parsed_manual_path(pdf_path)
    output_path.write_text(
        json.dumps(manual_to_dict(manual), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manual, output_path.resolve()
