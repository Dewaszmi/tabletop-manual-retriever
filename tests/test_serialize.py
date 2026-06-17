from pathlib import Path

import fitz
import pytest

from tabletop_manual_retriever.ingest.serialize import (
    manual_to_dict,
    parsed_manual_path,
    save_parsed_manual,
)


def _make_sample_manual(path: Path) -> None:
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), "Setup\nPlace the board.", fontsize=14)
        document.save(path)
    finally:
        document.close()


def test_save_parsed_manual_writes_json_next_to_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "rules.pdf"
    _make_sample_manual(pdf_path)

    manual, json_path = save_parsed_manual(pdf_path)

    assert json_path == parsed_manual_path(pdf_path)
    assert json_path.is_file()
    assert manual.page_count == 1
    assert json_path.read_text(encoding="utf-8").strip().startswith("{")


def test_manual_to_dict_includes_blocks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "rules.pdf"
    _make_sample_manual(pdf_path)
    manual, _ = save_parsed_manual(pdf_path)

    payload = manual_to_dict(manual)

    assert payload["page_count"] == 1
    assert payload["blocks"]
    assert payload["blocks"][0]["text"]
