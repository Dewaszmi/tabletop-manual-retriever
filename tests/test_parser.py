from pathlib import Path

import fitz
import pytest

from tabletop_manual_retriever.ingest import parse_pdf
from tabletop_manual_retriever.ingest.models import ParsedManual


def _make_sample_manual(path: Path) -> None:
    document = fitz.open()
    try:
        page_one = document.new_page()
        page_one.insert_text(
            (72, 72),
            "Setup\nPlace the board in the center of the table.",
            fontsize=14,
        )
        page_one.insert_text(
            (72, 120),
            "Each player takes five cards.",
            fontsize=11,
        )

        page_two = document.new_page()
        page_two.insert_text(
            (72, 72),
            "Victory\nThe first player to reach 10 points wins.",
            fontsize=14,
        )

        document.save(path)
    finally:
        document.close()


def test_parse_pdf_extracts_text_with_page_numbers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample-rules.pdf"
    _make_sample_manual(pdf_path)

    manual = parse_pdf(pdf_path)

    assert isinstance(manual, ParsedManual)
    assert manual.page_count == 2
    assert manual.source_path == pdf_path.resolve()
    assert len(manual.blocks) >= 3

    page_numbers = {block.page_number for block in manual.blocks}
    assert page_numbers == {1, 2}

    page_one_text = manual.text_for_page(1)
    assert "Setup" in page_one_text
    assert "five cards" in page_one_text

    page_two_text = manual.text_for_page(2)
    assert "Victory" in page_two_text
    assert "10 points" in page_two_text


def test_parse_pdf_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / "missing.pdf")


def test_parse_pdf_raises_for_non_pdf(tmp_path: Path) -> None:
    text_path = tmp_path / "notes.txt"
    text_path.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a .pdf file"):
        parse_pdf(text_path)
