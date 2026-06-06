import argparse
import json
import sys
from pathlib import Path

from tabletop_manual_retriever.ingest import parse_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text and page numbers from a rulebook PDF."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF manual")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of a human-readable summary",
    )
    parser.add_argument(
        "--page",
        type=int,
        metavar="N",
        help="Show text for a single page only",
    )
    args = parser.parse_args()

    try:
        manual = parse_pdf(args.pdf)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc

    if args.json:
        payload = {
            "source_path": str(manual.source_path),
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
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"Source: {manual.source_path}")
    print(f"Pages:  {manual.page_count}")
    print(f"Blocks: {len(manual.blocks)}")
    print()

    pages = (
        [(args.page, manual.text_for_page(args.page))]
        if args.page is not None
        else manual.pages()
    )

    if args.page is not None and not pages[0][1].strip():
        print(f"No text found on page {args.page}.", file=sys.stderr)
        raise SystemExit(1)

    for page_number, text in pages:
        print(f"--- Page {page_number} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
