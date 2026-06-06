# tabletop-manual-retriever

LLM-powered system for fast retrieval of information from tabletop game manuals.

Users provide their own rulebook PDFs. The pipeline starts with structured text extraction, then will add chunking, embeddings, and RAG query.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Parse a manual

Drop a PDF into `data/uploads/`, then run:

```bash
parse-manual data/uploads/your-manual.pdf
```

Show a single page:

```bash
parse-manual data/uploads/your-manual.pdf --page 3
```

Structured JSON output:

```bash
parse-manual data/uploads/your-manual.pdf --json
```

## Project layout

```
src/tabletop_manual_retriever/
  ingest/
    models.py    # ParsedManual, TextBlock
    parser.py    # PDF text extraction
  cli.py         # parse-manual command
tests/
data/uploads/    # place test PDFs here
```

## Tests

```bash
pytest
```
