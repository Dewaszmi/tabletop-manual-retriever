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

## API server

```bash
pip install -r requirements.txt
PYTHONPATH=src uvicorn tabletop_manual_retriever.main:app --reload
```

Parse a PDF:

```bash
curl -F "file=@data/uploads/your-manual.pdf" http://127.0.0.1:8000/parse-pdf
```

Docker:

```bash
docker build -t tabletop-manual-retriever .
docker run --rm -p 8000:8000 tabletop-manual-retriever
```

Docker Compose:

```bash
docker compose up --build -d
docker compose down
```

Services:

```text
API:    http://127.0.0.1:8000
Qdrant: http://127.0.0.1:6333
```

## Project layout

```
src/tabletop_manual_retriever/
  main.py        # FastAPI app
  ingest/
    models.py    # ParsedManual, TextBlock
    parser.py    # PDF text extraction
    router.py    # parse-pdf endpoint
  cli.py         # parse-manual command
tests/
data/uploads/    # place test PDFs here
```

## Tests

```bash
pytest
```
