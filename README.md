# tabletop-manual-retriever

LLM-powered retrieval from tabletop game manuals.

Upload rulebook PDFs, index them into a vector store, then ask natural-language questions grounded in the manual text.

## Quick start (Docker Compose)

Recommended way to run everything: API, Qdrant, and Ollama.

**1. Pull the LLM model** (once, on the host — Compose reuses `~/.ollama`):

```bash
ollama pull llama3.2
```

**2. Start the stack:**

```bash
docker compose up --build -d
```

**3. Open the web UI:**

```text
http://127.0.0.1:8000/
```

**4. Upload a manual** for a game (e.g. `monopoly`), click **Ingest** on the PDF, then use **Ask the rules** to query it.

Or use the API:

```bash
# Upload
curl -F "file=@/path/to/rules.pdf" http://127.0.0.1:8000/games/monopoly/manuals

# Index into Qdrant
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"game_slug":"monopoly"}'

# Ask a question (retrieval + LLM)
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"game_slug":"monopoly","question":"How do I win?"}' | jq '{answer_mode, answer}'
```

The first LLM query may take 30–60 seconds while the model loads.

**Stop:**

```bash
docker compose down
```

### Services

| Service      | URL                    |
| ------------ | ---------------------- |
| Web UI / API | http://127.0.0.1:8000/ |
| Qdrant       | http://127.0.0.1:6333  |

Compose runs Ollama internally at `http://ollama:11434` (not exposed on the host). It mounts `${HOME}/.ollama` so models you already pulled with the host `ollama` CLI are reused.

### Environment variables

Defaults are set in `docker-compose.yaml`. Common overrides:

| Variable              | Default                                                   | Purpose                                         |
| --------------------- | --------------------------------------------------------- | ----------------------------------------------- |
| `GEMINI_API_KEY`      | required                                                  | Google AI Studio API key used by Docker Compose |
| `LLM_MODEL`           | `gemini-2.5-flash`                                        | OpenAI-compatible chat model                    |
| `LLM_BASE_URL`        | `https://generativelanguage.googleapis.com/v1beta/openai` | OpenAI-compatible chat API                      |
| `LLM_TIMEOUT_SECONDS` | `120`                                                     | LLM request timeout                             |
| `QDRANT_COLLECTION`   | `manual_chunks`                                           | Vector collection name                          |
| `RAG_TOP_K`           | `10`                                                      | Number of chunks retrieved per query            |

If `LLM_BASE_URL` / `LLM_MODEL` are unset, `/query` falls back to returning retrieved excerpts (`answer_mode: "excerpt"`).

## Local development (without Docker)

**Setup:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -r requirements.txt
```

**Run Qdrant** (required for ingest/query):

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

**Run Ollama** (required for LLM answers):

```bash
ollama pull llama3.2
ollama serve
```

**Start the API:**

```bash
export LLM_BASE_URL=http://127.0.0.1:11434/v1
export LLM_MODEL=llama3.2:latest
PYTHONPATH=src uvicorn tabletop_manual_retriever.main:app --reload
```

Then open http://127.0.0.1:8000/

## CLI: parse a manual

Drop a PDF into `data/uploads/<game-slug>/`, then:

```bash
parse-manual data/uploads/catan/rules.pdf
parse-manual data/uploads/catan/rules.pdf --page 3
parse-manual data/uploads/catan/rules.pdf --json
```

## API overview

Upload manuals:

```bash
curl -F "file=@/path/to/rules.pdf" http://127.0.0.1:8000/games/catan/manuals
curl -F "files=@/path/to/base-rules.pdf" -F "files=@/path/to/seafarers.pdf" http://127.0.0.1:8000/games/catan/manuals
```

Replace an existing file:

```bash
curl -F "file=@/path/to/rules.pdf" "http://127.0.0.1:8000/games/catan/manuals?overwrite=true"
```

List library:

```bash
curl http://127.0.0.1:8000/games/library
```

Ingest into Qdrant:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"game_slug":"catan"}'
```

Query with RAG:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"game_slug":"catan","question":"How many cards do I draw?"}'
```

## Project layout

```
src/tabletop_manual_retriever/
  main.py        # FastAPI app
  config.py      # env config
  ingest/        # parse, chunk, embed, upsert to Qdrant
  rag/           # retrieve chunks + LLM answer
  storage/       # save/list uploaded PDFs
  upload/        # upload/list game manuals
  web/           # web UI
  cli.py         # parse-manual command
tests/
data/uploads/    # uploaded manuals, one folder per game
```

## Tests

```bash
pytest
```
