# tabletop-manual-retriever

Upload tabletop rulebook PDFs, index them into a vector store, and ask natural-language rules questions grounded in the manual text.

The web UI covers the full workflow: upload manuals, ingest them into Qdrant, and chat with retrieved passages plus an LLM answer. Conversations are saved and can be resumed later.

## Features

- **PDF library** — upload one or many rulebooks per game (`/games/{game_slug}/manuals`)
- **Ingestion** — parse, chunk, embed, and upsert into [Qdrant](https://qdrant.tech/)
- **Hybrid retrieval** — dense vector search + BM25 keyword search, fused with reciprocal rank fusion (RRF)
- **Reranking** — cross-encoder rescores candidates before the top passages are sent to the LLM
- **Grounded answers** — any OpenAI-compatible chat API (Gemini by default); falls back to raw excerpts when no LLM is configured
- **Chat history** — multi-turn conversations stored in SQLite, with source citations per answer

## Quick start (Docker Compose)

**1. Configure environment**

```bash
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` (free key from [Google AI Studio](https://aistudio.google.com/apikey)). See [`.env.example`](.env.example) for all options.

**2. Start the stack**

```bash
docker compose up --build -d
```

This starts the API, Qdrant, and an optional Ollama service (only needed if you switch the LLM settings to a local model).

**3. Open the web UI**

```text
http://127.0.0.1:8000/
```

**4. Use the app**

1. **Upload** — add PDF rulebooks for a game (e.g. `catan`, `monopoly`)
2. **Library** — click **Ingest** on each manual (or ingest all) to index chunks into Qdrant
3. **Ask the rules** — pick a game and ask questions; expand **Sources** on each answer to see ranked passages

The first query after a fresh install may take a minute while embedding and reranker models download into the Docker volume cache.

**Stop:**

```bash
docker compose down
```

### Services

| Service      | URL                    | Notes                                      |
| ------------ | ---------------------- | ------------------------------------------ |
| Web UI / API | http://127.0.0.1:8000/ | FastAPI + static frontend                  |
| Qdrant       | http://127.0.0.1:6333  | Vector database                            |
| Ollama       | internal only          | Optional local LLM; not exposed on host    |

Persistent data:

- `./data/` — uploaded PDFs, parsed JSON, chat database
- Docker volume `qdrant_data` — vector index
- Docker volume `hf_cache` — Hugging Face / SentenceTransformers model cache

## How retrieval works

For each question the pipeline:

1. **Embeds** the question (plus recent conversation context) with `BAAI/bge-small-en-v1.5`
2. **Searches** Qdrant for the closest chunk vectors
3. **Optionally merges** BM25 keyword hits with vector hits via RRF (`RAG_HYBRID_ENABLED`)
4. **Reranks** the candidate pool with a cross-encoder (`RAG_RERANK_MODEL`)
5. **Returns** the top `RAG_TOP_K` passages, sorted by rerank score (highest first)
6. **Generates** an answer from the LLM using numbered source citations `[1]`, `[2]`, …

Source scores are reranker outputs. With `BAAI/bge-reranker-base` they are typically in the 0–1 range. Older MiniLM rerankers emit raw logits that can be negative; higher is always better.

The LLM may cite `[3]` before `[1]` in prose if that passage fits the sentence better — the **Sources** list is always ordered by score.

### Using Ollama instead of Gemini

Ollama is included in Compose but not used unless you point the LLM settings at it:

```env
LLM_BASE_URL=http://ollama:11434/v1
LLM_MODEL=llama3.2:latest
GEMINI_API_KEY=ollama
```

Pull the model once on the host (Compose mounts `~/.ollama`):

```bash
ollama pull llama3.2
```

## Local development (without Docker)

**Setup**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -r requirements.txt
cp .env.example .env
# edit .env — set QDRANT_URL=http://127.0.0.1:6333 and LLM_* as needed
```

**Run Qdrant** (required for ingest and query):

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

**Start the API**

```bash
set -a && source .env && set +a
PYTHONPATH=src uvicorn tabletop_manual_retriever.main:app --reload
```

Open http://127.0.0.1:8000/

Note: local runs use Python defaults from `config.py` unless you export the `RAG_*` variables. Hybrid search and reranking are **off** by default outside Docker unless you set `RAG_HYBRID_ENABLED=true` and `RAG_RERANK_ENABLED=true`.


## CLI: parse a manual

Drop a PDF into `data/uploads/<game-slug>/`, or point at any file:

```bash
parse-manual data/uploads/catan/rules.pdf
parse-manual data/uploads/catan/rules.pdf --page 3
parse-manual data/uploads/catan/rules.pdf --json
```

