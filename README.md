# rag-chatbot

A production-scale RAG chatbot: hybrid (dense + sparse) retrieval over 1M+
documents, variable chunking, zero-downtime reindexing, grounded answers with
citations, and a full web console.

No vector-DB API key needed: Qdrant runs embedded or self-hosted, and
embeddings run locally via fastembed. The only key is `ANTHROPIC_API_KEY` for
answer generation (Claude).

## Stack

- **Store**: Qdrant - one collection per index version, named `dense` +
  `sparse` vectors
- **Embeddings** (local, ONNX): `BAAI/bge-small-en-v1.5` dense +
  `Qdrant/bm25` sparse with server-side IDF
- **Retrieval**: dense and sparse searched separately, fused with an
  alpha-weighted combination of max-normalized scores
  (alpha 1 = semantic, 0 = keyword)
- **Chunking**: content-aware (markdown / code / prose separators) and
  length-adaptive chunk sizes
- **Answers**: Claude (`claude-opus-5`) streamed over SSE, cited chunk by chunk
- **Registry**: SQLite tracks every document's content hash, chunk count, and
  the active index version
- **API**: FastAPI. **Console**: Next.js + Tailwind (`frontend/`)

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env               # add ANTHROPIC_API_KEY
uvicorn app.main:app --reload      # API on :8000

cd frontend && npm install && npm run dev   # console on :3000
```

First ingest downloads the embedding models (~130MB, one time). Without
`QDRANT_URL`, Qdrant runs embedded on disk under `data/` - fine for dev and
small corpora. For production scale run a server:

```bash
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
# then set QDRANT_URL=http://localhost:6333
```

## The 1M-document path

```bash
python scripts/bulk_ingest.py ./corpus_dir --workers 8
python scripts/bulk_ingest.py ./docs.jsonl --workers 8   # {"source":..., "text":...} per line
```

Checkpointed through the registry: unchanged documents (by content hash) are
skipped, so a crashed or repeated run resumes where it left off. Changed
documents replace their old vectors.

## Reindexing (zero downtime)

`POST /reindex` (or the Index page in the console) re-chunks and re-embeds
every registered document into a fresh collection (`rag_v2`, ...), then swaps
the active pointer atomically and drops the old collection. Queries keep
serving from the old version until the swap. Progress streams to `/stats`.

## API

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{question, k, alpha}` - SSE stream: `token` events then a `sources` event |
| `POST /ingest` | multipart PDF/TXT/MD upload; dedupes by content hash |
| `GET /documents?page=&q=` | paginated registry listing |
| `DELETE /documents/{id}` | remove from index + registry |
| `POST /reindex` | start background reindex |
| `GET /stats` | counts, namespaces, reindex progress |
| `GET /health` | liveness |

Set `DEMO_TOKEN` to require an `X-Demo-Token` header on mutating/LLM
endpoints (the console stores it in localStorage under `demo_token`).

## Console (`frontend/`)

Dark developer-console UI (Space Grotesk / Inter, mint accent - design system
authored in Stitch, components in the 21st.dev shadcn-format style):

- **Chat** - streaming answers, expandable citation cards with relevance
  bars, top-k and keyword-semantic alpha controls
- **Documents** - drag-and-drop ingest, searchable paginated table, delete
- **Index** - live stats, namespace list, one-click zero-downtime reindex
  with progress

## Layout

```
app/main.py          # FastAPI endpoints + Claude answer streaming
app/chunking.py      # variable chunking (profiles + adaptive sizing)
app/store.py         # Qdrant hybrid store (embed, upsert, query, fuse)
app/registry.py      # SQLite registry (hashes, versions, reindex state)
scripts/bulk_ingest.py  # checkpointed concurrent bulk loader
frontend/            # Next.js console
tests/               # offline unit tests (no network, no keys)
docs/superpowers/specs/  # design spec
```

## Tests

```bash
pytest tests/ -q
```

## Author

Abbas Mustafa — published NLP research (IEEE FIT 2022), MSc Cybersecurity @ IMT Atlantique.
Hire me on [Fiverr](https://www.fiverr.com/abbasmsh1).
