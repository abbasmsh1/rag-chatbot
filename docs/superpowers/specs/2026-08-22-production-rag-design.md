# Production RAG Chatbot — Design Spec (2026-08-22)

## Goal

Rebuild the demo RAG chatbot into a system that handles 1M+ documents in
production: Qdrant vector store (keyless: embedded or self-hosted), hybrid
(dense + sparse) indexing, variable chunking, zero-downtime reindexing, and a
full web console frontend. Revised 2026-08-22: Pinecone replaced with Qdrant
per user request (no vector-DB API key), answer LLM switched to Claude.

## Non-goals

- Multi-tenant auth (existing DEMO_TOKEN gate is kept as-is).
- Distributed task queue (Celery/Kafka). Bulk ingest is a checkpointed CLI
  script; in-app reindex uses FastAPI BackgroundTasks. Upgrade path noted in
  code where it matters.
- Cross-encoder rerank stage (RRF-style weighted fusion only; noted in code).

## Architecture

```
                       +------------------ frontend/ (Next.js) ------------------+
                       |  /chat        /documents        /index                  |
                       +----------------------|----------------------------------+
                                              | HTTP (JSON + SSE)
+---------------------------------------- app/ (FastAPI) -----------------------+
| main.py      endpoints: /ask /ingest /documents /reindex /stats /health       |
| chunking.py  variable chunking (profiles + adaptive size)                     |
| store.py     Qdrant hybrid store (embed, upsert, query, fuse, namespaces)     |
| registry.py  SQLite document registry (hash, chunk count, namespace version)  |
+--------------------------------------------------------------------------------+
        |                                   |
   Claude (claude-opus-5)          Qdrant (embedded local, or QDRANT_URL server)
                                      - named vectors: dense + sparse
                                      - dense: BAAI/bge-small-en-v1.5 (384d, fastembed)
                                      - sparse: Qdrant/bm25 + server-side IDF
                                      - fusion: alpha-weighted, max-normalized
                                      - collections: rag_v1, rag_v2, ... (reindex)
```

## Components

### Variable chunking (`app/chunking.py`)

Token-aware recursive splitter, no LangChain dependency. Content profile is
detected per document (markdown / code / prose) and sets separators and target
chunk size; chunk size additionally adapts to document length (short docs get
one chunk, very long docs get larger chunks to bound chunk count). Overlap is
a fraction of chunk size. Output: list of `{text, ordinal}`.

### Hybrid store (`app/store.py`)

- Qdrant, no API key: embedded local mode by default (`data/qdrant`), or a
  self-hosted server via `QDRANT_URL` for production scale.
- One collection per namespace version (`rag_v1`, ...), named vectors:
  `dense` (cosine) + `sparse` (BM25 with server-side IDF modifier), created on
  first use.
- Embeddings local via fastembed (ONNX, no key): dense
  `BAAI/bge-small-en-v1.5` (384d), sparse `Qdrant/bm25`. IDF lives in Qdrant,
  so incremental ingest at 1M-doc scale never needs corpus re-fitting.
- Upserts batched (256 points/batch). Point ids: UUID5 of `{doc_id}#{ordinal}`;
  deletes filter on the `doc_id` payload field.
- Query: dense and sparse searched separately (4x oversample), fused with an
  alpha-weighted convex combination of max-normalized scores.
- All reads/writes target the active namespace from the registry.

### Registry + reindexing (`app/registry.py`)

SQLite (stdlib) at `data/registry.db`:

- `documents(id, source, content_hash, chunk_count, chunk_profile, created_at, updated_at)`
- `meta(key, value)` — holds `active_namespace` and reindex state.

Content hash means re-ingesting an unchanged document is a no-op; a changed
document deletes its old vectors (by id prefix) and upserts new ones.

Reindex = re-chunk + re-embed every registered document into namespace
`v{n+1}`, then atomically swap `active_namespace`, then delete the old
namespace. Queries keep hitting the old namespace until the swap, so
reindexing is zero-downtime. Progress (done/total) is stored in `meta` and
exposed via `/stats`. Requires original document text, so ingest stores raw
text in `data/docs/{id}.txt`.

### API (`app/main.py`)

- `POST /ask` — body `{question, k, alpha}`; hybrid retrieve + fuse, answer
  via claude-opus-5 streamed as SSE; final event carries citations
  (source, score, excerpt).
- `POST /ingest` — multipart file (pdf/txt/md); chunk, embed, upsert, register.
- `GET /documents?page=&q=` — paginated registry listing.
- `DELETE /documents/{id}` — remove vectors + registry row + stored text.
- `POST /reindex` — start background reindex; `GET /stats` reports progress,
  vector counts, active namespace.
- `GET /health`.
- DEMO_TOKEN header gate retained on mutating + LLM endpoints.

### Bulk ingest (`scripts/bulk_ingest.py`)

CLI for the 1M-document path: walks a directory or JSONL file, concurrent
workers (thread pool over embed+upsert), checkpointed through the registry
(rerun skips already-ingested hashes), exponential backoff on API errors.

### Frontend (`frontend/`, Next.js App Router)

Tailwind + shadcn-style components in the 21st.dev aesthetic; visual comps
generated with Stitch (dark, Space Grotesk/Inter, mint accent) and committed
as reference. Routes:

- `/chat` — streaming answers, inline citation chips, expandable source cards
  with score bars; composer with top-k and alpha slider.
- `/documents` — dropzone upload, paginated searchable table, delete.
- `/index` — vector/document counts, active namespace, reindex trigger with
  live progress.

`NEXT_PUBLIC_API_URL` points at the FastAPI service.

## Error handling

- Embed/upsert failures: retry w/ exponential backoff (3 attempts) then fail
  the document, never the batch; bulk ingester logs and continues.
- `/ask` with empty index or below-threshold scores answers "I don't know"
  (kept from existing code).
- Reindex failure leaves `active_namespace` untouched (old data keeps serving).

## Testing

Offline unit tests (no network): chunking profiles/adaptivity, hybrid score
normalization, registry CRUD + namespace swap, prompt building, API endpoints
with a fake store. Existing test file replaced.

## Env

`ANTHROPIC_API_KEY` (answers only), optional `QDRANT_URL` (unset = embedded
local mode), `QDRANT_COLLECTION` (default `rag`), optional `DEMO_TOKEN`.
