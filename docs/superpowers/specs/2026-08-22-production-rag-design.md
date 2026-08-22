# Production RAG Chatbot — Design Spec (2026-08-22)

## Goal

Rebuild the demo RAG chatbot into a system that handles 1M+ documents in
production: Pinecone vector store, hybrid (dense + sparse) indexing, variable
chunking, zero-downtime reindexing, and a full web console frontend.

## Non-goals

- Multi-tenant auth (existing DEMO_TOKEN gate is kept as-is).
- Distributed task queue (Celery/Kafka). Bulk ingest is a checkpointed CLI
  script; in-app reindex uses FastAPI BackgroundTasks. Upgrade path noted in
  code where it matters.
- Self-hosted embeddings.

## Architecture

```
                       +------------------ frontend/ (Next.js) ------------------+
                       |  /chat        /documents        /index                  |
                       +----------------------|----------------------------------+
                                              | HTTP (JSON + SSE)
+---------------------------------------- app/ (FastAPI) -----------------------+
| main.py      endpoints: /ask /ingest /documents /reindex /stats /health       |
| chunking.py  variable chunking (profiles + adaptive size)                     |
| store.py     Pinecone hybrid store (embed, upsert, query, rerank, namespaces) |
| registry.py  SQLite document registry (hash, chunk count, namespace version)  |
+--------------------------------------------------------------------------------+
        |                                   |
   Claude (claude-opus-5)          Pinecone serverless index
                                      - metric: dotproduct
                                      - dense: llama-text-embed-v2 (1024d)
                                      - sparse: pinecone-sparse-english-v0
                                      - rerank: bge-reranker-v2-m3
                                      - namespaces: v1, v2, ... (reindex versions)
```

## Components

### Variable chunking (`app/chunking.py`)

Token-aware recursive splitter, no LangChain dependency. Content profile is
detected per document (markdown / code / prose) and sets separators and target
chunk size; chunk size additionally adapts to document length (short docs get
one chunk, very long docs get larger chunks to bound chunk count). Overlap is
a fraction of chunk size. Output: list of `{text, ordinal}`.

### Hybrid store (`app/store.py`)

- One serverless Pinecone index, `metric=dotproduct`, created on first use.
- Embeddings via `pc.inference.embed`: dense `llama-text-embed-v2`,
  sparse `pinecone-sparse-english-v0`. The sparse model is stateless — unlike
  BM25 it needs no corpus fitting, so incremental ingest at 1M-doc scale never
  requires re-fitting statistics.
- Upserts batched (100 vectors/batch). Vector ids: `{doc_id}#{ordinal}`.
- Query: alpha-weighted convex combination of dense and sparse query vectors
  (Pinecone-documented `hybrid_score_norm`), `top_k` oversampled, then
  `pc.inference.rerank` down to final k when rerank is enabled.
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

- `POST /ask` — body `{question, k, alpha, rerank}`; hybrid retrieve, rerank,
  answer via claude-opus-5 streamed as SSE; final event carries citations
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
  with score bars; composer with top-k, alpha slider, rerank toggle.
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

`PINECONE_API_KEY`, `ANTHROPIC_API_KEY`, optional `DEMO_TOKEN`,
`PINECONE_INDEX` (default `rag-chatbot`), `PINECONE_CLOUD`/`PINECONE_REGION`
(default aws/us-east-1).
