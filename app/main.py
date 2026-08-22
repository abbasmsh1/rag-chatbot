"""FastAPI endpoints for the production RAG chatbot."""
import hmac
import io
import json
import os

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .chunking import chunk_document
from .registry import Registry, content_hash, doc_id_for

app = FastAPI(title="rag-chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# lazy singletons so tests can inject fakes and importing never needs API keys
_store = None
_registry = None


def get_store():
    global _store
    if _store is None:
        from .store import PineconeStore

        _store = PineconeStore()
    return _store


def get_registry():
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def check_token(x_demo_token):
    """If DEMO_TOKEN is set, require it - keeps a public demo from burning API keys."""
    required = os.environ.get("DEMO_TOKEN")
    if required and not hmac.compare_digest(x_demo_token or "", required):
        raise HTTPException(401, "missing or wrong X-Demo-Token header")


class Question(BaseModel):
    question: str
    k: int = 5
    alpha: float = 0.7  # 1 = pure semantic, 0 = pure keyword
    rerank: bool = True


def build_prompt(question, contexts):
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        "Answer the question using ONLY the context below. "
        "Cite chunk numbers like [1]. If the context does not contain the answer, "
        "say you don't know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )


def extract_text(raw, name):
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return raw.decode("utf-8", errors="replace")


def ingest_text(text, source, store, registry):
    """Chunk + embed + upsert one document. Returns chunks indexed (0 = unchanged)."""
    doc_id = doc_id_for(source)
    chash = content_hash(text)
    existing = registry.get_document(doc_id)
    if existing and existing["content_hash"] == chash:
        return doc_id, 0
    ns = registry.active_namespace
    if existing:  # content changed: drop stale vectors first
        store.delete_document(doc_id, existing["chunk_count"], ns)
    chunks = chunk_document(text, source)
    if not chunks:
        raise HTTPException(400, "no extractable text")
    n = store.upsert_chunks(doc_id, source, chunks, ns)
    registry.save_text(doc_id, text)
    registry.upsert_document(doc_id, source, chash, n, chunks[0]["profile"])
    return doc_id, n


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile, x_demo_token: str | None = Header(default=None)):
    check_token(x_demo_token)
    raw = await file.read()
    name = file.filename or "upload"
    text = extract_text(raw, name)
    if not text.strip():
        raise HTTPException(400, "no extractable text in file")
    doc_id, n = ingest_text(text, name, get_store(), get_registry())
    return {"id": doc_id, "source": name, "chunks_indexed": n, "unchanged": n == 0}


@app.post("/ask")
def ask(q: Question, x_demo_token: str | None = Header(default=None)):
    check_token(x_demo_token)
    if not q.question.strip():
        raise HTTPException(400, "question is empty")
    registry = get_registry()
    hits = get_store().query(
        q.question, registry.active_namespace, k=q.k, alpha=q.alpha, rerank=q.rerank
    )

    def stream():
        if not hits:
            yield _sse("token", {"text": "I don't know - nothing indexed covers this."})
            yield _sse("sources", {"sources": []})
            return
        import anthropic

        client = anthropic.Anthropic()
        with client.messages.stream(
            model=os.environ.get("ANSWER_MODEL", "claude-opus-5"),
            max_tokens=4096,
            messages=[{"role": "user", "content": build_prompt(q.question, [h["text"] for h in hits])}],
        ) as resp:
            for text in resp.text_stream:
                yield _sse("token", {"text": text})
        yield _sse("sources", {"sources": hits})

    return StreamingResponse(stream(), media_type="text/event-stream")


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/documents")
def documents(page: int = 1, q: str = ""):
    return get_registry().list_documents(page=page, q=q)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, x_demo_token: str | None = Header(default=None)):
    check_token(x_demo_token)
    registry = get_registry()
    doc = registry.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "unknown document")
    get_store().delete_document(doc_id, doc["chunk_count"], registry.active_namespace)
    registry.delete_document(doc_id)
    return {"deleted": doc_id}


def run_reindex(store, registry):
    """Re-chunk + re-embed everything into a fresh namespace, then swap."""
    new_ns = registry.next_namespace()
    old_ns = registry.active_namespace
    ids = registry.all_doc_ids()
    registry.set_meta("reindex_total", len(ids))
    registry.set_meta("reindex_done", 0)
    registry.set_meta("reindex_status", "running")
    try:
        for i, doc_id in enumerate(ids, 1):
            doc = registry.get_document(doc_id)
            text = registry.load_text(doc_id)
            chunks = chunk_document(text, doc["source"])
            n = store.upsert_chunks(doc_id, doc["source"], chunks, new_ns)
            registry.upsert_document(
                doc_id, doc["source"], doc["content_hash"], n, chunks[0]["profile"]
            )
            registry.set_meta("reindex_done", i)
        registry.set_meta("active_namespace", new_ns)  # atomic swap: queries move over here
        store.delete_namespace(old_ns)
        registry.set_meta("reindex_status", "done")
    except Exception as e:  # old namespace untouched, still serving
        registry.set_meta("reindex_status", f"failed: {e}")


@app.post("/reindex")
def reindex(background: BackgroundTasks, x_demo_token: str | None = Header(default=None)):
    check_token(x_demo_token)
    registry = get_registry()
    if registry.get_meta("reindex_status") == "running":
        raise HTTPException(409, "reindex already running")
    # ponytail: in-process background task; move to a worker queue if reindex
    # time ever exceeds the deploy/restart cadence
    background.add_task(run_reindex, get_store(), registry)
    return {"started": True, "target_namespace": registry.next_namespace()}


@app.get("/stats")
def stats():
    registry = get_registry()
    counts = registry.counts()
    try:
        vec = get_store().stats()
    except Exception:
        vec = {"total_vectors": None, "namespaces": {}}
    return {
        **counts,
        **vec,
        "active_namespace": registry.active_namespace,
        "reindex": {
            "status": registry.get_meta("reindex_status", "idle"),
            "done": int(registry.get_meta("reindex_done", 0) or 0),
            "total": int(registry.get_meta("reindex_total", 0) or 0),
        },
    }
