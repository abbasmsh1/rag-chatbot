"""FastAPI endpoints for the RAG chatbot."""
import io
import os
import re
from collections import OrderedDict

from fastapi import FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .rag import RagStore

app = FastAPI(title="rag-chatbot")

_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html")

# one store per browser session so users on a shared deployment never see
# each other's documents; LRU-capped to bound memory
MAX_SESSIONS = 20
_stores = OrderedDict()
_SID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def store_for(sid):
    if not sid or not _SID_RE.match(sid):
        raise HTTPException(400, "missing or invalid X-Session-Id header")
    if sid in _stores:
        _stores.move_to_end(sid)
    else:
        _stores[sid] = RagStore()
        while len(_stores) > MAX_SESSIONS:
            _stores.popitem(last=False)
    return _stores[sid]


def check_token(x_demo_token):
    """If DEMO_TOKEN is set, require it - keeps a public demo from burning your API key."""
    required = os.environ.get("DEMO_TOKEN")
    if required and x_demo_token != required:
        raise HTTPException(401, "missing or wrong X-Demo-Token header")


class Question(BaseModel):
    question: str
    k: int = 4


@app.get("/", response_class=HTMLResponse)
def ui():
    with open(_WEB, encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(
    file: UploadFile,
    x_session_id: str | None = Header(default=None),
    x_demo_token: str | None = Header(default=None),
):
    check_token(x_demo_token)
    raw = await file.read()
    name = file.filename or "upload"
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        raise HTTPException(400, "no extractable text in file")
    n = store_for(x_session_id).ingest(text, source=name)
    return {"source": name, "chunks_indexed": n}


@app.post("/ask")
def ask(
    q: Question,
    x_session_id: str | None = Header(default=None),
    x_demo_token: str | None = Header(default=None),
):
    check_token(x_demo_token)
    if not q.question.strip():
        raise HTTPException(400, "question is empty")
    return store_for(x_session_id).ask(q.question, k=q.k)
