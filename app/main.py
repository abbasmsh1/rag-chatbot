"""FastAPI endpoints for the RAG chatbot."""
import io
import os

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .rag import RagStore

app = FastAPI(title="rag-chatbot")
_store = None

_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html")


@app.get("/", response_class=HTMLResponse)
def ui():
    with open(_WEB, encoding="utf-8") as f:
        return f.read()


def store():
    global _store
    if _store is None:
        _store = RagStore()
    return _store


class Question(BaseModel):
    question: str
    k: int = 4


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile):
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
    n = store().ingest(text, source=name)
    return {"source": name, "chunks_indexed": n}


@app.post("/ask")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(400, "question is empty")
    return store().ask(q.question, k=q.k)
