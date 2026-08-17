"""FastAPI endpoints for the RAG chatbot."""
import io

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from .rag import RagStore

app = FastAPI(title="rag-chatbot")
_store = None


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
