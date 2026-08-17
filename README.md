# rag-chatbot

A chatbot that answers from **your** documents, with source citations. LangChain + ChromaDB + FastAPI.

Reference implementation behind my [Fiverr RAG gig](https://www.fiverr.com/abbasmsh1): ingest PDFs or text, ask questions over HTTP, get grounded answers with the exact source chunks that produced them.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY

uvicorn app.main:app --reload
```

```bash
# ingest a document
curl -F "file=@docs/handbook.pdf" http://localhost:8000/ingest

# ask
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

Response:

```json
{
  "answer": "Refunds are accepted within 30 days...",
  "sources": [{"source": "handbook.pdf", "chunk": "...", "score": 0.87}]
}
```

## How it works

```
PDF / TXT  →  chunker  →  embeddings  →  ChromaDB
                                            ↓
question  →  retriever (top-k)  →  LLM  →  answer + citations
```

- Chunking with overlap so answers never lose context at boundaries
- Every answer cites the retrieved chunks; no chunk retrieved above threshold means the bot says "I don't know" instead of hallucinating
- Swap the LLM (OpenAI, Anthropic, Ollama) or store (FAISS, Pinecone) in `app/rag.py`

## Layout

```
app/main.py   # FastAPI endpoints: /ingest, /ask, /health
app/rag.py    # chunking, embedding, retrieval, answer generation
tests/        # chunker + prompt unit tests (offline)
Dockerfile
```

## Author

Abbas Mustafa — published NLP research (IEEE FIT 2022), MSc Cybersecurity @ IMT Atlantique.
Hire me on [Fiverr](https://www.fiverr.com/abbasmsh1).
