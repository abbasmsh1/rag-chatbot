# rag-chatbot

A chatbot that answers from **your** documents, with source citations. LangChain + ChromaDB + FastAPI.

Reference implementation behind my [Fiverr RAG gig](https://www.fiverr.com/abbasmsh1): ingest PDFs or text, ask questions over HTTP, get grounded answers with the exact source chunks that produced them.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY

uvicorn app.main:app --reload
```

Open http://localhost:8000 for the chat UI: upload a document, ask questions, expand the cited source chunks under each answer.

## Deploy to Vercel

```bash
vercel --prod
```

Set `OPENAI_API_KEY` in the Vercel project settings. The vector store is in-memory (numpy cosine) and partitioned per browser session, so users on a shared deployment never see each other's documents; ingested content lives for the lifetime of the function instance. For a persistent corpus, swap `RagStore` for a hosted vector DB. Optional: set `DEMO_TOKEN` to require an `X-Demo-Token` header, protecting your OpenAI credits on a public URL.

```bash
# ingest a document
curl -H "X-Session-Id: my-session-1" -F "file=@docs/handbook.pdf" http://localhost:8000/ingest

# ask
curl -X POST http://localhost:8000/ask \
  -H "X-Session-Id: my-session-1" \
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
- Swap the LLM (OpenAI, Anthropic, Ollama) or store (FAISS, Pinecone, Chroma) in `app/rag.py`

## Layout

```
app/main.py    # FastAPI: / (chat UI), /ingest, /ask, /health
app/rag.py     # chunking, embedding, retrieval, answer generation
web/index.html # single-file chat UI
api/index.py   # Vercel serverless entry
tests/         # chunker + prompt unit tests (offline)
Dockerfile
```

## Author

Abbas Mustafa — published NLP research (IEEE FIT 2022), MSc Cybersecurity @ IMT Atlantique.
Hire me on [Fiverr](https://www.fiverr.com/abbasmsh1).
