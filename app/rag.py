"""Chunking, in-memory vector store, and answer generation.

The store is plain numpy cosine similarity: no external vector DB, so the same
code runs locally and on serverless (Vercel). Swap InMemoryStore for a hosted
store (Pinecone, Chroma server) when your corpus outgrows RAM.
"""
import os

import numpy as np

MIN_SCORE = 0.25  # below this similarity, refuse instead of hallucinate


def chunk_text(text, chunk_size=800, overlap=150):
    """Split text into overlapping chunks on whitespace boundaries."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must exceed overlap")
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    # approximate words per chunk from average word length
    step = max(1, chunk_size // 6)
    ov = max(0, overlap // 6)
    while start < len(words):
        piece = " ".join(words[start:start + step])
        chunks.append(piece)
        if start + step >= len(words):
            break
        start += step - ov
    return chunks


def build_prompt(question, contexts):
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        "Answer the question using ONLY the context below. "
        "Cite chunk numbers like [1]. If the context does not contain the answer, "
        "say you don't know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )


class RagStore:
    """In-memory store with OpenAI embeddings and chat completion."""

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.vectors = np.zeros((0, 1536), dtype=np.float32)
        self.chunks = []
        self.sources = []

    def _embed(self, texts):
        resp = self.client.embeddings.create(model="text-embedding-3-small", input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    def ingest(self, text, source):
        chunks = chunk_text(text)
        if not chunks:
            return 0
        emb = self._embed(chunks)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        self.vectors = np.vstack([self.vectors, emb])
        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))
        return len(chunks)

    def ask(self, question, k=4):
        if not self.chunks:
            return {"answer": "No documents ingested yet - upload one first.", "sources": []}
        q = self._embed([question])[0]
        q /= np.linalg.norm(q)
        sims = self.vectors @ q
        order = np.argsort(sims)[::-1][:k]
        hits = [
            {"chunk": self.chunks[i], "source": self.sources[i], "score": round(float(sims[i]), 3)}
            for i in order
            if sims[i] >= MIN_SCORE
        ]
        if not hits:
            return {"answer": "I don't know - nothing in the ingested documents covers this.", "sources": []}

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": build_prompt(question, [h["chunk"] for h in hits])}],
            temperature=0,
        )
        return {"answer": completion.choices[0].message.content, "sources": hits}
