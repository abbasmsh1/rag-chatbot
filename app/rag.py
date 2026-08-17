"""Chunking, vector store, and answer generation."""
import os

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
    """ChromaDB-backed store with OpenAI embeddings and chat completion."""

    def __init__(self, persist_dir="chroma_db"):
        import chromadb
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.db = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.db.get_or_create_collection("docs")

    def _embed(self, texts):
        resp = self.client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in resp.data]

    def ingest(self, text, source):
        chunks = chunk_text(text)
        if not chunks:
            return 0
        ids = [f"{source}-{i}" for i in range(len(chunks))]
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=self._embed(chunks),
            metadatas=[{"source": source}] * len(chunks),
        )
        return len(chunks)

    def ask(self, question, k=4):
        q_emb = self._embed([question])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        # chroma returns distances; convert to similarity
        hits = [
            {"chunk": d, "source": m["source"], "score": round(1 - dist, 3)}
            for d, m, dist in zip(docs, metas, dists)
            if 1 - dist >= MIN_SCORE
        ]
        if not hits:
            return {"answer": "I don't know - nothing in the ingested documents covers this.", "sources": []}

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": build_prompt(question, [h["chunk"] for h in hits])}],
            temperature=0,
        )
        return {"answer": completion.choices[0].message.content, "sources": hits}
