"""Pinecone hybrid store: dense + sparse in one serverless index.

Dense: llama-text-embed-v2 (1024d). Sparse: pinecone-sparse-english-v0 —
stateless, so incremental ingest never needs BM25-style corpus re-fitting.
Hybrid queries use the Pinecone-documented alpha convex combination; optional
rerank via bge-reranker-v2-m3.
"""
import os
import time

DENSE_MODEL = "llama-text-embed-v2"
SPARSE_MODEL = "pinecone-sparse-english-v0"
RERANK_MODEL = "bge-reranker-v2-m3"
DIMENSION = 1024
UPSERT_BATCH = 100
EMBED_BATCH = 96  # inference API batch limit headroom
MIN_SCORE = 0.1


def hybrid_score_norm(dense, sparse, alpha):
    """Convex combination: alpha=1 pure semantic, alpha=0 pure keyword."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    hs = {"indices": sparse["indices"], "values": [v * (1 - alpha) for v in sparse["values"]]}
    return [v * alpha for v in dense], hs


def _retry(fn, attempts=3):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)


class PineconeStore:
    def __init__(self):
        from pinecone import Pinecone, ServerlessSpec

        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        name = os.environ.get("PINECONE_INDEX", "rag-chatbot")
        if not self.pc.has_index(name):
            self.pc.create_index(
                name=name,
                dimension=DIMENSION,
                metric="dotproduct",  # required for hybrid dense+sparse
                spec=ServerlessSpec(
                    cloud=os.environ.get("PINECONE_CLOUD", "aws"),
                    region=os.environ.get("PINECONE_REGION", "us-east-1"),
                ),
            )
        self.index = self.pc.Index(name)

    # -- embeddings -----------------------------------------------------------
    def _embed(self, texts, input_type):
        dense, sparse = [], []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            d = _retry(lambda: self.pc.inference.embed(
                model=DENSE_MODEL, inputs=batch,
                parameters={"input_type": input_type, "truncate": "END"}))
            s = _retry(lambda: self.pc.inference.embed(
                model=SPARSE_MODEL, inputs=batch,
                parameters={"input_type": input_type, "truncate": "END"}))
            dense.extend(e["values"] for e in d)
            sparse.extend({"indices": e["sparse_indices"], "values": e["sparse_values"]} for e in s)
        return dense, sparse

    # -- writes ---------------------------------------------------------------
    def upsert_chunks(self, doc_id, source, chunks, namespace):
        """Embed and upsert a document's chunks. Vector id = doc_id#ordinal."""
        dense, sparse = self._embed([c["text"] for c in chunks], "passage")
        vectors = [
            {
                "id": f"{doc_id}#{c['ordinal']}",
                "values": dv,
                "sparse_values": sv,
                "metadata": {"text": c["text"][:2000], "source": source, "doc_id": doc_id},
            }
            for c, dv, sv in zip(chunks, dense, sparse)
        ]
        for i in range(0, len(vectors), UPSERT_BATCH):
            batch = vectors[i:i + UPSERT_BATCH]
            _retry(lambda: self.index.upsert(vectors=batch, namespace=namespace))
        return len(vectors)

    def delete_document(self, doc_id, chunk_count, namespace):
        ids = [f"{doc_id}#{i}" for i in range(chunk_count)]
        for i in range(0, len(ids), 1000):
            _retry(lambda b=ids[i:i + 1000]: self.index.delete(ids=b, namespace=namespace))

    def delete_namespace(self, namespace):
        self.index.delete(delete_all=True, namespace=namespace)

    # -- reads ----------------------------------------------------------------
    def query(self, question, namespace, k=5, alpha=0.7, rerank=True):
        dense, sparse = self._embed([question], "query")
        hdense, hsparse = hybrid_score_norm(dense[0], sparse[0], alpha)
        oversample = k * 4 if rerank else k
        res = _retry(lambda: self.index.query(
            namespace=namespace, top_k=oversample, vector=hdense,
            sparse_vector=hsparse, include_metadata=True))
        hits = [
            {
                "id": m["id"],
                "score": float(m["score"]),
                "text": m["metadata"].get("text", ""),
                "source": m["metadata"].get("source", ""),
            }
            for m in res["matches"] if float(m["score"]) >= MIN_SCORE
        ]
        if rerank and len(hits) > k:
            rr = _retry(lambda: self.pc.inference.rerank(
                model=RERANK_MODEL, query=question,
                documents=[h["text"] for h in hits], top_n=k))
            hits = [
                {**hits[r["index"]], "score": round(float(r["score"]), 4)}
                for r in rr.data
            ]
        return hits[:k]

    def stats(self):
        s = self.index.describe_index_stats()
        return {
            "total_vectors": s["total_vector_count"],
            "namespaces": {ns: n["vector_count"] for ns, n in s.get("namespaces", {}).items()},
        }
