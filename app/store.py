"""Qdrant hybrid store: dense + sparse (BM25) named vectors, no API key needed.

Embeddings run locally via fastembed (ONNX): dense BAAI/bge-small-en-v1.5
(384d) + sparse Qdrant/bm25 with server-side IDF, so incremental ingest never
needs corpus re-fitting. Runs embedded on disk by default; set QDRANT_URL to
point at a Qdrant server (docker run -p 6333:6333 qdrant/qdrant) for
production scale.

Hybrid retrieval: dense and sparse queries fused client-side with an
alpha-weighted convex combination of max-normalized scores
(alpha=1 pure semantic, alpha=0 pure keyword).
"""
import os
import uuid

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DIMENSION = 384
UPSERT_BATCH = 256
MIN_SCORE = 0.05
COLLECTION_BASE = os.environ.get("QDRANT_COLLECTION", "rag")


def fuse(dense_hits, sparse_hits, alpha, k):
    """Weighted fusion of two hit lists ({id, score, ...}). Scores are
    max-normalized per list first, since BM25 scores are unbounded."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    merged = {}
    for hits, weight in ((dense_hits, alpha), (sparse_hits, 1 - alpha)):
        top = max((h["score"] for h in hits), default=0) or 1
        for h in hits:
            entry = merged.setdefault(h["id"], {**h, "score": 0.0})
            entry["score"] += weight * (h["score"] / top)
    ranked = sorted(merged.values(), key=lambda h: h["score"], reverse=True)
    return [{**h, "score": round(h["score"], 4)} for h in ranked[:k]]


def point_id(doc_id, ordinal):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#{ordinal}"))


class QdrantStore:
    def __init__(self):
        from fastembed import SparseTextEmbedding, TextEmbedding
        from qdrant_client import QdrantClient

        url = os.environ.get("QDRANT_URL")
        if url:
            self.client = QdrantClient(url=url)
        else:
            path = os.path.join(os.environ.get("RAG_DATA_DIR", "data"), "qdrant")
            self.client = QdrantClient(path=path)
        self.dense_model = TextEmbedding(DENSE_MODEL)
        self.sparse_model = SparseTextEmbedding(SPARSE_MODEL)

    def _collection(self, namespace):
        """Collection per namespace version; created on first use."""
        from qdrant_client import models

        name = f"{COLLECTION_BASE}_{namespace}"
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name,
                vectors_config={"dense": models.VectorParams(
                    size=DIMENSION, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF)},
            )
        return name

    # -- writes ---------------------------------------------------------------
    def upsert_chunks(self, doc_id, source, chunks, namespace):
        from qdrant_client import models

        name = self._collection(namespace)
        texts = [c["text"] for c in chunks]
        dense = list(self.dense_model.passage_embed(texts))
        sparse = list(self.sparse_model.passage_embed(texts))
        points = [
            models.PointStruct(
                id=point_id(doc_id, c["ordinal"]),
                vector={
                    "dense": d.tolist(),
                    "sparse": models.SparseVector(
                        indices=s.indices.tolist(), values=s.values.tolist()),
                },
                payload={"text": c["text"][:2000], "source": source, "doc_id": doc_id},
            )
            for c, d, s in zip(chunks, dense, sparse)
        ]
        for i in range(0, len(points), UPSERT_BATCH):
            self.client.upsert(name, points[i:i + UPSERT_BATCH])
        return len(points)

    def delete_document(self, doc_id, chunk_count, namespace):
        from qdrant_client import models

        self.client.delete(
            self._collection(namespace),
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
            ])),
        )

    def delete_namespace(self, namespace):
        self.client.delete_collection(f"{COLLECTION_BASE}_{namespace}")

    # -- reads ----------------------------------------------------------------
    def _query_one(self, query, using, namespace, limit):
        res = self.client.query_points(
            self._collection(namespace), query=query, using=using,
            limit=limit, with_payload=True)
        return [
            {
                "id": str(p.id),
                "score": float(p.score),
                "text": p.payload.get("text", ""),
                "source": p.payload.get("source", ""),
            }
            for p in res.points
        ]

    def query(self, question, namespace, k=5, alpha=0.7):
        from qdrant_client import models

        qd = list(self.dense_model.query_embed(question))[0]
        qs = list(self.sparse_model.query_embed(question))[0]
        oversample = k * 4
        dense_hits = self._query_one(qd.tolist(), "dense", namespace, oversample)
        sparse_hits = self._query_one(
            models.SparseVector(indices=qs.indices.tolist(), values=qs.values.tolist()),
            "sparse", namespace, oversample)
        # ponytail: no cross-encoder rerank stage; add a fastembed reranker if
        # fusion quality ever measurably falls short
        return [h for h in fuse(dense_hits, sparse_hits, alpha, k) if h["score"] >= MIN_SCORE]

    def stats(self):
        out, total = {}, 0
        for c in self.client.get_collections().collections:
            if c.name.startswith(f"{COLLECTION_BASE}_"):
                n = self.client.count(c.name).count
                out[c.name.removeprefix(f"{COLLECTION_BASE}_")] = n
                total += n
        return {"total_vectors": total, "namespaces": out}
