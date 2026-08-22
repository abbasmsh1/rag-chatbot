import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["RAG_DATA_DIR"] = "/tmp/rag-test-data"

from app.chunking import chunk_document, detect_profile, adaptive_target  # noqa: E402
from app.main import app, build_prompt, get_registry, ingest_text  # noqa: E402
from app.registry import Registry, content_hash, doc_id_for  # noqa: E402
from app.store import hybrid_score_norm  # noqa: E402


# -- chunking -----------------------------------------------------------------

def test_chunks_cover_text_and_overlap():
    text = "\n\n".join(" ".join(f"word{p}_{i}" for i in range(120)) for p in range(30))
    chunks = chunk_document(text, "doc.txt")
    assert len(chunks) > 1
    joined = " ".join(c["text"] for c in chunks)
    assert "word0_0" in joined and "word29_119" in joined
    assert chunks[1]["ordinal"] == 1


def test_profiles():
    assert detect_profile("# Title\n\nbody", "readme.md") == "markdown"
    assert detect_profile("def f():\n    pass", "x.py") == "code"
    assert detect_profile("Plain sentences here.", "a.txt") == "prose"
    code = "\n".join(f"def f{i}(): pass" for i in range(6))
    assert detect_profile(code) == "code"


def test_adaptive_target():
    assert adaptive_target(400, 100) == 100          # short doc: single chunk
    assert adaptive_target(400, 10_000) == 400       # normal doc: base size
    assert adaptive_target(400, 100_000) == 800      # huge doc: bigger chunks


def test_empty_document():
    assert chunk_document("") == []
    assert chunk_document("   \n  ") == []


# -- hybrid scoring -------------------------------------------------------------

def test_hybrid_score_norm_weights():
    dense, sparse = hybrid_score_norm([1.0, 2.0], {"indices": [3], "values": [4.0]}, alpha=0.75)
    assert dense == [0.75, 1.5]
    assert sparse["values"] == [1.0]
    with pytest.raises(ValueError):
        hybrid_score_norm([1.0], {"indices": [], "values": []}, alpha=1.5)


# -- registry -----------------------------------------------------------------

def test_registry_crud_and_namespace_swap(tmp_path):
    r = Registry(path=str(tmp_path / "reg.db"))
    assert r.active_namespace == "v1"
    assert r.next_namespace() == "v2"
    did = doc_id_for("a.txt")
    r.upsert_document(did, "a.txt", content_hash("hello"), 3, "prose")
    assert r.get_document(did)["chunk_count"] == 3
    assert r.counts() == {"documents": 1, "chunks": 3}
    assert r.list_documents(q="a.")["total"] == 1
    r.set_meta("active_namespace", "v2")
    assert r.active_namespace == "v2"
    r.delete_document(did)
    assert r.get_document(did) is None


# -- ingest flow with fake store -------------------------------------------------

class FakeStore:
    def __init__(self):
        self.upserts, self.deletes = [], []

    def upsert_chunks(self, doc_id, source, chunks, namespace):
        self.upserts.append((doc_id, namespace, len(chunks)))
        return len(chunks)

    def delete_document(self, doc_id, chunk_count, namespace):
        self.deletes.append(doc_id)


def test_ingest_dedupes_unchanged_and_replaces_changed(tmp_path):
    store, reg = FakeStore(), Registry(path=str(tmp_path / "reg.db"))
    text = "\n\n".join(f"paragraph {i} " * 40 for i in range(20))
    _, n1 = ingest_text(text, "doc.txt", store, reg)
    assert n1 > 0
    _, n2 = ingest_text(text, "doc.txt", store, reg)      # unchanged: no-op
    assert n2 == 0 and len(store.upserts) == 1
    _, n3 = ingest_text(text + " changed", "doc.txt", store, reg)
    assert n3 > 0 and store.deletes == [doc_id_for("doc.txt")]
    assert reg.load_text(doc_id_for("doc.txt")).endswith("changed")


def test_prompt_numbers_contexts_and_guards():
    p = build_prompt("what is x?", ["ctx one", "ctx two"])
    assert "[1] ctx one" in p and "[2] ctx two" in p
    assert "don't know" in p


# -- api ------------------------------------------------------------------------

def test_health_and_documents_endpoint():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    resp = client.get("/documents")
    assert resp.status_code == 200 and "documents" in resp.json()
