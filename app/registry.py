"""SQLite document registry: source of truth for what is indexed where.

Tracks every document's content hash (dedupe + resume), chunk count, and the
active Pinecone namespace (reindex versioning). stdlib only.
"""
import hashlib
import os
import sqlite3
import time
from contextlib import contextmanager

DATA_DIR = os.environ.get("RAG_DATA_DIR", "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    chunk_profile TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO meta VALUES ('active_namespace', 'v1');
"""


def doc_id_for(source):
    return hashlib.sha1(source.encode()).hexdigest()[:16]


def content_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Registry:
    def __init__(self, path=None):
        os.makedirs(DOCS_DIR, exist_ok=True)
        self.path = path or os.path.join(DATA_DIR, "registry.db")
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        # ponytail: one connection per call, fine for single-process FastAPI;
        # move to a pool if this ever fronts multiple workers
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- namespace / meta ---------------------------------------------------
    def get_meta(self, key, default=None):
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default

    def set_meta(self, key, value):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, str(value)))

    @property
    def active_namespace(self):
        return self.get_meta("active_namespace", "v1")

    def next_namespace(self):
        return f"v{int(self.active_namespace.lstrip('v')) + 1}"

    # -- documents ----------------------------------------------------------
    def upsert_document(self, doc_id, source, chash, chunk_count, profile):
        now = time.time()
        with self._conn() as c:
            c.execute(
                """INSERT INTO documents VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET content_hash=excluded.content_hash,
                   chunk_count=excluded.chunk_count, chunk_profile=excluded.chunk_profile,
                   updated_at=excluded.updated_at""",
                (doc_id, source, chash, chunk_count, profile, now, now),
            )

    def get_document(self, doc_id):
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def delete_document(self, doc_id):
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        txt = os.path.join(DOCS_DIR, f"{doc_id}.txt")
        if os.path.exists(txt):
            os.remove(txt)

    def list_documents(self, page=1, per_page=50, q=""):
        offset = (max(page, 1) - 1) * per_page
        like = f"%{q}%"
        with self._conn() as c:
            total = c.execute(
                "SELECT COUNT(*) n FROM documents WHERE source LIKE ?", (like,)
            ).fetchone()["n"]
            rows = c.execute(
                "SELECT * FROM documents WHERE source LIKE ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (like, per_page, offset),
            ).fetchall()
        return {"total": total, "page": page, "documents": [dict(r) for r in rows]}

    def all_doc_ids(self):
        with self._conn() as c:
            return [r["id"] for r in c.execute("SELECT id FROM documents")]

    def counts(self):
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) docs, COALESCE(SUM(chunk_count),0) chunks FROM documents"
            ).fetchone()
            return {"documents": row["docs"], "chunks": row["chunks"]}

    # -- raw text storage (needed for reindex) --------------------------------
    def save_text(self, doc_id, text):
        with open(os.path.join(DOCS_DIR, f"{doc_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    def load_text(self, doc_id):
        with open(os.path.join(DOCS_DIR, f"{doc_id}.txt"), encoding="utf-8") as f:
            return f.read()
