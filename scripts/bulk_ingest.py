"""Checkpointed bulk ingester for large corpora (built for the 1M-doc path).

Usage:
    python scripts/bulk_ingest.py ./corpus_dir --workers 8
    python scripts/bulk_ingest.py ./docs.jsonl --workers 8   # {"source":..., "text":...} per line

Rerunning is safe: documents whose content hash is already registered are
skipped, so a crashed run resumes where it left off.
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import ingest_text  # noqa: E402
from app.registry import Registry, content_hash, doc_id_for  # noqa: E402
from app.store import PineconeStore  # noqa: E402

TEXT_EXT = {".txt", ".md", ".mdx", ".markdown", ".py", ".js", ".ts", ".rst", ".html", ".json", ".csv"}


def iter_corpus(path):
    p = Path(path)
    if p.is_file() and p.suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    yield rec["source"], rec["text"]
    else:
        for fp in p.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in TEXT_EXT:
                yield str(fp.relative_to(p)), fp.read_text(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="directory or .jsonl file")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    registry = Registry()
    store = PineconeStore()
    lock = threading.Lock()
    done = skipped = failed = 0

    def work(source, text):
        doc = registry.get_document(doc_id_for(source))
        if doc and doc["content_hash"] == content_hash(text):
            return "skipped"
        ingest_text(text, source, store, registry)  # retries live in the store layer
        return "done"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, s, t): s for s, t in iter_corpus(args.corpus)}
        for fut in as_completed(futures):
            source = futures[fut]
            try:
                result = fut.result()
                with lock:
                    if result == "skipped":
                        skipped += 1
                    else:
                        done += 1
            except Exception as e:
                with lock:
                    failed += 1
                print(f"FAILED {source}: {e}", file=sys.stderr)
            total = done + skipped + failed
            if total % 500 == 0:
                print(f"{total} processed ({done} indexed, {skipped} skipped, {failed} failed)")

    print(f"finished: {done} indexed, {skipped} skipped, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
