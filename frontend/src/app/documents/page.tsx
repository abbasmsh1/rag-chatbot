"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { deleteDocument, listDocuments, uploadFile, type Doc } from "@/lib/api";

const PER_PAGE = 50;

export default function DocumentsPage() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    listDocuments(page, q)
      .then((r) => {
        setDocs(r.documents);
        setTotal(r.total);
      })
      .catch((e) => setStatus(`Failed to load: ${e.message}`));
  }, [page, q]);

  useEffect(refresh, [refresh]);

  async function handleFiles(files: FileList | null) {
    if (!files?.length) return;
    for (const file of Array.from(files)) {
      setStatus(`Indexing ${file.name}...`);
      try {
        const r = await uploadFile(file);
        setStatus(
          r.unchanged
            ? `${r.source}: unchanged, skipped`
            : `${r.source}: ${r.chunks_indexed} chunks indexed`,
        );
      } catch (e) {
        setStatus(`${file.name} failed: ${e instanceof Error ? e.message : e}`);
      }
    }
    refresh();
  }

  const pages = Math.max(1, Math.ceil(total / PER_PAGE));

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <h2 className="text-xl font-medium tracking-tight">Documents</h2>
      <p className="mt-1 text-sm text-muted">
        {total.toLocaleString()} document{total === 1 ? "" : "s"} in the corpus
      </p>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => fileRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && fileRef.current?.click()}
        className={`mt-6 grid cursor-pointer place-items-center rounded-2xl border border-dashed px-6 py-10 text-center transition-colors ${
          dragging ? "border-accent bg-accent-dim" : "border-border-strong hover:bg-surface"
        }`}
      >
        <input
          ref={fileRef}
          type="file"
          multiple
          accept=".pdf,.txt,.md"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="text-sm">
          Drop PDF, TXT, or MD files here <span className="text-muted">or click to browse</span>
        </div>
        <div className="mt-1 text-xs text-muted">
          Bulk corpora: <code className="font-mono">python scripts/bulk_ingest.py ./dir</code>
        </div>
      </div>

      {status && <div className="mt-3 text-xs text-muted">{status}</div>}

      <div className="mt-6 flex items-center gap-3">
        <input
          value={q}
          onChange={(e) => {
            setPage(1);
            setQ(e.target.value);
          }}
          placeholder="Filter by source..."
          className="w-64 rounded-xl border border-border bg-surface px-3 py-1.5 text-sm outline-none placeholder:text-muted focus:border-border-strong"
        />
        {pages > 1 && (
          <div className="ml-auto flex items-center gap-2 text-xs text-muted">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="rounded-lg border border-border px-2 py-1 disabled:opacity-30"
            >
              prev
            </button>
            <span className="font-mono">
              {page}/{pages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(pages, p + 1))}
              disabled={page === pages}
              className="rounded-lg border border-border px-2 py-1 disabled:opacity-30"
            >
              next
            </button>
          </div>
        )}
      </div>

      <div className="mt-3 overflow-x-auto rounded-2xl border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-surface text-left text-xs text-muted">
              <th className="px-4 py-2.5 font-medium">Source</th>
              <th className="px-4 py-2.5 font-medium">Chunks</th>
              <th className="px-4 py-2.5 font-medium">Profile</th>
              <th className="px-4 py-2.5 font-medium">Updated</th>
              <th className="w-12 px-4 py-2.5" />
            </tr>
          </thead>
          <tbody>
            {docs.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-muted">
                  Nothing indexed yet.
                </td>
              </tr>
            )}
            {docs.map((d) => (
              <tr key={d.id} className="border-b border-border last:border-0 hover:bg-surface">
                <td className="max-w-64 truncate px-4 py-2.5">{d.source}</td>
                <td className="px-4 py-2.5 font-mono text-xs">{d.chunk_count}</td>
                <td className="px-4 py-2.5">
                  <span className="rounded bg-surface-2 px-1.5 py-0.5 text-xs text-muted">
                    {d.chunk_profile}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-xs text-muted">
                  {new Date(d.updated_at * 1000).toLocaleString()}
                </td>
                <td className="px-4 py-2.5">
                  <button
                    aria-label={`Delete ${d.source}`}
                    onClick={async () => {
                      if (!confirm(`Delete ${d.source} from the index?`)) return;
                      await deleteDocument(d.id).catch((e) => setStatus(`Delete failed: ${e.message}`));
                      refresh();
                    }}
                    className="text-muted transition-colors hover:text-danger"
                  >
                    <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M4 7h16M10 11v6m4-6v6M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13M9 7V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v3" />
                    </svg>
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
