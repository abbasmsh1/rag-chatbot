"use client";

import { useEffect, useState } from "react";
import { getStats, startReindex, type Stats } from "@/lib/api";

export default function IndexPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const tick = () =>
      getStats()
        .then((s) => live && (setStats(s), setError(null)))
        .catch((e) => live && setError(e.message));
    tick();
    const id = setInterval(tick, 3000);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, []);

  const reindexing = stats?.reindex.status === "running";
  const progress =
    stats && stats.reindex.total > 0 ? stats.reindex.done / stats.reindex.total : 0;

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <h2 className="text-xl font-medium tracking-tight">Index</h2>
      <p className="mt-1 text-sm text-muted">Hybrid index health and lifecycle</p>
      {error && <p className="mt-4 text-sm text-danger">API unreachable: {error}</p>}

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Documents" value={stats?.documents} />
        <Stat label="Chunks" value={stats?.chunks} />
        <Stat label="Vectors" value={stats?.total_vectors} />
        <Stat label="Active namespace" value={stats?.active_namespace} mono />
      </div>

      <div className="mt-6 rounded-2xl border border-border bg-surface p-5">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-medium">Reindex</h3>
            <p className="mt-1 max-w-md text-xs leading-relaxed text-muted">
              Re-chunks and re-embeds every document into a fresh namespace, then swaps it in
              atomically. Queries keep serving from the current namespace until the swap - zero
              downtime.
            </p>
          </div>
          <button
            onClick={() =>
              startReindex()
                .then(() => setStats((s) => s && { ...s, reindex: { ...s.reindex, status: "running" } }))
                .catch((e) => setError(e.message))
            }
            disabled={reindexing || !stats}
            className="shrink-0 rounded-xl bg-accent px-4 py-2 text-sm font-medium text-background transition-opacity disabled:opacity-30"
          >
            {reindexing ? "Running..." : "Start reindex"}
          </button>
        </div>

        {reindexing && stats && (
          <div className="mt-4">
            <div className="flex justify-between text-xs text-muted">
              <span>
                {stats.reindex.done.toLocaleString()} / {stats.reindex.total.toLocaleString()} documents
              </span>
              <span className="font-mono">{Math.round(progress * 100)}%</span>
            </div>
            <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-2">
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-500"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>
        )}
        {stats?.reindex.status.startsWith("failed") && (
          <p className="mt-3 text-xs text-danger">{stats.reindex.status}</p>
        )}
      </div>

      <div className="mt-6 rounded-2xl border border-border bg-surface p-5">
        <h3 className="text-sm font-medium">Namespaces</h3>
        <div className="mt-3 flex flex-col gap-2">
          {stats && Object.keys(stats.namespaces).length === 0 && (
            <p className="text-xs text-muted">No collections yet - ingest a document first.</p>
          )}
          {stats &&
            Object.entries(stats.namespaces).map(([ns, count]) => (
              <div key={ns} className="flex items-center gap-3 text-sm">
                <span className="font-mono text-xs">{ns}</span>
                {ns === stats.active_namespace && (
                  <span className="rounded bg-accent-dim px-1.5 py-0.5 text-[10px] font-medium text-accent">
                    ACTIVE
                  </span>
                )}
                <span className="ml-auto font-mono text-xs text-muted">
                  {count.toLocaleString()} vectors
                </span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, mono }: { label: string; value?: number | string | null; mono?: boolean }) {
  return (
    <div className="rounded-2xl border border-border bg-surface p-4">
      <div className="text-xs text-muted">{label}</div>
      <div className={`mt-1 text-lg font-medium tabular-nums ${mono ? "font-mono text-base" : ""}`}>
        {value == null ? "—" : typeof value === "number" ? value.toLocaleString() : value}
      </div>
    </div>
  );
}
