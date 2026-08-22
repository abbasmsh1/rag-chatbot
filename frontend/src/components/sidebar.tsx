"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { getStats, type Stats } from "@/lib/api";

const NAV = [
  { href: "/chat", label: "Chat", icon: "M8 10h8M8 14h5m-9 7 3-3h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v14Z" },
  { href: "/documents", label: "Documents", icon: "M9 12h6m-6 4h6M7 3h7l5 5v11a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" },
  { href: "/indexing", label: "Index", icon: "M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3Zm0 0v10c0 1.7 3.6 3 8 3s8-1.3 8-3V7M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    getStats().then(setStats).catch(() => setStats(null));
  }, [pathname]);

  return (
    <aside className="glass flex w-56 shrink-0 flex-col border-r">
      <div className="flex items-center gap-2.5 px-5 py-5">
        <span className="grid size-8 place-items-center rounded-xl bg-accent-dim text-accent">
          <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M12 3v3m0 12v3m9-9h-3M6 12H3m14.5-6.5-2 2m-7 7-2 2m11 0-2-2m-7-7-2-2" />
            <circle cx="12" cy="12" r="3.5" />
          </svg>
        </span>
        <h1 className="text-[15px] font-semibold tracking-tight">RAG Console</h1>
      </div>

      <nav className="flex flex-col gap-1 px-3">
        {NAV.map(({ href, label, icon }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={`flex items-center gap-3 rounded-xl px-3 py-2 text-sm transition-colors ${
                active
                  ? "bg-accent-dim font-medium text-accent"
                  : "text-muted hover:bg-white/[0.04] hover:text-foreground"
              }`}
            >
              <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d={icon} />
              </svg>
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto border-t border-[color:var(--glass-border)] p-4">
        {stats ? (
          <div className="flex items-center gap-2 text-xs text-muted">
            <span
              className="size-1.5 rounded-full bg-accent"
              style={{ animation: "pulse-dot 2.4s ease-in-out infinite" }}
            />
            <span className="font-mono">{(stats.total_vectors ?? 0).toLocaleString()}</span>
            vectors · {stats.active_namespace}
          </div>
        ) : (
          <div className="flex items-center gap-2 text-xs text-muted">
            <span className="size-1.5 rounded-full bg-danger" />
            API offline
          </div>
        )}
      </div>
    </aside>
  );
}
