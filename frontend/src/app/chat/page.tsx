"use client";

import { useRef, useState } from "react";
import { ask, type Source } from "@/lib/api";

type Message = {
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  error?: boolean;
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [k, setK] = useState(5);
  const [alpha, setAlpha] = useState(0.7);
  const threadRef = useRef<HTMLDivElement>(null);

  async function send() {
    const question = input.trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text: question }, { role: "assistant", text: "" }]);

    const patchLast = (patch: Partial<Message> | ((prev: Message) => Partial<Message>)) =>
      setMessages((m) => {
        const last = m[m.length - 1];
        const p = typeof patch === "function" ? patch(last) : patch;
        return [...m.slice(0, -1), { ...last, ...p }];
      });

    try {
      await ask(
        { question, k, alpha },
        (t) => {
          patchLast((prev) => ({ text: prev.text + t }));
          threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight });
        },
        (sources) => patchLast({ sources }),
      );
    } catch (e) {
      patchLast({ text: e instanceof Error ? e.message : "Request failed", error: true });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div ref={threadRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-6 px-6 py-8">
          {messages.length === 0 && (
            <div className="mt-24 text-center">
              <h2 className="text-xl font-medium tracking-tight">Ask your corpus</h2>
              <p className="mx-auto mt-2 max-w-sm text-sm text-muted">
                Hybrid retrieval over every indexed document. Answers are grounded and cite their
                source chunks.
              </p>
            </div>
          )}
          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="pane ml-auto max-w-[80%] rounded-2xl rounded-br-md px-4 py-2.5 text-sm">
                {m.text}
              </div>
            ) : (
              <div key={i} className="max-w-[92%]">
                <div
                  className={`whitespace-pre-wrap text-sm leading-relaxed ${m.error ? "text-danger" : ""}`}
                >
                  {m.text || (
                    <span className="inline-flex gap-1 text-muted">
                      retrieving
                      <span style={{ animation: "pulse-dot 1.2s ease-in-out infinite" }}>...</span>
                    </span>
                  )}
                </div>
                {m.sources && m.sources.length > 0 && <SourceList sources={m.sources} />}
              </div>
            ),
          )}
        </div>
      </div>

      <div className="glass border-t px-6 py-4">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2 rounded-2xl border border-border bg-white/[0.03] p-2 transition-colors focus-within:border-[color:rgba(110,231,183,0.35)]">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={1}
              placeholder="Ask a question..."
              className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-muted"
            />
            <button
              onClick={send}
              disabled={busy || !input.trim()}
              aria-label="Send"
              className="grid size-9 shrink-0 place-items-center rounded-xl bg-accent text-background shadow-[0_2px_12px_rgba(110,231,183,0.25)] transition-opacity disabled:opacity-30 disabled:shadow-none"
            >
              <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 19V5m-6 6 6-6 6 6" />
              </svg>
            </button>
          </div>
          <div className="mt-2 flex items-center gap-6 px-1 text-xs text-muted">
            <label className="flex items-center gap-2">
              top-k
              <select
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="rounded-md border border-border bg-surface-2 px-1.5 py-0.5 outline-none"
              >
                {[3, 5, 8, 12].map((n) => (
                  <option key={n}>{n}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-1 items-center gap-2 sm:max-w-64">
              keyword
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                className="flex-1"
              />
              semantic
              <span className="w-8 font-mono">{alpha.toFixed(2)}</span>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}

function SourceList({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs text-muted transition-colors hover:text-foreground"
      >
        <svg
          viewBox="0 0 24 24"
          className={`size-3 transition-transform ${open ? "rotate-90" : ""}`}
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="m9 6 6 6-6 6" />
        </svg>
        {sources.length} source{sources.length > 1 ? "s" : ""}
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-2">
          {sources.map((s, i) => (
            <div key={s.id} className="pane rounded-xl p-3">
              <div className="flex items-center gap-2 text-xs">
                <span className="rounded bg-accent-dim px-1.5 py-0.5 font-mono text-accent">
                  [{i + 1}]
                </span>
                <span className="truncate font-medium">{s.source}</span>
                <div className="ml-auto flex items-center gap-1.5">
                  <div className="h-1 w-16 overflow-hidden rounded-full bg-white/[0.08]">
                    <div
                      className="h-full rounded-full bg-accent"
                      style={{ width: `${Math.min(s.score, 1) * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-muted">{s.score.toFixed(2)}</span>
                </div>
              </div>
              <p className="mt-2 line-clamp-2 font-mono text-xs leading-relaxed text-muted">
                {s.text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
