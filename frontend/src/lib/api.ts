export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Source = { id: string; score: number; text: string; source: string };
export type Doc = {
  id: string;
  source: string;
  chunk_count: number;
  chunk_profile: string;
  updated_at: number;
};
export type Stats = {
  documents: number;
  chunks: number;
  total_vectors: number | null;
  namespaces: Record<string, number>;
  active_namespace: string;
  reindex: { status: string; done: number; total: number };
};

function headers(): Record<string, string> {
  const token = typeof window !== "undefined" ? localStorage.getItem("demo_token") : null;
  return token ? { "X-Demo-Token": token } : {};
}

export async function ask(
  body: { question: string; k: number; alpha: number },
  onToken: (t: string) => void,
  onSources: (s: Source[]) => void,
  signal?: AbortSignal,
) {
  const res = await fetch(`${API}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers() },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) throw new Error((await res.text()) || `HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const ev of events) {
      const type = ev.match(/^event: (.+)$/m)?.[1];
      const data = ev.match(/^data: (.+)$/m)?.[1];
      if (!type || !data) continue;
      const parsed = JSON.parse(data);
      if (type === "token") onToken(parsed.text);
      if (type === "sources") onSources(parsed.sources);
    }
  }
}

export async function uploadFile(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/ingest`, { method: "POST", headers: headers(), body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ id: string; source: string; chunks_indexed: number; unchanged: boolean }>;
}

export async function listDocuments(page = 1, q = "") {
  const res = await fetch(`${API}/documents?page=${page}&q=${encodeURIComponent(q)}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ total: number; page: number; documents: Doc[] }>;
}

export async function deleteDocument(id: string) {
  const res = await fetch(`${API}/documents/${id}`, { method: "DELETE", headers: headers() });
  if (!res.ok) throw new Error(await res.text());
}

export async function getStats() {
  const res = await fetch(`${API}/stats`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<Stats>;
}

export async function startReindex() {
  const res = await fetch(`${API}/reindex`, { method: "POST", headers: headers() });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ started: boolean; target_namespace: string }>;
}
