// Calls to the FastAPI backend. The browser only ever sends app_id; no keys or secrets.

export type Citation = {
  ref: number;
  chunk_id: string;
  source: string;
  title: string;
  section: string;
  distance: number | null; // null = neighbouring passage added for context, not a search match
  metric: string;
};

export type ChatResponse = {
  answer: string;
  citations: Citation[];
  sources: Citation[];
  app_id: string;
  session_id: string | null;
  request_id: string;
  model: string;
  duration_ms: number;
};

export type AppInfo = {
  app_id: string;
  name: string;
  welcome_message: string;
  example_questions: string[];
};

export class ApiError extends Error {
  constructor(message: string, readonly code: string, readonly status: number) {
    super(message);
  }
}

async function request<T>(url: string, init: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError("Cannot reach the chatbot server. Check that it is running.", "network_error", 0);
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    throw new ApiError(data?.error?.message ?? `Request failed (${res.status}).`, data?.error?.code ?? "http_error", res.status);
  }
  return data as T;
}

export function sendChat(
  apiBase: string,
  body: { question: string; app_id: string; session_id: string },
  signal: AbortSignal,
): Promise<ChatResponse> {
  return request(`${apiBase}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

export function getAppInfo(apiBase: string, appId: string, signal: AbortSignal): Promise<AppInfo> {
  return request(`${apiBase}/api/v1/apps/${encodeURIComponent(appId)}`, { signal });
}
