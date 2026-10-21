// The chat UI shared by the full-page app and the embeddable widget.
import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { ApiError, getAppInfo, sendChat, type AppInfo } from "../api";
import { MessageBubble, type Message } from "./MessageBubble";

const MAX_CHARS = 2000; // matches the server default RAG_MAX_QUESTION_CHARS
const TIMEOUT_MS = 120_000; // local LLM answers can take a while

type Props = {
  apiBase: string;
  appId: string;
  title?: string;
  headerActions?: ReactNode;
  autoFocus?: boolean;
  debug?: boolean; // show retrieved passages with chunk ids and distances (developers only)
};

function newId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function ChatPanel({ apiBase, appId, title, headerActions, autoFocus, debug = false }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ message: string; question: string } | null>(null);
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState(newId);

  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Load the app's public info (name, welcome text, example questions) from the backend.
  useEffect(() => {
    const ctrl = new AbortController();
    setInfo(null);
    setInfoError(null);
    getAppInfo(apiBase, appId, ctrl.signal)
      .then(setInfo)
      .catch((err) => {
        if (err.name !== "AbortError") setInfoError(err instanceof ApiError ? err.message : "Could not load app info.");
      });
    return () => ctrl.abort();
  }, [apiBase, appId]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading, error]);

  useEffect(() => () => abortRef.current?.abort(), []);

  // Focus the input when the panel becomes visible (the widget keeps the panel mounted while closed).
  useEffect(() => {
    if (autoFocus) inputRef.current?.focus();
  }, [autoFocus]);

  // Grow the textarea with its content, up to the CSS max-height.
  // Re-runs when the widget opens: while hidden every size reads 0.
  useEffect(() => {
    const el = inputRef.current;
    if (!el || el.offsetParent === null) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight + el.offsetHeight - el.clientHeight}px`; // + borders, so no scrollbar appears
  }, [input, autoFocus]);

  async function ask(question: string, isRetry = false) {
    const q = question.trim();
    if (!q || loading) return;
    if (!isRetry) {
      setMessages((m) => [...m, { id: newId(), role: "user", text: q }]);
      setInput("");
    }
    setError(null);
    setLoading(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      ctrl.abort();
    }, TIMEOUT_MS);

    try {
      const res = await sendChat(apiBase, { question: q, app_id: appId, session_id: sessionId }, ctrl.signal);
      setMessages((m) => [
        ...m,
        { id: res.request_id, role: "assistant", text: res.answer, citations: res.citations, sources: res.sources },
      ]);
    } catch (err) {
      if ((err as Error).name === "AbortError" && !timedOut) return; // user started a new chat
      const message = timedOut
        ? "The answer took too long. Please try again."
        : err instanceof ApiError
          ? err.message
          : "Something went wrong. Please try again.";
      setError({ message, question: q });
    } finally {
      clearTimeout(timer);
      if (abortRef.current === ctrl) {
        abortRef.current = null;
        setLoading(false);
      }
    }
  }

  function newChat() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
    setMessages([]);
    setError(null);
    setInput("");
    setSessionId(newId());
    inputRef.current?.focus();
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      ask(input);
    }
  }

  const heading = title || info?.name || "Assistant";
  const tooLong = input.length > MAX_CHARS;

  return (
    <section className="chat" aria-label={heading}>
      <header className="chat-header">
        <div className="chat-title">
          <span className="chat-dot" aria-hidden="true" />
          <h1>{heading}</h1>
        </div>
        <div className="chat-header-actions">
          <button type="button" className="btn-ghost" onClick={newChat} disabled={messages.length === 0 && !error}>
            New chat
          </button>
          {headerActions}
        </div>
      </header>

      <div className="chat-body" ref={listRef} role="log" aria-live="polite" aria-relevant="additions">
        {messages.length === 0 && !error && (
          <div className="welcome">
            <h2>How can I help?</h2>
            <p>{infoError ?? info?.welcome_message ?? "Ask a question about the knowledge base."}</p>
            {info && info.example_questions.length > 0 && (
              <ul className="examples" aria-label="Example questions">
                {info.example_questions.map((q) => (
                  <li key={q}>
                    <button type="button" className="example" onClick={() => ask(q)} disabled={loading}>
                      {q}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {messages.map((m) => <MessageBubble key={m.id} message={m} debug={debug} />)}

        {loading && (
          <div className="msg msg-assistant">
            <div className="bubble typing" role="status" aria-label="Assistant is writing an answer">
              <span /><span /><span />
            </div>
          </div>
        )}

        {error && (
          <div className="error" role="alert">
            <p>{error.message}</p>
            <button type="button" className="btn-secondary btn-small" onClick={() => ask(error.question, true)}>
              Retry
            </button>
          </div>
        )}
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
      >
        <label htmlFor={`q-${sessionId}`} className="sr-only">Your question</label>
        <textarea
          id={`q-${sessionId}`}
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a question…"
          rows={1}
          aria-describedby={`hint-${sessionId}`}
          aria-invalid={tooLong}
        />
        <button type="submit" className="btn-primary" disabled={loading || !input.trim() || tooLong} aria-label="Send">
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path d="M3.4 20.4 21 12 3.4 3.6 3.4 10l12.6 2-12.6 2z" fill="currentColor" />
          </svg>
        </button>
        <div id={`hint-${sessionId}`} className={`composer-hint ${tooLong ? "over" : ""}`}>
          {input.length > MAX_CHARS * 0.8 ? `${input.length}/${MAX_CHARS} · ` : ""}Enter to send · Shift+Enter for a new line
        </div>
      </form>
    </section>
  );
}
