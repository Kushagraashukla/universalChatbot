import { useState } from "react";
import type { Citation } from "../api";
import { FormattedText } from "./FormattedText";

export type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  sources?: Citation[];
};

/** "Workflows: accounts and access > Admin creates a student" -> "Workflows: accounts and access › Admin creates a student" */
function sourceLabel(c: Citation): string {
  return c.section.replace(/`/g, "").split(" > ").join(" › ") || c.title;
}

/** One line per section; several [n] markers that point into the same section are shown together. */
function groupBySection(citations: Citation[]): { label: string; refs: number[] }[] {
  const groups = new Map<string, number[]>();
  for (const c of citations) {
    const label = sourceLabel(c);
    groups.set(label, [...(groups.get(label) ?? []), c.ref]);
  }
  return [...groups].map(([label, refs]) => ({ label, refs }));
}

// Developer view only: raw file, chunk id and distance for each passage the model saw.
function DebugItem({ c }: { c: Citation }) {
  return (
    <li>
      <span className="source-ref">{c.ref}</span>
      <span className="source-file">{c.chunk_id}</span>
      <span className="source-section">
        {c.distance === null ? "neighbouring passage" : `${c.metric} distance ${c.distance}`} · {c.section}
      </span>
    </li>
  );
}

export function MessageBubble({ message, debug = false }: { message: Message; debug?: boolean }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";
  const citations = message.citations ?? [];
  const sources = message.sources ?? [];

  async function copy() {
    try {
      await navigator.clipboard.writeText(message.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked (e.g. insecure context); nothing else to do */
    }
  }

  return (
    <div className={`msg ${isUser ? "msg-user" : "msg-assistant"}`}>
      <div className="bubble">
        <span className="sr-only">{isUser ? "You said:" : "Assistant said:"}</span>
        <div className="bubble-text">{isUser ? message.text : <FormattedText text={message.text} />}</div>

        {!isUser && citations.length > 0 && (
          <div className="sources">
            <div className="sources-label">Sources</div>
            <ul>
              {groupBySection(citations).map(({ label, refs }) => (
                <li key={label}>
                  <span className="source-ref">{refs.join(", ")}</span>
                  <span className="source-section">{label}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {!isUser && debug && sources.length > 0 && (
          <details className="retrieved">
            <summary>Debug: retrieved passages ({sources.length})</summary>
            <ul>{sources.map((c) => <DebugItem key={c.chunk_id} c={c} />)}</ul>
          </details>
        )}

        {!isUser && (
          <div className="msg-actions">
            <button type="button" className="btn-ghost btn-small" onClick={copy} aria-label="Copy answer">
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
