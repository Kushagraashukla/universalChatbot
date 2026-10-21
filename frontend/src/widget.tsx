// Embeddable widget. Usage on any web page:
//   <script src="http://localhost:8000/widget/widget.js" data-app-id="default" defer></script>
// Optional attributes: data-api-base (default: the script's own origin), data-title, data-position="left|right",
// data-debug="true" (developers: show retrieved passages with chunk ids and distances).
// Renders inside a Shadow DOM so the host page's CSS and the widget's CSS don't affect each other.
import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ChatPanel } from "./components/ChatPanel";
import css from "./styles.css?inline";

const script = document.currentScript as HTMLScriptElement | null; // only available while the script first runs

type Config = { appId: string; apiBase: string; title?: string; position: "left" | "right"; debug: boolean };

function readConfig(): Config {
  const d = script?.dataset ?? {};
  const fallbackBase = script?.src ? new URL(script.src).origin : window.location.origin;
  return {
    appId: d.appId || "default",
    apiBase: (d.apiBase || fallbackBase).replace(/\/$/, ""),
    title: d.title,
    position: d.position === "left" ? "left" : "right",
    debug: d.debug === "true",
  };
}

function Widget({ config }: { config: Config }) {
  const [open, setOpen] = useState(false);
  const launcherRef = useRef<HTMLButtonElement>(null);

  function close() {
    setOpen(false);
    launcherRef.current?.focus(); // keep keyboard focus somewhere visible
  }

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const closeButton = (
    <button type="button" className="btn-ghost btn-icon" onClick={close} aria-label="Close chat">
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
        <path d="M6 6l12 12M18 6 6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
    </button>
  );

  return (
    <div className={`rag-chat widget widget-${config.position}`}>
      {/* Kept mounted while closed so the conversation survives closing and reopening. */}
      <div id="rag-widget-panel" className="widget-panel" role="dialog" aria-label={config.title || "Chat"} hidden={!open}>
        <ChatPanel apiBase={config.apiBase} appId={config.appId} title={config.title} headerActions={closeButton} autoFocus={open} debug={config.debug} />
      </div>
      <button
        ref={launcherRef}
        type="button"
        className="widget-launcher"
        aria-expanded={open}
        aria-controls="rag-widget-panel"
        aria-label={open ? "Close chat" : "Open chat"}
        onClick={() => (open ? close() : setOpen(true))}
      >
        <svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
          {open ? (
            <path d="M6 6l12 12M18 6 6 18" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
          ) : (
            <path d="M4 5h16v11H8l-4 4z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          )}
        </svg>
      </button>
    </div>
  );
}

function mount() {
  if (document.getElementById("rag-chat-widget")) return; // script included twice
  const host = document.createElement("div");
  host.id = "rag-chat-widget";
  document.body.appendChild(host);
  const shadow = host.attachShadow({ mode: "open" });
  const style = document.createElement("style");
  style.textContent = css;
  const root = document.createElement("div");
  shadow.append(style, root);
  createRoot(root).render(
    <StrictMode>
      <Widget config={readConfig()} />
    </StrictMode>,
  );
}

if (document.body) mount();
else document.addEventListener("DOMContentLoaded", mount, { once: true });
