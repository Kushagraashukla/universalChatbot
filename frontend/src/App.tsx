// Full-page chat app. App ID is configurable in a small settings popover and remembered per browser.
import { useState } from "react";
import { ChatPanel } from "./components/ChatPanel";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000").replace(/\/$/, "");
const DEFAULT_APP_ID = import.meta.env.VITE_APP_ID ?? "default";
const STORAGE_KEY = "rag-chat-app-id";
// Developer view (raw retrieved passages, chunk ids, distances): open the app with ?debug=1
const DEBUG = new URLSearchParams(window.location.search).get("debug") === "1";

function readAppId(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) || DEFAULT_APP_ID;
  } catch {
    return DEFAULT_APP_ID;
  }
}

export function App() {
  const [appId, setAppId] = useState(readAppId);
  const [draft, setDraft] = useState(appId);
  const [open, setOpen] = useState(false);

  function save() {
    const value = draft.trim() || DEFAULT_APP_ID;
    setAppId(value);
    setOpen(false);
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch {
      /* storage blocked: setting just won't be remembered */
    }
  }

  const settings = (
    <div className="settings">
      <button type="button" className="btn-ghost" aria-expanded={open} aria-controls="settings-panel" onClick={() => setOpen(!open)}>
        Settings
      </button>
      {open && (
        <form
          id="settings-panel"
          className="settings-panel"
          onSubmit={(e) => {
            e.preventDefault();
            save();
          }}
        >
          <label htmlFor="app-id">App ID</label>
          <input id="app-id" value={draft} onChange={(e) => setDraft(e.target.value)} pattern="[A-Za-z0-9_\-]+" maxLength={64} required />
          <p className="settings-note">Server: {API_BASE}</p>
          <button type="submit" className="btn-primary btn-small">Save</button>
        </form>
      )}
    </div>
  );

  return (
    <div className="rag-chat page">
      <ChatPanel key={appId} apiBase={API_BASE} appId={appId} headerActions={settings} autoFocus debug={DEBUG} />
    </div>
  );
}
