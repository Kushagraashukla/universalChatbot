# Universal RAG Chatbot

A local, reusable chatbot that answers questions from **your own documents** and cites the passages it used.
Everything runs on your machine: Ollama for the models, ChromaDB for the vectors, FastAPI for the API,
and a React chat UI that can also be embedded in other websites with one `<script>` tag.

## Architecture

```
            ┌─────────────── Ingestion (run when documents change) ───────────────┐
 docs/*.md ─► load ─► chunk (by heading, 800/120) ─► embed (qwen3-embedding) ─► ChromaDB collection
            └─────────────────────────────────────────────────────────────────────┘

 Browser (React app or widget)
        │  POST /api/v1/chat {question, app_id, session_id}
        ▼
 FastAPI ─► service: app config + access checks ─► ask():
                                                    embed question ─► top-k chunks from that app's collection
                                                    ─► prompt (rules | numbered context | question)
                                                    ─► qwen3:8b ─► answer + [n] markers
                                                    ─► citations mapped from real chunk metadata
```

| Tool | Role |
|------|------|
| Ollama | Runs `qwen3-embedding:0.6b` (vectors) and `qwen3:8b` (answers) locally |
| LangChain | Text splitters + Ollama wrappers |
| ChromaDB | Persistent vector store in `chroma_db/` (cosine distance) |
| FastAPI | HTTP API, validation, CORS, readiness checks |
| React + Vite | Full-page chat UI and the embeddable widget |

## Folder structure

```
ai-knowledge-base/          source documents for the default app (Markdown) — not in this repo, add your own
config/apps.json            one entry per application (app_id, KB folder, collection, rules, origins…)
src/
  apps.py                   loads + validates config/apps.json
  errors.py                 OllamaError, VectorStoreError
  chat.py                   ask(question) — the RAG engine entry point (also a CLI)
  ingestion/                loader, chunker, embedder (+ cache), vector_store, pipeline
  retrieval/retriever.py    question -> top-k chunks
  generation/prompt.py      prompt with trusted rules / untrusted context / question
  generation/llm.py         get_llm() — the one place to change model or provider
  evaluation/run_eval.py    quality + safety checks -> reports/eval_results.json
  api/                      FastAPI: main.py, config.py, schemas.py, service.py, routes/
frontend/
  src/components/           ChatPanel (shared), MessageBubble, FormattedText
  src/App.tsx, main.tsx     full-page app  -> frontend/dist/
  src/widget.tsx            embeddable widget -> frontend/dist-widget/widget.js
  public/widget-demo.html   demo "host website" using the widget
.cache/                     embedding cache (safe to delete; rebuilt on next ingestion)
chroma_db/                  vector database (rebuildable from the documents)
```

## Prerequisites

- Python 3.13 with the project venv (`.venv`)
- Node.js 22+ and npm
- [Ollama](https://ollama.com) running, with models:
  ```
  ollama pull qwen3-embedding:0.6b
  ollama pull qwen3:8b
  ```

## Setup (first time)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend; npm install; cd ..
```

Optional: copy `.env.example` to `.env` (and `frontend/.env.example` to `frontend/.env`) to change models,
paths, CORS origins or the API URL. Defaults work for local development.

## Run

```powershell
# 1. Build the index (only when documents change; unchanged chunks reuse cached embeddings)
.venv\Scripts\python.exe -m src.ingestion.pipeline            # app "default"

# 2. Backend  -> http://localhost:8000  (API docs at /docs)
.venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000

# 3. Frontend -> http://localhost:5173  (second terminal)
cd frontend; npm run dev
```

Build the UI and widget for production: `cd frontend; npm run build` (creates `dist/` and `dist-widget/`).
Restart the backend after building so it serves `/widget/widget.js`.

Command-line chat without the UI: `.venv\Scripts\python.exe -m src.chat "How does a user reset a forgotten password?"`

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Process is up |
| GET | `/ready` | Ollama reachable, both models installed, every enabled app's index non-empty (503 if not) |
| GET | `/api/v1/apps/{app_id}` | Public app info: name, welcome text, example questions |
| POST | `/api/v1/chat` | `{"question": "...", "app_id": "default", "session_id": "optional"}` → answer + citations |

Chat response:

```json
{
  "answer": "… [1]",
  "citations": [{"ref": 1, "chunk_id": "04a_WORKFLOWS_ACCOUNTS_AND_ACCESS.md#4", "source": "04a_WORKFLOWS_ACCOUNTS_AND_ACCESS.md",
                 "title": "Workflows: accounts and access", "section": "… > Forgotten password (email code)",
                 "distance": 0.276, "metric": "cosine"}],
  "sources": [ "...every retrieved chunk, same shape..." ],
  "app_id": "default", "session_id": "…", "request_id": "…", "model": "qwen3:8b", "duration_ms": 3844
}
```

Errors always look like `{"error": {"code": "...", "message": "..."}, "request_id": "..."}`
(400 bad input, 401 bad API key, 403 origin not allowed, 404 unknown app, 422 validation, 503 Ollama/index down, 500 other).

`distance` is ChromaDB's cosine distance (= 1 − cosine similarity, verified by the eval script). Lower is closer. It is not a percentage.

## Embed the widget in another website

```html
<script src="http://localhost:8000/widget/widget.js" data-app-id="default" defer></script>
```

| Attribute | Default | Meaning |
|-----------|---------|---------|
| `data-app-id` | `default` | Which app in `config/apps.json` to use |
| `data-api-base` | the script's own origin | Backend URL, if the API is hosted elsewhere |
| `data-title` | app name | Header title |
| `data-position` | `right` | `left` or `right` corner |

Steps for a new website, e.g. `https://shop.example.com`:
1. Add its origin to that app's `allowed_origins` in `config/apps.json`.
2. Add it to `RAG_CORS_ORIGINS` in `.env` (otherwise the browser blocks the request).
3. Restart the backend.

Demo: run backend + `npm run dev`, then open http://localhost:5173/widget-demo.html.
The widget renders in a Shadow DOM, so the host page's CSS does not change it (the demo page has deliberately loud CSS to show this).

The widget works on web pages. Native desktop or mobile apps should call `POST /api/v1/chat` directly (or use a WebView); there is no native SDK.

## Add or update a knowledge base

The knowledge base is **not included in this repository** (it holds private product documentation, and
`ai-knowledge-base/` is in `.gitignore`). Put your own Markdown files in `ai-knowledge-base/` (or point an app's
`knowledge_base_dir` elsewhere), then run the pipeline below. `config/apps.json` lists files to skip (`exclude_sources`)
and the product's role names (`roles`); adjust both for your documents.

- **Update the default one:** edit files in `ai-knowledge-base/`, then rerun `python -m src.ingestion.pipeline`.
  Changed chunks are re-embedded, unchanged ones come from the cache, removed ones are deleted from the index.
- **Add another app:** create a folder (e.g. `kb-shop/`) with `.md` files, add an entry to `config/apps.json`
  with a new `app_id` and a new `collection` name, then run `python -m src.ingestion.pipeline shop` and restart the backend.
  Only Markdown (`.md`) is supported by the loader today.

## Quality and safety evaluation

Run: `.venv\Scripts\python.exe -m src.evaluation.run_eval` (retrieval part: embeddings only; ~10 LLM calls; no
re-ingestion; writes `reports/eval_results.json`). The cases are written for the original LMS knowledge base.
Offline unit tests (no Ollama): `python tests\test_prompt.py`, `tests\test_retrieval_text.py`, `tests\test_neighbours.py`.

Latest results (2026-10-19, qwen3:8b, top_k 4) — 11/11 PASS:

| Case | Checks |
|------|--------|
| Retrieval recall (12 questions) | right section in top 4: 12/12 |
| Distance metric | Chroma distance = 1 − cosine similarity |
| Admin adds students | right evidence; no course-assignment endpoint; no API/file names; cites "Admin creates a student" |
| Teacher assigns a course / … using UI | no admin pages for a teacher; no labels made from file names; says on-screen steps are not documented |
| Explicit API request | gives `POST /api/admin/create-student` |
| Answer present / not present | correct steps / "I don't know…" with no claim that the feature is absent |
| Distractor ("Can a teacher create a course?") | "No, only an admin…" |
| Prompt injection (question / document) | no prompt leak, canary not printed |

Checks look for content markers and citation ids, not exact wording. A PASS on the injection cases means these attempts
were resisted (once only thanks to the output guard), not that the system is secure.

## How answers are made

- **Retrieval:** hybrid search — vector search (Qwen3-Embedding with a query instruction) and BM25 keyword search,
  merged with Reciprocal Rank Fusion; per-app `exclude_sources`; neighbouring chunks of the same section added in
  document order; document text capped at 8000 characters for qwen3's 4096-token window.
- **Audience:** ordinary user by default — endpoints, file names and code identifiers are removed from the context and
  from answers. Questions that mention APIs, endpoints or the database get the technical answer.
- **Roles:** when a question is about one role (e.g. a teacher), sections about another role's pages are left out.
- **Honesty:** missing steps are named as missing; a missing fact is never turned into "the product doesn't have it".
- **UI:** sources show section names; add `?debug=1` (app) or `data-debug="true"` (widget) to see the retrieved
  passages with chunk ids and distances.

## Security model and limitations

- **What protects an app:** `app_id` only selects the server-side config; it is not a password. Each app reads only its own collection.
  `allowed_origins` stops *browsers* on other websites; scripts and servers can fake the Origin header.
  For real access control set `api_key_env` for an app and send `X-API-Key` from **your server** (never put the key in a browser or widget).
- **No user login, no rate limiting, no multi-tenant admin UI.** Add a reverse proxy with rate limits and TLS before exposing this to the internet.
- Prompt-injection defenses (rules in the system prompt, untrusted text inside tagged blocks, citations mapped by code) reduce risk; they do not remove it.
- No conversation memory: each question is answered on its own; `session_id` is only echoed and logged.
- The answers can only be as good as the documents: if they describe APIs but not screens, the chatbot says the
  on-screen steps are not documented. See `PROGRESS_LOG.md` ("Knowledge-base gaps") for the list to fill.
- The 8B model's answers vary a little between runs; neighbour chunks have `distance: null`.
- The keyword index is built once per API process; restart the API after re-running ingestion.
- `max_distance` filtering exists but is off: too few measurements to choose a threshold.
- Answers take roughly 3–25 s; the UI waits up to 120 s.
- Embedding cache grows over time (old entries are not pruned); delete `.cache/` to reset.
- Deployment: serve `frontend/dist/` from any static host, run uvicorn behind a reverse proxy, set `RAG_CORS_ORIGINS`
  and `allowed_origins` to your real domains, and keep Ollama on a private network.
