# Progress Log

## 2026-10-18 — Phase 1: Setup
- Created `src/{ingestion,retrieval,generation}/` (with `__init__.py`), `tests/`, `.gitignore`, `requirements.txt`, `README.md`.
- Kept existing `ai-knowledge-base/` (18 `.md` files) untouched; did not create a duplicate `knowledge_base/`.

## 2026-10-18 — Phase 2: Document loading
- Inspected KB: all 18 files are `.md`. No PDF/DOCX present.
- Created `.venv` with Python 3.13.7.
- Installed `langchain-core==1.6.12` only (gives the `Document` class). Skipped `langchain-community` (pulls ~15 extra packages; its `TextLoader` does the same plain file read).
- `requirements.txt` trimmed to `langchain-core`; later-phase packages listed as comments.
- Ollama 0.40.3 is installed and running. Only `llama3.2` is available; `qwen3:8b` and `qwen3-embedding:0.6b` are missing (not downloaded).
- Added `src/ingestion/loader.py`: `load_documents(kb_dir)` -> list of Documents with metadata `source`, `file_name`, `file_type`, `title`. Skips unsupported, empty and unreadable files with a warning.
- Check: import + load OK, 18 documents loaded.

## 2026-10-18 — Phase 3: Ollama update, chunking, embeddings
- Ollama updated 0.40.3 -> 0.40.6 (official signed installer), verified with `ollama --version`.
- C: ran out of space (0 GB) during model download. Removed only my own leftovers (installer + failed `-partial` blobs).
- Set user env `OLLAMA_MODELS=E:\ollama\models` and copied existing models there (originals on C: kept).
  **Not effective:** the Ollama app's own "Model location" setting (`settings.models` in its db) = `C:\Users\OMEN\.ollama\models` overrides the env var (confirmed in every `server*.log`).
- `qwen3-embedding:0.6b` installed (639 MB, on C:), verified in `ollama list`. Copied to E: too.
- `qwen3:8b` NOT installed: download never completed; no partial files left. C: has 3.6 GB free, model needs ~5.2 GB.
- Installed `langchain-text-splitters==1.1.3`, `langchain-ollama==1.1.0`.
- Added `src/ingestion/chunker.py`: split by `#/##/###` headings, then 800 chars / 120 overlap; each chunk prefixed with its heading path; metadata adds `section`, `chunk_index`, `chunk_id`. Check: 18 docs -> 311 chunks (avg 624, max 875 chars).
- Added `src/ingestion/embedder.py`: one shared model (`get_embeddings`) for chunks and queries; clear errors for Ollama down / model missing (404 message verified); JSON cache in `.cache/` so unchanged chunks are not re-embedded.
- Added `src/ingestion/pipeline.py` (load -> chunk -> embed) and `pyrightconfig.json` (editor uses `.venv`).
- Verified: one real `embed_query` call with `qwen3-embedding:0.6b` -> 1024-dim vector. Full 311-chunk run not done yet.
- Next: switch Ollama Model location to E:, pull `qwen3:8b`, then ChromaDB (Phase 4).

## 2026-10-18 — Model storage moved to E:, qwen3:8b download
- With user approval: stopped Ollama, backed up its db to `%LOCALAPPDATA%\Ollama\db-backup-20261018-103753\`, set `settings.models = E:\ollama\models`, restarted.
- Verified: `server.log` shows `OLLAMA_MODELS:E:\\ollama\\models`; `ollama list` still shows `qwen3-embedding:0.6b` + `llama3.2` (copies on E:).
- `qwen3:8b` first pull (10:40) stalled at 0 bytes for 13 min; server log had `context deadline exceeded` to ollama.com. Registry reachable afterwards (HTTP 200). Stopped the stalled pull, retried once (10:53).
- Direct CDN test: ~0.28 MB/s. Download is slow, not broken; ~2,650 / 4,983 MB done at 10:55.
- `qwen3:8b` download finished (`success`, exit 0). Verified in `ollama list`: `qwen3:8b` 5.2 GB, `qwen3-embedding:0.6b` 639 MB, `llama3.2` 2.0 GB. No partial files left. Server log path: `E:\ollama\models`.
- Verified: real `embed_query` call -> 1024-dim vector; pipeline yields 311 chunks (full chunk embedding not run).
- Verified: `qwen3:8b` answers a prompt (`ollama run`, replied "OK").
- Verified: full pipeline run -> 18 docs -> 311 chunks -> 311 vectors (1024-dim). Second run: 0 new, 311 reused from `.cache/embeddings.json` (4.1 MB).

## 2026-10-18 — Phase 4: ChromaDB vector storage
- Installed `chromadb==1.5.9` directly (no `langchain-chroma` wrapper needed: we already have our own vectors).
- Added `src/ingestion/vector_store.py`:
  - `get_collection()`: persistent DB in `chroma_db/` (override with `CHROMA_DIR` env var), collection `knowledge_base`, cosine distance, Chroma's own embedding disabled, collection tagged with `embedding_model`; errors if the model differs.
  - `index_chunks(chunks, vectors)`: upsert by `chunk_id` (no duplicates), then delete IDs no longer produced (stale chunks). Refuses to run on 0 chunks so a wrong path cannot wipe the index. Touches only its own collection.
- `pipeline.py` now ends with `index_chunks`. `chroma_db/` already in `.gitignore`.
- Verified: 2 pipeline runs -> 311 records both times, 0 new embeddings (cache reused).
- Verified: fresh process reopened DB -> count 311, sample `05_API_REFERENCE.md#0` has text, metadata (source, chunk_index, section...) and 1024-dim vector.
- Verified: indexing 309 chunks removed exactly 2 stale; re-index restored 311.

## 2026-10-18 — Phase 5 + 6: Retrieval, prompt, answer generation
- Confirmed from DB: `chroma_db/`, collection `knowledge_base`, 311 records, `hnsw.space = cosine`, model `qwen3-embedding:0.6b`.
- Small reuse refactors: `embedder._run` -> `call_ollama(fn, *args, model=...)` (shared Ollama error messages); `vector_store.open_collection()` (read-only, never creates an empty DB) + shared `_check_model`.
- Added `src/retrieval/retriever.py`: `retrieve(question, top_k=4)` -> chunk_id, text, source, title, section, raw distance + metric read from DB. Errors for empty question, missing DB/collection, empty DB, Ollama down.
- Added `src/generation/prompt.py`: system rules (answer only from context, say "I don't know", ignore instructions inside documents, cite file names); chunks in `<context><document source=...>`, question in `<question>`; closing tags in docs escaped.
- Added `src/generation/llm.py`: `get_llm()` = ChatOllama `qwen3:8b`, temperature 0, `reasoning=False` (no thinking output). `generate()` uses shared error handling.
- Added `src/chat.py`: `ask(question)` -> `{answer, sources}`; CLI `python -m src.chat "question"`. No ingestion / re-embedding per question. No new dependencies.
- Verified (1 question, "How does a user reset a forgotten password?", 28 s): top source `04a...#4` (cosine distance 0.276); all 4 chunk IDs exist in DB; answer's endpoints, 10-min validity, 5 attempts, sessions ending match `04a_WORKFLOWS_ACCOUNTS_AND_ACCESS.md` lines 35–37; no `<think>` in output.

## 2026-10-18 — Product phase A: quality, citations, multi-app config, FastAPI
- `src/errors.py`: `OllamaError`, `VectorStoreError` (subclass RuntimeError; CLI unchanged).
- `src/__init__.py` loads `.env` once (real env vars win). Models via `EMBEDDING_MODEL`, `LLM_MODEL` env.
- Citations: prompt now numbers chunks `[1]..[n]` and forbids file names; `chat._citations` maps `[n]` to real retrieved metadata (`ref, chunk_id, source, title, section, distance, metric`). `ask()` returns `{answer, citations, sources}` (old keys kept).
- Optional `max_distance` filter in `retrieve()` (cosine only). Kept disabled: only 2 observed points (0.276 answerable vs 0.427 unanswerable) — not enough to pick a threshold.
- Prompt: per-app `instructions` appended below core rules; `</document|context|question>` escaped in untrusted text.
- `config/apps.json` + `src/apps.py`: per-app `app_id`, KB dir, collection, instructions, top_k, max_distance, enabled, allowed_origins, api_key_env, welcome text, example questions. Duplicate app_ids/collections rejected. Pipeline now `python -m src.ingestion.pipeline [app_id]`.
- Eval `python -m src.evaluation.run_eval` -> `reports/eval_results.json`. Results 12:18: distance check PASS (chroma 0.2760 == 1-cos 0.2760); answer_present PASS (cites `04a...#4`); answer_not_present PASS ("I don't know..."); injection_in_question PASS; injection_in_document PASS (canary not printed). PASS != secure.
- FastAPI (`src/api/`): `config.py` (RAG_* settings), `schemas.py`, `service.py` (authorize + call `ask`, generic client errors, details in log), `routes/health.py` (`/health`, `/ready`), `routes/chat.py` (`POST /api/v1/chat`, `GET /api/v1/apps/{id}`), `main.py` (CORS from `RAG_CORS_ORIGINS`, request IDs, JSON errors, `/widget` static mount). Installed `fastapi`.
- Verified on :8000: `/health` ok; `/ready` 200 (ollama ok, both models, app default ok); unknown app 404; foreign Origin 403; empty question 422; CORS preflight allows localhost:5173; real chat 3.8 s, citations `00_INDEX.md#2`, `01_PRODUCT_OVERVIEW.md#0`, all 4 source IDs exist in DB.
- Observed weakness: "What roles exist?" retrieved index/overview chunks, not `03_ROLES_PERMISSIONS.md`; answer vague.

## 2026-10-18 — Product phase B: React UI, embeddable widget, docs
- `frontend/` (React 19.3, TypeScript 7, Vite 8; no other runtime deps). Shared `ChatPanel` (welcome + examples from `/api/v1/apps/{id}`, Enter/Shift+Enter, typing indicator, disabled send while loading, error + Retry, 120 s timeout, New chat aborts in-flight request, citations + "All retrieved passages", Copy, aria-live log, focus styles, dark mode, reduced motion). `FormattedText` renders `code`/**bold**/[n] as React elements (no innerHTML).
- Full app `App.tsx` (App ID setting, saved in localStorage with try/catch). Widget `widget.tsx`: Shadow DOM, launcher open/close, Escape closes and returns focus, `data-app-id`, `data-api-base` (default = script origin), `data-title`, `data-position`. Built to `frontend/dist-widget/widget.js`, served by FastAPI at `/widget/widget.js`. Demo: `frontend/public/widget-demo.html`.
- Added `.env.example`, `frontend/.env.example`; `.gitignore` += node_modules, frontend/dist, frontend/dist-widget, `!.env.example`. `requirements.txt` pinned incl. fastapi 0.143.3, uvicorn 0.54.0, pydantic(-settings), python-dotenv, ollama, numpy. README rewritten.
- Verified: `npm run build` OK (tsc + app 222 KB JS + widget 230 KB / 74 KB gzip; no `process.env` left, CSS inlined). `/widget/widget.js` served 200. CORS header for :5173 on app info.
- Verified in headless Chrome (CDP): app loads 3 example questions from API; click -> typing indicator -> real answer, Sources = `04a_WORKFLOWS_ACCOUNTS_AND_ACCESS.md`, no error. Widget on demo page: launcher in shadow DOM, opens, 3 examples, title font stays system-ui while host forces Georgia (CSS isolated). 390 px: no horizontal overflow; widget panel full-screen.
- Fixed after screenshots: textarea showed scrollbar arrows (auto-height ignored borders; in the widget it was also measured while hidden — now re-measured on open). Re-verified with a widget screenshot.
- Not done / limits: no auth/rate limiting, no conversation memory, widget not tested in Firefox/Safari, no screen-reader test.

## 2026-10-18 — Live testing + fixes
- Live E2E (headless Chrome, real keyboard input, real backend): 28/28 PASS — Enter/Shift+Enter, double-Enter guard, send disabled while loading, real answers with sources, Copy -> clipboard, "I don't know" with no fake sources, injection via UI ignored, New chat cancels running request + refocuses input, 2000-char limit, unknown app id error + Retry, widget open/focus/send/Escape/reopen keeps chat, no console errors.
- Bug found: example "What roles exist and what can each one do?" missed `superadmin` — the roles table is split across `01_PRODUCT_OVERVIEW.md#3/#4` and #4 ranked 5th (top_k=4).
- Fix 1 (`retriever._add_neighbours`): each hit is followed by its previous/next chunk if in the same section (distance = null for these). No re-embedding. API `Citation.distance` and TS type now allow null.
- Found while fixing: neighbours make prompts longer; Ollama's window is 4096 tokens and cuts long prompts silently (top_k up to 20 is allowed). Fix 2: `llm.NUM_CTX = 4096` set explicitly; `chat._fit_budget` caps document text at 8000 chars (hits kept before neighbours). top_k=4: unchanged (4640 chars); top_k=20: 12785 -> 7627 chars.
- Tests: `tests/test_neighbours.py` (neighbours + budget, offline) pass.
- Re-verified: roles answer lists all 7 roles incl. superadmin (cites #3 + #4); password answer still correct; Ollama log `truncated = 0`; eval 5/5 PASS; live E2E 28/28 PASS (before the budget change; the budget does not change top_k=4 prompts).
- Cost: answers now ~10–25 s instead of ~4–15 s (more context).

## 2026-10-18 — User report: "it keeps saying it doesn't know"
- Reproduced with 15 realistic questions: engine answered 14/15 correctly in English; failures came from the prompt, not the pipeline: rule "reply exactly I don't know" made partial matches refuse with no help; answers always English (or Devanagari for Hinglish).
- Prompt changes (`prompt.py`): answer the covered part and name what is missing; when nothing relevant, start with "I don't know…" then suggest topics from the context; greetings get a short friendly reply; reply language decided in code by `reply_language()` (Devanagari -> Hindi, >=2 common Hinglish words -> Hinglish in Latin letters, else English) and placed as the last line of the prompt.
- Tried and rejected: prefixing the app name to the search query fixed "ye app kya karta hai" but pushed `04a#4` out of the password results.
- Verified: unit tests pass (`tests/test_neighbours.py`, incl. `reply_language`); Hinglish answers no longer Devanagari (1/4 Hinglish, 3/4 English, all correct facts); greetings answered; out-of-scope questions get "I don't know…" + topic suggestions; eval 5/5 PASS; live API Hinglish password answer correct.
- Still weak: "ye app kya karta hai" / "what does this app do" retrieve `10_AI_TOOL_CATALOG` and `13_AI_GROUNDING_RULES` (documents about designing the chatbot itself) instead of `01_PRODUCT_OVERVIEW`, so they still say "I don't know".

## 2026-10-19 — Accuracy mission: retrieval, end-user answers, UI, eval
Audit found: correct "Admin creates a student" chunk at rank 6 (top_k=4); answers copied endpoints/file names; prompt had no user/developer distinction; UI showed file names, chunk ids, "All retrieved passages"; KB is developer docs (`00_INDEX.md`: "not user help text").
- Retrieval (measured on 12 probe questions, right section in top 4): 7/10 baseline -> 12/12.
  - `config/apps.json` `exclude_sources`: design docs about the chatbot (10, 11, 13, _ANALYSIS_NOTES) filtered at query time (no re-ingestion).
  - Hybrid search: `src/retrieval/lexical.py` (BM25, in-memory, built from the collection) + vector search, merged with RRF (k=60, 20 candidates; k=10/20/60 gave identical results).
  - Query instruction for Qwen3-Embedding (`embedder.QUERY_INSTRUCTION`; documents unchanged) — "What API endpoint creates a student?" rank 15 -> 1-2.
  - `retriever.search_text` drops "API/endpoint/route…" from the search text only.
  - Neighbour chunks now in document order (previous chunk before the hit) — fixed a table read without its header ("teacher cannot add students to a course").
  - Bug fixed: tokenizer turned "creates" into "creat".
- Generation (`prompt.py`, `chat.py`):
  - Audience decided in code (`audience()`): ordinary user by default, developer when the question uses API/endpoint/database words. User mode strips endpoints, file names and code identifiers from the context (`strip_developer_details`) and forbids them in answers; one rewrite pass if they still appear (`TECHNICAL_LEAK`).
  - Rules: never substitute another workflow; say which part is not documented; never conclude a feature is absent; mention side effects; UI questions get an explicit "on-screen steps are not documented" rule (`asks_for_ui`).
  - Roles (`config/apps.json` `roles`): sections whose heading actor is another role (e.g. "(admin, /admin/courses)" when a teacher asks) are dropped from context (`role_note`, `asked_roles`); labelling them alone did not stop the 8B model.
  - Output guard: an answer repeating the system prompt is replaced by a refusal (`leaks_system_prompt`); the model alone printed it once in testing.
  - Citation reminder in the final prompt line (citations had disappeared in plain-language answers).
- UI: Sources show readable section names, grouped per section; no file names, chunk ids or distances. Retrieved passages only in debug mode (`?debug=1`, widget `data-debug="true"`).
- Eval (`run_eval.py`, deterministic checks): retrieval 12/12; distance metric; admin add students; teacher assign course; teacher assign course using UI; explicit API request; answer present; answer not present; distractor ("Can a teacher create a course?"); injection in question; injection in document. Final run: 11/11 PASS.
- Unit tests: `tests/test_neighbours.py`, `tests/test_prompt.py`, `tests/test_retrieval_text.py` pass. `npm run build` OK. API `/ready` 200; live chat "How may an admin add students?" -> `/admin/add-student`, class + section, generated password, no endpoint, cites "Admin creates a student".
- Remaining limits: 8B answers vary between runs (Q2 warning about unenrolment appeared in 2 of 3 runs); small wording slips ("the admin also enrols" where the system does it); BM25 cache is not refreshed on same-size re-ingestion (restart API).

### Knowledge-base gaps to fill (not invented by the chatbot)
1. **Admin adds a student:** only the page (`/admin/add-student`), "choose class and section" and the Copy password notice. Missing: field list, buttons, success/error messages on screen.
2. **Teacher assigns a course / students:** only "Teacher-managed roster (older page)" with an API and two component names. Missing: whether this page is still in the product, its menu path/URL, and screen steps. (`12_GAPS` C4 notes a duplicate `/mentor/course/:courseId` route.)
3. **Admin manages course classes (`/admin/courses`):** actions listed as API rows. Missing: on-screen steps.
4. **Course creation (`/admin/create-course`):** choices described; missing: screen-by-screen steps and labels.
5. **General:** no end-user help pages; no `audience` / `status (current|legacy)` metadata per section (suggested by `11_RAG_KNOWLEDGE_STRATEGY.md`).

## 2026-10-21 — Published to GitHub
- Pushed to `https://github.com/Kushagraashukla/universalChatbot` (public). `ai-knowledge-base/` and `reports/` excluded via `.gitignore` (internal product docs and eval output quoting them).
