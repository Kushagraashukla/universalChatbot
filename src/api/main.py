"""FastAPI application.

Run: .venv\\Scripts\\python.exe -m uvicorn src.api.main:app --port 8000
Docs: http://localhost:8000/docs
"""

import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.config import get_settings
from src.api.routes import chat, health
from src.api.service import ApiError
from src.apps import load_apps

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rag.api")
load_apps()  # fail fast at startup if config/apps.json is invalid

app = FastAPI(title="Universal RAG Chatbot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    logger.info("%s %s -> %d (%d ms) request=%s", request.method, request.url.path, response.status_code,
                (time.perf_counter() - start) * 1000, request.state.request_id)
    return response


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message},
                         "request_id": getattr(request.state, "request_id", None)}, status_code=status)


@app.exception_handler(ApiError)
async def api_error(request: Request, exc: ApiError):
    return _error(request, exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Field names and rules only; the submitted values are not echoed back.
    details = "; ".join(f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors())
    return _error(request, 422, "invalid_request", details or "Invalid request.")


app.include_router(health.router)
app.include_router(chat.router)

# Serve the built embeddable widget (frontend: npm run build) at /widget/widget.js.
WIDGET_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist-widget"
if WIDGET_DIR.is_dir():
    app.mount("/widget", StaticFiles(directory=WIDGET_DIR), name="widget")
else:
    logger.warning("Widget not built yet (%s missing); /widget is disabled. Run: cd frontend && npm run build", WIDGET_DIR.name)
