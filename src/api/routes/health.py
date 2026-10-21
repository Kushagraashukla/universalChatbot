"""Liveness (/health) and readiness (/ready) checks."""

import logging

import ollama
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.apps import load_apps
from src.generation.llm import LLM_MODEL
from src.ingestion.embedder import EMBEDDING_MODEL
from src.ingestion.vector_store import open_collection

router = APIRouter(tags=["health"])
logger = logging.getLogger("rag.api")


@router.get("/health")
def health():
    """The API process is up. Says nothing about Ollama or the database."""
    return {"status": "ok"}


@router.get("/ready")
def ready():
    """Everything a chat request needs: Ollama reachable, both models installed, each enabled app's index non-empty."""
    checks: dict = {}
    try:
        installed = {m.model for m in ollama.Client().list().models}
        checks["ollama"] = "ok"
        checks["models"] = {name: name in installed for name in (EMBEDDING_MODEL, LLM_MODEL)}
    except Exception:
        logger.exception("readiness: Ollama unreachable")
        checks["ollama"] = "unreachable"
        checks["models"] = {EMBEDDING_MODEL: False, LLM_MODEL: False}

    checks["apps"] = {}
    for app in load_apps().values():
        if not app.enabled:
            continue
        try:
            checks["apps"][app.app_id] = "ok" if open_collection(app.collection).count() > 0 else "empty_index"
        except Exception:
            logger.exception("readiness: index for app=%s unavailable", app.app_id)
            checks["apps"][app.app_id] = "index_unavailable"

    is_ready = (checks["ollama"] == "ok" and all(checks["models"].values())
                and all(v == "ok" for v in checks["apps"].values()))
    return JSONResponse({"status": "ready" if is_ready else "not_ready", "checks": checks},
                        status_code=200 if is_ready else 503)
