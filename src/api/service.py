"""Service layer: checks access, then calls the existing RAG engine (src.chat.ask).

Error messages sent to clients are generic; full details (paths, model names)
go to the server log only.
"""

import hmac
import logging
import os
import time

from src.api.config import get_settings
from src.api.schemas import ChatRequest, ChatResponse
from src.apps import AppConfig, get_app
from src.chat import ask
from src.errors import OllamaError, VectorStoreError
from src.generation.llm import LLM_MODEL

logger = logging.getLogger("rag.api")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def authorize(app_id: str, origin: str | None, api_key: str | None) -> AppConfig:
    """Resolve app_id to its server-side config and apply that app's access rules.

    app_id alone proves nothing. Real protection comes from the per-app API key
    (server-to-server). The Origin check only stops *browsers* on other websites;
    non-browser clients can send any Origin header.
    """
    app = get_app(app_id)
    if app is None:
        raise ApiError(404, "unknown_app", "Unknown or disabled app_id.")

    if origin and app.allowed_origins and origin.rstrip("/") not in app.allowed_origins:
        raise ApiError(403, "origin_not_allowed", "This website is not allowed to use this app.")

    if app.api_key_env:
        expected = os.environ.get(app.api_key_env)
        if not expected:
            logger.error("app=%s requires env var %s but it is not set", app.app_id, app.api_key_env)
            raise ApiError(503, "app_misconfigured", "This app is not configured correctly.")
        if not api_key or not hmac.compare_digest(api_key.encode(), expected.encode()):
            raise ApiError(401, "invalid_api_key", "Missing or invalid API key.")
    return app


def answer(req: ChatRequest, origin: str | None, api_key: str | None, request_id: str) -> ChatResponse:
    app = authorize(req.app_id, origin, api_key)

    question = req.question.strip()
    limit = get_settings().max_question_chars
    if not question:
        raise ApiError(400, "empty_question", "Question is empty.")
    if len(question) > limit:
        raise ApiError(400, "question_too_long", f"Question is longer than {limit} characters.")

    start = time.perf_counter()
    try:
        result = ask(question, app.top_k, app.collection, app.max_distance, app.instructions, app.exclude_sources,
                     app.roles)
    except OllamaError:
        logger.exception("request=%s app=%s Ollama error", request_id, app.app_id)
        raise ApiError(503, "llm_unavailable", "The AI model service is unavailable. Please try again shortly.")
    except VectorStoreError:
        logger.exception("request=%s app=%s vector store error", request_id, app.app_id)
        raise ApiError(503, "index_unavailable", "The knowledge base is not available right now.")
    except ValueError as exc:
        raise ApiError(400, "invalid_request", str(exc))
    except Exception:
        logger.exception("request=%s app=%s unexpected error", request_id, app.app_id)
        raise ApiError(500, "generation_failed", "Something went wrong while generating the answer.")

    duration_ms = int((time.perf_counter() - start) * 1000)
    # Question text is not logged on purpose (may contain personal data).
    logger.info("request=%s app=%s chars=%d retrieved=%d cited=%d ms=%d", request_id, app.app_id,
                len(question), len(result["sources"]), len(result["citations"]), duration_ms)
    return ChatResponse(
        answer=result["answer"], citations=result["citations"], sources=result["sources"],
        app_id=app.app_id, session_id=req.session_id, request_id=request_id,
        model=LLM_MODEL, duration_ms=duration_ms,
    )
